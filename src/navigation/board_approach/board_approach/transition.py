"""Prepare precise docking from the latest valid deskewed moving observation."""
import math
import numpy as np


def moving_transition_target(reports, reference, pose, now, standoff):
    """Use the latest valid near-end fit; multi-frame agreement is diagnostic.

    Only this run's observations are eligible. The selected frame must meet the
    6-second freshness bound, geometry/identity checks and view-distance limits.
    Transform its geometry to the current base pose. Never average older fits
    into the selected goal or require another frame merely for consensus.
    """
    if reference.get('geometry_source') != 'lidar' or reference.get('use_aruco', True):
        return None
    pose = np.asarray(pose, float)
    selected, stamps = [], set()
    for report in reversed(reports):
        try:
            if report.get('error') or report.get('geometry_source') != 'lidar':
                continue
            age = now-float(report['captured_monotonic'])
            prior = np.asarray(report['map_from_base'], float)
            center = np.asarray(report['object_map'], float)
            normal = np.asarray(report['surface_normal_map'], float)
            if (not 0 <= age <= 10. or report['timestamp_ns'] in stamps
                    or prior.shape != (4,4) or center.shape != (3,) or normal.shape != (3,)
                    or not np.isfinite(prior).all() or not np.isfinite(center).all()
                    or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1) > .02):
                continue
            yaw_change = math.atan2(prior[1,0],prior[0,0])-math.atan2(pose[1,0],pose[0,0])
            yaw_change = math.atan2(math.sin(yaw_change),math.cos(yaw_change))
            if (np.linalg.norm(prior[:2,3]-pose[:2,3]) > .6
                    or abs(yaw_change) > math.radians(45)
                    or np.linalg.norm(center[:2]-prior[:2,3]) > max(2.7,standoff+1.3)):
                continue
            selected.append(report)
            stamps.add(report['timestamp_ns'])
            if len(selected) == 3:
                break
        except (KeyError, ValueError, TypeError):
            continue
    if not selected or now-selected[0]['captured_monotonic'] > 6.:
        return None
    centers = np.array([r['object_map'] for r in selected])
    normals = np.array([r['surface_normal_map'] for r in selected])
    angles = np.unwrap(np.arctan2(-normals[:,1],-normals[:,0]))
    spread = float(np.max(np.linalg.norm(centers[:,:2]-centers[:,:2].mean(axis=0),axis=1)))
    heading_span = float(np.ptp(angles))
    # Use only the latest fit. Older fits cannot veto or shift a valid goal.
    center = centers[0].copy()
    normal = normals[0].copy()
    normal[2] = 0.
    normal /= np.linalg.norm(normal)
    if (np.linalg.norm(center[:2]-np.array(reference['board_map'])[:2]) > .18
            or normal @ np.asarray(reference['surface_normal_map']) < math.cos(math.radians(25))):
        return None
    local_center = pose[:3,:3].T @ (center-pose[:3,3])
    local_normal = pose[:3,:3].T @ normal
    n = local_normal[:2]/np.linalg.norm(local_normal[:2])
    distance = float(-local_center[:2] @ n)
    if distance < standoff-.1:
        return None
    goal = center+standoff*normal
    return dict(reference,board_map=center.tolist(),object_map=center.tolist(),
                surface_normal_map=normal.tolist(),goal_map=[float(goal[0]),float(goal[1]),0.],
                goal_yaw=float(math.atan2(-normal[1],-normal[0])),
                distance=distance,lateral=float(local_center[:2] @ np.array([-n[1],n[0]])),
                heading_error=float(math.atan2(-n[1],-n[0])),
                observed_robot_xy=pose[:2,3].tolist(),coarse_orientation=False,
                approach_phase='precise_docking',center_spread_m=spread,heading_span_rad=heading_span,
                heading_limit_rad=.12,prefetched_observations=1,stationary_observations=0,
                diagnostic_observations=len(selected),multi_frame_consistent=(spread<=.06 and heading_span<=.12) if len(selected)>1 else None,
                timestamp_ns=selected[0]['timestamp_ns'],captured_monotonic=selected[0]['captured_monotonic'],
                transition_evidence=selected)
