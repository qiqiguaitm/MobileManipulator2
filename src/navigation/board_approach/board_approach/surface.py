"""Live RGB-D front-surface estimation and normal-axis docking geometry."""
import math
import numpy as np
from .perception import DetectionError, fit_board_plane, target_geometry


def normal_angle_limit(distance):
    if not np.isfinite(distance) or distance <= 0:
        raise DetectionError('Invalid depth for normal consistency threshold')
    growth = np.clip((distance**2 - 1.5**2)/(4.0**2 - 1.5**2), 0., 1.)
    return float(8. + 12.*growth)


def observation_heading_limit(targets, standoff):
    # Only permit rough orientation well outside the final docking zone.
    depth = min(t.get('plane_fit', {}).get('depth_m', 0.) for t in targets)
    clearance = min(t['distance'] for t in targets)
    if depth <= 2. or clearance <= max(1.8, standoff + .6):
        return .12
    return float(np.interp(depth, [2., 4.], [.12, math.radians(20)]))


def average_surface_normals(left, right, reference, distance):
    normals = []
    for value in (left, right, reference):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-6:
            raise DetectionError('Invalid surface normal')
        normals.append(value / np.linalg.norm(value))
    left, right, reference = normals
    # Fits all point toward the camera; opposite signs indicate inconsistent evidence.
    limit = normal_angle_limit(distance)
    angles = [math.degrees(math.acos(float(np.clip(n @ reference, -1., 1.)))) for n in (left, right)]
    if max(angles) > limit:
        raise DetectionError(f'Front surface normal ambiguous across object depth patches: angles={angles}, limit={limit:.2f}deg; depth={distance:.3f}m')
    average = left + right
    average /= np.linalg.norm(average)
    return average, dict(normal_limit_deg=limit, half_to_full_angles_deg=angles,
        left_normal=left.tolist(), right_normal=right.tolist(),
        full_normal=reference.tolist(), used_normal=average.tolist(), normal_method='normalized_left_right_mean')


def surface_observation(observation, depth, k):
    if observation.source != 'object_depth':
        return observation  # Marker/appearance modes already fit a depth plane.
    polygon = np.asarray(observation.polygon, np.float32)
    uv = polygon.mean(axis=0)
    # Exclude box borders, which frequently contain the object's background.
    inner = (uv + .7 * (polygon - uv)).astype(np.float32)
    plane = fit_board_plane(depth, k, inner, uv, [])
    # A low residual alone does not establish a stable normal on a tiny patch.
    x1, y1 = inner.min(axis=0)
    x2, y2 = inner.max(axis=0)
    if min((x2-x1)*plane.center[2]/k[0, 0], (y2-y1)*plane.center[2]/k[1, 1]) < .08:
        raise DetectionError('Visible front surface too small to estimate its normal')
    halves = []
    for lo, hi in ((x1, uv[0]), (uv[0], x2)):
        half = np.float32([[lo,y1],[hi,y1],[hi,y2],[lo,y2]])
        check = fit_board_plane(depth, k, half, [(lo+hi)/2, uv[1]], [])
        halves.append(check)
    plane.normal, diagnostics = average_surface_normals(
        halves[0].normal, halves[1].normal, plane.normal, plane.plane_fit['depth_m'])
    plane.plane_fit.update(diagnostics)
    plane.plane_fit['half_plane_fits'] = [half.plane_fit for half in halves]
    plane.source, plane.score, plane.polygon = observation.source, observation.score, polygon
    return plane


def surface_target_geometry(observation, base_from_camera, map_from_base, standoff):
    normal = np.asarray(observation.normal)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < .9:
        raise DetectionError('A reliable surface normal is required for perpendicular alignment')
    target = target_geometry(observation, base_from_camera, map_from_base, standoff)
    world_normal = map_from_base[:3,:3] @ base_from_camera[:3,:3] @ normal
    world_normal[2] = 0
    world_normal /= np.linalg.norm(world_normal)
    target.update(object_map=target['board_map'], surface_normal_map=world_normal.tolist(),
                  alignment='perpendicular_to_surface')
    return target


def approach_goal(target, standoff):
    """Reach a normal-axis staging point before entering the final approach."""
    result = dict(target)
    staging_distance = max(1.5, standoff + .5)
    off_axis = abs(target['lateral']) > .10
    if off_axis and target['distance'] < staging_distance - .10:
        raise DetectionError('Too close for lateral realignment; reposition farther from the surface before retrying')
    if off_axis or target['distance'] > staging_distance + .10:
        point = np.asarray(target['board_map']) + staging_distance*np.asarray(target['surface_normal_map'])
        result['goal_map'] = [float(point[0]), float(point[1]), 0.]
        result['approach_phase'] = 'reach_surface_normal_axis'
    else:
        result['approach_phase'] = 'final_normal_approach'
    return result
