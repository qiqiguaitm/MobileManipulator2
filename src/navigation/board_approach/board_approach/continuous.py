"""Continuous target approach and timestamped visual feedback (no motor access)."""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from .perception import detect_board, detect_board_region
from .object_center import center_target_geometry
from .surface import surface_observation, surface_target_geometry
from .lidar_geometry import measure_lidar

MAX_VISUAL_AGE = 6.0
MAX_START_VISUAL_AGE = 30.0
START_FEEDBACK_TIMEOUT = 45.0
NEAR_GOAL_DISTANCE = .40


class FeedbackExpired(RuntimeError):
    """An observation is too old to update a goal or verify arrival."""


def feedback_required_near_goal(goal, robot_xy):
    # Legacy/segmented callers retain strict feedback unless a bounded navigation
    # leg is explicitly identified. Distance is to this leg, not to the object.
    if goal.get('approach_phase') not in ('continuous_approach', 'continuous_coarse', 'precise_docking'):
        return True
    xy = np.asarray(robot_xy, float)
    endpoint = np.asarray(goal['goal_map'], float)[:2]
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(endpoint).all():
        raise RuntimeError('Invalid navigation position for feedback gate')
    return bool(np.linalg.norm(xy-endpoint) <= NEAR_GOAL_DISTANCE)



def stage_goal(target, standoff, precise):
    result = dict(target)
    if precise:
        if target['coarse_orientation']:
            raise RuntimeError('Precise stationary orientation is not stable; no additional approach attempts')
        result['approach_phase'] = 'precise_docking'
    else:
        distance = max(2., standoff + .8)
        if target['distance'] < distance - .15:
            raise RuntimeError('Already inside staging clearance; reposition farther away for the two-phase approach')
        goal = np.array(target['board_map']) + distance*np.array(target['surface_normal_map'])
        result['goal_map'] = [float(goal[0]), float(goal[1]), 0.]
        result['approach_phase'] = 'continuous_coarse'
    return result


def validate_feedback(feedback, target, now, max_age=MAX_VISUAL_AGE):
    age = now - feedback['captured_monotonic']
    if not math.isfinite(age) or not 0 <= age <= max_age:
        raise FeedbackExpired('Asynchronous visual feedback expired; stop')
    center = np.asarray(feedback['object_map'], float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise RuntimeError('Invalid asynchronous object position')
    delta = center - np.array(target['board_map'])
    # A pillar face can extend above the image. Its visible vertical crop changes
    # during approach; docking constrains the horizontal target centre only.
    if target.get('geometry_source')=='lidar':
        delta = delta[:2]
    if np.linalg.norm(delta) > .25:
        raise RuntimeError('Asynchronous target moved or detection identity changed')
    return center


class VisualMonitor:
    """One pending inference; all ROS access stays on the parent's main thread."""
    def __init__(self, node, detector, aruco, base_from_camera, target, directory, geometry_source='depth'):
        self.node, self.detector, self.aruco = node, detector, aruco
        self.transform, self.target = base_from_camera.copy(), dict(target)
        self.geometry_source = geometry_source
        self.directory = Path(directory)/'async'
        self.directory.mkdir()
        self.path = self.directory/'feedback.json'
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future, self.frame = None, 0
        self.next_capture = 0.
        self.valid_updates = 0
        self.recent_reports = []
        self.closed = False
        self.error_counts = {}
        self.last_error = None
        self.last_valid_capture = None

    def infer(self, frame):
        image, depth, k, pose, timestamp, captured, index = frame[:7]
        lidar_frame = frame[7] if len(frame)>7 else None
        started = time.monotonic()
        prefix = self.directory/('frame_%04d' % index)
        cv2.imwrite(str(prefix)+'.png', image)
        if depth is not None:
            np.save(str(prefix)+'_depth.npy', depth)
        if lidar_frame is not None:
            np.savez_compressed(str(prefix)+'_sensor.npz',points_base=lidar_frame['points_base'],
                                rings=lidar_frame['rings'],distortion=lidar_frame['distortion'])
        report = dict(timestamp_ns=timestamp, captured_monotonic=captured,
                      camera_matrix=k.tolist(), map_from_base=pose.tolist(),geometry_source=self.geometry_source,
                      lidar=lidar_frame['metadata'] if lidar_frame else None,
                      base_from_camera=self.transform.tolist())
        timings = dict(start_delay_seconds=started-captured,storage_seconds=time.monotonic()-started)
        try:
            detection_started = time.monotonic()
            if self.geometry_source=='lidar':
                region = detect_board_region(image) if self.aruco else self.detector.detect_region(image)
                timings['detector_seconds'] = time.monotonic()-detection_started
                geometry_started = time.monotonic()
                obs = measure_lidar(region,lidar_frame,k,self.transform,image,prefix)
                timings['geometry_and_evidence_seconds'] = time.monotonic()-geometry_started
            else:
                obs = detect_board(image, depth, k) if self.aruco else self.detector.detect(image, depth, k)
            annotated = image.copy()
            cv2.polylines(annotated, [obs.polygon.astype(np.int32)], True, (0,220,255), 2)
            cv2.imwrite(str(prefix)+'_detection.png', annotated)
            # LiDAR already established a surface: use exactly the same geometry
            # as stationary observation, not the legacy RGB-D centre-only path.
            surface = (surface_target_geometry(obs, self.transform, pose, 1.)
                       if self.geometry_source=='lidar' else None)
            center = (surface['object_map'] if surface is not None else
                      center_target_geometry(obs, self.transform, pose, 1.)['object_map'])
            report.update(object_map=center, detector=getattr(self.detector, 'last_metadata', None))
            if self.geometry_source=='lidar' and not self.aruco:
                from .door_filter import door_model
                report['door_model'] = door_model(region,obs,k,lidar_frame['distortion'],self.transform,pose,captured)
            try:
                if self.geometry_source!='lidar':
                    obs = surface_observation(obs, depth, k)
                if surface is None:
                    surface = surface_target_geometry(obs, self.transform, pose, 1.)
                report['surface_normal_map'] = surface['surface_normal_map']
                report['plane_fit'] = obs.plane_fit
            except Exception as error:
                if self.geometry_source=='lidar':
                    raise
                report['normal_error'] = str(error)
        except Exception as error:
            report['error'] = str(error)
            metadata = getattr(self.detector, 'last_metadata', None)
            if metadata:
                report['detector'] = metadata
                annotated = image.copy()
                x1,y1,x2,y2 = np.rint(metadata['bbox']).astype(int)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.imwrite(str(prefix)+'_detection.png',annotated)
        raw_response = getattr(self.detector, 'last_response', None)
        if raw_response is not None:
            Path(str(prefix)+'_response.json').write_text(json.dumps(raw_response, ensure_ascii=False, indent=2))
        timings.update(processing_seconds=time.monotonic()-started,
                       age_at_completion_seconds=time.monotonic()-captured)
        report['timings'] = timings
        Path(str(prefix)+'.json').write_text(json.dumps(report, indent=2))
        return report

    def record_status(self, error=None, captured=None):
        if error:
            self.last_error = str(error)
            self.error_counts[self.last_error] = self.error_counts.get(self.last_error, 0)+1
        if captured is not None:
            self.last_valid_capture = captured
        status = dict(last_error=self.last_error, error_counts=self.error_counts,
                      valid_updates=self.valid_updates, last_valid_capture=self.last_valid_capture,
                      updated_monotonic=time.monotonic())
        temporary = self.directory/'status.tmp'
        temporary.write_text(json.dumps(status, ensure_ascii=False))
        temporary.replace(self.directory/'status.json')

    def failure_summary(self):
        return (f'valid_updates={self.valid_updates}; last_error={self.last_error or "inference pending"}; '
                f'error_counts={self.error_counts}; evidence={self.directory}')

    def poll(self, capture=True):
        if self.closed:
            return
        if self.future is not None and self.future.done():
            report = self.future.result()
            self.future = None
            if 'error' in report:
                self.record_status(report['error'])
            else:
                try:
                    self.publish_start_feedback(report)
                    validate_feedback(report, self.target, time.monotonic())
                except FeedbackExpired as error:
                    # Keep evidence, but never publish an old frame as fresh.
                    # The motion process decides whether this leg needs feedback.
                    self.record_status(error)
                else:
                    temporary = self.path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(report))
                    temporary.replace(self.path)
                    self.recent_reports.append(report)
                    self.recent_reports = self.recent_reports[-6:]
                    self.valid_updates += 1
                    self.record_status(captured=report['captured_monotonic'])
        if self.future is None:
            (self.directory/'pending.json').unlink(missing_ok=True)
        capture = capture and not (self.directory/'endpoint_wait.json').exists()
        if capture and self.future is None and time.monotonic() >= self.next_capture:
            try:
                image, depth, k, pose, timestamp = self.node.snapshot(moving=True)
            except Exception as error:
                failure=dict(error=str(error),monotonic=time.monotonic(),raw_odom_samples=[])
                for message in list(getattr(self.node,'odometry',[]))[-6:]:
                    p=message.pose.pose.position
                    failure['raw_odom_samples'].append(dict(
                        timestamp=message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                        frame=message.header.frame_id,child=message.child_frame_id,xyz=[p.x,p.y,p.z]))
                (self.directory/'capture_error.json').write_text(json.dumps(failure))
                with (self.directory/'capture_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(failure)+'\n')
                self.record_status(error)
                self.next_capture = time.monotonic()+.2
                return
            age = (self.node.get_clock().now().nanoseconds-timestamp)/1e9
            self.future = self.pool.submit(self.infer, (image,depth,k,pose,timestamp,
                                             time.monotonic()-max(0.,age),self.frame,
                                             self.node.last_lidar_frame if self.geometry_source=='lidar' else None))
            pending=self.directory/'pending.tmp'
            pending.write_text(json.dumps(dict(timestamp_ns=timestamp,captured_monotonic=time.monotonic()-max(0.,age))))
            pending.replace(self.directory/'pending.json')
            self.frame += 1
            self.next_capture = time.monotonic()+2.

    def publish_start_feedback(self, report):
        # Separate startup channel: never present a delayed frame as fresh in motion.
        validate_feedback(report, self.target, time.monotonic(), max_age=MAX_START_VISUAL_AGE)
        path = self.directory/"startup_feedback.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report))
        temporary.replace(path)

    def seed_from_observation(self, target):
        # A freshly validated stationary frame is already a valid start observation;
        # no extra model round-trip is necessary just to enable the next leg.
        if 'captured_monotonic' not in target:
            return
        report = dict(captured_monotonic=target['captured_monotonic'],
                      timestamp_ns=target['timestamp_ns'], object_map=target['board_map'],
                      surface_normal_map=target['surface_normal_map'],
                      geometry_source=target['geometry_source'],
                      source='moving_transition' if target.get('prefetched_observations') else 'stationary_confirmation')
        try:
            self.publish_start_feedback(report)
            validate_feedback(report, self.target, time.monotonic())
        except FeedbackExpired:
            return
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report))
        temporary.replace(self.path)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        try:
            if self.future is not None and not self.future.cancelled():
                self.poll(capture=False)  # Preserve the last in-flight near-end observation.
        finally:
            self.closed = True


def corrected_navigation_goal(feedback, target, current_goal, robot_xy=None):
    """Small anchored corrections; never let noisy observations walk the goal away."""
    if target.get('approach_phase') not in ('continuous_approach', 'continuous_coarse', 'precise_docking') or 'surface_normal_map' not in feedback:
        return None
    observed = np.asarray(feedback['surface_normal_map'], float)
    reference = np.asarray(target['surface_normal_map'], float)
    if observed.shape != (3,) or not np.isfinite(observed).all() or np.linalg.norm(observed) < .9:
        raise RuntimeError('Invalid asynchronous surface normal')
    observed /= np.linalg.norm(observed)
    if observed @ reference < math.cos(math.radians(25)):
        return None  # Keep the validated rough direction; wait for a consistent new normal.
    fine = target.get('approach_phase') == 'precise_docking'
    # Fine alignment follows the close observation more strongly, still bounded
    # by the original anchor, identity checks and normal agreement.
    blend = proximity_weight(target, robot_xy) if target.get('approach_phase') == 'continuous_approach' else float(fine)
    gain = 1. if current_goal.get('approach_observation_refreshed') else .2 + .6*blend
    normal = (1-gain)*reference + gain*observed
    normal /= np.linalg.norm(normal)
    anchor = np.asarray(target['board_map'])
    center = (1-gain)*anchor + gain*np.asarray(feedback['object_map'])
    stand = np.linalg.norm(np.asarray(target['goal_map'])[:2]-anchor[:2])
    point = center+stand*normal
    if np.linalg.norm(point[:2]-np.asarray(target['goal_map'])[:2]) > .35:
        return None
    yaw = math.atan2(-normal[1],-normal[0])
    yaw_delta = math.atan2(math.sin(yaw-current_goal['goal_yaw']), math.cos(yaw-current_goal['goal_yaw']))
    fine = target.get('approach_phase') == 'precise_docking'
    if np.linalg.norm(point[:2]-np.asarray(current_goal['goal_map'])[:2]) < (.12-.11*blend) and abs(yaw_delta) < math.radians(5.-4.5*blend):
        return None
    return dict(current_goal, goal_map=[float(point[0]),float(point[1]),0.],goal_yaw=yaw)


def fine_alignment_ready(feedback, target, base_in_map, now):
    """Live geometry can end navigation, but the parent must still verify at rest."""
    from .perception import arrived
    if target.get('approach_phase') not in ('continuous_approach','precise_docking') or feedback is None:
        return False
    try:
        center = validate_feedback(feedback, target, now, max_age=MAX_VISUAL_AGE)
    except FeedbackExpired:
        return False
    normal = np.asarray(feedback.get('surface_normal_map'), float)
    transform = np.asarray(base_in_map, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal[:2]) < .9:
        return False
    if transform.shape != (4,4) or not np.isfinite(transform).all():
        return False
    normal = normal / np.linalg.norm(normal[:2])
    reference = np.asarray(target['surface_normal_map'],float)
    if normal @ reference < math.cos(math.radians(25)):
        return False
    relative = transform[:3,:3].T @ (center-transform[:3,3])
    n = transform[:3,:3].T @ normal
    n = n[:2]/np.linalg.norm(n[:2])
    measurement = dict(distance=float(-relative[:2]@n),
                       lateral=float(relative[:2]@np.array([-n[1],n[0]])),
                       heading_error=math.atan2(-n[1],-n[0]),
                       detection_source='object_lidar')
    stand = np.linalg.norm(np.asarray(target['goal_map'])[:2]-np.asarray(target['board_map'])[:2])
    return arrived(measurement,stand,require_aruco=False)


def continuous_goal(target, standoff):
    """One final face-normal goal, no staging waypoint or mode transition."""
    result=dict(target)
    normal=np.asarray(target['surface_normal_map'],float).copy()
    normal[2]=0.
    if not np.isfinite(normal).all() or np.linalg.norm(normal)<.9:
        raise RuntimeError('Invalid target surface normal')
    normal/=np.linalg.norm(normal)
    center=np.asarray(target['board_map'],float)
    goal=center+standoff*normal
    result.update(approach_phase='continuous_approach', standoff=standoff,
                  goal_map=[float(goal[0]),float(goal[1]),0.],
                  goal_yaw=math.atan2(-normal[1],-normal[0]))
    return result


def proximity_weight(target, robot_xy):
    """Blend continuously over 2.5 to 1.2 m; missing pose keeps conservative gain."""
    if robot_xy is None:return 0.
    xy=np.asarray(robot_xy,float)
    center=np.asarray(target['board_map'],float)[:2]
    normal=np.asarray(target['surface_normal_map'],float)[:2]
    if xy.shape!=(2,) or not np.isfinite(xy).all() or np.linalg.norm(normal)<.9:
        return 0.
    distance=float((xy-center)@(normal/np.linalg.norm(normal)))
    return float(np.clip((2.5-distance)/1.3,0.,1.))
