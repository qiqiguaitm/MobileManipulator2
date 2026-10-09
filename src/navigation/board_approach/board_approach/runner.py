"""One command: observe -> plan -> guarded approach -> re-observe -> verify."""
from .limits import MAX_TOTAL_PATH_M, MOTION_PROCESS_TIMEOUT_S
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
import cv2
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory
import yaml
from .perception import BoardUnavailable, DetectionError, detect_board, detect_board_region, pose_matrix, target_geometry, arrived
from .ros_io import RobotIO
from .safety import InputFreshnessError
from .motion import parent_death_signal, write_atomic
from .route import short_waypoint
from .coarse import AppearanceDetector
from .object_center import ObjectDetector, center_target_geometry
from .gpt_detector import GPTDetector, DeepSeekDetector, DetectorFormatError
from .qwen_detector import QwenDetector
from .diagnostics import save_failure
from .surface import surface_observation, surface_target_geometry, approach_goal, observation_heading_limit
from .continuous import continuous_goal, VisualMonitor
from .lidar_geometry import measure_lidar
from .transition import moving_transition_target


def motion_process_setup():
    # Separate killable process group, retaining the operator's controlling terminal
    # so sudo -v authentication remains valid in the motion child.
    os.setpgrp()
    parent_death_signal()


def stop_group(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def camera_transform(node):
    path = Path(get_package_share_directory('camera_driver'))/'config/extrinsics_lidar_to_top_camera_optical_frame.yaml'
    calibration = yaml.safe_load(path.read_text())
    if calibration['frame_id'] != 'rs16_lidar':
        raise RuntimeError('Unexpected calibration parent frame')
    # robot02 calls the same physical laser frame rslidar in its current URDF.
    t = calibration['transform']
    lidar_from_camera = pose_matrix([t['translation'][a] for a in 'xyz'],
                                   [t['rotation'][a] for a in ('x', 'y', 'z', 'w')])
    return node.transform('base_link', 'rslidar') @ lidar_from_camera


def record_timing(directory, stage, event):
    with (Path(directory)/'timing.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,event=event,monotonic=time.monotonic()))+'\n')


def stationary_snapshot(node, directory, timeout=3.):
    """Retry temporary stationary synchronization failures, never unsafe geometry."""
    deadline = time.monotonic()+timeout
    failures = []
    while True:
        try:
            return node.snapshot()
        except (RuntimeError, DetectionError) as error:
            message = str(error)
            retryable = isinstance(error, InputFreshnessError) or message.startswith((
                'No fresh RGB', 'No synchronized RGB / aligned depth pair',
                'No pose close to camera timestamp'))
            if message.startswith('No synchronized RGB/LiDAR/odometry observation:'):
                retryable = any(reason in message for reason in (
                    'Odometry history does not bracket', 'Odometry gap too large',
                    'No LiDAR scan within 60 ms'))
            if not retryable:
                raise
            failures.append(message)
            with (Path(directory)/'snapshot_retries.jsonl').open('a') as f:
                f.write(json.dumps({'monotonic':time.monotonic(), 'error':message})+'\n')
            if time.monotonic() >= deadline:
                raise InputFreshnessError('Stationary synchronization retry budget exhausted: '+
                                          '; '.join(dict.fromkeys(failures))) from error
            node.pump(min(.05, max(0.,deadline-time.monotonic())))


def reuse_moving_transition(node, monitor, standoff, pending_timeout=3.):
    """Drain only an ALREADY captured inference before declaring fallback.

    Never capture a stopped confirmation frame here. A completed worker is kept
    alive for precise docking, and all deadlines/consistency limits still apply.
    """
    started = time.monotonic()
    while True:
        node.health()
        monitor.poll(capture=False)
        p = node.data['pose'].pose.pose
        pose = pose_matrix([p.position.x,p.position.y,p.position.z],
                           [p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
        target = moving_transition_target(monitor.recent_reports, monitor.target, pose,
                                          time.monotonic(), standoff)
        if target is not None:
            target['transition_pending_wait_seconds'] = time.monotonic()-started
            return target
        if monitor.future is None or time.monotonic()-started >= pending_timeout:
            return None
        node.pump(.05)


def wait_for_stationary_transition(node, timeout=3.):
    """Only the fallback path waits for measured rest, never accepts moving frames."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        node.pump(.05)
        try:
            node.health()
        except InputFreshnessError:
            continue
        velocity = node.data['odom'].twist.twist
        if abs(velocity.linear.x) <= .03 and abs(velocity.angular.z) <= .04:
            return
    raise RuntimeError('Robot did not settle for stationary transition fallback')


def observe(node, directory, standoff, use_aruco=True, object_detector=None, precise=False, geometry_source='depth', required_observations=3):
    if required_observations not in (1,2,3):
        raise ValueError('Observation count must be 1, 2 or 3')
    targets = []
    appearance = AppearanceDetector()
    last_detection_error = None
    for index in range(10):
        image, depth, k, pose, timestamp = stationary_snapshot(node, directory)
        captured_monotonic = time.monotonic()-.8  # Conservative bound on accepted snapshot age.
        lidar_frame = node.last_lidar_frame if geometry_source=='lidar' else None
        cv2.imwrite(str(directory/('rgb_%d.png' % index)), image)
        if depth is not None:
            np.save(directory/('depth_%d.npy' % index), depth)
        if lidar_frame is not None:
            np.savez_compressed(directory/('sensor_%d.npz'%index),points_base=lidar_frame['points_base'],
                                rings=lidar_frame['rings'],distortion=lidar_frame['distortion'])
        (directory/('frame_%d.json' % index)).write_text(json.dumps({
            'timestamp_ns': timestamp, 'camera_matrix': k.tolist(), 'robot_pose': pose.tolist(),
            'geometry_source':geometry_source,'lidar':lidar_frame['metadata'] if lidar_frame else None}))
        if not use_aruco and object_detector is None:
            raise RuntimeError('Object detector is required when ArUco is disabled')
        detect = detect_board if use_aruco else object_detector.detect
        try:
            try:
                if geometry_source=='lidar':
                    observation = detect_board_region(image) if use_aruco else object_detector.detect_region(image)
                else:
                    observation = detect(image, depth, k)
            except BoardUnavailable as error:
                last_detection_error = error
                (directory/('frame_%d_error.json' % index)).write_text(json.dumps({'error': str(error)}))
                if geometry_source=='lidar' or not use_aruco or index < 3 or any(t['detection_source'] != 'appearance_depth' for t in targets):
                    continue
                observation = appearance.detect(image, depth, k)
        except BoardUnavailable as error:
            last_detection_error = error
            (directory/('frame_%d_error.json' % index)).write_text(json.dumps({'error': str(error)}))
            continue
        except DetectorFormatError as error:
            last_detection_error = error
            (directory/('frame_%d_error.json' % index)).write_text(json.dumps({'error': str(error), 'phase':'detector_format', 'retryable':True}))
            continue
        except Exception as error:
            (directory/('frame_%d_error.json' % index)).write_text(json.dumps({'error': str(error)}))
            raise
        finally:
            raw_response = getattr(object_detector, 'last_response', None) if not use_aruco else None
            if raw_response is not None:
                (directory/('detection_%d_response.json' % index)).write_text(json.dumps(raw_response, ensure_ascii=False, indent=2))
            # Persist model boxes even when depth validation or later consistency checks fail.
            metadata = getattr(object_detector, 'last_metadata', None) if not use_aruco else None
            if metadata:
                (directory/('detection_%d.json' % index)).write_text(json.dumps(metadata, indent=2))
                annotated = image.copy()
                x1, y1, x2, y2 = np.rint(metadata['bbox']).astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(annotated, 'Frame %d: model detection (not yet validated)' % index,
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
                cv2.imwrite(str(directory/('detection_%d.png' % index)), annotated)
        if not (directory/('detection_%d.png' % index)).exists():
            annotated = image.copy()
            cv2.polylines(annotated, [observation.polygon.astype(np.int32)], True, (0, 220, 255), 2)
            cv2.imwrite(str(directory/('detection_%d.png' % index)), annotated)
        if observation.source != 'appearance_depth' and targets and targets[0]['detection_source'] == 'appearance_depth':
            targets.clear()  # Three consistent precise observations, never a coarse/fine mixture.
        base_from_camera = camera_transform(node)
        frame_path = directory/('frame_%d.json'%index)
        frame_metadata = json.loads(frame_path.read_text())
        frame_metadata['base_from_camera'] = base_from_camera.tolist()
        frame_path.write_text(json.dumps(frame_metadata))
        try:
            if geometry_source=='lidar':
                observation = measure_lidar(observation,lidar_frame,k,base_from_camera,image,
                                            directory/('frame_%d'%index))
            else:
                observation = surface_observation(observation, depth, k)
            target = surface_target_geometry(observation, base_from_camera, pose, standoff)
            target['geometry_source'] = geometry_source
        except DetectionError as error:
            # A bad depth frame must not abort acquisition after two good frames.
            # Retain evidence and retry within the existing ten-frame budget.
            last_detection_error = error
            (directory/('frame_%d_error.json' % index)).write_text(json.dumps({
                'error': str(error), 'phase': 'surface_geometry', 'retryable': True}))
            node.verify_stationary_since(pose)
            continue
        target['observed_robot_xy'] = pose[:2, 3].tolist()
        target['scan_frame'] = node.data['scan'].header.frame_id
        target['base_from_scan'] = node.transform('base_link', target['scan_frame']).tolist()
        target['timestamp_ns'] = timestamp
        target['captured_monotonic'] = captured_monotonic
        if not use_aruco and getattr(object_detector, 'last_metadata', None):
            target['detector'] = object_detector.last_metadata
        node.verify_stationary_since(pose)
        targets.append(target)
        (directory/'observations.json').write_text(json.dumps(targets, indent=2))
        if len(targets) == required_observations:
            break
    if len(targets) < required_observations:
        if not targets and isinstance(last_detection_error, BoardUnavailable):
            raise BoardUnavailable(str(last_detection_error))
        raise DetectionError(f'Fewer than {required_observations} valid board observations: '+str(last_detection_error))
    centers = np.array([x['board_map'] for x in targets])
    robot_positions = np.array([x['observed_robot_xy'] for x in targets])
    yaws = np.unwrap([x['goal_yaw'] for x in targets])
    # Tall objects have a variable visible vertical crop. Docking is constrained
    # in the ground plane; retain the same 6 cm XY tolerance for LiDAR mode.
    consistency_centers = centers[:,:2] if geometry_source=='lidar' else centers
    center_spread = float(np.max(np.linalg.norm(consistency_centers-consistency_centers.mean(axis=0), axis=1)))
    yaw_spread = float(np.ptp(yaws))
    yaw_limit = .12 if precise else observation_heading_limit(targets, standoff)
    if center_spread > .06 or yaw_spread > yaw_limit:
        raise RuntimeError(f'Object estimate unstable across {len(targets)} independent frames: centre spread={center_spread:.3f}m (limit 0.060m), heading span={yaw_spread:.3f}rad (limit {yaw_limit:.3f}rad)')
    if np.max(np.linalg.norm(robot_positions-robot_positions[0], axis=1)) > .025:
        raise RuntimeError('Robot moved during observation')
    target = dict(targets[-1])
    target['coarse_orientation'] = yaw_limit > .12
    target['heading_span_rad'] = yaw_spread
    target['center_spread_m'] = center_spread
    target['center_check_axes'] = 'map_xy' if geometry_source=='lidar' else 'map_xyz'
    target['visible_center_height_span_m'] = float(np.ptp(centers[:,2]))
    target['heading_limit_rad'] = yaw_limit
    target['stationary_observations'] = len(targets)
    target['use_aruco'] = use_aruco
    target['identity_verified'] = all(t['identity_verified'] for t in targets)
    for field in ('distance', 'lateral', 'heading_error'):
        target[field] = float(np.mean([x[field] for x in targets]))
    target['board_map'] = centers.mean(axis=0).tolist()
    target['object_map'] = target['board_map']
    target['goal_map'] = np.mean([x['goal_map'] for x in targets], axis=0).tolist()
    target['goal_yaw'] = float(np.mean(yaws))
    normals = np.array([t['surface_normal_map'] for t in targets])
    normal = normals.mean(axis=0)
    normal /= np.linalg.norm(normal)
    target['surface_normal_map'] = normal.tolist()
    goal = centers.mean(axis=0) + standoff * normal
    target['goal_map'] = [float(goal[0]), float(goal[1]), 0.]
    cv2.polylines(image, [observation.polygon.astype(np.int32)], True, (0, 220, 0), 2)
    cv2.putText(image, 'Object: %.2f m | markers: %d' % (target['distance'], len(observation.ids)),
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 0), 2)
    cv2.imwrite(str(directory/'observation.png'), image)
    (directory/'observations.json').write_text(json.dumps(targets, indent=2))
    return target


def main():
    parser = argparse.ArgumentParser(description='robot02 object-centre approach; defaults to stationary check')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Enable real movement; operator must remain on site')
    mode.add_argument('--check', action='store_true', help='Observe and plan only; never enable chassis')
    parser.add_argument('--resume-precise', action='store_true', help='Deprecated compatibility option; all runs now use continuous approach')
    parser.add_argument('--segmented', action='store_true', help='Legacy compatibility flag; the runner uses continuous guarded Nav2')
    detection_mode = parser.add_mutually_exclusive_group()
    detection_mode.add_argument('--aruco', action='store_true', dest='aruco', help='Use ArUco board detection without model API calls')
    detection_mode.add_argument('--no-aruco', action='store_false', dest='aruco', help='Use object detection and depth (default)')
    parser.set_defaults(aruco=False)
    parser.add_argument('--target', default='空气净化器', help='Unique object or front-surface region description')
    parser.add_argument('--detector', choices=('deepseek','qwen','gpt','dinox'), default='deepseek', help='Object detector backend (default: deepseek)')
    parser.add_argument('--geometry-source',choices=('lidar','depth'),default=None,
                        help='Default: LiDAR with object detection; camera depth with --aruco. Never silently falls back.')
    parser.add_argument('--startup-observations', type=int, choices=(2,3), default=2, help='Required valid startup observations within 10 attempts (default: 2); consistency checks still apply')
    parser.add_argument('--qwen-model', default=None, help='Default configured Qwen vision model or qwen3-vl-plus')
    parser.add_argument('--deepseek-model', default=None, help='Default configured model or deepseek-flash')
    parser.add_argument('--gpt-model', default=None, help='OpenAI image-capable model; default configured model or gpt-4.1')
    parser.add_argument('--detector-url', default=os.environ.get('OBJECT_DETECTOR_URL', 'http://192.168.112.14:10086'), help='DINO-X HTTP service base URL')
    parser.add_argument('--standoff', type=float, default=0.934, help='Distance from base_link, 0.5–2.0 m; default 0.934 gives about 0.5 m front clearance')
    parser.add_argument('--project-root', default=os.environ.get('MM2_PROJECT_ROOT', '/home/didi/junbin.zhang/MobileManipulator2'))
    args = parser.parse_args()
    args.geometry_source = args.geometry_source or ('depth' if args.aruco else 'lidar')
    cv2.setNumThreads(2)
    if not math.isfinite(args.standoff) or not 0.5 <= args.standoff <= 2.0:
        parser.error('standoff must be between 0.5 and 2.0 m')
    lock = open('/tmp/board-approach-%s.lock' % os.getuid(), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another board approach is running')
    root = Path(args.project_root).resolve()
    directory = root/'log/board_approach'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    result = {'success': False, 'execute': args.execute, 'report_dir': str(directory), 'use_aruco': args.aruco, 'target_prompt': args.target, 'alignment': 'perpendicular_to_surface', 'detector': 'aruco' if args.aruco else args.detector}
    result['geometry_source'] = args.geometry_source
    print('模式：' + ('实际行走' if args.execute else '只检查，不行驶'), flush=True)
    print('对准方式：' + ('ArUco 编号验证' if args.aruco else args.detector+' 物体检测') +
          ('＋LiDAR 距离和表面法向' if args.geometry_source=='lidar' else '＋双目深度平面'), flush=True)
    print('报告目录：'+str(directory), flush=True)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    node = RobotIO(geometry_source=args.geometry_source)
    camera = motion = None
    monitor = None
    current = None
    try:
        # Check only the narrowly permitted, read-only CAN command before paid inference.
        if args.execute:
            permission = subprocess.run(['sudo', '-n', '/usr/sbin/ip', 'link', 'show', 'can2'],
                                        capture_output=True, text=True, timeout=5)
            if permission.returncode:
                raise RuntimeError('CAN sudo permission unavailable; install the can2-only sudoers rule: '+permission.stderr.strip())
        # Fail before any camera start or movement if GPT credentials are missing.
        object_detector = None
        if not args.aruco:
            if args.detector == 'qwen':
                object_detector = QwenDetector(args.target, model=args.qwen_model, max_calls=120)
            elif args.detector == 'deepseek':
                object_detector = DeepSeekDetector(args.target, model=args.deepseek_model, max_calls=120)
            elif args.detector == 'gpt':
                object_detector = GPTDetector(args.target, model=args.gpt_model, max_calls=120)
            else:
                object_detector = ObjectDetector(args.detector_url, args.target + ' . visible front face')
        discovery_deadline = time.monotonic()+10
        while time.monotonic() < discovery_deadline:
            node.pump(.2)
            if (all(k in node.data for k in ('pose', 'odom', 'scan', 'status'))
                    and node.tf.can_transform('map', 'base_link', rclpy.time.Time())
                    and node.tf.can_transform('base_link', 'rslidar', rclpy.time.Time())):
                break
        node.health()  # Do not start or reset localization implicitly.
        if node.active_goals:
            raise RuntimeError('Another Nav2 goal is active; finish it first')
        if 'rgb' not in node.data or time.monotonic()-node.received.get('rgb', 0) > 1:
            # Never start a second process against an occupied camera.
            existing = subprocess.run(['pgrep', '-f', '^/.*realsense2_camera_node( |$)'], capture_output=True)
            if existing.returncode == 0:
                # ROS discovery of the image stream may lag localization/TF discovery.
                deadline = time.monotonic()+10
                while time.monotonic()-node.received.get('rgb', 0) > 1 and time.monotonic() < deadline:
                    node.pump(.2)
                if time.monotonic()-node.received.get('rgb', 0) > 1:
                    raise RuntimeError('Camera process exists but is not publishing here; check ROS domain/localhost settings')
            else:
                camera = subprocess.Popen([str(root/'scripts/run_top_camera_rsusb.sh')],
                    stdout=open(directory/'camera.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+25
            while 'rgb' not in node.data and time.monotonic() < deadline:
                node.pump(.2)
                if camera is not None and camera.poll() is not None:
                    raise RuntimeError('Camera startup failed; inspect camera.log')
        previous_board = None
        previous_normal = None
        total_path = 0.
        for stage in (0, 2):  # Initial acquisition, then endpoint verification; one motion leg.
            current = directory/('approach' if stage == 0 else 'arrival')
            current.mkdir()
            record_timing(directory,stage,'observation_start')
            reused = None
            if stage == 2 and monitor is not None:
                reused = reuse_moving_transition(node, monitor, args.standoff,
                                                 pending_timeout=0.)
            if reused is not None:
                target = reused
                target['arrival_verification_source'] = 'latest_moving_observation_at_current_pose'
                (current/'arrival_observation.json').write_text(json.dumps(target, indent=2))
                print('终点使用最新有效异步测量，按当前位姿验收，不重复检测。', flush=True)
                record_timing(directory,stage,'moving_observations_reused')
            else:
                if monitor is not None:
                    monitor.close()
                    monitor = None
                if stage == 2:
                    wait_for_stationary_transition(node)
                    record_timing(directory,stage,'stationary_fallback')
                print('终点无可用新鲜测量，补测一帧。' if stage == 2 else '自动检测并测量目标（第 %d 次）' % (stage+1), flush=True)
                target = observe(node, current, args.standoff, use_aruco=args.aruco, object_detector=object_detector,
                                 precise=stage > 0,geometry_source=args.geometry_source,
                                 required_observations=1 if stage == 2 else args.startup_observations)
            record_timing(directory,stage,'observation_complete')
            visual = target['identity_verified'] or (not args.aruco and target['detection_source'] in ('object_depth','object_lidar'))
            if not visual:
                raise RuntimeError('Target identity unavailable for continuous approach')
            reference = previous_board
            if reference is not None and np.linalg.norm(np.array(target['board_map'])[:2]-np.array(reference)[:2]) > .18:
                raise RuntimeError('Target moved / localization shifted after approach')
            previous_board = np.array(target['board_map'])
            normal = np.array(target['surface_normal_map'])
            if previous_normal is not None and np.dot(previous_normal, normal) < math.cos(math.radians(25)):
                raise RuntimeError('Target surface normal changed after approach; refusing an inconsistent docking pose')
            previous_normal = normal
            print(json.dumps({k: target[k] for k in ('distance', 'lateral', 'heading_error', 'marker_ids')}, ensure_ascii=False), flush=True)
            if not target['coarse_orientation'] and arrived(target, args.standoff, require_aruco=args.aruco):
                result.update(success=True, state='already_arrived' if stage == 0 else 'arrived', measurement=target)
                break
            if target['distance'] < args.standoff-.10:
                raise RuntimeError('Already too close; automatic reversing is disabled')
            if stage == 2:
                raise RuntimeError('Final alignment verification failed; no additional stop-and-go attempts')
            target = continuous_goal(target, args.standoff)
            target['path_length'] = node.plan(target)
            total_path += target['path_length']
            if total_path > MAX_TOTAL_PATH_M:
                raise RuntimeError(f'Total planned path {total_path:.2f} m exceeds {MAX_TOTAL_PATH_M:.1f} m limit')
            (current/'goal.json').write_text(json.dumps(target, indent=2))
            monitor = VisualMonitor(node, object_detector, args.aruco, camera_transform(node), target, current,
                                    geometry_source=args.geometry_source)
            # Start inference now; chassis initialization can overlap it while held at zero.
            # The motion child still requires fresh validated feedback before enabling Nav2.
            monitor.seed_from_observation(target)
            monitor.poll()
            if not args.execute:
                deadline = time.monotonic()+25
                while monitor.valid_updates < 3 and time.monotonic() < deadline:
                    node.pump(.1)
                    monitor.poll()
                if monitor.valid_updates < 3:
                    raise RuntimeError('No fresh visual/geometry feedback before motion: '+monitor.failure_summary())
            if not args.execute:
                result.update(success=True, state='checked', measurement=target,async_preflight=True,
                              async_valid_updates=monitor.valid_updates)
                monitor.close()
                monitor = None
                break
            command = [sys.executable, '-m', 'board_approach.motion', '--goal', str(current/'goal.json'),
                       '--result', str(current/'motion.json'), '--speed', '.10']
            command.extend(['--visual-feedback', str(monitor.path)])
            record_timing(directory,stage,'motion_process_start')
            motion = subprocess.Popen(command, preexec_fn=motion_process_setup)
            deadline = time.monotonic()+MOTION_PROCESS_TIMEOUT_S
            stopping = current/'motion.stopping.json'
            while motion.poll() is None:
                node.pump(.2)
                monitor.poll(capture=motion.poll() is None and not stopping.exists())
                if time.monotonic() > deadline:
                    raise RuntimeError('Motion process deadline exceeded')
            record_timing(directory,stage,'motion_process_complete')
            motion_result = current/'motion.json'
            leg_result = json.loads(motion_result.read_text()) if motion_result.exists() else {}
            if not leg_result.get('success') or motion.poll() not in (None, 0):
                motion_result = current/'motion.json'
                reason = json.loads(motion_result.read_text()).get('error', 'Unknown motion error') if motion_result.exists() else 'Motion process exited without a report'
                raise RuntimeError('Motion guard stopped: '+reason+'; report: '+str(motion_result))
            motion = None
            # Preserve the same-run recent fits for endpoint verification.
            # Stage 2 does not start a new inference if a fresh fit is available.
            record_timing(directory,stage,'observer_continued')
            # The chassis is stopped before endpoint verification.
        if not result['success']:
            raise RuntimeError('Arrival not verified')
    except (Exception, KeyboardInterrupt) as error:
        result.update(success=False, state='stopped', error=str(error) or 'Interrupted')
    finally:
        for owned_process in (motion,):
            try:
                stop_group(owned_process)
            except Exception as error:
                result.update(success=False, cleanup_error=str(error))
        if monitor is not None:
            monitor.close()
        if not result['success']:
            try:
                result['failure_dir'] = save_failure(directory, current,
                    result.get('error', result.get('cleanup_error', 'Unknown failure')), node, camera_transform=camera_transform)
                print('失败图片及原因：' + result['failure_dir'], flush=True)
            except Exception as error:
                result['diagnostic_error'] = str(error)
        try:
            stop_group(camera)
        except Exception as error:
            result.update(success=False, cleanup_error=str(error))
        node.destroy_node()
        rclpy.shutdown()
        (directory/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        lock.close()
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
