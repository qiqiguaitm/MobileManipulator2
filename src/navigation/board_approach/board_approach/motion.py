"""Separate process for bounded Nav2 motion. No image processing runs here."""
from .limits import (MAX_STAGE_TRAVEL_M, MAX_STAGE_DISPLACEMENT_M, MAX_TOTAL_PATH_M, MOTION_TIMEOUT_S,
                     FINE_XY_GOAL_TOLERANCE_M, FINE_YAW_GOAL_TOLERANCE_RAD)
from collections import deque
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import uuid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from action_msgs.srv import CancelGoal
from unique_identifier_msgs.msg import UUID
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from .ros_io import RobotIO, stamp
from .perception import pose_matrix
from .endpoint import ApproachObservationGate, approach_refresh_goal, latency_speed_limit, endpoint_correction, MAX_ENDPOINT_CORRECTIONS, ENDPOINT_BUDGET_S, PENDING_WAIT_S
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from .safety import forward_bounds, can_error, guarded_velocity, InputFreshnessError, FreshDataGate, smooth_steering, APPROACH_MAX_WZ, APPROACH_MAX_AZ
from .continuous import (validate_feedback, corrected_navigation_goal, FeedbackExpired,
                         fine_alignment_ready, proximity_weight, feedback_required_near_goal, MAX_START_VISUAL_AGE, START_FEEDBACK_TIMEOUT)


def approach_controller_parameters(speed, fine=False):
    # TEB treats a non-positive backwards limit as UNLIMITED. A tiny positive
    # bound plus a strong forward penalty is required; the guard clamps all
    # remaining negative commands to zero. Keep epsilon below that bound.
    values = {'FollowPath.max_vel_x': speed,
              'FollowPath.max_vel_x_backwards': .001,
              'FollowPath.max_vel_y': 0.,
              'FollowPath.max_vel_theta': APPROACH_MAX_WZ,
              'FollowPath.acc_lim_theta': APPROACH_MAX_AZ,
              'FollowPath.penalty_epsilon': .0001,
              'FollowPath.weight_kinematics_forward_drive': 1000.}
    if fine:
        values.update({'goal_checker.xy_goal_tolerance': FINE_XY_GOAL_TOLERANCE_M,
                       'goal_checker.yaw_goal_tolerance': FINE_YAW_GOAL_TOLERANCE_RAD,
                       'progress_checker.required_movement_radius': .01,
                       'progress_checker.movement_time_allowance': 40.})
    return values


def configure_approach_controller(node, desired, originals):
    # Save before setting so even a partial rejection can be restored by finally.
    for name, value in node.parameters(names=list(desired)).items():
        originals.setdefault(name, value)
    node.parameters(desired)
    actual = node.parameters(names=list(desired))
    if any(abs(actual[k]-v) > 1e-9 for k, v in desired.items()):
        raise RuntimeError('Approach controller parameters did not take effect')


def hold_for_navigation_retry(node, driver, attempt, action_status, timeout=1.5):
    """Only retry an aborted action, at zero velocity, with healthy live sensors."""
    node.enabled = False
    node.command, node.command_time = Twist(), time.monotonic()
    node.output.publish(Twist())
    if action_status != 6 or attempt > 2:
        raise RuntimeError(f'Navigation failed: status={action_status}, retries={attempt-1}')
    if hasattr(node, 'capture_navigation_failure'):
        node.capture_navigation_failure(attempt, action_status)
    node.record({'stage':'navigation_retry_wait', 'attempt':attempt, 'status':action_status})
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        node.pump(.1)
        node.output.publish(Twist())
        if node.failure or driver.poll() is not None:
            raise RuntimeError(node.failure or 'Chassis driver exited during retry')
        node.health()
        problem=can_error(node.can_state,time.monotonic()-node.can_time)
        if problem: raise RuntimeError(problem)
    node.command, node.command_time = Twist(), time.monotonic()



def endpoint_observation(node, driver, deadline):
    """Stop, drain the in-flight observation without starting another capture."""
    node.enabled=False
    node.command,node.command_time=Twist(),time.monotonic()
    node.output.publish(Twist())
    directory=node.visual_feedback.parent
    write_atomic(directory/'endpoint_wait.json',dict(monotonic=time.monotonic()))
    node.record({'stage':'endpoint_wait_start'})
    # Give the parent one poll to see the hold, then drain only its pending work.
    until=min(deadline,time.monotonic()+PENDING_WAIT_S)
    node.pump(.25)
    while True:
        node.output.publish(Twist())
        if node.failure or driver.poll() is not None:
            raise RuntimeError(node.failure or 'Chassis driver exited at endpoint')
        node.health()
        problem=can_error(node.can_state,time.monotonic()-node.can_time)
        if problem:raise RuntimeError(problem)
        if not (directory/'pending.json').exists() or time.monotonic()>=until:break
        node.pump(.05)
    feedback=motion_feedback(node)
    p=node.data['pose'].pose.pose
    transform=pose_matrix([p.position.x,p.position.y,p.position.z],
                          [p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
    node.record({'stage':'endpoint_wait_complete','has_fresh_feedback':feedback is not None})
    return feedback,transform


def parent_death_signal():
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(2)
    if os.getppid() == 1:
        os._exit(2)


def interrupt(signum, frame):
    raise KeyboardInterrupt()


def can_command(*args):
    completed = subprocess.run(['sudo', '-n', '/usr/sbin/ip', 'link', 'set', 'can2', *args],
                               timeout=5, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError('CAN interface command failed ('+' '.join(args)+'): '+completed.stderr.strip())


def wait_for_start_feedback(node, timeout=START_FEEDBACK_TIMEOUT):
    """Keep zero output during initialization until feedback is freshly confirmed."""
    if node.enabled:
        raise RuntimeError('Startup feedback wait must keep motion disabled')
    if not node.visual_feedback:
        return
    deadline=time.monotonic()+timeout
    last_error='No feedback file'
    while time.monotonic()<deadline:
        node.pump(.1)  # enabled remains False: guard continues publishing zero.
        node.health()
        try:
            startup_path=node.visual_feedback.with_name('startup_feedback.json')
            feedback=json.loads((startup_path if startup_path.exists() else node.visual_feedback).read_text())
            validate_feedback(feedback,node.target,time.monotonic(), max_age=MAX_START_VISUAL_AGE)
            # Motion remains disabled; normal goal updates retain the 6 s bound.
            return
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            last_error=str(error)
    raise RuntimeError('No fresh visual/geometry feedback after chassis initialization: '+last_error)


def wait_for_driver_subscription(node, driver, timeout=12.):
    """Wait for graph discovery without enabling movement; reject extra owners."""
    deadline = time.monotonic()+timeout
    last = None
    while time.monotonic() < deadline:
        node.pump(.1)
        code = driver.poll()
        count = node.output.get_subscription_count()
        endpoints = [{'name':e.node_name, 'namespace':e.node_namespace}
                     for e in node.get_subscriptions_info_by_topic('/board_approach/safe_cmd_vel')]
        current = {'returncode':code, 'subscription_count':count, 'endpoints':endpoints}
        if current != last:
            node.record(dict(stage='driver_discovery', **current))
            last = current
        if code is not None:
            raise RuntimeError(f'Chassis driver exited during startup: returncode={code}')
        problem = can_error(node.can_state, time.monotonic()-node.can_time)
        if problem:
            raise RuntimeError(problem)
        if count > 1 or len(endpoints) > 1:
            raise RuntimeError('Multiple guarded command subscribers: '+json.dumps(current))
        if (count == 1 and len(endpoints) == 1
                and endpoints[0]['name'] and not endpoints[0]['name'].startswith('_NODE_')
                and endpoints[0]['namespace'].startswith('/')):
            return
    raise RuntimeError('Chassis driver alive but guarded ROS subscription not ready: '+json.dumps(last))


def wait_for_can_mode(node, driver, timeout=7.):
    """Wait at zero output for fresh CAN-mode feedback; never clear faults."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        node.pump(.05)
        if driver.poll() is not None:
            raise RuntimeError('Chassis driver exited during CAN initialization')
        age = time.monotonic()-node.can_time
        problem = can_error(node.can_state, age, require_can=False)
        if problem:
            raise RuntimeError(problem)
        if node.can_state[1] == 1:
            node.record({'stage':'can_mode_ready', 'raw_can_state':list(node.can_state), 'age':age})
            return
    raise RuntimeError('CAN control mode transition timed out; '+str(
        can_error(node.can_state, time.monotonic()-node.can_time)))


def motion_feedback(node, force=False):
    """Fresh valid observations may update a leg; missing/stale ones keep its goal.

    The parent performs new stationary observations after each completed leg.
    force is retained for callers at completion; it does not require camera freshness.
    """
    if not node.visual_feedback:
        return None
    try:
        feedback = json.loads(node.visual_feedback.read_text())
        validate_feedback(feedback, node.target, time.monotonic())
        return feedback
    except (FeedbackExpired, FileNotFoundError):
        if node.target.get('approach_phase') in ('continuous_approach', 'continuous_coarse', 'precise_docking'):
            return None
        raise


class Motion(RobotIO):
    def __init__(self, target, speed, segmented=False):
        super().__init__(with_camera=False)
        self.target, self.speed = target, speed
        self.enabled, self.failure = False, None
        self.visual_feedback = None
        self.paused, self.fresh_gate, self.events = False, FreshDataGate(), []
        self.can = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.can.bind(('can2',))
        self.can.setblocking(False)
        self.can_state, self.can_time = None, 0.
        self.command, self.command_time = Twist(), 0.
        self.output = self.create_publisher(Twist, '/board_approach/safe_cmd_vel', 10)
        self.segmented = segmented
        if not segmented:
            self.subs.append(self.create_subscription(Twist, '/cmd_vel', self.on_command, 10))
        self.started, self.start_xy = 0., None
        self.travel, self.last_xy = 0., None
        self.steering_w, self.steering_time = 0., time.monotonic()
        self.costmaps, self.navigation_failures = {}, []
        for name in ('global', 'local'):
            self.subs.append(self.create_subscription(OccupancyGrid, f'/{name}_costmap/costmap',
                lambda message, key=name: self.costmaps.update({key:message}),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        self.control_trace = deque(maxlen=8000)
        self.trace_time = 0.
        self.door_clouds = deque(maxlen=5)
        self.door_filter_cache = None
        self.last_door_filter = None
        self.subs.append(self.create_subscription(PointCloud2, '/rslidar_points_delayed',
            self.door_clouds.append, QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT)))
        self.guard_timer = self.create_timer(.02, self.guard)

    def near_door_ranges(self, scan, feedback):
        # Fail closed to raw /scan if the exact original cloud/pose is unavailable.
        from .door_filter import cloud_xyz, reflection_mask, reconstruct_scan
        raw = np.asarray(scan.ranges,float)
        if not feedback or not feedback.get('door_model') or not hasattr(self,'door_clouds'):
            return raw
        try:
            cloud = next((m for m in reversed(self.door_clouds)
                          if stamp(m)==stamp(scan) and m.header.frame_id==scan.header.frame_id), None)
            if cloud is None or not self.poses:
                return raw
            pose_msg = min(self.poses,key=lambda m:abs(stamp(m)-stamp(scan)))
            if abs(stamp(pose_msg)-stamp(scan))>100_000_000:
                return raw
            key=(stamp(scan),feedback['captured_monotonic'],self.target.get('approach_phase'))
            if self.door_filter_cache is not None and self.door_filter_cache[0]==key:
                return self.door_filter_cache[1]
            p,q=pose_msg.pose.pose.position,pose_msg.pose.pose.orientation
            pose=pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            xyz=cloud_xyz(cloud)
            transform=np.asarray(self.target['base_from_scan'])
            points=xyz@transform[:3,:3].T+transform[:3,3]
            removed=reflection_mask(points,feedback['door_model'],pose,
                                    self.target.get('approach_phase'),time.monotonic())
            ranges,changed=reconstruct_scan(xyz,scan,removed)
            self.door_filter_cache=(key,ranges)
            if changed.any():
                self.last_door_filter=dict(points_base=points,removed=removed,
                    raw_ranges=raw,filtered_ranges=ranges,changed_bins=changed,
                    scan_timestamp_ns=stamp(scan),map_from_base=pose,
                    model_json=json.dumps(feedback['door_model']))
                self.record(dict(stage='near_door_filter',scan_timestamp_ns=stamp(scan),
                                 candidate_points=int(removed.sum()),changed_bins=int(changed.sum())))
            return ranges
        except (ValueError,KeyError,TypeError,StopIteration):
            return raw

    def on_command(self, message):
        self.command, self.command_time = message, time.monotonic()

    def capture_navigation_failure(self, attempt, status):
        # Copy in memory while stopped; write evidence only after CAN teardown.
        metadata=dict(attempt=attempt, action_status=status, monotonic=time.monotonic(),
                      goal=getattr(self,'current_goal',{}), costmaps={})
        arrays={}
        for name,m in self.costmaps.items():
            arrays[name]=np.asarray(m.data,dtype=np.int8).reshape(m.info.height,m.info.width).copy()
            o=m.info.origin
            metadata['costmaps'][name]=dict(frame=m.header.frame_id, stamp_ns=stamp(m),
                resolution=m.info.resolution, origin=[o.position.x,o.position.y,o.position.z],
                orientation=[o.orientation.x,o.orientation.y,o.orientation.z,o.orientation.w])
        pose=self.data['pose'].pose.pose
        metadata['robot_map_xy']=[pose.position.x,pose.position.y]
        self.navigation_failures.append((metadata,arrays))

    def guard(self):
        out = Twist()
        now = time.monotonic()
        dt = now-getattr(self, "steering_time", now-.02)
        self.steering_time = now
        try:
            while True:
                try:
                    raw = self.can.recv(16)
                except BlockingIOError:
                    break
                ident, size, data = struct.unpack('=IB3x8s', raw)
                if ident & 0x7ff == 0x211:
                    self.can_state, self.can_time = list(data[:size]), time.monotonic()
            if self.enabled and not self.failure:
                feedback = None
                if getattr(self, 'visual_feedback', None):
                    feedback = motion_feedback(self)
                if self.segmented and self.active_goals:
                    raise RuntimeError('Another navigation goal started')
                problem = can_error(self.can_state, time.monotonic()-self.can_time)
                if problem:
                    raise RuntimeError(problem)
                reason = None
                try:
                    self.health()
                except InputFreshnessError as error:
                    reason = str(error)
                was_paused = self.paused
                scan_stamp = stamp(self.data['scan']) if 'scan' in self.data else None
                self.paused = not self.fresh_gate.update(time.monotonic(), reason is None, scan_stamp)
                if self.paused != was_paused:
                    self.record({'stage': 'sensor_pause' if self.paused else 'sensor_resumed',
                                 'reason': reason})
                if self.paused:
                    self.command, self.command_time = Twist(), time.monotonic()
                    self.steering_w = 0.
                    self.output.publish(out)
                    return
                p = self.data['pose'].pose.pose.position
                xy = np.array([p.x, p.y])
                if self.last_xy is not None:
                    self.travel += float(np.linalg.norm(xy-self.last_xy))
                self.last_xy = xy
                continuous = self.target.get('approach_phase') == 'continuous_approach'
                travel_limit = MAX_TOTAL_PATH_M if continuous else MAX_STAGE_TRAVEL_M
                displacement_limit = MAX_TOTAL_PATH_M if continuous else MAX_STAGE_DISPLACEMENT_M
                if self.travel > travel_limit or np.linalg.norm(xy-self.start_xy) > displacement_limit:
                    raise RuntimeError('Motion distance budget exceeded')
                scan = self.data['scan']
                if scan.header.frame_id != self.target['scan_frame']:
                    raise RuntimeError('Lidar frame changed')
                ranges = self.near_door_ranges(scan, feedback) if hasattr(self, "near_door_ranges") else np.array(scan.ranges)
                angles = scan.angle_min + np.arange(len(ranges))*scan.angle_increment
                valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
                points = np.column_stack((ranges[valid]*np.cos(angles[valid]),
                                          ranges[valid]*np.sin(angles[valid]), np.zeros(valid.sum())))
                transform = np.array(self.target['base_from_scan'])
                points = points @ transform[:3, :3].T + transform[:3, 3]
                envelope = forward_bounds(self.target,xy)
                age = time.monotonic()-self.command_time
                if age > 1. and time.monotonic()-self.started > 3:
                    raise RuntimeError('Navigation velocity stream stopped')
                speed_limit = self.speed
                if self.target.get('approach_phase') == 'continuous_approach':
                    speed_limit = latency_speed_limit(self.target,xy,feedback,self.speed)
                if age <= .35:
                    out.linear.x, out.angular.z = guarded_velocity(
                        self.command.linear.x, self.command.linear.y,
                        self.command.angular.z, points, speed_limit, bounds=envelope)
                    smoothed = smooth_steering(out.linear.x,out.angular.z,
                                               getattr(self,'steering_w',0.),dt)
                    # Check the ACTUAL smoothed turn too: residual turning may remain
                    # when the new requested angular velocity has reached zero.
                    out.linear.x,out.angular.z = guarded_velocity(out.linear.x,0.,smoothed,points,speed_limit,bounds=envelope)
        except Exception as error:
            if self.failure:
                self.steering_w = 0.
                self.output.publish(out)
                return  # Preserve the first cause during CAN teardown.
            self.failure = str(error)
            # Stop FIRST. Only retain memory here; rendering/file I/O runs after cleanup.
            self.output.publish(Twist())
            if self.failure.startswith('Obstacle in '):
                self.obstacle_evidence = dict(points_base=points.copy(), metadata=dict(
                    error=self.failure,monotonic=time.monotonic(),scan_timestamp_ns=stamp(scan),
                    forward_bounds_m=envelope,
                    scan_frame=scan.header.frame_id,points_frame='base_link',
                    source='exact_LaserScan_used_by_guard_not_full_3D_cloud',
                    base_from_scan=transform.tolist(),command=[self.command.linear.x,self.command.linear.y,self.command.angular.z]))
            self.record({'stage': 'guard_failure', 'error': self.failure,
                         'command_vx': self.command.linear.x,
                         'command_vy': self.command.linear.y,
                         'command_wz': self.command.angular.z,
                         'command_age': time.monotonic()-self.command_time})
            out = Twist()
        self.steering_w = out.angular.z
        self.output.publish(out)
        if hasattr(self,'control_trace') and now-getattr(self,'trace_time',0.) >= .1:
            self.trace_time = now
            entry=dict(monotonic=now,phase=self.target.get('approach_phase'),
                       requested_vx=self.command.linear.x,requested_wz=self.command.angular.z,
                       sent_vx=out.linear.x,sent_wz=out.angular.z,paused=self.paused,failure=self.failure)
            if 'odom' in self.data:
                odom=self.data['odom']
                entry.update(odom_vx=odom.twist.twist.linear.x,odom_wz=odom.twist.twist.angular.z,
                             odom_timestamp_ns=stamp(odom))
            if 'pose' in self.data:
                pose=self.data['pose'];q=pose.pose.pose.orientation
                entry.update(pose_yaw=float(np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))),
                             pose_timestamp_ns=stamp(pose))
            self.control_trace.append(entry)

    def record(self, event):
        event = dict(event, monotonic=time.monotonic())
        self.events.append(event)
        print(json.dumps(event), flush=True)

    def parameters(self, values=None, names=None):
        names = names or ['goal_checker.xy_goal_tolerance', 'goal_checker.yaw_goal_tolerance']
        if values is None:
            client = self.create_client(GetParameters, '/controller_server/get_parameters')
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError('Controller parameter service missing')
            response = self.await_result(client.call_async(GetParameters.Request(names=names)), 4)
            if len(response.values) != len(names) or any(v.type != 3 for v in response.values):
                raise RuntimeError('Cannot read controller parameters: '+', '.join(
                    f'{name} (type={value.type})' for name,value in zip(names,response.values)
                    if value.type != 3))
            return {name: value.double_value for name, value in zip(names, response.values)}
        client = self.create_client(SetParameters, '/controller_server/set_parameters')
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError('Controller parameter service missing')
        request = SetParameters.Request(parameters=[Parameter(name=k,
            value=ParameterValue(type=3, double_value=v)) for k, v in values.items()])
        response = self.await_result(client.call_async(request), 4)
        if len(response.results) != len(values) or not all(x.successful for x in response.results):
            raise RuntimeError('Controller rejected parameters: '+str([x.reason for x in response.results]))


def run_segmented(node, driver, target, speed):
    from .segmented import SegmentedController, validate_straight_path
    node.plan(target)
    validate_straight_path(node.last_plan_points, node.start_xy, target['goal_map'][:2])
    controller = SegmentedController(node.start_xy, target['goal_map'][:2],
                                     target['goal_yaw'], speed)
    node.started, node.enabled = time.monotonic(), True
    last_report = 0.
    while time.monotonic()-node.started < 120:
        if node.failure or driver.poll() is not None:
            raise RuntimeError(node.failure or 'Chassis driver exited')
        try:
            node.health()
        except InputFreshnessError:
            node.on_command(Twist())
            node.output.publish(Twist())
            node.pump(.04)
            continue
        if node.paused:
            node.on_command(Twist())
            node.pump(.04)
            continue
        p = node.data['pose'].pose.pose
        q = p.orientation
        yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        v, w = controller.command([p.position.x, p.position.y], yaw)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        node.on_command(cmd)
        node.pump(.04)
        if node.failure:
            raise RuntimeError(node.failure)
        if controller.phase == 'done':
            node.enabled = False
            node.output.publish(Twist())
            return
        if time.monotonic()-last_report > 3:
            node.record({'stage': controller.phase, 'travel': node.travel,
                         'xy': [p.position.x, p.position.y]})
            last_report = time.monotonic()
    raise RuntimeError('Segmented motion timeout')


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def wait_for_inputs(node, timeout=5.):
    """Start once sensor/CAN discovery is complete, without a fixed sleep."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        node.pump(.05)
        if node.failure:
            raise RuntimeError(node.failure)
        try:
            node.health()
        except InputFreshnessError:
            continue
        if node.can_state is None or time.monotonic()-node.can_time > .4:
            continue
        problem = can_error(node.can_state, time.monotonic()-node.can_time, require_can=False)
        if problem:
            raise RuntimeError(problem)
        return
    raise RuntimeError('Fresh sensor/CAN startup timeout')


def hold_for_next_leg(node, driver, request_path, timeout=60.):
    """Keep the same chassis owner at zero while the parent measures/plans."""
    node.enabled = False
    node.command = Twist()
    node.output.publish(Twist())
    node.record({'stage':'transition_hold_start'})
    deadline = time.monotonic()+timeout
    position = node.data['pose'].pose.pose.position
    anchor = np.array([position.x, position.y])
    fresh = FreshDataGate()
    while time.monotonic() < deadline:
        node.pump(.05)
        if node.failure or driver.poll() is not None:
            raise RuntimeError(node.failure or 'Chassis driver exited during transition')
        problem = can_error(node.can_state, time.monotonic()-node.can_time)
        if problem:
            raise RuntimeError(problem)
        healthy = True
        try:
            node.health()
        except InputFreshnessError:
            healthy = False
        scan_stamp = stamp(node.data['scan']) if 'scan' in node.data else None
        ready = fresh.update(time.monotonic(), healthy, scan_stamp)
        if healthy:
            position = node.data['pose'].pose.pose.position
            if np.linalg.norm(np.array([position.x, position.y])-anchor) > .025:
                raise RuntimeError('Robot moved during transition hold')
        if ready and Path(request_path).exists():
            request = json.loads(Path(request_path).read_text())
            if request.get('action') not in ('finish', 'precise'):
                raise RuntimeError('Invalid transition request')
            node.record({'stage':'transition_hold_complete'})
            return request
    raise RuntimeError('Stationary transition timed out')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--speed', type=float, choices=[.05, .08, .10], required=True)
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--next-request', help='Hold at zero after coarse success until a precise/finish request')
    parser.add_argument('--segmented', action='store_true')
    parser.add_argument('--visual-feedback', help='Atomic timestamped feedback written by the camera process')
    args = parser.parse_args()
    target = json.loads(Path(args.goal).read_text())
    if not np.isfinite(target['goal_map'] + [target['goal_yaw']]).all():
        raise RuntimeError('Invalid goal')
    # Refuse an already owned CAN interface / driver, rather than stealing another task.
    if subprocess.run(['pgrep', '-x', 'tracer_base_nod'], capture_output=True).returncode == 0:
        raise RuntimeError('Another chassis driver is running')
    links = json.loads(subprocess.check_output(['/usr/sbin/ip', '-json', 'link', 'show', 'can2']))
    if 'UP' in links[0]['flags']:
        raise RuntimeError('can2 is already up; stop its existing owner first')
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    node = driver = None
    old_parameters = {}
    owned = False
    goal_id = UUID(uuid=list(uuid.uuid4().bytes))
    owned_goal_ids = [goal_id]
    sent = False
    result = {'success': False}
    try:
        can_command('type', 'can', 'bitrate', '500000')
        owned = True
        can_command('up')
        node = Motion(target, args.speed, args.segmented)
        node.visual_feedback = Path(args.visual_feedback) if args.visual_feedback else None
        wait_for_inputs(node)
        if node.active_goals:
            raise RuntimeError('Another navigation goal is active')
        node.record({'stage':'before_driver_start', 'raw_can_state':node.can_state,
                     'age':time.monotonic()-node.can_time})
        problem = can_error(node.can_state, time.monotonic()-node.can_time, require_can=False)
        if problem:
            raise RuntimeError(problem)
        now_pose = node.data['pose'].pose.pose.position
        if np.linalg.norm(np.array([now_pose.x, now_pose.y])-target['observed_robot_xy']) > .15:
            raise RuntimeError('Robot moved after target observation')
        # Require the compiled driver watchdog; param declaration is also checked below.
        executable = str(Path(get_package_prefix('tracer_base'))/'lib/tracer_base/tracer_base_node')
        driver = subprocess.Popen([executable, '--ros-args',
            '-r', '/cmd_vel:=/board_approach/safe_cmd_vel', '-p', 'port_name:=can2',
            '-p', 'publish_tf:=false', '-p', 'odom_topic_name:=/board_approach/wheel_odom',
            '-p', 'odom_frame:=wheel_odom', '-p', 'command_timeout:=0.4'],
            stdout=open(str(Path(args.result).with_suffix('.driver.log')), 'w'),
            stderr=subprocess.STDOUT, preexec_fn=parent_death_signal)
        wait_for_can_mode(node, driver)
        wait_for_driver_subscription(node, driver)
        if len(node.get_publishers_info_by_topic('/board_approach/safe_cmd_vel')) != 1:
            raise RuntimeError('Unexpected motor command publisher')
        # Driver's actual node name is obtained from the guarded subscriber endpoint.
        endpoint = node.get_subscriptions_info_by_topic('/board_approach/safe_cmd_vel')[0]
        check = node.create_client(GetParameters,
            endpoint.node_namespace.rstrip('/')+'/'+endpoint.node_name+'/get_parameters')
        if not check.wait_for_service(timeout_sec=3):
            raise RuntimeError('Cannot verify chassis command watchdog')
        response = node.await_result(check.call_async(GetParameters.Request(names=['command_timeout'])), 3)
        if len(response.values) != 1 or abs(response.values[0].double_value-.4) > .001:
            raise RuntimeError('Chassis command watchdog unavailable; rebuild tracer_base')
        while True:
            if not args.segmented:
                configure_approach_controller(node,
                    approach_controller_parameters(args.speed, args.fine or target.get('approach_phase')=='continuous_approach'), old_parameters)
            node.wait_healthy()
            now_pose = node.data['pose'].pose.pose.position
            if np.linalg.norm(np.array([now_pose.x, now_pose.y])-target['observed_robot_xy']) > .15:
                raise RuntimeError('Robot moved during chassis initialization')
            wait_for_start_feedback(node)
            if node.active_goals - {bytes(owned.uuid) for owned in owned_goal_ids}:
                raise RuntimeError('Another navigation goal became active during chassis initialization')
            now_pose = node.data['pose'].pose.pose.position
            if np.linalg.norm(np.array([now_pose.x, now_pose.y])-target['observed_robot_xy']) > .15:
                raise RuntimeError('Robot moved while waiting for fresh start feedback')
            problem = can_error(node.can_state, time.monotonic()-node.can_time)
            if problem:
                raise RuntimeError(problem)
            node.start_xy = np.array([now_pose.x, now_pose.y])
            node.last_xy = node.start_xy.copy()
            if args.segmented:
                run_segmented(node, driver, target, args.speed)
                p = node.data['pose'].pose.pose.position
                result.update(success=True, position=[p.x, p.y], travel=node.travel)
            else:
                client = ActionClient(node, NavigateToPose, '/navigate_to_pose')
                if not client.wait_for_server(timeout_sec=3):
                    raise RuntimeError('Nav2 unavailable')
                goal = NavigateToPose.Goal()
                goal.pose.header.frame_id = 'map'
                goal.pose.header.stamp = node.get_clock().now().to_msg()
                goal.pose.pose.position.x, goal.pose.pose.position.y = target['goal_map'][:2]
                goal.pose.pose.orientation.z = float(np.sin(target['goal_yaw']/2))
                goal.pose.pose.orientation.w = float(np.cos(target['goal_yaw']/2))
                from ament_index_python.packages import get_package_share_directory
                goal.behavior_tree = str(Path(get_package_share_directory('board_approach'))/'config/no_recovery.xml')
                node.current_goal = dict(target)
                node.started, node.enabled = time.monotonic(), True
                node.record({'stage':'navigation_enabled'})
                sent = True
                handle = node.await_result(client.send_goal_async(goal, goal_uuid=goal_id), 5)
                if not handle.accepted:
                    raise RuntimeError('Navigation goal rejected')
                pending = handle.get_result_async()
                last_report = 0.
                last_update = 0.
                geometry_complete = False
                navigation_retries = 0
                endpoint_corrections = 0
                endpoint_deadline = None
                endpoint_released = False
                approach_gate = ApproachObservationGate()
                current_goal = dict(target)
                while time.monotonic()-node.started < MOTION_TIMEOUT_S:
                    node.pump(.05)
                    if node.failure or driver.poll() is not None:
                        raise RuntimeError(node.failure or 'Chassis driver exited')
                    terminal = target.get('approach_phase')=='continuous_approach' and node.visual_feedback
                    if terminal and not pending.done() and not approach_gate.complete:
                        feedback=motion_feedback(node)
                        pose=node.data['pose'].pose.pose
                        p,q=pose.position,pose.orientation
                        transform=pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
                        if fine_alignment_ready(feedback,target,transform,time.monotonic()):
                            approach_gate.complete=True
                        action=approach_gate.update(target,[p.x,p.y],feedback,time.monotonic())
                        if action=='hold':
                            if node.enabled:node.record({'stage':'approach_observation_hold','reserve':approach_gate.reserve})
                            node.enabled=False
                            node.command,node.command_time=Twist(),time.monotonic()
                            node.output.publish(Twist())
                            node.health()
                            problem=can_error(node.can_state,time.monotonic()-node.can_time)
                            if problem:raise RuntimeError(problem)
                            continue
                        if action=='refresh':
                            updated=approach_refresh_goal(feedback,target,transform,time.monotonic())
                            goal.pose.header.stamp=node.get_clock().now().to_msg()
                            goal.pose.pose.position.x,goal.pose.pose.position.y=updated['goal_map'][:2]
                            goal.pose.pose.orientation.z=float(np.sin(updated['goal_yaw']/2))
                            goal.pose.pose.orientation.w=float(np.cos(updated['goal_yaw']/2))
                            goal_id=UUID(uuid=list(uuid.uuid4().bytes));owned_goal_ids.append(goal_id)
                            handle=node.await_result(client.send_goal_async(goal,goal_uuid=goal_id),5)
                            if not handle.accepted:raise RuntimeError('Approach refreshed goal rejected')
                            pending=handle.get_result_async()
                            current_goal=updated;node.current_goal=dict(updated)
                            node.command,node.command_time=Twist(),time.monotonic()
                            node.enabled=True
                            last_update=time.monotonic()
                            node.record({'stage':'approach_observation_refresh','goal_map':updated['goal_map'],
                                         'camera_timestamp_ns':feedback['timestamp_ns'],'reserve':approach_gate.reserve})
                    nav_arrived = pending.done() and pending.result().status == 4
                    if terminal and not pending.done():
                        feedback = motion_feedback(node)
                        pose = node.data['pose'].pose.pose
                        p,q = pose.position,pose.orientation
                        transform=pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
                        if fine_alignment_ready(feedback,target,transform,time.monotonic()):
                            node.enabled=False
                            node.output.publish(Twist())
                            cancellation=node.await_result(handle.cancel_goal_async(),5)
                            if not cancellation.goals_canceling:
                                raise RuntimeError('Cannot cancel navigation for endpoint verification')
                            node.await_result(pending,5)
                            nav_arrived=True
                    if nav_arrived and terminal:
                        if endpoint_deadline is None:
                            endpoint_deadline=min(node.started+MOTION_TIMEOUT_S,time.monotonic()+ENDPOINT_BUDGET_S)
                        feedback,transform=endpoint_observation(node,driver,endpoint_deadline)
                        if feedback is None:
                            endpoint_released=True  # Parent may acquire one stationary fallback frame.
                            break
                        if fine_alignment_ready(feedback,target,transform,time.monotonic()):
                            geometry_complete=True
                            node.record({'stage':'endpoint_geometry_verified','camera_timestamp_ns':feedback['timestamp_ns']})
                            break
                        if endpoint_corrections>=MAX_ENDPOINT_CORRECTIONS or time.monotonic()>=endpoint_deadline:
                            raise RuntimeError('Endpoint correction budget exhausted')
                        updated=endpoint_correction(feedback,target,transform,time.monotonic())
                        goal.pose.header.stamp=node.get_clock().now().to_msg()
                        goal.pose.pose.position.x,goal.pose.pose.position.y=updated['goal_map'][:2]
                        goal.pose.pose.orientation.z=float(np.sin(updated['goal_yaw']/2))
                        goal.pose.pose.orientation.w=float(np.cos(updated['goal_yaw']/2))
                        goal_id=UUID(uuid=list(uuid.uuid4().bytes));owned_goal_ids.append(goal_id)
                        handle=node.await_result(client.send_goal_async(goal,goal_uuid=goal_id),5)
                        if not handle.accepted:raise RuntimeError('Endpoint correction goal rejected')
                        pending=handle.get_result_async()
                        current_goal=updated;node.current_goal=dict(updated)
                        node.command,node.command_time=Twist(),time.monotonic()
                        (node.visual_feedback.parent/'endpoint_wait.json').unlink(missing_ok=True)
                        node.enabled=True
                        last_update=time.monotonic();endpoint_corrections+=1
                        node.record({'stage':'endpoint_correction','attempt':endpoint_corrections,
                                     'goal_map':updated['goal_map'],'goal_yaw':updated['goal_yaw']})
                        continue
                    if endpoint_deadline is not None and time.monotonic()>=endpoint_deadline:
                        raise RuntimeError('Endpoint correction time budget exceeded')
                    if pending.done():
                        status=pending.result().status
                        if status==4:break
                        navigation_retries += 1
                        hold_for_navigation_retry(node, driver, navigation_retries, status)
                        if time.monotonic()-node.started >= MOTION_TIMEOUT_S:
                            raise RuntimeError('Navigation time budget exceeded during retry')
                        goal.pose.header.stamp = node.get_clock().now().to_msg()
                        goal_id = UUID(uuid=list(uuid.uuid4().bytes))
                        owned_goal_ids.append(goal_id)
                        handle = node.await_result(client.send_goal_async(goal, goal_uuid=goal_id), 5)
                        if not handle.accepted:
                            raise RuntimeError('Retried navigation goal rejected')
                        pending = handle.get_result_async()
                        node.command, node.command_time = Twist(), time.monotonic()
                        node.enabled = True
                        last_update = time.monotonic()
                        node.record({'stage':'navigation_retry_started', 'attempt':navigation_retries})
                        continue
                    if endpoint_deadline is None and node.visual_feedback and time.monotonic()-last_update > 3:
                        feedback = motion_feedback(node)
                        position = node.data['pose'].pose.pose.position
                        near_endpoint = feedback_required_near_goal(current_goal, [position.x, position.y])
                        updated = (corrected_navigation_goal(feedback, target, current_goal, [position.x, position.y])
                                   if feedback is not None and (target.get('approach_phase') in ('continuous_approach','precise_docking')
                                                                or not near_endpoint) else None)
                        if updated is not None:
                            goal.pose.header.stamp = node.get_clock().now().to_msg()
                            goal.pose.pose.position.x, goal.pose.pose.position.y = updated['goal_map'][:2]
                            goal.pose.pose.orientation.z = float(np.sin(updated['goal_yaw']/2))
                            goal.pose.pose.orientation.w = float(np.cos(updated['goal_yaw']/2))
                            # Nav2 preempts its own goal without restarting the motor process.
                            goal_id = UUID(uuid=list(uuid.uuid4().bytes))
                            owned_goal_ids.append(goal_id)
                            node.current_goal = dict(updated)
                            handle = node.await_result(client.send_goal_async(goal, goal_uuid=goal_id), 5)
                            if not handle.accepted:
                                raise RuntimeError('Updated navigation goal rejected')
                            pending = handle.get_result_async()
                            current_goal = updated
                            node.record({'stage':'async_goal_update', 'goal_map':updated['goal_map'],
                                         'goal_yaw':updated['goal_yaw'], 'camera_timestamp_ns':feedback['timestamp_ns']})
                        last_update = time.monotonic()
                    if time.monotonic()-last_report > 3:
                        print(json.dumps({'stage': 'moving', 'travel': node.travel,
                            'inlier': node.data['status'].inlier_fraction}), flush=True)
                        last_report = time.monotonic()
                node.enabled = False
                node.output.publish(Twist())
                if node.failure:
                    raise RuntimeError(node.failure)
                if not (geometry_complete or endpoint_released) and (not pending.done() or pending.result().status != 4):
                    raise RuntimeError('Navigation timeout or failure')
                motion_feedback(node, force=True)
                p = node.data['pose'].pose.pose.position
                result.update(success=True, position=[p.x, p.y], travel=node.travel)
            if not args.next_request:
                break
            # The action has completed. Gate all subsequent commands before publishing
            # the leg report; the parent can immediately begin stationary observation.
            node.enabled = False
            node.command = Twist()
            node.output.publish(Twist())
            write_atomic(Path(args.result).with_suffix('.stopping.json'),
                         {'monotonic':time.monotonic(), 'success':True})
            result.update(events=list(node.events), chassis_held=True)
            write_atomic(args.result, result)
            request = hold_for_next_leg(node, driver, args.next_request)
            if request['action'] == 'finish':
                break
            # Only one coarse-to-precise transition is allowed in a session.
            args.next_request = None
            args.goal, args.result = request['goal'], request['result']
            args.fine = True
            target = json.loads(Path(args.goal).read_text())
            if target.get('approach_phase') != 'precise_docking' or not np.isfinite(target['goal_map']+[target['goal_yaw']]).all():
                raise RuntimeError('Invalid precise transition goal')
            node.target = target
            node.visual_feedback = Path(request['visual_feedback'])
            node.travel, node.last_xy = 0., None
            node.paused, node.fresh_gate = False, FreshDataGate()
            node.command, node.command_time = Twist(), 0.
            node.events = []
            result = {'success':False}
            goal_id = UUID(uuid=list(uuid.uuid4().bytes))
    except (Exception, KeyboardInterrupt) as error:
        result.update(success=False, error=str(error) or 'Interrupted')
    finally:
        if node:
            result['can_state_before_cleanup'] = node.can_state
            result['can_age_before_cleanup'] = time.monotonic()-node.can_time
            result['events'] = node.events
            node.enabled = False
            node.output.publish(Twist())
            # Signal the observer to stop launching inference during the existing zero-output cleanup.
            stopping_path=Path(args.result).with_suffix('.stopping.json')
            stopping_tmp=stopping_path.with_suffix('.tmp')
            try:
                stopping_tmp.write_text(json.dumps({'monotonic':time.monotonic(),'success':result.get('success',False)}))
                stopping_tmp.replace(stopping_path)
            except OSError as error:
                result['timing_marker_error']=str(error)  # Never interrupt motor cleanup for logging.
            if sent:
                try:
                    cancel = node.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
                    for owned_id in owned_goal_ids:
                        request = CancelGoal.Request()
                        request.goal_info.goal_id = owned_id
                        cancel.call_async(request)
                except Exception as error:
                    result.update(success=False, cancel_error=str(error))
            try:
                node.pump(2)
            except Exception:
                pass
        if driver and driver.poll() is None:
            try:
                driver.send_signal(signal.SIGINT)
                try:
                    driver.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    driver.kill()
                    driver.wait(timeout=3)
            except Exception as error:
                result.update(success=False, driver_cleanup_error=str(error))
        if node:
            node.guard_timer.cancel()
        if owned:
            try:
                can_command('down')
            except Exception as error:
                result.update(success=False, cleanup_error=str(error))
        if node:
            if old_parameters:
                try:
                    node.parameters(old_parameters)
                except Exception as error:
                    result.update(success=False, parameter_restore_error=str(error))
            node.can.close()
            node.destroy_node()
        rclpy.shutdown()
        if node and getattr(node,'obstacle_evidence',None) is not None:
            try:
                from .obstacle_evidence import save_trigger
                save_trigger(Path(args.result).parent,node.obstacle_evidence)
                result['obstacle_evidence']='obstacle_trigger.npz / .pcd / .png / .json'
            except Exception as error:
                result['obstacle_evidence_error']=str(error)
        if node and getattr(node,'last_door_filter',None) is not None:
            try:
                np.savez_compressed(Path(args.result).parent/'door_filter.npz',**node.last_door_filter)
                result['door_filter_evidence']='door_filter.npz'
            except Exception as error:
                result['door_filter_evidence_error']=str(error)
        if node and getattr(node,'navigation_failures',None):
            try:
                for index,(metadata,arrays) in enumerate(node.navigation_failures):
                    name=Path(args.result).parent/f'navigation_failure_{index}'
                    name.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
                    np.savez_compressed(name.with_suffix('.npz'),**arrays)
                result['navigation_failure_evidence_count']=len(node.navigation_failures)
            except Exception as error:
                result['navigation_failure_evidence_error']=str(error)
        if node and getattr(node,'control_trace',None):
            try:
                trace_path=Path(args.result).parent.parent/'control_trace.jsonl'
                trace_path.write_text(''.join(json.dumps(row)+'\n' for row in node.control_trace))
                result['control_trace']=str(trace_path)
            except Exception as error:
                result['control_trace_error']=str(error)
        write_atomic(args.result, result)
        print(json.dumps(result), flush=True)
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
