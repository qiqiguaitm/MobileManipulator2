from .limits import MAX_STAGE_PATH_M, MAX_TOTAL_PATH_M
import json
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatusArray
from hdl_localization.msg import ScanMatchingStatus
from nav2_msgs.action import ComputePathToPose
from tf2_ros import Buffer, TransformListener, TransformException
from .perception import pose_matrix, DetectionError
from .safety import freshness_error, InputFreshnessError
from .lidar_snapshot import capture_lidar


def stamp(message):
    return message.header.stamp.sec*10**9 + message.header.stamp.nanosec


class RobotIO(Node):
    def __init__(self, with_camera=True, geometry_source='depth'):
        super().__init__('board_approach_io')
        self.data, self.received = {}, {}
        self.require_lidar = with_camera and geometry_source=='lidar'
        self.clouds = deque(maxlen=15)
        self.odometry = deque(maxlen=250)
        self.last_lidar_frame = None
        self.images = {'rgb': {}, 'depth': {}}
        self.poses = deque(maxlen=100)
        self.active_goals = set()
        self.last_snapshot_stamp = 0
        self.subs = []
        for key, topic, kind in [
            ('rgb', '/camera/top/color/image_raw', Image),
            ('depth', '/camera/top/aligned_depth_to_color/image_raw', Image),
            ('info', '/camera/top/color/camera_info', CameraInfo),
            ('pose', '/pose', Odometry), ('odom', '/odom', Odometry),
            ('scan', '/scan', LaserScan), ('status', '/status', ScanMatchingStatus)]:
            if not with_camera and key in ('rgb', 'depth', 'info'):
                continue
            self.subs.append(self.create_subscription(kind, topic,
                lambda m, key=key: self.receive(key, m),
                QoSProfile(depth=1, reliability=(ReliabilityPolicy.RELIABLE
                    if key in ('rgb', 'depth', 'info') else ReliabilityPolicy.BEST_EFFORT))))
        if self.require_lidar:
            self.subs.append(self.create_subscription(PointCloud2,'/rslidar_points',
                lambda m:self.receive('cloud',m),QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)))
            self.subs.append(self.create_subscription(Odometry,'/fastlio/odom_measurement',
                lambda m:self.receive('deskew_odom',m),QoSProfile(depth=20,reliability=ReliabilityPolicy.RELIABLE)))
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subs.append(self.create_subscription(GoalStatusArray,
            '/navigate_to_pose/_action/status', self.goal_status, q))
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)

    def receive(self, key, message):
        self.data[key] = message
        self.received[key] = time.monotonic()
        if key in self.images:
            frames = self.images[key]
            frames[stamp(message)] = message
            while len(frames) > 12:
                del frames[min(frames)]
        if key == 'pose':
            self.poses.append(message)
        if key == 'cloud':
            self.clouds.append(message)
        if key == 'deskew_odom':
            self.odometry.append(message)

    def goal_status(self, message):
        self.active_goals = {bytes(s.goal_info.goal_id.uuid) for s in message.status_list
                             if s.status in (1, 2, 3)}

    def pump(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.03)

    def await_result(self, future, timeout):
        end = time.monotonic()+timeout
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.03)
        if not future.done():
            raise RuntimeError('ROS request timed out')
        return future.result()

    def health(self, camera=False):
        keys = ['pose', 'odom', 'scan', 'status']
        if camera:
            keys += ['rgb','info'] + (['cloud','deskew_odom'] if self.require_lidar else ['depth'])
        error = freshness_error(time.monotonic(), self.received, keys)
        if error:
            raise InputFreshnessError(error)
        now = self.get_clock().now().nanoseconds
        for key in ('pose', 'odom', 'scan'):
            age = (now-stamp(self.data[key]))/1e9
            if not -.1 <= age <= .8:
                raise InputFreshnessError('%s message timestamp is stale: age=%.3fs, allowed=-0.100..0.800s' % (key, age))
        status = self.data['status']
        if not status.has_converged or not np.isfinite(status.inlier_fraction) or status.inlier_fraction < .90:
            raise RuntimeError('Localization unreliable; set initial pose in RViz first')

    def wait_healthy(self, timeout=5.):
        deadline = time.monotonic()+timeout
        last_error = None
        while time.monotonic() < deadline:
            self.pump(.1)
            try:
                self.health()
                return
            except InputFreshnessError as error:
                last_error = error
        raise InputFreshnessError('Fresh sensor startup timeout: '+str(last_error))

    def transform(self, parent, child, timestamp_ns=None):
        query_time = (rclpy.time.Time() if timestamp_ns is None else
                      rclpy.time.Time(nanoseconds=int(timestamp_ns),
                                      clock_type=self.get_clock().clock_type))
        t = self.tf.lookup_transform(parent, child, query_time).transform
        return pose_matrix([t.translation.x, t.translation.y, t.translation.z],
                           [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])

    def snapshot(self, moving=False):
        self.pump(.03 if moving else .5)
        self.health(camera=True)
        motion = self.data['odom'].twist.twist
        if not moving and (abs(motion.linear.x) > .03 or abs(motion.angular.z) > .04):
            raise RuntimeError('Robot must be stationary while observing board')
        lidar = getattr(self,'require_lidar',False)
        common = self.images['rgb'].keys() if lidar else self.images['rgb'].keys() & self.images['depth'].keys()
        if not common:
            raise RuntimeError('No synchronized RGB / aligned depth pair')
        now = self.get_clock().now().nanoseconds
        pose_tolerance_ns = 100_000_000 if moving else 250_000_000
        candidates = [t for t in sorted(common, reverse=True)
                      if t > self.last_snapshot_stamp and -.1 <= (now-t)/1e9 <= .8
                      and self.poses and min(abs(stamp(p)-t) for p in self.poses) <= pose_tolerance_ns]
        if lidar:
            # Leave enough history on either side for per-point odometry deskew.
            candidates = [t for t in candidates if self.clouds and self.odometry
                          and min(abs(stamp(m)-t) for m in self.clouds)<=60_000_000
                          and stamp(self.odometry[-1])>=t+120_000_000
                          and stamp(self.odometry[0])<=t-180_000_000]
        if not candidates:
            source = 'RGB/LiDAR' if lidar else 'RGB-D'
            raise RuntimeError('No fresh '+source+' pair with a matching localization pose')
        timestamp = candidates[0]
        if lidar:
            last_error = None
            for timestamp in candidates:
                try:
                    self.last_lidar_frame = capture_lidar(self,timestamp)
                    break
                except DetectionError as error:
                    last_error = error
            else:
                raise DetectionError('No synchronized RGB/LiDAR/odometry observation: '+str(last_error))
        rgb = self.images['rgb'][timestamp]
        depth = self.images['depth'].get(timestamp)
        if rgb.encoding not in ('rgb8', 'bgr8') or (not lidar and depth.encoding != '16UC1'):
            raise RuntimeError('Unexpected image encoding')
        if (not lidar and rgb.header.frame_id != depth.header.frame_id) or rgb.header.frame_id != self.data['info'].header.frame_id:
            raise RuntimeError('RGB, aligned depth and calibration frames disagree')
        pixels = np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.step)
        pixels = pixels[:, :rgb.width*3].reshape(rgb.height, rgb.width, 3).copy()
        if rgb.encoding == 'rgb8':
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        meters = None
        if depth is not None and depth.encoding=='16UC1':
            dtype = '>u2' if depth.is_bigendian else '<u2'
            meters = np.frombuffer(depth.data, dtype).reshape(depth.height, depth.step//2)[:, :depth.width].copy()/1000.
        pose = min(self.poses, key=lambda p: abs(stamp(p)-timestamp))
        if abs(stamp(pose)-timestamp) > pose_tolerance_ns:
            raise RuntimeError('No pose close to camera timestamp')
        p = pose.pose.pose
        matrix = pose_matrix([p.position.x, p.position.y, p.position.z],
                             [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        # Compare navigation TF with the localization pose at exactly its own
        # timestamp. The pose was matched to the image above; latest TF would
        # include robot motion after capture and falsely fail during turns.
        pose_timestamp = stamp(pose)
        try:
            matched_tf = self.transform('map', 'base_link', timestamp_ns=pose_timestamp)
        except TransformException as error:
            raise InputFreshnessError(
                f'Navigation TF unavailable at localization timestamp {pose_timestamp} '
                f'(camera={timestamp}): {error}') from error
        rotation_error = float(np.arccos(np.clip(
            (np.trace(matrix[:3, :3].T @ matched_tf[:3, :3])-1)/2, -1, 1)))
        position_error = float(np.linalg.norm(matrix[:2, 3]-matched_tf[:2, 3]))
        if position_error > .10 or rotation_error > .12:
            raise RuntimeError(
                f'Localization pose and navigation TF disagree at matched timestamp: '
                f'position={position_error:.3f}m (limit 0.100m), '
                f'rotation={rotation_error:.3f}rad (limit 0.120rad), '
                f'pose={pose_timestamp}, camera={timestamp}')
        if lidar:
            if (self.data['info'].width!=rgb.width or self.data['info'].height!=rgb.height):
                raise RuntimeError('Camera intrinsics resolution does not match RGB image')
        self.last_snapshot_stamp = timestamp
        return pixels, meters, np.array(self.data['info'].k).reshape(3, 3), matrix, timestamp

    def verify_stationary_since(self, observed_pose):
        # GPT may take seconds. Drain sensors and reject robot movement during inference.
        self.pump(.5)
        self.health()
        current = self.data['pose'].pose.pose
        pose = pose_matrix([current.position.x,current.position.y,current.position.z],
                           [current.orientation.x,current.orientation.y,current.orientation.z,current.orientation.w])
        angle = np.arccos(np.clip((np.trace(observed_pose[:3,:3].T @ pose[:3,:3])-1)/2,-1,1))
        motion = self.data['odom'].twist.twist
        if (np.linalg.norm(pose[:2,3]-observed_pose[:2,3]) > .025 or angle > .04
                or abs(motion.linear.x) > .03 or abs(motion.angular.z) > .04):
            raise RuntimeError('Robot moved while detector was processing the frame')

    def plan(self, target):
        client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError('Nav2 planner is not running')
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = 'map'
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose.position.x, goal.goal.pose.position.y = target['goal_map'][:2]
        goal.goal.pose.orientation.z = float(np.sin(target['goal_yaw']/2))
        goal.goal.pose.orientation.w = float(np.cos(target['goal_yaw']/2))
        goal.planner_id = 'GridBased'
        handle = self.await_result(client.send_goal_async(goal), 8)
        if not handle.accepted:
            raise RuntimeError('Planner rejected goal')
        result = self.await_result(handle.get_result_async(), 10)
        if result.status != 4 or not result.result.path.poses:
            raise RuntimeError('No collision-free path to board')
        points = np.array([[p.pose.position.x, p.pose.position.y] for p in result.result.path.poses])
        self.last_plan_points = points
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        limit = MAX_TOTAL_PATH_M if target.get('approach_phase') == 'continuous_approach' else MAX_STAGE_PATH_M
        if length > limit:
            raise RuntimeError(f'Approach path {length:.2f} m exceeds {limit:.1f} m limit')
        return length
