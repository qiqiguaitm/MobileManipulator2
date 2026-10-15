"""Localization/data gate shared by capture startup and runtime monitoring."""
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from hdl_localization.msg import ScanMatchingStatus
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException


class Readiness(Node):
    def __init__(self, args):
        super().__init__('navigation_scene_readiness')
        self.args = args
        self.data = {}
        self.received = {}
        self.buffer = Buffer(cache_time=Duration(seconds=10))
        self.scan_tf_ok_at = 0.
        self.pending_scans = deque(maxlen=20)
        self.map_ok = False
        self.map_reason = 'waiting for map'
        self.tf_history = deque()
        self.static_transforms = {}
        self.expected = None
        if args.expected_map:
            spec = yaml.safe_load(args.expected_map.read_text())
            if spec.get('mode', 'trinary') != 'trinary':
                raise ValueError('Only trinary map verification is supported')
            image = Path(spec['image'])
            if not image.is_absolute():
                image = args.expected_map.parent / image
            pixels = np.asarray(Image.open(image).convert('L'), dtype=float)
            prob = pixels / 255 if spec.get('negate', 0) else (255-pixels) / 255
            grid = np.full(pixels.shape, -1, dtype=np.int8)
            grid[prob > spec['occupied_thresh']] = 100
            grid[prob < spec['free_thresh']] = 0
            self.expected = spec, pixels.shape, np.flipud(grid).ravel()
        live = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.kept = []
        for key, topic, kind, qos in [
            ('status', args.status_topic, ScanMatchingStatus, live),
            ('scan', '/scan', LaserScan, live),
            ('odom', args.odom_topic, Odometry, live),
            ('map', '/map', OccupancyGrid, latched),
            ('tf', '/tf', TFMessage, live),
            ('tf_static', '/tf_static', TFMessage, latched),
        ]:
            self.kept.append(self.create_subscription(kind, topic,
                lambda msg, key=key: self.receive(key, msg), qos))

    def receive(self, key, msg):
        self.data[key] = msg
        self.received[key] = time.monotonic()
        if key == 'scan':
            self.pending_scans.append((self.received[key], msg))
        if key in ('tf', 'tf_static'):
            if key == 'tf':
                self.tf_history.append((time.monotonic(), msg))
                while self.tf_history and time.monotonic()-self.tf_history[0][0] > 2:
                    self.tf_history.popleft()
            for transform in msg.transforms:
                if key == 'tf_static':
                    self.static_transforms[transform.child_frame_id] = transform
                    self.buffer.set_transform_static(transform, 'live')
                else:
                    self.buffer.set_transform(transform, 'live')
        if key == 'map':
            self.map_ok = bool(msg.data) and msg.header.frame_id == self.args.map_frame
            if self.expected:
                spec, shape, grid = self.expected
                p, q = msg.info.origin.position, msg.info.origin.orientation
                yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                self.map_ok = bool(self.map_ok and (msg.info.height, msg.info.width) == shape
                    and math.isclose(msg.info.resolution, spec['resolution'], abs_tol=1e-7)
                    and np.allclose([p.x,p.y,yaw], spec['origin'], atol=1e-6)
                    and np.array_equal(np.asarray(msg.data), grid))
            self.map_reason = 'map differs from expected map/frame or is empty'

    def problems(self):
        result = []
        now = time.monotonic()
        for key in ('status', 'scan', 'odom', 'tf', 'tf_static', 'map'):
            if key not in self.data:
                result.append('missing '+key)
            elif key in ('status', 'scan', 'odom', 'tf') and now-self.received[key] > 0.8:
                result.append('stale '+key)
        if not self.map_ok:
            result.append(self.map_reason)
        status = self.data.get('status')
        if status is not None and (not status.has_converged
                or not math.isfinite(status.inlier_fraction)
                or status.inlier_fraction < self.args.min_inlier):
            result.append(f'localization not ready: converged={status.has_converged}, inlier={status.inlier_fraction:.3f}')
        now_ns = self.get_clock().now().nanoseconds
        for key in ('scan', 'odom'):
            msg = self.data.get(key)
            if msg is not None:
                age = (now_ns-Time.from_msg(msg.header.stamp).nanoseconds)/1e9
                if not -0.1 <= age <= 0.8:
                    result.append(f'{key} timestamp age={age:.3f}s')
        try:
            pose = self.buffer.lookup_transform(self.args.map_frame, self.args.base_frame, Time())
            age = (now_ns-Time.from_msg(pose.header.stamp).nanoseconds)/1e9
            if not -0.1 <= age <= 0.8:
                result.append('stale map->base TF')
        except TransformException:
            result.append('missing map->base TF')
        # TF may arrive after its scan. Retry recent scans instead of repeatedly
        # replacing the pending lookup with a newer, not-yet-transformable scan.
        while self.pending_scans and now-self.pending_scans[0][0] > 0.8:
            self.pending_scans.popleft()
        for received, scan in reversed(self.pending_scans):
            try:
                self.buffer.lookup_transform(self.args.map_frame, scan.header.frame_id,
                                             Time.from_msg(scan.header.stamp))
                self.scan_tf_ok_at = received
                break
            except TransformException:
                pass
        if now-self.scan_tf_ok_at > 0.8:
            result.append('scan timestamp TF unavailable')
        return result
