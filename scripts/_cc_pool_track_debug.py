#!/usr/bin/env python3
"""
Debug: 记录感知层 track 输出 + 目标池变化，用于分析漂移 track 是否污染目标池。

订阅:
  /multi_camera_perception/fused/objects_3d  — 感知 track 结果 (每帧)
  /cleaner_manager/status                    — 目标池快照 (1Hz)
  TF map → base_link                         — 机器人位姿

输出: JSONL 文件，两种事件:
  percept  — 每帧感知输出 + 机器人运动状态
  pool     — 目标池快照 + 新增/消失标记

用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_pool_track_debug.py [--duration 120]
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from perception.msg import Object3DArray
from cleaner_manager.msg import CleanerManagerStatus


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class PoolTrackDebugger(Node):
    MOVE_THRESH = 0.02  # m/s, 高于此认为在运动
    ROT_THRESH = 0.03   # rad/s
    POOL_MATCH_DIST = 0.15  # m, pool diff 匹配阈值

    def __init__(self, out_path: str, duration: float):
        super().__init__('pool_track_debugger')
        self._out = open(out_path, 'w')
        self._deadline = time.time() + duration if duration > 0 else float('inf')
        self._start_time = time.time()

        # TF
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # Robot state
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._robot_vx = 0.0
        self._robot_vy = 0.0
        self._robot_wz = 0.0
        self._last_tf_time = 0.0
        self._robot_moving = False

        # Pool state tracking (for diff)
        self._prev_pool = []  # list of (x, y, cat, obs)
        self._pool_event_count = 0
        self._new_target_events = []  # 最近的新增事件

        # Recent perception frames (for correlation)
        self._recent_percepts = []  # deque-like, keep last 30 frames
        self._percept_count = 0

        # Stats
        self._stats = {
            'percept_frames': 0,
            'pool_snapshots': 0,
            'new_targets_total': 0,
            'new_targets_while_moving': 0,
            'confirmed_tracks_total': 0,
            'unconfirmed_tracks_total': 0,
        }

        # Subscriptions
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(
            Object3DArray,
            '/multi_camera_perception/fused/objects_3d',
            self._percept_cb, qos)

        self.create_subscription(
            CleanerManagerStatus,
            '/cleaner_manager_node/status',
            self._pool_cb, 10)

        # TF poll timer (10Hz)
        self.create_timer(0.1, self._tf_poll)
        # Live summary timer (5s)
        self.create_timer(5.0, self._print_summary)

        self.get_logger().info(f'Recording to {out_path} ...')

    def _tf_poll(self):
        """Poll TF to update robot position and compute velocity."""
        if time.time() > self._deadline:
            self.get_logger().info('Duration reached, shutting down.')
            self._close()
            raise SystemExit(0)

        try:
            tf = self._tf_buf.lookup_transform('map', 'base_link',
                                                rclpy.time.Time())
        except Exception:
            return

        now = time.time()
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        dt = now - self._last_tf_time if self._last_tf_time > 0 else 0.0
        if dt > 0.01:
            self._robot_vx = (x - self._robot_x) / dt
            self._robot_vy = (y - self._robot_y) / dt
            # Unwrap yaw diff
            dyaw = yaw - self._robot_yaw
            if dyaw > math.pi:
                dyaw -= 2 * math.pi
            elif dyaw < -math.pi:
                dyaw += 2 * math.pi
            self._robot_wz = dyaw / dt

        self._robot_x = x
        self._robot_y = y
        self._robot_yaw = yaw
        self._last_tf_time = now

        v_lin = math.sqrt(self._robot_vx**2 + self._robot_vy**2)
        self._robot_moving = (v_lin > self.MOVE_THRESH or
                              abs(self._robot_wz) > self.ROT_THRESH)

    def _robot_state(self):
        return {
            'x': round(self._robot_x, 4),
            'y': round(self._robot_y, 4),
            'yaw': round(self._robot_yaw, 3),
            'vx': round(self._robot_vx, 4),
            'vy': round(self._robot_vy, 4),
            'wz': round(self._robot_wz, 4),
            'moving': self._robot_moving,
        }

    # ------------------------------------------------------------------
    # Perception callback
    # ------------------------------------------------------------------

    def _percept_cb(self, msg: Object3DArray):
        t = time.time()
        objects = []
        n_confirmed = 0
        n_unconfirmed = 0

        for obj in msg.objects:
            is_confirmed = obj.object_id.startswith('track_')
            if is_confirmed:
                n_confirmed += 1
            else:
                n_unconfirmed += 1

            objects.append({
                'id': obj.object_id,
                'cat': obj.category,
                'x': round(obj.position.x, 4),
                'y': round(obj.position.y, 4),
                'z': round(obj.position.z, 4),
                'conf': round(obj.confidence, 3),
                'score': round(obj.score, 3),
                'dist': round(obj.distance, 3),
                'phys': round(obj.physical_size, 4),
                'confirmed': is_confirmed,
            })

        record = {
            'type': 'percept',
            't': round(t - self._start_time, 3),
            'wall': round(t, 3),
            'robot': self._robot_state(),
            'n_obj': len(objects),
            'n_confirmed': n_confirmed,
            'n_unconfirmed': n_unconfirmed,
            'objects': objects,
        }
        self._out.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Keep recent frames for pool correlation
        self._recent_percepts.append(record)
        if len(self._recent_percepts) > 30:
            self._recent_percepts.pop(0)

        self._stats['percept_frames'] += 1
        self._stats['confirmed_tracks_total'] += n_confirmed
        self._stats['unconfirmed_tracks_total'] += n_unconfirmed

    # ------------------------------------------------------------------
    # Pool status callback
    # ------------------------------------------------------------------

    def _pool_cb(self, msg: CleanerManagerStatus):
        t = time.time()

        # Build current pool snapshot
        current_pool = []
        for e in msg.pool_targets:
            current_pool.append({
                'cat': e.category,
                'x': round(e.pos_x, 4),
                'y': round(e.pos_y, 4),
                'z': round(e.pos_z, 4),
                'obs': e.observations,
                'status': e.status,
                'viable': e.viable,
                'dist': round(e.distance, 3),
                'phys': round(e.physical_size, 4),
                'score': round(e.score, 3),
                'age': round(e.last_seen_age, 2),
                'attempts': e.attempts,
            })

        # Diff: find new targets (in current but not in prev)
        new_targets = self._diff_pool(self._prev_pool, current_pool)
        removed_targets = self._diff_pool(current_pool, self._prev_pool)

        # For each new target, find likely source track from recent perception
        for nt in new_targets:
            source = self._find_source_track(nt)
            nt['likely_source'] = source

        record = {
            'type': 'pool',
            't': round(t - self._start_time, 3),
            'wall': round(t, 3),
            'state': msg.state_name,
            'robot': self._robot_state(),
            'pool_size': len(current_pool),
            'remaining': msg.targets_remaining,
            'targets': current_pool,
            'new': new_targets,
            'removed': removed_targets,
        }
        self._out.write(json.dumps(record, ensure_ascii=False) + '\n')

        if new_targets:
            for nt in new_targets:
                moving_tag = ' [MOVING]' if self._robot_moving else ' [STATIC]'
                src = nt.get('likely_source', '?')
                self.get_logger().warn(
                    f'NEW POOL TARGET: {nt["cat"]} '
                    f'({nt["x"]:.2f},{nt["y"]:.2f}) '
                    f'src={src}{moving_tag}')

            self._stats['new_targets_total'] += len(new_targets)
            if self._robot_moving:
                self._stats['new_targets_while_moving'] += len(new_targets)

        self._prev_pool = current_pool
        self._stats['pool_snapshots'] += 1

    def _diff_pool(self, base, target):
        """Find entries in target that are not in base (by 2D position)."""
        new = []
        for t in target:
            if t['status'] != 0:  # only care about ACTIVE
                continue
            found = False
            for b in base:
                if b['status'] != 0:
                    continue
                dx = t['x'] - b['x']
                dy = t['y'] - b['y']
                if math.sqrt(dx*dx + dy*dy) < self.POOL_MATCH_DIST:
                    found = True
                    break
            if not found:
                new.append(dict(t))  # copy
        return new

    def _find_source_track(self, pool_target):
        """Search recent perception frames for the track closest to this pool target."""
        px, py = pool_target['x'], pool_target['y']
        best_id = None
        best_dist = float('inf')
        best_frame_t = None

        for frame in reversed(self._recent_percepts):
            for obj in frame['objects']:
                dx = obj['x'] - px
                dy = obj['y'] - py
                d = math.sqrt(dx*dx + dy*dy)
                if d < best_dist:
                    best_dist = d
                    best_id = obj['id']
                    best_frame_t = frame['t']

        if best_id and best_dist < 0.20:
            return {
                'track_id': best_id,
                'dist': round(best_dist, 4),
                'frame_t': best_frame_t,
                'confirmed': best_id.startswith('track_'),
            }
        return None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        s = self._stats
        elapsed = time.time() - self._start_time
        self.get_logger().info(
            f'[{elapsed:.0f}s] '
            f'percept={s["percept_frames"]}frames '
            f'pool={s["pool_snapshots"]}snaps '
            f'new_targets={s["new_targets_total"]}'
            f'(moving={s["new_targets_while_moving"]}) '
            f'confirmed={s["confirmed_tracks_total"]} '
            f'unconfirmed={s["unconfirmed_tracks_total"]}')

    def _close(self):
        self._out.flush()
        self._out.close()
        self.get_logger().info(f'Final stats: {json.dumps(self._stats)}')


def main():
    parser = argparse.ArgumentParser(description='Pool-Track debug recorder')
    parser.add_argument('--duration', type=float, default=120,
                        help='Recording duration in seconds (0=infinite)')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(os.path.dirname(__file__),
                            f'pool_track_debug_{ts}.jsonl')

    rclpy.init()
    node = PoolTrackDebugger(out_path, args.duration)

    def shutdown(*_):
        node._close()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        try:
            node._close()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
