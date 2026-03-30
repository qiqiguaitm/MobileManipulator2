#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF / HDL / FastLIO / Camera 时间戳监控脚本

监控所有时间戳源，检测异常（断供、时钟偏移、延迟跳变），
生成 JSONL 日志 + 终端实时摘要。

用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_tf_timestamp_monitor.py [--duration 60] [--output ts_monitor.jsonl]
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TimestampTracker:
    """跟踪单个源的时间戳统计"""

    def __init__(self, name, window=100):
        self.name = name
        self.count = 0
        self.first_wall = None
        self.last_wall = None
        self.last_stamp = None
        self.dt_history = deque(maxlen=window)
        self.lag_history = deque(maxlen=window)
        self.gap_events = []       # (wall_time, gap_sec)
        self.future_events = []    # (wall_time, ahead_sec)

    def update(self, msg_stamp_sec, wall_now=None):
        if wall_now is None:
            wall_now = time.time()

        if self.first_wall is None:
            self.first_wall = wall_now

        lag = wall_now - msg_stamp_sec
        self.lag_history.append(lag)

        if self.last_stamp is not None:
            dt = msg_stamp_sec - self.last_stamp
            self.dt_history.append(dt)
            # 检测断供 (dt > 500ms 视为 gap)
            if dt > 0.5:
                self.gap_events.append((wall_now, dt))
            # 检测时间戳回退
            if dt < -0.01:
                self.gap_events.append((wall_now, dt))

        # 检测未来时间戳
        if lag < -0.01:
            self.future_events.append((wall_now, -lag))

        self.last_stamp = msg_stamp_sec
        self.last_wall = wall_now
        self.count += 1

    def stats(self):
        result = {'name': self.name, 'count': self.count}
        if self.first_wall and self.last_wall and self.last_wall > self.first_wall:
            result['avg_hz'] = round(self.count / (self.last_wall - self.first_wall), 1)

        if self.dt_history:
            dts = list(self.dt_history)
            result['dt_min_ms'] = round(min(dts) * 1000, 1)
            result['dt_max_ms'] = round(max(dts) * 1000, 1)
            result['dt_mean_ms'] = round(sum(dts) / len(dts) * 1000, 1)

        if self.lag_history:
            lags = list(self.lag_history)
            result['lag_min_ms'] = round(min(lags) * 1000, 1)
            result['lag_max_ms'] = round(max(lags) * 1000, 1)
            result['lag_mean_ms'] = round(sum(lags) / len(lags) * 1000, 1)

        result['gaps'] = len(self.gap_events)
        result['future_stamps'] = len(self.future_events)
        return result


class TimestampMonitor(Node):

    def __init__(self, duration, output_path):
        super().__init__('tf_timestamp_monitor')
        self.duration = duration
        self.output_path = output_path
        self.start_time = time.time()
        self.log_entries = []
        self._lock = threading.Lock()

        # 所有需要监控的源
        self.trackers = {}
        self._init_tracker('tf_map_odom', 'TF map→odom')
        self._init_tracker('tf_map_odom_hdl', 'TF map→odom (HDL, 老时间戳)')
        self._init_tracker('tf_map_odom_repub', 'TF map→odom (tf_republisher, 新时间戳)')
        self._init_tracker('tf_camerainit_body', 'TF camera_init→body')
        self._init_tracker('hdl_pose', 'HDL /pose')
        self._init_tracker('fastlio_odom', 'FastLIO /fastlio/odom')
        self._init_tracker('cam_top_color', '相机 top color')
        self._init_tracker('cam_chassis_color', '相机 chassis color')
        self._init_tracker('cam_top_depth', '相机 top depth')
        self._init_tracker('cam_chassis_depth', '相机 chassis depth')

        # 用于区分 HDL 原始 vs tf_republisher
        self._last_map_odom_stamps = deque(maxlen=5)

        # 订阅
        best_effort = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(TFMessage, '/tf', self._tf_cb, 50)
        self.create_subscription(Odometry, '/pose', self._hdl_pose_cb, 10)
        self.create_subscription(Odometry, '/fastlio/odom', self._fastlio_cb, 10)

        # 相机 — best_effort 避免 QoS 不匹配
        for topic, key in [
            ('/camera/top/color/image_raw', 'cam_top_color'),
            ('/camera/chassis/color/image_raw', 'cam_chassis_color'),
            ('/camera/top/aligned_depth_to_color/image_raw', 'cam_top_depth'),
            ('/camera/chassis/aligned_depth_to_color/image_raw', 'cam_chassis_depth'),
        ]:
            self.create_subscription(
                Image, topic,
                lambda msg, k=key: self._image_cb(msg, k),
                best_effort)

        # 定时采样写日志
        self.create_timer(1.0, self._sample_cb)
        # 终端输出
        self.create_timer(5.0, self._print_summary)

        self.get_logger().info(
            f'开始监控 {duration}s, 日志 → {output_path}')

    def _init_tracker(self, key, desc):
        self.trackers[key] = TimestampTracker(desc)

    def _tf_cb(self, msg):
        now = time.time()
        with self._lock:
            for t in msg.transforms:
                fid = t.header.frame_id
                cid = t.child_frame_id
                ts = stamp_to_sec(t.header.stamp)

                if fid == 'map' and cid == 'odom':
                    self.trackers['tf_map_odom'].update(ts, now)

                    # 区分 HDL 原始 (老时间戳, lag>100ms) vs tf_republisher (新/未来时间戳)
                    lag = now - ts
                    if lag > 0.1:
                        self.trackers['tf_map_odom_hdl'].update(ts, now)
                    else:
                        self.trackers['tf_map_odom_repub'].update(ts, now)

                elif fid == 'camera_init' and cid == 'body':
                    self.trackers['tf_camerainit_body'].update(ts, now)

    def _hdl_pose_cb(self, msg):
        now = time.time()
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self.trackers['hdl_pose'].update(ts, now)

    def _fastlio_cb(self, msg):
        now = time.time()
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self.trackers['fastlio_odom'].update(ts, now)

    def _image_cb(self, msg, key):
        now = time.time()
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self.trackers[key].update(ts, now)

    def _sample_cb(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            self._write_report()
            self.get_logger().info(f'监控结束 ({self.duration}s), 报告已写入 {self.output_path}')
            raise SystemExit()

        with self._lock:
            entry = {
                'elapsed': round(elapsed, 1),
                'wall': round(time.time(), 3),
            }
            for key, tracker in self.trackers.items():
                if tracker.count > 0:
                    s = tracker.stats()
                    entry[key] = {
                        'hz': s.get('avg_hz', 0),
                        'lag_ms': s.get('lag_mean_ms', 0),
                        'dt_max_ms': s.get('dt_max_ms', 0),
                        'gaps': s['gaps'],
                        'future': s['future_stamps'],
                    }
            self.log_entries.append(entry)

    def _print_summary(self):
        elapsed = time.time() - self.start_time
        lines = [f'\n===== 时间戳监控 [{elapsed:.0f}s / {self.duration}s] =====']
        with self._lock:
            for key, tracker in self.trackers.items():
                if tracker.count == 0:
                    continue
                s = tracker.stats()
                lag_str = f"lag={s.get('lag_mean_ms', 0):+.0f}ms"
                dt_str = f"dt=[{s.get('dt_min_ms', 0):.0f},{s.get('dt_max_ms', 0):.0f}]ms"
                warn = ''
                if s['gaps'] > 0:
                    warn += f' ⚠ {s["gaps"]}gaps'
                if s['future_stamps'] > 0:
                    warn += f' ⏩ {s["future_stamps"]}future'
                lines.append(
                    f'  {s["name"]:<40s} '
                    f'{s.get("avg_hz", 0):6.1f}Hz  {lag_str:>12s}  {dt_str:>20s}  '
                    f'n={s["count"]}{warn}')
        lines.append('')
        print('\n'.join(lines), flush=True)

    def _write_report(self):
        # 写 JSONL 采样数据
        with open(self.output_path, 'w') as f:
            for entry in self.log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # 写可读报告
        report_path = self.output_path.replace('.jsonl', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(f'TF 时间戳监控报告\n')
            f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'监控时长: {self.duration}s\n')
            f.write(f'{"=" * 80}\n\n')

            with self._lock:
                for key, tracker in self.trackers.items():
                    if tracker.count == 0:
                        f.write(f'[{tracker.name}] — 无数据\n\n')
                        continue
                    s = tracker.stats()
                    f.write(f'[{s["name"]}]\n')
                    f.write(f'  消息数: {s["count"]}\n')
                    f.write(f'  平均频率: {s.get("avg_hz", 0):.1f} Hz\n')
                    f.write(f'  时间戳延迟 (stamp vs wall): '
                            f'min={s.get("lag_min_ms", 0):.1f}ms '
                            f'mean={s.get("lag_mean_ms", 0):.1f}ms '
                            f'max={s.get("lag_max_ms", 0):.1f}ms\n')
                    f.write(f'  消息间隔 (dt): '
                            f'min={s.get("dt_min_ms", 0):.1f}ms '
                            f'mean={s.get("dt_mean_ms", 0):.1f}ms '
                            f'max={s.get("dt_max_ms", 0):.1f}ms\n')

                    if tracker.gap_events:
                        f.write(f'  ⚠ 断供/回退事件 ({len(tracker.gap_events)}):\n')
                        for wall_t, gap in tracker.gap_events[:20]:
                            rel = wall_t - self.start_time
                            if gap > 0:
                                f.write(f'    [{rel:.1f}s] 断供 {gap*1000:.0f}ms\n')
                            else:
                                f.write(f'    [{rel:.1f}s] 时间戳回退 {gap*1000:.0f}ms\n')
                        if len(tracker.gap_events) > 20:
                            f.write(f'    ... 共 {len(tracker.gap_events)} 事件\n')

                    if tracker.future_events:
                        f.write(f'  ⏩ 未来时间戳 ({len(tracker.future_events)}):\n')
                        f.write(f'    (stamp 超前 wall clock, 前5个):\n')
                        for wall_t, ahead in tracker.future_events[:5]:
                            rel = wall_t - self.start_time
                            f.write(f'    [{rel:.1f}s] 超前 {ahead*1000:.1f}ms\n')

                    f.write('\n')

                # 交叉对比
                f.write(f'{"=" * 80}\n')
                f.write(f'交叉对比分析\n')
                f.write(f'{"=" * 80}\n\n')

                hdl = self.trackers['hdl_pose']
                fastlio = self.trackers['fastlio_odom']
                repub = self.trackers['tf_map_odom_repub']
                hdl_tf = self.trackers['tf_map_odom_hdl']

                if hdl.count > 0 and hdl_tf.count > 0:
                    hdl_s = hdl.stats()
                    hdl_tf_s = hdl_tf.stats()
                    f.write(f'HDL /pose vs HDL TF:\n')
                    f.write(f'  /pose: {hdl_s.get("avg_hz", 0):.1f}Hz, '
                            f'lag={hdl_s.get("lag_mean_ms", 0):.0f}ms\n')
                    f.write(f'  TF:    {hdl_tf_s.get("avg_hz", 0):.1f}Hz, '
                            f'lag={hdl_tf_s.get("lag_mean_ms", 0):.0f}ms\n')
                    if hdl_s.get('avg_hz', 0) > 0 and hdl_tf_s.get('avg_hz', 0) > 0:
                        ratio = hdl_tf_s['avg_hz'] / hdl_s['avg_hz']
                        if ratio < 0.8:
                            f.write(f'  ⚠ HDL TF 频率偏低 ({ratio:.1%}), 可能有 TF 发布失败\n')
                    f.write('\n')

                if repub.count > 0:
                    repub_s = repub.stats()
                    f.write(f'tf_republisher:\n')
                    f.write(f'  频率: {repub_s.get("avg_hz", 0):.1f}Hz (期望 100Hz)\n')
                    f.write(f'  stamp 超前: {-repub_s.get("lag_mean_ms", 0):.0f}ms '
                            f'(配置 stamp_ahead=50ms)\n')
                    if repub_s['gaps'] > 0:
                        f.write(f'  ⚠ {repub_s["gaps"]} 次断供!\n')
                    f.write('\n')

                cam_top = self.trackers['cam_top_color']
                cam_chassis = self.trackers['cam_chassis_color']
                if cam_top.count > 0 and cam_chassis.count > 0:
                    top_s = cam_top.stats()
                    ch_s = cam_chassis.stats()
                    f.write(f'相机时间戳对比:\n')
                    f.write(f'  top:     {top_s.get("avg_hz", 0):.1f}Hz, '
                            f'lag={top_s.get("lag_mean_ms", 0):.0f}ms\n')
                    f.write(f'  chassis: {ch_s.get("avg_hz", 0):.1f}Hz, '
                            f'lag={ch_s.get("lag_mean_ms", 0):.0f}ms\n')
                    lag_diff = abs(top_s.get('lag_mean_ms', 0) - ch_s.get('lag_mean_ms', 0))
                    if lag_diff > 50:
                        f.write(f'  ⚠ 两相机时间戳差异 {lag_diff:.0f}ms, 可能影响 TF 查询\n')
                    else:
                        f.write(f'  ✓ 两相机时间戳差异 {lag_diff:.0f}ms, 正常\n')
                    f.write('\n')

        self.get_logger().info(f'报告已写入: {report_path}')


def main():
    parser = argparse.ArgumentParser(description='TF/HDL/FastLIO/Camera 时间戳监控')
    parser.add_argument('--duration', type=int, default=30, help='监控时长 (秒)')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'scripts/ts_monitor_{ts}.jsonl'

    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    rclpy.init()
    node = TimestampMonitor(args.duration, args.output)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
