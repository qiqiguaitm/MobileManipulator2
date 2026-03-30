#!/usr/bin/env python3
"""感知位置漂移诊断 — 基于 track_id 追踪同一目标在机器人运动中的 map 系位置变化

订阅 /multi_camera_perception/fused/objects_3d (带 ByteTracker3D track_id),
按 object_id 分组追踪, 记录每个 track 在不同运动阶段的位置漂移。

实验: 静止→前进→静止→左转→静止→右转→静止→后退→静止

用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_position_drift_diag.py
"""

import time
import json
import math
import signal
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from perception.msg import Object3DArray


@dataclass
class Phase:
    name: str
    linear: float
    angular: float
    duration: float
    desc: str


PHASES = [
    Phase('static_0',   0.0,   0.0, 5.0, '静止基线'),
    Phase('L1',         0.0,   0.6, 0.7, '左转 ~24°'),       # 0.6×0.7=0.42rad≈24°
    Phase('pause_1',    0.0,   0.0, 2.0, '停顿'),
    Phase('R1',         0.0,  -0.6, 0.7, '右转 ~24° (回正)'),
    Phase('pause_2',    0.0,   0.0, 2.0, '停顿'),
    Phase('R2',         0.0,  -0.6, 0.7, '右转 ~24°'),
    Phase('pause_3',    0.0,   0.0, 2.0, '停顿'),
    Phase('L2',         0.0,   0.6, 0.7, '左转 ~24° (回正)'),
    Phase('pause_4',    0.0,   0.0, 2.0, '停顿'),
    Phase('L3',         0.0,   0.6, 0.7, '左转 ~24°'),
    Phase('pause_5',    0.0,   0.0, 2.0, '停顿'),
    Phase('R3',         0.0,  -0.6, 0.7, '右转 ~24° (回正)'),
    Phase('pause_6',    0.0,   0.0, 2.0, '停顿'),
    Phase('R4',         0.0,  -0.6, 0.7, '右转 ~24°'),
    Phase('pause_7',    0.0,   0.0, 2.0, '停顿'),
    Phase('L4',         0.0,   0.6, 0.7, '左转 ~24° (回正)'),
    Phase('static_end', 0.0,   0.0, 5.0, '静止收尾'),
]


class PositionDriftDiag(Node):
    def __init__(self):
        super().__init__('position_drift_diag')
        cbg = ReentrantCallbackGroup()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._lock = threading.Lock()
        self._cmd_vel = (0.0, 0.0)
        self._current_phase = ''
        self._running = True

        # track_id → list of observations
        self._tracks = defaultdict(list)
        # track_id → first observed position (baseline)
        self._baselines = {}

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_path = f'scripts/pos_drift_{ts}.jsonl'
        self._report_path = f'scripts/pos_drift_{ts}_report.txt'
        self._log_file = open(self._log_path, 'w')

        self.create_subscription(
            Object3DArray,
            '/multi_camera_perception/fused/objects_3d',
            self._fused_cb, 10, callback_group=cbg)

        self.get_logger().info(f'日志: {self._log_path}')

    # ------------------------------------------------------------------
    def _fused_cb(self, msg: Object3DArray):
        if not self._running or not self._current_phase:
            return

        frame = msg.header.frame_id
        now = time.time()
        cmd_lin, cmd_ang = self._cmd_vel
        is_static = abs(cmd_lin) <= 0.01 and abs(cmd_ang) <= 0.01

        for obj in msg.objects:
            oid = obj.object_id
            # 只追踪有 track_id 的（ByteTracker3D 确认的轨迹）
            if not oid or not oid.startswith('track_'):
                continue

            pos = np.array([obj.position.x, obj.position.y, obj.position.z])

            # 第一次见到此 track → 记基线
            if oid not in self._baselines:
                self._baselines[oid] = {
                    'pos': pos.copy(),
                    'cat': obj.category,
                    'phase': self._current_phase,
                    't0': now,
                }
                print(f'  [baseline] {oid} ({obj.category}) = '
                      f'({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) '
                      f'frame={frame}')

            bl = self._baselines[oid]
            drift_3d = float(np.linalg.norm(pos - bl['pos'])) * 100  # cm
            drift_2d = float(np.linalg.norm(pos[:2] - bl['pos'][:2])) * 100
            dx = (pos[0] - bl['pos'][0]) * 100
            dy = (pos[1] - bl['pos'][1]) * 100
            dz = (pos[2] - bl['pos'][2]) * 100

            rec = {
                'ts': round(now, 4),
                'phase': self._current_phase,
                'track_id': oid,
                'cat': obj.category,
                'src': obj.source_camera,
                'frame': frame,
                'x': round(pos[0], 4),
                'y': round(pos[1], 4),
                'z': round(pos[2], 4),
                'dx_cm': round(dx, 2),
                'dy_cm': round(dy, 2),
                'dz_cm': round(dz, 2),
                'drift_2d_cm': round(drift_2d, 2),
                'drift_3d_cm': round(drift_3d, 2),
                'score': round(obj.score, 3),
                'confidence': round(obj.confidence, 3),
                'depth': round(obj.depth, 3),
                'phys': round(getattr(obj, 'physical_size', 0.0) or 0.0, 4),
                'cmd_lin': round(cmd_lin, 3),
                'cmd_ang': round(cmd_ang, 3),
                'is_static': is_static,
            }
            with self._lock:
                self._tracks[oid].append(rec)
            self._log_file.write(json.dumps(rec) + '\n')

        # 打印当前帧摘要
        tracked = [o for o in msg.objects if o.object_id.startswith('track_')]
        if tracked:
            parts = []
            for obj in tracked:
                bl = self._baselines.get(obj.object_id)
                if bl:
                    p = np.array([obj.position.x, obj.position.y, obj.position.z])
                    d = np.linalg.norm(p[:2] - bl['pos'][:2]) * 100
                    parts.append(f'{obj.object_id}:{d:.1f}cm')
            status = '静止' if is_static else '移动'
            print(f'  [{self._current_phase:10s}] {status} {len(tracked)}t '
                  f'drift=[{", ".join(parts)}]')

    # ------------------------------------------------------------------
    def _send_vel(self, linear, angular):
        self._cmd_vel = (linear, angular)
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)

    def _stop(self):
        for _ in range(10):
            self._send_vel(0.0, 0.0)
            time.sleep(0.02)

    # ------------------------------------------------------------------
    def run_experiment(self):
        print('\n等待 TF + 感知...')
        for i in range(100):
            time.sleep(0.1)
            try:
                self._tf_buffer.lookup_transform(
                    'map', 'base_link', Time(),
                    timeout=Duration(seconds=0))
                print(f'  TF 就绪 ({(i+1)*0.1:.1f}s)')
                break
            except Exception:
                pass
        else:
            print('TF 不可用'); self._running = False; return

        pubs = self.count_publishers('/multi_camera_perception/fused/objects_3d')
        print(f'  fused topic publishers: {pubs}')
        if pubs == 0:
            print('感知未运行!'); self._running = False; return

        total = sum(p.duration for p in PHASES)
        print(f'\n实验: {len(PHASES)} 阶段, {total:.0f}s')
        print('=' * 70)

        for i, phase in enumerate(PHASES):
            self._current_phase = phase.name
            moving = phase.linear != 0 or phase.angular != 0
            marker = '>>>' if moving else '---'
            print(f'\n{marker} Phase {i}: {phase.desc} ({phase.duration:.0f}s)')
            print('-' * 70)

            t0 = time.time()
            while time.time() - t0 < phase.duration:
                if not self._running:
                    self._stop(); return
                self._send_vel(phase.linear, phase.angular)
                time.sleep(0.05)
            self._stop()

        self._current_phase = 'done'
        self._stop()
        print('\n' + '=' * 70)

    # ------------------------------------------------------------------
    def generate_report(self):
        self._log_file.close()
        with self._lock:
            tracks = dict(self._tracks)

        if not tracks:
            print('无数据')
            return

        def pct(arr, p):
            if not arr: return 0.0
            s = sorted(arr)
            return s[min(int(len(s) * p), len(s) - 1)]

        lines = []
        lines.append('=' * 70)
        lines.append('track_id 位置漂移诊断 (map 坐标系, ByteTracker3D)')
        lines.append(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        total_obs = sum(len(v) for v in tracks.values())
        lines.append(f'轨迹: {len(tracks)}, 总观测: {total_obs}')
        lines.append('=' * 70)

        # 按观测数排序（最多的排前面）
        for tid, recs in sorted(tracks.items(), key=lambda x: -len(x[1])):
            if len(recs) < 3:
                continue
            bl = self._baselines.get(tid)
            if not bl:
                continue

            lines.append(f'\n{"─" * 70}')
            lines.append(f'轨迹: {tid} ({bl["cat"]}) — {len(recs)} 观测')
            lines.append(f'基线: ({bl["pos"][0]:.4f}, {bl["pos"][1]:.4f}, {bl["pos"][2]:.4f})')
            lines.append(f'{"─" * 70}')

            # 静止 vs 运动 统计
            static = [r for r in recs if r['is_static']]
            moving = [r for r in recs if not r['is_static']]

            s_2d = [r['drift_2d_cm'] for r in static]
            m_2d = [r['drift_2d_cm'] for r in moving]

            if s_2d:
                lines.append(f'  静止 ({len(static):3d}帧): '
                             f'2D P50={pct(s_2d,0.5):.1f}cm  '
                             f'P95={pct(s_2d,0.95):.1f}cm  '
                             f'max={max(s_2d):.1f}cm')
            if m_2d:
                lines.append(f'  运动 ({len(moving):3d}帧): '
                             f'2D P50={pct(m_2d,0.5):.1f}cm  '
                             f'P95={pct(m_2d,0.95):.1f}cm  '
                             f'max={max(m_2d):.1f}cm')

            # 各 Phase 详情
            lines.append(f'  各Phase:')
            for phase in PHASES:
                pr = [r for r in recs if r['phase'] == phase.name]
                if not pr:
                    lines.append(f'    {phase.desc:22s}  --')
                    continue
                d2d = [r['drift_2d_cm'] for r in pr]
                dxs = [r['dx_cm'] for r in pr]
                dys = [r['dy_cm'] for r in pr]
                dzs = [r['dz_cm'] for r in pr]
                lines.append(
                    f'    {phase.desc:22s}  n={len(pr):3d}  '
                    f'Δ2D={np.mean(d2d):5.1f}cm(max={max(d2d):.1f})  '
                    f'dx={np.mean(dxs):+5.1f} dy={np.mean(dys):+5.1f} '
                    f'dz={np.mean(dzs):+5.1f}cm')

            # 位置时间线（每2秒采样一个点）
            lines.append(f'  时间线 (每2s采样):')
            t0 = recs[0]['ts']
            bucket_size = 2.0
            buckets = defaultdict(list)
            for r in recs:
                b = int((r['ts'] - t0) / bucket_size)
                buckets[b].append(r)
            for b in sorted(buckets.keys()):
                br = buckets[b]
                mx = np.mean([r['x'] for r in br])
                my = np.mean([r['y'] for r in br])
                d2d = np.mean([r['drift_2d_cm'] for r in br])
                ph = br[0]['phase']
                t_sec = b * bucket_size
                lines.append(f'    t={t_sec:5.0f}s  ({mx:.3f},{my:.3f})  '
                             f'Δ2D={d2d:5.1f}cm  [{ph}]')

        # 汇总
        lines.append(f'\n{"=" * 70}')
        lines.append('汇总 (所有 track)')
        lines.append(f'{"=" * 70}')

        all_static_2d = []
        all_moving_2d = []
        for tid, recs in tracks.items():
            if tid not in self._baselines:
                continue
            for r in recs:
                if r['is_static']:
                    all_static_2d.append(r['drift_2d_cm'])
                else:
                    all_moving_2d.append(r['drift_2d_cm'])

        if all_static_2d:
            lines.append(f'  静止 ({len(all_static_2d)}帧): '
                         f'2D P50={pct(all_static_2d,0.5):.1f}cm  '
                         f'P95={pct(all_static_2d,0.95):.1f}cm  '
                         f'max={max(all_static_2d):.1f}cm')
        if all_moving_2d:
            lines.append(f'  运动 ({len(all_moving_2d)}帧): '
                         f'2D P50={pct(all_moving_2d,0.5):.1f}cm  '
                         f'P95={pct(all_moving_2d,0.95):.1f}cm  '
                         f'max={max(all_moving_2d):.1f}cm')

        if all_static_2d and all_moving_2d:
            ratio = pct(all_moving_2d, 0.5) / max(pct(all_static_2d, 0.5), 0.01)
            lines.append(f'  运动/静止 P50 比值: {ratio:.1f}x')

        report = '\n'.join(lines)
        print(report)
        with open(self._report_path, 'w') as f:
            f.write(report)
        print(f'\n日志: {self._log_path}')
        print(f'报告: {self._report_path}')


def main():
    rclpy.init()
    node = PositionDriftDiag()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    signal.signal(signal.SIGINT,
                  lambda s, f: (setattr(node, '_running', False), node._stop()))

    try:
        node.run_experiment()
        node.generate_report()
    finally:
        node._running = False
        node._stop()
        try:
            executor.shutdown()
            spin_thread.join(timeout=2.0)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
