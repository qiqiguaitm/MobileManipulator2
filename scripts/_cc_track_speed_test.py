#!/usr/bin/env python3
"""
Track Stability Speed Test — 测试不同速度下 ByteTracker3D 的匹配稳定性

用法:
    # 前置: 确保感知 + 底盘驱动已运行
    make percept-full          # 终端1
    make can-bringup-auto      # 终端2 (如果还没做)
    ros2 launch tracer_base tracer_base.launch.py port_name:=can1 publish_tf:=false  # 终端2

    # 运行测试
    python3 scripts/_cc_track_speed_test.py                  # 默认 0.3 m/s
    python3 scripts/_cc_track_speed_test.py --speed 0.5      # 测试 0.5 m/s
    python3 scripts/_cc_track_speed_test.py --speed 0.1 --duration 8  # 慢速长测
    python3 scripts/_cc_track_speed_test.py --angular 0.3    # 原地旋转测试
"""

import argparse
import atexit
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist, Point

from perception.msg import Object3DArray, Object3D


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TrackSnapshot:
    track_id: str
    category: str
    position: np.ndarray  # [x, y, z] in base_link


@dataclass
class FrameRecord:
    timestamp: float
    phase: str
    speed: float
    angular: float
    tracks: Dict[str, TrackSnapshot] = field(default_factory=dict)
    kept: List[str] = field(default_factory=list)
    lost: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    mutations: List[Tuple[str, str]] = field(default_factory=list)  # (old_id, new_id)


# ============================================================
# 主节点
# ============================================================

class TrackSpeedTest(Node):

    MUTATION_DIST_THRESH = 0.08  # 8cm: 新旧 track 位置差小于此 → 判为 ID 突变
    MAX_SPEED = 0.5              # 硬编码速度上限
    CMD_VEL_HZ = 10             # cmd_vel 发布频率

    def __init__(self, args):
        super().__init__('track_speed_test')

        self.target_linear = min(abs(args.speed), self.MAX_SPEED)
        self.target_angular = args.angular
        self.static_duration = args.static
        self.moving_duration = args.duration
        self.stop_duration = args.stop

        # Phase: WAIT → STATIC → MOVING → STOP → DONE
        self.phase = 'WAIT'
        self.phase_start = 0.0
        self.start_time = time.time()
        self.current_linear = 0.0
        self.current_angular = 0.0

        # Track data
        self.baseline_tracks: Dict[str, TrackSnapshot] = {}
        self.prev_tracks: Dict[str, TrackSnapshot] = {}
        self.records: List[FrameRecord] = []
        self.frame_count = 0

        # ROS interfaces
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.fused_sub = self.create_subscription(
            Object3DArray,
            '/multi_camera_perception/fused/objects_3d',
            self._fused_callback, qos)
        self.vel_timer = self.create_timer(1.0 / self.CMD_VEL_HZ, self._vel_tick)

        # Safety
        atexit.register(self._emergency_stop)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.get_logger().info(
            f'Track Speed Test 启动: linear={self.target_linear} m/s, '
            f'angular={self.target_angular} rad/s, '
            f'phases: static={self.static_duration}s move={self.moving_duration}s stop={self.stop_duration}s')

    # ── Safety ──

    def _emergency_stop(self):
        """确保底盘停车"""
        try:
            twist = Twist()
            for _ in range(10):
                self.cmd_pub.publish(twist)
                time.sleep(0.05)
        except Exception:
            pass

    def _signal_handler(self, sig, frame):
        self.get_logger().warn('收到中断信号，紧急停车...')
        self._emergency_stop()
        self._print_summary()
        sys.exit(0)

    # ── Velocity Control ──

    def _vel_tick(self):
        now = time.time()
        elapsed = now - self.start_time

        # 状态机转换
        if self.phase == 'WAIT':
            if self.frame_count > 0:
                self.phase = 'STATIC'
                self.phase_start = now
                self.get_logger().info(f'[STATIC] 开始静止基线采集 ({self.static_duration}s)...')
        elif self.phase == 'STATIC':
            if now - self.phase_start >= self.static_duration:
                self.phase = 'MOVING'
                self.phase_start = now
                self.baseline_tracks = dict(self.prev_tracks)
                n = len(self.baseline_tracks)
                ids = ', '.join(sorted(self.baseline_tracks.keys()))
                self.get_logger().info(
                    f'[MOVING] 基线 {n} tracks: [{ids}]')
                self.get_logger().info(
                    f'[MOVING] 开始运动: linear={self.target_linear} m/s, '
                    f'angular={self.target_angular} rad/s ({self.moving_duration}s)...')
        elif self.phase == 'MOVING':
            if now - self.phase_start >= self.moving_duration:
                self.phase = 'STOP'
                self.phase_start = now
                self.get_logger().info(f'[STOP] 停车，观察恢复 ({self.stop_duration}s)...')
        elif self.phase == 'STOP':
            if now - self.phase_start >= self.stop_duration:
                self.phase = 'DONE'
                self.get_logger().info('[DONE] 测试完成')
                self._emergency_stop()
                self._print_summary()
                rclpy.shutdown()
                return

        # 速度输出
        if self.phase == 'MOVING':
            self.current_linear = self.target_linear
            self.current_angular = self.target_angular
        else:
            self.current_linear = 0.0
            self.current_angular = 0.0

        twist = Twist()
        twist.linear.x = self.current_linear
        twist.angular.z = self.current_angular
        self.cmd_pub.publish(twist)

        # 超时保护
        total_max = self.static_duration + self.moving_duration + self.stop_duration + 30
        if elapsed > total_max:
            self.get_logger().error(f'超时 ({total_max}s)，紧急停车')
            self._emergency_stop()
            self._print_summary()
            rclpy.shutdown()

    # ── Fused Callback ──

    def _fused_callback(self, msg: Object3DArray):
        now = time.time()
        self.frame_count += 1

        # 解析当前帧 tracks
        current: Dict[str, TrackSnapshot] = {}
        for obj in msg.objects:
            oid = obj.object_id
            if not oid.startswith('track_'):
                continue
            current[oid] = TrackSnapshot(
                track_id=oid,
                category=obj.category,
                position=np.array([obj.position.x, obj.position.y, obj.position.z])
            )

        # 与上一帧对比
        prev_ids = set(self.prev_tracks.keys())
        curr_ids = set(current.keys())
        kept = sorted(curr_ids & prev_ids)
        lost = sorted(prev_ids - curr_ids)
        new = sorted(curr_ids - prev_ids)

        # 检测 ID 突变: lost 和 new 中位置接近的对
        mutations = []
        if lost and new:
            for lid in lost:
                lpos = self.prev_tracks[lid].position
                for nid in new:
                    npos = current[nid].position
                    dist = np.linalg.norm(lpos - npos)
                    if dist < self.MUTATION_DIST_THRESH:
                        mutations.append((lid, nid))

        # 记录
        rec = FrameRecord(
            timestamp=now - self.start_time,
            phase=self.phase,
            speed=self.current_linear,
            angular=self.current_angular,
            tracks=current,
            kept=kept,
            lost=lost,
            new=new,
            mutations=mutations,
        )
        self.records.append(rec)

        # 实时日志
        if self.phase in ('MOVING', 'STOP'):
            baseline_ids = set(self.baseline_tracks.keys())
            surviving = curr_ids & baseline_ids
            retention = len(surviving) / max(len(baseline_ids), 1) * 100

            parts = [
                f'[{rec.timestamp:5.1f}s] {self.phase:6s}',
                f'v={self.current_linear:.2f}+w={self.current_angular:.2f}',
                f'tracks={len(current):2d}',
                f'ret={retention:3.0f}%({len(surviving)}/{len(baseline_ids)})',
            ]
            if lost:
                parts.append(f'LOST={lost}')
            if new:
                parts.append(f'NEW={new}')
            if mutations:
                for old_id, new_id in mutations:
                    parts.append(f'MUTATION:{old_id}→{new_id}')
            self.get_logger().info(' | '.join(parts))
        elif self.phase == 'STATIC' and self.frame_count % 5 == 0:
            self.get_logger().info(
                f'[{rec.timestamp:5.1f}s] STATIC | tracks={len(current):2d} '
                f'[{", ".join(sorted(current.keys()))}]')

        self.prev_tracks = current

    # ── Summary ──

    def _print_summary(self):
        if not self.records:
            print('\n  无数据，跳过统计')
            return

        baseline_ids = set(self.baseline_tracks.keys())
        n_baseline = len(baseline_ids)
        if n_baseline == 0:
            print('\n  基线无 track，无法统计')
            return

        # 分阶段统计
        moving_recs = [r for r in self.records if r.phase == 'MOVING']
        stop_recs = [r for r in self.records if r.phase == 'STOP']

        # ── Moving phase ──
        total_lost_events = 0
        total_mutations = 0
        retention_list = []
        lost_ids_all = []
        mutation_log = []

        for r in moving_recs:
            curr_ids = set(r.tracks.keys())
            surviving = curr_ids & baseline_ids
            retention_list.append(len(surviving) / n_baseline * 100)
            total_lost_events += len(r.lost)
            total_mutations += len(r.mutations)
            for lid in r.lost:
                lost_ids_all.append((r.timestamp, lid))
            for old_id, new_id in r.mutations:
                mutation_log.append((r.timestamp, old_id, new_id))

        # ── Stop phase recovery ──
        if stop_recs:
            last_stop = stop_recs[-1]
            final_ids = set(last_stop.tracks.keys())
            recovered = final_ids & baseline_ids
        else:
            recovered = set()

        # ── Print ──
        print('\n')
        print('=' * 65)
        print('  Track Stability Report')
        print('=' * 65)
        print(f'\n  速度: linear={self.target_linear} m/s, angular={self.target_angular} rad/s')
        print(f'  阶段: static={self.static_duration}s → move={self.moving_duration}s → stop={self.stop_duration}s')

        print(f'\n  ── 基线 (静止) ──')
        print(f'  Track 数量: {n_baseline}')
        for tid in sorted(baseline_ids):
            t = self.baseline_tracks[tid]
            print(f'    {tid:12s} ({t.category:15s}) pos=({t.position[0]:.2f}, {t.position[1]:.2f}, {t.position[2]:.2f})')

        if moving_recs:
            print(f'\n  ── 运动阶段 ({len(moving_recs)} frames) ──')
            ret_arr = np.array(retention_list) if retention_list else np.array([0])
            print(f'  Retention: avg={ret_arr.mean():.1f}%  min={ret_arr.min():.0f}%  max={ret_arr.max():.0f}%')
            print(f'  Lost events:  {total_lost_events} (帧间丢失次数)')
            print(f'  ID mutations: {total_mutations}')

            if mutation_log:
                print(f'\n  ── ID 突变详情 ──')
                for ts, old_id, new_id in mutation_log:
                    print(f'    {ts:5.1f}s: {old_id} → {new_id}')

            if lost_ids_all:
                # 统计哪些 ID 丢失最频繁
                from collections import Counter
                lost_counter = Counter(lid for _, lid in lost_ids_all)
                print(f'\n  ── 丢失频次 (top 5) ──')
                for lid, cnt in lost_counter.most_common(5):
                    print(f'    {lid}: {cnt} 次')

        if stop_recs:
            print(f'\n  ── 停车恢复 ──')
            print(f'  基线 {n_baseline} tracks → 恢复 {len(recovered)}/{n_baseline}')
            missing = baseline_ids - recovered
            if missing:
                print(f'  未恢复: {sorted(missing)}')
            else:
                print(f'  全部恢复 ✓')

        # Verdict
        print()
        if retention_list:
            avg_ret = np.mean(retention_list)
            if avg_ret >= 90 and total_mutations == 0:
                verdict = f'✓ {self.target_linear} m/s 可用 (retention {avg_ret:.0f}%, 0 mutations)'
            elif avg_ret >= 75:
                verdict = f'⚠ {self.target_linear} m/s 勉强可用 (retention {avg_ret:.0f}%, {total_mutations} mutations)'
            else:
                verdict = f'✗ {self.target_linear} m/s 不稳定 (retention {avg_ret:.0f}%, {total_mutations} mutations)'
            print(f'  结论: {verdict}')
        print('=' * 65)
        print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Track Stability Speed Test')
    parser.add_argument('--speed', type=float, default=0.3,
                        help='线性速度 m/s (default: 0.3)')
    parser.add_argument('--angular', type=float, default=0.0,
                        help='角速度 rad/s (default: 0, 纯直线)')
    parser.add_argument('--static', type=float, default=3.0,
                        help='静止基线采集时间 s (default: 3)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='运动持续时间 s (default: 5)')
    parser.add_argument('--stop', type=float, default=3.0,
                        help='停车观察时间 s (default: 3)')
    args = parser.parse_args()

    rclpy.init()
    node = TrackSpeedTest(args)

    # 前置检查
    time.sleep(0.5)  # 等 discovery
    topics = [t[0] for t in node.get_topic_names_and_types()]

    fused_topic = '/multi_camera_perception/fused/objects_3d'
    if fused_topic not in topics:
        print(f'\n  ✗ 感知未运行 (找不到 {fused_topic})')
        print(f'  请先: make percept-full\n')
        rclpy.shutdown()
        return

    has_chassis = any('tracer' in t for t in topics)
    if not has_chassis:
        print('\n  ⚠ 底盘驱动可能未运行 (找不到 tracer 相关 topic)')
        print('  如需实际运动，请先:')
        print('    make can-bringup-auto')
        print('    ros2 launch tracer_base tracer_base.launch.py port_name:=can1 publish_tf:=false')
        print('  继续运行（仅监控 track，不驱动底盘）...\n')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._emergency_stop()
        node.destroy_node()


if __name__ == '__main__':
    main()
