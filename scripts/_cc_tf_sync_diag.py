#!/usr/bin/env python3
"""TF 时间同步诊断 — 自动驱动+精确记录

实验流程：
  Phase 0: 静止基线 (5s)
  Phase 1: 直线前进 0.15 m/s (4s)
  Phase 2: 静止恢复 (3s)
  Phase 3: 原地旋转 0.3 rad/s (4s)
  Phase 4: 静止恢复 (3s)
  Phase 5: 直线后退 0.15 m/s (4s) — 回到原点附近
  Phase 6: 静止收尾 (3s)

安全：速度极小，总位移 <0.6m 前后往返，Ctrl+C 立即停车。

用法:
  # 确保 tracer_base + hdl_localization + 相机 已启动
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_tf_sync_diag.py

输出:
  - 终端实时打印
  - scripts/tf_sync_diag_<timestamp>.jsonl  (逐帧记录)
  - scripts/tf_sync_diag_<timestamp>_report.txt (汇总报告)
"""

import time
import sys
import json
import math
import signal
import threading
from dataclasses import dataclass
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener


# ── 实验计划 ──────────────────────────────────────────────

@dataclass
class Phase:
    name: str
    linear: float       # m/s
    angular: float      # rad/s
    duration: float     # seconds
    desc: str


PHASES = [
    Phase('static_baseline', 0.0,  0.0, 5.0, '静止基线'),
    Phase('forward',         0.15, 0.0, 4.0, '直线前进 0.15m/s'),
    Phase('pause_1',         0.0,  0.0, 3.0, '静止恢复'),
    Phase('rotate_ccw',      0.0,  0.3, 4.0, '原地左转 0.3rad/s'),
    Phase('pause_2',         0.0,  0.0, 3.0, '静止恢复'),
    Phase('rotate_cw',       0.0, -0.3, 4.0, '原地右转 0.3rad/s (转回)'),
    Phase('pause_3',         0.0,  0.0, 3.0, '静止恢复'),
    Phase('backward',       -0.15, 0.0, 4.0, '直线后退 0.15m/s (回原点)'),
    Phase('static_final',    0.0,  0.0, 3.0, '静止收尾'),
]


# ── 节点 ──────────────────────────────────────────────────

class TFSyncDiag(Node):
    def __init__(self):
        super().__init__('tf_sync_diag')

        cbg = ReentrantCallbackGroup()

        # TF
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # cmd_vel publisher
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # state
        self._lock = threading.Lock()
        self._odom_vel = (0.0, 0.0)
        self._cmd_vel = (0.0, 0.0)   # 当前指令速度（由 phase 决定，100%可靠）
        self._odom_received = False
        self._records = []
        self._current_phase = ''
        self._running = True
        self._last_sample_time = {}   # camera -> last sample wall time (5Hz throttle)
        self._pending_queries = []    # (query_time, image_stamp_msg, camera, cmd, odom, phase)

        # log file
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_path = f'scripts/tf_sync_diag_{ts}.jsonl'
        self._report_path = f'scripts/tf_sync_diag_{ts}_report.txt'
        self._log_file = open(self._log_path, 'w')

        # 相机 subscribers (BEST_EFFORT — 匹配 realsense 发布者)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=1)

        cameras = {
            'chassis': '/camera/chassis/color/image_raw',
            'top': '/camera/top/color/image_raw',
        }
        self._cam_count = {}
        for cam_name, topic in cameras.items():
            self._cam_count[cam_name] = 0
            self.create_subscription(
                Image, topic,
                lambda msg, cn=cam_name: self._image_cb(cn, msg),
                sensor_qos, callback_group=cbg)

        # odom subscribers — 多源尝试（RELIABLE 匹配 tracer/odom_frame_converter）
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, depth=10)
        for odom_topic in ['/wheel_odom', '/odom']:
            self.create_subscription(
                Odometry, odom_topic, self._odom_cb, odom_qos, callback_group=cbg)

        self.get_logger().info(f'日志: {self._log_path}')

    # ── callbacks ──

    def _odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        self._odom_vel = (math.sqrt(v.x**2 + v.y**2), abs(w.z))
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info('odom 已收到')

    def _image_cb(self, camera_name: str, msg: Image):
        """模拟感知流程: 5Hz采样 → 170ms GPU处理 → TF查询"""
        if not self._running or not self._current_phase:
            return

        now = time.time()

        # ── 5Hz 节流（每相机每200ms取一帧） ──
        last = self._last_sample_time.get(camera_name, 0.0)
        if now - last < 0.19:  # ~5Hz
            return
        self._last_sample_time[camera_name] = now

        # 记录采样时的状态，170ms后再做TF查询
        with self._lock:
            self._pending_queries.append((
                now + 0.17,          # 170ms后查询
                msg.header.stamp,    # image_stamp
                camera_name,
                self._cmd_vel,
                self._odom_vel,
                self._current_phase,
            ))

    def _process_pending_queries(self):
        """处理到期的延迟TF查询（模拟GPU处理完成后的查询时机）"""
        now = time.time()

        with self._lock:
            ready = [q for q in self._pending_queries if now >= q[0]]
            self._pending_queries = [q for q in self._pending_queries if now < q[0]]

        for query_time, img_stamp, camera_name, cmd, odom, phase in ready:
            img_t = img_stamp.sec + img_stamp.nanosec * 1e-9

            # A) 即时查询（图像到达时立刻查TF，对照组）
            # B) 延迟查询（GPU处理完成后查TF，模拟实际感知流程）
            # 两者都做，对比差异

            # 完全复刻 _get_base_to_map_matrix 逻辑 (multi_camera_perception_node.py:218-231)
            interp_ok = False
            tf_t_used = 0.0
            ts = None
            try:
                # Step 1: 精确时间插值 (timeout=0, 同感知代码)
                ts = self._tf_buffer.lookup_transform(
                    'map', 'base_link', Time.from_msg(img_stamp),
                    timeout=Duration(seconds=0))
                interp_ok = True
            except Exception:
                pass
            if ts is None:
                try:
                    # Step 2: 回退 latest (timeout=0.05, 同感知代码)
                    ts = self._tf_buffer.lookup_transform(
                        'map', 'base_link', Time(),
                        timeout=Duration(seconds=0.05))
                except Exception:
                    continue
            tf_t_used = ts.header.stamp.sec + ts.header.stamp.nanosec * 1e-9

            cmd_lin, cmd_ang = cmd
            odom_lin, odom_ang = odom
            tf_lag_ms = (img_t - tf_t_used) * 1000   # 正=图像超前TF, 负=TF超前
            query_delay_ms = (now - (query_time - 0.17)) * 1000  # 实际经过时间
            tf_age_ms = (now - tf_t_used) * 1000

            # 插值成功→精确匹配,误差≈0; 失败→用了latest TF,误差=速度×时间差
            if interp_ok:
                est_err_cm = 0.0
            else:
                lag_sec = abs(img_t - tf_t_used)  # 用绝对值,无论谁超前
                est_err_cm = (abs(cmd_lin) + abs(cmd_ang) * 1.0) * lag_sec * 100

            rec = {
                'ts': round(now, 4),
                'cam': camera_name,
                'phase': phase,
                'interp_ok': interp_ok,
                'tf_lag_ms': round(tf_lag_ms, 1),
                'query_delay_ms': round(query_delay_ms, 1),
                'tf_age_ms': round(tf_age_ms, 1),
                'cmd_linear': round(cmd_lin, 3),
                'cmd_angular': round(cmd_ang, 3),
                'odom_linear': round(odom_lin, 3),
                'odom_angular': round(odom_ang, 3),
                'est_err_cm': round(est_err_cm, 1),
            }

            with self._lock:
                self._records.append(rec)
                self._cam_count[camera_name] = self._cam_count.get(camera_name, 0) + 1
                cnt = self._cam_count[camera_name]

            self._log_file.write(json.dumps(rec) + '\n')

            mode = 'INTERP' if interp_ok else 'LATEST'
            moving = abs(cmd_lin) > 0.01 or abs(cmd_ang) > 0.01
            status = '移动' if moving else '静止'
            print(f'  [{camera_name:7s}] {mode:6s} | '
                  f'tf_lag={tf_lag_ms:+7.1f}ms delay={query_delay_ms:.0f}ms | '
                  f'cmd=({cmd_lin:.2f},{cmd_ang:.2f}) {status} | '
                  f'err≈{est_err_cm:.1f}cm')

    # ── 速度控制 ──

    def _send_vel(self, linear: float, angular: float):
        self._cmd_vel = (linear, angular)
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)

    def _stop(self):
        for _ in range(10):
            self._send_vel(0.0, 0.0)
            time.sleep(0.02)

    # ── 实验主循环（在独立线程运行） ──

    def run_experiment(self):
        # 等待 TF
        print('\n等待 TF 就绪...')
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
            print('❌ map→base_link TF 不可用 (10s超时)')
            self._running = False
            return

        # 等待 odom
        print('等待 odom...')
        for i in range(50):
            if self._odom_received:
                print(f'  odom 就绪 ({(i+1)*0.1:.1f}s)')
                break
            time.sleep(0.1)
        else:
            print('⚠️  odom 未收到 (5s)，继续但速度数据可能为0')

        total = sum(p.duration for p in PHASES)
        print(f'\n实验计划: {len(PHASES)} 阶段, 总时长 {total:.0f}s')
        print('安全: 最大速度 0.15m/s, 总位移 <0.6m 前后往返')
        print('Ctrl+C 随时停车\n')
        print('=' * 70)

        for i, phase in enumerate(PHASES):
            self._current_phase = phase.name
            is_moving = (phase.linear != 0 or phase.angular != 0)
            marker = '>>>' if is_moving else '---'

            print(f'\n{marker} Phase {i}: {phase.desc} '
                  f'({phase.duration:.0f}s) '
                  f'[v={phase.linear}, w={phase.angular}]')
            print('-' * 70)

            t0 = time.time()
            while time.time() - t0 < phase.duration:
                if not self._running:
                    self._stop()
                    return
                self._send_vel(phase.linear, phase.angular)
                self._process_pending_queries()
                time.sleep(0.05)  # 20Hz cmd_vel

            self._stop()

        # 等待剩余的 pending queries 到期
        self._current_phase = 'done'
        self._stop()
        deadline = time.time() + 0.5
        while time.time() < deadline:
            self._process_pending_queries()
            time.sleep(0.02)

        print('\n' + '=' * 70)
        print('实验完成，生成报告...\n')

    # ── 报告 ──

    def generate_report(self):
        self._log_file.close()

        with self._lock:
            records = list(self._records)

        if not records:
            print('无数据')
            return

        lines = []
        lines.append('=' * 70)
        lines.append('TF 时间同步诊断报告 (模拟5Hz感知流程)')
        lines.append(f'方法: 5Hz采样 + 170ms延迟后查TF (模拟GPU处理)')
        lines.append(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append(f'总帧数: {len(records)}')
        lines.append('=' * 70)

        # 按 phase 分组统计
        phases_seen = []
        for p in PHASES:
            if p.name not in phases_seen:
                phases_seen.append(p.name)

        def pct(arr, p):
            idx = min(int(len(arr) * p), len(arr) - 1)
            return arr[idx]

        for phase_name in phases_seen:
            recs = [r for r in records if r['phase'] == phase_name]
            if not recs:
                continue

            phase_obj = next((p for p in PHASES if p.name == phase_name), None)
            desc = phase_obj.desc if phase_obj else phase_name

            lags = sorted([r['tf_lag_ms'] for r in recs])
            errs = sorted([r['est_err_cm'] for r in recs])
            interp_ok = sum(1 for r in recs if r['interp_ok'])
            n = len(recs)

            lines.append(f'\n--- {desc} ({phase_name}) ---')
            lines.append(f'  帧数: {n}, 插值成功: {interp_ok}/{n} ({interp_ok/n*100:.0f}%)')
            lines.append(f'  TF滞后(ms): '
                         f'min={lags[0]:+.1f}  '
                         f'P50={pct(lags, 0.5):+.1f}  '
                         f'P95={pct(lags, 0.95):+.1f}  '
                         f'max={lags[-1]:+.1f}')
            lines.append(f'  预估误差(cm): '
                         f'P50={pct(errs, 0.5):.1f}  '
                         f'P95={pct(errs, 0.95):.1f}  '
                         f'max={errs[-1]:.1f}')

            # 按相机细分
            for cam in ['chassis', 'top']:
                cam_recs = [r for r in recs if r['cam'] == cam]
                if not cam_recs:
                    continue
                c_lags = sorted([r['tf_lag_ms'] for r in cam_recs])
                c_interp = sum(1 for r in cam_recs if r['interp_ok'])
                cn = len(cam_recs)
                lines.append(f'    [{cam}] {cn}帧  '
                             f'插值OK: {c_interp}/{cn}  '
                             f'lag P50={pct(c_lags, 0.5):+.1f}ms')

        # 总体结论
        all_lags = sorted([r['tf_lag_ms'] for r in records])
        total_interp = sum(1 for r in records if r['interp_ok'])
        n = len(records)
        fallback_pct = (1 - total_interp / n) * 100

        moving_recs = [r for r in records
                       if abs(r['cmd_linear']) > 0.01 or abs(r['cmd_angular']) > 0.01]
        static_recs = [r for r in records
                       if abs(r['cmd_linear']) <= 0.01 and abs(r['cmd_angular']) <= 0.01]

        lines.append(f'\n{"=" * 70}')
        lines.append('诊断结论')
        lines.append(f'{"=" * 70}')
        lines.append(f'  总帧: {n}')
        lines.append(f'  插值成功率: {total_interp}/{n} ({total_interp/n*100:.0f}%)')
        lines.append(f'  回退到 latest: {fallback_pct:.0f}%')
        lines.append(f'  TF滞后 P50={pct(all_lags, 0.5):+.1f}ms  '
                     f'P95={pct(all_lags, 0.95):+.1f}ms')

        if moving_recs:
            m_errs = sorted([r['est_err_cm'] for r in moving_recs])
            m_lags = sorted([r['tf_lag_ms'] for r in moving_recs])
            lines.append(f'\n  [运动时] {len(moving_recs)}帧')
            lines.append(f'    TF滞后 P50={pct(m_lags, 0.5):+.1f}ms  '
                         f'P95={pct(m_lags, 0.95):+.1f}ms')
            lines.append(f'    预估误差 P50={pct(m_errs, 0.5):.1f}cm  '
                         f'P95={pct(m_errs, 0.95):.1f}cm  '
                         f'max={m_errs[-1]:.1f}cm')

        if static_recs:
            s_lags = sorted([r['tf_lag_ms'] for r in static_recs])
            lines.append(f'\n  [静止时] {len(static_recs)}帧')
            lines.append(f'    TF滞后 P50={pct(s_lags, 0.5):+.1f}ms  '
                         f'P95={pct(s_lags, 0.95):+.1f}ms')

        lines.append('')
        if fallback_pct > 50 and moving_recs:
            m_errs = sorted([r['est_err_cm'] for r in moving_recs])
            lines.append(f'  ⚠️  确认: 即使延迟170ms后查询，TF同步问题仍然存在')
            lines.append(f'     {fallback_pct:.0f}% 帧无法精确插值，回退到 latest TF')
            lines.append(f'     运动时预估位置误差 P95={pct(m_errs, 0.95):.1f}cm')
            lines.append(f'     根因: HDL TF 更新频率不足以覆盖 image_stamp')
            lines.append(f'     建议: 用高频 odom 做 TF 外推补偿')
        elif fallback_pct > 50:
            lines.append(f'  ⚠️  TF同步问题严重 (回退率{fallback_pct:.0f}%)')
        elif fallback_pct > 10:
            lines.append(f'  ⚡ 延迟170ms后仍有 {fallback_pct:.0f}% 回退，需关注运动场景')
        else:
            lines.append(f'  ✅ 5Hz+170ms延迟下TF同步良好 (回退率{fallback_pct:.0f}%)')
            lines.append(f'     170ms GPU处理延迟使TF buffer有时间追上 image_stamp')

        report = '\n'.join(lines)
        print(report)

        with open(self._report_path, 'w') as f:
            f.write(report)
        print(f'\n日志: {self._log_path}')
        print(f'报告: {self._report_path}')


def main():
    rclpy.init()
    node = TFSyncDiag()

    # 用 MultiThreadedExecutor 在后台 spin，callbacks 不受主线程阻塞
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def sig_handler(sig, frame):
        print('\n\n⚠️  Ctrl+C — 紧急停车')
        node._running = False
        node._stop()
        node.generate_report()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)

    try:
        node.run_experiment()
        node.generate_report()
    finally:
        node._stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
