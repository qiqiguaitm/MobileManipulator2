#!/usr/bin/env python3
"""三源 odom + HDL 定位 + TF 链 对比诊断.

观测维度:
  1. 频率 & 消息延迟 (fastlio, wheel_odom, odom_fused, HDL /pose)
  2. 位姿增量 (Δx, Δy, Δyaw) 四源对比
  3. 速度对比 (vx, wz)
  4. FastLIO 校准引起的 fused 跳变
  5. TF 链实时查询: map→base_link, map→odom, odom→base_link
  6. TF 回溯对比: @now vs @(now-200ms) 量化插值/外推误差
  7. 按运动状态 (STATIC/TRANSLATE/ROTATE) 分桶统计

运动状态由 wheel_odom.twist 判定 (CAN 编码器真值).

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/_cc_odom_compare.py
"""

import json
import math
import os
import signal
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy,
)
from nav_msgs.msg import Odometry

import tf2_ros

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,  # depth=1 会被 TF listener 的高频回调抢占导致丢消息
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 运动状态阈值 ──
WZ_ROTATE_THRESH = 0.05     # rad/s
VX_TRANSLATE_THRESH = 0.03  # m/s

# ── TF 查询参数 ──
TF_QUERY_HZ = 50.0
TF_BACKTEST_SEC = 0.2       # 回溯 200ms


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quat_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def tf_to_xyy(transform) -> tuple:
    """TransformStamped → (x, y, yaw)"""
    t = transform.transform
    x = t.translation.x
    y = t.translation.y
    siny = 2.0 * (t.rotation.w * t.rotation.z + t.rotation.x * t.rotation.y)
    cosy = 1.0 - 2.0 * (t.rotation.y ** 2 + t.rotation.z ** 2)
    yaw = math.atan2(siny, cosy)
    return x, y, yaw


# ── 每个 odom 源的状态 ──
@dataclass
class OdomState:
    name: str
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    wz: float = 0.0
    stamp: float = 0.0          # header.stamp (sec)
    arrive: float = 0.0         # time.time() 到达时刻
    # 增量
    dx: float = 0.0
    dy: float = 0.0
    dyaw: float = 0.0
    dt: float = 0.0
    # 频率计算
    arrive_times: deque = field(default_factory=lambda: deque(maxlen=50))
    count: int = 0
    inited: bool = False        # 首帧跳过增量计算

    @property
    def hz(self) -> float:
        if len(self.arrive_times) < 2:
            return 0.0
        span = self.arrive_times[-1] - self.arrive_times[0]
        return (len(self.arrive_times) - 1) / span if span > 0.01 else 0.0

    @property
    def age_ms(self) -> float:
        if self.arrive == 0 or self.stamp == 0:
            return -1.0
        return (self.arrive - self.stamp) * 1000

    def update(self, msg: Odometry):
        now = time.time()
        s = stamp_to_sec(msg.header.stamp)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        t = msg.twist.twist

        new_x, new_y = p.x, p.y
        new_yaw = quat_to_yaw(q)

        if self.inited:
            self.dt = s - self.stamp if s > self.stamp else 0.0
            self.dx = new_x - self.x
            self.dy = new_y - self.y
            self.dyaw = normalize_angle(new_yaw - self.yaw)
        else:
            self.inited = True

        self.x, self.y, self.yaw = new_x, new_y, new_yaw
        self.vx = t.linear.x
        self.wz = t.angular.z
        self.stamp = s
        self.arrive = now
        self.arrive_times.append(now)
        self.count += 1


# ── TF 查询结果 ──
@dataclass
class TFSnapshot:
    """一次 TF 查询的三条链 + 回溯对比"""
    ts: float = 0.0             # wall time
    motion: str = 'STATIC'
    # map→base_link @Time(0)
    m2b_ok: bool = False
    m2b_x: float = 0.0
    m2b_y: float = 0.0
    m2b_yaw: float = 0.0
    # map→odom @Time(0)  (HDL+tf_republisher 贡献)
    m2o_ok: bool = False
    m2o_x: float = 0.0
    m2o_y: float = 0.0
    m2o_yaw: float = 0.0
    # odom→base_link @Time(0)  (FastLIO 贡献)
    o2b_ok: bool = False
    o2b_x: float = 0.0
    o2b_y: float = 0.0
    o2b_yaw: float = 0.0
    # 回溯: map→base_link @(now - 200ms)
    bt_ok: bool = False
    bt_x: float = 0.0
    bt_y: float = 0.0
    bt_yaw: float = 0.0
    # 增量 (与上一次查询的差)
    d_m2b_xy: float = 0.0      # mm
    d_m2b_yaw: float = 0.0     # rad
    d_m2o_xy: float = 0.0      # mm
    d_m2o_yaw: float = 0.0     # rad
    d_o2b_xy: float = 0.0      # mm
    d_o2b_yaw: float = 0.0     # rad
    # 回溯差 (now vs backtest)
    bt_gap_xy: float = 0.0     # mm
    bt_gap_yaw: float = 0.0    # rad


class OdomCompare(Node):
    def __init__(self):
        super().__init__('odom_compare')

        # 日志
        ts_str = time.strftime('%Y%m%d_%H%M%S')
        self._log_path = os.path.join(SCRIPT_DIR, f'odom_compare_{ts_str}.jsonl')
        self._log_file = open(self._log_path, 'w')
        self.get_logger().info(f'日志: {self._log_path}')

        # 四个 odom 源
        self.fl = OdomState(name='fastlio')
        self.wh = OdomState(name='wheel')
        self.fu = OdomState(name='fused')
        self.hdl = OdomState(name='hdl')

        self.create_subscription(
            Odometry, '/fastlio/odom', self._fl_cb, SENSOR_QOS)
        self.create_subscription(
            Odometry, '/wheel_odom', self._wh_cb, SENSOR_QOS)
        self.create_subscription(
            Odometry, '/odom/fused', self._fu_cb, SENSOR_QOS)
        self.create_subscription(
            Odometry, '/pose', self._hdl_cb, SENSOR_QOS)

        # FastLIO 校准跳变检测
        self._fu_before_fl = None
        self._fl_arrive_time = 0.0
        self._calibration_events = []

        # ── TF 查询 ──
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        # 上一次 TF 快照 (用于计算增量)
        self._prev_tf = None
        self._tf_snaps = []  # 所有 TFSnapshot
        # TF 分桶统计: {motion_state: {metric_name: [values]}}
        self._tf_stats = {
            ms: {
                'd_m2b_xy': [], 'd_m2o_xy': [], 'd_o2b_xy': [],
                'bt_gap_xy': [], 'bt_gap_yaw': [],
            } for ms in ['STATIC', 'TRANSLATE', 'ROTATE']
        }
        self.create_timer(1.0 / TF_QUERY_HZ, self._tf_query_cb)

        # 按运动状态分桶的增量统计
        self._stats = {
            'STATIC':    {'fl': [], 'wh': [], 'fu': [], 'hdl': [], 'wz_diff': [], 'dyaw_diff': []},
            'TRANSLATE': {'fl': [], 'wh': [], 'fu': [], 'hdl': [], 'wz_diff': [], 'dyaw_diff': []},
            'ROTATE':    {'fl': [], 'wh': [], 'fu': [], 'hdl': [], 'wz_diff': [], 'dyaw_diff': []},
        }
        self._motion_state = 'STATIC'
        self._t0 = time.time()

        # Dashboard 定时器
        self.create_timer(0.5, self._dashboard)
        self.get_logger().info('等待 odom + TF 数据...')

    # ── odom 回调 ──
    def _fl_cb(self, msg: Odometry):
        if self.fu.inited:
            self._fu_before_fl = (self.fu.x, self.fu.y, self.fu.yaw, self.fu.stamp)
        self.fl.update(msg)
        self._fl_arrive_time = time.time()
        self._log_odom('fastlio', self.fl)

    def _wh_cb(self, msg: Odometry):
        self.wh.update(msg)
        if abs(self.wh.wz) > WZ_ROTATE_THRESH:
            self._motion_state = 'ROTATE'
        elif abs(self.wh.vx) > VX_TRANSLATE_THRESH:
            self._motion_state = 'TRANSLATE'
        else:
            self._motion_state = 'STATIC'
        self._log_odom('wheel', self.wh)

    def _fu_cb(self, msg: Odometry):
        self.fu.update(msg)
        self._log_odom('fused', self.fu)

        # 校准跳变检测
        if self._fu_before_fl and (time.time() - self._fl_arrive_time) < 0.05:
            bx, by, byaw, _ = self._fu_before_fl
            jump_xy = math.sqrt((self.fu.x - bx)**2 + (self.fu.y - by)**2) * 1000
            jump_yaw = abs(normalize_angle(self.fu.yaw - byaw))
            if jump_xy > 5 or jump_yaw > math.radians(0.5):
                evt = {
                    'src': 'calibration_jump',
                    'ts': time.time(),
                    'motion': self._motion_state,
                    'jump_xy_mm': round(jump_xy, 1),
                    'jump_yaw_deg': round(math.degrees(jump_yaw), 2),
                }
                self._calibration_events.append(evt)
                self._log_file.write(json.dumps(evt) + '\n')
            self._fu_before_fl = None

        # wheel vs fused 增量/速度对比
        if self.wh.inited and self.fu.inited:
            ms = self._motion_state
            wz_diff = abs(self.fu.wz - self.wh.wz)
            self._stats[ms]['wz_diff'].append(wz_diff)
            if self.fu.dt > 0.001 and self.wh.dt > 0.001:
                dyaw_diff = abs(abs(self.fu.dyaw) - abs(self.wh.dyaw))
                self._stats[ms]['dyaw_diff'].append(dyaw_diff)

    def _hdl_cb(self, msg: Odometry):
        self.hdl.update(msg)
        self._log_odom('hdl', self.hdl)

    # ── TF 查询 (10Hz) ──
    def _tf_query_cb(self):
        if not self.wh.inited:
            return

        snap = TFSnapshot(ts=time.time(), motion=self._motion_state)
        now = self.get_clock().now()

        # 1) map→base_link @Time(0)
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link', Time())
            snap.m2b_x, snap.m2b_y, snap.m2b_yaw = tf_to_xyy(t)
            snap.m2b_ok = True
        except Exception:
            pass

        # 2) map→odom @Time(0)  (HDL + tf_republisher 平滑后)
        try:
            t = self._tf_buffer.lookup_transform('map', 'odom', Time())
            snap.m2o_x, snap.m2o_y, snap.m2o_yaw = tf_to_xyy(t)
            snap.m2o_ok = True
        except Exception:
            pass

        # 3) odom→base_link @Time(0)  (FastLIO 链: odom→camera_init→body→base_link)
        try:
            t = self._tf_buffer.lookup_transform('odom', 'base_link', Time())
            snap.o2b_x, snap.o2b_y, snap.o2b_yaw = tf_to_xyy(t)
            snap.o2b_ok = True
        except Exception:
            pass

        # 4) 回溯: map→base_link @(now - 200ms)
        try:
            bt_time = now - Duration(nanoseconds=int(TF_BACKTEST_SEC * 1e9))
            t = self._tf_buffer.lookup_transform('map', 'base_link', bt_time)
            snap.bt_x, snap.bt_y, snap.bt_yaw = tf_to_xyy(t)
            snap.bt_ok = True
        except Exception:
            pass

        # 增量 (与上一次查询)
        if self._prev_tf and self._prev_tf.m2b_ok and snap.m2b_ok:
            snap.d_m2b_xy = math.sqrt(
                (snap.m2b_x - self._prev_tf.m2b_x)**2 +
                (snap.m2b_y - self._prev_tf.m2b_y)**2) * 1000
            snap.d_m2b_yaw = abs(normalize_angle(snap.m2b_yaw - self._prev_tf.m2b_yaw))
        if self._prev_tf and self._prev_tf.m2o_ok and snap.m2o_ok:
            snap.d_m2o_xy = math.sqrt(
                (snap.m2o_x - self._prev_tf.m2o_x)**2 +
                (snap.m2o_y - self._prev_tf.m2o_y)**2) * 1000
            snap.d_m2o_yaw = abs(normalize_angle(snap.m2o_yaw - self._prev_tf.m2o_yaw))
        if self._prev_tf and self._prev_tf.o2b_ok and snap.o2b_ok:
            snap.d_o2b_xy = math.sqrt(
                (snap.o2b_x - self._prev_tf.o2b_x)**2 +
                (snap.o2b_y - self._prev_tf.o2b_y)**2) * 1000
            snap.d_o2b_yaw = abs(normalize_angle(snap.o2b_yaw - self._prev_tf.o2b_yaw))

        # 回溯差: @now vs @(-200ms)
        if snap.m2b_ok and snap.bt_ok:
            snap.bt_gap_xy = math.sqrt(
                (snap.m2b_x - snap.bt_x)**2 +
                (snap.m2b_y - snap.bt_y)**2) * 1000
            snap.bt_gap_yaw = abs(normalize_angle(snap.m2b_yaw - snap.bt_yaw))

        # 分桶统计
        ms = self._motion_state
        if self._prev_tf:
            if snap.m2b_ok and self._prev_tf.m2b_ok:
                self._tf_stats[ms]['d_m2b_xy'].append(snap.d_m2b_xy)
            if snap.m2o_ok and self._prev_tf.m2o_ok:
                self._tf_stats[ms]['d_m2o_xy'].append(snap.d_m2o_xy)
            if snap.o2b_ok and self._prev_tf.o2b_ok:
                self._tf_stats[ms]['d_o2b_xy'].append(snap.d_o2b_xy)
        if snap.m2b_ok and snap.bt_ok:
            self._tf_stats[ms]['bt_gap_xy'].append(snap.bt_gap_xy)
            self._tf_stats[ms]['bt_gap_yaw'].append(snap.bt_gap_yaw)

        self._prev_tf = snap
        self._tf_snaps.append(snap)

        # JSONL 日志
        rec = {
            'src': 'tf_query',
            'ts': snap.ts,
            'motion': snap.motion,
        }
        if snap.m2b_ok:
            rec.update(m2b_x=round(snap.m2b_x, 4), m2b_y=round(snap.m2b_y, 4),
                       m2b_yaw=round(snap.m2b_yaw, 4), d_m2b_xy=round(snap.d_m2b_xy, 1),
                       d_m2b_yaw=round(snap.d_m2b_yaw, 5))
        if snap.m2o_ok:
            rec.update(m2o_x=round(snap.m2o_x, 4), m2o_y=round(snap.m2o_y, 4),
                       m2o_yaw=round(snap.m2o_yaw, 4), d_m2o_xy=round(snap.d_m2o_xy, 1),
                       d_m2o_yaw=round(snap.d_m2o_yaw, 5))
        if snap.o2b_ok:
            rec.update(o2b_x=round(snap.o2b_x, 4), o2b_y=round(snap.o2b_y, 4),
                       o2b_yaw=round(snap.o2b_yaw, 4), d_o2b_xy=round(snap.d_o2b_xy, 1),
                       d_o2b_yaw=round(snap.d_o2b_yaw, 5))
        if snap.bt_ok:
            rec.update(bt_x=round(snap.bt_x, 4), bt_y=round(snap.bt_y, 4),
                       bt_yaw=round(snap.bt_yaw, 4),
                       bt_gap_xy=round(snap.bt_gap_xy, 1),
                       bt_gap_yaw=round(snap.bt_gap_yaw, 5))
        self._log_file.write(json.dumps(rec) + '\n')

    # ── odom 日志 ──
    def _log_odom(self, src: str, st: OdomState):
        if not st.inited:
            return
        rec = {
            'src': src,
            'ts': time.time(),
            'stamp': round(st.stamp, 4),
            'age_ms': round(st.age_ms, 1),
            'x': round(st.x, 4),
            'y': round(st.y, 4),
            'yaw': round(st.yaw, 4),
            'vx': round(st.vx, 4),
            'wz': round(st.wz, 4),
            'dx': round(st.dx, 5),
            'dy': round(st.dy, 5),
            'dyaw': round(st.dyaw, 5),
            'dt': round(st.dt, 4),
            'motion': self._motion_state,
        }
        self._log_file.write(json.dumps(rec) + '\n')

        # 分桶统计
        if st.dt > 0.001:
            ms = self._motion_state
            dxy = math.sqrt(st.dx**2 + st.dy**2) * 1000
            key = {'fastlio': 'fl', 'wheel': 'wh', 'fused': 'fu', 'hdl': 'hdl'}[src]
            self._stats[ms][key].append(dxy)

    # ── Dashboard ──
    def _dashboard(self):
        elapsed = time.time() - self._t0
        if not self.wh.inited:
            return

        m = self._motion_state
        if m == 'ROTATE':
            m_disp = '\033[91m◆ ROTATE\033[0m'
        elif m == 'TRANSLATE':
            m_disp = '\033[93m◆ TRANSLATE\033[0m'
        else:
            m_disp = '\033[92m◆ STATIC\033[0m'

        lines = [
            '\033[2J\033[H',
            f'═══ Odom+TF Compare  t={elapsed:.0f}s  {m_disp}  wz={self.wh.wz:+.3f} vx={self.wh.vx:+.3f} ═══',
            '',
            f' {"Source":<10} {"Hz":>5} {"age":>6} │ {"yaw°":>7} {"vx":>7} {"wz":>7} │ {"Δxy mm":>8} {"Δyaw°":>8}',
            f' {"─"*10} {"─"*5} {"─"*6} ┼ {"─"*7} {"─"*7} {"─"*7} ┼ {"─"*8} {"─"*8}',
        ]

        for label, st in [('fastlio', self.fl), ('wheel', self.wh),
                          ('fused', self.fu), ('hdl', self.hdl)]:
            if not st.inited:
                lines.append(f' {label:<10}  --')
                continue
            dxy = math.sqrt(st.dx**2 + st.dy**2) * 1000
            dyaw_deg = math.degrees(st.dyaw)
            yaw_deg = math.degrees(st.yaw)
            age_str = f'{st.age_ms:.0f}ms' if st.age_ms >= 0 else '--'
            lines.append(
                f' {label:<10} {st.hz:5.1f} {age_str:>6} │ '
                f'{yaw_deg:7.1f} {st.vx:+7.3f} {st.wz:+7.3f} │ '
                f'{dxy:8.1f} {dyaw_deg:+8.3f}'
            )

        # wheel vs fused 差异
        if self.wh.inited and self.fu.inited:
            wz_diff = self.fu.wz - self.wh.wz
            dyaw_diff_deg = math.degrees(abs(self.fu.dyaw) - abs(self.wh.dyaw))
            lines.append(f' wheel─fused差异 │ Δwz={wz_diff:+.4f} rad/s  Δ|dyaw|={dyaw_diff_deg:+.3f}°')

        # ── TF 链实时 ──
        tf = self._prev_tf
        if tf:
            lines.append('')
            lines.append(f' ─── TF 链查询 (@10Hz, 回溯={TF_BACKTEST_SEC*1000:.0f}ms) ───')
            lines.append(f' {"TF Link":<18} {"ok":>3} │ {"x":>8} {"y":>8} {"yaw°":>8} │ {"Δxy mm":>8} {"Δyaw°":>8}')
            lines.append(f' {"─"*18} {"─"*3} ┼ {"─"*8} {"─"*8} {"─"*8} ┼ {"─"*8} {"─"*8}')
            for lbl, ok, x, y, yaw, dxy, dyaw in [
                ('map→base_link', tf.m2b_ok, tf.m2b_x, tf.m2b_y, tf.m2b_yaw, tf.d_m2b_xy, tf.d_m2b_yaw),
                ('map→odom(HDL)', tf.m2o_ok, tf.m2o_x, tf.m2o_y, tf.m2o_yaw, tf.d_m2o_xy, tf.d_m2o_yaw),
                ('odom→blink(FL)', tf.o2b_ok, tf.o2b_x, tf.o2b_y, tf.o2b_yaw, tf.d_o2b_xy, tf.d_o2b_yaw),
            ]:
                if ok:
                    lines.append(
                        f' {lbl:<18} {"✓":>3} │ '
                        f'{x:8.3f} {y:8.3f} {math.degrees(yaw):8.1f} │ '
                        f'{dxy:8.1f} {math.degrees(dyaw):+8.3f}'
                    )
                else:
                    lines.append(f' {lbl:<18} {"✗":>3} │ {"--":>8}')
            # 回溯对比
            if tf.m2b_ok and tf.bt_ok:
                lines.append(
                    f' \033[96mbacktest gap\033[0m      '
                    f'    │ xy={tf.bt_gap_xy:6.1f}mm  yaw={math.degrees(tf.bt_gap_yaw):+.3f}°'
                )

        # FastLIO 校准跳变
        recent_cal = [e for e in self._calibration_events if time.time() - e['ts'] < 10]
        if recent_cal:
            last = recent_cal[-1]
            lines.append(f' \033[95mFastLIO校准跳变\033[0m │ xy={last["jump_xy_mm"]:.0f}mm yaw={last["jump_yaw_deg"]:.2f}° ({last["motion"]})')

        # ── 分桶统计: odom ──
        lines.append('')
        lines.append(f' ─── odom Δxy 统计 (mm) ───')
        lines.append(f' {"State":<10} │ {"fastlio":>20} │ {"wheel":>20} │ {"fused":>20} │ {"hdl":>20}')
        for ms in ['STATIC', 'TRANSLATE', 'ROTATE']:
            parts = []
            for key in ['fl', 'wh', 'fu', 'hdl']:
                vals = self._stats[ms][key]
                if len(vals) >= 2:
                    p50 = statistics.median(vals)
                    avg = statistics.mean(vals)
                    mx = max(vals)
                    parts.append(f'P50={p50:5.0f} x̄={avg:5.0f} M={mx:5.0f}')
                else:
                    parts.append(f'{"n=" + str(len(vals)):>20}')
            lines.append(f' {ms:<10} │ {parts[0]:>20} │ {parts[1]:>20} │ {parts[2]:>20} │ {parts[3]:>20}')

        # ── 分桶统计: TF ──
        lines.append(f' ─── TF Δxy 统计 (mm) ───')
        lines.append(f' {"State":<10} │ {"map→blink":>16} │ {"map→odom":>16} │ {"odom→blink":>16} │ {"bt_gap":>16} │ {"bt_yaw°":>10}')
        for ms in ['STATIC', 'TRANSLATE', 'ROTATE']:
            parts = []
            for key in ['d_m2b_xy', 'd_m2o_xy', 'd_o2b_xy', 'bt_gap_xy']:
                vals = self._tf_stats[ms][key]
                if len(vals) >= 2:
                    p50 = statistics.median(vals)
                    mx = max(vals)
                    parts.append(f'P50={p50:5.1f} M={mx:5.0f}')
                else:
                    parts.append(f'{"n=" + str(len(vals)):>16}')
            yaw_vals = self._tf_stats[ms]['bt_gap_yaw']
            if len(yaw_vals) >= 2:
                yaw_str = f'P50={math.degrees(statistics.median(yaw_vals)):.2f}'
            else:
                yaw_str = f'n={len(yaw_vals)}'
            lines.append(f' {ms:<10} │ {parts[0]:>16} │ {parts[1]:>16} │ {parts[2]:>16} │ {parts[3]:>16} │ {yaw_str:>10}')

        # 校准跳变统计
        n_cal = len(self._calibration_events)
        if n_cal > 0:
            cal_xy = [e['jump_xy_mm'] for e in self._calibration_events]
            lines.append(f'\n FastLIO校准跳变: {n_cal}次, xy P50={statistics.median(cal_xy):.0f}mm max={max(cal_xy):.0f}mm')

        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()

    def shutdown(self):
        self._log_file.flush()
        self._log_file.close()
        self.get_logger().info(f'日志已保存: {self._log_path}')

        # 最终统计
        print('\n\n═══ 最终统计 ═══')

        print('\n── odom 增量 ──')
        for ms in ['STATIC', 'TRANSLATE', 'ROTATE']:
            print(f'\n  [{ms}]')
            for key, label in [('fl', 'fastlio'), ('wh', 'wheel'), ('fu', 'fused'), ('hdl', 'hdl')]:
                vals = self._stats[ms][key]
                if len(vals) >= 2:
                    print(f'    {label:>8} Δxy(mm): n={len(vals)} P50={statistics.median(vals):.1f} '
                          f'mean={statistics.mean(vals):.1f} max={max(vals):.1f}')
            wz_vals = self._stats[ms]['wz_diff']
            if len(wz_vals) >= 2:
                print(f'    wz_diff(rad/s): n={len(wz_vals)} P50={statistics.median(wz_vals):.4f} '
                      f'mean={statistics.mean(wz_vals):.4f} max={max(wz_vals):.4f}')

        print('\n── TF 链增量 ──')
        for ms in ['STATIC', 'TRANSLATE', 'ROTATE']:
            print(f'\n  [{ms}]')
            for key, label in [('d_m2b_xy', 'map→blink'), ('d_m2o_xy', 'map→odom'),
                               ('d_o2b_xy', 'odom→blink')]:
                vals = self._tf_stats[ms][key]
                if len(vals) >= 2:
                    print(f'    {label:>12} Δxy(mm): n={len(vals)} P50={statistics.median(vals):.1f} '
                          f'mean={statistics.mean(vals):.1f} max={max(vals):.1f}')
            bt_vals = self._tf_stats[ms]['bt_gap_xy']
            if len(bt_vals) >= 2:
                print(f'    {"bt_gap":>12} xy(mm): n={len(bt_vals)} P50={statistics.median(bt_vals):.1f} '
                      f'mean={statistics.mean(bt_vals):.1f} max={max(bt_vals):.1f}')
            bt_yaw = self._tf_stats[ms]['bt_gap_yaw']
            if len(bt_yaw) >= 2:
                print(f'    {"bt_gap":>12} yaw(°): n={len(bt_yaw)} P50={math.degrees(statistics.median(bt_yaw)):.3f} '
                      f'mean={math.degrees(statistics.mean(bt_yaw)):.3f} max={math.degrees(max(bt_yaw)):.3f}')

        n_cal = len(self._calibration_events)
        if n_cal:
            cal_xy = [e['jump_xy_mm'] for e in self._calibration_events]
            per_motion = {}
            for e in self._calibration_events:
                per_motion.setdefault(e['motion'], []).append(e['jump_xy_mm'])
            print(f'\n  FastLIO校准跳变: {n_cal}次')
            for ms, vals in per_motion.items():
                print(f'    {ms}: {len(vals)}次, P50={statistics.median(vals):.0f}mm max={max(vals):.0f}mm')


def main():
    rclpy.init()
    node = OdomCompare()

    def _sig(signum, frame):
        node.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
