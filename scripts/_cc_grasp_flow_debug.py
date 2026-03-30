#!/usr/bin/env python3
"""
Piper 全链路 Debug — 多层交叉验证版
四层独立数据源, 交叉验证arm真实位姿:
  Layer 0: candump 被动监听原始CAN帧 (硬件真实值, 不干扰full launch)
  Layer 1: /piper/status  (piper_grasp_node发布, node计算值)
  Layer 2: /arm_status    (node透传SDK, 含ctrl_mode/err_code)
  Layer 3: /joint_states  (node发布关节角)

用法 (必须在ROS2环境下):
  source /opt/ros/humble/setup.bash
  source /data/workspace/MobileManipulator2/install/setup.bash
  python3 scripts/_cc_grasp_flow_debug.py
"""

import os, sys, re, struct, time, math, threading, signal, subprocess
from datetime import datetime
from collections import deque


# ─── Tee: 同时写终端和log文件 ────────────────────────────────────────────────

class Tee:
    """将所有print输出同时写到终端和log文件"""
    def __init__(self, path: str):
        self.terminal = sys.stdout
        self.log      = open(path, 'w', buffering=1, encoding='utf-8')
        # ANSI strip regex for log file
        self._ansi = re.compile(r'\x1b\[[0-9;]*m')

    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.log.write(self._ansi.sub('', msg))
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        return self.terminal.fileno()

    def close(self):
        self.log.close()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from piper_msgs.msg import PiperStatus, PiperStatusMsg

# ─── Config ───────────────────────────────────────────────────────────────────
CAN_PORT       = "can0"
READY_GC       = [317.0, 15.0, 248.0]   # gripper_center ready target (mm)
READY_TOL      = 30.0                    # mm
GRIPPER_OFFSET = 135.03                  # mm flange→GC
FACTOR         = 1000

# CAN IDs
ID_STATUS   = 0x2A1   # arm status feedback
ID_EP1      = 0x2A2   # end pose X, Y
ID_EP2      = 0x2A3   # end pose Z, RX
ID_EP3      = 0x2A4   # end pose RY, RZ
ID_JT12     = 0x2A5   # joint 1,2
ID_JT34     = 0x2A6   # joint 3,4
ID_JT56     = 0x2A7   # joint 5,6
ID_GRIP     = 0x2A8   # gripper
# Control command IDs (sent from host to arm)
ID_MCTRL1   = 0x150   # MotionCtrl_1
ID_MCTRL2   = 0x151   # MotionCtrl_2: mode, speed
ID_EP_CMD1  = 0x152   # EndPoseCtrl X, Y
ID_EP_CMD2  = 0x153   # EndPoseCtrl Z, RX
ID_EP_CMD3  = 0x154   # EndPoseCtrl RY, RZ
ID_ENABLER  = 0x471   # enable/disable

FEEDBACK_IDS = {ID_STATUS, ID_EP1, ID_EP2, ID_EP3, ID_JT12, ID_JT34, ID_JT56, ID_GRIP}
CMD_IDS      = {ID_MCTRL1, ID_MCTRL2, ID_EP_CMD1, ID_EP_CMD2, ID_EP_CMD3, ID_ENABLER}

G="\033[92m"; Y="\033[93m"; R="\033[91m"; B="\033[94m"; C="\033[96m"
M="\033[95m"; BOL="\033[1m"; GR="\033[90m"; RST="\033[0m"

def ts(t=None):
    return datetime.fromtimestamp(t or time.time()).strftime("%H:%M:%S.%f")[:-3]

def d3(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def i32be(data: bytes, offset=0) -> int:
    """Big-endian signed int32 from bytes"""
    return struct.unpack('>i', data[offset:offset+4])[0]

def flange_to_gc(f):
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation as Rot
        R = Rot.from_euler('xyz', [math.radians(a) for a in f[3:6]]).as_matrix()
        off = R @ [0, 0, GRIPPER_OFFSET]
        return [f[0]+off[0], f[1]+off[1], f[2]+off[2], f[3], f[4], f[5]]
    except:
        return list(f)


# ─── Layer 0: CAN passive monitor (candump) ────────────────────────────────────

class CANLayer:
    """
    Passively read raw CAN frames via candump subprocess.
    No SDK connection — no conflict with piper_grasp_node.
    Decodes: end_pose, joints, status, gripper + control commands.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._proc   = None
        self._thread = None
        self.running = False

        # Decoded state
        self.ep  = [0.0]*6       # flange [x,y,z,rx,ry,rz] mm/deg
        self.gc  = [0.0]*6       # gripper center
        self.jts = [0.0]*6       # joints deg
        self.grip = 0.0           # gripper mm
        self.arm_st   = -1       # arm_status byte
        self.err_code = -1       # error code
        self.dist_ready = 999.0

        # Last seen control commands
        self.last_mctrl2 = None  # (ctrl_mode, move_mode, speed)
        self.last_ep_cmd = [None,None,None]  # target pose parts
        self.last_enable = None

        # History
        self.hist    = deque(maxlen=200)
        self.cmd_log = deque(maxlen=50)
        self.raw_log = deque(maxlen=200)

        # Freshness
        self.last_ts = {id_: 0 for id_ in FEEDBACK_IDS | CMD_IDS}
        self.ep_complete = [False, False, False]   # have all 3 EP frames
        self.jt_complete = [False, False, False]   # have all 3 joint frames

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._proc:
            self._proc.terminate()

    def _loop(self):
        # candump -t a  = absolute timestamp, all frames
        cmd = ['candump', '-t', 'a', CAN_PORT]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1)
        except Exception as e:
            print(f"[CAN] candump failed: {e}")
            return

        # candump output format:
        # (1234567890.123456) can0  2A2   [8]  01 02 03 04 05 06 07 08
        pat = re.compile(
            r'\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)\s+\[\d+\]\s+((?:[0-9A-Fa-f]{2}\s*)+)')

        for line in self._proc.stdout:
            if not self.running:
                break
            m = pat.match(line.strip())
            if not m:
                continue
            try:
                t_hw   = float(m.group(1))
                can_id = int(m.group(2), 16)
                raw    = bytes.fromhex(m.group(3).replace(' ', ''))
                self._decode(t_hw, can_id, raw)
                with self._lock:
                    self.raw_log.append((t_hw, can_id, raw))
            except Exception:
                pass

    def _decode(self, t: float, can_id: int, data: bytes):
        if len(data) < 8:
            return

        with self._lock:
            self.last_ts[can_id] = t
            now = time.time()

            # ── Feedback frames ──────────────────────────────────────────────
            if can_id == ID_EP1:
                self.ep[0] = i32be(data, 0) / FACTOR   # X mm
                self.ep[1] = i32be(data, 4) / FACTOR   # Y mm
                self.ep_complete[0] = True
            elif can_id == ID_EP2:
                self.ep[2] = i32be(data, 0) / FACTOR   # Z mm
                self.ep[3] = i32be(data, 4) / FACTOR   # RX deg
                self.ep_complete[1] = True
            elif can_id == ID_EP3:
                self.ep[4] = i32be(data, 0) / FACTOR   # RY deg
                self.ep[5] = i32be(data, 4) / FACTOR   # RZ deg
                self.ep_complete[2] = True
                # Full pose received — compute GC and record
                if all(self.ep_complete):
                    self.gc = flange_to_gc(self.ep)
                    self.dist_ready = d3(self.gc[:3], READY_GC)
                    self.hist.append({'ts': now, 'gc': self.gc[:3].copy(),
                                      'dist': self.dist_ready, 'ep': self.ep.copy()})

            elif can_id == ID_JT12:
                self.jts[0] = i32be(data, 0) / FACTOR
                self.jts[1] = i32be(data, 4) / FACTOR
                self.jt_complete[0] = True
            elif can_id == ID_JT34:
                self.jts[2] = i32be(data, 0) / FACTOR
                self.jts[3] = i32be(data, 4) / FACTOR
                self.jt_complete[1] = True
            elif can_id == ID_JT56:
                self.jts[4] = i32be(data, 0) / FACTOR
                self.jts[5] = i32be(data, 4) / FACTOR
                self.jt_complete[2] = True

            elif can_id == ID_STATUS:
                # byte0=ctrl_mode, byte1=motion_status, byte2-3=err_code
                self.arm_st   = data[1]
                self.err_code = (data[2] << 8) | data[3]

            elif can_id == ID_GRIP:
                angle = i32be(data, 0)
                self.grip = angle / FACTOR

            # ── Control commands ─────────────────────────────────────────────
            elif can_id == ID_MCTRL2:
                ctrl_mode  = data[0]
                move_mode  = data[1]
                speed      = data[2]
                self.last_mctrl2 = (ctrl_mode, move_mode, speed)
                self.cmd_log.append(
                    {'ts': now, 'id': can_id,
                     'msg': f"MotionCtrl2: ctrl={ctrl_mode:#04x} mode={move_mode:#04x} spd={speed}%"})

            elif can_id == ID_EP_CMD1:
                x = i32be(data, 0) / FACTOR
                y = i32be(data, 4) / FACTOR
                self.last_ep_cmd[0] = (x, y)
                self.cmd_log.append(
                    {'ts': now, 'id': can_id, 'msg': f"EndPoseCmd XY: x={x:.2f} y={y:.2f}"})

            elif can_id == ID_EP_CMD2:
                z  = i32be(data, 0) / FACTOR
                rx = i32be(data, 4) / FACTOR
                self.last_ep_cmd[1] = (z, rx)
                self.cmd_log.append(
                    {'ts': now, 'id': can_id, 'msg': f"EndPoseCmd ZRX: z={z:.2f} rx={rx:.2f}"})

            elif can_id == ID_EP_CMD3:
                ry = i32be(data, 0) / FACTOR
                rz = i32be(data, 4) / FACTOR
                self.last_ep_cmd[2] = (ry, rz)
                self.cmd_log.append(
                    {'ts': now, 'id': can_id, 'msg': f"EndPoseCmd RYRZ: ry={ry:.2f} rz={rz:.2f}"})

            elif can_id == ID_MCTRL1:
                self.cmd_log.append(
                    {'ts': now, 'id': can_id,
                     'msg': f"MotionCtrl1: {data.hex(' ')}"})

            elif can_id == ID_ENABLER:
                joint = data[0]
                en    = data[1]
                self.last_enable = (joint, en)
                self.cmd_log.append(
                    {'ts': now, 'id': can_id,
                     'msg': f"Enable: joint={joint} en={en:#04x}"})

    def get_state(self) -> dict:
        with self._lock:
            ep_age  = time.time() - max(self.last_ts.get(ID_EP3, 0), 0.001)
            jt_age  = time.time() - max(self.last_ts.get(ID_JT56, 0), 0.001)
            st_age  = time.time() - max(self.last_ts.get(ID_STATUS, 0), 0.001)
            return {
                'ep':           self.ep.copy(),
                'gc':           self.gc.copy(),
                'jts':          self.jts.copy(),
                'grip':         self.grip,
                'arm_st':       self.arm_st,
                'err_code':     self.err_code,
                'dist_ready':   self.dist_ready,
                'ep_complete':  all(self.ep_complete),
                'jt_complete':  all(self.jt_complete),
                'ep_age':       ep_age,
                'jt_age':       jt_age,
                'st_age':       st_age,
                'last_mctrl2':  self.last_mctrl2,
                'last_ep_cmd':  self.last_ep_cmd.copy(),
                'last_enable':  self.last_enable,
            }

    def get_hist(self):
        with self._lock:
            return list(self.hist)

    def get_cmd_log(self, n=10):
        with self._lock:
            return list(self.cmd_log)[-n:]


# ─── Layer 1/2/3: ROS Monitor ────────────────────────────────────────────────

class ROSLayer(Node):
    def __init__(self):
        super().__init__('_cc_grasp_debug')
        self._lock = threading.Lock()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)

        self.last_status  = None   # PiperStatus
        self.last_arm_st  = None   # PiperStatusMsg
        self.last_js      = None
        self.last_err     = None
        self.status_ts    = 0.0
        self.arm_st_ts    = 0.0
        self.hist         = deque(maxlen=200)

        self.create_subscription(PiperStatus,    '/piper/status', self._on_st,  qos)
        self.create_subscription(PiperStatusMsg, '/arm_status',   self._on_ast, qos)
        self.create_subscription(JointState,     '/joint_states', self._on_js,  qos)
        self.create_subscription(String, '/piper/error_state',    self._on_err, qos)

    def _on_st(self, m):
        with self._lock:
            self.last_status = m
            self.status_ts   = time.time()
            if len(m.gripper_center) >= 3:
                gc   = list(m.gripper_center)
                dist = d3(gc[:3], READY_GC)
                self.hist.append({'ts': time.time(), 'gc': gc[:3], 'dist': dist,
                                  'en': m.enabled, 'err': m.error_state})

    def _on_ast(self, m):
        with self._lock:
            self.last_arm_st = m
            self.arm_st_ts   = time.time()

    def _on_js(self, m):
        with self._lock:
            self.last_js = m

    def _on_err(self, m):
        with self._lock:
            self.last_err = m.data

    def get_all(self):
        with self._lock:
            return {
                'status':    self.last_status,
                'arm_st':    self.last_arm_st,
                'js':        self.last_js,
                'err':       self.last_err,
                'status_age': time.time()-self.status_ts if self.status_ts else 9999,
                'arm_st_age': time.time()-self.arm_st_ts if self.arm_st_ts else 9999,
            }

    def get_hist(self):
        with self._lock:
            return list(self.hist)


# ─── Cross-Validation ─────────────────────────────────────────────────────────

def cross_validate(can: CANLayer, ros: ROSLayer) -> list:
    """比较CAN原始值与ROS发布值, 返回差异列表"""
    issues = []
    cs  = can.get_state()
    rd  = ros.get_all()
    rst = rd['status']

    if not rst or len(rst.gripper_center) < 3:
        return [f"{Y}ROS /piper/status 无数据{RST}"]

    if not cs['ep_complete']:
        return [f"{Y}CAN 末端位姿不完整 (ep_age={cs['ep_age']:.1f}s){RST}"]

    gc_can = cs['gc'][:3]
    gc_ros = list(rst.gripper_center)[:3]

    diff = [abs(gc_can[i]-gc_ros[i]) for i in range(3)]
    max_diff = max(diff)

    if max_diff > 5.0:
        issues.append(
            f"{R}⚠ GC不一致: CAN=({gc_can[0]:.1f},{gc_can[1]:.1f},{gc_can[2]:.1f}) "
            f"ROS=({gc_ros[0]:.1f},{gc_ros[1]:.1f},{gc_ros[2]:.1f}) "
            f"diff=({diff[0]:.1f},{diff[1]:.1f},{diff[2]:.1f})mm{RST}")
    else:
        issues.append(
            f"{G}✓ GC一致: max_diff={max_diff:.2f}mm{RST}")

    # Joint angle comparison
    jts_can = cs['jts']
    jts_ros = list(rst.joints) if rst.joints else []
    if jts_ros and len(jts_ros) >= 6:
        jdiff = [abs(jts_can[i]-jts_ros[i]) for i in range(6)]
        max_jd = max(jdiff)
        if max_jd > 1.0:
            issues.append(
                f"{R}⚠ 关节角不一致: CAN={[f'{j:.1f}' for j in jts_can]} "
                f"ROS={[f'{j:.1f}' for j in jts_ros]} "
                f"max_diff={max_jd:.2f}°{RST}")
        else:
            issues.append(f"{G}✓ 关节角一致: max_diff={max_jd:.2f}°{RST}")

    return issues


# ─── Phase 1: 多层新鲜度验证 ──────────────────────────────────────────────────

def phase1_validation(can: CANLayer, ros: ROSLayer, duration=8.0):
    print(f"\n{BOL}{M}{'═'*72}{RST}")
    print(f"{BOL}{M}  Phase 1: 四层交叉验证  ({duration:.0f}s 连续采样){RST}")
    print(f"{BOL}{M}{'═'*72}{RST}")
    print(f"  Layer0=CAN raw  Layer1=/piper/status  Layer2=/arm_status  Layer3=/joint_states\n")

    t_end = time.time() + duration
    row   = 0
    prev_can_gc = None

    print(f"  {'时间':12s}  {'[CAN]GC-X':10s}  {'[CAN]GC-Z':10s}  {'[ROS]GC-X':10s}  "
          f"{'[ROS]GC-Z':10s}  {'dist_ready':10s}  {'Δpos_CAN':9s}  {'同步?'}")
    print(f"  {'─'*90}")

    while time.time() < t_end:
        cs  = can.get_state()
        rd  = ros.get_all()
        rst = rd['status']

        # CAN data
        gc_can  = cs['gc'][:3] if cs['ep_complete'] else [0,0,0]
        dist_c  = cs['dist_ready']
        ep_age  = cs['ep_age']

        # ROS data
        gc_ros  = list(rst.gripper_center)[:3] if (rst and len(rst.gripper_center)>=3) else [0,0,0]
        dist_r  = d3(gc_ros, READY_GC) if rst else 999

        # CAN delta
        dpos_can = d3(gc_can, prev_can_gc) if prev_can_gc else 0
        prev_can_gc = gc_can.copy()

        # Agreement
        agree_d = d3(gc_can, gc_ros) if cs['ep_complete'] and rst else 999
        agree_s = f"{G}✓{RST}" if agree_d < 3 else f"{Y}△{agree_d:.0f}mm{RST}" if agree_d < 20 else f"{R}✗{agree_d:.0f}mm{RST}"

        dc = G if dist_c < READY_TOL else Y if dist_c < 100 else R
        mc = G if dpos_can > 1.0 else GR

        age_str = f"{GR}({ep_age:.1f}s ago){RST}" if ep_age > 0.5 else ""

        print(f"  [{ts()}]  {gc_can[0]:9.2f}  {gc_can[2]:9.2f}  "
              f"{gc_ros[0]:9.2f}  {gc_ros[2]:9.2f}  "
              f"{dc}{dist_c:8.1f}mm{RST}  {mc}{dpos_can:7.2f}mm{RST}  "
              f"{agree_s} {age_str}")

        row += 1
        time.sleep(0.5)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"\n{BOL}[ Phase 1 分析 ]{RST}")

    # CAN freshness
    cs = can.get_state()
    print(f"\n  {BOL}Layer 0 - CAN raw:{RST}")
    print(f"    ep_complete={cs['ep_complete']}  ep_age={cs['ep_age']:.2f}s")
    print(f"    jt_complete={cs['jt_complete']}  jt_age={cs['jt_age']:.2f}s")
    print(f"    arm_status={cs['arm_st']}  err_code={cs['err_code']:#06x}")
    ep = cs['ep']
    gc = cs['gc']
    print(f"    Flange: ({ep[0]:.2f},{ep[1]:.2f},{ep[2]:.2f})mm  "
          f"rpy=({ep[3]:.1f},{ep[4]:.1f},{ep[5]:.1f})°")
    print(f"    GC:     ({gc[0]:.2f},{gc[1]:.2f},{gc[2]:.2f})mm  "
          f"dist_ready={cs['dist_ready']:.1f}mm")
    jts = cs['jts']
    if jts:
        print(f"    Joints: {' '.join(f'J{i+1}:{jts[i]:.2f}°' for i in range(6))}")

    # ROS status
    rd  = ros.get_all()
    rst = rd['status']
    print(f"\n  {BOL}Layer 1 - /piper/status:{RST}")
    if rst:
        en_c = G if rst.enabled else R
        print(f"    enabled={en_c}{rst.enabled}{RST}  err={rst.error_state}")
        if len(rst.gripper_center) >= 3:
            gc_r = rst.gripper_center
            print(f"    GC: ({gc_r[0]:.2f},{gc_r[1]:.2f},{gc_r[2]:.2f})mm")
        if rst.joints:
            jros = list(rst.joints)
            print(f"    Joints: {' '.join(f'J{i+1}:{jros[i]:.2f}°' for i in range(len(jros)))}")
    else:
        print(f"    {R}无数据{RST}")

    ast = rd['arm_st']
    print(f"\n  {BOL}Layer 2 - /arm_status:{RST}")
    if ast:
        print(f"    ctrl_mode={ast.ctrl_mode}  motion_status={ast.motion_status}"
              f"  err_code={ast.err_code}")
        en_bits = [ast.communication_status_joint_1, ast.communication_status_joint_2,
                   ast.communication_status_joint_3, ast.communication_status_joint_4,
                   ast.communication_status_joint_5, ast.communication_status_joint_6]
        en_str = '  '.join(f"{G if e else R}J{i+1}{'✓' if e else '✗'}{RST}"
                           for i, e in enumerate(en_bits))
        print(f"    关节使能: {en_str}")
    else:
        print(f"    {R}无数据{RST}")

    # Cross validation
    issues = cross_validate(can, ros)
    print(f"\n  {BOL}四层交叉验证:{RST}")
    for issue in issues:
        print(f"    {issue}")

    # CAN history analysis
    ch = can.get_hist()
    if len(ch) >= 5:
        dists = [h['dist'] for h in ch]
        gcs   = [h['gc']   for h in ch]
        deltas = [d3(gcs[i], gcs[i-1]) for i in range(1, len(gcs))]
        total  = sum(deltas)
        print(f"\n  {BOL}CAN位置历史:{RST}  帧数={len(ch)}")
        print(f"    dist_ready: {dists[0]:.1f}→{dists[-1]:.1f}mm")
        print(f"    总移动={total:.2f}mm  "
              f"{''+R+'Arm完全静止!'+RST if total<2.0 else G+'Arm有移动'+RST}")


# ─── Phase 2: go_ready + 多层追踪 ────────────────────────────────────────────

def phase2_trace(can: CANLayer, ros: ROSLayer):
    print(f"\n{BOL}{C}{'═'*72}{RST}")
    print(f"{BOL}{C}  Phase 2: go_ready 多层时间链路追踪{RST}")
    print(f"{BOL}{C}{'═'*72}{RST}\n")

    call_done   = threading.Event()
    call_result = [None]
    t_call      = [time.time()]
    trace_can   = []
    trace_ros   = []

    def _tracker():
        prev_gc_can = can.get_state()['gc'][:3]
        prev_gc_ros = [0,0,0]
        st = ros.get_all()['status']
        if st and len(st.gripper_center) >= 3:
            prev_gc_ros = list(st.gripper_center)[:3]

        print(f"  {'时间':12s} {'Δt':5s} {'[CAN]X':7s} {'[CAN]Z':7s} {'dist':7s} "
              f"{'Δcan':6s} {'[ROS]X':7s} {'[ROS]Z':7s} {'Δros':6s} {'enabled':7s}")
        print(f"  {'─'*80}")

        while not call_done.wait(timeout=0.1):
            cs  = can.get_state()
            rd  = ros.get_all()
            rst = rd['status']

            gc_can = cs['gc'][:3]
            gc_ros = list(rst.gripper_center)[:3] if (rst and len(rst.gripper_center)>=3) else prev_gc_ros

            dpos_can = d3(gc_can, prev_gc_can)
            dpos_ros = d3(gc_ros, prev_gc_ros)
            dt       = time.time() - t_call[0]
            dist     = cs['dist_ready']
            dc       = G if dist < READY_TOL else Y if dist < 100 else R
            mc       = G if dpos_can > 1.0 else GR
            mr       = G if dpos_ros > 1.0 else GR
            en_c     = G if (rst and rst.enabled) else R
            en_s     = str(rst.enabled if rst else '?')[:5]

            print(f"  [{ts()}] {dt:4.1f}s  {gc_can[0]:6.1f}  {gc_can[2]:6.1f}  "
                  f"{dc}{dist:5.1f}{RST}  {mc}{dpos_can:4.1f}{RST}  "
                  f"{gc_ros[0]:6.1f}  {gc_ros[2]:6.1f}  {mr}{dpos_ros:4.1f}{RST}  "
                  f"{en_c}{en_s}{RST}")

            trace_can.append({'ts': time.time(), 'gc': gc_can.copy(), 'dpos': dpos_can, 'dist': dist})
            trace_ros.append({'ts': time.time(), 'gc': gc_ros.copy(), 'dpos': dpos_ros})
            prev_gc_can = gc_can.copy()
            prev_gc_ros = gc_ros.copy()

    # CAN command pre-log
    print(f"  {BOL}开始追踪CAN命令帧...{RST}")
    cmd_before = can.get_cmd_log(5)
    print(f"  调用前最近CAN命令:")
    for cmd in cmd_before:
        print(f"    [{ts(cmd['ts'])}] {GR}{cmd['msg']}{RST}")

    tracker = threading.Thread(target=_tracker, daemon=True)
    tracker.start()

    # Call go_ready via ros2 service call subprocess
    print(f"\n  {BOL}调用 /piper/go_ready...{RST}")
    t_call[0] = time.time()

    def _call():
        cmd = ['ros2', 'service', 'call', '/piper/go_ready',
               'piper_msgs/srv/GoReady', '{speed: 30, open_gripper: true}']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            call_result[0] = {'out': r.stdout.strip(), 'err': r.stderr.strip(),
                              'ret': r.returncode, 'elapsed': time.time()-t_call[0]}
        except subprocess.TimeoutExpired:
            call_result[0] = {'out': '', 'err': 'TIMEOUT 20s', 'ret': -1,
                              'elapsed': 20.0}
        except Exception as e:
            call_result[0] = {'out': '', 'err': str(e), 'ret': -1,
                              'elapsed': time.time()-t_call[0]}
        finally:
            call_done.set()

    threading.Thread(target=_call, daemon=True).start()
    call_done.wait(timeout=25)
    tracker.join(timeout=0.5)

    # Results
    res = call_result[0] or {}
    print(f"\n  {'─'*80}")
    print(f"  {BOL}go_ready返回:{RST}  耗时={res.get('elapsed',0):.2f}s  ret={res.get('ret','?')}")
    print(f"  stdout: {res.get('out','')[:300]}")
    if res.get('err'):
        print(f"  stderr: {R}{res['err'][:200]}{RST}")

    # CAN commands that arrived
    print(f"\n  {BOL}go_ready期间CAN命令帧:{RST}")
    cmd_after = can.get_cmd_log(20)
    # Show only commands after t_call
    t0 = t_call[0]
    new_cmds = [c for c in cmd_after if c['ts'] >= t0 - 0.5]
    if new_cmds:
        for cmd in new_cmds:
            print(f"    [{ts(cmd['ts'])}] {C}{cmd['msg']}{RST}")
    else:
        print(f"    {R}!! 期间没有检测到任何控制命令 — arm命令根本没发出!{RST}")

    # Movement analysis
    print(f"\n  {BOL}运动分析:{RST}")
    if trace_can:
        total_can = sum(t['dpos'] for t in trace_can)
        total_ros = sum(t['dpos'] for t in trace_ros) if trace_ros else 0
        dists = [t['dist'] for t in trace_can]
        print(f"  CAN总移动={total_can:.2f}mm  ROS总移动={total_ros:.2f}mm")
        print(f"  dist_ready: {dists[0]:.1f}mm → {dists[-1]:.1f}mm")
        if total_can < 3.0:
            print(f"  {R}! Arm完全没动 (CAN层确认){RST}")
            # Diagnose
            cs = can.get_state()
            if cs['arm_st'] != 0:
                print(f"  {R}→ arm_status={cs['arm_st']} (非零=错误/急停){RST}")
            if not new_cmds:
                print(f"  {R}→ go_ready期间没有CAN命令帧 — node可能被锁/busy/rejected{RST}")
        else:
            print(f"  {G}Arm有运动!{RST}")

    return trace_can


# ─── Phase 3: 持续监控 ────────────────────────────────────────────────────────

def phase3_monitor(can: CANLayer, ros: ROSLayer):
    stop  = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    frame = 0

    while not stop.is_set():
        os.system('clear')
        cs = can.get_state()
        rd = ros.get_all()
        rst = rd['status']
        ast = rd['arm_st']

        print(f"{BOL}{B}{'═'*72}{RST}")
        print(f"{BOL}{B}  Piper 全链路监控  {ts()}  frame={frame}{RST}")
        print(f"{BOL}{B}{'═'*72}{RST}\n")

        # Layer 0: CAN
        ep  = cs['ep']; gc = cs['gc']
        jts = cs['jts']; dist = cs['dist_ready']
        dc  = G if dist < READY_TOL else Y if dist < 100 else R
        st_c = G if cs['arm_st']==0 else R
        age_c = GR if cs['ep_age']>0.5 else ''
        print(f"{BOL}[ Layer 0 — CAN raw ]{RST}  {age_c}ep_age={cs['ep_age']:.2f}s{RST}")
        print(f"  Flange: ({ep[0]:.2f},{ep[1]:.2f},{ep[2]:.2f})mm  "
              f"rpy=({ep[3]:.1f},{ep[4]:.1f},{ep[5]:.1f})°")
        print(f"  GC:     ({gc[0]:.2f},{gc[1]:.2f},{gc[2]:.2f})mm  "
              f"dist={dc}{dist:.1f}mm{RST}")
        print(f"  Joints: {' '.join(f'J{i+1}:{jts[i]:.1f}°' for i in range(6))}")
        print(f"  arm_status={st_c}{cs['arm_st']}{RST}  err_code={cs['err_code']:#06x}  "
              f"gripper={cs['grip']:.1f}mm")

        # Last command
        m2 = cs['last_mctrl2']
        ec = cs['last_ep_cmd']
        if m2:
            print(f"  LastCmd: MotionCtrl2 ctrl={m2[0]:#04x} mode={m2[1]:#04x} spd={m2[2]}%")
        if ec[0]:
            x,y = ec[0]; z_,rx_ = ec[1] if ec[1] else (0,0); ry_,rz_ = ec[2] if ec[2] else (0,0)
            print(f"  LastCmd: EndPose=({x:.1f},{y:.1f},{z_:.1f})mm rpy=({rx_:.1f},{ry_:.1f},{rz_:.1f})°")

        # Layer 1: ROS status
        print(f"\n{BOL}[ Layer 1 — /piper/status ]{RST}  age={rd['status_age']:.1f}s")
        if rst:
            en_c = G if rst.enabled else R
            print(f"  enabled={en_c}{rst.enabled}{RST}  error={rst.error_state}")
            if len(rst.gripper_center) >= 3:
                gc_r = rst.gripper_center
                dr   = d3(list(gc_r[:3]), READY_GC)
                dc_r = G if dr < READY_TOL else Y if dr < 100 else R
                print(f"  GC: ({gc_r[0]:.2f},{gc_r[1]:.2f},{gc_r[2]:.2f})mm  "
                      f"dist={dc_r}{dr:.1f}mm{RST}")
            if rst.joints:
                jros = list(rst.joints)
                print(f"  Joints: {' '.join(f'J{i+1}:{jros[i]:.1f}°' for i in range(len(jros)))}")
        else:
            print(f"  {R}无数据{RST}")

        # Layer 2: arm_status
        print(f"\n{BOL}[ Layer 2 — /arm_status ]{RST}  age={rd['arm_st_age']:.1f}s")
        if ast:
            print(f"  ctrl_mode={ast.ctrl_mode}  motion_status={ast.motion_status}  "
                  f"err_code={ast.err_code}")
            en_bits = [ast.communication_status_joint_1, ast.communication_status_joint_2,
                       ast.communication_status_joint_3, ast.communication_status_joint_4,
                       ast.communication_status_joint_5, ast.communication_status_joint_6]
            en_str = '  '.join(f"{G if e else R}J{i+1}{'✓' if e else '✗'}{RST}"
                               for i, e in enumerate(en_bits))
            print(f"  使能: {en_str}")
        else:
            print(f"  {GR}无数据{RST}")

        # Cross-validation
        issues = cross_validate(can, ros)
        print(f"\n{BOL}[ 交叉验证 ]{RST}")
        for iss in issues:
            print(f"  {iss}")

        # CAN cmd log
        cmds = can.get_cmd_log(4)
        if cmds:
            print(f"\n{BOL}[ 最近CAN控制命令 ]{RST}")
            for cmd in cmds:
                print(f"  [{ts(cmd['ts'])}] {C}{cmd['msg']}{RST}")

        # Trend
        ch = can.get_hist()
        if len(ch) >= 3:
            dists  = [h['dist'] for h in ch]
            gcs_h  = [h['gc']   for h in ch]
            deltas = [d3(gcs_h[i], gcs_h[i-1]) for i in range(1, len(gcs_h))]
            n_still= sum(1 for d in deltas[-10:] if d < 0.3)
            mc     = R if n_still >= 8 else G
            print(f"\n{BOL}[ 位置趋势 ]{RST}  "
                  f"dist: {dists[0]:.1f}→{dists[-1]:.1f}mm  "
                  f"{mc}静止={n_still}/min(10,{len(deltas)}){RST}")

        frame += 1
        time.sleep(0.3)

    print(f"\n{Y}监控结束.{RST}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log_path = f"/tmp/grasp_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"{BOL}Piper 全链路多层交叉验证 Debug{RST}")
    print(f"Log: {log_path}")
    print(f"CAN={CAN_PORT}  ready={READY_GC}mm  tol={READY_TOL}mm\n")

    # Start CAN passive monitor
    can = CANLayer()
    can.start()
    print(f"[CAN] 被动监听 {CAN_PORT}...")

    # Start ROS2 monitor
    rclpy.init()
    ros = ROSLayer()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros), daemon=True)
    ros_thread.start()

    # Wait for data
    print(f"  等待数据 (3s)...")
    time.sleep(3.0)

    cs = can.get_state()
    rd = ros.get_all()
    can_ok = cs['ep_complete']
    ros_ok = rd['status'] is not None

    print(f"  CAN数据: {''+G+'OK'+RST if can_ok else Y+'不完整 ep_age='+str(round(cs['ep_age'],1))+'s'+RST}")
    print(f"  ROS数据: {''+G+'OK'+RST if ros_ok else R+'无 /piper/status'+RST}")

    if not can_ok and not ros_ok:
        print(f"\n{R}两层都无数据! 请确认 full launch 运行中.{RST}")
    else:
        phase1_validation(can, ros, duration=6.0)

        input(f"\n{BOL}{Y}  按Enter开始Phase 2 (go_ready追踪)...{RST}")
        phase2_trace(can, ros)

        input(f"\n{BOL}{Y}  按Enter进入Phase 3 (持续监控, Ctrl+C退出)...{RST}")

    try:
        phase3_monitor(can, ros)
    except KeyboardInterrupt:
        pass
    finally:
        can.stop()
        ros.destroy_node()
        rclpy.shutdown()
        print(f"\n{Y}结束. Log已保存: {log_path}{RST}")
        sys.stdout = tee.terminal
        tee.close()


if __name__ == '__main__':
    main()
