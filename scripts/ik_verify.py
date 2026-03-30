#!/usr/bin/env python3
"""
IK 路径诊断与执行工具

功能:
  1. 调用 /piper/observe 自动获取感知结果
  2. 按与 piper_grasp_node 完全相同的逻辑计算抓取路径
  3. 诊断当前机械臂 IK 构型是否能求解该路径
  4. 可选: 执行路径（go_zero 复位构型 → 完整抓取）

用法:
    # 仅诊断（先执行 observe，再分析 IK 构型，不动臂）
    ros2 run piper_grasp _cc_ik_verify --diagnose

    # 诊断 + 测试 step1（臂会移动！）
    ros2 run piper_grasp _cc_ik_verify --test-step1

    # 执行完整路径（go_zero 复位 + 抓取）
    ros2 run piper_grasp _cc_ik_verify --execute

    # 不调用 observe，手动指定目标（纯诊断，不需要 ROS）
    python3 _cc_ik_verify.py --no-ros --diagnose \
        --target-gc-x 406 --target-gc-y 148 \
        --approach-gc-z -123 --grasp-gc-z -143
"""

import sys
import os
import math
import time
import argparse
import threading

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

# ── 路径: piper_grasp scripts ──────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPER_SCRIPTS = os.path.join(_SCRIPT_DIR, '../src/piper_grasp/scripts')
sys.path.insert(0, _PIPER_SCRIPTS)

from piper_api_v2 import PiperAPI


# ════════════════════════════════════════════════════════════════════════════════
# 关节角插值表 (来自 piper_api_v2.py Z_KEYPOINT_CONFIGS)
# 适用: GC_X≈317-400mm, GC_Y≈0, pitch=30° 标准路径
# ════════════════════════════════════════════════════════════════════════════════
Z_KEYPOINTS_DEG = {
    # GC_Z(mm): [J1, J2, J3, J4, J5, J6] degrees
    200:  [0.0,  79.4, -37.1, 0.0,  52.7, 0.0],
    150:  [0.0,  80.7, -27.2, 0.0,  41.4, 0.0],
    100:  [0.0,  84.1, -19.2, 0.0,  30.0, 0.0],
    50:   [0.0,  89.4, -13.0, 0.0,  18.6, 0.0],
    0:    [0.0,  96.7,  -8.8, 0.0,   7.1, 0.0],
    -50:  [0.0, 105.5,  -6.7, 0.0,  -3.8, 0.0],
    -100: [0.0, 115.5,  -6.8, 0.0, -13.7, 0.0],
    -150: [0.0, 126.1,  -9.1, 0.0, -22.0, 0.0],
    -200: [0.0, 136.7, -13.5, 0.0, -28.2, 0.0],
    -250: [0.0, 147.1, -19.8, 0.0, -32.3, 0.0],
}

GRIPPER_OFFSET_MM = 135.03  # 法兰→夹爪中心 (Z轴)
JOINT_LIMITS_DEG = [(-154,154), (-20,155), (-145,145), (-154,154), (-90,90), (-154,154)]


# ════════════════════════════════════════════════════════════════════════════════
# 几何工具
# ════════════════════════════════════════════════════════════════════════════════

def interp_joints_at_z(gc_z: float) -> list:
    """在关键帧表中线性插值，返回 GC_Z 处预期的 [J1..J6] 关节角（度）。"""
    keys = sorted(Z_KEYPOINTS_DEG.keys())
    if gc_z <= keys[0]:
        return list(Z_KEYPOINTS_DEG[keys[0]])
    if gc_z >= keys[-1]:
        return list(Z_KEYPOINTS_DEG[keys[-1]])
    for i in range(len(keys) - 1):
        if keys[i] <= gc_z <= keys[i + 1]:
            t = (gc_z - keys[i]) / (keys[i + 1] - keys[i])
            a = Z_KEYPOINTS_DEG[keys[i]]
            b = Z_KEYPOINTS_DEG[keys[i + 1]]
            return [a[k] + t * (b[k] - a[k]) for k in range(6)]
    return list(Z_KEYPOINTS_DEG[keys[0]])


def gc_to_flange(gc_pos: list) -> list:
    """夹爪中心坐标 → 法兰坐标"""
    angles_rad = [math.radians(a) for a in gc_pos[3:6]]
    R_mat = ScipyR.from_euler('xyz', angles_rad).as_matrix()
    offset = R_mat @ np.array([0, 0, GRIPPER_OFFSET_MM])
    return [gc_pos[0] - offset[0], gc_pos[1] - offset[1], gc_pos[2] - offset[2],
            gc_pos[3], gc_pos[4], gc_pos[5]]


def classify_j2(j2: float) -> str:
    if j2 < 75:   return "ELBOW-UP (高位)"
    if j2 < 100:  return "ELBOW-UP (标准)"
    if j2 < 130:  return "ELBOW-DOWN (过渡)"
    if j2 < 148:  return "ELBOW-DOWN (深度)"
    return "‼ 接近极限(≥148°)"


def check_limits(joints: list) -> list:
    warns = []
    for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
        margin = min(joints[i] - lo, hi - joints[i])
        if margin < 8:
            warns.append(f"J{i+1}={joints[i]:.1f}° 距极限仅{margin:.1f}° [{lo},{hi}]")
    return warns


# ════════════════════════════════════════════════════════════════════════════════
# ROS2 服务调用 (go_ready + observe 在同一个 rclpy 会话里完成)
# ════════════════════════════════════════════════════════════════════════════════

def _call_service_sync(node, cli, req, timeout_sec: float):
    """在当前线程里同步等待一个 async service call 完成。"""
    import rclpy
    fut = cli.call_async(req)
    done = threading.Event()

    def _spin():
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout_sec)
        done.set()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    done.wait(timeout=timeout_sec + 2)
    return fut


def go_ready_then_observe(prompt: str = '',
                          go_ready_timeout: float = 20.0,
                          observe_timeout: float = 10.0) -> dict:
    """
    单次 rclpy 会话：先 /piper/go_ready，再 /piper/observe。

    observe 前必须确认臂在 ready 位，否则 offset = point3d_base - EEF_GC 会用
    错误的 EEF 位置，导致 offset_z 不符合 working_area 范围而被误拒绝。

    Returns:
        observe result dict，失败返回 None
    """
    try:
        import rclpy
        from rclpy.node import Node
        from piper_msgs.srv import GoReady, Observe

        rclpy.init()
        node = Node('_ik_verify_ros_client')

        # ── 1. go_ready ──────────────────────────────────────────────────────
        cli_ready = node.create_client(GoReady, '/piper/go_ready')
        if not cli_ready.wait_for_service(timeout_sec=5.0):
            print("[ROS] ⚠ /piper/go_ready 不可用，跳过 go_ready")
        else:
            print("[ROS] go_ready ...")
            req_ready = GoReady.Request()
            req_ready.speed = 30
            req_ready.open_gripper = True
            fut = _call_service_sync(node, cli_ready, req_ready, go_ready_timeout)
            if fut.done() and fut.result() is not None:
                r = fut.result()
                if r.success:
                    pos = list(r.position)
                    print(f"[ROS] ✅ ready: GC=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]")
                else:
                    print(f"[ROS] ⚠ go_ready 失败: {r.message}，继续 observe...")
            else:
                print("[ROS] ⚠ go_ready 超时，继续 observe...")

        # ── 2. observe ───────────────────────────────────────────────────────
        cli_obs = node.create_client(Observe, '/piper/observe')
        if not cli_obs.wait_for_service(timeout_sec=5.0):
            print("[ROS] ⚠ /piper/observe 不可用")
            node.destroy_node()
            rclpy.shutdown()
            return None

        print(f"[ROS] observe (prompt='{prompt}') ...")
        req_obs = Observe.Request()
        req_obs.prompt = prompt
        req_obs.enable_cdm = True
        fut = _call_service_sync(node, cli_obs, req_obs, observe_timeout)

        node.destroy_node()
        rclpy.shutdown()

        if fut.done() and fut.result() is not None:
            resp = fut.result()
            if resp.success:
                return {
                    'category':      resp.category,
                    'score':         resp.score,
                    'point3d_base':  list(resp.point3d_base),
                    'offset':        list(resp.offset),
                    'angle_base':    resp.angle_base,
                    'gripper_width': resp.gripper_width,
                }
            else:
                print(f"[ROS] ✗ observe 失败: {resp.error_message}")
                return None
        else:
            print("[ROS] ✗ observe 超时")
            return None

    except ImportError:
        print("[ROS] ⚠ rclpy 不可用")
        return None
    except Exception as e:
        print(f"[ROS] ✗ 异常: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# 从 observe 结果计算抓取路径 (与 piper_grasp_node pick_action 逻辑完全一致)
# ════════════════════════════════════════════════════════════════════════════════

def compute_grasp_targets(obs: dict,
                           current_gc: list,
                           config: dict = None) -> dict:
    """
    根据 observe 结果计算抓取路径各关键点。
    复现 piper_grasp_node.py pick_action 中的目标计算逻辑。

    Returns:
        dict with keys: target_x, target_y, target_z (grasp),
                        approach_z, step1_z, approach_pitch, target_yaw,
                        gripper_open, category, score
    """
    cfg = config or {}
    grasp_cfg = cfg.get('grasp', {})

    target_x = obs['point3d_base'][0]
    target_y = obs['point3d_base'][1]
    target_z = obs['point3d_base'][2]

    # grasp_depth (category-specific)
    category_depths = grasp_cfg.get('category_depths', {'bottle': 30, 'pen': 15})
    default_depth   = grasp_cfg.get('grasp_depth_mm', 20)
    grasp_depth     = category_depths.get(obs['category'], default_depth)
    target_z       -= grasp_depth

    # min_z clamp
    min_z   = grasp_cfg.get('min_z', -245)
    target_z = max(target_z, min_z)

    # approach_z = target_z + 100mm
    approach_z = target_z + 100

    # intermediate step1 height
    step1_z                = grasp_cfg.get('approach_intermediate_z', 150)
    approach_pitch         = grasp_cfg.get('approach_pitch', 10)
    approach_intermediate_z = step1_z
    use_two_step           = approach_z < approach_intermediate_z

    # target yaw: exactly matches piper_grasp_node._calculate_target_yaw
    angle  = obs['angle_base']
    cur_yaw = current_gc[5] if current_gc else 180.0
    angle_cam = max(0.0, min(180.0, angle))   # clamp to [0,180]
    target_yaw = cur_yaw - angle_cam           # SUBTRACT (not add!)
    if target_yaw < 90:
        target_yaw += 180
    target_yaw = max(120.0, min(240.0, target_yaw))  # clamp [120,240]

    # gripper width
    w = obs['gripper_width']
    gripper_max = cfg.get('gripper', {}).get('max_mm', 90.0)
    gripper_open = min(max(w * 1.2, 20.0), gripper_max)  # add 20% margin

    return {
        'target_x':      target_x,
        'target_y':      target_y,
        'target_z':      target_z,       # grasp GC_Z
        'approach_z':    approach_z,     # approach GC_Z (target_z + 100)
        'step1_z':       step1_z,        # 150mm intermediate
        'use_two_step':  use_two_step,
        'approach_pitch': approach_pitch,
        'target_yaw':    target_yaw,
        'gripper_open':  gripper_open,
        'category':      obs['category'],
        'score':         obs['score'],
        'angle_base':    angle,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 诊断打印
# ════════════════════════════════════════════════════════════════════════════════

def print_observe_result(obs: dict):
    print(f"\n{'═'*60}")
    print(f"  Observe 结果")
    print(f"{'═'*60}")
    print(f"  类别: {obs['category']}  置信度: {obs['score']:.2f}")
    p = obs['point3d_base']
    print(f"  物体 GC (base frame): x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f} mm")
    o = obs['offset']
    print(f"  offset (相对EEF):     dx={o[0]:.1f} dy={o[1]:.1f} dz={o[2]:.1f} mm")
    print(f"  抓取角度: {obs['angle_base']:.1f}°   夹爪宽度: {obs['gripper_width']:.1f} mm")


def print_grasp_plan(plan: dict):
    print(f"\n{'═'*60}")
    print(f"  抓取路径计划")
    print(f"{'═'*60}")
    print(f"  类别: {plan['category']}  score={plan['score']:.2f}")
    print(f"  目标 GC: x={plan['target_x']:.1f} y={plan['target_y']:.1f}")
    print(f"  step1_z  = {plan['step1_z']:.0f} mm  (approach_intermediate_z)")
    print(f"  approach = {plan['approach_z']:.1f} mm  (grasp_z + 100)")
    print(f"  grasp_z  = {plan['target_z']:.1f} mm")
    print(f"  pitch    = {plan['approach_pitch']}°   yaw = {plan['target_yaw']:.1f}°")
    print(f"  策略: {'两步法 (step1→descent)' if plan['use_two_step'] else '单步 MoveJ'}")


def print_ik_analysis(joints_now: list, plan: dict):
    print(f"\n{'═'*60}")
    print(f"  IK 构型分析")
    print(f"{'═'*60}")

    cur_j2 = joints_now[1]
    print(f"\n  当前关节: [{' '.join(f'{v:6.1f}' for v in joints_now)}]°")
    print(f"  当前 J2 = {cur_j2:.1f}°  → {classify_j2(cur_j2)}")
    warns = check_limits(joints_now)
    for w in warns:
        print(f"  ⚠ {w}")

    print()
    checkpoints = [
        ("READY  (248mm)",             248.0),
        ("STEP1  (intermediate)",       plan['step1_z']),
        ("APPROACH",                    plan['approach_z']),
        ("GRASP",                       plan['target_z']),
    ]
    for label, gz in checkpoints:
        j = interp_joints_at_z(gz)
        flange_z = gc_to_flange([plan['target_x'], plan['target_y'], gz,
                                  180, plan['approach_pitch'], plan['target_yaw']])[2]
        warns_pt = check_limits(j)
        warn_str = "  ⚠ " + " | ".join(warns_pt) if warns_pt else ""
        print(f"  {label:<20} GC_Z={gz:7.1f}  法兰Z={flange_z:7.1f}  "
              f"期望J2={j[1]:6.1f}°  {classify_j2(j[1])}{warn_str}")

    # 诊断结论
    step1_j2 = interp_joints_at_z(plan['step1_z'])[1]
    delta = cur_j2 - step1_j2
    print(f"\n  Step1 期望 J2 = {step1_j2:.1f}°,  当前 J2 = {cur_j2:.1f}°,  差值 = {delta:+.1f}°")

    if abs(delta) < 15:
        print("  ✅ 构型正常: 固件 IK 应能从当前构型求解 step1")
        return False
    elif delta > 25:
        print("  ❌ 构型异常: 当前 J2 过大 (elbow-down)，固件 IK 可能静默拒绝 step1")
        print("     → 建议: --execute 模式（go_zero 复位 → 重新执行）")
        return True
    else:
        print("  ⚠ 构型边界: 建议用 --test-step1 验证固件能否求解")
        return False


# ════════════════════════════════════════════════════════════════════════════════
# 执行模块
# ════════════════════════════════════════════════════════════════════════════════

def test_step1(arm: PiperAPI, plan: dict) -> bool:
    """测试 EndPoseCtrl 到 step1 是否能触发臂移动。"""
    print(f"\n{'═'*60}")
    print("  EndPoseCtrl step1 移动测试")
    print(f"{'═'*60}")

    j_before = arm.get_joint_degrees()
    _, pos_before = arm.get_position(return_gripper_center=True)
    print(f"  当前 GC: [{' '.join(f'{v:.1f}' for v in pos_before[:3])}]  J2={j_before[1]:.1f}°")
    print(f"  发送 step1 → GC({plan['target_x']:.0f},{plan['target_y']:.0f},{plan['step1_z']:.0f}) ...")

    ok = arm.set_position(
        x=plan['target_x'], y=plan['target_y'], z=plan['step1_z'],
        pitch=plan['approach_pitch'],
        # !! 不设置 yaw，与 piper_grasp_node 一致
        speed=20, wait=True, use_gripper_center=True,
        timeout=13.0, tolerance=5000
    )

    j_after = arm.get_joint_degrees()
    _, pos_after = arm.get_position(return_gripper_center=True)
    moved = any(abs(j_after[i] - j_before[i]) > 2.0 for i in range(6))
    gc_dist = math.sqrt(sum((pos_after[i]-pos_before[i])**2 for i in range(3)))

    print(f"  结果: {'✅ 成功' if ok else '❌ 失败'}")
    print(f"  臂移动: {'✅ 是' if moved else '❌ 否 (固件静默拒绝)'} (GC 位移={gc_dist:.1f}mm)")
    print(f"  移动后 J2 = {j_after[1]:.1f}°")

    if not moved:
        print("  → 固件 IK 在当前构型下无解，使用 --execute 修复。")
    return ok and moved


def safe_retreat(arm: PiperAPI, label: str = ""):
    """
    安全退出序列: 分段抬升 → go_ready → (go_zero 在 finally 里)
    从深处直接 go_zero 会经过雷达区域，必须先到 ready 位。
    """
    ready_cfg = {'x': 317, 'y': 15, 'z': 248, 'roll': 180, 'pitch': 30, 'yaw': 180}
    _, gc = arm.get_position(return_gripper_center=True)
    cur_z = gc[2]
    print(f"  [{label}] 当前 GC_Z={cur_z:.0f}mm → 分段抬升到安全高度...")

    # 分段抬升到 step1 高度 (150mm)，再 go_ready
    safe_z = 150.0
    if cur_z < safe_z:
        ok = arm.move_z_segmented(target_z=safe_z, max_step=60, speed=20,
                                   use_gripper_center=True, lock_xy=True)
        if not ok:
            print(f"  [{label}] ⚠ 分段抬升失败，尝试直接 go_ready...")

    print(f"  [{label}] go_ready...")
    arm.go_ready(ready_pos=ready_cfg, open_gripper=False, speed=30)
    _, gc = arm.get_position(return_gripper_center=True)
    print(f"  [{label}] 回到 GC=[{' '.join(f'{v:.1f}' for v in gc[:3])}]")


class JointLogger:
    """后台线程：每 0.2s 采样关节角，执行完后打印汇总。"""
    def __init__(self, arm: PiperAPI, interval: float = 0.2):
        self._arm = arm
        self._interval = interval
        self._records: list = []   # [(t, label, j1..j6)]
        self._label = "idle"
        self._running = False
        self._thread = None

    def set_label(self, label: str):
        self._label = label

    def start(self):
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while self._running:
            try:
                j = self._arm.get_joint_degrees()
                t = time.time() - self._t0
                self._records.append((t, self._label, j))
            except Exception:
                pass
            time.sleep(self._interval)

    def print_summary(self):
        if not self._records:
            return
        print(f"\n{'─'*70}")
        print(f"  关节角记录 ({len(self._records)} 采样点)")
        print(f"{'─'*70}")
        print(f"  {'时间':>6}  {'阶段':<18}  {'J1':>6} {'J2':>6} {'J3':>6} {'J4':>6} {'J5':>6} {'J6':>6}")
        prev_label = None
        for t, label, j in self._records:
            sep = "  ─" if label != prev_label else "   "
            print(f"{sep} {t:5.1f}s  {label:<18}  "
                  f"{j[0]:6.1f} {j[1]:6.1f} {j[2]:6.1f} {j[3]:6.1f} {j[4]:6.1f} {j[5]:6.1f}")
            prev_label = label
        # J4 变化统计
        j4s = [r[2][3] for r in self._records]
        print(f"\n  J4 范围: {min(j4s):.1f}° ~ {max(j4s):.1f}°  "
              f"(峰峰值 {max(j4s)-min(j4s):.1f}°)")
        print(f"{'─'*70}")


def execute_path(arm: PiperAPI, plan: dict) -> bool:
    """完整执行: go_zero → go_ready → step1 → approach → grasp → lift → go_ready。"""
    print(f"\n{'═'*60}")
    print("  完整路径执行 (go_zero → pick)")
    print(f"{'═'*60}")
    print(f"  目标: GC({plan['target_x']:.0f},{plan['target_y']:.0f})")
    print(f"  approach={plan['approach_z']:.0f}mm  grasp={plan['target_z']:.0f}mm")
    print(f"\n  ⚠ 机械臂将会移动! 确认物理安全后按 Enter (Ctrl+C 取消)...")
    try:
        input()
    except KeyboardInterrupt:
        print("  取消。")
        return False

    logger = JointLogger(arm, interval=0.2)
    logger.start()

    # ── 1. go_zero ──────────────────────────────────────────────────────────
    logger.set_label("go_zero")
    print("\n[1/5] go_zero: 复位关节构型...")
    arm.go_zero()
    time.sleep(1.0)
    j = arm.get_joint_degrees()
    print(f"  Zero 后关节: [{' '.join(f'{v:.1f}' for v in j)}]°  J2={j[1]:.1f}°")

    # ── 2. go_ready ─────────────────────────────────────────────────────────
    ready_cfg = {'x': 317, 'y': 15, 'z': 248, 'roll': 180, 'pitch': 30, 'yaw': 180}
    logger.set_label("go_ready")
    print("\n[2/5] go_ready...")
    ok = arm.go_ready(ready_pos=ready_cfg, open_gripper=True, speed=30)
    time.sleep(0.3)
    j = arm.get_joint_degrees()
    _, gc = arm.get_position(return_gripper_center=True)
    print(f"  Ready 后 J2={j[1]:.1f}°  GC=[{' '.join(f'{v:.1f}' for v in gc[:3])}]")
    if not ok:
        print("  ⚠ go_ready 未完全到达，继续执行（误差在容忍范围）")

    # ── 3. step1: 水平到目标XY，保持 pitch=30°（ready 高度）────────────────
    # 原因: 只改 XY，与 ready 构型 IK 种子相同 → 固件 IK 必然可解
    logger.set_label("step1")
    print(f"\n[3/7] step1 → GC({plan['target_x']:.0f},{plan['target_y']:.0f},{ready_cfg['z']:.0f}) pitch={ready_cfg['pitch']}° (水平平移)...")
    ok = arm.set_position(
        x=plan['target_x'], y=plan['target_y'], z=ready_cfg['z'],
        pitch=ready_cfg['pitch'],
        speed=20, wait=True, use_gripper_center=True,
        timeout=13.0, tolerance=5000
    )
    j = arm.get_joint_degrees()
    _, gc = arm.get_position(return_gripper_center=True)
    print(f"  Step1 后 J2={j[1]:.1f}° J4={j[3]:.1f}°  GC=[{' '.join(f'{v:.1f}' for v in gc[:3])}]")
    if not ok:
        logger.stop(); logger.print_summary()
        print("  ❌ step1 失败，safe_retreat → go_zero...")
        safe_retreat(arm, "step1 fail")
        return False

    # ── 3b. step1b: 下降到 intermediate_z=120mm 并调整 pitch=10°（合并 MoveJ）─
    # 关键: Z=120mm + pitch=10° → flange_total=434mm (GC_X=376mm) → 工作空间内
    # 不在 Z=248mm 调 pitch: flange_Z 会达 381mm → 超出工作空间 (TARGET_LIMIT)
    # 此步把 pitch 调好后，后续下降全程 J5 > 0°，避免腕部奇点导致 J4 乱转
    logger.set_label("step1b")
    print(f"\n[3b/7] step1b → GC_Z={plan['step1_z']:.0f}mm pitch={plan['approach_pitch']}° (合并下降+pitch调整)...")
    ok = arm.set_position(
        x=plan['target_x'], y=plan['target_y'], z=plan['step1_z'],
        pitch=plan['approach_pitch'],
        speed=20, wait=True, use_gripper_center=True,
        timeout=10.0, tolerance=5000
    )
    j = arm.get_joint_degrees()
    _, gc = arm.get_position(return_gripper_center=True)
    print(f"  Step1b 后 J2={j[1]:.1f}° J5={j[4]:.1f}°  GC=[{' '.join(f'{v:.1f}' for v in gc[:3])}]")
    if not ok:
        logger.stop(); logger.print_summary()
        print("  ❌ step1b 失败，safe_retreat → go_zero...")
        safe_retreat(arm, "step1b fail")
        return False

    # ── 4. approach: pitch=10° 锁定下降到 approach_z ─────────────────────────
    # pitch=10° 全程保持 → J5 保持 >0° → 无腕部奇点 → J4 不乱转
    logger.set_label("approach")
    print(f"\n[4/7] approach → GC_Z={plan['approach_z']:.0f} mm (pitch={plan['approach_pitch']}° 锁定)...")
    ok = arm.move_z_segmented(
        target_z=plan['approach_z'], max_step=60, speed=20,
        use_gripper_center=True, lock_xy=True
    )
    j = arm.get_joint_degrees()
    _, gc = arm.get_position(return_gripper_center=True)
    print(f"  Approach 后 J2={j[1]:.1f}° J4={j[3]:.1f}° J5={j[4]:.1f}°  GC=[{' '.join(f'{v:.1f}' for v in gc[:3])}]")
    if not ok:
        logger.stop(); logger.print_summary()
        print("  ❌ Approach 失败，safe_retreat → go_zero...")
        safe_retreat(arm, "approach fail")
        return False

    # ── 5. 调整 yaw，开夹爪 ──────────────────────────────────────────────────
    logger.set_label("adjust_yaw")
    print(f"\n[5/7] 调整 yaw → {plan['target_yaw']:.1f}° (开夹爪)...")
    arm.set_position(yaw=plan['target_yaw'], speed=20, wait=True, use_gripper_center=True)
    arm.set_gripper_position(plan['gripper_open'], speed=500, wait=True)

    # ── 6. grasp: 下降到抓取深度 + 夹持 ─────────────────────────────────────
    # 注: 夹爪在此之前已打开 (step5)，在臂到达 target_z 后才闭合
    # _wait_pos t=0.0s 显示的是等待开始时的位置，不是结束时的位置
    logger.set_label("grasp")
    print(f"\n[6/7] grasp → GC_Z={plan['target_z']:.0f} mm ...")
    arm.move_z_segmented(
        target_z=plan['target_z'], max_step=60, speed=20,
        use_gripper_center=True, lock_xy=True
    )
    print("  夹爪闭合（臂已到达 target_z）...")
    arm.set_gripper_position(0, speed=500, wait=True)
    time.sleep(0.5)

    logger.set_label("lift")
    safe_retreat(arm, "lift")
    logger.stop()
    logger.print_summary()
    print("\n✅ 执行完成。请检查是否成功抓取物体。")
    return True


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Piper IK 路径诊断 + 执行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--can',      default='can0', help='CAN 接口 (默认 can0)')
    p.add_argument('--prompt',   default='',     help='observe 提示词 (如 "bottle.cup")')

    # 模式
    p.add_argument('--diagnose',   action='store_true', help='诊断 IK 构型（默认）')
    p.add_argument('--test-step1', action='store_true', help='测试 step1 EndPoseCtrl（臂会动）')
    p.add_argument('--execute',    action='store_true', help='执行完整抓取路径')
    p.add_argument('--no-ros',     action='store_true', help='跳过 observe，手动指定目标')

    # 手动目标 (--no-ros 时使用)
    p.add_argument('--target-gc-x',  type=float, default=400.0, metavar='X', dest='tx')
    p.add_argument('--target-gc-y',  type=float, default=0.0,   metavar='Y', dest='ty')
    p.add_argument('--approach-gc-z',type=float, default=-100.0,metavar='Z', dest='az')
    p.add_argument('--grasp-gc-z',   type=float, default=-120.0,metavar='Z', dest='gz')

    args = p.parse_args()
    if not (args.diagnose or args.test_step1 or args.execute):
        args.diagnose = True
    return args


def main():
    args = parse_args()

    # ── go_ready + observe（同一 ROS 会话）────────────────────────────────
    obs = None
    if not args.no_ros:
        # offset = point3d_base - EEF_GC，必须在 ready 位才能得到正确的 offset
        print(f"\n[IK Verify] go_ready → observe (prompt='{args.prompt}')...")
        obs = go_ready_then_observe(prompt=args.prompt,
                                    go_ready_timeout=20.0,
                                    observe_timeout=10.0)

    if obs is None:
        # observe 失败：execute/test-step1 模式必须有真实目标，否则终止
        if (args.execute or args.test_step1) and not args.no_ros:
            print("\n[IK Verify] ❌ observe 失败，无法执行 --execute/--test-step1。")
            print("  可能原因:")
            print("    1. 物体离手太近 (offset_z > offset_zmax=-150mm)")
            print("    2. 物体不在工作区 (offset_xmin/xmax, offset_ymin/ymax)")
            print("    3. 感知置信度太低 (score < min_score=0.3)")
            print("  建议: 调整机器人/物体位置后重试，或用 --no-ros 手动指定坐标")
            return

        # --diagnose 或 --no-ros: 使用手动参数构造假目标（仅用于分析）
        print(f"\n[IK Verify] 使用手动目标: GC({args.tx:.0f},{args.ty:.0f}) "
              f"approach={args.az:.0f} grasp={args.gz:.0f} mm")
        obs = {
            'category':      'manual',
            'score':         1.0,
            'point3d_base':  [args.tx, args.ty, args.gz + 20],  # grasp_depth=20 → target_z=gz
            'offset':        [0.0, 0.0, args.gz],
            'angle_base':    0.0,
            'gripper_width': 50.0,
        }

    print_observe_result(obs)

    # ── 连接机械臂 ─────────────────────────────────────────────────────────
    arm = PiperAPI(can_name=args.can, gripper_max_mm=90.0, gripper_offset_mm=GRIPPER_OFFSET_MM)
    try:
        arm.connect()
        time.sleep(0.5)

        # 获取当前状态
        joints_now = arm.get_joint_degrees()
        _, gc_now   = arm.get_position(return_gripper_center=True)

        # ── 计算抓取路径 ───────────────────────────────────────────────────
        # config 使用默认值，如需读 yaml 可扩展
        config = {
            'grasp': {
                'grasp_depth_mm': 20,
                'category_depths': {'bottle': 30, 'pen': 15},
                'min_z': -245,
                'approach_intermediate_z': 120,  # see config: lowered from 150 to avoid workspace limit at pitch=10°
                'approach_pitch': 10,
            },
            'gripper': {'max_mm': 90.0}
        }
        # 如果手动模式，直接覆盖 target_z
        if args.no_ros:
            obs['point3d_base'] = [args.tx, args.ty, args.gz + 20]

        plan = compute_grasp_targets(obs, gc_now, config)

        print_grasp_plan(plan)

        # ── 诊断 ──────────────────────────────────────────────────────────
        mismatch = print_ik_analysis(joints_now, plan)

        # ── 测试 step1 ────────────────────────────────────────────────────
        if args.test_step1:
            test_step1(arm, plan)

        # ── 执行 ──────────────────────────────────────────────────────────
        if args.execute:
            execute_path(arm, plan)

    except KeyboardInterrupt:
        print("\n中断。")
    finally:
        print("\n[IK Verify] go_zero 后断开连接...")
        try:
            arm.go_zero()
        except Exception:
            pass
        try:
            arm.disconnect(safe=True)
        except Exception:
            pass
        print("[IK Verify] 完成。")


if __name__ == '__main__':
    main()
