#!/usr/bin/env python3
"""
_cc_piper_diag.py — Piper 机械臂多层诊断工具

分层检查: CAN总线 → SDK → PiperAPI → ROS服务 → 应用层

使用方法:
  # 先确保 piper_grasp_node 已启动
  # 然后在另一个终端执行:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_piper_diag.py

  # 或者不依赖 piper_grasp_node，直接测试 SDK:
  python3 scripts/_cc_piper_diag.py --sdk-only

测试流程 (按序执行):
  Phase 1: CAN 总线健康检查
  Phase 2: SDK 直连测试 (--sdk-only 模式)
  Phase 3: ROS 服务测试 (通过 piper_grasp_node)
  Phase 4: 运动命令跟踪 (go_ready + approach)

诊断结果会保存到 /tmp/piper_diag_<timestamp>.log
"""

import os
import sys
import time
import struct
import subprocess
import threading
import argparse
import json
import re
from datetime import datetime
from collections import defaultdict

# ROS2 (optional, only for Phase 3)
try:
    import rclpy
    from rclpy.node import Node
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False

# ============================================================
# Configuration
# ============================================================
CAN_PORT = "can0"
DIAG_LOG = f"/tmp/piper_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# CAN IDs
CAN_IDS = {
    # Control (Host → Arm)
    0x150: "MotionCtrl_1",
    0x151: "MotionCtrl_2",
    0x152: "EndPose_XY",
    0x153: "EndPose_ZRX",
    0x154: "EndPose_RYRZ",
    0x155: "Joint_12",
    0x156: "Joint_34",
    0x157: "Joint_56",
    0x159: "Gripper",
    0x471: "EnableArm",
    # Feedback (Arm → Host)
    0x2A1: "ArmStatus",
    0x2A2: "EndPose_FB_XY",
    0x2A3: "EndPose_FB_ZRX",
    0x2A4: "EndPose_FB_RYRZ",
    0x2A5: "Joint_FB_12",
    0x2A6: "Joint_FB_34",
    0x2A7: "Joint_FB_56",
    0x2A8: "Gripper_FB",
}

# Status codes for 0x2A1
ARM_STATUS_CODES = {
    0x00: "NORMAL",
    0x01: "EMERGENCY_STOP",
    0x02: "NO_SOLUTION",
    0x03: "COLLISION",
    0x04: "TARGET_LIMIT",
    0x05: "JOINT_LIMIT",
    0x06: "OVER_SPEED",
    0x07: "COLLISION_PROT",
}

MOTION_STATUS_CODES = {
    0x00: "ARRIVED",
    0x01: "NOT_ARRIVED",
}

MOVE_MODE_CODES = {
    0x00: "MoveP",
    0x01: "MoveJ",
    0x02: "MoveL",
    0x03: "MoveC",
}


def _safe_int(val):
    """Convert SDK enum/value to int safely (piper_sdk returns enum, not int)"""
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return val.value  # enum
        except AttributeError:
            return 0


def _fmt_status(val):
    """Format arm status value for display"""
    iv = _safe_int(val)
    name = ARM_STATUS_CODES.get(iv, f"0x{iv:02X}")
    return iv, name


class DiagLog:
    """Dual output: terminal + log file"""
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w')
        self.f.write(f"# Piper Arm Diagnostic Log - {datetime.now().isoformat()}\n\n")

    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def section(self, title):
        sep = "=" * 60
        self.log("")
        self.log(sep)
        self.log(f"  {title}")
        self.log(sep)

    def close(self):
        self.f.close()


log = None  # Global logger


# ============================================================
# Phase 1: CAN Bus Health Check
# ============================================================
def phase1_can_health():
    log.section("Phase 1: CAN 总线健康检查")

    # 1.1 Check CAN interface exists
    log.log(f"检查 {CAN_PORT} 接口...")
    result = subprocess.run(["ip", "link", "show", CAN_PORT],
                            capture_output=True, text=True)
    if result.returncode != 0:
        log.log(f"FAIL: {CAN_PORT} 接口不存在!", "ERROR")
        return False

    # Parse state
    for line in result.stdout.strip().split('\n'):
        log.log(f"  {line.strip()}")

    if "UP" not in result.stdout:
        log.log(f"FAIL: {CAN_PORT} 未启动 (state != UP)", "ERROR")
        return False

    log.log(f"OK: {CAN_PORT} 接口正常", "PASS")

    # 1.2 Check CAN traffic (are feedback frames coming?)
    log.log(f"\n捕获 CAN 帧 (2秒)...")
    try:
        proc = subprocess.Popen(
            ["candump", CAN_PORT, "-t", "A"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(2.0)
        proc.terminate()
        stdout, _ = proc.communicate(timeout=3)

        lines = stdout.strip().split('\n') if stdout.strip() else []
        log.log(f"  捕获到 {len(lines)} 帧")

        if len(lines) == 0:
            log.log("FAIL: 2秒内没有收到任何CAN帧! 机械臂可能断电或CAN线断开", "ERROR")
            return False

        # Count by CAN ID
        # candump format: "(timestamp)  can0  2A1   [8]  00 01 ..."
        id_counts = defaultdict(int)
        can_id_re = re.compile(r'\s+([0-9A-Fa-f]{1,3})\s+\[')
        for line in lines:
            m = can_id_re.search(line)
            if m:
                try:
                    can_id = int(m.group(1), 16)
                    id_counts[can_id] += 1
                except ValueError:
                    pass

        # Display frame counts
        log.log("\n  CAN 帧统计:")
        for can_id in sorted(id_counts.keys()):
            name = CAN_IDS.get(can_id, "Unknown")
            direction = "←ARM" if can_id >= 0x200 else "→ARM"
            log.log(f"    0x{can_id:03X} ({name:20s}) {direction}: {id_counts[can_id]:4d} 帧")

        # Check for essential feedback
        essential = [0x2A1, 0x2A2, 0x2A3, 0x2A4]
        missing = [f"0x{eid:03X}" for eid in essential if id_counts[eid] == 0]
        if missing:
            log.log(f"WARN: 缺少关键反馈帧: {', '.join(missing)}", "WARN")
        else:
            log.log("OK: 所有关键反馈帧正常", "PASS")

        # Parse latest arm status (0x2A1)
        for line in reversed(lines):
            if " 2A1 " in line.upper():
                log.log(f"\n  最新 ArmStatus (0x2A1): {line.strip()}")
                _parse_arm_status_line(line)
                break

    except FileNotFoundError:
        log.log("WARN: candump 不可用，跳过CAN流量检查", "WARN")
    except Exception as e:
        log.log(f"WARN: CAN检查异常: {e}", "WARN")

    return True


def _parse_arm_status_line(line):
    """Parse 0x2A1 frame data"""
    try:
        # Extract hex data bytes from candump output
        parts = line.strip().split()
        # Find data bytes (after the frame ID and length)
        data_start = None
        for i, p in enumerate(parts):
            if p.upper() in ('2A1', '02A1'):
                # Next part might be [8] or just data bytes
                data_start = i + 1
                break

        if data_start is None:
            return

        # Collect hex bytes
        hex_bytes = []
        for p in parts[data_start:]:
            p = p.strip('[]')
            if len(p) == 2:
                try:
                    hex_bytes.append(int(p, 16))
                except ValueError:
                    continue
            elif len(p) == 1:
                try:
                    int(p, 16)
                    hex_bytes.append(int(p, 16))
                except ValueError:
                    continue

        if len(hex_bytes) >= 7:
            ctrl_mode = hex_bytes[0]
            arm_status = hex_bytes[1]
            mode_feed = hex_bytes[2]
            teach_status = hex_bytes[3]
            motion_status = hex_bytes[4]
            traj_num = hex_bytes[5]
            err_code = (hex_bytes[6] << 8) | hex_bytes[7] if len(hex_bytes) >= 8 else hex_bytes[6]

            status_name = ARM_STATUS_CODES.get(arm_status, f"UNKNOWN(0x{arm_status:02X})")
            motion_name = MOTION_STATUS_CODES.get(motion_status, f"UNKNOWN(0x{motion_status:02X})")
            mode_name = MOVE_MODE_CODES.get(mode_feed, f"0x{mode_feed:02X}")

            log.log(f"    ctrl_mode=0x{ctrl_mode:02X} arm_status={status_name} "
                     f"move_mode={mode_name} motion_status={motion_name} "
                     f"err_code=0x{err_code:04X}")
    except Exception as e:
        log.log(f"    (解析失败: {e})")


# ============================================================
# Phase 2: SDK Direct Test (bypass piper_grasp_node)
# ============================================================
def phase2_sdk_test():
    log.section("Phase 2: SDK 直连测试")
    log.log("注意: 此测试直接操作机械臂，不经过 piper_grasp_node!")
    log.log("如果 piper_grasp_node 正在运行，会有 CAN 总线冲突!")

    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError:
        log.log("FAIL: piper_sdk 未安装", "ERROR")
        return False

    piper = None
    try:
        # 2.1 Connect
        log.log("\n[2.1] 连接...")
        piper = C_PiperInterface_V2(CAN_PORT)
        piper.ConnectPort()
        time.sleep(0.5)
        log.log("OK: ConnectPort 成功", "PASS")

        # 2.2 Read initial state
        log.log("\n[2.2] 读取初始状态...")
        _sdk_read_state(piper)

        # 2.3 Enable
        log.log("\n[2.3] 使能...")
        for attempt in range(10):
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            time.sleep(0.5)

            msg = piper.GetArmLowSpdInfoMsgs()
            enable_list = [
                msg.motor_1.foc_status.driver_enable_status,
                msg.motor_2.foc_status.driver_enable_status,
                msg.motor_3.foc_status.driver_enable_status,
                msg.motor_4.foc_status.driver_enable_status,
                msg.motor_5.foc_status.driver_enable_status,
                msg.motor_6.foc_status.driver_enable_status,
            ]
            log.log(f"  尝试 {attempt+1}: enable={enable_list}")
            if all(enable_list):
                log.log("OK: 使能成功", "PASS")
                break
        else:
            log.log("FAIL: 使能超时!", "ERROR")
            return False

        # 2.4 Read position after enable
        log.log("\n[2.4] 使能后位置...")
        pos = _sdk_read_state(piper)

        # 2.5 Test motion command - go zero first
        log.log("\n[2.5] 测试运动: go_zero (Joint模式)...")
        piper.MotionCtrl_2(0x01, 0x01, 50, 0)  # Joint mode
        piper.JointCtrl(0, 0, 0, 0, 0, 0)
        log.log("  已发送 go_zero 命令")

        # Monitor position for 3 seconds
        start = time.time()
        positions = []
        while time.time() - start < 3.0:
            msg = piper.GetArmEndPoseMsgs()
            p = [msg.end_pose.X_axis/1000, msg.end_pose.Y_axis/1000,
                 msg.end_pose.Z_axis/1000]
            status = piper.GetArmStatus()
            _, motion_stat = _fmt_status(status.arm_status.arm_status)
            positions.append((time.time()-start, p, motion_stat))
            time.sleep(0.2)

        for t, p, m in positions[::3]:  # Print every 3rd
            log.log(f"  t={t:.1f}s pos=[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]mm arm_status={m}")

        # Check if position changed
        if len(positions) >= 2:
            p0 = positions[0][1]
            pn = positions[-1][1]
            delta = sum(abs(pn[i]-p0[i]) for i in range(3))
            if delta > 5:
                log.log(f"OK: 机械臂移动了 {delta:.1f}mm", "PASS")
            else:
                log.log(f"WARN: 机械臂移动量只有 {delta:.1f}mm (可能已在零位)", "WARN")

        time.sleep(1.0)

        # 2.6 Test MoveJ to ready position (THE critical test)
        log.log("\n[2.6] 测试运动: MoveJ 到 ready 位置...")
        log.log("  这是关键测试! 等同于 go_ready 的底层命令")

        # Target: ready position in flange coordinates
        # gripper_center = (317, 15, 248, 180, 30, 180)
        # offset = R @ [0,0,135.03] with R=Rz(180)*Ry(30)*Rx(180)
        # flange = gc - offset ≈ (249.5, 15, 364.9, 180, 30, 180)
        # In SDK units (×1000): (249500, 15000, 364900, 180000, 30000, 180000)
        target = [249500, 15000, 364900, 180000, 30000, 180000]

        log.log(f"  目标 (flange, SDK units): {target}")
        log.log(f"  目标 (flange, mm/deg): [{target[0]/1000:.1f}, {target[1]/1000:.1f}, {target[2]/1000:.1f}, "
                f"{target[3]/1000:.1f}, {target[4]/1000:.1f}, {target[5]/1000:.1f}]")

        # Send command sequence (same as set_position)
        log.log("  发送: MotionCtrl_1(0x00, 0x00, 0x00)")
        piper.MotionCtrl_1(0x00, 0x00, 0x00)

        log.log("  发送: MotionCtrl_2(0x01, 0x00, 30, 0x00)  # CAN ctrl, MoveJ, 30%")
        piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)

        log.log(f"  发送: EndPoseCtrl({target})")
        piper.EndPoseCtrl(*target)

        # Monitor for 8 seconds
        log.log("\n  监控运动 (8秒)...")
        start = time.time()
        positions = []
        moved = False
        while time.time() - start < 8.0:
            msg = piper.GetArmEndPoseMsgs()
            p = [msg.end_pose.X_axis/1000, msg.end_pose.Y_axis/1000,
                 msg.end_pose.Z_axis/1000,
                 msg.end_pose.RX_axis/1000, msg.end_pose.RY_axis/1000,
                 msg.end_pose.RZ_axis/1000]
            status = piper.GetArmStatus()
            arm_st = _safe_int(status.arm_status.arm_status)
            err_code = _safe_int(status.arm_status.err_code)
            ctrl_mode = _safe_int(status.arm_status.ctrl_mode)

            t = time.time() - start
            positions.append((t, p, arm_st, err_code, ctrl_mode))

            # Resend command every 0.5s (matching _wait_position_reached)
            if t > 0.2 and int(t * 5) % 3 == 0:
                piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
                piper.EndPoseCtrl(*target)

            time.sleep(0.1)

        # Print position trace
        for i, (t, p, st, err, ctrl) in enumerate(positions):
            if i % 5 == 0 or i == len(positions)-1:
                st_name = ARM_STATUS_CODES.get(st, f"0x{st:02X}")
                log.log(f"  t={t:.1f}s pos=[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}] "
                        f"orient=[{p[3]:.1f},{p[4]:.1f},{p[5]:.1f}] "
                        f"ctrl=0x{ctrl:02X} status={st_name} err=0x{err:04X}")

        # Analyze motion
        if len(positions) >= 2:
            p0 = positions[0][1]
            pn = positions[-1][1]
            delta_pos = sum(abs(pn[i]-p0[i]) for i in range(3))
            target_mm = [t/1000 for t in target[:3]]
            final_err = sum(abs(pn[i]-target_mm[i]) for i in range(3))

            log.log(f"\n  总位移: {delta_pos:.1f}mm")
            log.log(f"  与目标距离: {final_err:.1f}mm")

            if delta_pos > 10:
                if final_err < 10:
                    log.log("OK: SDK 直连运动正常，到达目标!", "PASS")
                else:
                    log.log(f"WARN: 机械臂移动了但未到达目标 (误差={final_err:.1f}mm)", "WARN")
            else:
                log.log("FAIL: 机械臂没有移动! SDK层面就有问题!", "ERROR")
                if any(p[2] != 0 for p in positions):
                    errors = set(p[2] for p in positions if p[2] != 0)
                    log.log(f"  检测到错误状态: {[ARM_STATUS_CODES.get(e, 'unknown') for e in errors]}", "ERROR")
                else:
                    log.log("  arm_status 一直是 NORMAL, 但不动 → 可能是固件/模式问题", "ERROR")

        return True

    except Exception as e:
        log.log(f"SDK 测试异常: {e}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        return False

    finally:
        if piper:
            try:
                _sdk_safe_shutdown(piper)
            except Exception:
                pass


def _sdk_safe_shutdown(piper):
    """归零 → 失能 → 断开"""
    log.log("\n[Cleanup] 归零 + 失能...")
    try:
        # Go zero
        piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        piper.JointCtrl(0, 0, 0, 0, 0, 0)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(3.0)

        # Disable
        piper.DisableArm(7)
        piper.GripperCtrl(0, 1000, 0x02, 0)  # Gripper disable
        time.sleep(0.5)

        # Verify disabled
        msg = piper.GetArmLowSpdInfoMsgs()
        enable_list = [
            msg.motor_1.foc_status.driver_enable_status,
            msg.motor_2.foc_status.driver_enable_status,
            msg.motor_3.foc_status.driver_enable_status,
            msg.motor_4.foc_status.driver_enable_status,
            msg.motor_5.foc_status.driver_enable_status,
            msg.motor_6.foc_status.driver_enable_status,
        ]
        if not any(enable_list):
            log.log("OK: 已失能", "PASS")
        else:
            log.log(f"WARN: 失能不完全: {enable_list}", "WARN")
    except Exception as e:
        log.log(f"  归零/失能异常: {e}", "WARN")

    try:
        piper.DisconnectPort()
        log.log("OK: 已断开连接", "PASS")
    except Exception:
        pass


def _sdk_read_state(piper):
    """Read and log current SDK state"""
    try:
        msg = piper.GetArmEndPoseMsgs()
        pos = [msg.end_pose.X_axis/1000, msg.end_pose.Y_axis/1000,
               msg.end_pose.Z_axis/1000,
               msg.end_pose.RX_axis/1000, msg.end_pose.RY_axis/1000,
               msg.end_pose.RZ_axis/1000]
        log.log(f"  末端位姿: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}, "
                f"{pos[3]:.1f}, {pos[4]:.1f}, {pos[5]:.1f}] mm/deg")

        jmsg = piper.GetArmJointMsgs()
        joints = [jmsg.joint_state.joint_1/1000, jmsg.joint_state.joint_2/1000,
                  jmsg.joint_state.joint_3/1000, jmsg.joint_state.joint_4/1000,
                  jmsg.joint_state.joint_5/1000, jmsg.joint_state.joint_6/1000]
        log.log(f"  关节角度: [{', '.join(f'{j:.1f}' for j in joints)}] deg")

        status = piper.GetArmStatus()
        arm_st = _safe_int(status.arm_status.arm_status)
        err = _safe_int(status.arm_status.err_code)
        ctrl = _safe_int(status.arm_status.ctrl_mode)
        log.log(f"  状态: ctrl_mode=0x{ctrl:02X} arm_status=0x{arm_st:02X} err_code=0x{err:04X}")

        return pos
    except Exception as e:
        log.log(f"  读取失败: {e}", "ERROR")
        return None


# ============================================================
# Phase 3: ROS Service Test (via piper_grasp_node)
# ============================================================
def phase3_ros_test():
    log.section("Phase 3: ROS 服务测试 (通过 piper_grasp_node)")

    if not HAS_RCLPY:
        log.log("FAIL: rclpy 不可用", "ERROR")
        return False

    rclpy.init()
    node = None

    try:
        node = rclpy.create_node('piper_diag_node')

        # 3.1 Check piper_grasp_node is running
        log.log("\n[3.1] 检查 piper_grasp_node 是否运行...")
        from piper_msgs.srv import GetStatus, GoReady
        from piper_msgs.msg import PiperStatus

        status_client = node.create_client(GetStatus, '/piper/get_status')
        ready_client = node.create_client(GoReady, '/piper/go_ready')

        if not status_client.wait_for_service(timeout_sec=3.0):
            log.log("FAIL: /piper/get_status 服务不可用 (piper_grasp_node 未运行?)", "ERROR")
            return False
        log.log("OK: piper_grasp_node 服务可用", "PASS")

        # 3.2 Get current status
        log.log("\n[3.2] 获取当前状态...")
        req = GetStatus.Request()
        future = status_client.call_async(req)
        _spin_wait(node, future, timeout=5.0)
        if future.done():
            resp = future.result()
            log.log(f"  connected={resp.connected}, enabled={resp.enabled}")
            log.log(f"  error_state={resp.error_state}")
            if len(resp.gripper_center) >= 3:
                gc = resp.gripper_center
                log.log(f"  gripper_center=[{gc[0]:.1f}, {gc[1]:.1f}, {gc[2]:.1f}]mm")
            if len(resp.joints) >= 6:
                log.log(f"  joints=[{', '.join(f'{j:.1f}' for j in resp.joints[:6])}] deg")
            if not resp.connected:
                log.log("FAIL: 机械臂未连接!", "ERROR")
                return False
            if not resp.enabled:
                log.log("FAIL: 机械臂未使能!", "ERROR")
                return False
            log.log("OK: 状态正常", "PASS")
        else:
            log.log("FAIL: GetStatus 超时", "ERROR")
            return False

        # 3.3 Subscribe to /piper/status for monitoring
        log.log("\n[3.3] 订阅 /piper/status 实时监控...")
        status_history = []
        last_pos = [None]

        def status_cb(msg):
            gc = list(msg.gripper_center) if len(msg.gripper_center) >= 3 else [0,0,0]
            ts = time.time()

            # Only log if position changed by >1mm or first message
            if last_pos[0] is None or sum(abs(gc[i]-last_pos[0][i]) for i in range(3)) > 1.0:
                status_history.append({
                    'time': ts,
                    'gc': gc[:3],
                    'enabled': msg.enabled,
                    'error': msg.error_state,
                })
                last_pos[0] = gc[:3]

        sub = node.create_subscription(PiperStatus, '/piper/status', status_cb, 10)
        # Let it collect a few samples
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

        if status_history:
            h = status_history[-1]
            log.log(f"  最新: gc=[{h['gc'][0]:.1f},{h['gc'][1]:.1f},{h['gc'][2]:.1f}] "
                    f"enabled={h['enabled']} error={h['error']}")

        # 3.4 Test go_ready via ROS service
        log.log("\n[3.4] 测试 go_ready 服务调用...")
        log.log("  这是关键测试! 等同于面板点击 Let's Grasp 的第一步")

        if not ready_client.wait_for_service(timeout_sec=3.0):
            log.log("FAIL: /piper/go_ready 服务不可用", "ERROR")
            return False

        # Start CAN sniffer in background
        can_thread = None
        can_frames = []
        try:
            can_thread = _start_can_sniffer(can_frames, duration=15.0)
        except Exception:
            log.log("WARN: CAN嗅探器启动失败 (candump不可用)", "WARN")

        # Record start position
        start_status = status_history[-1] if status_history else None
        start_time = time.time()

        req = GoReady.Request()
        req.speed = 30
        req.open_gripper = True

        log.log(f"  发送 go_ready(speed=30, open_gripper=true)...")
        future = ready_client.call_async(req)

        # Spin while waiting, collecting status updates
        deadline = time.time() + 30.0
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        elapsed = time.time() - start_time

        if future.done():
            resp = future.result()
            log.log(f"  响应: success={resp.success}, message={resp.message}")
            log.log(f"  耗时: {elapsed:.1f}s")
            if resp.position:
                log.log(f"  最终位置: [{resp.position[0]:.1f}, {resp.position[1]:.1f}, {resp.position[2]:.1f}]")

            if resp.success:
                log.log("OK: go_ready 成功!", "PASS")
            else:
                log.log("FAIL: go_ready 失败!", "ERROR")
        else:
            log.log("FAIL: go_ready 超时 (30s)", "ERROR")

        # Collect remaining status updates
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)

        # Report position trace during go_ready
        if len(status_history) > 1:
            log.log(f"\n  位置轨迹 (共 {len(status_history)} 个采样点):")
            for h in status_history:
                t = h['time'] - start_time
                log.log(f"    t={t:+.1f}s gc=[{h['gc'][0]:.1f},{h['gc'][1]:.1f},{h['gc'][2]:.1f}] "
                        f"error={h['error']}")

        # Analyze CAN frames during go_ready
        if can_thread:
            can_thread.join(timeout=3)
            _analyze_can_frames(can_frames, start_time)

        return True

    except Exception as e:
        log.log(f"ROS 测试异常: {e}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
        return False

    finally:
        if node:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _spin_wait(node, future, timeout=5.0):
    """Spin until future done or timeout"""
    deadline = time.time() + timeout
    while not future.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def _start_can_sniffer(frames_list, duration=15.0):
    """Start CAN sniffer in background thread"""
    def _sniffer():
        try:
            proc = subprocess.Popen(
                ["candump", CAN_PORT, "-t", "A"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            deadline = time.time() + duration
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break
                frames_list.append((time.time(), line.strip()))
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass

    t = threading.Thread(target=_sniffer, daemon=True)
    t.start()
    return t


def _analyze_can_frames(frames, start_time):
    """Analyze captured CAN frames"""
    if not frames:
        return

    log.log(f"\n  CAN 帧分析 (共 {len(frames)} 帧):")

    # Count control frames (0x150-0x159)
    ctrl_frames = defaultdict(int)
    first_ctrl = None
    for ts, line in frames:
        for can_id_hex in ['150', '151', '152', '153', '154', '155', '156', '157', '159']:
            if f" {can_id_hex.upper()} " in line.upper() or f" {can_id_hex} " in line:
                can_id = int(can_id_hex, 16)
                ctrl_frames[can_id] += 1
                if first_ctrl is None:
                    first_ctrl = (ts, can_id)

    if ctrl_frames:
        log.log("  发送的控制帧:")
        for can_id in sorted(ctrl_frames.keys()):
            name = CAN_IDS.get(can_id, "Unknown")
            log.log(f"    0x{can_id:03X} ({name}): {ctrl_frames[can_id]} 帧")

        if first_ctrl:
            log.log(f"  第一个控制帧: 0x{first_ctrl[1]:03X} at t={first_ctrl[0]-start_time:+.2f}s")
    else:
        log.log("  WARNING: 没有检测到任何控制帧! 命令可能没有发出!", "WARN")


# ============================================================
# Phase 4: Motion Command Trace with CAN Sniffing
# ============================================================
def phase4_motion_trace():
    log.section("Phase 4: 运动命令跟踪 (CAN + SDK)")
    log.log("此测试需要 piper_grasp_node 未运行 (直接 SDK 操作)")

    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError:
        log.log("SKIP: piper_sdk 不可用", "WARN")
        return

    # Start CAN sniffer
    can_frames = []
    try:
        can_thread = _start_can_sniffer(can_frames, duration=20.0)
    except Exception:
        can_thread = None

    piper = None
    try:
        piper = C_PiperInterface_V2(CAN_PORT)
        piper.ConnectPort()
        time.sleep(0.5)

        # Enable
        for _ in range(10):
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            time.sleep(0.5)
            msg = piper.GetArmLowSpdInfoMsgs()
            if all([msg.motor_1.foc_status.driver_enable_status,
                    msg.motor_2.foc_status.driver_enable_status,
                    msg.motor_3.foc_status.driver_enable_status,
                    msg.motor_4.foc_status.driver_enable_status,
                    msg.motor_5.foc_status.driver_enable_status,
                    msg.motor_6.foc_status.driver_enable_status]):
                break

        log.log("使能完成")

        # Go zero first
        log.log("\n[4.1] go_zero...")
        piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        piper.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(3.0)

        pos_msg = piper.GetArmEndPoseMsgs()
        log.log(f"  零位: [{pos_msg.end_pose.X_axis/1000:.1f}, {pos_msg.end_pose.Y_axis/1000:.1f}, "
                f"{pos_msg.end_pose.Z_axis/1000:.1f}]mm")

        # 4.2 Test: Send MoveJ commands with/without MotionCtrl_1(0x00)
        log.log("\n[4.2] 测试A: 带 MotionCtrl_1(0x00) 的 MoveJ (与 set_position 相同)")
        target = [249500, 15000, 364900, 180000, 30000, 180000]

        t0 = time.time()
        piper.MotionCtrl_1(0x00, 0x00, 0x00)
        piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        piper.EndPoseCtrl(*target)

        moved = _monitor_motion(piper, target, timeout=8.0, label="测试A")

        if not moved:
            log.log("\n[4.3] 测试B: 不带 MotionCtrl_1 的 MoveJ (跳过 standby 指令)")
            # Go back to zero first
            piper.MotionCtrl_2(0x01, 0x01, 50, 0)
            piper.JointCtrl(0, 0, 0, 0, 0, 0)
            time.sleep(3.0)

            t0 = time.time()
            # Skip MotionCtrl_1!
            piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
            piper.EndPoseCtrl(*target)

            moved = _monitor_motion(piper, target, timeout=8.0, label="测试B")

            if moved:
                log.log("KEY FINDING: 不发 MotionCtrl_1(0x00) 才能正常运动!", "ERROR")
                log.log("根因: MotionCtrl_1(0x00, 0x00, 0x00) 可能被固件解读为 standby/停止", "ERROR")

    except Exception as e:
        log.log(f"异常: {e}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
    finally:
        if piper:
            try:
                _sdk_safe_shutdown(piper)
            except Exception:
                pass

    if can_thread:
        can_thread.join(timeout=3)
        if can_frames:
            # Count relevant frames
            ctrl_count = sum(1 for _, line in can_frames
                            if any(f" {hex(cid)[2:].upper()} " in line.upper()
                                  for cid in range(0x150, 0x15A)))
            fb_count = sum(1 for _, line in can_frames
                          if any(f" {hex(cid)[2:].upper()} " in line.upper()
                                for cid in range(0x2A1, 0x2A9)))
            log.log(f"\n  CAN 统计: 控制帧={ctrl_count}, 反馈帧={fb_count}")


def _monitor_motion(piper, target, timeout=8.0, label=""):
    """Monitor arm motion and return True if it moved"""
    start = time.time()
    positions = []
    resend_interval = 0.5
    last_resend = 0

    while time.time() - start < timeout:
        msg = piper.GetArmEndPoseMsgs()
        p = [msg.end_pose.X_axis, msg.end_pose.Y_axis, msg.end_pose.Z_axis,
             msg.end_pose.RX_axis, msg.end_pose.RY_axis, msg.end_pose.RZ_axis]

        status = piper.GetArmStatus()
        arm_st = _safe_int(status.arm_status.arm_status)
        err = _safe_int(status.arm_status.err_code)

        t = time.time() - start
        positions.append((t, p, arm_st, err))

        # Check if reached
        pos_ok = all(abs(p[i] - target[i]) < 5000 for i in range(3))
        if pos_ok:
            log.log(f"  {label}: 到达目标! (t={t:.1f}s)", "PASS")
            return True

        # Resend
        if t - last_resend > resend_interval:
            piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
            piper.EndPoseCtrl(*target)
            last_resend = t

        time.sleep(0.1)

    # Print trace
    for i, (t, p, st, err) in enumerate(positions):
        if i % 5 == 0 or i == len(positions)-1:
            st_name = ARM_STATUS_CODES.get(st, f"0x{st:02X}")
            log.log(f"  {label} t={t:.1f}s pos=[{p[0]/1000:.1f},{p[1]/1000:.1f},{p[2]/1000:.1f}] "
                    f"status={st_name} err=0x{err:04X}")

    # Check total movement
    if len(positions) >= 2:
        p0 = positions[0][1]
        pn = positions[-1][1]
        delta = sum(abs(pn[i]-p0[i]) for i in range(3)) / 1000
        if delta > 5:
            log.log(f"  {label}: 移动了 {delta:.1f}mm 但未到达目标", "WARN")
            return True
        else:
            log.log(f"  {label}: 未移动 ({delta:.1f}mm)", "ERROR")
            return False
    return False


# ============================================================
# Summary
# ============================================================
def print_summary():
    log.section("诊断总结")
    log.log(f"完整日志已保存到: {DIAG_LOG}")
    log.log("")
    log.log("请将日志文件内容发给开发者进行分析。")
    log.log("重点关注:")
    log.log("  1. Phase 2 [2.6]: SDK 直连 MoveJ 是否成功?")
    log.log("  2. Phase 3 [3.4]: ROS go_ready 是否成功?")
    log.log("  3. Phase 4 [4.2]/[4.3]: MotionCtrl_1(0x00) 是否是问题根因?")
    log.log("")
    log.log("如果 Phase 2 成功但 Phase 3 失败:")
    log.log("  → 问题在 piper_grasp_node.py (并发/锁/定时器干扰)")
    log.log("如果 Phase 2 也失败:")
    log.log("  → 问题在 SDK 或固件层面 (不是 piper_grasp_node 的问题)")
    log.log("如果 Phase 4 测试A失败但测试B成功:")
    log.log("  → 根因是 MotionCtrl_1(0x00,0,0) 导致固件进入 standby")


# ============================================================
# Main
# ============================================================
def main():
    global log, CAN_PORT

    parser = argparse.ArgumentParser(description='Piper Arm Multi-Layer Diagnostics')
    parser.add_argument('--sdk-only', action='store_true',
                        help='Only run SDK direct test (Phase 2+4), skip ROS')
    parser.add_argument('--ros-only', action='store_true',
                        help='Only run ROS test (Phase 3), skip SDK')
    parser.add_argument('--can-port', default='can0',
                        help='CAN port (default: can0)')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                        help='Run only specific phase')
    args = parser.parse_args()

    CAN_PORT = args.can_port
    log = DiagLog(DIAG_LOG)

    log.log(f"Piper Arm 多层诊断工具")
    log.log(f"CAN 端口: {CAN_PORT}")
    log.log(f"日志文件: {DIAG_LOG}")
    log.log(f"时间: {datetime.now().isoformat()}")

    try:
        if args.phase:
            # Run specific phase
            if args.phase == 1:
                phase1_can_health()
            elif args.phase == 2:
                phase2_sdk_test()
            elif args.phase == 3:
                phase3_ros_test()
            elif args.phase == 4:
                phase4_motion_trace()
        elif args.sdk_only:
            phase1_can_health()
            phase2_sdk_test()
            phase4_motion_trace()
        elif args.ros_only:
            phase1_can_health()
            phase3_ros_test()
        else:
            # Full diagnostic
            log.log("\n完整诊断流程:")
            log.log("  Phase 1: CAN 总线 → Phase 2: SDK → Phase 3: ROS → Phase 4: 运动跟踪")
            log.log("  注意: Phase 2 和 4 需要 piper_grasp_node 不在运行!")
            log.log("")

            ok = phase1_can_health()
            if not ok:
                log.log("CAN 总线异常，终止诊断", "ERROR")
                return

            # Ask user which mode
            print("\n" + "=" * 50)
            print("选择诊断模式:")
            print("  1 = SDK 直连测试 (Phase 2+4, 需要先停止 piper_grasp_node)")
            print("  2 = ROS 服务测试 (Phase 3, 需要 piper_grasp_node 运行中)")
            print("  3 = 全部 (先 SDK, 再 ROS)")
            print("=" * 50)

            choice = input("请选择 [1/2/3]: ").strip()

            if choice == '1':
                phase2_sdk_test()
                phase4_motion_trace()
            elif choice == '2':
                phase3_ros_test()
            elif choice == '3':
                log.log("\n>>> 请确保 piper_grasp_node 已停止! <<<")
                input("按 Enter 继续 SDK 测试...")
                phase2_sdk_test()
                phase4_motion_trace()

                log.log("\n>>> 请启动 piper_grasp_node <<<")
                input("按 Enter 继续 ROS 测试...")
                phase3_ros_test()
            else:
                log.log("无效选择", "ERROR")
                return

        print_summary()

    except KeyboardInterrupt:
        log.log("\n用户中断", "WARN")
    finally:
        log.close()

    print(f"\n日志已保存: {DIAG_LOG}")


if __name__ == '__main__':
    main()
