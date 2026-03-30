#!/usr/bin/env python3
"""
Place 动作诊断与执行工具

功能:
  1. 从 ready 位置执行 place 动作（与 _execute_place 完全相同的逻辑）
  2. 逐步打印每一步的位置和关节状态
  3. 支持 --dry-run 仅计算位置，不移动臂

用法:
    # 诊断并执行 place（臂会移动！）
    export PATH="/usr/bin:$PATH" && python3 scripts/_cc_place_test.py

    # 仅计算位置，不移动臂
    python3 scripts/_cc_place_test.py --dry-run
"""

import sys
import os
import math
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '../src/piper_grasp/scripts'))

import yaml
from piper_api_v2 import PiperAPI

DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, '../src/piper_grasp/config/piper_grasp_node.yaml')


def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get('piper_ros', data)


def compute_place_gc(cfg):
    """Compute GC coords from flange config (matches _execute_place)."""
    place_cfg = cfg.get('positions', {}).get('place', {})
    flange_x = place_cfg.get('x', 16)
    flange_y = place_cfg.get('y', -200)
    flange_z = place_cfg.get('z', 200)
    quat = place_cfg.get('quat', [0.6626, 0.7281, -0.1011, 0.1435])

    rot = R.from_quat(quat)
    rot_matrix = rot.as_matrix()
    euler = rot.as_euler('xyz', degrees=True)
    place_roll, place_pitch, place_yaw = euler

    gripper_offset = cfg.get('gripper', {}).get('offset_mm', 135.03)
    offset_base = rot_matrix @ np.array([0, 0, gripper_offset])

    place_x = flange_x + offset_base[0]
    place_y = flange_y + offset_base[1]
    place_z = flange_z + offset_base[2]

    return {
        'flange_x': flange_x, 'flange_y': flange_y, 'flange_z': flange_z,
        'gc_x': place_x, 'gc_y': place_y, 'gc_z': place_z,
        'roll': place_roll, 'pitch': place_pitch, 'yaw': place_yaw,
        'quat': quat,
    }


def print_sep(title=""):
    w = 60
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='仅计算，不移动臂')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--speed', type=int, default=20, help='运动速度 1-100%%')
    args = parser.parse_args()

    cfg = load_config(args.config)
    place = compute_place_gc(cfg)

    ready_cfg = cfg.get('positions', {}).get('ready', {})
    ready_x    = ready_cfg.get('x', 317)
    ready_y    = ready_cfg.get('y', 15)
    ready_z    = ready_cfg.get('z', 248)
    ready_roll = ready_cfg.get('roll', 180)
    ready_pitch = ready_cfg.get('pitch', 30)
    ready_yaw  = ready_cfg.get('yaw', 180)

    grasp_cfg  = cfg.get('grasp', {})
    safe_z     = grasp_cfg.get('safe_z', 200)
    seg_size   = grasp_cfg.get('descent_segment_size', 60)
    gripper_max = cfg.get('gripper', {}).get('max_mm', 70)

    # ─────────────────────────────────────────────────────────────
    print_sep("Place 位置计算")
    print(f"  法兰 (flange): [{place['flange_x']:.1f}, {place['flange_y']:.1f}, {place['flange_z']:.1f}] mm")
    print(f"  夹爪中心 (GC): [{place['gc_x']:.1f}, {place['gc_y']:.1f}, {place['gc_z']:.1f}] mm")
    print(f"  姿态 RPY:      roll={place['roll']:.1f}° pitch={place['pitch']:.1f}° yaw={place['yaw']:.1f}°")

    # Reach check
    fx, fy, fz = place['flange_x'], place['flange_y'], place['flange_z']
    flange_reach = math.sqrt(fx**2 + fy**2 + fz**2)
    j1_deg = math.degrees(math.atan2(fy, fx))
    print(f"\n  法兰臂展: {flange_reach:.1f}mm   J1方向: {j1_deg:.1f}°")

    # Plan
    lift_target = max(ready_z, safe_z)
    delta_z = place['gc_z'] - lift_target
    print_sep("运动规划")
    print(f"  Step1: MOVE XY+姿态  ready({ready_x},{ready_y},{ready_z}) → GC({place['gc_x']:.1f},{place['gc_y']:.1f},{lift_target})")
    print(f"  Step2: DESCEND       {lift_target}mm → {place['gc_z']:.1f}mm  (delta={delta_z:.1f}mm)")
    print(f"  Step3: OPEN gripper  → {gripper_max}mm")
    print(f"  Step4: LIFT          +100mm")
    print(f"  Step5: RETURN READY  go_ready (gripper close)")

    if args.dry_run:
        print("\n  --dry-run: 不执行运动")
        return

    # ─────────────────────────────────────────────────────────────
    input(f"\n  ⚠ 机械臂将会移动! 确认物理安全后按 Enter (Ctrl+C 取消)...")

    arm = PiperAPI(can_port=cfg.get('can_port', 'can0'),
                   gripper_max_mm=gripper_max)
    arm.connect(go_zero=False)
    arm.enable_arm(go_zero=False)

    try:
        # ── 1. Go to ready ────────────────────────────────────────
        print_sep("执行 Place")
        ready_pos_dict = {
            'x': ready_x, 'y': ready_y, 'z': ready_z,
            'roll': ready_roll, 'pitch': ready_pitch, 'yaw': ready_yaw,
        }
        print(f"\n[1/5] go_ready (speed=30%)...")
        arm.go_ready(ready_pos=ready_pos_dict, open_gripper=False, speed=30)
        _, gc = arm.get_position(return_gripper_center=True)
        j = arm.get_joint_degrees()
        print(f"  GC=[{gc[0]:.1f},{gc[1]:.1f},{gc[2]:.1f}]  J=[{','.join(f'{v:.1f}' for v in j)}]")

        # ── 2. Move XY + orientation at lift_target height ────────
        print(f"\n[2/5] MOVE XY+姿态 → GC({place['gc_x']:.1f},{place['gc_y']:.1f},{lift_target:.0f}mm) "
              f"rpy=[{place['roll']:.1f},{place['pitch']:.1f},{place['yaw']:.1f}]  speed={args.speed}%...")
        ok = arm.set_position(
            x=place['gc_x'], y=place['gc_y'], z=lift_target,
            roll=place['roll'], pitch=place['pitch'], yaw=place['yaw'],
            speed=args.speed, wait=True, use_gripper_center=True)
        _, gc = arm.get_position(return_gripper_center=True)
        j = arm.get_joint_degrees()
        print(f"  ok={ok}  GC=[{gc[0]:.1f},{gc[1]:.1f},{gc[2]:.1f}]")
        print(f"  J=[{','.join(f'{v:.1f}' for v in j)}]")
        if not ok:
            print("  ❌ MOVE XY 失败!")
            return

        # ── 3. Descend to place_z ─────────────────────────────────
        if abs(place['gc_z'] - lift_target) > 5:
            print(f"\n[3/5] DESCEND → GC_Z={place['gc_z']:.1f}mm (segmented, step={seg_size}mm)...")
            ok = arm.move_z_segmented(
                target_z=place['gc_z'], max_step=seg_size,
                speed=args.speed, use_gripper_center=True)
            _, gc = arm.get_position(return_gripper_center=True)
            j = arm.get_joint_degrees()
            print(f"  ok={ok}  GC=[{gc[0]:.1f},{gc[1]:.1f},{gc[2]:.1f}]")
            print(f"  J=[{','.join(f'{v:.1f}' for v in j)}]")
            if not ok:
                print("  ❌ DESCEND 失败!")
                return
        else:
            print(f"\n[3/5] DESCEND: 跳过 (|delta_z|={abs(delta_z):.1f}mm ≤ 5mm)")

        # ── 4. Open gripper ───────────────────────────────────────
        print(f"\n[4/5] OPEN gripper → {gripper_max}mm...")
        arm.set_gripper_position(gripper_max, speed=500, wait=True)
        time.sleep(0.3)
        print(f"  夹爪已打开")

        # ── 5. Lift 100mm ─────────────────────────────────────────
        _, cur = arm.get_position(return_gripper_center=True)
        lift_z = cur[2] + 100
        print(f"\n[5/5] LIFT to {lift_z:.1f}mm → close gripper → go_ready...")
        arm.set_position(z=lift_z, wait=True, speed=args.speed, use_gripper_center=True)
        arm.set_gripper_position(0, speed=500, wait=True)
        arm.go_ready(ready_pos=ready_pos_dict, open_gripper=False, speed=30)
        _, gc = arm.get_position(return_gripper_center=True)
        print(f"  最终 GC=[{gc[0]:.1f},{gc[1]:.1f},{gc[2]:.1f}]")

        print("\n  ✅ Place 测试完成!")

    except Exception as e:
        print(f"\n  ❌ 异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        arm.disconnect(safe=True)


if __name__ == '__main__':
    main()
