#!/usr/bin/env python3
"""
验证arm预定义状态是否在关节限制范围内
"""

import math

# 关节限制（从joint_limits.yaml）
JOINT_LIMITS = {
    'joint1': (-2.687, 2.687),   # ±154°
    'joint2': (0, 3.403),        # 0°~195°
    'joint3': (-3.054, 0),       # -175°~0°
    'joint4': (-1.780, 1.780),   # ±102°
    'joint5': (-1.309, 1.309),   # ±75°
    'joint6': (-2.094, 2.094),   # ±120°
    'joint7': (0, 0.05),         # 0~50mm
    'joint8': (-0.05, 0),        # -50mm~0
}

# 预定义状态
ARM_STATES = {
    'zero': {
        'joint1': 0,
        'joint2': 0.7854,    # 45°
        'joint3': -1.5708,   # -90°
        'joint4': 0,
        'joint5': 0,
        'joint6': 0,
        'joint7': 0,
        'joint8': 0,
    },
    'ready': {
        'joint1': 0,
        'joint2': 1.5708,    # 90°
        'joint3': -2.0944,   # -120°
        'joint4': 0,
        'joint5': 0.5236,    # 30°
        'joint6': 0,
        'joint7': 0,
        'joint8': 0,
    },
    'home': {
        'joint1': 0,
        'joint2': 0.7854,    # 45°
        'joint3': -1.5708,   # -90°
        'joint4': 0,
        'joint5': 0.7854,    # 45°
        'joint6': 0,
    }
}

def rad_to_deg(rad):
    """弧度转角度"""
    return rad * 180 / math.pi

def verify_state(state_name, state_values):
    """验证状态是否在关节限制内"""
    print(f"\n检查 {state_name} 状态:")
    print("-" * 60)
    
    all_valid = True
    for joint, value in state_values.items():
        if joint in JOINT_LIMITS:
            min_val, max_val = JOINT_LIMITS[joint]
            is_valid = min_val <= value <= max_val
            
            if joint.startswith('joint') and int(joint[-1]) <= 6:
                # 角度关节
                print(f"  {joint}: {value:.4f} rad ({rad_to_deg(value):.1f}°) "
                      f"[范围: {min_val:.3f}~{max_val:.3f} rad "
                      f"({rad_to_deg(min_val):.0f}°~{rad_to_deg(max_val):.0f}°)] "
                      f"{'✓' if is_valid else '✗ 超出范围!'}")
            else:
                # 位移关节
                print(f"  {joint}: {value:.3f} m ({value*1000:.0f}mm) "
                      f"[范围: {min_val:.3f}~{max_val:.3f} m] "
                      f"{'✓' if is_valid else '✗ 超出范围!'}")
            
            if not is_valid:
                all_valid = False
    
    return all_valid

def main():
    print("=== 验证Arm预定义状态 ===")
    
    for state_name, state_values in ARM_STATES.items():
        is_valid = verify_state(state_name, state_values)
        if is_valid:
            print(f"\n{state_name} 状态: ✓ 所有关节都在限制范围内")
        else:
            print(f"\n{state_name} 状态: ✗ 有关节超出限制范围!")
    
    print("\n=== 建议 ===")
    print("- zero状态：机械臂处于折叠状态，适合存储和运输")
    print("- ready状态：机械臂处于准备工作状态，便于开始各种任务")
    print("- home状态：Piper组的默认位置，不包含夹爪关节")

if __name__ == '__main__':
    main()