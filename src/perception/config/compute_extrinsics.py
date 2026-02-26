#!/usr/bin/env python3
"""
外参计算脚本

基于两个基础外参计算所有派生外参：
1. rslidar_to_top_camera_optical_frame.yaml - 标定结果
2. base_link_to_arm_base_link - URDF 值

坐标系关系 (来自 URDF):
  base_link
    └── box_link (0, 0, 0)
         └── arm_base (0.118, 0, 0.1)
    └── lidar_link (0.27254, 0.00035292, 0.0315)
         └── rslidar (0, 0, 0.0635)

注意: arm_base 在代码中也称为 arm_base_link
"""

import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from datetime import datetime


def load_tf_extrinsics(yaml_path):
    """加载 TF 约定的外参，返回 R, t"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    trans = data['transform']['translation']
    t = np.array([trans['x'], trans['y'], trans['z']])

    rot = data['transform']['rotation']
    quat = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
    R = Rotation.from_quat(quat).as_matrix()

    return R, t


def tf_to_matrix(R, t):
    """TF R,t 转为 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_tf(T):
    """4x4 矩阵转 TF R, t, quat"""
    R = T[:3, :3]
    t = T[:3, 3]
    quat = Rotation.from_matrix(R).as_quat()  # xyzw
    return R, t, quat


def save_extrinsics(path, frame_id, child_frame_id, t, quat, source):
    """保存外参为 YAML"""
    data = {
        'header': {
            'calibration_date': datetime.now().strftime('%Y-%m-%d'),
            'source': source,
            'frame_id': frame_id,
            'child_frame_id': child_frame_id,
        },
        'transform': {
            'translation': {
                'x': float(t[0]),
                'y': float(t[1]),
                'z': float(t[2]),
            },
            'rotation': {
                'x': float(quat[0]),
                'y': float(quat[1]),
                'z': float(quat[2]),
                'w': float(quat[3]),
            }
        }
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  saved: {os.path.basename(path)}")


def compute_all_extrinsics(config_dir=None):
    """计算所有外参"""
    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("外参计算脚本")
    print("=" * 70)

    # ========== 基础外参 ==========
    print("\n[1/5] 基础外参 (来自标定和 URDF)")
    print("-" * 70)

    # 基础1: rslidar → optical (标定结果)
    rslidar_to_optical_path = os.path.join(config_dir,
        'extrinsics_rslidar_to_top_camera_optical_frame.yaml')
    R_rslidar_optical, t_rslidar_optical = load_tf_extrinsics(rslidar_to_optical_path)
    T_rslidar_optical = tf_to_matrix(R_rslidar_optical, t_rslidar_optical)
    euler = Rotation.from_matrix(R_rslidar_optical).as_euler('xyz', degrees=True)
    print(f"  rslidar → optical (标定):")
    print(f"    t = [{t_rslidar_optical[0]:.4f}, {t_rslidar_optical[1]:.4f}, {t_rslidar_optical[2]:.4f}]")
    print(f"    euler = [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")

    # 基础2: base_link → arm_base (URDF)
    # 从 URDF: base_link → box_link (0,0,0) → arm_base (0.118, 0, 0.1)
    t_base_arm = np.array([0.118, 0.0, 0.1])
    R_base_arm = np.eye(3)  # 无旋转
    quat_base_arm = np.array([0.0, 0.0, 0.0, 1.0])
    T_base_arm = tf_to_matrix(R_base_arm, t_base_arm)
    print(f"  base_link → arm_base (URDF):")
    print(f"    t = [{t_base_arm[0]:.4f}, {t_base_arm[1]:.4f}, {t_base_arm[2]:.4f}]")

    # 基础3: base_link → rslidar (URDF)
    # 从 URDF: base_link → lidar_link (0.27254, 0.00035292, 0.0315) → rslidar (0, 0, 0.0635)
    t_base_rslidar = np.array([0.27254, 0.00035292, 0.0315 + 0.0635])
    R_base_rslidar = np.eye(3)  # 无旋转 (忽略微小角度)
    quat_base_rslidar = np.array([0.0, 0.0, 0.0, 1.0])
    T_base_rslidar = tf_to_matrix(R_base_rslidar, t_base_rslidar)
    print(f"  base_link → rslidar (URDF):")
    print(f"    t = [{t_base_rslidar[0]:.4f}, {t_base_rslidar[1]:.4f}, {t_base_rslidar[2]:.4f}]")

    # ========== 派生外参 ==========
    print("\n[2/5] 计算派生外参...")
    print("-" * 70)

    # arm_base → rslidar = inv(base → arm) @ (base → rslidar)
    T_arm_rslidar = np.linalg.inv(T_base_arm) @ T_base_rslidar
    _, t_arm_rslidar, quat_arm_rslidar = matrix_to_tf(T_arm_rslidar)
    print(f"  arm_base → rslidar: t = [{t_arm_rslidar[0]:.4f}, {t_arm_rslidar[1]:.4f}, {t_arm_rslidar[2]:.4f}]")

    # arm_base → optical = arm_to_rslidar @ rslidar_to_optical
    # TF 链式变换: 从右到左读，先找 rslidar 在 arm 中，再找 optical 在 rslidar 中
    T_arm_optical = T_arm_rslidar @ T_rslidar_optical
    _, t_arm_optical, quat_arm_optical = matrix_to_tf(T_arm_optical)
    euler = Rotation.from_quat(quat_arm_optical).as_euler('xyz', degrees=True)
    print(f"  arm_base → optical: t = [{t_arm_optical[0]:.4f}, {t_arm_optical[1]:.4f}, {t_arm_optical[2]:.4f}]")

    # base_link → optical = base_to_rslidar @ rslidar_to_optical
    T_base_optical = T_base_rslidar @ T_rslidar_optical
    _, t_base_optical, quat_base_optical = matrix_to_tf(T_base_optical)
    print(f"  base_link → optical: t = [{t_base_optical[0]:.4f}, {t_base_optical[1]:.4f}, {t_base_optical[2]:.4f}]")

    # 逆变换
    T_rslidar_arm = np.linalg.inv(T_arm_rslidar)
    _, t_rslidar_arm, quat_rslidar_arm = matrix_to_tf(T_rslidar_arm)

    T_optical_arm = np.linalg.inv(T_arm_optical)
    _, t_optical_arm, quat_optical_arm = matrix_to_tf(T_optical_arm)

    T_rslidar_base = np.linalg.inv(T_base_rslidar)
    _, t_rslidar_base, quat_rslidar_base = matrix_to_tf(T_rslidar_base)

    T_optical_rslidar = np.linalg.inv(T_rslidar_optical)
    _, t_optical_rslidar, quat_optical_rslidar = matrix_to_tf(T_optical_rslidar)

    T_optical_base = np.linalg.inv(T_base_optical)
    _, t_optical_base, quat_optical_base = matrix_to_tf(T_optical_base)

    T_arm_base = np.linalg.inv(T_base_arm)
    _, t_arm_base, quat_arm_base = matrix_to_tf(T_arm_base)

    # ========== 保存外参文件 ==========
    print("\n[3/5] 保存外参文件...")
    print("-" * 70)

    # base_link → arm_base (URDF)
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_base_link_to_arm_base_link.yaml'),
        'base_link', 'arm_base_link',
        t_base_arm, quat_base_arm, 'URDF: base_link → box_link → arm_base')

    # arm_base → base_link
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_arm_base_link_to_base_link.yaml'),
        'arm_base_link', 'base_link',
        t_arm_base, quat_arm_base, 'Computed: inv(base_link → arm_base)')

    # base_link → rslidar (URDF)
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_base_link_to_rslidar.yaml'),
        'base_link', 'rslidar',
        t_base_rslidar, quat_base_rslidar, 'URDF: base_link → lidar_link → rslidar')

    # rslidar → base_link
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_rslidar_to_base_link.yaml'),
        'rslidar', 'base_link',
        t_rslidar_base, quat_rslidar_base, 'Computed: inv(base_link → rslidar)')

    # arm_base → rslidar
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_arm_base_link_to_rslidar.yaml'),
        'arm_base_link', 'rslidar',
        t_arm_rslidar, quat_arm_rslidar, 'Computed: inv(base → arm) @ (base → rslidar)')

    # rslidar → arm_base
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_rslidar_to_arm_base_link.yaml'),
        'rslidar', 'arm_base_link',
        t_rslidar_arm, quat_rslidar_arm, 'Computed: inv(arm_base → rslidar)')

    # arm_base → optical (链式)
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_arm_base_link_to_top_camera_optical_frame.yaml'),
        'arm_base_link', 'top_camera_optical_frame',
        t_arm_optical, quat_arm_optical, 'Computed: arm_to_rslidar @ rslidar_to_optical')

    # optical → arm_base
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_top_camera_optical_frame_to_arm_base_link.yaml'),
        'top_camera_optical_frame', 'arm_base_link',
        t_optical_arm, quat_optical_arm, 'Computed: inv(arm_base → optical)')

    # base_link → optical
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_base_link_to_top_camera_optical_frame.yaml'),
        'base_link', 'top_camera_optical_frame',
        t_base_optical, quat_base_optical, 'Computed: base_to_rslidar @ rslidar_to_optical')

    # optical → base_link
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_top_camera_optical_frame_to_base_link.yaml'),
        'top_camera_optical_frame', 'base_link',
        t_optical_base, quat_optical_base, 'Computed: inv(base_link → optical)')

    # optical → rslidar
    save_extrinsics(
        os.path.join(config_dir, 'extrinsics_top_camera_optical_frame_to_rslidar.yaml'),
        'top_camera_optical_frame', 'rslidar',
        t_optical_rslidar, quat_optical_rslidar, 'Computed: inv(rslidar → optical)')

    # ========== 验证一致性 ==========
    print("\n[4/5] 验证变换链一致性...")
    print("-" * 70)

    # 测试点 (rslidar 坐标系)
    test_point = np.array([1.5, 0.2, 0.1, 1])  # 齐次坐标

    # 验证1: 往返一致性 (rslidar → optical → rslidar)
    p_optical = T_rslidar_optical @ test_point
    p_rslidar_back = T_optical_rslidar @ p_optical
    diff1 = np.linalg.norm(test_point[:3] - p_rslidar_back[:3])
    print(f"  往返测试 (rslidar → optical → rslidar):")
    print(f"    差异: {diff1:.10f} m {'✓' if diff1 < 1e-10 else '✗'}")

    # 验证2: 往返一致性 (arm → optical → arm)
    test_point_arm = np.array([0.5, 0.1, 0.2, 1])
    p_optical2 = T_arm_optical @ test_point_arm
    p_arm_back = T_optical_arm @ p_optical2
    diff2 = np.linalg.norm(test_point_arm[:3] - p_arm_back[:3])
    print(f"  往返测试 (arm → optical → arm):")
    print(f"    差异: {diff2:.10f} m {'✓' if diff2 < 1e-10 else '✗'}")

    # 说明 URDF 与标定的差异
    print(f"\n  URDF 与标定链的差异分析:")
    p_arm_via_optical = T_optical_arm @ (T_rslidar_optical @ test_point)
    p_arm_urdf = T_rslidar_arm @ test_point
    urdf_diff = np.linalg.norm(p_arm_via_optical[:3] - p_arm_urdf[:3])
    print(f"    标定链 (rslidar→optical→arm): {p_arm_via_optical[:3]}")
    print(f"    URDF链 (rslidar→arm 直接):   {p_arm_urdf[:3]}")
    print(f"    差异: {urdf_diff:.4f} m")
    if urdf_diff > 0.01:
        print(f"    注意: URDF 中 rslidar 位置与实际标定存在 {urdf_diff*100:.1f}cm 偏差")

    # ========== 汇总 ==========
    print("\n[5/5] 外参汇总")
    print("-" * 70)
    print("""
基础外参 (不可修改):
  1. rslidar → optical         (标定)
  2. base_link → arm_base      (URDF: 0.118, 0, 0.1)
  3. base_link → rslidar       (URDF: 0.27254, 0.00035292, 0.095)

派生外参 (自动计算):
  4. arm_base → rslidar        = inv(base→arm) @ (base→rslidar)
  5. arm_base → optical        = arm_to_rslidar @ rslidar_to_optical
  6. base_link → optical       = base_to_rslidar @ rslidar_to_optical
  7. 所有逆变换
""")

    print("=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    compute_all_extrinsics()
