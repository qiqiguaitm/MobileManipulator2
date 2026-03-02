#!/usr/bin/env python3
"""MATLAB Stereo Camera Calibrator 结果 → YAML 外参转换工具

将 MATLAB Stereo Camera Calibrator 输出的 R, T 转换为项目的 YAML 外参格式。
支持直接输出 cam1↔cam2 外参，以及通过手眼标定链计算 chassis_cam→base_link。

用法:
  # 交互模式 — 粘贴 MATLAB R, T 即可
  python3 scripts/matlab_stereo_to_yaml.py

  # 命令行模式 — 指定 R, T 和 frame 名称
  python3 scripts/matlab_stereo_to_yaml.py \
      --R "0.9996,0.0203,-0.0180;-0.0069,0.8326,0.5538;0.0263,-0.5535,0.8325" \
      --T "-12.255,-733.685,-75.404" \
      --frame-id chassis_camera_optical_frame \
      --child-frame-id hand_camera_optical_frame

  # 同时计算手臂链式外参 (chassis_cam → base_link)
  python3 scripts/matlab_stereo_to_yaml.py \
      --R "..." --T "..." \
      --arm-pose calibration_data/multi_eye_20260302_113545/arm_pose.yaml \
      --hand-eye src/perception/config/extrinsics_flan_to_hand_camera.yaml

MATLAB 约定:
  Camera1 = 参考相机 (frame_id), Camera2 = 目标相机 (child_frame_id)
  P_Camera1 = R * P_Camera2 + T   (T 单位: mm)

YAML TF 约定:
  P_parent = R_tf * P_child + t_tf
  frame_id = parent, child_frame_id = child

两者直接对应: R_tf = R_matlab, t_tf = T_matlab / 1000
"""

import argparse
import os
import sys
import re
from datetime import date

import numpy as np
from scipy.spatial.transform import Rotation


def parse_matrix_string(s):
    """解析 MATLAB 矩阵字符串，支持多种格式:
    - 分号分行: "1,2,3;4,5,6;7,8,9"
    - 方括号:   "[1 2 3; 4 5 6; 7 8 9]"
    - 多行粘贴
    """
    s = s.strip().strip('[]')
    s = re.sub(r'[,\s]+', ' ', s)  # 统一分隔符为空格
    rows = [r.strip() for r in s.split(';') if r.strip()]
    if len(rows) == 1:
        # 可能是单行向量
        vals = [float(x) for x in rows[0].split()]
        return np.array(vals)
    return np.array([[float(x) for x in row.split()] for row in rows])


def load_yaml_transform(path):
    """从 YAML 外参文件加载 4x4 TF 矩阵"""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    t = data['transform']['translation']
    r = data['transform']['rotation']
    R_mat = Rotation.from_quat([r['x'], r['y'], r['z'], r['w']]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [t['x'], t['y'], t['z']]
    return T


def load_arm_pose(path):
    """从 arm_pose.yaml 加载法兰位姿，返回 4x4 T_{arm_base←flange}"""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    p = data['flange_pose']
    xyz_m = np.array([p['x_mm'], p['y_mm'], p['z_mm']]) / 1000.0
    rpy_deg = [p['roll_deg'], p['pitch_deg'], p['yaw_deg']]
    R_mat = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = xyz_m
    return T


def mat_to_quat_t(T):
    """4x4 齐次矩阵 → (quat_xyzw, translation)"""
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]
    return q, T[:3, 3]


def format_yaml(frame_id, child_frame_id, q, t, header_lines=None):
    """生成 YAML 外参字符串"""
    lines = []
    if header_lines:
        for h in header_lines:
            lines.append(f'# {h}')
        lines.append('')

    lines.append('header:')
    lines.append(f"  calibration_date: '{date.today().isoformat()}'")
    lines.append(f"  method: matlab_stereo_calibration")
    lines.append(f"  source: MATLAB Stereo Camera Calibrator App")
    lines.append('')
    lines.append(f'frame_id: {frame_id}')
    lines.append(f'child_frame_id: {child_frame_id}')
    lines.append('')
    lines.append('transform:')
    lines.append('  translation:')
    lines.append(f'    x: {t[0]:.16f}')
    lines.append(f'    y: {t[1]:.16f}')
    lines.append(f'    z: {t[2]:.16f}')
    lines.append('  rotation:')
    lines.append(f'    x: {q[0]:.16f}')
    lines.append(f'    y: {q[1]:.16f}')
    lines.append(f'    z: {q[2]:.16f}')
    lines.append(f'    w: {q[3]:.16f}')
    lines.append('')
    return '\n'.join(lines)


def interactive_input():
    """交互模式：引导用户粘贴 MATLAB 输出"""
    print('=' * 60)
    print('MATLAB Stereo Calibrator → YAML 外参转换')
    print('=' * 60)

    print('\n请粘贴 MATLAB 的旋转矩阵 R (3x3):')
    print('  格式: 分号分行，如 "0.999,0.020,-0.018;-0.007,0.833,0.554;0.026,-0.553,0.832"')
    R_str = input('R = ').strip()
    R_mat = parse_matrix_string(R_str)
    if R_mat.shape != (3, 3):
        print(f'错误: R 应为 3x3，得到 {R_mat.shape}')
        sys.exit(1)

    print('\n请粘贴 MATLAB 的平移向量 T (mm):')
    print('  格式: 如 "-12.255,-733.685,-75.404"')
    T_str = input('T = ').strip()
    T_mm = parse_matrix_string(T_str)
    if T_mm.size != 3:
        print(f'错误: T 应为 3 个元素，得到 {T_mm.size}')
        sys.exit(1)
    T_mm = T_mm.flatten()

    print('\nCamera1 (参考相机) frame_id:')
    frame_id = input('  [chassis_camera_optical_frame]: ').strip()
    if not frame_id:
        frame_id = 'chassis_camera_optical_frame'

    print('Camera2 (目标相机) child_frame_id:')
    child_id = input('  [hand_camera_optical_frame]: ').strip()
    if not child_id:
        child_id = 'hand_camera_optical_frame'

    return R_mat, T_mm, frame_id, child_id


def compute_chain(T_cam1_cam2, arm_pose_path, hand_eye_path, arm_base_path):
    """计算链式变换: cam1 → cam2 → flange → arm_base → base_link

    Chain:
      T_{cam1 ← base_link}
        = T_{cam1 ← cam2}
        @ inv(T_{flan ← cam2})       [hand-eye]
        @ inv(T_{arm_base ← flan})   [flange pose]
        @ T_{arm_base ← base_link}
    """
    T_flan_cam2 = load_yaml_transform(hand_eye_path)
    T_arm_flan = load_arm_pose(arm_pose_path)
    T_arm_base = load_yaml_transform(arm_base_path)

    T_cam2_flan = np.linalg.inv(T_flan_cam2)
    T_flan_arm = np.linalg.inv(T_arm_flan)

    T_cam1_base = T_cam1_cam2 @ T_cam2_flan @ T_flan_arm @ T_arm_base

    return T_cam1_base


def main():
    parser = argparse.ArgumentParser(
        description='MATLAB Stereo Calibrator → YAML 外参转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--R', help='旋转矩阵 (分号分行)')
    parser.add_argument('--T', '--translation', help='平移向量 (逗号分隔, mm)')
    parser.add_argument('--frame-id', default='chassis_camera_optical_frame')
    parser.add_argument('--child-frame-id', default='hand_camera_optical_frame')
    parser.add_argument('--arm-pose', help='arm_pose.yaml 路径 (计算链式外参)')
    parser.add_argument('--hand-eye',
                        default='src/perception/config/extrinsics_flan_to_hand_camera.yaml',
                        help='手眼标定 YAML')
    parser.add_argument('--arm-base',
                        default='src/perception/config/extrinsics_arm_base_link_to_base_link.yaml',
                        help='arm_base→base_link YAML')
    parser.add_argument('-o', '--output-dir',
                        default='src/perception/config',
                        help='输出目录')
    parser.add_argument('--dry-run', action='store_true', help='只打印不写文件')
    args = parser.parse_args()

    # 输入
    if args.R and args.T:
        R_mat = parse_matrix_string(args.R)
        T_mm = parse_matrix_string(args.T).flatten()
        frame_id = args.frame_id
        child_id = args.child_frame_id
    else:
        R_mat, T_mm, frame_id, child_id = interactive_input()

    # 验证旋转矩阵
    det = np.linalg.det(R_mat)
    if abs(det - 1.0) > 0.01:
        print(f'警告: det(R) = {det:.6f}，不是有效旋转矩阵')
        sys.exit(1)

    T_m = T_mm / 1000.0

    # 构建 4x4 TF 矩阵 (MATLAB convention = TF convention)
    T_4x4 = np.eye(4)
    T_4x4[:3, :3] = R_mat
    T_4x4[:3, 3] = T_m

    q, t = mat_to_quat_t(T_4x4)

    # 输出 1: 直接外参 (cam1 ↔ cam2)
    header = [
        f'Extrinsics: {frame_id} -> {child_id}',
        f'P_{frame_id.split("_")[0]} = R * P_{child_id.split("_")[0]} + T',
        f'MATLAB: Camera1={frame_id.split("_")[0]}, Camera2={child_id.split("_")[0]}',
    ]
    yaml_direct = format_yaml(frame_id, child_id, q, t, header)

    print('\n' + '=' * 60)
    print(f'[1] 直接外参: {frame_id} → {child_id}')
    print('=' * 60)
    print(yaml_direct)

    if not args.dry_run:
        fname = f'extrinsics_{frame_id}_to_{child_id}.yaml'
        path = os.path.join(args.output_dir, fname)
        with open(path, 'w') as f:
            f.write(yaml_direct)
        print(f'  → 已保存: {path}')

    # 输出 2: 链式外参 (cam1 → base_link)
    if args.arm_pose:
        print('\n' + '=' * 60)
        print(f'[2] 链式外参: {frame_id} → base_link')
        print('=' * 60)

        T_cam1_base = compute_chain(T_4x4, args.arm_pose, args.hand_eye, args.arm_base)
        q2, t2 = mat_to_quat_t(T_cam1_base)

        header2 = [
            f'Extrinsics: {frame_id} -> base_link',
            f'Chain: {frame_id} → {child_id} → flange → arm_base → base_link',
            f'hand_eye: {args.hand_eye}',
            f'arm_pose: {args.arm_pose}',
        ]
        yaml_chain = format_yaml(frame_id, 'base_link', q2, t2, header2)
        print(yaml_chain)

        if not args.dry_run:
            fname2 = f'extrinsics_{frame_id}_to_base_link_stereo_hand.yaml'
            path2 = os.path.join(args.output_dir, fname2)
            with open(path2, 'w') as f:
                f.write(yaml_chain)
            print(f'  → 已保存: {path2}')

        # 打印对比信息
        print('  链式变换详情:')
        print(f'    translation: [{t2[0]*1000:.1f}, {t2[1]*1000:.1f}, {t2[2]*1000:.1f}] mm')
        rot_deg = Rotation.from_quat(q2).as_euler('xyz', degrees=True)
        print(f'    euler(xyz):  [{rot_deg[0]:.2f}, {rot_deg[1]:.2f}, {rot_deg[2]:.2f}] deg')
    else:
        print('\n提示: 添加 --arm-pose <path> 可同时计算 chassis_cam→base_link 链式外参')

    print('\n完成。')


if __name__ == '__main__':
    main()
