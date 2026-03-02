#!/usr/bin/env python3
"""OpenCV 双目立体标定工具

从棋盘格图像对直接计算两相机的相对外参 (R, T)，替代 MATLAB Stereo Camera Calibrator。

约定:
  P_cam1 = R * P_cam2 + T
  cam1 = chassis_camera (frame_id / parent)
  cam2 = hand_camera    (child_frame_id / child)

用法:
  # 基本用法
  python3 scripts/stereo_calibrate.py \\
      --data calibration_data/multi_eye_20260302_113545 \\
      --square-size 0.072

  # 同时计算链式外参 (chassis→base_link)
  python3 scripts/stereo_calibrate.py \\
      --data calibration_data/multi_eye_20260302_113545 \\
      --square-size 0.072 \\
      --arm-pose calibration_data/multi_eye_20260302_113545/arm_pose.yaml
"""

import argparse
import glob
import os
import sys
from datetime import date

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# --- Project imports ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'cam_lidar_calibration', 'lib'))
sys.path.insert(0, _SCRIPT_DIR)

from chessboard import detect_chessboard, draw_chessboard
from matlab_stereo_to_yaml import compute_chain, mat_to_quat_t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_intrinsics(meta, camera):
    """从 metadata dict 提取相机内参矩阵和畸变系数。"""
    intr = meta[f'{camera}_intrinsics']
    K = np.array([
        [intr['fx'], 0, intr['cx']],
        [0, intr['fy'], intr['cy']],
        [0, 0, 1],
    ], dtype=np.float64)
    D = np.array(intr['dist_coeffs'], dtype=np.float64)
    size = (intr['width'], intr['height'])
    return K, D, size


def format_yaml(frame_id, child_frame_id, q, t, header_lines=None):
    """生成 YAML 外参字符串 (OpenCV 标定)。"""
    lines = []
    if header_lines:
        for h in header_lines:
            lines.append(f'# {h}')
        lines.append('')
    lines.append('header:')
    lines.append(f"  calibration_date: '{date.today().isoformat()}'")
    lines.append(f"  method: opencv_stereo_calibration")
    lines.append(f"  source: OpenCV stereoCalibrate (CALIB_FIX_INTRINSIC)")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='OpenCV 双目立体标定',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data', required=True,
                        help='标定数据目录 (包含 cam1/, cam2/, metadata.yaml)')
    parser.add_argument('--cam1', default='chassis',
                        help='相机1 目录名 / metadata key 前缀 (default: chassis)')
    parser.add_argument('--cam2', default='hand',
                        help='相机2 目录名 / metadata key 前缀 (default: hand)')
    parser.add_argument('--square-size', type=float, default=None,
                        help='棋盘格方格边长 (米), 默认从 metadata.yaml 读取')
    parser.add_argument('--pattern', default='7x6',
                        help='棋盘格内角点数 (列x行)')
    parser.add_argument('--frame-id', default=None,
                        help='TF parent frame (default: {cam1}_camera_optical_frame)')
    parser.add_argument('--child-frame-id', default=None,
                        help='TF child frame (default: {cam2}_camera_optical_frame)')
    parser.add_argument('--arm-pose',
                        help='arm_pose.yaml 路径 (计算链式外参)')
    parser.add_argument('--hand-eye',
                        default='src/perception/config/extrinsics_flan_to_hand_camera.yaml')
    parser.add_argument('--arm-base',
                        default='src/perception/config/extrinsics_arm_base_link_to_base_link.yaml')
    parser.add_argument('-o', '--output-dir', default='src/perception/config')
    parser.add_argument('--refine-intrinsics', action='store_true',
                        help='允许 OpenCV 优化内参 (内参不可靠时使用)')
    parser.add_argument('--show', action='store_true',
                        help='显示角点检测可视化')
    args = parser.parse_args()

    # --- Defaults derived from cam names ---
    if args.frame_id is None:
        args.frame_id = f'{args.cam1}_camera_optical_frame'
    if args.child_frame_id is None:
        args.child_frame_id = f'{args.cam2}_camera_optical_frame'

    # --- Parse pattern ---
    cols, rows = map(int, args.pattern.split('x'))
    pattern_size = (cols, rows)

    # --- Load metadata & intrinsics ---
    with open(os.path.join(args.data, 'metadata.yaml')) as f:
        meta = yaml.safe_load(f)

    K1, D1, size1 = load_intrinsics(meta, args.cam1)
    K2, D2, size2 = load_intrinsics(meta, args.cam2)

    square_size = args.square_size or meta.get('square_size_m', 0.072)

    print(f'数据目录: {args.data}')
    print(f'相机对: {args.cam1} (parent) ↔ {args.cam2} (child)')
    print(f'棋盘格: {cols}x{rows}, 方格 {square_size * 1000:.1f} mm')
    print(f'图像尺寸: {size1[0]}x{size1[1]}')
    print()

    # --- Detect chessboard in all image pairs ---
    cam1_dir = os.path.join(args.data, args.cam1)
    cam2_dir = os.path.join(args.data, args.cam2)
    cam1_files = sorted(glob.glob(os.path.join(cam1_dir, '*.png')))

    obj_points = []
    img_points1 = []
    img_points2 = []
    used = []
    failed = []

    for cam1_path in cam1_files:
        fname = os.path.basename(cam1_path)
        cam2_path = os.path.join(cam2_dir, fname)
        if not os.path.exists(cam2_path):
            continue

        img1 = cv2.imread(cam1_path)
        img2 = cv2.imread(cam2_path)
        if img1 is None or img2 is None:
            failed.append((fname, 'read_error'))
            continue

        c1, objp1, _ = detect_chessboard(img1, pattern_size, square_size)
        c2, objp2, _ = detect_chessboard(img2, pattern_size, square_size)

        expected_n = cols * rows
        if c1 is not None and c2 is not None:
            if c1.shape[0] != expected_n or c2.shape[0] != expected_n:
                failed.append((fname, f'count c1={c1.shape[0]} c2={c2.shape[0]}'))
                status = f'✗ (count mismatch)'
                print(f'  {fname}: {status}')
                continue
            obj_points.append(objp1.reshape(-1, 1, 3))
            img_points1.append(c1.reshape(-1, 1, 2))
            img_points2.append(c2.reshape(-1, 1, 2))
            used.append(fname)
            status = '✓'
        else:
            side = ('both' if c1 is None and c2 is None
                    else args.cam1 if c1 is None else args.cam2)
            failed.append((fname, side))
            status = f'✗ ({side})'

        print(f'  {fname}: {status}')

        if args.show and c1 is not None and c2 is not None:
            vis1 = draw_chessboard(img1, c1, pattern_size)
            vis2 = draw_chessboard(img2, c2, pattern_size)
            vis1 = cv2.resize(vis1, None, fx=0.5, fy=0.5)
            vis2 = cv2.resize(vis2, None, fx=0.5, fy=0.5)
            cv2.imshow('Stereo Detection', np.hstack([vis1, vis2]))
            if cv2.waitKey(200) == 27:
                args.show = False
                cv2.destroyAllWindows()

    if args.show:
        cv2.destroyAllWindows()

    print()
    print(f'检测结果: {len(used)}/{len(used) + len(failed)} 对成功')
    if failed:
        print(f'  失败: {", ".join(f"{n}({s})" for n, s in failed)}')

    if len(used) < 3:
        print('错误: 成功检测的图像对不足 (<3), 无法标定')
        sys.exit(1)

    # --- Stereo calibration ---
    if args.refine_intrinsics:
        flags = 0
        mode_str = '优化内参+外参'
    else:
        flags = cv2.CALIB_FIX_INTRINSIC
        mode_str = 'CALIB_FIX_INTRINSIC'

    print()
    print(f'运行 stereoCalibrate ({mode_str}) ...')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    ret, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        K1, D1, K2, D2, size1,
        flags=flags,
        criteria=criteria,
    )

    if args.refine_intrinsics:
        print(f'  优化后 cam1 fx={K1_out[0,0]:.2f} fy={K1_out[1,1]:.2f} '
              f'cx={K1_out[0,2]:.2f} cy={K1_out[1,2]:.2f}')
        print(f'  优化后 cam2 fx={K2_out[0,0]:.2f} fy={K2_out[1,1]:.2f} '
              f'cx={K2_out[0,2]:.2f} cy={K2_out[1,2]:.2f}')

    T_vec = T.flatten()

    print(f'  RMS 重投影误差: {ret:.4f} px')

    # stereoCalibrate 约定: P_cam2 = R * P_cam1 + T  (chassis→hand)
    # TF 约定需要:         P_parent = R * P_child + T  (hand→chassis)
    # 反转变换
    R_tf = R.T
    T_tf = -R.T @ T_vec

    print(f'  [OpenCV raw]  T = [{T_vec[0]*1000:.3f}, {T_vec[1]*1000:.3f}, {T_vec[2]*1000:.3f}] mm')
    print(f'  [TF inverted] T = [{T_tf[0]*1000:.3f}, {T_tf[1]*1000:.3f}, {T_tf[2]*1000:.3f}] mm')
    euler = Rotation.from_matrix(R_tf).as_euler('xyz', degrees=True)
    print(f'  [TF inverted] Euler (xyz): [{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}] deg')

    # --- Build 4x4 transform (TF convention) ---
    T_4x4 = np.eye(4)
    T_4x4[:3, :3] = R_tf
    T_4x4[:3, 3] = T_tf

    q, t = mat_to_quat_t(T_4x4)

    # --- Output 1: direct extrinsics (cam1 ↔ cam2) ---
    header = [
        f'Extrinsics: {args.frame_id} -> {args.child_frame_id}',
        f'P_{args.frame_id.split("_")[0]} = R * P_{args.child_frame_id.split("_")[0]} + T',
        f'OpenCV stereoCalibrate, {len(used)} image pairs, RMS={ret:.4f} px',
    ]
    yaml_str = format_yaml(args.frame_id, args.child_frame_id, q, t, header)

    print()
    print('=' * 60)
    print(f'直接外参: {args.frame_id} → {args.child_frame_id}')
    print('=' * 60)
    print(yaml_str)

    os.makedirs(args.output_dir, exist_ok=True)
    fname = f'extrinsics_{args.frame_id}_to_{args.child_frame_id}.yaml'
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, 'w') as f:
        f.write(yaml_str)
    print(f'  → 已保存: {out_path}')

    # --- Output 2: chain extrinsics (cam1 → base_link) ---
    if args.arm_pose:
        print()
        print('=' * 60)
        print(f'链式外参: {args.frame_id} → base_link')
        print('=' * 60)

        T_cam1_base = compute_chain(
            T_4x4, args.arm_pose, args.hand_eye, args.arm_base)
        q2, t2 = mat_to_quat_t(T_cam1_base)

        header2 = [
            f'Extrinsics: {args.frame_id} -> base_link',
            f'Chain: {args.frame_id} → {args.child_frame_id} → flange → arm_base → base_link',
            f'hand_eye: {args.hand_eye}',
            f'arm_pose: {args.arm_pose}',
        ]
        yaml_chain = format_yaml(args.frame_id, 'base_link', q2, t2, header2)
        print(yaml_chain)

        fname2 = f'extrinsics_{args.frame_id}_to_base_link_stereo_hand.yaml'
        out_path2 = os.path.join(args.output_dir, fname2)
        with open(out_path2, 'w') as f:
            f.write(yaml_chain)
        print(f'  → 已保存: {out_path2}')

        print(f'  translation: [{t2[0]*1000:.1f}, {t2[1]*1000:.1f}, {t2[2]*1000:.1f}] mm')
        euler2 = Rotation.from_quat(q2).as_euler('xyz', degrees=True)
        print(f'  euler(xyz):  [{euler2[0]:.2f}, {euler2[1]:.2f}, {euler2[2]:.2f}] deg')
    else:
        print()
        print('提示: 添加 --arm-pose <path> 可同时计算 chassis_cam→base_link 链式外参')

    print()
    print('完成。')


if __name__ == '__main__':
    main()
