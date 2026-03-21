#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标定结果部署工具

将标定结果目录中的 extrinsics YAML 部署到 perception/config/，
自动处理 frame_id 映射 (rs16_lidar -> rslidar) 和文件备份。

Usage:
    export PATH="/usr/bin:$PATH"

    # 部署 chassis 相机标定结果
    python3 src/cam_lidar_calibration/deploy_calibration.py \
      src/cam_lidar_calibration/data/calib_data_chassis_20260302_164551/config/results/chassis_camera2lidar_20260302_165026 \
      --camera chassis

    # dry-run 模式 (只打印差异，不复制)
    python3 src/cam_lidar_calibration/deploy_calibration.py \
      src/cam_lidar_calibration/data/calib_data_chassis_20260302_164551/config/results/chassis_camera2lidar_20260302_165026 \
      --camera chassis --dry-run
"""

import argparse
import glob
import os
import shutil
import sys
import datetime

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPLOY_DIR = os.path.join(PROJECT_ROOT, 'src', 'perception', 'config')

# 标定输出 frame_id -> 部署 frame_id
FRAME_ID_MAP = {
    'rs16_lidar': 'rslidar',
}

# camera type -> 部署目标文件名中的 child_frame 部分
CAMERA_CHILD_FRAME = {
    'chassis': 'chassis_camera_optical_frame',
    'top': 'top_camera_optical_frame',
}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_Rt(data):
    """从 YAML data 提取 (R, t) — TF 约定。"""
    tf = data['transform']
    t = np.array([tf['translation']['x'], tf['translation']['y'], tf['translation']['z']])
    q = tf['rotation']
    R = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    return R, t


def rotation_angle_diff(R1, R2):
    """两个旋转矩阵之间的角度差 (度)。"""
    dR = R1.T @ R2
    cos_angle = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def print_diff(old_data, new_data):
    """打印新旧外参差异。"""
    R_old, t_old = extract_Rt(old_data)
    R_new, t_new = extract_Rt(new_data)

    angle_diff = rotation_angle_diff(R_old, R_new)
    trans_diff = np.linalg.norm(t_old - t_new)

    print(f"  旋转差异: {angle_diff:.4f} deg")
    print(f"  平移差异: {trans_diff:.6f} m ({trans_diff*1000:.2f} mm)")
    print(f"  旧平移: [{t_old[0]:.6f}, {t_old[1]:.6f}, {t_old[2]:.6f}]")
    print(f"  新平移: [{t_new[0]:.6f}, {t_new[1]:.6f}, {t_new[2]:.6f}]")

    old_date = old_data.get('header', {}).get('calibration_date', '?')
    new_date = new_data.get('header', {}).get('calibration_date', '?')
    print(f"  旧日期: {old_date}")
    print(f"  新日期: {new_date}")


def remap_frame_ids(data):
    """将标定输出的 frame_id 映射为部署约定。"""
    data = dict(data)
    fid = data.get('frame_id', '')
    if fid in FRAME_ID_MAP:
        data['frame_id'] = FRAME_ID_MAP[fid]
    if 'header' in data:
        header = dict(data['header'])
        if header.get('frame_id') in FRAME_ID_MAP:
            header['frame_id'] = FRAME_ID_MAP[header['frame_id']]
        data['header'] = header
    return data


def find_calib_yaml(result_dir, camera_type):
    """在标定结果目录中找到 extrinsics YAML (排除 apply_ 文件)。"""
    pattern = os.path.join(result_dir, 'extrinsics_*.yaml')
    candidates = [f for f in glob.glob(pattern) if 'apply_' not in os.path.basename(f)]
    if not candidates:
        print(f"错误: 在 {result_dir} 中未找到 extrinsics_*.yaml")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"警告: 找到多个标定文件，使用第一个:")
        for c in candidates:
            print(f"  {c}")
    return candidates[0]


def deploy_target_path(camera_type):
    """生成部署目标路径。"""
    child_frame = CAMERA_CHILD_FRAME[camera_type]
    filename = f'extrinsics_rslidar_to_{child_frame}.yaml'
    return os.path.join(DEPLOY_DIR, filename)


def backup_file(path):
    """备份文件为 .bak.时间戳。"""
    if not os.path.exists(path):
        return None
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = f"{path}.bak.{timestamp}"
    shutil.copy2(path, bak)
    return bak


def main():
    parser = argparse.ArgumentParser(description='部署标定结果到 perception/config/')
    parser.add_argument('result_dir', help='标定结果目录路径')
    parser.add_argument('--camera', required=True, choices=['chassis', 'top'],
                        help='相机类型')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印差异，不实际部署')
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    if not os.path.isdir(result_dir):
        print(f"错误: 目录不存在: {result_dir}")
        sys.exit(1)

    # 找到标定结果文件
    src_yaml = find_calib_yaml(result_dir, args.camera)
    dst_path = deploy_target_path(args.camera)

    print(f"{'='*60}")
    print(f" 标定结果部署 {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*60}")
    print(f"  源文件:   {src_yaml}")
    print(f"  目标文件: {dst_path}")
    print(f"  相机类型: {args.camera}")
    print()

    # 加载新标定数据
    new_data = load_yaml(src_yaml)

    # frame_id 映射
    new_data = remap_frame_ids(new_data)

    # 比较差异
    if os.path.exists(dst_path):
        old_data = load_yaml(dst_path)
        print("[差异对比]")
        print_diff(old_data, new_data)
    else:
        print(f"  目标文件不存在，将创建新文件")
    print()

    if args.dry_run:
        print("[DRY RUN] 不执行实际部署")
        return

    # 确认
    answer = input("确认部署? [y/N] ").strip().lower()
    if answer != 'y':
        print("取消")
        return

    # 备份
    bak = backup_file(dst_path)
    if bak:
        print(f"  备份: {bak}")

    # 写入
    with open(dst_path, 'w') as f:
        # 写注释头
        child_frame = CAMERA_CHILD_FRAME[args.camera]
        f.write(f"# Extrinsics: rslidar -> {child_frame}\n")
        f.write(f"# Source: direct {args.camera}-lidar calibration (interactive_calibrate_v2)\n")
        f.write(f"# Deployed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        yaml.dump(new_data, f, default_flow_style=False)

    print(f"  已部署到: {dst_path}")
    print("完成")


if __name__ == '__main__':
    main()
