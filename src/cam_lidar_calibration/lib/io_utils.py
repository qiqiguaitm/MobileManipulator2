# -*- coding: utf-8 -*-
"""
IO utilities for camera-LiDAR calibration.
Extracted from interactive_calibrate.py - keeps exact same logic and output.
"""

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_extrinsics(yaml_path, as_point_transform=False):
    """加载外参文件 (TF约定格式)

    Args:
        yaml_path: YAML文件路径
        as_point_transform: True 时返回点变换形式 (P_cam = R @ P_lidar + t)，
                           False 时返回原始 TF 约定 (R_tf, t_tf)
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    t = data['transform']['translation']
    r = data['transform']['rotation']

    t_tf = np.array([t['x'], t['y'], t['z']])
    quat = [r['x'], r['y'], r['z'], r['w']]  # xyzw
    R_tf = Rotation.from_quat(quat).as_matrix()

    if as_point_transform:
        return R_tf.T, -R_tf.T @ t_tf

    return R_tf, t_tf


def load_intrinsics(yaml_path):
    """加载内参文件"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # 支持两种格式
    if 'camera_matrix' in data:
        K = np.array(data['camera_matrix']['data']).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'])
    else:
        K = np.array(data['K']).reshape(3, 3)
        D = np.array(data['D'])
    return K, D


def load_pcd(pcd_path):
    """加载PCD点云 (纯numpy实现，无需open3d)"""
    with open(pcd_path, 'rb') as f:
        # 解析header
        n_points = 0
        fields = []
        data_type = 'ascii'
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('POINTS'):
                n_points = int(line.split()[1])
            elif line.startswith('DATA'):
                data_type = line.split()[1]
                break

        n_fields = len(fields)
        if data_type == 'binary':
            raw = f.read(n_points * n_fields * 4)
            points = np.frombuffer(raw, dtype=np.float32).reshape(n_points, n_fields)
        else:
            points = np.loadtxt(f, dtype=np.float32).reshape(-1, n_fields)

    # 只取xyz前三列
    points = points[:, :3]
    valid = ~np.isnan(points).any(axis=1)
    return points[valid]


def save_extrinsics(yaml_path, R, t, frame_id='rs16_lidar', child_frame_id='camera_optical_frame',
                    n_frames=0, method='interactive_multi_frame', calibration_date=None):
    """保存外参到YAML文件 (TF约定格式)

    Args:
        calibration_date: 标定日期字符串，默认取当天日期
    """
    quat = Rotation.from_matrix(R).as_quat()  # xyzw

    result = {
        'header': {
            'calibration_date': calibration_date or str(np.datetime64('today')),
            'method': method,
            'n_frames': n_frames
        },
        'frame_id': frame_id,
        'child_frame_id': child_frame_id,
        'transform': {
            'translation': {
                'x': float(t[0]),
                'y': float(t[1]),
                'z': float(t[2])
            },
            'rotation': {
                'x': float(quat[0]),
                'y': float(quat[1]),
                'z': float(quat[2]),
                'w': float(quat[3])
            }
        }
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
