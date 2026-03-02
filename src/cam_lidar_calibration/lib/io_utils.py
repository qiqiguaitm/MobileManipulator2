# -*- coding: utf-8 -*-
"""
IO utilities for camera-LiDAR calibration.
Extracted from interactive_calibrate.py - keeps exact same logic and output.
"""

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_extrinsics(yaml_path):
    """加载外参文件"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    t = data['transform']['translation']
    r = data['transform']['rotation']

    translation = np.array([t['x'], t['y'], t['z']])
    quat = [r['x'], r['y'], r['z'], r['w']]  # xyzw
    rotation = Rotation.from_quat(quat).as_matrix()

    return rotation, translation


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
    """加载PCD点云"""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    # 移除NaN
    valid = ~np.isnan(points).any(axis=1)
    return points[valid]


def save_extrinsics(yaml_path, R, t, frame_id='rs16_lidar', child_frame_id='camera_optical_frame',
                    n_frames=0, method='interactive_multi_frame'):
    """保存外参到YAML文件 (TF约定格式)"""
    quat = Rotation.from_matrix(R).as_quat()  # xyzw

    result = {
        'header': {
            'calibration_date': str(np.datetime64('today')),
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
