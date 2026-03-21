# -*- coding: utf-8 -*-
"""
Coordinate transformation utilities for camera-LiDAR calibration.
"""

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def invert_tf_extrinsics(R_tf, t_tf):
    """Convert TF convention extrinsics to point transformation form.

    TF convention:
        t_tf is camera origin position in lidar frame
        R_tf columns are camera axes directions in lidar frame

    Point transformation:
        p_camera = R_inv @ p_lidar + t_inv
        where R_inv = R_tf^T, t_inv = -R_tf^T @ t_tf

    Args:
        R_tf: rotation matrix in TF convention
        t_tf: translation vector in TF convention

    Returns:
        R_inv: rotation matrix for point transformation
        t_inv: translation vector for point transformation
    """
    R_inv = R_tf.T
    t_inv = -R_tf.T @ t_tf
    return R_inv, t_inv


def tf_to_point_transform(R_tf, t_tf):
    """Alias for invert_tf_extrinsics."""
    return invert_tf_extrinsics(R_tf, t_tf)


def point_transform_to_tf(R, t):
    """Convert point transformation form back to TF convention.

    Args:
        R: rotation matrix (point transform form)
        t: translation vector (point transform form)

    Returns:
        R_tf: rotation matrix in TF convention
        t_tf: translation vector in TF convention
    """
    R_tf = R.T
    t_tf = -R.T @ t
    return R_tf, t_tf


def transform_link_to_optical(R, t):
    """Convert camera_link frame to camera_optical_frame.

    camera_link: X-forward, Y-left, Z-up (ROS convention)
    camera_optical_frame: X-right, Y-down, Z-forward (OpenCV convention)

    Args:
        R: rotation matrix in camera_link frame
        t: translation vector in camera_link frame

    Returns:
        R_optical: rotation matrix in optical frame
        t_optical: translation vector in optical frame
    """
    R_link_to_optical = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ], dtype=float)

    R_optical = R_link_to_optical @ R
    t_optical = R_link_to_optical @ t

    return R_optical, t_optical


def project_camera_center_to_lidar(cam_center, R_c2l, t_c2l):
    """Project camera-detected board center to LiDAR frame.

    Args:
        cam_center: board center in camera frame (3,)
        R_c2l: rotation matrix (point transform form)
        t_c2l: translation vector (point transform form)

    Returns:
        lidar_center: board center in LiDAR frame (3,)
    """
    # Inverse transform: p_lidar = R_c2l^T @ (p_camera - t_c2l)
    R_inv = R_c2l.T
    lidar_center = R_inv @ (cam_center - t_c2l)
    return lidar_center


def transform_points_lidar_to_camera(points, R, t):
    """Transform points from LiDAR frame to camera frame.

    Args:
        points: point cloud in LiDAR frame (N, 3)
        R: rotation matrix (point transform form)
        t: translation vector (point transform form)

    Returns:
        points_cam: points in camera frame (N, 3)
    """
    return (R @ points.T).T + t


def transform_points_camera_to_lidar(points, R, t):
    """Transform points from camera frame to LiDAR frame.

    Args:
        points: point cloud in camera frame (N, 3)
        R: rotation matrix (point transform form)
        t: translation vector (point transform form)

    Returns:
        points_lidar: points in LiDAR frame (N, 3)
    """
    R_inv = R.T
    t_inv = -R.T @ t
    return (R_inv @ points.T).T + t_inv


# ============================================================================
# Camera Link <-> Optical Frame Transformations
# ============================================================================

# 静态点变换: camera_optical_frame -> camera_link
# 命名 _PT = Point Transform，即 p_link = R(q) @ p_optical + t
# 来源: RealSense 驱动发布的 tf (camera/top_link -> camera/top_color_optical_frame)
#
# ROS TF 约定: tf_echo A B 输出的四元数 q 用于点变换 p_A = R(q) @ p_B + t
# 所以 tf_echo camera/{}_link camera/{}_color_optical_frame 的输出
# 就是 optical -> link 的点变换，可直接使用
#
# 验证: optical [0,0,1] (前) -> link [1,0,0] (前) ✓
CAMERA_OPTICAL_TO_LINK_PT = {
    'top': {
        # tf_echo camera/top_link camera/top_color_optical_frame (ROS 实际值)
        'translation': np.array([-0.000283, -0.059184, -0.000029]),
        'rotation_quat': np.array([-0.499962, 0.499032, -0.499347, 0.501655]),  # xyzw
    },
    'chassis': {
        # tf_echo camera/chassis_link camera/chassis_color_optical_frame (ROS 实际值)
        'translation': np.array([-0.000245, 0.014810, 0.000133]),
        'rotation_quat': np.array([-0.498668, 0.505262, -0.495065, 0.500949]),  # xyzw
    }
}


def get_camera_optical_to_link_transform(camera_type='top'):
    """获取 optical_frame 到 camera_link 的点变换

    用于: p_link = R @ p_optical + t

    Args:
        camera_type: 'top' or 'chassis'

    Returns:
        R: 旋转矩阵 (3x3)，用于点变换
        t: 平移向量 (3,)
    """
    if camera_type not in CAMERA_OPTICAL_TO_LINK_PT:
        raise ValueError(f"Unknown camera type: {camera_type}. Must be 'top' or 'chassis'")

    tf_data = CAMERA_OPTICAL_TO_LINK_PT[camera_type]
    t = tf_data['translation']
    quat = tf_data['rotation_quat']  # xyzw format
    R = Rotation.from_quat(quat).as_matrix()

    return R, t


def convert_optical_to_link_extrinsics(input_yaml, output_yaml, camera_type='top'):
    """将标定结果从optical_frame转换到camera_link坐标系

    坐标系变换链：
    - T_rslidar_to_camera_optical: 标定直接输出 (rs16_lidar -> camera_optical_frame)
    - T_camera_optical_to_link: RealSense 静态 TF (camera_optical_frame -> camera_link)
    - T_rslidar_to_camera_link: SLAM 最终需要 (rs16_lidar -> camera_link)

    变换链（点变换形式）：
    p_link = R_o2l @ (R_l2o @ p_lid + t_l2o) + t_o2l
           = (R_o2l @ R_l2o) @ p_lid + (R_o2l @ t_l2o + t_o2l)

    Args:
        input_yaml: 输入文件路径 (rs16_lidar -> camera_optical_frame)
        output_yaml: 输出文件路径 (rs16_lidar -> camera_link)
        camera_type: 'top' or 'chassis'
    """
    # 读取标定结果: T_rslidar_to_camera_optical (TF格式)
    with open(input_yaml, 'r') as f:
        calib_data = yaml.safe_load(f)

    # 提取变换参数 (TF约定格式)
    trans = calib_data['transform']['translation']
    rot = calib_data['transform']['rotation']

    t_rslidar_to_optical_tf = np.array([trans['x'], trans['y'], trans['z']])
    q_rslidar_to_optical_tf = np.array([rot['x'], rot['y'], rot['z'], rot['w']])  # xyzw
    R_rslidar_to_optical_tf = Rotation.from_quat(q_rslidar_to_optical_tf).as_matrix()

    # 转换为点变换形式: p_optical = R @ p_lidar + t
    R_rslidar_to_optical, t_rslidar_to_optical = invert_tf_extrinsics(
        R_rslidar_to_optical_tf, t_rslidar_to_optical_tf
    )

    # 获取 T_camera_optical_to_link (RealSense 静态 TF，已经是点变换形式)
    # ROS tf_echo 输出的四元数直接用于点变换: p_link = R @ p_optical + t
    R_camera_optical_to_link, t_camera_optical_to_link = get_camera_optical_to_link_transform(camera_type)

    # 应用变换链: p_link = R_o2l @ (R_l2o @ p_lid + t_l2o) + t_o2l
    # 展开: p_link = (R_o2l @ R_l2o) @ p_lid + (R_o2l @ t_l2o + t_o2l)
    R_rslidar_to_camera_link = R_camera_optical_to_link @ R_rslidar_to_optical
    t_rslidar_to_camera_link = R_camera_optical_to_link @ t_rslidar_to_optical + t_camera_optical_to_link

    # 转换回TF约定格式
    R_rslidar_to_camera_link_tf, t_rslidar_to_camera_link_tf = point_transform_to_tf(
        R_rslidar_to_camera_link, t_rslidar_to_camera_link
    )
    q_rslidar_to_camera_link_tf = Rotation.from_matrix(R_rslidar_to_camera_link_tf).as_quat()  # xyzw

    # 更新child_frame_id
    old_child_frame = calib_data.get('child_frame_id', '')
    if 'optical' in old_child_frame:
        new_child_frame = old_child_frame.replace('_optical_frame', '_link').replace('_color_optical_frame', '_link')
    else:
        new_child_frame = f'camera/{camera_type}_link'

    # 构建输出数据
    output_data = {
        'header': {
            'calibration_date': calib_data['header'].get('calibration_date', ''),
            'method': calib_data['header'].get('method', 'unknown'),
            'n_frames': calib_data['header'].get('n_frames', 0),
            'note': f'Converted from optical_frame to camera_link using static TF'
        },
        'frame_id': calib_data.get('frame_id', 'rs16_lidar'),
        'child_frame_id': new_child_frame,
        'transform': {
            'translation': {
                'x': float(t_rslidar_to_camera_link_tf[0]),
                'y': float(t_rslidar_to_camera_link_tf[1]),
                'z': float(t_rslidar_to_camera_link_tf[2])
            },
            'rotation': {
                'x': float(q_rslidar_to_camera_link_tf[0]),
                'y': float(q_rslidar_to_camera_link_tf[1]),
                'z': float(q_rslidar_to_camera_link_tf[2]),
                'w': float(q_rslidar_to_camera_link_tf[3])
            }
        }
    }

    # 保存结果
    with open(output_yaml, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False)

    print(f"\n[坐标系转换]")
    print(f"  输入: {input_yaml}")
    print(f"    {calib_data['frame_id']} -> {calib_data['child_frame_id']}")
    print(f"  输出: {output_yaml}")
    print(f"    {output_data['frame_id']} -> {output_data['child_frame_id']}")
    print(f"  相机类型: {camera_type}")
    print(f"\n  变换详情:")
    print(f"    T_rslidar_to_camera_optical: t=[{t_rslidar_to_optical_tf[0]:.4f}, {t_rslidar_to_optical_tf[1]:.4f}, {t_rslidar_to_optical_tf[2]:.4f}]")
    print(f"    T_camera_optical_to_link:    t=[{t_camera_optical_to_link[0]:.6f}, {t_camera_optical_to_link[1]:.6f}, {t_camera_optical_to_link[2]:.6f}]")
    print(f"    T_rslidar_to_camera_link:    t=[{t_rslidar_to_camera_link_tf[0]:.4f}, {t_rslidar_to_camera_link_tf[1]:.4f}, {t_rslidar_to_camera_link_tf[2]:.4f}]")

    return output_data
