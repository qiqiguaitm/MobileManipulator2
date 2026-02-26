#!/usr/bin/env python3
"""
坐标变换工具

加载外参文件，执行坐标系间的变换。
支持: rslidar, top_camera_optical_frame, arm_base_link, base_link

这是一个纯 Python 模块，不依赖 ROS。
"""

import os
from typing import Optional, Dict

import yaml
import numpy as np
from scipy.spatial.transform import Rotation


class CoordinateTransformer:
    """坐标变换器，管理多个坐标系间的变换关系"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: 外参配置文件目录
        """
        self.config_dir = config_dir
        self._transforms: Dict[str, np.ndarray] = {}

    def set_config_dir(self, config_dir: str):
        """设置配置文件目录"""
        self.config_dir = os.path.abspath(config_dir)

    def load_extrinsics(self, yaml_path: str, as_point_transform: bool = True) -> np.ndarray:
        """
        加载外参 YAML 文件，返回 4x4 变换矩阵

        Args:
            yaml_path: YAML 文件路径 (绝对或相对于 config_dir)
            as_point_transform: 是否转换为点变换形式 (默认 True)
                - True: 返回点变换矩阵，用于 p_cam = T @ p_lidar
                - False: 返回原始 TF 约定矩阵

        Returns:
            T: 4x4 齐次变换矩阵

        注意:
            外参 YAML 文件使用 TF 约定存储:
            - t_tf: 目标坐标系原点在源坐标系中的位置
            - R_tf: 目标坐标系坐标轴在源坐标系中的方向

            点变换需要转换:
            - R_point = R_tf^T
            - t_point = -R_tf^T @ t_tf
        """
        # 处理路径
        if not os.path.isabs(yaml_path) and self.config_dir:
            yaml_path = os.path.join(self.config_dir, yaml_path)

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # 解析 translation
        trans = data['transform']['translation']
        t_tf = np.array([trans['x'], trans['y'], trans['z']])

        # 解析 rotation (quaternion xyzw)
        rot = data['transform']['rotation']
        quat = np.array([rot['x'], rot['y'], rot['z'], rot['w']])

        # 四元数转旋转矩阵 (TF 约定)
        R_tf = Rotation.from_quat(quat).as_matrix()

        if as_point_transform:
            # 转换为点变换形式: p_target = R_point @ p_source + t_point
            R = R_tf.T
            t = -R_tf.T @ t_tf
        else:
            # 保持 TF 约定
            R = R_tf
            t = t_tf

        # 构建 4x4 齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def load_all_extrinsics(self, camera_name: str = 'top', suffix: str = ''):
        """加载所有需要的外参文件

        外参使用链式变换，支持多个目标坐标系:
        - arm_base_link: 机械臂基座
        - base_link: 底盘中心 (默认)

        变换链:
        - optical → arm_base_link
        - optical → base_link
        - rslidar → arm_base_link
        - rslidar → base_link

        Args:
            camera_name: 相机名称 ('top' 或 'chassis')
            suffix: 外参文件后缀 (如 '_new'，用于测试新标定)
        """
        if self.config_dir is None:
            print("[WARN] config_dir not set, skip loading extrinsics")
            return

        # 根据 camera_name 动态选择外参文件名
        optical_frame = f'{camera_name}_camera_optical_frame'
        optical_to_base_file = f'extrinsics_{optical_frame}_to_base_link{suffix}.yaml'

        extrinsics_files = {
            # 原有变换
            'rslidar_to_optical': f'extrinsics_rslidar_to_{optical_frame}.yaml',
            'arm_to_rslidar': 'extrinsics_arm_base_link_to_rslidar.yaml',
            'arm_to_optical': f'extrinsics_arm_base_link_to_{optical_frame}.yaml',
            # base_link 变换 (应用后缀)
            'optical_to_base': optical_to_base_file,
            'rslidar_to_base': 'extrinsics_rslidar_to_base_link.yaml',
        }

        for name, filename in extrinsics_files.items():
            try:
                T = self.load_extrinsics(filename)
                self._transforms[name] = T
                print(f"[CoordinateTransformer] Loaded {filename}")
            except Exception as e:
                print(f"[WARN] Failed to load {filename}: {e}")

        # 计算派生变换
        if 'arm_to_optical' in self._transforms:
            self._transforms['optical_to_arm'] = np.linalg.inv(self._transforms['arm_to_optical'])
        if 'arm_to_rslidar' in self._transforms:
            self._transforms['rslidar_to_arm'] = np.linalg.inv(self._transforms['arm_to_rslidar'])
        if 'optical_to_base' in self._transforms:
            self._transforms['base_to_optical'] = np.linalg.inv(self._transforms['optical_to_base'])
        if 'rslidar_to_base' in self._transforms:
            self._transforms['base_to_rslidar'] = np.linalg.inv(self._transforms['rslidar_to_base'])

    def get_transform(self, name: str) -> np.ndarray:
        """获取指定的变换矩阵"""
        if name not in self._transforms:
            raise KeyError(f"Transform '{name}' not found. Available: {list(self._transforms.keys())}")
        return self._transforms[name]

    def add_transform(self, name: str, T: np.ndarray):
        """手动添加变换矩阵"""
        self._transforms[name] = T

    def transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        变换点云 (优化版: 避免额外内存分配)

        Args:
            points: (N, 3) 或 (N, 4) 点云数组
            T: 4x4 变换矩阵

        Returns:
            transformed: 变换后的点云，保持输入的列数
        """
        if len(points) == 0:
            return points

        # 提取旋转和平移 (避免创建齐次坐标)
        R = T[:3, :3]
        t = T[:3, 3]

        # 直接矩阵运算: xyz_t = xyz @ R.T + t
        xyz = points[:, :3]
        xyz_t = xyz @ R.T + t  # 广播加法

        # 保持原有的额外列 (如 intensity)
        if points.shape[1] > 3:
            result = np.hstack([xyz_t, points[:, 3:]])
        else:
            result = xyz_t

        return result

    def rslidar_to_optical(self, points: np.ndarray) -> np.ndarray:
        """rslidar → top_camera_optical_frame"""
        T = self.get_transform('rslidar_to_optical')
        return self.transform_points(points, T)

    def optical_to_arm(self, points: np.ndarray) -> np.ndarray:
        """top_camera_optical_frame → arm_base_link"""
        T = self.get_transform('optical_to_arm')
        return self.transform_points(points, T)

    def rslidar_to_arm(self, points: np.ndarray) -> np.ndarray:
        """rslidar → arm_base_link"""
        T = self.get_transform('rslidar_to_arm')
        return self.transform_points(points, T)

    def optical_to_base(self, points: np.ndarray) -> np.ndarray:
        """top_camera_optical_frame → base_link"""
        T = self.get_transform('optical_to_base')
        return self.transform_points(points, T)

    def rslidar_to_base(self, points: np.ndarray) -> np.ndarray:
        """rslidar → base_link"""
        T = self.get_transform('rslidar_to_base')
        return self.transform_points(points, T)

    def optical_to_target(self, points: np.ndarray, target_frame: str) -> np.ndarray:
        """将 optical 坐标系的点变换到目标坐标系

        Args:
            points: (N, 3) 点云，optical 坐标系
            target_frame: 'base_link' 或 'arm_base_link'

        Returns:
            (N, 3) 变换后的点云
        """
        if target_frame == 'base_link':
            return self.optical_to_base(points)
        elif target_frame == 'arm_base_link':
            return self.optical_to_arm(points)
        else:
            raise ValueError(f"Unsupported target frame: {target_frame}. Use: base_link, arm_base_link")

    def rslidar_to_target(self, points: np.ndarray, target_frame: str) -> np.ndarray:
        """将 rslidar 坐标系的点变换到目标坐标系

        Args:
            points: (N, 3) 点云，rslidar 坐标系
            target_frame: 'base_link' 或 'arm_base_link'

        Returns:
            (N, 3) 变换后的点云
        """
        if target_frame == 'base_link':
            return self.rslidar_to_base(points)
        elif target_frame == 'arm_base_link':
            return self.rslidar_to_arm(points)
        else:
            raise ValueError(f"Unsupported target frame: {target_frame}. Use: base_link, arm_base_link")

    def project_to_image(self, points_optical: np.ndarray, intrinsics: dict) -> np.ndarray:
        """
        将 optical_frame 中的点投影到图像平面

        Args:
            points_optical: (N, 3+) 点云，optical_frame 坐标系
            intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}

        Returns:
            uv: (N, 2) 图像坐标 [u, v]
        """
        if len(points_optical) == 0:
            return np.empty((0, 2))

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        X = points_optical[:, 0]
        Y = points_optical[:, 1]
        Z = points_optical[:, 2]

        # 避免除零
        Z_safe = np.where(Z > 0.01, Z, 0.01)

        u = fx * X / Z_safe + cx
        v = fy * Y / Z_safe + cy

        return np.column_stack([u, v])


def test_transformer():
    """测试坐标变换器"""
    print("=" * 60)
    print("CoordinateTransformer Test")
    print("=" * 60)

    transformer = CoordinateTransformer()

    # 测试手动添加变换
    T_identity = np.eye(4)
    transformer.add_transform('test', T_identity)
    print(f"Added identity transform: {transformer.get_transform('test')}")

    # 测试点变换
    test_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = transformer.transform_points(test_points, T_identity)
    print(f"Transformed points:\n{transformed}")

    print("\nTest completed")


if __name__ == '__main__':
    test_transformer()
