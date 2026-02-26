#!/usr/bin/env python3
"""
坐标变换工具

加载外参文件，执行坐标系间的变换。
支持: rslidar, top_camera_optical_frame, arm_base_link
"""

import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation


class CoordinateTransformer:
    """坐标变换器，管理多个坐标系间的变换关系"""

    def __init__(self, config_dir=None):
        """
        Args:
            config_dir: 外参配置文件目录，默认 perception/config/
        """
        if config_dir is None:
            # 默认: src/coordinate_transformer.py → ../config/
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        self.config_dir = os.path.abspath(config_dir)

        # 存储变换矩阵
        self._transforms = {}

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
        if not os.path.isabs(yaml_path):
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

    def load_all_extrinsics(self):
        """加载所有需要的外参文件

        外参使用链式变换，支持多个目标坐标系:
        - arm_base_link: 机械臂基座
        - base_link: 底盘中心 (默认)

        变换链:
        - optical → arm_base_link
        - optical → base_link
        - rslidar → arm_base_link
        - rslidar → base_link
        """
        extrinsics_files = {
            # 原有变换
            'rslidar_to_optical': 'extrinsics_rslidar_to_top_camera_optical_frame.yaml',
            'arm_to_rslidar': 'extrinsics_arm_base_link_to_rslidar.yaml',
            'arm_to_optical': 'extrinsics_arm_base_link_to_top_camera_optical_frame.yaml',
            # 新增 base_link 变换
            'optical_to_base': 'extrinsics_top_camera_optical_frame_to_base_link.yaml',
            'rslidar_to_base': 'extrinsics_rslidar_to_base_link.yaml',
        }

        for name, filename in extrinsics_files.items():
            try:
                T = self.load_extrinsics(filename)
                self._transforms[name] = T
                print(f"[CoordinateTransformer] 已加载 {filename}")
            except Exception as e:
                print(f"[WARN] 加载 {filename} 失败: {e}")

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
            raise KeyError(f"未找到变换 '{name}'，可用: {list(self._transforms.keys())}")
        return self._transforms[name]

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
            raise ValueError(f"不支持的目标坐标系: {target_frame}, 可用: base_link, arm_base_link")

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
            raise ValueError(f"不支持的目标坐标系: {target_frame}, 可用: base_link, arm_base_link")

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
    print("CoordinateTransformer 测试")
    print("=" * 60)

    transformer = CoordinateTransformer()

    # 加载所有外参
    transformer.load_all_extrinsics()

    # 显示变换矩阵
    print("\n可用的变换:")
    for name in transformer._transforms.keys():
        T = transformer.get_transform(name)
        t = T[:3, 3]
        R = Rotation.from_matrix(T[:3, :3])
        euler = R.as_euler('xyz', degrees=True)
        print(f"  {name}:")
        print(f"    平移: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
        print(f"    旋转: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]° (xyz)")

    # 测试点云变换
    print("\n测试点云变换:")

    # 假设一个在 rslidar 坐标系前方 1.2m 的点
    test_point = np.array([[1.2, 0.0, 0.0, 100]])  # [x, y, z, intensity]
    print(f"原始点 (rslidar): {test_point[0, :3]}")

    # rslidar → optical
    point_optical = transformer.rslidar_to_optical(test_point)
    print(f"变换后 (optical): {point_optical[0, :3]}")

    # rslidar → arm_base
    point_arm = transformer.rslidar_to_arm(test_point)
    print(f"变换后 (arm_base): {point_arm[0, :3]}")

    # rslidar → base_link (新增)
    if 'rslidar_to_base' in transformer._transforms:
        point_base = transformer.rslidar_to_base(test_point)
        print(f"变换后 (base_link): {point_base[0, :3]}")

    # 测试统一接口
    print("\n测试 *_to_target 统一接口:")
    for target in ['base_link', 'arm_base_link']:
        try:
            result = transformer.rslidar_to_target(test_point, target)
            print(f"  rslidar → {target}: {result[0, :3]}")
        except (KeyError, ValueError) as e:
            print(f"  rslidar → {target}: 失败 ({e})")

    # 测试投影
    print("\n测试投影到图像平面:")
    intrinsics = {'fx': 383.5, 'fy': 383.5, 'cx': 640.0, 'cy': 360.0}  # 假设 1280x720
    uv = transformer.project_to_image(point_optical, intrinsics)
    print(f"图像坐标: u={uv[0, 0]:.1f}, v={uv[0, 1]:.1f}")


if __name__ == '__main__':
    test_transformer()
