#!/usr/bin/env python3
"""
深度传感器处理模块 - 分离架构

感知线程 (15Hz): 深度图 → 点云 → RANSAC去地面 → 聚类 → 更新结果
控制线程 (50Hz): 只读取最新结果，不做计算

支持:
  1. 深度图投影到3D点云
  2. 外参变换到 base_link 坐标系
  3. RANSAC 地面拟合去除
  4. 简单聚类找物体
  5. 返回最近障碍物
"""

import time
import threading
import yaml
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from .config import ApproachConfig


class DepthSensor:
    """深度传感器处理器 - 分离架构

    感知线程 (15Hz):
      深度图 → 点云 → RANSAC去地面 → 聚类 → 更新结果

    控制线程调用:
      get_nearest_obstacle() - 直接返回最新结果，无计算
    """

    def __init__(self, node: Node, config: ApproachConfig):
        """初始化深度传感器

        Args:
            node: ROS2 节点实例
            config: 配置对象
        """
        self._node = node
        self._config = config
        self._running = True

        # ========== 深度数据 (订阅回调写入) ==========
        self._depth_lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_timestamp: float = 0.0
        self._bridge = CvBridge()

        # ========== 感知结果 (感知线程写入，控制线程读取) ==========
        self._result_lock = threading.Lock()
        self._nearest_obstacle: Optional[Tuple[float, float, float]] = None
        self._obstacle_points: Optional[np.ndarray] = None
        self._clusters: List[np.ndarray] = []
        self._result_timestamp: float = 0.0

        # ========== 相机内参 ==========
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._has_intrinsics = False

        # ========== 相机外参 (optical_frame -> base_link) ==========
        self._R_cam_to_base: Optional[np.ndarray] = None
        self._t_cam_to_base: Optional[np.ndarray] = None
        self._has_extrinsics = False
        self._load_extrinsics()

        # ========== ROS 订阅 ==========
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._sub = node.create_subscription(
            Image,
            config.depth_topic,
            self._depth_callback,
            sensor_qos
        )

        info_topic = config.depth_topic.replace('/image_raw', '/camera_info')
        self._info_sub = node.create_subscription(
            CameraInfo,
            info_topic,
            self._camera_info_callback,
            sensor_qos
        )

        # ========== 启动感知线程 ==========
        self._perception_thread = threading.Thread(
            target=self._perception_loop,
            daemon=True,
            name="depth_perception"
        )
        self._perception_thread.start()

        self._node.get_logger().info(
            f"深度传感器初始化完成 (分离架构):\n"
            f"  感知线程: 25Hz, 降采样4x\n"
            f"  地面过滤: 高度预过滤 + RANSAC (支持相机倾斜)\n"
            f"  检测宽度: ±{config.depth_detect_width}m\n"
            f"  外参: {'已加载' if self._has_extrinsics else '未加载'}"
        )

    def shutdown(self):
        """关闭感知线程"""
        self._running = False

    # =========================================================================
    # 感知线程
    # =========================================================================

    def _perception_loop(self):
        """感知线程主循环 (25Hz)

        处理流程:
        1. 获取最新深度图
        2. 转换为点云
        3. RANSAC 去除地面
        4. 聚类
        5. 找最近障碍物
        6. 更新共享结果
        """
        perception_rate = 25  # Hz
        interval = 1.0 / perception_rate

        while self._running and rclpy.ok():
            start_time = time.time()

            try:
                self._do_perception()
            except Exception as e:
                self._node.get_logger().error(
                    f"感知线程错误: {e}",
                    throttle_duration_sec=2.0
                )

            # 保持固定频率
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _do_perception(self):
        """执行一次完整的感知处理"""
        # 1. 获取深度图
        with self._depth_lock:
            if self._latest_depth is None:
                return
            if time.time() - self._depth_timestamp > self._config.depth_data_timeout:
                return
            depth_image = self._latest_depth.copy()

        if not self._has_intrinsics:
            return

        # 2. 深度图转点云 (降采样4倍，加速处理)
        points = self._depth_to_pointcloud(depth_image, downsample=4)
        if points is None or len(points) < 100:
            return

        # 3. 过滤前方区域
        x_base = points[:, 0]
        y_base = points[:, 1]
        z_base = points[:, 2]

        # 前方区域基本过滤
        front_mask = (x_base > 0.1) & (x_base < self._config.depth_max_valid) & \
                     (np.abs(y_base) < self._config.depth_detect_width) & \
                     (z_base > -0.5) & (z_base < self._config.depth_obstacle_max_height)

        front_points = points[front_mask]
        if len(front_points) < 50:
            with self._result_lock:
                self._nearest_obstacle = None
                self._obstacle_points = None
                self._clusters = []
            return

        # 4. 两阶段地面去除：高度预过滤 + RANSAC精确拟合
        obstacle_points = self._remove_ground(front_points)
        if len(obstacle_points) < 20:
            with self._result_lock:
                self._nearest_obstacle = None
                self._obstacle_points = obstacle_points
                self._clusters = []
            return

        # 5. 聚类
        clusters = self._simple_clustering(obstacle_points)

        # 6. 找最近点
        nearest_point = None
        if clusters:
            nearest_dist = float('inf')
            for cluster in clusters:
                min_x = np.min(cluster[:, 0])
                if min_x < nearest_dist:
                    nearest_dist = min_x
                    min_idx = np.argmin(cluster[:, 0])
                    nearest_point = cluster[min_idx]
        elif len(obstacle_points) > 0:
            # 无聚类时用最近点
            min_idx = np.argmin(obstacle_points[:, 0])
            nearest_point = obstacle_points[min_idx]

        # 7. 更新共享结果
        with self._result_lock:
            if nearest_point is not None:
                self._nearest_obstacle = (
                    float(nearest_point[0]),
                    float(nearest_point[1]),
                    float(nearest_point[2])
                )
            else:
                self._nearest_obstacle = None
            self._obstacle_points = obstacle_points
            self._clusters = clusters
            self._result_timestamp = time.time()

    # =========================================================================
    # 公共接口 (控制线程调用，只读取结果)
    # =========================================================================

    def get_nearest_obstacle(self) -> Optional[Tuple[float, float, float]]:
        """获取最近障碍物信息 (只读取，无计算)

        Returns:
            (x, y, z): 最近障碍物在 base_link 坐标系中的位置
            x: 前向距离, y: 侧向距离 (左正右负), z: 高度
            如果无有效数据返回 None
        """
        with self._result_lock:
            # 检查结果是否过期 (200ms)
            if time.time() - self._result_timestamp > 0.2:
                return None
            return self._nearest_obstacle

    def get_target_depth(self) -> Optional[float]:
        """获取前方最近障碍物深度

        Returns:
            float: 最近障碍物到 base_link 原点的前向距离 (米)
        """
        result = self.get_nearest_obstacle()
        if result is None:
            return None
        return result[0]

    def get_obstacle_points(self, max_points: int = 5000) -> Optional[np.ndarray]:
        """获取障碍物点云 (用于调试/可视化)

        Returns:
            np.ndarray: (N, 3) 点云，在 base_link 坐标系
        """
        with self._result_lock:
            if self._obstacle_points is None:
                return None
            points = self._obstacle_points.copy()

        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        return points

    def get_clusters(self) -> List[np.ndarray]:
        """获取聚类结果 (用于调试)

        Returns:
            List[np.ndarray]: 聚类列表
        """
        with self._result_lock:
            return [c.copy() for c in self._clusters]

    @property
    def has_data(self) -> bool:
        """检查是否有深度数据"""
        with self._depth_lock:
            return self._latest_depth is not None

    @property
    def has_intrinsics(self) -> bool:
        """检查是否有相机内参"""
        return self._has_intrinsics

    @property
    def has_extrinsics(self) -> bool:
        """检查是否有相机外参"""
        return self._has_extrinsics

    # =========================================================================
    # ROS 回调
    # =========================================================================

    def _depth_callback(self, msg: Image):
        """深度图回调"""
        try:
            depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            with self._depth_lock:
                self._latest_depth = depth_image.astype(np.float32)
                if msg.encoding == '16UC1':
                    self._latest_depth = self._latest_depth / 1000.0
                self._depth_timestamp = time.time()

        except Exception as e:
            self._node.get_logger().error(f"深度回调错误: {e}")

    def _camera_info_callback(self, msg: CameraInfo):
        """相机内参回调"""
        if not self._has_intrinsics:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]
            self._has_intrinsics = True
            self._node.get_logger().info(
                f"相机内参: fx={self._fx:.1f}, fy={self._fy:.1f}, "
                f"cx={self._cx:.1f}, cy={self._cy:.1f}"
            )

    # =========================================================================
    # 内部算法
    # =========================================================================

    def _load_extrinsics(self):
        """从标定文件加载外参"""
        try:
            path = Path(self._config.extrinsics_file)
            if not path.exists():
                self._node.get_logger().warn(f"外参文件不存在: {path}")
                return

            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            transform = data['transform']
            trans = transform['translation']
            rot = transform['rotation']

            t_file = np.array([trans['x'], trans['y'], trans['z']])
            q_file = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
            R_file = Rotation.from_quat(q_file).as_matrix()

            # 求逆变换: P_base = R^T * P_optical - R^T * t
            self._R_cam_to_base = R_file.T
            self._t_cam_to_base = -R_file.T @ t_file

            self._has_extrinsics = True
            self._node.get_logger().info(
                f"外参已加载: t=[{self._t_cam_to_base[0]:.3f}, "
                f"{self._t_cam_to_base[1]:.3f}, {self._t_cam_to_base[2]:.3f}]"
            )

        except Exception as e:
            self._node.get_logger().error(f"加载外参失败: {e}")

    def _depth_to_pointcloud(self, depth_image: np.ndarray, downsample: int = 1) -> Optional[np.ndarray]:
        """深度图转点云 (base_link 坐标系)"""
        if not self._has_intrinsics:
            return None

        h, w = depth_image.shape[:2]

        if downsample > 1:
            depth_ds = depth_image[::downsample, ::downsample]
            v_indices = np.arange(0, h, downsample)
            u_indices = np.arange(0, w, downsample)
        else:
            depth_ds = depth_image
            v_indices = np.arange(h)
            u_indices = np.arange(w)

        u_grid, v_grid = np.meshgrid(u_indices, v_indices)
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        d_flat = depth_ds.flatten()

        valid_mask = (d_flat > self._config.depth_min_valid) & \
                     (d_flat < self._config.depth_max_valid)

        if not np.any(valid_mask):
            return None

        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        d_valid = d_flat[valid_mask]

        # 投影到相机坐标系
        x_cam = (u_valid - self._cx) * d_valid / self._fx
        y_cam = (v_valid - self._cy) * d_valid / self._fy
        z_cam = d_valid
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # 变换到 base_link
        if self._has_extrinsics:
            points_base = (self._R_cam_to_base @ points_cam.T).T + self._t_cam_to_base
        else:
            points_base = np.stack([z_cam, -x_cam, -y_cam], axis=1)

        return points_base

    def _remove_ground(self, points: np.ndarray) -> np.ndarray:
        """RANSAC地面去除，保留低矮障碍物（如躺倒的瓶子）

        基于实测数据:
        - 地面z值: 约 -0.156m (主要在 -0.20 ~ -0.15m)
        - 躺倒瓶子: z约 -0.10m ~ -0.07m (需要保留)
        - 站立障碍物: z > -0.05m

        策略: 主要依赖RANSAC判断平面，而不是简单高度阈值

        Args:
            points: (N, 3) 点云 in base_link

        Returns:
            去除地面后的障碍物点云
        """
        if len(points) < 20:
            return points

        z_base = points[:, 2]

        # 阶段1: 高度预过滤
        # 明显的高处物体直接保留 (z > 0m 肯定不是地面)
        high_threshold = 0.0
        high_mask = z_base > high_threshold
        high_points = points[high_mask]

        # 低处点需要RANSAC判断 (包括地面和低矮障碍物)
        low_mask = z_base <= high_threshold
        low_points = points[low_mask]

        if len(low_points) < 20:
            return high_points if len(high_points) > 0 else points

        # 阶段2: RANSAC拟合地面平面
        ground_inliers = self._ransac_fit_ground(low_points)

        if ground_inliers is not None:
            # 去除地面点，保留非地面点（包括低矮障碍物）
            non_ground_low = low_points[~ground_inliers]
        else:
            # RANSAC失败，用保守高度阈值 (只去除最低的点)
            non_ground_low = low_points[low_points[:, 2] > -0.14]

        # 合并
        if len(high_points) > 0 and len(non_ground_low) > 0:
            return np.vstack([high_points, non_ground_low])
        elif len(high_points) > 0:
            return high_points
        else:
            return non_ground_low

    def _ransac_fit_ground(self, points: np.ndarray,
                           distance_threshold: float = 0.02,
                           max_iterations: int = 100,
                           min_inlier_ratio: float = 0.2) -> np.ndarray:
        """RANSAC拟合地面平面

        考虑相机可能有俯仰角，放宽法向量约束。

        Args:
            points: 低处点云
            distance_threshold: 点到平面距离阈值
            max_iterations: 最大迭代次数
            min_inlier_ratio: 最小内点比例

        Returns:
            地面点的布尔掩码，失败返回None
        """
        n_points = len(points)
        if n_points < 10:
            return None

        best_inliers = None
        best_count = 0

        for _ in range(max_iterations):
            # 随机选3个点
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # 计算平面法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # 确保法向量朝上 (z分量为正)
            if normal[2] < 0:
                normal = -normal

            # 法向量约束：考虑相机俯仰，允许±30度倾斜
            # cos(30°) ≈ 0.866，放宽到0.7允许更大倾斜
            if normal[2] < 0.7:
                continue

            # 计算所有点到平面的距离
            d = np.dot(normal, p1)
            distances = np.abs(np.dot(points, normal) - d)
            inliers = distances < distance_threshold
            count = np.sum(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers

        # 检查内点比例是否足够
        if best_inliers is None or best_count < n_points * min_inlier_ratio:
            return None

        return best_inliers

    def _simple_clustering(self, points: np.ndarray,
                            distance_threshold: float = 0.15,
                            min_points: int = 20) -> List[np.ndarray]:
        """简单欧式聚类 (基于X距离分段)"""
        if len(points) < min_points:
            return []

        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]

        clusters = []
        cluster_start = 0

        for i in range(1, len(sorted_points)):
            if sorted_points[i, 0] - sorted_points[i-1, 0] > distance_threshold:
                if i - cluster_start >= min_points:
                    clusters.append(sorted_points[cluster_start:i])
                cluster_start = i

        if len(sorted_points) - cluster_start >= min_points:
            clusters.append(sorted_points[cluster_start:])

        return clusters
