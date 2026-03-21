#!/usr/bin/env python3
"""
深度传感器处理模块

相机坐标系 (D435 标准):
  - x: right  (右为正)
  - y: down   (向下为正, 地面 y ≈ +0.15m, 空中 y < 0)
  - z: forward (前向深度)

相机离地约 15cm → 地面在 y ≈ +0.15m

感知线程 (30Hz): 深度图 → 点云 → ROI过滤 → RANSAC去地面 → 聚类 → 更新结果
控制线程:        只读最新结果, 不做计算
"""

import time
import threading
from collections import deque
from typing import Optional, Tuple, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import StaticTransformBroadcaster

from .config import ApproachConfig


class DepthSensor:
    """深度传感器处理器"""

    def __init__(self, node: Node, config: ApproachConfig,
                 tf_buffer: tf2_ros.Buffer = None):
        self._node = node
        self._config = config
        self._tf_buffer = tf_buffer
        self._running = True
        self._active = threading.Event()   # 仅在阶段3激活时 set()

        # ========== 深度数据 (回调写入) ==========
        self._depth_lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_timestamp: float = 0.0
        self._bridge = CvBridge()

        # ========== 感知结果 (感知线程写入, 控制线程读取) ==========
        self._result_lock = threading.Lock()
        self._nearest_obstacle: Optional[Tuple[float, float, float]] = None
        self._obstacle_points: Optional[np.ndarray] = None
        self._front_points: Optional[np.ndarray] = None
        self._ground_points: Optional[np.ndarray] = None
        self._clusters: List[np.ndarray] = []
        self._result_timestamp: float = 0.0
        self._processing_time_ms: float = 0.0

        # ========== 相机内参 ==========
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._has_intrinsics = False
        self._camera_frame_id: str = 'camera_optical_frame'  # 由 CameraInfo 更新

        # ========== 点云生成缓存 (避免每帧重建 meshgrid) ==========
        self._grid_cache_key: Optional[Tuple] = None
        self._u_flat_cache: Optional[np.ndarray] = None
        self._v_flat_cache: Optional[np.ndarray] = None

        # ========== 感知日志控制 ==========
        self._log_enabled: bool = True   # 可在 input() 期间关闭，避免淹没终端

        # ========== 调试点云发布节流 (2Hz, 避免 Jetson 过载) ==========
        self._debug_publish_interval: float = 0.5  # 500ms
        self._last_debug_publish: float = 0.0

        # ========== 聚类后端检测 ==========
        self._use_sklearn = False
        try:
            from sklearn.cluster import DBSCAN  # noqa: F401
            self._use_sklearn = True
            node.get_logger().info("sklearn DBSCAN 可用 → 高性能聚类")
        except ImportError:
            node.get_logger().warn(
                "sklearn 不可用, 使用 KDTree-BFS 聚类\n"
                "  建议安装: pip install scikit-learn"
            )

        # ========== ROS 订阅 (按需创建，activate 时启用) ==========
        self._sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._depth_topic = config.depth_topic
        self._info_topic = config.depth_topic.replace('/image_raw', '/camera_info')
        self._sub = None       # activate() 时创建
        self._info_sub = None  # activate() 时创建

        # ========== 调试点云发布 (已禁用，太卡) ==========
        # self._debug_front_pub    = node.create_publisher(PointCloud2, '~/debug_front_cloud', 1)
        # self._debug_obstacle_pub = node.create_publisher(PointCloud2, '~/debug_obstacle_cloud', 1)
        # self._debug_ground_pub   = node.create_publisher(PointCloud2, '~/debug_ground_cloud', 1)
        # self._debug_cluster_pub  = node.create_publisher(PointCloud2, '~/debug_cluster_cloud', 1)

        # ========== 桥接相机驱动 TF 孤岛到机器人 TF 树 ==========
        # RealSense 驱动以 chassis_link 为根发布 TF，但机器人 URDF 用 chassis_camera_link
        # 发布静态 TF: chassis_link → chassis_camera_link (identity)
        # 完整链: chassis_color_optical_frame → chassis_color_frame → chassis_link
        #          → chassis_camera_link → lidar_link → base_link
        self._static_tf_pub = StaticTransformBroadcaster(node)
        bridge = TransformStamped()
        bridge.header.stamp = node.get_clock().now().to_msg()
        bridge.header.frame_id = 'chassis_camera_link'
        bridge.child_frame_id  = 'chassis_link'
        bridge.transform.rotation.w = 1.0
        self._static_tf_pub.sendTransform(bridge)

        # ========== 启动感知线程 ==========
        self._perception_thread = threading.Thread(
            target=self._perception_loop, daemon=True, name="depth_perception"
        )
        self._perception_thread.start()

        self._node.get_logger().info(
            f"深度传感器初始化:\n"
            f"  坐标系: x=右 y=下(地面y≈+{config.depth_ground_max_height}m) z=前\n"
            f"  ROI y范围: [{config.depth_obstacle_min_height}, {config.depth_obstacle_max_height}]m\n"
            f"  降采样: {config.depth_downsample}x  感知频率: 30Hz"
        )

    def shutdown(self):
        self._running = False
        self._active.clear()
        if self._perception_thread.is_alive():
            self._perception_thread.join(timeout=1.0)
        # 仅在 shutdown 时销毁订阅 (不在 deactivate 中销毁，避免 DDS 重匹配失败)
        if self._sub is not None:
            self._node.destroy_subscription(self._sub)
            self._sub = None
        if self._info_sub is not None:
            self._node.destroy_subscription(self._info_sub)
            self._info_sub = None

    # =========================================================================
    # 感知线程 (30Hz)
    # =========================================================================

    def activate(self):
        """激活感知线程 + 订阅 (阶段2/3开始时调用)

        订阅只创建一次，后续调用仅恢复感知线程。
        避免反复 destroy/create 导致 DDS 重匹配失败。
        """
        if self._sub is None:
            self._sub = self._node.create_subscription(
                Image, self._depth_topic, self._depth_callback, self._sensor_qos)
        if self._info_sub is None:
            self._info_sub = self._node.create_subscription(
                CameraInfo, self._info_topic, self._camera_info_callback, self._sensor_qos)
        self._active.set()
        self._node.get_logger().info("深度感知已激活")

    def deactivate(self):
        """暂停感知线程 (保留订阅，避免 DDS 重匹配问题)"""
        self._active.clear()
        self._clear_results()
        with self._depth_lock:
            self._latest_depth = None
        self._node.get_logger().info("深度感知已暂停")

    def _perception_loop(self):
        interval = 1.0 / 30.0
        while self._running and rclpy.ok():
            if not self._active.wait(timeout=0.5):
                continue
            t0 = time.time()
            try:
                self._do_perception()
            except Exception as e:
                self._node.get_logger().error(f"感知线程错误: {e}", throttle_duration_sec=1.0)
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    def _do_perception(self):
        t0 = time.time()

        # 1. 获取深度图
        with self._depth_lock:
            if self._latest_depth is None:
                self._node.get_logger().info(
                    "[depth] 无深度帧", throttle_duration_sec=2.0)
                return
            if time.time() - self._depth_timestamp > self._config.depth_data_timeout:
                self._node.get_logger().info(
                    "[depth] 深度帧超时", throttle_duration_sec=2.0)
                return
            depth_image = self._latest_depth.copy()

        if not self._has_intrinsics:
            self._node.get_logger().info(
                "[depth] 等待相机内参...", throttle_duration_sec=2.0)
            return

        t1 = time.time()

        # 2. 深度图 → 相机坐标系点云
        points = self._depth_to_pointcloud(depth_image, self._config.depth_downsample)
        if points is None or len(points) < 100:
            self._node.get_logger().info(
                f"[depth] 点云不足: {0 if points is None else len(points)}点 (<100)",
                throttle_duration_sec=2.0)
            self._clear_results()
            return

        t2 = time.time()

        # 3. ROI 过滤
        # y 向下为正: 地面 y ≈ +0.15m (相机离地15cm)
        # min_height=-0.50 (物体最高50cm超出相机), max_height=+0.22 (含地面供RANSAC)
        xc, yc, zc = points[:, 0], points[:, 1], points[:, 2]
        roi_mask = (
            (zc > self._config.depth_min_valid) &
            (zc < self._config.depth_max_valid) &
            (np.abs(xc) < self._config.depth_detect_width) &
            (yc > self._config.depth_obstacle_min_height) &
            (yc < self._config.depth_obstacle_max_height)
        )
        front_points = points[roi_mask]
        if len(front_points) < 50:
            self._node.get_logger().info(
                f"[depth] ROI过滤后不足: total={len(points)} roi={len(front_points)} (<50)",
                throttle_duration_sec=2.0)
            self._clear_results()
            return

        t3 = time.time()

        # 3.5 去除孤立噪点 (RealSense 飞点 / 深度不连续处错误点)
        front_points = self._remove_outliers(front_points)
        if len(front_points) < 50:
            self._node.get_logger().info(
                f"[depth] 去噪后不足: {len(front_points)}点 (<50)",
                throttle_duration_sec=2.0)
            self._clear_results()
            return

        # 4. RANSAC 地面去除
        obstacle_points, ground_points = self._remove_ground(front_points)

        t4 = time.time()

        # 5. 聚类
        clusters = self._cluster(obstacle_points)

        t5 = time.time()

        # 6. 最近障碍物 (前沿 min-z 作为距离)
        nearest = self._find_nearest(clusters, obstacle_points)

        t6 = time.time()

        # 7. 更新结果
        with self._result_lock:
            self._front_points    = front_points
            self._obstacle_points = obstacle_points if len(obstacle_points) > 0 else None
            self._ground_points   = ground_points   if len(ground_points)   > 0 else None
            self._clusters        = clusters
            self._nearest_obstacle = nearest
            self._result_timestamp = time.time()
            self._processing_time_ms = (time.time() - t0) * 1000

        # 8. 调试点云发布 (已禁用，太卡)
        # now_mono = time.time()
        # if now_mono - self._last_debug_publish >= self._debug_publish_interval:
        #     self._last_debug_publish = now_mono
        #     self._publish_debug_clouds(front_points, obstacle_points, ground_points, clusters)

        t7 = time.time()
        total_ms = (t7 - t0) * 1000

        if self._log_enabled:
            nz = f"nearest_z={nearest[2]:.3f}m" if nearest else "nearest=None"
            self._node.get_logger().info(
                f"[depth] front={len(front_points)} obs={len(obstacle_points)} "
                f"gnd={len(ground_points)} cls={len(clusters)} {nz}  "
                f"{total_ms:.0f}ms",
                throttle_duration_sec=2.0
            )

    def _clear_results(self):
        with self._result_lock:
            self._nearest_obstacle = None
            self._obstacle_points  = None
            self._front_points     = None
            self._ground_points    = None
            self._clusters         = []

    # =========================================================================
    # 点云生成
    # =========================================================================

    def _depth_to_pointcloud(self, depth_image: np.ndarray,
                             downsample: int = 4) -> Optional[np.ndarray]:
        """深度图 → 相机坐标系点云, 缓存 u/v 网格避免每帧重建"""
        if not self._has_intrinsics:
            return None

        h, w = depth_image.shape[:2]
        cache_key = (h, w, downsample)

        if self._grid_cache_key != cache_key:
            v_arr = np.arange(0, h, downsample)
            u_arr = np.arange(0, w, downsample)
            u_grid, v_grid = np.meshgrid(u_arr, v_arr)
            self._u_flat_cache = u_grid.ravel()
            self._v_flat_cache = v_grid.ravel()
            self._grid_cache_key = cache_key

        d_flat = depth_image[::downsample, ::downsample].ravel()

        valid = (d_flat > self._config.depth_min_valid) & \
                (d_flat < self._config.depth_max_valid)
        if not np.any(valid):
            return None

        d = d_flat[valid]
        u = self._u_flat_cache[valid]
        v = self._v_flat_cache[valid]

        x = (u - self._cx) * d / self._fx
        y = (v - self._cy) * d / self._fy   # y 向下为正
        z = d

        return np.column_stack([x, y, z])

    # =========================================================================
    # 地面去除 (RANSAC)
    # =========================================================================

    def _remove_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC 拟合地面平面并去除

        y 向下为正, 地面在 y ≈ +ground_max_height (+0.15m).
        potential_ground: y >= ground_max - 0.05 (接近地面高度的点)
        前提: ROI 上限 depth_obstacle_max_height > depth_ground_max_height
        """
        empty = np.empty((0, 3), dtype=np.float32)
        if len(points) < 50:
            return points, empty

        ground_max = abs(self._config.depth_ground_max_height)  # 确保正值
        margin = self._config.ground_candidate_margin
        yc = points[:, 1]

        potential_ground_mask = yc >= (ground_max - margin)
        potential_ground = points[potential_ground_mask]

        if len(potential_ground) < 30:
            # 地面点不足 (ROI 上限可能设置不够高), 全当障碍物
            return points, empty

        # RANSAC 输入上限 2000 点，避免大场景下性能下降
        if len(potential_ground) > 2000:
            idx = np.random.choice(len(potential_ground), 2000, replace=False)
            ransac_input = potential_ground[idx]
        else:
            ransac_input = potential_ground

        # RANSAC 拟合水平面，直接返回法向量+截距，不需要 SVD 二次拟合
        normal, d_plane = self._ransac_ground(
            ransac_input,
            distance_threshold=self._config.ransac_distance_threshold,
            max_iterations=self._config.ransac_max_iterations,
            min_inlier_ratio=self._config.ransac_min_inlier_ratio,
        )

        # 将平面参数应用到全量地面候选点
        if normal is not None:
            # 有符号距离: 正值=地面以下(或在地面上), 负值=地面以上(物体)
            # 只将位于地面平面 1cm 以上至 3cm 以下范围内的点视为地面,
            # 避免将低矮物体顶面(可能距地面仅 2-3cm)误判为地面点而去除
            signed = potential_ground @ normal - d_plane
            inliers = (signed >= self._config.ground_inlier_min_dist) & \
                       (signed < self._config.ground_inlier_max_dist)
        else:
            inliers = None

        if inliers is None:
            # RANSAC 失败: 回退到简单高度阈值
            ground_mask = yc >= ground_max
            return points[~ground_mask], points[ground_mask]

        ground_points   = potential_ground[inliers]
        near_non_ground = potential_ground[~inliers]
        far_points      = points[~potential_ground_mask]

        if len(near_non_ground) > 0 and len(far_points) > 0:
            obstacle_points = np.vstack([far_points, near_non_ground])
        elif len(far_points) > 0:
            obstacle_points = far_points
        else:
            obstacle_points = near_non_ground

        return obstacle_points, ground_points

    def _remove_outliers(self, points: np.ndarray) -> np.ndarray:
        """去除孤立噪点 (半径异常值去除)

        对每个点统计 radius 球内邻居数，不足 min_neighbors 的视为噪点。
        有效去除 RealSense 飞点 (flying pixels) 和深度不连续处的错误点。
        """
        radius = self._config.outlier_radius
        min_neighbors = self._config.outlier_min_neighbors
        if len(points) <= min_neighbors:
            return points
        try:
            from scipy.spatial import cKDTree
            # return_length=True 只返回数量，比返回索引快很多
            counts = cKDTree(points).query_ball_point(points, radius, return_length=True)
            # counts 包含点自身，所以用 > 而不是 >=
            return points[counts > min_neighbors]
        except ImportError:
            return points

    def _ransac_ground(self, points: np.ndarray,
                       distance_threshold: float = 0.03,
                       max_iterations: int = 50,
                       min_inlier_ratio: float = 0.3):
        """RANSAC 拟合水平面, 返回 (normal, d) 或 (None, None)"""
        n = len(points)
        if n < 10:
            return None, None

        best_normal = None
        best_d      = 0.0
        best_count  = 0

        for _ in range(max_iterations):
            idx = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = points[idx]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm

            if abs(normal[1]) < 0.7:
                continue
            if normal[1] < 0:
                normal = -normal   # 确保法向量指向+y方向(向下)，使有符号距离符号一致

            d     = float(np.dot(normal, p1))
            count = int(np.sum(np.abs(points @ normal - d) < distance_threshold))

            if count > best_count:
                best_count  = count
                best_normal = normal.copy()
                best_d      = d

        if best_normal is None or best_count < n * min_inlier_ratio:
            return None, None
        return best_normal, best_d

    # =========================================================================
    # 聚类
    # =========================================================================

    def _cluster(self, points: np.ndarray) -> List[np.ndarray]:
        tolerance = self._config.cluster_tolerance
        min_size = self._config.cluster_min_size
        max_size = self._config.cluster_max_size
        if len(points) < min_size:
            return []
        if self._use_sklearn:
            return self._cluster_dbscan(points, tolerance, min_size, max_size)
        try:
            from scipy.spatial import cKDTree  # noqa: F401
            return self._cluster_kdtree(points, tolerance, min_size, max_size)
        except ImportError:
            return []

    def _cluster_dbscan(self, points: np.ndarray, tolerance: float,
                        min_size: int, max_size: int) -> List[np.ndarray]:
        """sklearn DBSCAN (最快, C++ 实现)"""
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(
            eps=tolerance, min_samples=min_size,
            algorithm='ball_tree', n_jobs=1
        ).fit_predict(points)

        clusters = []
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = labels == label
            cnt  = int(np.sum(mask))
            if min_size <= cnt <= max_size:
                clusters.append(points[mask])
        return clusters

    def _cluster_kdtree(self, points: np.ndarray, tolerance: float,
                        min_size: int, max_size: int) -> List[np.ndarray]:
        """KDTree BFS 聚类 (scipy), deque 保证 O(n) BFS"""
        from scipy.spatial import cKDTree
        tree      = cKDTree(points)
        processed = np.zeros(len(points), dtype=bool)
        clusters  = []

        for i in range(len(points)):
            if processed[i]:
                continue
            cluster_idx: list = []
            q = deque([i])
            processed[i] = True

            while q:
                idx = q.popleft()
                cluster_idx.append(idx)
                if len(cluster_idx) > max_size:
                    break
                for nb in tree.query_ball_point(points[idx], tolerance):
                    if not processed[nb]:
                        processed[nb] = True
                        q.append(nb)

            if min_size <= len(cluster_idx) <= max_size:
                clusters.append(points[cluster_idx])

        return clusters

    # =========================================================================
    # 最近障碍物
    # =========================================================================

    def _find_nearest(self, clusters: List[np.ndarray],
                      fallback_points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """找最近障碍物并返回 (centroid_x, centroid_y, front_z)

        - 按聚类前沿 min-z 排序找最近聚类 (正确反映实际距离)
        - x/y 用质心 (用于侧向对齐), z 用前沿最小值 (到相机的真实距离)
        """
        if clusters:
            nearest  = min(clusters, key=lambda c: float(np.min(c[:, 2])))
            centroid = np.mean(nearest, axis=0)
            front_z  = float(np.min(nearest[:, 2]))
            return (float(centroid[0]), float(centroid[1]), front_z)

        if len(fallback_points) > 0:
            idx = int(np.argmin(fallback_points[:, 2]))
            p   = fallback_points[idx]
            return (float(p[0]), float(p[1]), float(p[2]))

        return None

    # =========================================================================
    # ROS 回调
    # =========================================================================

    def _depth_callback(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            ).astype(np.float32)
            if msg.encoding == '16UC1':
                img /= 1000.0
            with self._depth_lock:
                self._latest_depth    = img
                self._depth_timestamp = time.time()
        except Exception as e:
            self._node.get_logger().error(f"深度回调错误: {e}")

    def _camera_info_callback(self, msg: CameraInfo):
        if not self._has_intrinsics:
            self._fx, self._fy      = msg.k[0], msg.k[4]
            self._cx, self._cy      = msg.k[2], msg.k[5]
            self._camera_frame_id   = msg.header.frame_id  # 从消息取真实 frame_id
            self._has_intrinsics    = True
            self._node.get_logger().info(
                f"相机内参: fx={self._fx:.1f} fy={self._fy:.1f} "
                f"cx={self._cx:.1f} cy={self._cy:.1f}  "
                f"frame_id={self._camera_frame_id}"
            )

    # =========================================================================
    # 调试点云发布
    # =========================================================================

    def _transform_to_base_link(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        """将点云从相机坐标系变换到 base_link，返回 (transformed_points, frame_id)。
        若 TF 不可用则原样返回相机坐标系的点。"""
        if self._tf_buffer is None or not self._has_intrinsics:
            return points, self._camera_frame_id
        try:
            t = self._tf_buffer.lookup_transform(
                self._config.base_frame,
                self._camera_frame_id,
                rclpy.time.Time()
            )
            tr = t.transform.translation
            q  = t.transform.rotation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            # 四元数 → 旋转矩阵
            R = np.array([
                [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
                [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
                [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
            ], dtype=np.float32)
            t_vec = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
            transformed = (R @ points.T).T + t_vec
            return transformed, self._config.base_frame
        except Exception:
            return points, self._camera_frame_id

    # 聚类调试用颜色表 (最多支持 12 个聚类, 循环使用)
    _CLUSTER_COLORS = [
        (255, 200,   0), (  0, 180, 255), (255,  80, 200), (  0, 255, 180),
        (255, 120,   0), ( 80, 120, 255), (200, 255,   0), (255,   0, 100),
        (  0, 220, 100), (180,   0, 255), (255, 180, 100), (100, 255, 100),
    ]

    def _publish_debug_clouds(self, front: np.ndarray,
                              obstacle: np.ndarray,
                              ground: np.ndarray,
                              clusters: List[np.ndarray]):
        # 只在有订阅者时才序列化发布，避免无谓的 DDS 开销
        has_front    = self._debug_front_pub.get_subscription_count() > 0
        has_obstacle = self._debug_obstacle_pub.get_subscription_count() > 0
        has_ground   = self._debug_ground_pub.get_subscription_count() > 0
        has_cluster  = self._debug_cluster_pub.get_subscription_count() > 0

        if not (has_front or has_obstacle or has_ground or has_cluster):
            return

        stamp = self._node.get_clock().now().to_msg()

        if has_front and len(front) > 0:
            pts, frame = self._transform_to_base_link(front)
            self._publish_cloud(self._debug_front_pub,    pts, stamp, frame, (200, 200, 200))
        if has_obstacle and len(obstacle) > 0:
            pts, frame = self._transform_to_base_link(obstacle)
            self._publish_cloud(self._debug_obstacle_pub, pts, stamp, frame, (255,  50,  50))
        if has_ground and len(ground) > 0:
            pts, frame = self._transform_to_base_link(ground)
            self._publish_cloud(self._debug_ground_pub,   pts, stamp, frame, ( 50, 255,  50))
        if has_cluster and clusters:
            # 合并所有聚类，每个聚类染不同颜色
            parts = []
            for i, cluster in enumerate(clusters):
                r, g, b = self._CLUSTER_COLORS[i % len(self._CLUSTER_COLORS)]
                rgb_val = np.uint32((r << 16) | (g << 8) | b)
                pts = cluster.astype(np.float32)
                chunk = np.empty(len(pts),
                                 dtype=[('x','f4'),('y','f4'),('z','f4'),('rgb','u4')])
                chunk['x'] = pts[:, 0]
                chunk['y'] = pts[:, 1]
                chunk['z'] = pts[:, 2]
                chunk['rgb'] = rgb_val
                parts.append(chunk)
            merged = np.concatenate(parts)
            # 转换到 base_link
            xyz = np.column_stack([merged['x'], merged['y'], merged['z']])
            xyz_t, frame = self._transform_to_base_link(xyz)
            merged['x'] = xyz_t[:, 0]
            merged['y'] = xyz_t[:, 1]
            merged['z'] = xyz_t[:, 2]
            msg = PointCloud2()
            msg.header.stamp    = stamp
            msg.header.frame_id = frame
            msg.height   = 1
            msg.width    = len(merged)
            msg.is_dense = True
            msg.is_bigendian = False
            msg.fields = [
                PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
            ]
            msg.point_step = 16
            msg.row_step   = 16 * len(merged)
            msg.data       = merged.tobytes()
            self._debug_cluster_pub.publish(msg)

    def _publish_cloud(self, pub, points: np.ndarray, stamp, frame_id: str,
                       color: Tuple[int, int, int] = None):
        if len(points) == 0:
            return

        pts = points.astype(np.float32)
        msg = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.height          = 1
        msg.width           = len(pts)
        msg.is_bigendian    = False
        msg.is_dense        = True

        if color:
            # 标准 ROS RGB 打包: (R<<16)|(G<<8)|B 存为 uint32
            r, g, b   = color
            rgb_val   = np.uint32((r << 16) | (g << 8) | b)
            msg.fields = [
                PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
            ]
            msg.point_step = 16
            data = np.empty(len(pts),
                            dtype=[('x','f4'), ('y','f4'), ('z','f4'), ('rgb','u4')])
            data['x']   = pts[:, 0]
            data['y']   = pts[:, 1]
            data['z']   = pts[:, 2]
            data['rgb'] = rgb_val
            msg.data = data.tobytes()
        else:
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 12
            msg.data = pts.tobytes()

        msg.row_step = msg.point_step * msg.width
        pub.publish(msg)

    # =========================================================================
    # 公共接口 (线程安全)
    # =========================================================================

    def get_nearest_obstacle(self) -> Optional[Tuple[float, float, float]]:
        """获取最近障碍物 (centroid_x, centroid_y, front_z)

        超时设为 1.5s，允许感知线程在重负载下仍能提供有效数据
        """
        with self._result_lock:
            if time.time() - self._result_timestamp > 1.5:
                return None
            return self._nearest_obstacle

    def get_all_cluster_centroids(self) -> List[Tuple[float, float, float]]:
        """获取所有聚类的质心信息 (用于阶段2目标选择)

        Returns:
            list of (centroid_x, centroid_y, front_z)，空列表表示无数据
        """
        with self._result_lock:
            if time.time() - self._result_timestamp > 1.5:
                return []
            result = []
            for cluster in self._clusters:
                centroid = np.mean(cluster, axis=0)
                front_z = float(np.min(cluster[:, 2]))
                result.append((float(centroid[0]), float(centroid[1]), front_z))
            return result

    def get_target_depth(self) -> Optional[float]:
        """获取到最近障碍物前沿的距离 (相机 z 轴, 即到底盘相机的距离)"""
        obs = self.get_nearest_obstacle()
        return obs[2] if obs else None

    def get_obstacle_points(self, max_points: int = 5000) -> Optional[np.ndarray]:
        with self._result_lock:
            if self._obstacle_points is None:
                return None
            pts = self._obstacle_points.copy()
        if len(pts) > max_points:
            pts = pts[np.random.choice(len(pts), max_points, replace=False)]
        return pts

    @property
    def has_data(self) -> bool:
        with self._depth_lock:
            return self._latest_depth is not None

    @property
    def has_intrinsics(self) -> bool:
        return self._has_intrinsics

    def set_logging_enabled(self, enabled: bool):
        """控制感知线程日志输出 (input() 期间可关闭, 避免淹没终端)"""
        self._log_enabled = enabled
