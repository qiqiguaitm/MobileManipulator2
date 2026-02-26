#!/usr/bin/env python3
"""
Perception RViz Node - 感知结果 RViz 可视化

发布:
- RGB-D 点云 (base_link 坐标系)
- 物体 Mask 3D 投影 (不同颜色)
- 距离标签 (Camera/LiDAR)
"""

import os
import sys
import struct
import threading
import numpy as np

# 添加 src 目录
_src_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_src_dir, '..', 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import cv2
import rospy
import message_filters
from common.logger import get_logger
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from perception.msg import Object3DArray

# 尝试导入 TrackedObject3DArray (可能未构建)
try:
    from perception.msg import TrackedObject3D, TrackedObject3DArray
    HAS_TRACKER_MSG = True
except ImportError:
    HAS_TRACKER_MSG = False
from coordinate_transformer import CoordinateTransformer


class PerceptionRVizNode:
    """感知结果 RViz 可视化节点"""

    # 物体颜色列表 (RGBA for 3D markers)
    COLORS = [
        (0.0, 1.0, 0.0, 0.8),   # 绿
        (1.0, 0.0, 0.0, 0.8),   # 红
        (0.0, 0.0, 1.0, 0.8),   # 蓝
        (1.0, 1.0, 0.0, 0.8),   # 黄
        (1.0, 0.0, 1.0, 0.8),   # 品红
        (0.0, 1.0, 1.0, 0.8),   # 青
        (1.0, 0.5, 0.0, 0.8),   # 橙
        (0.5, 0.0, 1.0, 0.8),   # 紫
        (0.0, 1.0, 0.5, 0.8),   # 青绿
        (1.0, 0.0, 0.5, 0.8),   # 玫红
    ]

    # BGR 颜色 (for 2D OpenCV drawing)
    COLORS_BGR = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 255, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128)
    ]

    # 类别到颜色的映射缓存 (保证同类别物体颜色一致)
    _category_color_cache = {}

    def __init__(self):
        rospy.init_node('perception_rviz_node')
        self.log = get_logger("PerceptRViz")

        # 参数
        self.camera_name = rospy.get_param('~camera_name', 'top')
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.publish_rgb_cloud = rospy.get_param('~publish_rgb_cloud', True)
        self.publish_rate = rospy.get_param('~publish_rate', 2.0)  # Hz
        self.depth_max = rospy.get_param('~depth_max', 5.0)  # m
        self.cloud_skip = rospy.get_param('~cloud_skip', 1)  # 降采样 (1=不降采样，配合 image_scale 使用)
        self.image_scale = rospy.get_param('~image_scale', 0.5)  # 图像缩放 (0.5=1/4点数，效果平滑)

        # 噪点过滤参数
        self.enable_outlier_filter = rospy.get_param('~enable_outlier_filter', True)  # 启用离群点过滤
        self.outlier_radius = rospy.get_param('~outlier_radius', 0.05)  # 邻域半径 (m)
        self.outlier_min_neighbors = rospy.get_param('~outlier_min_neighbors', 5)  # 最小邻居数

        # 深度预处理参数
        self.enable_depth_denoise = rospy.get_param('~enable_depth_denoise', True)  # 深度中值滤波
        self.depth_denoise_ksize = rospy.get_param('~depth_denoise_ksize', 5)  # 中值滤波核大小

        # 组件
        self._bridge = CvBridge()
        self._transformer = CoordinateTransformer()
        self._transformer.load_all_extrinsics()

        # 数据缓存
        self._intrinsics = None
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_objects = None
        self._cached_objects = None  # 优化: 缓存最后检测结果，避免消失
        self._data_lock = threading.Lock()

        # 发布器
        self.pub_rgb_cloud = rospy.Publisher(
            '~rgb_pointcloud', PointCloud2, queue_size=1
        )
        self.pub_object_clouds = rospy.Publisher(
            '~object_clouds', PointCloud2, queue_size=1
        )
        self.pub_object_markers = rospy.Publisher(
            '~object_markers', MarkerArray, queue_size=1
        )
        self.pub_distance_labels = rospy.Publisher(
            '~distance_labels', MarkerArray, queue_size=1
        )
        self.pub_lidar_colored = rospy.Publisher(
            '~lidar_colored', PointCloud2, queue_size=1
        )

        # 2D 图像可视化发布器 (合并自 perception_viz_node)
        self.pub_detection_image = rospy.Publisher(
            '/perception_viz/detection_image', Image, queue_size=1
        )
        if HAS_TRACKER_MSG:
            self.pub_tracking_image = rospy.Publisher(
                '/perception_viz/tracking_image', Image, queue_size=1
            )

        # LiDAR 距离着色参数
        self.lidar_topic = rospy.get_param('~lidar_topic', '/rslidar_points')  # 订阅 topic
        self.lidar_min_range = rospy.get_param('~lidar_min_range', 0.0)  # m
        self.lidar_max_range = rospy.get_param('~lidar_max_range', 10.0)  # m
        self.lidar_skip = rospy.get_param('~lidar_skip', 1)  # 降采样 (1=不降采样)
        self.lidar_publish_rate = rospy.get_param('~lidar_publish_rate', 10.0)  # Hz

        # LiDAR 处理节流
        self._last_lidar_time = rospy.Time(0)

        # 相机回调节流 (相机 15-30Hz，但我们只需要 publish_rate Hz)
        self._last_camera_time = rospy.Time(0)
        # 处理 publish_rate=0 的情况（perf_mode=0 时禁用实时可视化）
        if self.publish_rate > 0:
            self._camera_throttle_period = 1.0 / self.publish_rate  # 节流周期
            # 合理优化: 如果发布频率很低 (<0.5Hz)，适度放宽节流
            if self.publish_rate < 0.5:
                self._camera_throttle_period = max(self._camera_throttle_period, 1.5)  # 最少1.5秒间隔
        else:
            # publish_rate=0: 禁用相机实时可视化，但保留节点以可视化 service 结果
            self._camera_throttle_period = float('inf')  # 无限大，跳过所有相机回调
            self.log.info(" publish_rate=0, 禁用相机实时可视化（仅显示 service 结果）")

        # 获取相机内参
        self._get_camera_info()

        # 订阅相机
        self._setup_camera_subscribers()

        # 订阅检测结果
        self._objects_sub = rospy.Subscriber(
            '/scene_perception_3d/objects_3d',
            Object3DArray,
            self._objects_callback,
            queue_size=1
        )

        # 订阅跟踪结果 (如果可用)
        self._latest_tracked = None
        if HAS_TRACKER_MSG:
            self._tracked_sub = rospy.Subscriber(
                '/object_tracker/tracked_objects',
                TrackedObject3DArray,
                self._tracked_callback,
                queue_size=1
            )

        # 订阅 LiDAR 点云
        self._lidar_sub = rospy.Subscriber(
            self.lidar_topic,
            PointCloud2,
            self._lidar_callback,
            queue_size=1
        )

        # 定时发布
        if self.publish_rate > 0:
            self._timer = rospy.Timer(
                rospy.Duration(1.0 / self.publish_rate),
                self._publish_callback
            )
        else:
            # publish_rate=0: 禁用定时发布（仅通过回调发布）
            self._timer = None

        self.log.info("Node started")
        self.log.info(f"  Camera: {self.camera_name}")
        self.log.info(f"  Target frame: {self.target_frame}")
        self.log.info(f"  RGB-D: rate={self.publish_rate}Hz, scale={self.image_scale}")
        self.log.info(f"  2D viz: /perception_viz/detection_image" + (", /perception_viz/tracking_image" if HAS_TRACKER_MSG else ""))
        self.log.info(f"  LiDAR: {self.lidar_topic} -> ~lidar_colored")

    def _get_camera_info(self):
        """获取相机内参"""
        info_topic = f'/camera/{self.camera_name}/color/camera_info'
        self.log.info(f"Waiting for camera info: {info_topic}")

        try:
            info_msg = rospy.wait_for_message(info_topic, CameraInfo, timeout=10.0)
            self._intrinsics = {
                'fx': info_msg.K[0],
                'fy': info_msg.K[4],
                'cx': info_msg.K[2],
                'cy': info_msg.K[5],
                'width': info_msg.width,
                'height': info_msg.height,
            }
            self.log.info(f"Intrinsics: {self._intrinsics['width']}x{self._intrinsics['height']}")
        except rospy.ROSException as e:
            self.log.error(f"Get intrinsics failed: {e}")
            raise

    def _setup_camera_subscribers(self):
        """设置相机订阅"""
        color_topic = f'/camera/{self.camera_name}/color/image_raw'
        # 使用 CDM 优化后的深度图 (来自 scene_perception_3d)
        depth_topic = '/scene_perception_3d/optimized_depth'

        if self.publish_rate > 0:
            # 正常模式：使用 message_filters 同步 color 和 depth
            self._color_sub = message_filters.Subscriber(color_topic, Image)
            self._depth_sub = message_filters.Subscriber(depth_topic, Image)

            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub],
                queue_size=5,
                slop=2.0  # 增大到2.0s以适应scene_perception_3d的处理延迟
            )
            self._sync.registerCallback(self._camera_callback)
        else:
            # perf_mode=0：单独订阅，不同步时间戳（service 结果时间戳可能过时）
            self._color_sub = rospy.Subscriber(color_topic, Image, self._color_only_callback, queue_size=1)
            self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_only_callback, queue_size=1)

        self.log.info(f"Subscribe camera: {color_topic}")
        self.log.info(f"Subscribe optimized depth: {depth_topic}")

    def _color_only_callback(self, color_msg):
        """Color 图像独立回调 (perf_mode=0 使用)"""
        try:
            rgb = self._bridge.imgmsg_to_cv2(color_msg, 'passthrough')
            if color_msg.encoding == 'rgb8':
                rgb = rgb[:, :, ::-1]  # RGB -> BGR

            with self._data_lock:
                self._latest_rgb = rgb
        except Exception as e:
            self.log.warn(f"Color decode failed: {e}")

    def _depth_only_callback(self, depth_msg):
        """Depth 图像独立回调 (perf_mode=0 使用)"""
        try:
            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            # 深度预处理
            if self.enable_depth_denoise:
                valid_mask = depth_m > 0.1
                depth_m = cv2.medianBlur(depth_m, self.depth_denoise_ksize)
                depth_m[~valid_mask] = 0

            with self._data_lock:
                self._latest_depth = depth_m
        except Exception as e:
            self.log.warn(f"Depth decode failed: {e}")

    def _camera_callback(self, color_msg, depth_msg):
        """相机数据回调 (带节流优化)"""
        try:
            # 节流: 相机回调 15-30Hz，但我们只需要 2Hz
            # 但如果 throttle_period=inf（perf_mode=0），则不节流，接受所有同步的消息
            if self._camera_throttle_period < float('inf'):
                now = rospy.Time.now()
                dt = (now - self._last_camera_time).to_sec()
                if dt < self._camera_throttle_period:
                    return  # 跳过本次回调
                self._last_camera_time = now

            rgb = self._bridge.imgmsg_to_cv2(color_msg, 'passthrough')
            if color_msg.encoding == 'rgb8':
                rgb = rgb[:, :, ::-1]  # RGB -> BGR

            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            # 深度预处理: 中值滤波去除椒盐噪点
            if self.enable_depth_denoise:
                # 保存有效深度掩码
                valid_mask = depth_m > 0.1
                # 中值滤波 (对深度图有效区域)
                depth_m = cv2.medianBlur(depth_m, self.depth_denoise_ksize)
                # 恢复无效区域为 0
                depth_m[~valid_mask] = 0

            with self._data_lock:
                self._latest_rgb = rgb
                self._latest_depth = depth_m

        except Exception as e:
            self.log.warn(f"Camera data process failed: {e}")

    def _filter_outliers(self, points):
        """统计离群点过滤

        Args:
            points: (N, 3) 点云数组

        Returns:
            (M, 3) 过滤后的点云，M <= N
        """
        if not self.enable_outlier_filter or len(points) < self.outlier_min_neighbors:
            return points

        from scipy.spatial import cKDTree

        # 构建 KD-Tree
        tree = cKDTree(points)

        # 查询每个点的邻居数量
        neighbor_counts = tree.query_ball_point(points, self.outlier_radius, return_length=True)

        # 保留邻居数量 >= 阈值的点
        valid_mask = neighbor_counts >= self.outlier_min_neighbors

        return points[valid_mask]

    def _cluster_object_points(self, points, radius=0.05, min_neighbors=5):
        """快速统计过滤: 使用 KD-Tree 去除离群点 (比 DBSCAN 快 10x)

        Args:
            points: (N, 3) 点云数组
            radius: 邻域半径 (m)
            min_neighbors: 最小邻居数

        Returns:
            过滤后的点云
        """
        if len(points) < min_neighbors:
            return points

        from scipy.spatial import cKDTree

        # 随机降采样加速 KD-Tree 构建 (点数 > 2000 时)
        if len(points) > 2000:
            idx = np.random.choice(len(points), 2000, replace=False)
            sample_points = points[idx]
        else:
            sample_points = points
            idx = np.arange(len(points))

        # 构建 KD-Tree 并查询邻居数量
        tree = cKDTree(sample_points)
        counts = tree.query_ball_point(sample_points, radius, return_length=True)

        # 保留密集区域的点
        dense_mask = counts >= min_neighbors

        if len(points) > 2000:
            # 对原始点云，保留与密集采样点接近的点
            full_tree = cKDTree(points)
            dense_sample = sample_points[dense_mask]
            if len(dense_sample) > 0:
                # 查找原始点中在密集区域附近的点
                nearby = full_tree.query_ball_point(dense_sample, radius)
                # nearby 是列表的列表，需要展平
                all_indices = []
                for idx_list in nearby:
                    all_indices.extend(idx_list)
                if len(all_indices) > 0:
                    valid_idx = np.unique(all_indices)
                    return points[valid_idx]
            return points
        else:
            return points[dense_mask] if np.any(dense_mask) else points

    def _get_category_color(self, category):
        """根据类别获取固定颜色 (同类别物体颜色始终一致)

        Args:
            category: 物体类别字符串

        Returns:
            RGBA 颜色元组
        """
        if category not in self._category_color_cache:
            # 使用类别字符串的哈希值确定颜色索引
            color_idx = hash(category) % len(self.COLORS)
            self._category_color_cache[category] = self.COLORS[color_idx]
        return self._category_color_cache[category]

    def _get_category_color_bgr(self, category):
        """根据类别获取固定 BGR 颜色 (用于 2D 图像)"""
        color_idx = hash(category) % len(self.COLORS_BGR)
        return self.COLORS_BGR[color_idx]

    def _objects_callback(self, msg):
        """检测结果回调"""
        with self._data_lock:
            self._latest_objects = msg
            # 优化: 缓存最后的有效检测结果
            if msg is not None and len(msg.objects) > 0:
                self._cached_objects = msg
            rgb = self._latest_rgb

        # 发布 2D 检测图像
        if rgb is not None and self.pub_detection_image.get_num_connections() > 0:
            self._publish_detection_image(rgb, msg)

        # perf_mode=0: 无定时器，收到新检测需要延迟触发可视化
        # 延迟是为了让 latched 的 optimized_depth 有时间通过 message_filters 同步到相机回调
        if self.publish_rate <= 0 and msg is not None and len(msg.objects) > 0:
            rospy.Timer(rospy.Duration(0.5), lambda event: self._publish_callback(event), oneshot=True)

    def _tracked_callback(self, msg):
        """跟踪结果回调"""
        with self._data_lock:
            self._latest_tracked = msg
            rgb = self._latest_rgb

        # 发布 2D 跟踪图像
        if rgb is not None and HAS_TRACKER_MSG and self.pub_tracking_image.get_num_connections() > 0:
            self._publish_tracking_image(rgb, msg)

    def _publish_callback(self, event):
        """定时发布回调"""
        with self._data_lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            objects = self._latest_objects
            cached_objects = self._cached_objects

        if rgb is None or depth is None:
            self.log.debug_throttle(5.0, f"Data not ready: rgb={rgb is not None}, depth={depth is not None}")
            return

        stamp = rospy.Time.now()

        # 发布 RGB-D 点云
        if self.publish_rgb_cloud:
            self._publish_rgb_pointcloud(rgb, depth, stamp)

        # 发布物体可视化 (优化: 使用缓存避免消失)
        objects_to_visualize = objects if (objects is not None and len(objects.objects) > 0) else cached_objects
        if objects_to_visualize is not None and len(objects_to_visualize.objects) > 0:
            self.log.info_throttle(10.0, f"Visualizing {len(objects_to_visualize.objects)} objects, depth shape={depth.shape}")
            self._publish_object_visualization(objects_to_visualize, depth, stamp)
        else:
            self.log.debug_throttle(5.0, "No object detection results")

    def _publish_rgb_pointcloud(self, rgb, depth, stamp):
        """发布 RGB-D 点云 (优化: 图像缩放比点云降采样效果更平滑)"""
        # 图像缩放 (比点云降采样效果更平滑)
        scale = self.image_scale
        if scale < 1.0:
            import cv2
            new_h, new_w = int(depth.shape[0] * scale), int(depth.shape[1] * scale)
            depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # 调整内参
            fx = self._intrinsics['fx'] * scale
            fy = self._intrinsics['fy'] * scale
            cx = self._intrinsics['cx'] * scale
            cy = self._intrinsics['cy'] * scale
        else:
            depth_scaled = depth
            rgb_scaled = rgb
            fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
            cx, cy = self._intrinsics['cx'], self._intrinsics['cy']

        h, w = depth_scaled.shape

        # 点云降采样 (可选，配合 image_scale 使用)
        skip = self.cloud_skip
        ys, xs = np.mgrid[0:h:skip, 0:w:skip]
        ys = ys.flatten()
        xs = xs.flatten()

        # 获取深度
        ds = depth_scaled[ys, xs]
        valid = (ds > 0.1) & (ds < self.depth_max)
        ys_valid, xs_valid, ds_valid = ys[valid], xs[valid], ds[valid]

        # 反投影到 optical frame
        X = (xs_valid - cx) * ds_valid / fx
        Y = (ys_valid - cy) * ds_valid / fy
        Z = ds_valid

        points_optical = np.column_stack([X, Y, Z])

        # 变换到 base_link
        points_base = self._transformer.optical_to_target(points_optical, self.target_frame)

        # 离群点过滤
        if self.enable_outlier_filter and len(points_base) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(points_base)
            neighbor_counts = tree.query_ball_point(points_base, self.outlier_radius, return_length=True)
            outlier_mask = neighbor_counts >= self.outlier_min_neighbors
            points_base = points_base[outlier_mask]
            ys_valid = ys_valid[outlier_mask]
            xs_valid = xs_valid[outlier_mask]

        # 获取颜色 (使用缩放后的图像)
        colors = rgb_scaled[ys_valid, xs_valid]  # BGR

        # 创建 PointCloud2
        if len(points_base) > 0:
            cloud_msg = self._create_colored_pointcloud(points_base, colors, stamp, self.target_frame)
            self.pub_rgb_cloud.publish(cloud_msg)

    def _publish_object_visualization(self, objects_msg, depth, stamp):
        """发布物体可视化"""
        markers = MarkerArray()
        labels = MarkerArray()

        # 优化: 先清除旧 markers，避免残留
        delete_bbox = Marker()
        delete_bbox.action = Marker.DELETEALL
        delete_bbox.ns = "object_bbox"
        markers.markers.append(delete_bbox)

        delete_label = Marker()
        delete_label.action = Marker.DELETEALL
        delete_label.ns = "distance_labels"
        labels.markers.append(delete_label)

        all_object_points = []
        all_object_colors = []

        fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
        cx, cy = self._intrinsics['cx'], self._intrinsics['cy']

        for i, obj in enumerate(objects_msg.objects):
            # 根据类别获取固定颜色 (同类别物体颜色一致)
            color = self._get_category_color(obj.category)

            # 1. 创建边界框 Marker (立方体)
            bbox_marker = self._create_bbox_marker(obj, i, color, stamp)
            markers.markers.append(bbox_marker)

            # 2. 创建距离标签
            label_marker = self._create_distance_label(obj, i, stamp)
            labels.markers.append(label_marker)

            # 3. 生成物体 mask 点云 (基于 bbox 区域)
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # 限制范围
            h, w = depth.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 提取 bbox 区域的深度 (降采样: 每隔2像素)
            skip = 2
            ys_box, xs_box = np.mgrid[y1:y2:skip, x1:x2:skip]
            ys_flat = ys_box.flatten()
            xs_flat = xs_box.flatten()

            if len(ys_flat) == 0:
                continue

            ds_flat = depth[ys_flat, xs_flat]

            # 过滤有效深度
            valid = (ds_flat > 0.1) & (ds_flat < self.depth_max)

            if not np.any(valid):
                continue

            # 深度范围过滤: 基于中位数去除背景点
            depth_median = np.median(ds_flat[valid])
            depth_range = 0.5  # 保留中位数 ±0.5m 范围
            depth_mask = np.abs(ds_flat - depth_median) < depth_range
            final_mask = valid & depth_mask

            ys_v = ys_flat[final_mask]
            xs_v = xs_flat[final_mask]
            ds_v = ds_flat[final_mask]

            if len(ds_v) < 10:  # 至少 10 个点
                continue

            # 反投影到 3D
            X = (xs_v - cx) * ds_v / fx
            Y = (ys_v - cy) * ds_v / fy
            Z = ds_v
            pts = np.column_stack([X, Y, Z])

            # 简化聚类: 仅对大点云做过滤 (>1000点)
            if len(pts) > 1000:
                pts_filtered = self._cluster_object_points(pts)
            else:
                pts_filtered = pts

            if len(pts_filtered) > 0:
                all_object_points.append(pts_filtered)
                clr = np.array([[int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)]])
                all_object_colors.append(np.tile(clr, (len(pts_filtered), 1)))

        # 发布 markers
        self.pub_object_markers.publish(markers)
        self.pub_distance_labels.publish(labels)

        # 发布物体点云 (无缓存，避免重影)
        if len(all_object_points) > 0:
            points_optical = np.vstack(all_object_points)
            colors = np.vstack(all_object_colors).astype(np.uint8)

            self.log.info_throttle(10.0, f"Object pointcloud: {len(points_optical)} points")

            points_base = self._transformer.optical_to_target(points_optical, self.target_frame)
            cloud_msg = self._create_colored_pointcloud(points_base, colors, stamp, self.target_frame)
            self.pub_object_clouds.publish(cloud_msg)
        else:
            self.log.warn_throttle(10.0, "Object pointcloud empty")

    def _create_bbox_marker(self, obj, idx, color, stamp):
        """创建边界框 Marker"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = "object_bbox"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 位置 (物体质心)
        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z
        marker.pose.orientation.w = 1.0

        # 大小 (估计)
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        # 颜色 (半透明)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5

        # object_id (供 object_selector_node 使用)
        marker.text = obj.object_id

        marker.lifetime = rospy.Duration(3.0)  # 延长到3秒，覆盖0.5Hz发布周期
        return marker

    def _create_distance_label(self, obj, idx, stamp):
        """创建距离标签 Marker (合并版)"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = "distance_labels"
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # 位置 (物体上方)
        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z + 0.15  # 增加高度以容纳更多文字
        marker.pose.orientation.w = 1.0

        # 构建文字内容
        lines = [obj.object_id]

        # 第二行: base_link 距离
        cam_dist = f"dist:{obj.distance:.2f}m" if obj.distance > 0 else "dist:N/A"
        if obj.distance_lidar > 0:
            lidar_dist = f"L:{obj.distance_lidar:.2f}"
            lines.append(f"{cam_dist} {lidar_dist}")
        else:
            lines.append(cam_dist)

        # 第三行: 相机光学坐标系下的坐标和深度
        if hasattr(obj, 'depth') and obj.depth > 0:
            # 相机名称 + 深度
            cam_name = obj.source_camera if hasattr(obj, 'source_camera') and obj.source_camera else "cam"
            lines.append(f"[{cam_name}] z={obj.depth:.2f}m")

            # 第四行: optical 坐标 (x, y, z)
            if hasattr(obj, 'position_optical'):
                po = obj.position_optical
                lines.append(f"opt:({po.x:.2f},{po.y:.2f},{po.z:.2f})")

        marker.text = "\n".join(lines)

        # 屏幕固定大小: 根据距离动态缩放，远处字体更大以抵消透视
        base_scale = 0.022  # 稍微减小字体以容纳更多信息
        distance = max(obj.distance, 0.5) if obj.distance > 0 else 1.0
        marker.scale.z = base_scale * distance

        # 颜色 (白色)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(3.0)  # 延长到3秒，覆盖0.5Hz发布周期
        return marker

    def _create_colored_pointcloud(self, points, colors, stamp, frame_id):
        """创建带颜色的 PointCloud2 (向量化优化版)"""
        n_points = len(points)

        # 定义字段
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]

        # 向量化打包数据 (避免 Python for 循环!)
        # 创建结构化数组 (x, y, z, rgb)
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        cloud_arr = np.zeros(n_points, dtype=dtype)
        cloud_arr['x'] = points[:, 0].astype(np.float32)
        cloud_arr['y'] = points[:, 1].astype(np.float32)
        cloud_arr['z'] = points[:, 2].astype(np.float32)

        # 向量化 RGB 打包: (r << 16) | (g << 8) | b
        # colors 是 BGR 格式
        r = colors[:, 2].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 0].astype(np.uint32)
        cloud_arr['rgb'] = (r << 16) | (g << 8) | b

        # 创建消息
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = n_points
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * n_points
        msg.is_dense = True
        msg.data = cloud_arr.tobytes()  # 向量化转换！

        return msg

    def _distances_to_colors_vectorized(self, distances):
        """向量化: 将距离数组映射到颜色数组 (蓝→绿→黄→红)

        Args:
            distances: 距离数组 (N,) float32

        Returns:
            colors: RGB 颜色数组 (N, 3) uint8
        """
        # 归一化距离到 [0, 1]
        ratios = (distances - self.lidar_min_range) / (self.lidar_max_range - self.lidar_min_range)
        ratios = np.clip(ratios, 0.0, 1.0)

        n = len(ratios)
        colors = np.zeros((n, 3), dtype=np.uint8)

        # 蓝(0) → 青(0.25) → 绿(0.5) → 黄(0.75) → 红(1)
        # 段1: 蓝 → 青 [0, 0.25)
        mask1 = ratios < 0.25
        t1 = ratios[mask1] / 0.25
        colors[mask1, 0] = 0
        colors[mask1, 1] = (255 * t1).astype(np.uint8)
        colors[mask1, 2] = 255

        # 段2: 青 → 绿 [0.25, 0.5)
        mask2 = (ratios >= 0.25) & (ratios < 0.5)
        t2 = (ratios[mask2] - 0.25) / 0.25
        colors[mask2, 0] = 0
        colors[mask2, 1] = 255
        colors[mask2, 2] = (255 * (1 - t2)).astype(np.uint8)

        # 段3: 绿 → 黄 [0.5, 0.75)
        mask3 = (ratios >= 0.5) & (ratios < 0.75)
        t3 = (ratios[mask3] - 0.5) / 0.25
        colors[mask3, 0] = (255 * t3).astype(np.uint8)
        colors[mask3, 1] = 255
        colors[mask3, 2] = 0

        # 段4: 黄 → 红 [0.75, 1.0]
        mask4 = ratios >= 0.75
        t4 = (ratios[mask4] - 0.75) / 0.25
        colors[mask4, 0] = 255
        colors[mask4, 1] = (255 * (1 - t4)).astype(np.uint8)
        colors[mask4, 2] = 0

        return colors

    def _lidar_callback(self, msg):
        """LiDAR 点云回调 - 添加距离颜色 (优化版)"""
        try:
            # 频率节流 (减少 CPU 负载)
            # lidar_publish_rate=0 时禁用 LiDAR 可视化
            if self.lidar_publish_rate <= 0:
                return

            now = rospy.Time.now()
            dt = (now - self._last_lidar_time).to_sec()
            if dt < 1.0 / self.lidar_publish_rate:
                return
            self._last_lidar_time = now

            # 快速解析点云 (避免 for 循环)
            import sensor_msgs.point_cloud2 as pc2

            # 使用 read_points_numpy (如果可用) 或向量化读取
            points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

            if len(points_list) == 0:
                return

            # 降采样 (取每 N 个点)
            if self.lidar_skip > 1:
                points_list = points_list[::self.lidar_skip]

            points = np.array(points_list, dtype=np.float32)

            # 向量化: 计算径向距离
            distances = np.linalg.norm(points, axis=1)

            # 向量化: 计算颜色
            colors = self._distances_to_colors_vectorized(distances)

            # 创建带颜色的点云
            colored_cloud = self._create_colored_pointcloud(
                points, colors, msg.header.stamp, msg.header.frame_id
            )

            # 发布
            self.pub_lidar_colored.publish(colored_cloud)

        except Exception as e:
            self.log.warn_throttle(10.0, f"LiDAR coloring failed: {e}")

    # ==================== 2D 图像可视化 (合并自 perception_viz_node) ====================

    def _publish_detection_image(self, rgb, msg):
        """发布 2D 检测可视化图像"""
        try:
            img = rgb.copy()
            if msg is not None and len(msg.objects) > 0:
                img = self._draw_detections_2d(img, msg.objects)
            else:
                # 无检测结果时显示提示
                cv2.putText(img, "Detection: No objects", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = rospy.Time.now()
            self.pub_detection_image.publish(img_msg)
        except Exception as e:
            self.log.warn_throttle(5.0, f"Publish detection image failed: {e}")

    def _publish_tracking_image(self, rgb, msg):
        """发布 2D 跟踪可视化图像"""
        try:
            img = rgb.copy()
            if msg is not None and len(msg.objects) > 0:
                img = self._draw_tracking_2d(img, msg.objects)
            else:
                cv2.putText(img, "Tracking: No objects", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = rospy.Time.now()
            self.pub_tracking_image.publish(img_msg)
        except Exception as e:
            self.log.warn_throttle(5.0, f"Publish tracking image failed: {e}")

    def _draw_detections_2d(self, img, objects):
        """绘制 2D 检测结果"""
        for i, obj in enumerate(objects):
            # 根据类别获取固定颜色 (同类别物体颜色一致)
            color = self._get_category_color_bgr(obj.category)

            # 绘制 bbox
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 标签: 类别 + 分数 + 距离
            label1 = f"{obj.category} {obj.score:.2f}"
            label2 = f"{obj.distance:.2f}m"

            # 标签背景
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            max_tw = max(tw1, tw2)

            cv2.rectangle(img, (x1, y1 - th1 - th2 - 10), (x1 + max_tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(img, label1, (x1 + 3, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, label2, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 标题
        cv2.putText(img, f"Detection: {len(objects)} objects", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def _draw_tracking_2d(self, img, objects):
        """绘制 2D 跟踪结果"""
        for obj in objects:
            color = self.COLORS_BGR[obj.track_id % len(self.COLORS_BGR)]

            # 绘制 bbox (更粗的线)
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # 标签: ID + 类别 + 距离
            label1 = f"#{obj.track_id} {obj.category}"
            label2 = f"{obj.distance:.2f}m" if obj.distance > 0 else "N/A"

            # 标签背景
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_tw = max(tw1, tw2)

            cv2.rectangle(img, (x1, y1 - th1 - th2 - 12), (x1 + max_tw + 8, y1), color, -1)
            cv2.putText(img, label1, (x1 + 4, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, label2, (x1 + 4, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 标题
        cv2.putText(img, f"Tracking: {len(objects)} objects", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return img

    def run(self):
        """运行节点"""
        rospy.spin()


def main():
    try:
        node = PerceptionRVizNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[PerceptRViz] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
