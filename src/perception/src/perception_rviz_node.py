#!/usr/bin/env python3
"""
Perception RViz Node - 感知结果 RViz 可视化 (ROS2)

发布:
- RGB-D 点云 (base_link 坐标系)
- 物体 Mask 3D 投影 (不同颜色)
- 距离标签 (Camera/LiDAR)
- 2D 检测/跟踪可视化图像
"""

import struct
import threading
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy

import message_filters
from perception.utils import CvBridgeNumPy2 as CvBridge
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from perception.msg import Object3DArray, TrackedObject3DArray

from perception.coordinate_transformer import CoordinateTransformer

# 传感器 QoS
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)


class PerceptionRVizNode(Node):
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

    # 类别到颜色的映射缓存
    _category_color_cache: Dict[str, Tuple] = {}

    def __init__(self):
        super().__init__('perception_rviz_node')

        # === 参数 ===
        self._load_parameters()

        # === 组件 ===
        self._bridge = CvBridge()
        self._transformer = CoordinateTransformer()

        # 加载外参
        if self._extrinsics_dir:
            self._transformer.set_config_dir(self._extrinsics_dir)
        try:
            self._transformer.load_all_extrinsics()
        except Exception as e:
            self.get_logger().warn(f"Failed to load extrinsics: {e}")

        # === 数据缓存 ===
        self._intrinsics: Optional[Dict] = None
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_objects: Optional[Object3DArray] = None
        self._cached_objects: Optional[Object3DArray] = None
        self._latest_tracked: Optional[TrackedObject3DArray] = None
        self._data_lock = threading.Lock()

        # === 发布器 ===
        self.pub_rgb_cloud = self.create_publisher(
            PointCloud2, '~/rgb_pointcloud', 1
        )
        self.pub_object_clouds = self.create_publisher(
            PointCloud2, '~/object_clouds', 1
        )
        self.pub_object_markers = self.create_publisher(
            MarkerArray, '~/object_markers', 1
        )
        self.pub_distance_labels = self.create_publisher(
            MarkerArray, '~/distance_labels', 1
        )
        self.pub_lidar_colored = self.create_publisher(
            PointCloud2, '~/lidar_colored', 1
        )
        self.pub_detection_image = self.create_publisher(
            Image, '/perception_viz/detection_image', 1
        )
        self.pub_tracking_image = self.create_publisher(
            Image, '/perception_viz/tracking_image', 1
        )

        # === 节流控制 ===
        self._last_camera_time = self.get_clock().now()
        self._last_lidar_time = self.get_clock().now()

        if self._publish_rate > 0:
            self._camera_throttle_period = 1.0 / self._publish_rate
        else:
            self._camera_throttle_period = float('inf')

        # === 获取相机内参 ===
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            f'/camera/{self._camera_name}/color/camera_info',
            self._camera_info_callback,
            SENSOR_QOS
        )

        # === 订阅相机 ===
        self._setup_camera_subscribers()

        # === 订阅检测结果 ===
        self._objects_sub = self.create_subscription(
            Object3DArray,
            '/scene_perception_3d/objects_3d',
            self._objects_callback,
            1
        )

        # === 订阅跟踪结果 ===
        self._tracked_sub = self.create_subscription(
            TrackedObject3DArray,
            '/object_tracker/tracked_objects',
            self._tracked_callback,
            1
        )

        # === 订阅 LiDAR 点云 ===
        self._lidar_sub = self.create_subscription(
            PointCloud2,
            self._lidar_topic,
            self._lidar_callback,
            SENSOR_QOS
        )

        # === 定时发布 ===
        if self._publish_rate > 0:
            period = 1.0 / self._publish_rate
            self._timer = self.create_timer(period, self._publish_callback)
        else:
            self._timer = None

        self.get_logger().info("Node started")
        self.get_logger().info(f"  Camera: {self._camera_name}")
        self.get_logger().info(f"  Target frame: {self._target_frame}")
        self.get_logger().info(f"  RGB-D: rate={self._publish_rate}Hz, scale={self._image_scale}")

    def _load_parameters(self):
        """声明和加载参数"""
        self.declare_parameter('camera_name', 'top')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('publish_rgb_cloud', True)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('depth_max', 5.0)
        self.declare_parameter('cloud_skip', 1)
        self.declare_parameter('image_scale', 0.5)
        self.declare_parameter('enable_outlier_filter', True)
        self.declare_parameter('outlier_radius', 0.05)
        self.declare_parameter('outlier_min_neighbors', 5)
        self.declare_parameter('enable_depth_denoise', True)
        self.declare_parameter('depth_denoise_ksize', 5)
        self.declare_parameter('lidar_topic', '/rslidar_points')
        self.declare_parameter('lidar_min_range', 0.0)
        self.declare_parameter('lidar_max_range', 10.0)
        self.declare_parameter('lidar_skip', 1)
        self.declare_parameter('lidar_publish_rate', 10.0)
        self.declare_parameter('extrinsics_dir', '')

        self._camera_name = self.get_parameter('camera_name').value
        self._target_frame = self.get_parameter('target_frame').value
        self._publish_rgb_cloud = self.get_parameter('publish_rgb_cloud').value
        self._publish_rate = self.get_parameter('publish_rate').value
        self._depth_max = self.get_parameter('depth_max').value
        self._cloud_skip = self.get_parameter('cloud_skip').value
        self._image_scale = self.get_parameter('image_scale').value
        self._enable_outlier_filter = self.get_parameter('enable_outlier_filter').value
        self._outlier_radius = self.get_parameter('outlier_radius').value
        self._outlier_min_neighbors = self.get_parameter('outlier_min_neighbors').value
        self._enable_depth_denoise = self.get_parameter('enable_depth_denoise').value
        self._depth_denoise_ksize = self.get_parameter('depth_denoise_ksize').value
        self._lidar_topic = self.get_parameter('lidar_topic').value
        self._lidar_min_range = self.get_parameter('lidar_min_range').value
        self._lidar_max_range = self.get_parameter('lidar_max_range').value
        self._lidar_skip = self.get_parameter('lidar_skip').value
        self._lidar_publish_rate = self.get_parameter('lidar_publish_rate').value
        self._extrinsics_dir = self.get_parameter('extrinsics_dir').value

    def _camera_info_callback(self, msg: CameraInfo):
        """相机内参回调"""
        if self._intrinsics is None:
            self._intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height,
            }
            self.get_logger().info(f"Intrinsics: {msg.width}x{msg.height}")

    def _setup_camera_subscribers(self):
        """设置相机订阅"""
        color_topic = f'/camera/{self._camera_name}/color/image_raw'
        depth_topic = '/scene_perception_3d/optimized_depth'

        if self._publish_rate > 0:
            self._color_sub = message_filters.Subscriber(
                self, Image, color_topic, qos_profile=SENSOR_QOS
            )
            self._depth_sub = message_filters.Subscriber(
                self, Image, depth_topic, qos_profile=SENSOR_QOS
            )

            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub],
                queue_size=5,
                slop=2.0
            )
            self._sync.registerCallback(self._camera_callback)
        else:
            self._color_sub = self.create_subscription(
                Image, color_topic, self._color_only_callback, SENSOR_QOS
            )
            self._depth_sub = self.create_subscription(
                Image, depth_topic, self._depth_only_callback, SENSOR_QOS
            )

        self.get_logger().info(f"Subscribe camera: {color_topic}")
        self.get_logger().info(f"Subscribe optimized depth: {depth_topic}")

    def _color_only_callback(self, color_msg: Image):
        """Color 图像独立回调"""
        try:
            rgb = self._bridge.imgmsg_to_cv2(color_msg, 'passthrough')
            if color_msg.encoding == 'rgb8':
                rgb = rgb[:, :, ::-1]

            with self._data_lock:
                self._latest_rgb = rgb
        except Exception as e:
            self.get_logger().warn(f"Color decode failed: {e}")

    def _depth_only_callback(self, depth_msg: Image):
        """Depth 图像独立回调"""
        try:
            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            if self._enable_depth_denoise:
                valid_mask = depth_m > 0.1
                depth_m = cv2.medianBlur(depth_m, self._depth_denoise_ksize)
                depth_m[~valid_mask] = 0

            with self._data_lock:
                self._latest_depth = depth_m
        except Exception as e:
            self.get_logger().warn(f"Depth decode failed: {e}")

    def _camera_callback(self, color_msg: Image, depth_msg: Image):
        """相机数据回调 (带节流优化)"""
        try:
            # 节流
            if self._camera_throttle_period < float('inf'):
                now = self.get_clock().now()
                dt = (now - self._last_camera_time).nanoseconds / 1e9
                if dt < self._camera_throttle_period:
                    return
                self._last_camera_time = now

            rgb = self._bridge.imgmsg_to_cv2(color_msg, 'passthrough')
            if color_msg.encoding == 'rgb8':
                rgb = rgb[:, :, ::-1]

            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            if self._enable_depth_denoise:
                valid_mask = depth_m > 0.1
                depth_m = cv2.medianBlur(depth_m, self._depth_denoise_ksize)
                depth_m[~valid_mask] = 0

            with self._data_lock:
                self._latest_rgb = rgb
                self._latest_depth = depth_m

        except Exception as e:
            self.get_logger().warn(f"Camera data process failed: {e}")

    def _objects_callback(self, msg: Object3DArray):
        """检测结果回调"""
        with self._data_lock:
            self._latest_objects = msg
            if msg is not None and len(msg.objects) > 0:
                self._cached_objects = msg
            rgb = self._latest_rgb

        # 发布 2D 检测图像
        if rgb is not None:
            self._publish_detection_image(rgb, msg)

    def _tracked_callback(self, msg: TrackedObject3DArray):
        """跟踪结果回调"""
        with self._data_lock:
            self._latest_tracked = msg
            rgb = self._latest_rgb

        # 发布 2D 跟踪图像
        if rgb is not None:
            self._publish_tracking_image(rgb, msg)

    def _publish_callback(self):
        """定时发布回调"""
        with self._data_lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            objects = self._latest_objects
            cached_objects = self._cached_objects

        if rgb is None or depth is None or self._intrinsics is None:
            return

        stamp = self.get_clock().now().to_msg()

        # 发布 RGB-D 点云
        if self._publish_rgb_cloud:
            self._publish_rgb_pointcloud(rgb, depth, stamp)

        # 发布物体可视化
        objects_to_visualize = objects if (objects is not None and len(objects.objects) > 0) else cached_objects
        if objects_to_visualize is not None and len(objects_to_visualize.objects) > 0:
            self._publish_object_visualization(objects_to_visualize, depth, stamp)

    def _get_category_color(self, category: str) -> Tuple:
        """根据类别获取固定颜色"""
        if category not in self._category_color_cache:
            color_idx = hash(category) % len(self.COLORS)
            self._category_color_cache[category] = self.COLORS[color_idx]
        return self._category_color_cache[category]

    def _get_category_color_bgr(self, category: str) -> Tuple:
        """根据类别获取固定 BGR 颜色"""
        color_idx = hash(category) % len(self.COLORS_BGR)
        return self.COLORS_BGR[color_idx]

    def _publish_rgb_pointcloud(self, rgb: np.ndarray, depth: np.ndarray, stamp):
        """发布 RGB-D 点云"""
        scale = self._image_scale
        if scale < 1.0:
            new_h, new_w = int(depth.shape[0] * scale), int(depth.shape[1] * scale)
            depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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

        skip = self._cloud_skip
        ys, xs = np.mgrid[0:h:skip, 0:w:skip]
        ys = ys.flatten()
        xs = xs.flatten()

        ds = depth_scaled[ys, xs]
        valid = (ds > 0.1) & (ds < self._depth_max)
        ys_valid, xs_valid, ds_valid = ys[valid], xs[valid], ds[valid]

        # 反投影到 optical frame
        X = (xs_valid - cx) * ds_valid / fx
        Y = (ys_valid - cy) * ds_valid / fy
        Z = ds_valid

        points_optical = np.column_stack([X, Y, Z])

        # 变换到 base_link
        points_base = self._transformer.optical_to_target(points_optical, self._target_frame)

        # 离群点过滤
        if self._enable_outlier_filter and len(points_base) > 0:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(points_base)
                neighbor_counts = tree.query_ball_point(points_base, self._outlier_radius, return_length=True)
                outlier_mask = neighbor_counts >= self._outlier_min_neighbors
                points_base = points_base[outlier_mask]
                ys_valid = ys_valid[outlier_mask]
                xs_valid = xs_valid[outlier_mask]
            except ImportError:
                pass

        # 获取颜色
        colors = rgb_scaled[ys_valid, xs_valid]

        # 创建 PointCloud2
        if len(points_base) > 0:
            cloud_msg = self._create_colored_pointcloud(points_base, colors, stamp, self._target_frame)
            self.pub_rgb_cloud.publish(cloud_msg)

    def _publish_object_visualization(self, objects_msg: Object3DArray, depth: np.ndarray, stamp):
        """发布物体可视化"""
        markers = MarkerArray()
        labels = MarkerArray()

        # 清除旧 markers
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
            color = self._get_category_color(obj.category)

            # 创建边界框 Marker
            bbox_marker = self._create_bbox_marker(obj, i, color, stamp)
            markers.markers.append(bbox_marker)

            # 创建距离标签
            label_marker = self._create_distance_label(obj, i, stamp)
            labels.markers.append(label_marker)

            # 生成物体 mask 点云
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            h, w = depth.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            skip = 2
            ys_box, xs_box = np.mgrid[y1:y2:skip, x1:x2:skip]
            ys_flat = ys_box.flatten()
            xs_flat = xs_box.flatten()

            if len(ys_flat) == 0:
                continue

            ds_flat = depth[ys_flat, xs_flat]

            valid = (ds_flat > 0.1) & (ds_flat < self._depth_max)

            if not np.any(valid):
                continue

            depth_median = np.median(ds_flat[valid])
            depth_range = 0.5
            depth_mask = np.abs(ds_flat - depth_median) < depth_range
            final_mask = valid & depth_mask

            ys_v = ys_flat[final_mask]
            xs_v = xs_flat[final_mask]
            ds_v = ds_flat[final_mask]

            if len(ds_v) < 10:
                continue

            # 反投影到 3D
            X = (xs_v - cx) * ds_v / fx
            Y = (ys_v - cy) * ds_v / fy
            Z = ds_v
            pts = np.column_stack([X, Y, Z])

            # 对大点云进行聚类过滤
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

        # 发布物体点云
        if len(all_object_points) > 0:
            points_optical = np.vstack(all_object_points)
            colors = np.vstack(all_object_colors).astype(np.uint8)

            points_base = self._transformer.optical_to_target(points_optical, self._target_frame)
            cloud_msg = self._create_colored_pointcloud(points_base, colors, stamp, self._target_frame)
            self.pub_object_clouds.publish(cloud_msg)

    def _cluster_object_points(self, points: np.ndarray, radius: float = 0.05,
                                min_neighbors: int = 5) -> np.ndarray:
        """快速统计过滤: 使用 KD-Tree 去除离群点

        Args:
            points: (N, 3) 点云数组
            radius: 邻域半径 (m)
            min_neighbors: 最小邻居数

        Returns:
            过滤后的点云
        """
        if len(points) < min_neighbors:
            return points

        try:
            from scipy.spatial import cKDTree

            # 随机降采样加速 KD-Tree 构建 (点数 > 2000 时)
            if len(points) > 2000:
                idx = np.random.choice(len(points), 2000, replace=False)
                sample_points = points[idx]
            else:
                sample_points = points

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
                    nearby = full_tree.query_ball_point(dense_sample, radius)
                    all_indices = []
                    for idx_list in nearby:
                        all_indices.extend(idx_list)
                    if len(all_indices) > 0:
                        valid_idx = np.unique(all_indices)
                        return points[valid_idx]
                return points
            else:
                return points[dense_mask] if np.any(dense_mask) else points

        except ImportError:
            return points

    def _create_bbox_marker(self, obj, idx: int, color: Tuple, stamp) -> Marker:
        """创建边界框 Marker"""
        marker = Marker()
        marker.header.frame_id = self._target_frame
        marker.header.stamp = stamp
        marker.ns = "object_bbox"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5

        marker.text = obj.object_id
        marker.lifetime.sec = 3

        return marker

    def _create_distance_label(self, obj, idx: int, stamp) -> Marker:
        """创建距离标签 Marker"""
        marker = Marker()
        marker.header.frame_id = self._target_frame
        marker.header.stamp = stamp
        marker.ns = "distance_labels"
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z + 0.15  # 增加高度以容纳更多文字
        marker.pose.orientation.w = 1.0

        # 构建文字内容
        lines = [obj.object_id]

        # 第二行: base_link 距离
        cam_dist = f"dist:{obj.distance:.2f}m" if obj.distance > 0 else "dist:N/A"
        if hasattr(obj, 'distance_lidar') and obj.distance_lidar > 0:
            lidar_dist = f"L:{obj.distance_lidar:.2f}"
            lines.append(f"{cam_dist} {lidar_dist}")
        else:
            lines.append(cam_dist)

        # 第三行: 相机光学坐标系下的深度
        if hasattr(obj, 'depth') and obj.depth > 0:
            # 相机名称 + 深度
            cam_name = obj.source_camera if hasattr(obj, 'source_camera') and obj.source_camera else "cam"
            lines.append(f"[{cam_name}] z={obj.depth:.2f}m")

            # 第四行: optical 坐标 (x, y, z)
            if hasattr(obj, 'position_optical'):
                po = obj.position_optical
                lines.append(f"opt:({po.x:.2f},{po.y:.2f},{po.z:.2f})")

        marker.text = "\n".join(lines)

        base_scale = 0.022  # 稍微减小字体以容纳更多信息
        distance = max(obj.distance, 0.5) if obj.distance > 0 else 1.0
        marker.scale.z = base_scale * distance

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.lifetime.sec = 3

        return marker

    def _create_colored_pointcloud(self, points: np.ndarray, colors: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        """创建带颜色的 PointCloud2"""
        n_points = len(points)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

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

        # RGB 打包
        r = colors[:, 2].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 0].astype(np.uint32)
        cloud_arr['rgb'] = (r << 16) | (g << 8) | b

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
        msg.data = cloud_arr.tobytes()

        return msg

    def _lidar_callback(self, msg: PointCloud2):
        """LiDAR 点云回调 - 添加距离颜色"""
        try:
            if self._lidar_publish_rate <= 0:
                return

            now = self.get_clock().now()
            dt = (now - self._last_lidar_time).nanoseconds / 1e9
            if dt < 1.0 / self._lidar_publish_rate:
                return
            self._last_lidar_time = now

            # 解析点云
            points = self._parse_pointcloud2(msg)
            if len(points) == 0:
                return

            # 降采样
            if self._lidar_skip > 1:
                points = points[::self._lidar_skip]

            # 计算距离和颜色
            distances = np.linalg.norm(points, axis=1)
            colors = self._distances_to_colors(distances)

            # 创建带颜色的点云
            colored_cloud = self._create_colored_pointcloud(
                points, colors, msg.header.stamp, msg.header.frame_id
            )

            self.pub_lidar_colored.publish(colored_cloud)

        except Exception as e:
            self.get_logger().warn(f"LiDAR coloring failed: {e}")

    def _parse_pointcloud2(self, msg: PointCloud2) -> np.ndarray:
        """解析 PointCloud2 消息"""
        # 简单解析 XYZ 点
        point_step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)

        # 假设 x, y, z 是前三个 float32
        n_points = msg.width * msg.height
        points = np.zeros((n_points, 3), dtype=np.float32)

        for i in range(n_points):
            offset = i * point_step
            points[i, 0] = np.frombuffer(data[offset:offset+4], dtype=np.float32)[0]
            points[i, 1] = np.frombuffer(data[offset+4:offset+8], dtype=np.float32)[0]
            points[i, 2] = np.frombuffer(data[offset+8:offset+12], dtype=np.float32)[0]

        # 过滤 NaN
        valid = ~np.isnan(points).any(axis=1)
        return points[valid]

    def _distances_to_colors(self, distances: np.ndarray) -> np.ndarray:
        """将距离映射到颜色 (蓝→绿→黄→红)"""
        ratios = (distances - self._lidar_min_range) / (self._lidar_max_range - self._lidar_min_range)
        ratios = np.clip(ratios, 0.0, 1.0)

        n = len(ratios)
        colors = np.zeros((n, 3), dtype=np.uint8)

        # 蓝 → 青 → 绿 → 黄 → 红
        mask1 = ratios < 0.25
        t1 = ratios[mask1] / 0.25
        colors[mask1, 0] = 0
        colors[mask1, 1] = (255 * t1).astype(np.uint8)
        colors[mask1, 2] = 255

        mask2 = (ratios >= 0.25) & (ratios < 0.5)
        t2 = (ratios[mask2] - 0.25) / 0.25
        colors[mask2, 0] = 0
        colors[mask2, 1] = 255
        colors[mask2, 2] = (255 * (1 - t2)).astype(np.uint8)

        mask3 = (ratios >= 0.5) & (ratios < 0.75)
        t3 = (ratios[mask3] - 0.5) / 0.25
        colors[mask3, 0] = (255 * t3).astype(np.uint8)
        colors[mask3, 1] = 255
        colors[mask3, 2] = 0

        mask4 = ratios >= 0.75
        t4 = (ratios[mask4] - 0.75) / 0.25
        colors[mask4, 0] = 255
        colors[mask4, 1] = (255 * (1 - t4)).astype(np.uint8)
        colors[mask4, 2] = 0

        return colors

    # ==================== 2D 图像可视化 ====================

    def _publish_detection_image(self, rgb: np.ndarray, msg: Object3DArray):
        """发布 2D 检测可视化图像"""
        try:
            img = rgb.copy()
            if msg is not None and len(msg.objects) > 0:
                img = self._draw_detections_2d(img, msg.objects)
            else:
                cv2.putText(img, "Detection: No objects", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_detection_image.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Publish detection image failed: {e}")

    def _publish_tracking_image(self, rgb: np.ndarray, msg: TrackedObject3DArray):
        """发布 2D 跟踪可视化图像"""
        try:
            img = rgb.copy()
            if msg is not None and len(msg.objects) > 0:
                img = self._draw_tracking_2d(img, msg.objects)
            else:
                cv2.putText(img, "Tracking: No objects", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_tracking_image.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Publish tracking image failed: {e}")

    def _draw_detections_2d(self, img: np.ndarray, objects: List) -> np.ndarray:
        """绘制 2D 检测结果"""
        for i, obj in enumerate(objects):
            color = self._get_category_color_bgr(obj.category)

            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label1 = f"{obj.category} {obj.score:.2f}"
            label2 = f"{obj.distance:.2f}m"

            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            max_tw = max(tw1, tw2)

            cv2.rectangle(img, (x1, y1 - th1 - th2 - 10), (x1 + max_tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(img, label1, (x1 + 3, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, label2, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(img, f"Detection: {len(objects)} objects", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def _draw_tracking_2d(self, img: np.ndarray, objects: List) -> np.ndarray:
        """绘制 2D 跟踪结果"""
        for obj in objects:
            color = self.COLORS_BGR[obj.track_id % len(self.COLORS_BGR)]

            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            label1 = f"#{obj.track_id} {obj.category}"
            label2 = f"{obj.distance:.2f}m" if obj.distance > 0 else "N/A"

            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_tw = max(tw1, tw2)

            cv2.rectangle(img, (x1, y1 - th1 - th2 - 12), (x1 + max_tw + 8, y1), color, -1)
            cv2.putText(img, label1, (x1 + 4, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, label2, (x1 + 4, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(img, f"Tracking: {len(objects)} objects", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return img


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionRVizNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
