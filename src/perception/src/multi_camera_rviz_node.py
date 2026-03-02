#!/usr/bin/env python3
"""
Multi-Camera RViz Node - 双相机感知结果 RViz 可视化 (ROS2)

同时可视化 top 和 chassis 两个相机的检测结果：
- 独立的 3D 边界框 (不同颜色/namespace)
- 独立的 RGB-D 点云
- 独立的 2D 检测图像
- 共享的 LiDAR 着色点云

发布:
    ~/top/object_markers (MarkerArray) - Top相机物体边界框
    ~/chassis/object_markers (MarkerArray) - Chassis相机物体边界框
    ~/top/rgb_pointcloud (PointCloud2) - Top相机RGB-D点云
    ~/chassis/rgb_pointcloud (PointCloud2) - Chassis相机RGB-D点云
    ~/top/detection_image (Image) - Top相机2D检测可视化
    ~/chassis/detection_image (Image) - Chassis相机2D检测可视化
    ~/lidar_colored (PointCloud2) - LiDAR距离着色点云

Usage:
    ros2 run perception multi_camera_rviz_node
    ros2 launch perception multi_camera_rviz.launch.py
"""

import struct
import threading
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import message_filters
from perception.utils import CvBridgeNumPy2 as CvBridge, parse_pointcloud2_fast, LATCHED_QOS
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from perception.msg import Object3DArray
from perception.coordinate_transformer import CoordinateTransformer

# 传感器 QoS
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# 检测结果 QoS (深度更大，避免高频发布时丢失消息)
DETECTION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10  # 足够容纳10Hz的检测结果
)


@dataclass
class CameraVizState:
    """单个相机的可视化状态"""
    name: str
    color_offset: int  # 颜色偏移量
    intrinsics: Optional[Dict] = None
    latest_rgb: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    latest_objects: Optional[Object3DArray] = None
    transformer: CoordinateTransformer = field(default_factory=CoordinateTransformer)
    data_lock: threading.Lock = field(default_factory=threading.Lock)


class MultiCameraRVizNode(Node):
    """多相机感知结果 RViz 可视化节点"""

    # 物体颜色列表 (RGBA for 3D markers) - 20种颜色
    COLORS = [
        (0.0, 1.0, 0.0, 0.8),   # 0: 绿 (top相机主色)
        (0.0, 0.8, 0.2, 0.8),   # 1: 青绿
        (0.2, 1.0, 0.0, 0.8),   # 2: 黄绿
        (0.0, 1.0, 0.5, 0.8),   # 3: 春绿
        (0.5, 1.0, 0.0, 0.8),   # 4:  lime
        (1.0, 0.0, 0.0, 0.8),   # 5: 红 (chassis相机主色)
        (1.0, 0.2, 0.0, 0.8),   # 6: 橙红
        (1.0, 0.0, 0.2, 0.8),   # 7: 玫红
        (0.8, 0.0, 0.2, 0.8),   # 8: 紫红
        (1.0, 0.0, 0.5, 0.8),   # 9: 深粉
        (0.0, 0.0, 1.0, 0.8),   # 10: 蓝
        (1.0, 1.0, 0.0, 0.8),   # 11: 黄
        (1.0, 0.0, 1.0, 0.8),   # 12: 品红
        (0.0, 1.0, 1.0, 0.8),   # 13: 青
        (1.0, 0.5, 0.0, 0.8),   # 14: 橙
        (0.5, 0.0, 1.0, 0.8),   # 15: 紫
        (0.5, 1.0, 0.5, 0.8),   # 16: 浅绿
        (1.0, 0.5, 0.5, 0.8),   # 17: 浅红
        (0.5, 0.5, 1.0, 0.8),   # 18: 浅蓝
        (1.0, 1.0, 0.5, 0.8),   # 19: 浅黄
    ]

    # BGR 颜色 (for 2D OpenCV drawing)
    COLORS_BGR = [
        (0, 255, 0), (0, 204, 51), (51, 255, 0), (0, 255, 128), (0, 128, 255),
        (0, 0, 255), (0, 51, 255), (0, 255, 255), (51, 0, 255), (128, 0, 255),
        (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 255, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
    ]

    # 类别到颜色的映射缓存
    _category_color_cache: Dict[str, Tuple] = {}

    def __init__(self):
        super().__init__('multi_camera_rviz_node')

        # === 参数 ===
        self._load_parameters()

        # === 回调组（用于多线程执行器）===
        # 融合结果回调使用独立的回调组，避免被定时器阻塞
        self._fused_callback_group = MutuallyExclusiveCallbackGroup()
        self._timer_callback_group = MutuallyExclusiveCallbackGroup()

        # === 组件 ===
        self._bridge = CvBridge()

        # === 相机状态 (完全独立) ===
        self._cameras: Dict[str, CameraVizState] = {}
        if self._enable_top:
            self._cameras['top'] = CameraVizState(name='top', color_offset=0)
        if self._enable_chassis:
            self._cameras['chassis'] = CameraVizState(name='chassis', color_offset=5)

        # === 融合结果状态 ===
        self._latest_fused_objects = None
        self._fused_lock = threading.Lock()
        self._previous_fused_ids = set()  # 追踪之前的fused marker IDs，避免DELETEALL导致闪烁
        self._previous_camera_ids = {}  # 追踪每个相机的marker IDs {camera_name: set()}

        # 加载外参
        self._load_extrinsics()

        # === 发布器 ===
        self._init_publishers()

        # === 节流控制 ===
        self._last_publish_time = self.get_clock().now()
        if self._publish_rate > 0:
            self._publish_period = 1.0 / self._publish_rate
        else:
            self._publish_period = float('inf')

        # === 设置订阅 ===
        for camera_name in self._cameras.keys():
            self._setup_camera_subscribers(camera_name)
            self._setup_detection_subscribers(camera_name)

        # === 订阅 LiDAR ===
        self._lidar_sub = self.create_subscription(
            PointCloud2, self._lidar_topic, self._lidar_callback, SENSOR_QOS
        )

        # === 订阅融合结果（使用独立回调组，避免被定时器阻塞）===
        fused_topic = f'/{self._perception_node_name}/fused/objects_3d'
        self._fused_sub = self.create_subscription(
            Object3DArray, fused_topic, self._fused_objects_callback, DETECTION_QOS,
            callback_group=self._fused_callback_group
        )
        self.get_logger().info(f"Subscribe fused results: {fused_topic} (depth=10, dedicated callback group)")

        # === 定时发布（使用独立回调组）===
        if self._publish_rate > 0:
            period = 1.0 / self._publish_rate
            self._timer = self.create_timer(
                period, self._publish_callback,
                callback_group=self._timer_callback_group
            )
        else:
            self._timer = None

        self.get_logger().info("Multi-Camera RViz Node started")
        self.get_logger().info(f"  Cameras: {list(self._cameras.keys())}")
        self.get_logger().info(f"  Target frame: {self._target_frame}")
        self.get_logger().info(f"  Publish rate: {self._publish_rate}Hz")

    def _load_parameters(self):
        """声明和加载参数"""
        # 相机启用开关
        self.declare_parameter('enable_top', True)
        self.declare_parameter('enable_chassis', True)
        self._enable_top = self.get_parameter('enable_top').value
        self._enable_chassis = self.get_parameter('enable_chassis').value

        # 通用参数
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('depth_max', 5.0)
        self.declare_parameter('cloud_skip', 2)
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
        self.declare_parameter('extrinsics_suffix', '')

        self._target_frame = self.get_parameter('target_frame').value
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
        self._extrinsics_suffix = self.get_parameter('extrinsics_suffix').value

        # 感知节点名称（用于构建话题路径）
        self.declare_parameter('perception_node_name', 'multi_camera_perception')
        self._perception_node_name = self.get_parameter('perception_node_name').value

    def _load_extrinsics(self):
        """加载所有相机的外参"""
        for camera_name, state in self._cameras.items():
            if self._extrinsics_dir:
                state.transformer.set_config_dir(self._extrinsics_dir)
            try:
                suffix = self._extrinsics_suffix if camera_name == 'chassis' else ''
                state.transformer.load_all_extrinsics(camera_name=camera_name, suffix=suffix)
                self.get_logger().info(f"[{camera_name}] Extrinsics loaded (suffix='{suffix}')")
            except Exception as e:
                self.get_logger().warn(f"[{camera_name}] Failed to load extrinsics: {e}")

    def _init_publishers(self):
        """初始化发布器（每个相机独立）"""
        self.pubs = {}

        for camera_name in self._cameras.keys():
            self.pubs[camera_name] = {
                'object_markers': self.create_publisher(
                    MarkerArray, f'~/{camera_name}/object_markers', 1
                ),
                'rgb_pointcloud': self.create_publisher(
                    PointCloud2, f'~/{camera_name}/rgb_pointcloud', 1
                ),
                'detection_image': self.create_publisher(
                    Image, f'~/{camera_name}/detection_image', 1
                ),
            }

        # LiDAR 着色点云（共享）
        self.pub_lidar_colored = self.create_publisher(
            PointCloud2, '~/lidar_colored', 1
        )

        # 合并的 markers（可选，方便统一查看）
        self.pub_combined_markers = self.create_publisher(
            MarkerArray, '~/combined/all_markers', 1
        )

        # 融合结果 markers
        self.pub_fused_markers = self.create_publisher(
            MarkerArray, '~/fused/object_markers', 1
        )

    def _setup_camera_subscribers(self, camera_name: str):
        """设置相机订阅（RGB + 深度）"""
        state = self._cameras[camera_name]

        # 相机内参
        info_topic = f'/camera/{camera_name}/color/camera_info'
        self.create_subscription(
            CameraInfo, info_topic,
            lambda msg, name=camera_name: self._camera_info_callback(name, msg),
            SENSOR_QOS
        )

        color_topic = f'/camera/{camera_name}/color/image_raw'
        depth_topic = f'/{self._perception_node_name}/{camera_name}/optimized_depth'

        # === 独立 RGB 订阅器（用于检测图像叠加，不受深度图帧率限制）===
        self.create_subscription(
            Image, color_topic,
            lambda msg, name=camera_name: self._rgb_only_callback(name, msg),
            SENSOR_QOS
        )

        # === RGB + 深度同步订阅器（用于点云生成）===
        color_sub = message_filters.Subscriber(
            self, Image, color_topic, qos_profile=SENSOR_QOS
        )
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=SENSOR_QOS
        )

        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.1
        )
        sync.registerCallback(
            lambda color, depth, name=camera_name: self._camera_callback(name, color, depth)
        )

        self.get_logger().info(f"[{camera_name}] Subscribe RGB (detection overlay): {color_topic}")
        self.get_logger().info(f"[{camera_name}] Subscribe RGB+Depth (point cloud): {depth_topic}")

    def _setup_detection_subscribers(self, camera_name: str):
        """设置检测结果订阅"""
        topic = f'/{self._perception_node_name}/{camera_name}/objects_3d'
        self.create_subscription(
            Object3DArray, topic,
            lambda msg, name=camera_name: self._objects_callback(name, msg),
            1
        )
        self.get_logger().info(f"[{camera_name}] Subscribe detections: {topic}")

    def _camera_info_callback(self, camera_name: str, msg: CameraInfo):
        """相机内参回调"""
        state = self._cameras[camera_name]
        if state.intrinsics is None:
            state.intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5],
                'width': msg.width, 'height': msg.height,
            }
            self.get_logger().info(f"[{camera_name}] Intrinsics: {msg.width}x{msg.height}")

    def _rgb_only_callback(self, camera_name: str, color_msg: Image):
        """独立 RGB 回调（用于检测图像叠加，高帧率）"""
        state = self._cameras[camera_name]
        try:
            # 解码 RGB
            rgb = self._bridge.imgmsg_to_cv2(color_msg, 'passthrough')
            if color_msg.encoding == 'rgb8':
                rgb = rgb[:, :, ::-1]  # RGB to BGR

            with state.data_lock:
                state.latest_rgb = rgb
                objects = state.latest_objects
                # 复制 rgb 用于后续发布，避免数据竞争
                rgb_copy = rgb.copy() if rgb is not None else None

            # 有检测结果时立即发布检测图像（跟随 RGB 帧率）
            if objects is not None and rgb_copy is not None:
                self._publish_detection_image(camera_name, rgb_copy, objects)

        except Exception as e:
            self.get_logger().debug(f"[{camera_name}] RGB callback failed: {e}")

    def _camera_callback(self, camera_name: str, color_msg: Image, depth_msg: Image):
        """RGB + 深度同步回调（用于点云生成，帧率受 optimized_depth 限制）"""
        state = self._cameras[camera_name]
        try:
            # 解码深度（RGB 已在 _rgb_only_callback 中更新）
            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            # 去噪
            if self._enable_depth_denoise:
                valid_mask = depth_m > 0.1
                depth_m = cv2.medianBlur(depth_m, self._depth_denoise_ksize)
                depth_m[~valid_mask] = 0

            with state.data_lock:
                state.latest_depth = depth_m

            # 注：检测图像已在 _rgb_only_callback 中发布（高帧率）

        except Exception as e:
            self.get_logger().warn(f"[{camera_name}] Depth callback failed: {e}")

    def _objects_callback(self, camera_name: str, msg: Object3DArray):
        """检测结果回调"""
        state = self._cameras[camera_name]
        with state.data_lock:
            state.latest_objects = msg
            rgb = state.latest_rgb
            # 复制 rgb 用于后续发布，避免数据竞争
            rgb_copy = rgb.copy() if rgb is not None else None
            obj_count = len(msg.objects) if msg is not None else 0

        self.get_logger().debug(f'[{camera_name}] Objects callback: {obj_count} objects, rgb={rgb_copy is not None}')

        # 发布 2D 检测图像（即使 rgb 为 None 也要发布提示信息）
        if rgb_copy is not None:
            self._publish_detection_image(camera_name, rgb_copy, msg)
        else:
            # 创建空白图像提示没有相机数据
            self._publish_no_camera_image(camera_name, msg)

        # 立即发布 3D markers（事件驱动，跟随检测频率，避免被点云定时器拖慢）
        self._publish_camera_markers(camera_name, msg)

    def _publish_callback(self):
        """定时发布回调（仅发布点云，markers 已由 _objects_callback 事件驱动发布）"""
        now = self.get_clock().now()
        if self._publish_period < float('inf'):
            dt = (now - self._last_publish_time).nanoseconds / 1e9
            if dt < self._publish_period:
                return
            self._last_publish_time = now

        stamp = now.to_msg()

        for camera_name, state in self._cameras.items():
            with state.data_lock:
                rgb = state.latest_rgb
                depth = state.latest_depth
                # 复制数据用于后续处理，避免数据竞争
                rgb_copy = rgb.copy() if rgb is not None else None
                depth_copy = depth.copy() if depth is not None else None

            if rgb_copy is None or depth_copy is None or state.intrinsics is None:
                continue

            # 发布 RGB-D 点云
            self._publish_rgb_pointcloud(camera_name, rgb_copy, depth_copy, stamp)

    def _publish_camera_markers(self, camera_name: str, objects_msg: Optional[Object3DArray]):
        """发布 per-camera markers（事件驱动，跟随检测频率）

        与 _publish_fused_visualization 设计一致：检测结果到达时立即发布，
        不被点云定时器拖慢。使用增量更新避免闪烁。
        """
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        if camera_name not in self._previous_camera_ids:
            self._previous_camera_ids[camera_name] = set()

        current_ids = set()
        objects_list = objects_msg.objects if objects_msg is not None else []
        for i, obj in enumerate(objects_list):
            color = self._get_category_color(obj.category, camera_name)
            global_id = self._get_global_marker_id(camera_name, i)
            current_ids.add(global_id)

            bbox_marker = self._create_bbox_marker(obj, global_id, color, stamp, camera_name)
            markers.markers.append(bbox_marker)

            label_marker = self._create_distance_label(obj, global_id, stamp, camera_name)
            markers.markers.append(label_marker)

        # 增量删除旧 markers
        stale_ids = self._previous_camera_ids[camera_name] - current_ids
        for stale_id in stale_ids:
            del_bbox = Marker()
            del_bbox.header.frame_id = self._target_frame
            del_bbox.header.stamp = stamp
            del_bbox.ns = f"{camera_name}_object_bbox"
            del_bbox.id = stale_id
            del_bbox.action = Marker.DELETE
            markers.markers.append(del_bbox)

            del_label = Marker()
            del_label.header.frame_id = self._target_frame
            del_label.header.stamp = stamp
            del_label.ns = f"{camera_name}_distance_labels"
            del_label.id = stale_id
            del_label.action = Marker.DELETE
            markers.markers.append(del_label)

        self._previous_camera_ids[camera_name] = current_ids
        self.pubs[camera_name]['object_markers'].publish(markers)
        self.pub_combined_markers.publish(markers)

    def _get_category_color(self, category: str, camera_name: str) -> Tuple:
        """根据类别和相机获取固定颜色"""
        cache_key = f"{camera_name}_{category}"
        if cache_key not in self._category_color_cache:
            color_idx = (hash(category) + self._cameras[camera_name].color_offset) % len(self.COLORS)
            self._category_color_cache[cache_key] = self.COLORS[color_idx]
        return self._category_color_cache[cache_key]

    def _get_category_color_bgr(self, category: str, camera_name: str) -> Tuple:
        """根据类别和相机获取固定 BGR 颜色"""
        color_idx = (hash(category) + self._cameras[camera_name].color_offset) % len(self.COLORS_BGR)
        return self.COLORS_BGR[color_idx]

    def _get_global_marker_id(self, camera_name: str, local_id: int) -> int:
        """生成全局唯一的 marker id"""
        # top: 0-999, chassis: 1000-1999
        if camera_name == 'top':
            return local_id
        else:
            return local_id + 1000

    def _publish_rgb_pointcloud(self, camera_name: str, rgb: np.ndarray, 
                                depth: np.ndarray, stamp):
        """发布 RGB-D 点云"""
        state = self._cameras[camera_name]

        # 降采样
        scale = self._image_scale
        if scale < 1.0:
            new_h = int(depth.shape[0] * scale)
            new_w = int(depth.shape[1] * scale)
            depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            fx = state.intrinsics['fx'] * scale
            fy = state.intrinsics['fy'] * scale
            cx = state.intrinsics['cx'] * scale
            cy = state.intrinsics['cy'] * scale
        else:
            depth_scaled = depth
            rgb_scaled = rgb
            fx, fy = state.intrinsics['fx'], state.intrinsics['fy']
            cx, cy = state.intrinsics['cx'], state.intrinsics['cy']

        h, w = depth_scaled.shape
        skip = self._cloud_skip
        ys, xs = np.mgrid[0:h:skip, 0:w:skip]
        ys, xs = ys.flatten(), xs.flatten()

        ds = depth_scaled[ys, xs]
        valid = (ds > 0.1) & (ds < self._depth_max)
        ys_valid, xs_valid, ds_valid = ys[valid], xs[valid], ds[valid]

        if len(ds_valid) == 0:
            return

        # 反投影
        X = (xs_valid - cx) * ds_valid / fx
        Y = (ys_valid - cy) * ds_valid / fy
        Z = ds_valid
        points_optical = np.column_stack([X, Y, Z])

        # 变换到 base_link（使用各自相机的外参）
        try:
            points_base = state.transformer.optical_to_target(points_optical, self._target_frame)
        except Exception as e:
            self.get_logger().debug(f"[{camera_name}] Transform failed: {e}")
            return

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

        if len(points_base) == 0:
            return

        # 获取颜色
        colors = rgb_scaled[ys_valid, xs_valid]

        # 创建点云
        cloud_msg = self._create_colored_pointcloud(points_base, colors, stamp, self._target_frame)
        self.pubs[camera_name]['rgb_pointcloud'].publish(cloud_msg)

    def _publish_object_visualization(self, camera_name: str,
                                      objects_msg: Optional[Object3DArray],
                                      depth: np.ndarray, stamp):
        """发布物体可视化（增量更新，避免闪烁）"""
        state = self._cameras[camera_name]
        markers = MarkerArray()

        # 初始化该相机的marker ID追踪集合
        if camera_name not in self._previous_camera_ids:
            self._previous_camera_ids[camera_name] = set()

        # 收集本次的marker IDs
        current_ids = set()

        # 收集所有物体点云
        all_object_points = []
        all_object_colors = []

        fx, fy = state.intrinsics['fx'], state.intrinsics['fy']
        cx, cy = state.intrinsics['cx'], state.intrinsics['cy']

        # 处理物体（即使没有物体也要继续，以便清理旧markers）
        objects_list = objects_msg.objects if objects_msg is not None else []
        for i, obj in enumerate(objects_list):
            color = self._get_category_color(obj.category, camera_name)
            global_id = self._get_global_marker_id(camera_name, i)
            current_ids.add(global_id)

            # 边界框 marker
            bbox_marker = self._create_bbox_marker(obj, global_id, color, stamp, camera_name)
            markers.markers.append(bbox_marker)

            # 距离标签 marker
            label_marker = self._create_distance_label(obj, global_id, stamp, camera_name)
            markers.markers.append(label_marker)

            # 生成物体 mask 点云（基于 bbox）
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h, w = depth.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            skip = 2
            ys_box, xs_box = np.mgrid[y1:y2:skip, x1:x2:skip]
            ys_flat, xs_flat = ys_box.flatten(), xs_box.flatten()

            if len(ys_flat) == 0:
                continue

            ds_flat = depth[ys_flat, xs_flat]
            valid = (ds_flat > 0.1) & (ds_flat < self._depth_max)

            if not np.any(valid):
                continue

            depth_median = np.median(ds_flat[valid])
            depth_mask = np.abs(ds_flat - depth_median) < 0.5
            final_mask = valid & depth_mask

            ys_v, xs_v, ds_v = ys_flat[final_mask], xs_flat[final_mask], ds_flat[final_mask]

            if len(ds_v) < 10:
                continue

            # 反投影
            X = (xs_v - cx) * ds_v / fx
            Y = (ys_v - cy) * ds_v / fy
            Z = ds_v
            pts = np.column_stack([X, Y, Z])

            # 过滤
            if len(pts) > 1000:
                pts = self._cluster_object_points(pts)

            if len(pts) > 0:
                all_object_points.append(pts)
                clr = np.array([[int(color[2]*255), int(color[1]*255), int(color[0]*255)]])
                all_object_colors.append(np.tile(clr, (len(pts), 1)))

        # 删除不再需要的旧markers（增量删除，避免DELETEALL导致闪烁）
        stale_ids = self._previous_camera_ids[camera_name] - current_ids
        for stale_id in stale_ids:
            # 删除边界框
            del_bbox = Marker()
            del_bbox.header.frame_id = self._target_frame
            del_bbox.header.stamp = stamp
            del_bbox.ns = f"{camera_name}_object_bbox"
            del_bbox.id = stale_id
            del_bbox.action = Marker.DELETE
            markers.markers.append(del_bbox)

            # 删除标签
            del_label = Marker()
            del_label.header.frame_id = self._target_frame
            del_label.header.stamp = stamp
            del_label.ns = f"{camera_name}_distance_labels"
            del_label.id = stale_id
            del_label.action = Marker.DELETE
            markers.markers.append(del_label)

        # 更新追踪集合
        self._previous_camera_ids[camera_name] = current_ids

        # 发布 markers
        self.pubs[camera_name]['object_markers'].publish(markers)

        # 发布合并的 markers（方便统一查看）
        self.pub_combined_markers.publish(markers)

    def _create_bbox_marker(self, obj, idx: int, color: Tuple, stamp, camera_name: str) -> Marker:
        """创建边界框 Marker"""
        marker = Marker()
        marker.header.frame_id = self._target_frame
        marker.header.stamp = stamp
        marker.ns = f"{camera_name}_object_bbox"  # 独立 namespace
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
        marker.color.a = color[3]

        # 添加相机前缀
        prefix = "[TOP]" if camera_name == 'top' else "[CHS]"
        marker.text = f"{prefix} {obj.object_id}"
        marker.lifetime.sec = 3  # 固定3秒lifetime，避免检测间隔不稳定导致闪烁

        return marker

    def _create_distance_label(self, obj, idx: int, stamp, camera_name: str) -> Marker:
        """创建距离标签 Marker"""
        marker = Marker()
        marker.header.frame_id = self._target_frame
        marker.header.stamp = stamp
        marker.ns = f"{camera_name}_distance_labels"  # 独立 namespace
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z + 0.18  # 增加高度以容纳更多文字
        marker.pose.orientation.w = 1.0

        # 构建文字内容
        lines = [obj.object_id]

        # 第二行: base_link 距离
        prefix = "T" if camera_name == 'top' else "C"
        cam_dist = f"dist:{obj.distance:.2f}m" if obj.distance > 0 else "dist:N/A"
        lines.append(cam_dist)

        # 第三行: base_link 坐标 (用于验证外参)
        lines.append(f"base:({obj.position.x:.2f},{obj.position.y:.2f},{obj.position.z:.2f})")

        # 第四行: optical frame 坐标 (用于验证外参)
        if hasattr(obj, 'position_optical') and obj.position_optical is not None:
            opt = obj.position_optical
            lines.append(f"opt:({opt.x:.2f},{opt.y:.2f},{opt.z:.2f})")

        marker.text = "\n".join(lines)

        base_scale = 0.020  # 稍微减小字体以容纳更多信息
        distance = max(obj.distance, 0.5) if obj.distance > 0 else 1.0
        marker.scale.z = base_scale * distance

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.lifetime.sec = 3  # 固定3秒lifetime，避免检测间隔不稳定导致闪烁

        return marker

    def _cluster_object_points(self, points: np.ndarray, radius: float = 0.05,
                                min_neighbors: int = 5) -> np.ndarray:
        """快速统计过滤: 使用 KD-Tree 去除离群点"""
        if len(points) < min_neighbors:
            return points

        try:
            from scipy.spatial import cKDTree

            if len(points) > 2000:
                idx = np.random.choice(len(points), 2000, replace=False)
                sample_points = points[idx]
            else:
                sample_points = points

            tree = cKDTree(sample_points)
            counts = tree.query_ball_point(sample_points, radius, return_length=True)
            dense_mask = counts >= min_neighbors

            if len(points) > 2000:
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

    def _create_colored_pointcloud(self, points: np.ndarray, colors: np.ndarray, 
                                   stamp, frame_id: str) -> PointCloud2:
        """创建带颜色的 PointCloud2"""
        n_points = len(points)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)
        ])
        cloud_arr = np.zeros(n_points, dtype=dtype)
        cloud_arr['x'] = points[:, 0].astype(np.float32)
        cloud_arr['y'] = points[:, 1].astype(np.float32)
        cloud_arr['z'] = points[:, 2].astype(np.float32)

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

    def _publish_no_camera_image(self, camera_name: str, msg: Object3DArray):
        """发布无相机数据的提示图像"""
        try:
            # 创建空白图像
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            prefix = "TOP" if camera_name == 'top' else "CHS"
            
            # 显示无相机数据提示
            cv2.putText(img, f"{prefix}: No Camera Data", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 显示检测数量（如果有）
            obj_count = len(msg.objects) if msg is not None else 0
            cv2.putText(img, f"Detections: {obj_count}", (50, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pubs[camera_name]['detection_image'].publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"[{camera_name}] Publish no-camera image failed: {e}")

    def _publish_detection_image(self, camera_name: str, rgb: np.ndarray, msg: Object3DArray):
        """发布 2D 检测可视化图像"""
        try:
            img = rgb.copy()
            obj_count = len(msg.objects) if msg is not None else 0
            
            if msg is not None and len(msg.objects) > 0:
                self.get_logger().debug(f'[{camera_name}] Drawing {len(msg.objects)} detections on image {img.shape}')
                img = self._draw_detections_2d(img, msg.objects, camera_name)
            else:
                prefix = "TOP" if camera_name == 'top' else "CHS"
                cv2.putText(img, f"{prefix}: No objects", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pubs[camera_name]['detection_image'].publish(img_msg)
            self.get_logger().debug(f'[{camera_name}] Published detection image with {obj_count} objects')
        except Exception as e:
            self.get_logger().error(f"[{camera_name}] Publish detection image failed: {e}")
            import traceback
            traceback.print_exc()

    def _draw_detections_2d(self, img: np.ndarray, objects: List, camera_name: str) -> np.ndarray:
        """绘制 2D 检测结果"""
        h, w = img.shape[:2]
        
        for i, obj in enumerate(objects):
            try:
                color = self._get_category_color_bgr(obj.category, camera_name)

                bbox = obj.bbox
                if len(bbox) < 4:
                    continue
                    
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # 边界检查，防止越界
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                prefix = "T" if camera_name == 'top' else "C"
                label1 = f"{prefix}:{obj.category} {obj.score:.2f}"
                label2 = f"{obj.distance:.2f}m"

                (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                max_tw = max(tw1, tw2)
                
                # 文本背景框边界检查
                text_y1 = max(0, y1 - th1 - th2 - 10)
                text_y2 = y1
                text_x2 = min(w, x1 + max_tw + 6)
                
                if text_y1 < text_y2 and x1 < text_x2:
                    cv2.rectangle(img, (x1, text_y1), (text_x2, text_y2), (0, 0, 0), -1)
                    if text_y1 + th2 + 6 < h:
                        cv2.putText(img, label1, (x1 + 3, text_y1 + th2 + 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    if text_y1 + th1 + th2 + 3 < h:
                        cv2.putText(img, label2, (x1 + 3, text_y1 + th1 + th2 + 3),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            except Exception as e:
                self.get_logger().debug(f'[{camera_name}] Failed to draw object {i}: {e}')
                continue

        prefix = "TOP" if camera_name == 'top' else "CHS"
        cv2.putText(img, f"{prefix}: {len(objects)} objects", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    # ==================== LiDAR 着色（共享）====================

    def _lidar_callback(self, msg: PointCloud2):
        """LiDAR 点云回调 - 添加距离颜色"""
        try:
            if self._lidar_publish_rate <= 0:
                return

            now = self.get_clock().now()
            if not hasattr(self, '_last_lidar_time'):
                self._last_lidar_time = now
            dt = (now - self._last_lidar_time).nanoseconds / 1e9
            if dt < 1.0 / self._lidar_publish_rate:
                return
            self._last_lidar_time = now

            points = self._parse_pointcloud2(msg)
            if len(points) == 0:
                return

            if self._lidar_skip > 1:
                points = points[::self._lidar_skip]

            distances = np.linalg.norm(points, axis=1)
            colors = self._distances_to_colors(distances)

            colored_cloud = self._create_colored_pointcloud(
                points, colors, msg.header.stamp, msg.header.frame_id
            )
            self.pub_lidar_colored.publish(colored_cloud)

        except Exception as e:
            self.get_logger().warn(f"LiDAR coloring failed: {e}")

    def _parse_pointcloud2(self, msg: PointCloud2) -> np.ndarray:
        """使用 utils 快速向量化解析 PointCloud2"""
        points = parse_pointcloud2_fast(msg)
        # parse_pointcloud2_fast 返回 (N, 4)，只取 xyz
        return points[:, :3] if len(points) > 0 else np.empty((0, 3), dtype=np.float32)

    def _distances_to_colors(self, distances: np.ndarray) -> np.ndarray:
        """将距离映射到颜色 (蓝→绿→黄→红)"""
        ratios = (distances - self._lidar_min_range) / (self._lidar_max_range - self._lidar_min_range)
        ratios = np.clip(ratios, 0.0, 1.0)

        n = len(ratios)
        colors = np.zeros((n, 3), dtype=np.uint8)

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

    # ==================== 融合结果可视化 ====================

    def _fused_objects_callback(self, msg: Object3DArray):
        """融合结果回调"""
        obj_count = len(msg.objects) if msg else 0
        self.get_logger().info(f'[FUSED_VIZ] 收到融合结果: {obj_count} objects')

        with self._fused_lock:
            self._latest_fused_objects = msg

        # 立即发布可视化
        self._publish_fused_visualization(msg)

    def _publish_fused_visualization(self, objects_msg: Object3DArray):
        """发布融合结果可视化（增量更新，避免闪烁）"""
        try:
            markers = MarkerArray()
            stamp = self.get_clock().now().to_msg()

            # 融合结果使用独特的颜色（金色/黄色），区别于top绿色和chassis红色
            fused_color = (1.0, 0.84, 0.0, 0.8)  # 金色 (r, g, b, a)

            # 收集本次需要的marker IDs
            current_ids = set()
            valid_count = 0

            # 处理物体（即使没有物体也要继续，以便清理旧markers）
            if objects_msg is not None and len(objects_msg.objects) > 0:
                for i, obj in enumerate(objects_msg.objects):
                    try:
                        # 使用独立的marker ID范围 (2000-2999)
                        marker_id = 2000 + i
                        current_ids.add(marker_id)

                        # 边界框 marker
                        bbox_marker = self._create_fused_bbox_marker(obj, marker_id, fused_color, stamp)
                        markers.markers.append(bbox_marker)

                        # 距离标签 marker
                        label_marker = self._create_fused_distance_label(obj, marker_id, stamp)
                        markers.markers.append(label_marker)
                        valid_count += 1
                    except Exception as e:
                        self.get_logger().warn(f'[FUSED_VIZ] 创建 marker {i} 失败: {e}')

            # 删除不再需要的旧markers（增量删除，避免DELETEALL导致闪烁）
            stale_ids = self._previous_fused_ids - current_ids
            for stale_id in stale_ids:
                # 删除边界框
                del_bbox = Marker()
                del_bbox.header.frame_id = self._target_frame
                del_bbox.header.stamp = stamp
                del_bbox.ns = "fused_object_bbox"
                del_bbox.id = stale_id
                del_bbox.action = Marker.DELETE
                markers.markers.append(del_bbox)

                # 删除标签
                del_label = Marker()
                del_label.header.frame_id = self._target_frame
                del_label.header.stamp = stamp
                del_label.ns = "fused_distance_labels"
                del_label.id = stale_id
                del_label.action = Marker.DELETE
                markers.markers.append(del_label)

            # 更新追踪集合
            self._previous_fused_ids = current_ids

            # 发布融合结果 markers
            self.pub_fused_markers.publish(markers)
            self.get_logger().debug(f'[FUSED_VIZ] 发布 {valid_count} 个 fused markers, 删除 {len(stale_ids)} 个旧 markers')

            # 也添加到合并的markers中
            self.pub_combined_markers.publish(markers)
        except Exception as e:
            self.get_logger().error(f'[FUSED_VIZ] _publish_fused_visualization 异常: {e}')
            import traceback
            traceback.print_exc()

    def _create_fused_bbox_marker(self, obj, idx: int, color: Tuple, stamp) -> Marker:
        """创建融合结果的边界框 Marker"""
        marker = Marker()
        marker.header.frame_id = self._target_frame
        marker.header.stamp = stamp
        marker.ns = "fused_object_bbox"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z
        marker.pose.orientation.w = 1.0

        # 融合结果使用稍大的尺寸以区分
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        marker.text = f"[FUSED] {obj.object_id}"
        marker.lifetime.sec = 3  # 固定3秒lifetime，避免检测间隔不稳定导致闪烁

        return marker

    def _create_fused_distance_label(self, obj, idx: int, stamp) -> Marker:
        """创建融合结果的距离标签 Marker"""
        marker = Marker()
        marker.header.frame_id = self._target_frame
        marker.header.stamp = stamp
        marker.ns = "fused_distance_labels"
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = obj.position.x
        marker.pose.position.y = obj.position.y
        marker.pose.position.z = obj.position.z + 0.20  # 稍高一些避免与单相机标签重叠
        marker.pose.orientation.w = 1.0

        # 构建文字内容（格式与单相机保持一致，冒号后不加空格）
        lines = [obj.object_id]

        # 距离信息
        if obj.distance > 0:
            lines.append(f"dist:{obj.distance:.2f}m")
        else:
            dist = (obj.position.x**2 + obj.position.y**2 + obj.position.z**2) ** 0.5
            lines.append(f"dist:{dist:.2f}m" if dist > 0.01 else "dist:N/A")

        # 来源相机
        if obj.source_camera:
            lines.append(f"src:{obj.source_camera}")

        # 置信度
        if obj.confidence > 0:
            lines.append(f"conf:{obj.confidence:.2f}")

        marker.text = "\n".join(lines)

        base_scale = 0.022  # 与单相机一致
        distance = max(obj.distance, 0.5) if obj.distance > 0 else 1.0
        marker.scale.z = base_scale * distance

        # 使用金色文字
        marker.color.r = 1.0
        marker.color.g = 0.84
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 3  # 固定3秒lifetime，避免检测间隔不稳定导致闪烁

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraRVizNode()

    # 使用多线程执行器，让融合结果回调和定时器回调并行执行
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
