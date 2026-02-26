#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Tracker Node (ROS2)

基于 SAM2TrackerOnline 的 2D 跟踪，结合深度测量获取 3D 位置。
异步接收检测结果，同步执行跟踪循环。

架构:
    检测节点 (2Hz)  ──→  本节点 (5Hz)  ──→  TrackedObject3D
         │                    │
    Object3DArray      SAM2TrackerOnline
                              │
                       DepthMeasurer
                              │
                         3D 位置

订阅:
    /top_camera/color/image_raw (sensor_msgs/Image)
    /top_camera/aligned_depth_to_color/image_raw (sensor_msgs/Image)
    /perception_3d/objects (Object3DArray)

发布:
    ~/tracked_objects (TrackedObject3DArray)
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
from .utils import CvBridgeNumPy2 as CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo

from perception_interfaces.msg import Object3DArray, TrackedObject3D, TrackedObject3DArray

# pycocotools for mask RLE decoding
try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

# 传感器 QoS
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# 可靠 QoS
RELIABLE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)


# ============================================================================
# DepthMeasurer: 从 mask 测量 3D 位置
# ============================================================================

@dataclass
class MeasurementResult:
    """深度测量结果"""
    valid: bool = False
    position: Optional[np.ndarray] = None
    distance: float = 0.0
    confidence: float = 0.0
    error_msg: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class DepthMeasurer:
    """
    从 mask 测量 3D 位置

    算法：
    1. Mask 腐蚀去噪
    2. 深度范围过滤
    3. IQR 异常值剔除
    4. 向量化反投影
    5. 中值质心计算
    """

    DEPTH_MIN = 0.3   # m
    DEPTH_MAX = 10.0  # m
    MASK_ERODE_KERNEL = 5
    IQR_FACTOR = 1.5
    MIN_DEPTH_POINTS = 10

    def __init__(self, intrinsics: dict, target_frame: str = 'base_link'):
        """
        Args:
            intrinsics: 相机内参 {fx, fy, cx, cy}
            target_frame: 目标坐标系
        """
        self.intrinsics = intrinsics
        self.target_frame = target_frame
        self._erode_kernel = np.ones(
            (self.MASK_ERODE_KERNEL, self.MASK_ERODE_KERNEL), np.uint8
        )

    def measure(self, depth: np.ndarray, mask: np.ndarray) -> MeasurementResult:
        """
        测量 mask 区域的 3D 位置

        Args:
            depth: 深度图 (H, W) float32, 单位: 米
            mask: 二值 mask (H, W) uint8 或 bool

        Returns:
            MeasurementResult
        """
        result = MeasurementResult()

        # 确保 mask 是 uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        # 1. Mask 腐蚀
        eroded = cv2.erode(mask, self._erode_kernel, iterations=1)
        if eroded.sum() < self.MIN_DEPTH_POINTS:
            result.error_msg = f"mask 太小: {eroded.sum()}"
            return result

        # 2. 提取深度值
        ys, xs = np.where(eroded > 0)
        depths = depth[ys, xs]

        # 3. 深度范围过滤
        valid = (depths > self.DEPTH_MIN) & (depths < self.DEPTH_MAX)
        if valid.sum() < self.MIN_DEPTH_POINTS:
            result.error_msg = f"有效深度点太少: {valid.sum()}"
            return result

        depths_valid = depths[valid]

        # 4. IQR 异常值剔除
        q1, q3 = np.percentile(depths_valid, [25, 75])
        iqr = q3 - q1
        lower = q1 - self.IQR_FACTOR * iqr
        upper = q3 + self.IQR_FACTOR * iqr

        # 组合过滤条件
        final_valid = valid & (depths >= lower) & (depths <= upper)
        if final_valid.sum() < self.MIN_DEPTH_POINTS:
            result.error_msg = f"IQR 后点太少: {final_valid.sum()}"
            return result

        # 5. 向量化反投影
        xs_f = xs[final_valid].astype(np.float32)
        ys_f = ys[final_valid].astype(np.float32)
        ds_f = depths[final_valid]

        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']

        X = (xs_f - cx) * ds_f / fx
        Y = (ys_f - cy) * ds_f / fy
        Z = ds_f

        points = np.column_stack([X, Y, Z])

        # 6. 计算质心 (中值)
        centroid = np.median(points, axis=0)
        distance = np.linalg.norm(centroid)

        # 7. 计算置信度
        depth_std = np.std(ds_f)
        confidence = self._compute_confidence(len(points), depth_std)

        result.valid = True
        result.position = centroid
        result.distance = float(distance)
        result.confidence = float(confidence)
        result.stats = {
            'num_points': len(points),
            'depth_median': float(np.median(ds_f)),
            'depth_std': float(depth_std),
        }

        return result

    def _compute_confidence(self, num_points: int, depth_std: float) -> float:
        """计算测量置信度"""
        point_score = min(num_points / 100.0, 1.0)
        std_score = max(0, 1.0 - depth_std / 0.3)
        return 0.7 * point_score + 0.3 * std_score


# ============================================================================
# ObjectTrackerNode: ROS2 节点
# ============================================================================

class ObjectTrackerNode(Node):
    """物体跟踪节点"""

    def __init__(self):
        super().__init__('object_tracker_node')

        # === 参数 ===
        self._load_parameters()

        # === 内部状态 ===
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # 图像缓存
        self._rgb_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self._rgb_stamp = None

        # 检测缓存 (异步更新)
        self._pending_detections: List[Dict] = []
        self._detection_stamp = None

        # 相机内参
        self._intrinsics: Optional[dict] = None

        # 跟踪器 (延迟初始化)
        self._tracker = None
        self._depth_measurer: Optional[DepthMeasurer] = None

        # 跟踪状态
        self._frame_idx = 0
        self._tracker_initialized = False

        # ID 到类别的映射
        self._id_to_category: Dict[int, str] = {}

        # 活跃跟踪目标缓存
        self._active_tracks: Dict[int, Dict] = {}

        # 统计
        self._frame_count = 0
        self._track_time_avg = 0.0

        # 初始化
        self._init_components()
        self._init_ros()

        self.get_logger().info("Initialization complete")
        self.get_logger().info(f"  Track rate: {self._track_rate} Hz")
        self.get_logger().info(f"  Tracker URL: {self._tracker_url}")
        self.get_logger().info(f"  Target frame: {self._target_frame}")

    def _load_parameters(self):
        """声明和加载参数"""
        self.declare_parameter('track_rate', 5.0)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('tracker_url', 'http://192.168.112.14:11086')
        self.declare_parameter('rgb_topic', '/top_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/top_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/top_camera/color/camera_info')
        self.declare_parameter('detection_topic', '/perception_3d/objects')
        self.declare_parameter('output_topic', '~/tracked_objects')

        self._track_rate = self.get_parameter('track_rate').value
        self._target_frame = self.get_parameter('target_frame').value
        self._tracker_url = self.get_parameter('tracker_url').value
        self._rgb_topic = self.get_parameter('rgb_topic').value
        self._depth_topic = self.get_parameter('depth_topic').value
        self._camera_info_topic = self.get_parameter('camera_info_topic').value
        self._detection_topic = self.get_parameter('detection_topic').value
        self._output_topic = self.get_parameter('output_topic').value

    def _init_components(self):
        """初始化组件"""
        # 等待相机内参 (使用轮询代替 wait_for_message)
        self.get_logger().info(f"Waiting for camera info: {self._camera_info_topic}")
        self._intrinsics = None

        # 使用一次性订阅获取内参
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._camera_info_callback,
            SENSOR_QOS
        )

        # 初始化 SAM2 在线跟踪器
        try:
            from perception_core import SAM2TrackerOnline, SimpleConfig

            cfg = SimpleConfig(
                url=self._tracker_url,
                resize=(512, 512),
                jpeg_quality=50
            )

            self._tracker = SAM2TrackerOnline(cfg)
            self.get_logger().info(f"SAM2TrackerOnline initialized: {self._tracker_url}")
        except Exception as e:
            self.get_logger().warn(f"SAM2TrackerOnline init failed (will work without tracking): {e}")
            self._tracker = None

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
            self.get_logger().info(f"Camera intrinsics received: {msg.width}x{msg.height}")

            # 初始化深度测量器
            self._depth_measurer = DepthMeasurer(
                intrinsics=self._intrinsics,
                target_frame=self._target_frame,
            )

    def _init_ros(self):
        """初始化 ROS2 接口"""
        # 订阅
        self._rgb_sub = self.create_subscription(
            Image, self._rgb_topic, self._rgb_callback, SENSOR_QOS
        )
        self._depth_sub = self.create_subscription(
            Image, self._depth_topic, self._depth_callback, SENSOR_QOS
        )
        self._det_sub = self.create_subscription(
            Object3DArray, self._detection_topic, self._detection_callback, 1
        )

        # 发布
        self._pub = self.create_publisher(
            TrackedObject3DArray, self._output_topic, RELIABLE_QOS
        )

        # 跟踪定时器
        period = 1.0 / self._track_rate
        self._timer = self.create_timer(period, self._track_callback)

    def _rgb_callback(self, msg: Image):
        """RGB 图像回调"""
        try:
            img = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            with self._lock:
                self._rgb_image = img
                self._rgb_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")

    def _depth_callback(self, msg: Image):
        """深度图像回调"""
        try:
            if msg.encoding == '16UC1':
                depth = self._bridge.imgmsg_to_cv2(msg, '16UC1')
                depth = depth.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                depth = self._bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                depth = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 1000.0

            with self._lock:
                self._depth_image = depth
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def _detection_callback(self, msg: Object3DArray):
        """检测结果回调 (异步)"""
        detections = []
        for obj in msg.objects:
            detections.append({
                'bbox': list(obj.bbox),
                'score': obj.score,
                'name': obj.category,
                'object_id': obj.object_id,
            })

        with self._lock:
            self._pending_detections = detections
            self._detection_stamp = msg.header.stamp

        if detections:
            self.get_logger().debug(f"Received {len(detections)} detections")

    def _track_callback(self):
        """跟踪循环回调"""
        t_start = time.perf_counter()

        # 检查必要组件
        if self._depth_measurer is None or self._intrinsics is None:
            return

        # 获取当前帧数据
        with self._lock:
            rgb = self._rgb_image
            depth = self._depth_image
            stamp = self._rgb_stamp
            pending_dets = self._pending_detections
            self._pending_detections = []

        if rgb is None or depth is None:
            return

        # 如果没有跟踪器，直接从检测生成跟踪结果
        if self._tracker is None:
            if pending_dets:
                self._publish_from_detections(pending_dets, depth, stamp)
            return

        # 转换检测格式
        dets_input = None
        if pending_dets:
            new_dets = []
            for d in pending_dets:
                det_bbox = list(d['bbox'])
                # 检查是否与已跟踪目标重叠 (IoU > 0.5)
                is_tracked = False
                for track_info in self._active_tracks.values():
                    if self._compute_iou(det_bbox, track_info['bbox']) > 0.5:
                        is_tracked = True
                        break
                if not is_tracked:
                    new_dets.append({
                        'dt_bbox': det_bbox,
                        'dt_score': d['score'],
                        'name': d['name'],
                    })
            if new_dets:
                dets_input = new_dets
                self.get_logger().info(
                    f"Added {len(new_dets)} new targets "
                    f"(filtered {len(pending_dets) - len(new_dets)} tracked)"
                )

        # SAM2 跟踪
        if self._frame_idx == 0 and dets_input is None:
            return

        try:
            # 如果有新目标，先添加
            if dets_input is not None:
                add_result = self._tracker.forward(
                    rgb=rgb,
                    dets=dets_input,
                    frame_idx=self._frame_idx,
                )
                self._frame_idx += 1
                if not add_result.get('success', False):
                    self.get_logger().warn(f"Add new target failed: {add_result.get('error')}")

            # 请求完整跟踪结果
            result = self._tracker.forward(
                rgb=rgb,
                dets=None,
                frame_idx=self._frame_idx,
            )
            self._frame_idx += 1
        except Exception as e:
            self.get_logger().warn(f"Tracking failed: {e}")
            return

        if not result.get('success', False):
            self.get_logger().warn(f"Track result failed: {result.get('error', 'unknown')}")
            return

        # 获取跟踪结果
        ids = result.get('ids', [])
        boxes = result.get('boxes', [])
        scores = result.get('scores', [0.0] * len(ids))
        track_scores = result.get('track_scores', scores)
        cats = result.get('cats', ['object'] * len(ids))
        masks_rle = result.get('masks', [])

        # 更新 _active_tracks
        self._active_tracks.clear()
        for i, track_id in enumerate(ids):
            bbox = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
            self._active_tracks[track_id] = {'bbox': bbox}
            self._id_to_category[track_id] = cats[i] if i < len(cats) else 'object'

        if not ids:
            return

        # 解码 RLE masks 并缩放
        orig_h, orig_w = self._intrinsics['height'], self._intrinsics['width']
        masks = self._decode_masks(masks_rle, orig_h, orig_w)

        # 构建输出消息
        output_msg = TrackedObject3DArray()
        output_msg.header.stamp = stamp if stamp else self.get_clock().now().to_msg()
        output_msg.header.frame_id = self._target_frame

        for i, track_id in enumerate(ids):
            bbox = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
            score = track_scores[i] if i < len(track_scores) else 0.0
            name = cats[i] if i < len(cats) else 'object'
            mask = masks[i] if i < len(masks) else None

            # 深度测量
            position = Point()
            distance = 0.0
            pos_conf = 0.0

            if mask is not None:
                result_depth = self._depth_measurer.measure(depth, mask)
                if result_depth.valid:
                    position.x = float(result_depth.position[0])
                    position.y = float(result_depth.position[1])
                    position.z = float(result_depth.position[2])
                    distance = result_depth.distance
                    pos_conf = result_depth.confidence

            obj_msg = TrackedObject3D()
            obj_msg.header = output_msg.header
            obj_msg.track_id = track_id
            obj_msg.category = name
            obj_msg.bbox = [float(x) for x in bbox] if bbox else [0.0, 0.0, 0.0, 0.0]
            obj_msg.position = position
            obj_msg.distance = float(distance)
            obj_msg.track_score = float(score)
            obj_msg.position_confidence = float(pos_conf)
            output_msg.objects.append(obj_msg)

        # 发布
        self._pub.publish(output_msg)

        # 统计
        t_elapsed = (time.perf_counter() - t_start) * 1000
        self._frame_count += 1
        self._track_time_avg = 0.9 * self._track_time_avg + 0.1 * t_elapsed

        if self._frame_count % 50 == 0:
            self.get_logger().info(
                f"Frame {self._frame_count}, "
                f"tracking {len(output_msg.objects)} objects, "
                f"time {t_elapsed:.1f}ms (avg {self._track_time_avg:.1f}ms)"
            )

    def _publish_from_detections(self, detections: List[Dict], depth: np.ndarray, stamp):
        """从检测结果直接发布（无跟踪器时的降级模式）"""
        output_msg = TrackedObject3DArray()
        output_msg.header.stamp = stamp if stamp else self.get_clock().now().to_msg()
        output_msg.header.frame_id = self._target_frame

        for i, det in enumerate(detections):
            bbox = det.get('bbox', [0, 0, 0, 0])

            # 简单的 bbox 中心深度测量
            if len(bbox) >= 4:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                h, w = depth.shape
                if 0 <= cy < h and 0 <= cx < w:
                    depth_val = float(depth[cy, cx])
                else:
                    depth_val = 0.0
            else:
                depth_val = 0.0

            # 计算 3D 位置
            position = Point()
            if depth_val > 0.3 and depth_val < 10.0 and self._intrinsics:
                fx = self._intrinsics['fx']
                fy = self._intrinsics['fy']
                cx_i = self._intrinsics['cx']
                cy_i = self._intrinsics['cy']
                position.x = float((cx - cx_i) * depth_val / fx)
                position.y = float((cy - cy_i) * depth_val / fy)
                position.z = float(depth_val)

            obj_msg = TrackedObject3D()
            obj_msg.header = output_msg.header
            obj_msg.track_id = i  # 使用检测索引作为临时 ID
            obj_msg.category = det.get('name', 'object')
            obj_msg.bbox = [float(x) for x in bbox] if len(bbox) >= 4 else [0.0, 0.0, 0.0, 0.0]
            obj_msg.position = position
            obj_msg.distance = float(np.linalg.norm([position.x, position.y, position.z]))
            obj_msg.track_score = float(det.get('score', 0.0))
            obj_msg.position_confidence = 0.5  # 降级模式置信度较低
            output_msg.objects.append(obj_msg)

        self._pub.publish(output_msg)

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个 bbox 的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _decode_masks(self, masks_rle: List, target_h: int, target_w: int) -> List[Optional[np.ndarray]]:
        """解码 RLE masks 并缩放到目标尺寸"""
        masks = []
        for rle in masks_rle:
            if rle is None:
                masks.append(None)
                continue
            try:
                if isinstance(rle, dict) and HAS_COCO:
                    mask_512 = coco_mask.decode(rle)
                else:
                    masks.append(None)
                    continue

                # 缩放到原图尺寸
                if mask_512.shape[:2] != (target_h, target_w):
                    mask_resized = cv2.resize(
                        mask_512.astype(np.uint8),
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_resized = mask_512.astype(np.uint8)

                masks.append(mask_resized)
            except Exception as e:
                self.get_logger().warn(f"Mask decode failed: {e}")
                masks.append(None)

        return masks

    def reset_tracker(self):
        """重置跟踪器状态"""
        self._frame_idx = 0
        self._id_to_category.clear()
        self._active_tracks.clear()
        self.get_logger().info("Tracker reset")


def main(args=None):
    rclpy.init(args=args)

    node = ObjectTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
