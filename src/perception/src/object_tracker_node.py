#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Tracker Node

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

SAM2 服务端行为:
    - 当 dets 不为空: 只返回新检测的目标 (用于初始化)
    - 当 dets 为空: 返回所有正在跟踪的目标
    - frame_idx=0 会触发 tracker reset
    - masks 返回 RLE 格式, 256x256 尺寸, 需解码并缩放

订阅:
    /top_camera/color/image_raw (sensor_msgs/Image)
    /top_camera/aligned_depth_to_color/image_raw (sensor_msgs/Image)
    /perception_3d/objects (perception/Object3DArray)

发布:
    /object_tracker/tracked_objects (perception/TrackedObject3DArray)
"""

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from common.logger import get_logger
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from pycocotools import mask as coco_mask

# 添加 perception/src 路径
PERCEPTION_SRC = os.path.dirname(os.path.abspath(__file__))
if PERCEPTION_SRC not in sys.path:
    sys.path.insert(0, PERCEPTION_SRC)


# ============================================================================
# DepthMeasurer: 从 mask 测量 3D 位置
# ============================================================================

@dataclass
class MeasurementResult:
    """深度测量结果"""
    valid: bool = False
    position: Optional[np.ndarray] = None  # [x, y, z] in target frame
    distance: float = 0.0
    confidence: float = 0.0
    error_msg: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class DepthMeasurer:
    """
    从 mask 测量 3D 位置

    基于 ScenePerceptionCore.camera_3d_percept() 的核心算法：
    1. Mask 腐蚀去噪
    2. 深度范围过滤
    3. IQR 异常值剔除
    4. 向量化反投影
    5. 中值质心计算
    6. 坐标系变换
    """

    # 算法参数
    DEPTH_MIN = 0.3   # m
    DEPTH_MAX = 10.0  # m
    MASK_ERODE_KERNEL = 5
    IQR_FACTOR = 1.5
    MIN_DEPTH_POINTS = 10

    def __init__(
        self,
        intrinsics: dict,
        transformer=None,
        target_frame: str = 'base_link'
    ):
        """
        Args:
            intrinsics: 相机内参 {fx, fy, cx, cy}
            transformer: CoordinateTransformer (可选)
            target_frame: 目标坐标系
        """
        self.intrinsics = intrinsics
        self.transformer = transformer
        self.target_frame = target_frame

        # 预计算腐蚀核
        self._erode_kernel = np.ones(
            (self.MASK_ERODE_KERNEL, self.MASK_ERODE_KERNEL), np.uint8
        )

    def measure(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        skip_transform: bool = False
    ) -> MeasurementResult:
        """
        测量 mask 区域的 3D 位置

        Args:
            depth: 深度图 (H, W) float32, 单位: 米
            mask: 二值 mask (H, W) uint8 或 bool
            skip_transform: 跳过坐标变换 (返回 optical 坐标系)

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
        centroid_optical = np.median(points, axis=0)

        # 7. 坐标变换
        if skip_transform or self.transformer is None:
            centroid = centroid_optical
            distance = np.linalg.norm(centroid_optical)
        else:
            centroid = self.transformer.optical_to_target(
                centroid_optical.reshape(1, 3), self.target_frame
            )[0]
            distance = np.linalg.norm(centroid)

        # 8. 计算置信度
        depth_std = np.std(ds_f)
        confidence = self._compute_confidence(len(points), depth_std)

        result.valid = True
        result.position = centroid
        result.distance = distance
        result.confidence = confidence
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
# ObjectTrackerNode: ROS 节点
# ============================================================================

class ObjectTrackerNode:
    """
    物体跟踪节点

    架构:
        检测 (异步, 2Hz)                  跟踪 (5Hz, 200ms)
               │                               │
               │  Object3D[]                   │  RGB + Depth
               ▼                               ▼
        ┌──────────────┐              ┌──────────────────┐
        │ 新目标加入    │ ──────────→ │  SAM2 2D 跟踪    │
        │ 丢失目标移除  │              │  (bbox + mask)   │
        └──────────────┘              └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  深度测量 (mask)  │
                                      │  → 3D position   │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  发布结果         │
                                      │ TrackedObject3D[]│
                                      └──────────────────┘
    """

    def __init__(self):
        rospy.init_node('object_tracker_node', anonymous=False)
        self.log = get_logger("ObjectTracker")

        # 参数
        self.track_rate = rospy.get_param('~track_rate', 5.0)  # Hz
        self.target_frame = rospy.get_param('~target_frame', 'base_link')

        # SAM2 Tracker 服务地址
        self.tracker_url = rospy.get_param('~tracker_url', 'http://192.168.112.14:11086')

        # 相机话题
        self.rgb_topic = rospy.get_param('~rgb_topic', '/top_camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/top_camera/aligned_depth_to_color/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/top_camera/color/camera_info')
        self.detection_topic = rospy.get_param('~detection_topic', '/perception_3d/objects')
        self.output_topic = rospy.get_param('~output_topic', '/object_tracker/tracked_objects')

        # 内部状态
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # 图像缓存
        self._rgb_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self._rgb_stamp: Optional[rospy.Time] = None

        # 检测缓存 (异步更新，可增量添加新目标)
        self._pending_detections: List[Dict] = []
        self._detection_stamp: Optional[rospy.Time] = None

        # 相机内参
        self._intrinsics: Optional[dict] = None

        # 跟踪器 (延迟初始化)
        self._tracker = None
        self._depth_measurer: Optional[DepthMeasurer] = None
        self._transformer = None

        # 跟踪状态
        self._frame_idx = 0  # SAM2 帧计数器
        self._tracker_initialized = False

        # ID 到类别的映射
        self._id_to_category: Dict[int, str] = {}

        # 活跃跟踪目标缓存 (用于融合新检测和已有跟踪)
        # key: track_id, value: 最新的跟踪结果
        self._active_tracks: Dict[int, Dict] = {}

        # 统计
        self._frame_count = 0
        self._track_time_avg = 0.0

        # 初始化
        self._init_components()
        self._init_ros()

        self.log.info("Initialization complete")
        self.log.info(f"  Track rate: {self.track_rate} Hz")
        self.log.info(f"  Tracker URL: {self.tracker_url}")
        self.log.info(f"  Target frame: {self.target_frame}")

    def _init_components(self):
        """初始化组件"""
        # 等待相机内参
        self.log.info(f"Waiting for camera info: {self.camera_info_topic}")
        try:
            info_msg = rospy.wait_for_message(
                self.camera_info_topic, CameraInfo, timeout=10.0
            )
            self._intrinsics = {
                'fx': info_msg.K[0],
                'fy': info_msg.K[4],
                'cx': info_msg.K[2],
                'cy': info_msg.K[5],
                'width': info_msg.width,
                'height': info_msg.height,
            }
            self.log.info(f"Camera intrinsics: {self._intrinsics}")
        except rospy.ROSException:
            self.log.warn("No camera info received, using defaults")
            self._intrinsics = {
                'fx': 607.0, 'fy': 607.0,
                'cx': 640.0, 'cy': 360.0,
                'width': 1280, 'height': 720,
            }

        # 初始化坐标变换器
        try:
            from coordinate_transformer import CoordinateTransformer
            self._transformer = CoordinateTransformer()
            self._transformer.load_all_extrinsics()
            self.log.info("CoordinateTransformer loaded")
        except Exception as e:
            self.log.warn(f"CoordinateTransformer load failed: {e}")
            self._transformer = None

        # 初始化深度测量器
        self._depth_measurer = DepthMeasurer(
            intrinsics=self._intrinsics,
            transformer=self._transformer,
            target_frame=self.target_frame,
        )

        # 初始化 SAM2 在线跟踪器
        try:
            from percept import SAM2TrackerOnline
            from mmengine.config import Config

            cfg = Config()
            cfg.url = self.tracker_url
            cfg.resize = (512, 512)  # SAM2 tracker 固定尺寸
            cfg.jpeg_quality = 50

            self._tracker = SAM2TrackerOnline(cfg)
            self.log.info(f"SAM2TrackerOnline initialized: {self.tracker_url}")
        except Exception as e:
            self.log.error(f"SAM2TrackerOnline init failed: {e}")
            raise

    def _init_ros(self):
        """初始化 ROS 接口"""
        # 导入消息类型
        from perception.msg import Object3DArray, TrackedObject3D, TrackedObject3DArray

        self._TrackedObject3D = TrackedObject3D
        self._TrackedObject3DArray = TrackedObject3DArray

        # 订阅
        self._rgb_sub = rospy.Subscriber(
            self.rgb_topic, Image, self._rgb_callback, queue_size=1
        )
        self._depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self._depth_callback, queue_size=1
        )
        self._det_sub = rospy.Subscriber(
            self.detection_topic, Object3DArray, self._detection_callback, queue_size=1
        )

        # 发布
        self._pub = rospy.Publisher(
            self.output_topic, TrackedObject3DArray, queue_size=1
        )

        # 跟踪定时器
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self.track_rate), self._track_callback
        )

    def _rgb_callback(self, msg: Image):
        """RGB 图像回调"""
        try:
            # 使用 passthrough 避免调用 cv_bridge_boost.cvtColor2()
            # (cv_bridge_boost.so 针对 OpenCV 4.2 编译，与当前 OpenCV 4.12 不兼容)
            img = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
            # 如果是 RGB 格式，手动转换为 BGR
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            with self._lock:
                self._rgb_image = img
                self._rgb_stamp = msg.header.stamp
        except Exception as e:
            self.log.warn(f"RGB conversion failed: {e}")

    def _depth_callback(self, msg: Image):
        """深度图像回调"""
        try:
            if msg.encoding == '16UC1':
                depth = self._bridge.imgmsg_to_cv2(msg, '16UC1')
                depth = depth.astype(np.float32) / 1000.0  # mm -> m
            elif msg.encoding == '32FC1':
                depth = self._bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                depth = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 1000.0

            with self._lock:
                self._depth_image = depth
        except Exception as e:
            self.log.warn(f"Depth conversion failed: {e}")

    def _detection_callback(self, msg):
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
            self.log.debug(f"Received {len(detections)} detections")

    def _track_callback(self, event):
        """跟踪循环回调"""
        t_start = time.perf_counter()

        # 获取当前帧数据
        with self._lock:
            rgb = self._rgb_image
            depth = self._depth_image
            stamp = self._rgb_stamp
            pending_dets = self._pending_detections
            self._pending_detections = []  # 清空已处理的检测

        if rgb is None or depth is None:
            return

        # 转换检测格式: Object3D -> SAM2TrackerOnline 格式
        # 过滤已跟踪目标，只传入真正的新目标
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
                self.log.info(f"Added {len(new_dets)} new targets (filtered {len(pending_dets) - len(new_dets)} tracked)")

        # SAM2 跟踪
        # frame_idx=0 必须有检测结果来初始化 tracker，否则服务端返回 500
        if self._frame_idx == 0 and dets_input is None:
            return  # 等待第一个检测结果

        try:
            # 如果有新目标，先添加新目标
            if dets_input is not None:
                add_result = self._tracker.forward(
                    rgb=rgb,
                    dets=dets_input,
                    frame_idx=self._frame_idx,
                )
                self._frame_idx += 1
                if not add_result.get('success', False):
                    self.log.warn(f"Add new target failed: {add_result.get('error')}")

            # 请求完整跟踪结果 (dets=None 返回所有目标)
            result = self._tracker.forward(
                rgb=rgb,
                dets=None,
                frame_idx=self._frame_idx,
            )
            self._frame_idx += 1
        except Exception as e:
            self.log.warn(f"Tracking failed: {e}")
            return

        if not result.get('success', False):
            self.log.warn(f"Track result failed: {result.get('error', 'unknown')}")
            return

        # 获取跟踪结果
        ids = result.get('ids', [])
        boxes = result.get('boxes', [])
        scores = result.get('scores', [0.0] * len(ids))
        track_scores = result.get('track_scores', scores)
        cats = result.get('cats', ['object'] * len(ids))
        masks_rle = result.get('masks', [])

        # 更新 _active_tracks 用于下次 IoU 过滤
        # (只记录 bbox，不做缓存发布)
        self._active_tracks.clear()
        for i, track_id in enumerate(ids):
            bbox = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
            self._active_tracks[track_id] = {'bbox': bbox}
            self._id_to_category[track_id] = cats[i] if i < len(cats) else 'object'

        # 没有跟踪结果则返回
        if not ids:
            return

        # 解码 RLE masks 并缩放到原图尺寸
        orig_h, orig_w = self._intrinsics['height'], self._intrinsics['width']
        masks = self._decode_masks(masks_rle, orig_h, orig_w)

        # 构建输出消息 - 直接发布服务端返回的结果
        output_msg = self._TrackedObject3DArray()
        output_msg.header = Header()
        output_msg.header.stamp = stamp if stamp else rospy.Time.now()
        output_msg.header.frame_id = self.target_frame

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

            obj_msg = self._TrackedObject3D()
            obj_msg.header = output_msg.header
            obj_msg.track_id = track_id
            obj_msg.category = name
            obj_msg.bbox = bbox
            obj_msg.position = position
            obj_msg.distance = distance
            obj_msg.track_score = score
            obj_msg.position_confidence = pos_conf
            output_msg.objects.append(obj_msg)

        # 发布
        self._pub.publish(output_msg)

        # 统计
        t_elapsed = (time.perf_counter() - t_start) * 1000
        self._frame_count += 1
        self._track_time_avg = 0.9 * self._track_time_avg + 0.1 * t_elapsed

        if self._frame_count % 50 == 0:
            self.log.info(
                f"Frame {self._frame_count}, "
                f"tracking {len(output_msg.objects)} objects, "
                f"time {t_elapsed:.1f}ms (avg {self._track_time_avg:.1f}ms)"
            )

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个 bbox 的 IoU

        Args:
            box1, box2: [x1, y1, x2, y2] 格式的边界框

        Returns:
            IoU 值 (0~1)
        """
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

    def _decode_masks(self, masks_rle: List, target_h: int, target_w: int) -> List[np.ndarray]:
        """解码 RLE masks 并缩放到目标尺寸

        Args:
            masks_rle: RLE 格式的 mask 列表 (SAM2 返回 256x256)
            target_h: 目标高度 (原图高度)
            target_w: 目标宽度 (原图宽度)

        Returns:
            解码并缩放后的 mask 列表
        """
        masks = []
        for rle in masks_rle:
            if rle is None:
                masks.append(None)
                continue
            try:
                # 解码 RLE
                if isinstance(rle, dict):
                    # pycocotools 格式
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
                self.log.warn(f"Mask decode failed: {e}")
                masks.append(None)

        return masks

    def reset_tracker(self):
        """重置跟踪器状态"""
        self._frame_idx = 0
        self._id_to_category.clear()
        self._active_tracks.clear()
        self.log.info("Tracker reset")

    def run(self):
        """运行节点"""
        self.log.info("Starting")
        rospy.spin()


def main():
    try:
        node = ObjectTrackerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[ObjectTracker] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
