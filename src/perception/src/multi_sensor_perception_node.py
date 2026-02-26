#!/usr/bin/env python3
"""
Multi-Sensor Perception Node - 双相机+LiDAR联合感知节点

系统架构（针对N=20物体场景优化）:
1. 双相机并行检测 (DINO-X)
2. 3D空间匹配 (Hungarian Algorithm)
3. 传感器融合 (智能路由)
4. 多目标跟踪 (Track ID)

性能目标:
- 检测延迟: <350ms (双相机并行 + 匹配12ms)
- 跟踪频率: 5Hz
- 支持物体数: 20+ 个

Topics:
- Subscribe:
  - /camera/chassis/color/image_raw (sensor_msgs/Image)
  - /camera/chassis/aligned_depth_to_color/image_raw
  - /camera/top/color/image_raw
  - /camera/top/aligned_depth_to_color/image_raw
  - /rslidar_points (sensor_msgs/PointCloud2)

- Publish:
  - ~tracked_objects (perception/TrackedObjectArray): 跟踪结果
  - ~debug/matches (visualization_msgs/MarkerArray): 匹配可视化
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np
import cv2
import rospy
import rospkg
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

# 导入自定义消息
from perception.msg import TrackedObject3D, TrackedObject3DArray

# 导入日志
from common.logger import get_logger

# 将 src 目录添加到 Python 路径
rospack = rospkg.RosPack()
_pkg_path = rospack.get_path('perception')
_src_path = os.path.join(_pkg_path, 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# 导入检测服务和工具
from percept import DinoXDetectorOnline, DepthOptimizerOnline, DepthMeasurer

# 导入双相机匹配器
sys.path.append(os.path.dirname(__file__))
from dual_camera_matcher import (
    DualCameraMatcher, DetectionWithDepth, MatchPair
)


# ============================================================================
# 相机数据容器
# ============================================================================

class CameraData:
    """单个相机的同步数据"""
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.timestamp = None
        self.frame_id = ""


# ============================================================================
# 简单跟踪器
# ============================================================================

class SimpleTracker:
    """
    简单的多目标跟踪器

    策略：
    1. 根据3D位置 + 类别匹配轨迹
    2. 轨迹持续时间阈值（消失3帧删除）
    3. 新检测创建新轨迹
    """

    def __init__(self, max_missing_frames=3, distance_threshold=0.15):
        """
        Args:
            max_missing_frames: 轨迹消失最大帧数
            distance_threshold: 位置关联阈值（米）
        """
        self.tracks = {}  # {track_id: Track}
        self.next_track_id = 0
        self.max_missing = max_missing_frames
        self.dist_thresh = distance_threshold
        self.log = get_logger("Tracker")

    def update(self, fused_objects):
        """
        更新跟踪器

        Args:
            fused_objects: 融合后的物体列表（带position, category等）

        Returns:
            tracked_objects: 带track_id的物体列表
        """
        # Step 1: 关联现有轨迹
        unmatched_detections = []
        matched_tracks = set()

        for obj in fused_objects:
            best_track_id = None
            best_distance = float('inf')

            # 查找最近的轨迹（同类别）
            for track_id, track in self.tracks.items():
                if track.category != obj['category']:
                    continue

                dist = np.linalg.norm(track.position - obj['position'])
                if dist < self.dist_thresh and dist < best_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                # 更新现有轨迹
                self.tracks[best_track_id].update(obj)
                matched_tracks.add(best_track_id)
                obj['track_id'] = best_track_id
            else:
                # 新检测
                unmatched_detections.append(obj)

        # Step 2: 创建新轨迹
        for obj in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = Track(
                track_id=track_id,
                category=obj['category'],
                position=obj['position'],
                confidence=obj['confidence']
            )
            obj['track_id'] = track_id
            self.log.debug(f"New track {track_id}: {obj['category']}")

        # Step 3: 更新未匹配轨迹（增加缺失计数）
        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track.missing_frames += 1

                # 删除长时间缺失的轨迹
                if track.missing_frames > self.max_missing:
                    del self.tracks[track_id]
                    self.log.debug(f"Lost track {track_id}")

        return fused_objects


class Track:
    """单个轨迹"""
    def __init__(self, track_id, category, position, confidence):
        self.track_id = track_id
        self.category = category
        self.position = position
        self.confidence = confidence
        self.missing_frames = 0
        self.age = 0

    def update(self, detection):
        """更新轨迹"""
        # 简单平滑（指数移动平均）
        alpha = 0.7
        self.position = alpha * detection['position'] + (1 - alpha) * self.position
        self.confidence = detection['confidence']
        self.missing_frames = 0
        self.age += 1


# ============================================================================
# 主节点
# ============================================================================

class MultiSensorPerceptionNode:
    """多传感器感知节点"""

    def __init__(self):
        rospy.init_node('multi_sensor_perception_node')
        self.log = get_logger("MultiSensorPerception")
        self.log.info("Initializing...")

        # === 配置加载 ===
        self.config = rospy.get_param('~', {})

        # === 数据容器 ===
        self.chassis_data = CameraData()
        self.top_data = CameraData()
        self.lidar_points = None
        self.bridge = CvBridge()
        self._data_lock = threading.Lock()

        # === 线程池（双相机并行检测） ===
        self._executor = ThreadPoolExecutor(max_workers=2)

        # === 初始化检测器 ===
        self._init_detectors()

        # === 初始化匹配器 ===
        self.matcher = DualCameraMatcher(
            weight_distance=0.5,
            weight_feature=0.3,
            weight_category=0.2,
            distance_threshold=0.20,  # 20cm
            max_cost_threshold=0.7
        )
        self.log.info("Matcher initialized (Hungarian, N=20 optimized)")

        # === 初始化跟踪器 ===
        self.tracker = SimpleTracker(
            max_missing_frames=3,
            distance_threshold=0.15  # 15cm
        )

        # === 订阅相机（使用message_filters同步） ===
        self._init_camera_subscribers()

        # === 订阅LiDAR ===
        lidar_topic = self.config.get('lidar', {}).get('topic', '/rslidar_points')
        rospy.Subscriber(lidar_topic, PointCloud2, self._lidar_callback, queue_size=1)
        self.log.info(f"Subscribed LiDAR: {lidar_topic}")

        # === 发布器 ===
        self._result_pub = rospy.Publisher('~tracked_objects', TrackedObject3DArray, queue_size=1)
        self._debug_pub = rospy.Publisher('~debug/matches', MarkerArray, queue_size=1)

        # === 实时感知定时器 ===
        self._perception_rate = self.config.get('perception_rate', 5.0)  # Hz
        self._perception_timer = rospy.Timer(
            rospy.Duration(1.0 / self._perception_rate),
            self._perception_callback
        )

        self.log.info(f"Perception rate: {self._perception_rate} Hz")
        self.log.info("Ready")

    def _init_detectors(self):
        """初始化检测器（DINO-X + CDM）"""
        detector_cfg = self.config.get('dinox', {})

        # DINO-X检测器配置
        class SimpleConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        dinox_cfg = SimpleConfig(
            url=detector_cfg.get('url', 'http://localhost:7002/dinox'),
            timeout=detector_cfg.get('timeout', 30.0),
            enabled=detector_cfg.get('enabled', True),
        )

        self.detector = DinoXDetectorOnline(dinox_cfg, self.log)
        self.log.info("DINO-X detector initialized")

        # CDM深度优化器（可选）
        cdm_cfg = self.config.get('cdm', {})
        if cdm_cfg.get('enabled', False):
            cdm_config = SimpleConfig(
                url=cdm_cfg.get('url', 'http://localhost:7003/cdm'),
                timeout=cdm_cfg.get('timeout', 10.0),
                enabled=True
            )
            self.depth_optimizer = DepthOptimizerOnline(cdm_config, self.log)
            self.log.info("CDM depth optimizer initialized")
        else:
            self.depth_optimizer = None

        # 深度测量器（用于3D测量）
        self.depth_measurer_chassis = None  # 在获取内参后初始化
        self.depth_measurer_top = None

    def _init_camera_subscribers(self):
        """初始化相机订阅器"""
        chassis_cfg = self.config.get('camera_chassis', {})
        top_cfg = self.config.get('camera_top', {})

        # Chassis camera
        chassis_rgb = chassis_cfg.get('rgb_topic', '/camera/chassis/color/image_raw')
        chassis_depth = chassis_cfg.get('depth_topic', '/camera/chassis/aligned_depth_to_color/image_raw')
        chassis_info = chassis_cfg.get('info_topic', '/camera/chassis/color/camera_info')

        rospy.Subscriber(chassis_info, CameraInfo,
                        lambda msg: self._camera_info_callback(msg, 'chassis'),
                        queue_size=1)

        self._chassis_rgb_sub = message_filters.Subscriber(chassis_rgb, Image)
        self._chassis_depth_sub = message_filters.Subscriber(chassis_depth, Image)

        self._chassis_sync = message_filters.ApproximateTimeSynchronizer(
            [self._chassis_rgb_sub, self._chassis_depth_sub],
            queue_size=5, slop=0.1
        )
        self._chassis_sync.registerCallback(
            lambda rgb, depth: self._camera_sync_callback(rgb, depth, 'chassis')
        )

        self.log.info(f"Chassis camera: {chassis_rgb}, {chassis_depth}")

        # Top camera
        top_rgb = top_cfg.get('rgb_topic', '/camera/top/color/image_raw')
        top_depth = top_cfg.get('depth_topic', '/camera/top/aligned_depth_to_color/image_raw')
        top_info = top_cfg.get('info_topic', '/camera/top/color/camera_info')

        rospy.Subscriber(top_info, CameraInfo,
                        lambda msg: self._camera_info_callback(msg, 'top'),
                        queue_size=1)

        self._top_rgb_sub = message_filters.Subscriber(top_rgb, Image)
        self._top_depth_sub = message_filters.Subscriber(top_depth, Image)

        self._top_sync = message_filters.ApproximateTimeSynchronizer(
            [self._top_rgb_sub, self._top_depth_sub],
            queue_size=5, slop=0.1
        )
        self._top_sync.registerCallback(
            lambda rgb, depth: self._camera_sync_callback(rgb, depth, 'top')
        )

        self.log.info(f"Top camera: {top_rgb}, {top_depth}")

    def _camera_info_callback(self, msg, camera_name):
        """相机内参回调"""
        intrinsics = {
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5],
            'width': msg.width,
            'height': msg.height,
        }

        with self._data_lock:
            if camera_name == 'chassis':
                self.chassis_data.intrinsics = intrinsics
                self.chassis_data.frame_id = msg.header.frame_id

                # 初始化深度测量器
                if self.depth_measurer_chassis is None:
                    self.depth_measurer_chassis = DepthMeasurer(intrinsics, self.log)
                    self.log.info(f"Chassis DepthMeasurer initialized: {msg.header.frame_id}")
            else:
                self.top_data.intrinsics = intrinsics
                self.top_data.frame_id = msg.header.frame_id

                if self.depth_measurer_top is None:
                    self.depth_measurer_top = DepthMeasurer(intrinsics, self.log)
                    self.log.info(f"Top DepthMeasurer initialized: {msg.header.frame_id}")

    def _camera_sync_callback(self, rgb_msg, depth_msg, camera_name):
        """相机同步回调"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')

            # Depth: 处理不同编码
            if depth_msg.encoding == '16UC1':
                depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth = depth_raw.astype(np.float32) / 1000.0  # mm -> m
            elif depth_msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            else:
                self.log.warn(f"Unsupported depth encoding: {depth_msg.encoding}")
                return

            with self._data_lock:
                if camera_name == 'chassis':
                    self.chassis_data.rgb = rgb
                    self.chassis_data.depth = depth
                    self.chassis_data.timestamp = rgb_msg.header.stamp
                else:
                    self.top_data.rgb = rgb
                    self.top_data.depth = depth
                    self.top_data.timestamp = rgb_msg.header.stamp

        except Exception as e:
            self.log.error(f"Camera callback error ({camera_name}): {e}")

    def _lidar_callback(self, msg):
        """LiDAR点云回调"""
        # TODO: 解析PointCloud2并转换为numpy数组
        # 这里简化处理，实际需要使用sensor_msgs.point_cloud2
        pass

    def _perception_callback(self, event):
        """感知主循环"""
        try:
            t_start = time.time()

            # 检查数据是否就绪
            with self._data_lock:
                if (self.chassis_data.rgb is None or
                    self.top_data.rgb is None or
                    self.depth_measurer_chassis is None or
                    self.depth_measurer_top is None):
                    return

                # 复制数据（释放锁）
                chassis_rgb = self.chassis_data.rgb.copy()
                chassis_depth = self.chassis_data.depth.copy()
                top_rgb = self.top_data.rgb.copy()
                top_depth = self.top_data.depth.copy()

            # === Step 1: 双相机并行检测 ===
            prompt = self.config.get('detection_prompt', 'bottle.cup.box.person')

            future_chassis = self._executor.submit(
                self.detector.detect, prompt, chassis_rgb
            )
            future_top = self._executor.submit(
                self.detector.detect, prompt, top_rgb
            )

            dets_chassis_raw = future_chassis.result()
            dets_top_raw = future_top.result()

            if not dets_chassis_raw or not dets_top_raw:
                self.log.warn("Detection failed on one or both cameras")
                return

            t_detect = time.time()

            # === Step 2: 快速3D测量（用于匹配） ===
            dets_chassis = self._prepare_detections(
                dets_chassis_raw, chassis_depth, self.depth_measurer_chassis, 'chassis'
            )
            dets_top = self._prepare_detections(
                dets_top_raw, top_depth, self.depth_measurer_top, 'top'
            )

            t_measure = time.time()

            # === Step 3: 匹配 ===
            matches, unmatched_chassis, unmatched_top = self.matcher.process(
                dets_chassis, dets_top
            )

            t_match = time.time()

            # === Step 4: 构建融合物体列表 ===
            fused_objects = []

            # 匹配对
            for match in matches:
                fused_objects.append({
                    'category': match.fused_category,
                    'position': match.fused_position,
                    'confidence': match.fused_confidence,
                    'source': 'fused',
                    'quality': match.quality.value,
                })

            # 未匹配的chassis检测
            for det in unmatched_chassis:
                if det.position_3d is not None:
                    fused_objects.append({
                        'category': det.category,
                        'position': det.position_3d,
                        'confidence': det.detection_score,
                        'source': 'chassis_only',
                        'quality': 'single_view',
                    })

            # 未匹配的top检测
            for det in unmatched_top:
                if det.position_3d is not None:
                    fused_objects.append({
                        'category': det.category,
                        'position': det.position_3d,
                        'confidence': det.detection_score,
                        'source': 'top_only',
                        'quality': 'single_view',
                    })

            # === Step 5: 跟踪 ===
            tracked_objects = self.tracker.update(fused_objects)

            t_track = time.time()

            # === Step 6: 发布结果 ===
            self._publish_results(tracked_objects)

            # 性能统计
            total_time = (t_track - t_start) * 1000
            self.log.info(
                f"Perception: {len(tracked_objects)} objects | "
                f"Detect: {(t_detect-t_start)*1000:.0f}ms | "
                f"Measure: {(t_measure-t_detect)*1000:.0f}ms | "
                f"Match: {(t_match-t_measure)*1000:.0f}ms | "
                f"Track: {(t_track-t_match)*1000:.0f}ms | "
                f"Total: {total_time:.0f}ms"
            )

        except Exception as e:
            self.log.error(f"Perception callback error: {e}", exc_info=True)

    def _prepare_detections(self, raw_detections, depth, measurer, camera_name):
        """
        准备检测数据（添加3D位置和特征）

        Args:
            raw_detections: DINO-X原始检测
            depth: 深度图
            measurer: DepthMeasurer实例
            camera_name: 相机名称

        Returns:
            List[DetectionWithDepth]
        """
        detections = []

        for det in raw_detections:
            # 快速3D测量（bbox中心点）
            x1, y1, x2, y2 = det['bbox']
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            h, w = depth.shape
            if not (0 <= cy < h and 0 <= cx < w):
                continue

            depth_val = depth[cy, cx]
            if not (0.3 < depth_val < 3.0):  # 有效范围
                continue

            # 像素转3D（简化版，只计算中心点）
            try:
                point_3d = measurer.pixel_to_3d(cx, cy, depth_val)
                distance = np.linalg.norm(point_3d)
            except:
                continue

            # TODO: 提取视觉特征（DINO-X features或CLIP）
            # 这里暂时使用None，实际应该从detector获取或单独提取
            visual_features = None

            detection = DetectionWithDepth(
                bbox=np.array(det['bbox']),
                mask=det.get('mask'),  # 可能为None
                category=det['category'],
                detection_score=det['score'],
                visual_features=visual_features,
                position_3d=point_3d,
                distance=distance,
                camera_source=camera_name,
            )

            detections.append(detection)

        return detections

    def _publish_results(self, tracked_objects):
        """发布跟踪结果"""
        msg = TrackedObject3DArray()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'

        for obj in tracked_objects:
            tracked = TrackedObject3D()
            tracked.header = msg.header
            tracked.track_id = obj.get('track_id', -1)
            tracked.category = obj['category']

            # 3D位置
            tracked.position.x = obj['position'][0]
            tracked.position.y = obj['position'][1]
            tracked.position.z = obj['position'][2]
            tracked.distance = np.linalg.norm(obj['position'])

            # 置信度
            tracked.position_confidence = obj['confidence']
            tracked.track_score = obj['confidence']  # 简化：使用相同值

            # bbox（如果有的话，暂时用占位符）
            tracked.bbox = [0.0, 0.0, 0.0, 0.0]

            msg.objects.append(tracked)

        self._result_pub.publish(msg)


# ============================================================================
# 主函数
# ============================================================================

def main():
    try:
        node = MultiSensorPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node crashed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
