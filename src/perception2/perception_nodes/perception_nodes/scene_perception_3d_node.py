#!/usr/bin/env python3
"""
Scene Perception 3D - ROS2 节点

3D 场景感知节点，集成相机和 LiDAR 数据进行物体 3D 定位。

发布:
    ~/objects_3d (Object3DArray) - 检测结果
    ~/optimized_depth (Image) - CDM 优化后的深度图

服务:
    ~/detect (DetectObjects) - 按需检测服务
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from .utils import CvBridgeNumPy2 as CvBridge
from pycocotools import mask as coco_mask

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from perception_interfaces.msg import Object3D, Object3DArray, PerceptionConfig, PerceptionStatus
from perception_interfaces.srv import DetectObjects

from perception_core import (
    CoordinateTransformer,
    ScenePerceptionCore,
    DinoXDetectorOnline,
    DepthOptimizerOnline,
    SimpleConfig,
)
from .synced_sensor_subscriber import SyncedSensorSubscriber
from .utils import SENSOR_QOS, LATCHED_QOS, ThrottledLogger


class ScenePerception3DNode(Node):
    """场景 3D 感知 ROS2 节点"""

    def __init__(self):
        super().__init__('scene_perception_3d')
        self._log = ThrottledLogger(self.get_logger(), period=5.0)

        # 加载配置
        self._load_config()

        # 健康检查
        self._health_check()

        # 初始化组件
        self._init_components()

        # 内部状态
        self._current_prompt = self.default_prompt
        self._current_min_score = self.min_score
        self._current_iou_threshold = self.iou_threshold
        self._last_detect_time = 0.0
        self._last_object_count = 0
        self._last_categories = []

        # CV Bridge
        self._bridge = CvBridge()

        # 配置 QoS
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # 发布器
        self.pub_objects = self.create_publisher(
            Object3DArray, '~/objects_3d', latched_qos
        )
        self.pub_optimized_depth = self.create_publisher(
            Image, '~/optimized_depth', latched_qos
        )
        self.pub_status = self.create_publisher(
            PerceptionStatus, '/perception/status', 1
        )

        # 订阅配置 (可选)
        self._config_sub = self.create_subscription(
            PerceptionConfig, '/perception/config',
            self._config_callback, 1
        )

        # 服务
        self._detect_srv = self.create_service(
            DetectObjects, '~/detect', self.handle_detect
        )

        # 自动检测定时器
        self._auto_detect_timer = None
        if self.auto_detect_rate > 0:
            period = 1.0 / self.auto_detect_rate
            self._auto_detect_timer = self.create_timer(
                period, self._auto_detect_callback
            )
            self.get_logger().info(f'自动检测已启用: {self.auto_detect_rate} Hz')

        self.get_logger().info('节点启动完成')
        self.get_logger().info(f'  Camera: {self.camera_name}')
        self.get_logger().info(f'  LiDAR: {"启用" if self.enable_lidar else "禁用"}')
        self.get_logger().info(f'  Target frame: {self.target_frame}')
        self.get_logger().info(f'  Service: ~/detect')
        self.get_logger().info(f'  Topic: ~/objects_3d')

    def _load_config(self):
        """声明和加载 ROS2 参数"""
        # 相机配置
        self.declare_parameter('camera_name', 'top')
        self.camera_name = self.get_parameter('camera_name').value

        # LiDAR 配置
        self.declare_parameter('lidar_topic', '/lidar/chassis/point_cloud')
        self.declare_parameter('enable_lidar', True)
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.enable_lidar = self.get_parameter('enable_lidar').value

        # Prompt 配置
        self.declare_parameter('prompt_source', 'default')
        self.declare_parameter('prompt_topic', '/user/prompt')
        self.declare_parameter('default_prompt', 'bottle.cup.box')
        self.prompt_source = self.get_parameter('prompt_source').value
        self.prompt_topic = self.get_parameter('prompt_topic').value
        self.default_prompt = self.get_parameter('default_prompt').value

        # 检测服务
        self.declare_parameter('detector_url', 'http://192.168.112.14:10086')
        self.declare_parameter('detector_timeout', 3.0)
        self.declare_parameter('min_score', 0.25)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('detector_warmup', 0)
        self.detector_url = self.get_parameter('detector_url').value
        self.detector_timeout = self.get_parameter('detector_timeout').value
        self.min_score = self.get_parameter('min_score').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.detector_warmup = self.get_parameter('detector_warmup').value

        # 深度优化服务
        self.declare_parameter('depth_optimizer_url', 'http://192.168.112.14:8086')
        self.declare_parameter('enable_depth_optimizer', True)
        self.declare_parameter('depth_optimizer_warmup', 0)
        self.depth_optimizer_url = self.get_parameter('depth_optimizer_url').value
        self.enable_depth_optimizer = self.get_parameter('enable_depth_optimizer').value
        self.depth_optimizer_warmup = self.get_parameter('depth_optimizer_warmup').value

        # 坐标系配置
        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').value

        # 距离补偿
        self.declare_parameter('distance_offset', 0.30)
        self.distance_offset_default = self.get_parameter('distance_offset').value

        # 功能开关
        self.declare_parameter('enable_accuracy_log', False)
        self.enable_accuracy_log = self.get_parameter('enable_accuracy_log').value

        # 自动检测配置
        self.declare_parameter('auto_detect_rate', 1.0)
        self.auto_detect_rate = self.get_parameter('auto_detect_rate').value

        # 时间同步参数
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('data_max_age', 0.5)
        self.sync_slop = self.get_parameter('sync_slop').value
        self.data_max_age = self.get_parameter('data_max_age').value

        # 外参配置目录
        self.declare_parameter('extrinsics_dir', '')
        self.extrinsics_dir = self.get_parameter('extrinsics_dir').value

    def _health_check(self):
        """启动时健康检查"""
        import requests
        self.get_logger().info('健康检查...')

        def check_service(url, timeout=5.0):
            try:
                resp = requests.get(url, timeout=timeout, proxies={'http': None, 'https': None})
                return resp.status_code in (200, 404, 405)
            except Exception:
                return False

        # 检查 DINO-X 服务
        if not check_service(self.detector_url, timeout=5.0):
            self.get_logger().error(f'DINO-X 服务不可用: {self.detector_url}')
            raise RuntimeError('DINO-X 服务不可用')
        self.get_logger().info('  DINO-X service: OK')

        # CDM 服务可选，失败时自动禁用
        if self.enable_depth_optimizer:
            if not check_service(self.depth_optimizer_url, timeout=3.0):
                self.get_logger().warn('CDM 服务不可用，禁用深度优化')
                self.enable_depth_optimizer = False
            else:
                self.get_logger().info('  CDM service: OK')

    def _init_components(self):
        """初始化组件"""
        # 构建相机 topic
        camera_topics = {
            'color': f'/camera/{self.camera_name}/color/image_raw',
            'depth': f'/camera/{self.camera_name}/aligned_depth_to_color/image_raw',
            'info': f'/camera/{self.camera_name}/color/camera_info',
        }

        # 同步传感器订阅器
        self.sensor = SyncedSensorSubscriber(
            node=self,
            color_topic=camera_topics['color'],
            depth_topic=camera_topics['depth'],
            info_topic=camera_topics['info'],
            lidar_topic=self.lidar_topic,
            enable_lidar=self.enable_lidar,
            sync_slop=self.sync_slop,
        )

        if not self.sensor.connect(timeout=10.0):
            raise RuntimeError('传感器连接失败')

        # 坐标变换器
        self.transformer = CoordinateTransformer()
        if self.extrinsics_dir:
            self.transformer.set_config_dir(self.extrinsics_dir)
        self.transformer.load_all_extrinsics(camera_name=self.camera_name)

        # 检测器
        detector_cfg = SimpleConfig(
            url=self.detector_url,
            min_score=self.min_score,
            resize=(self.sensor.intrinsics['width'], self.sensor.intrinsics['height']),
            warmup=self.detector_warmup,
        )
        self.detector = DinoXDetectorOnline(detector_cfg)

        # 深度优化器 (可选)
        depth_optimizer = None
        if self.enable_depth_optimizer:
            optimizer_cfg = SimpleConfig(
                url=self.depth_optimizer_url,
                chosen_policy='dn',
                warmup=self.depth_optimizer_warmup,
            )
            depth_optimizer = DepthOptimizerOnline(optimizer_cfg)

        # 感知核心
        self.core = ScenePerceptionCore(
            transformer=self.transformer,
            intrinsics=self.sensor.intrinsics,
            target_frame=self.target_frame,
            depth_optimizer=depth_optimizer,
        )

    def _config_callback(self, msg: PerceptionConfig):
        """配置回调"""
        if msg.prompt:
            self._current_prompt = msg.prompt
        if msg.min_score > 0:
            self._current_min_score = msg.min_score
        if msg.iou_threshold > 0:
            self._current_iou_threshold = msg.iou_threshold
        self.get_logger().info(
            f'更新配置: prompt="{self._current_prompt}", '
            f'min_score={self._current_min_score:.2f}, iou={self._current_iou_threshold:.2f}'
        )

    def _publish_status(self):
        """发布当前状态"""
        status = PerceptionStatus()
        status.prompt = self._current_prompt
        status.min_score = float(self._current_min_score)
        status.iou_threshold = float(self._current_iou_threshold)
        status.object_count = self._last_object_count
        status.categories = self._last_categories
        status.last_detect_time = float(self._last_detect_time)
        self.pub_status.publish(status)

    def _get_distance_offset(self) -> float:
        """动态获取距离补偿参数"""
        return self.get_parameter('distance_offset').value

    def _auto_detect_callback(self):
        """自动检测定时器回调"""
        try:
            result, timings = self._run_detection(
                prompt=self._current_prompt,
                enable_lidar=self.enable_lidar
            )

            if result:
                self.pub_objects.publish(result)

            self._log.debug(f'Detection time: {timings.get("total", 0):.3f}s', period=10.0)

        except Exception as e:
            self._log.warn(f'自动检测失败: {e}', period=5.0)

    def handle_detect(self, request, response):
        """处理检测服务请求"""
        response.success = True

        try:
            result, timings = self._run_detection(
                prompt=request.prompt,
                enable_lidar=request.enable_lidar
            )

            if result:
                response.result = result
                self.pub_objects.publish(result)

            self._log_timings(timings, request.enable_lidar)

        except TimeoutError as e:
            response.success = False
            response.error_message = f'数据采集超时: {e}'
            self.get_logger().warn(response.error_message)
        except Exception as e:
            response.success = False
            response.error_message = f'检测失败: {e}'
            self.get_logger().error(response.error_message)
            import traceback
            traceback.print_exc()

        return response

    def _log_timings(self, timings: dict, enable_lidar: bool):
        """记录性能统计"""
        total = timings.get('total', 0)
        self.get_logger().info('===== 性能统计 =====')
        self.get_logger().info(f'  总耗时: {total*1000:.0f} ms')
        self.get_logger().info(f'  - 数据同步: {timings.get("data_sync", 0)*1000:.0f} ms')

        if 'parallel_time' in timings:
            detection_t = timings.get('detection', 0)
            depth_opt_t = timings.get('depth_optimize', 0)
            parallel_t = timings.get('parallel_time', 0)
            saved_t = (detection_t + depth_opt_t) - parallel_t
            self.get_logger().info(f'  - 并行执行: {parallel_t*1000:.0f} ms')
            self.get_logger().info(f'    * DINO-X: {detection_t*1000:.0f} ms')
            self.get_logger().info(f'    * CDM: {depth_opt_t*1000:.0f} ms')
            self.get_logger().info(f'    * 节省: {saved_t*1000:.0f} ms')

        self.get_logger().info(f'  - 3D测量: {timings.get("measurement_total", 0)*1000:.0f} ms')

    def _run_detection(self, prompt: str, enable_lidar: bool) -> tuple:
        """执行检测流程，返回 (Object3DArray, 性能统计字典)"""
        result = Object3DArray()
        result.header.stamp = self.get_clock().now().to_msg()
        result.frame_id = self.target_frame

        timings = {}
        total_start = time.time()

        # 1. 采集同步数据
        t0 = time.time()
        data = self.sensor.get_synced_data(
            timeout=2.0,
            max_age=self.data_max_age
        )
        timings['data_sync'] = time.time() - t0

        # 2. 获取 prompt
        final_prompt = prompt if prompt else self._current_prompt

        # 3. 并行调用 DINO-X 检测 和 CDM 深度优化
        t_parallel_start = time.time()

        def _timed_detection():
            t0 = time.time()
            result = self._detect_objects(final_prompt, data['rgb'])
            return result, time.time() - t0

        def _timed_depth_optimize():
            t0 = time.time()
            result = self.core.optimize_depth(data['rgb'], data['depth'])
            return result, time.time() - t0

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_detection = executor.submit(_timed_detection)
            future_depth = executor.submit(_timed_depth_optimize)

            detections, detection_time = future_detection.result()
            depth, depth_optimize_time = future_depth.result()

            timings['detection'] = detection_time
            timings['depth_optimize'] = depth_optimize_time

        timings['parallel_time'] = time.time() - t_parallel_start

        # 发布优化后的深度图
        try:
            depth_mm = (depth * 1000).astype(np.uint16)
            depth_msg = self._bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
            depth_msg.header.stamp = data['timestamp']
            depth_msg.header.frame_id = f'{self.camera_name}_camera_optical_frame'
            self.pub_optimized_depth.publish(depth_msg)
        except Exception as e:
            self._log.warn(f'发布优化深度图失败: {e}', period=10.0)

        if not detections:
            self.get_logger().info(f'DETECT: No objects found for "{final_prompt}"')
            timings['total'] = time.time() - total_start
            return result, timings

        self.get_logger().info(f'DETECT: Found {len(detections)} objects')

        # 5. 对每个物体进行 3D 测量
        lidar_points = data.get('lidar_points') if enable_lidar else None
        t0 = time.time()

        # 5.1 并行执行相机 3D 测量
        def _measure_camera(det):
            camera_result = self.core.camera_3d_percept(depth, det['mask'], skip_transform=True)
            return det, camera_result

        with ThreadPoolExecutor(max_workers=4) as executor:
            camera_results = list(executor.map(_measure_camera, detections))

        # 过滤无效测量
        valid_items = []
        optical_centroids = []
        for det, camera_result in camera_results:
            if camera_result.valid:
                valid_items.append((det, camera_result))
                optical_centroids.append(camera_result.centroid_optical)

        timings['measurement_camera'] = time.time() - t0

        if not valid_items:
            self.get_logger().warn('DETECT: All camera measurements failed')
            timings['measurement_total'] = time.time() - t0
            timings['total'] = time.time() - total_start
            return result, timings

        # 5.2 批量坐标变换
        optical_points = np.array(optical_centroids)
        target_centroids = self.transformer.optical_to_target(optical_points, self.target_frame)

        # 5.3 构建 Object3D 消息
        distance_offset = self._get_distance_offset()

        for i, (det, camera_result) in enumerate(valid_items):
            obj = Object3D()
            obj.object_id = det['object_id']
            obj.category = det['category']
            obj.score = float(det['score'])
            obj.bbox = [float(x) for x in det['bbox']]

            # 使用批量变换后的坐标
            target_centroid = target_centroids[i]

            # 应用距离补偿
            original_distance = np.linalg.norm(target_centroid)
            if original_distance > 0.01:
                scale_factor = (original_distance + distance_offset) / original_distance
                compensated_centroid = target_centroid * scale_factor
            else:
                compensated_centroid = target_centroid

            obj.position = Point(
                x=float(compensated_centroid[0]),
                y=float(compensated_centroid[1]),
                z=float(compensated_centroid[2])
            )
            obj.distance = float(np.linalg.norm(compensated_centroid))
            obj.confidence = float(camera_result.confidence)

            # 保存相机光学坐标系下的原始位置 (用于调试)
            optical_centroid = optical_centroids[i]
            obj.position_optical = Point(
                x=float(optical_centroid[0]),
                y=float(optical_centroid[1]),
                z=float(optical_centroid[2])
            )
            obj.depth = float(optical_centroid[2])  # z 值即为深度
            obj.source_camera = self.camera_name

            # LiDAR 3D 感知 (可选)
            if lidar_points is not None and len(lidar_points) > 0:
                camera_depth = camera_result.stats.get('depth_median', None)
                lidar_result = self.core.lidar_3d_percept(
                    lidar_points, det['bbox'], camera_depth
                )

                if lidar_result.valid:
                    obj.position_lidar = Point(
                        x=float(lidar_result.centroid[0]),
                        y=float(lidar_result.centroid[1]),
                        z=float(lidar_result.centroid[2])
                    )
                    obj.distance_lidar = float(lidar_result.distance)
                    obj.confidence_lidar = float(lidar_result.confidence)

            result.objects.append(obj)

        timings['measurement_total'] = time.time() - t0
        timings['total'] = time.time() - total_start

        # 更新状态
        self._last_detect_time = timings['total']
        self._last_object_count = len(result.objects)
        self._last_categories = list(set(obj.category for obj in result.objects))
        self._publish_status()

        self.get_logger().info(f'DETECT: Measured {len(result.objects)} objects successfully')

        return result, timings

    def _detect_objects(self, prompt: str, rgb: np.ndarray) -> list:
        """检测物体并解码 mask"""
        result = self.detector.forward(
            text=prompt, rgb=rgb,
            min_score=self._current_min_score,
            iou_threshold=self._current_iou_threshold
        )
        objects = result.result.get('objects', [])

        if not objects:
            return []

        # 批量解码 masks
        rle_masks = []
        bbox_masks = []
        obj_indices = []

        for i, obj in enumerate(objects):
            mask_rle = obj.get('mask')
            if mask_rle:
                rle_masks.append(mask_rle)
                obj_indices.append(i)
            else:
                bbox_masks.append((i, obj['bbox']))

        # 批量解码 RLE masks
        decoded_masks = {}
        if rle_masks:
            masks_array = coco_mask.decode(rle_masks)
            if masks_array.ndim == 2:
                masks_array = masks_array[:, :, np.newaxis]
            for idx, obj_idx in enumerate(obj_indices):
                decoded_masks[obj_idx] = masks_array[:, :, idx].astype(np.uint8)

        # 处理 bbox masks
        for obj_idx, bbox in bbox_masks:
            binary_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            binary_mask[y1:y2, x1:x2] = 1
            decoded_masks[obj_idx] = binary_mask

        # 构建检测结果
        detections = []
        category_counts = {}

        for i, obj in enumerate(objects):
            binary_mask = decoded_masks.get(i)
            if binary_mask is None:
                continue

            if binary_mask.sum() < 300:
                continue

            category = obj.get('category', prompt)
            score = obj.get('score', 0)

            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
            obj_id = f"{category}_{category_counts[category]}"

            detections.append({
                'object_id': obj_id,
                'category': category,
                'score': score,
                'bbox': obj['bbox'],
                'mask': binary_mask,
            })

        return detections


def main(args=None):
    rclpy.init(args=args)

    node = ScenePerception3DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
