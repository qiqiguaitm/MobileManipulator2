#!/usr/bin/env python3
"""
Perception Grasp Node - 基于手部相机的抓取感知节点 (ROS2)

功能:
- 订阅手部相机 RGB + Depth
- 三服务并行: GraspAnything + DinoX + CDM
- 输出相机光学坐标系下的 3D 抓取位姿
- 发布 GraspObjectArray (所有检测物体) 和优化后深度图

发布 Topics:
- ~/result (GraspObjectArray): 所有检测物体 + chosen_index
- ~/depth (sensor_msgs/Image): CDM优化后深度图 (16UC1, mm)

Service:
- ~/detect (GraspDetect): 按需检测服务
"""

import os
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from .utils import CvBridgeNumPy2 as CvBridge

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Point

from perception_interfaces.msg import GraspObject, GraspObjectArray
from perception_interfaces.srv import GraspDetect

from perception_core import (
    GraspAnythingOnline,
    DinoXDetectorOnline,
    DepthOptimizerOnline,
    SimpleConfig,
)

# pycocotools for mask RLE encoding
try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

# 传感器 QoS (兼容 RealSense 等 RELIABLE 发布者)
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# Latched QoS
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)


def _get_default_grasp_server_list() -> str:
    """获取默认的 GraspAnything 服务器配置文件路径"""
    try:
        config_dir = get_package_share_directory('perception_bringup')
        config_path = os.path.join(config_dir, 'config', 'server_grasp.json')
        if os.path.isfile(config_path):
            return config_path
    except Exception:
        pass
    return ''


class PerceptionGraspNode(Node):
    """抓取感知 ROS2 节点"""

    def __init__(self):
        super().__init__('perception_grasp_node')

        # === 加载参数 ===
        self._load_parameters()

        # === 并发控制 ===
        self._detect_lock = threading.Lock()
        self._detecting = False

        # === 持久化线程池 ===
        self._executor = ThreadPoolExecutor(max_workers=3)

        # === 图像缓存 ===
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self._bridge = CvBridge()
        self._data_lock = threading.Lock()

        # === 订阅 CameraInfo ===
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._camera_info_callback,
            SENSOR_QOS
        )
        self.get_logger().info(f'订阅 CameraInfo: {self._camera_info_topic}')

        # === 订阅 RGB 和 Depth (使用 message_filters 同步) ===
        try:
            import message_filters
            self._rgb_sub = message_filters.Subscriber(self, Image, self._rgb_topic, qos_profile=SENSOR_QOS)
            self._depth_sub = message_filters.Subscriber(self, Image, self._depth_topic, qos_profile=SENSOR_QOS)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub],
                queue_size=5,
                slop=0.1
            )
            self._sync.registerCallback(self._sync_callback)
            self.get_logger().info(f'同步订阅: {self._rgb_topic}, {self._depth_topic}')
        except ImportError:
            self.get_logger().error('message_filters 不可用，请安装 ros-humble-message-filters')
            raise

        # === 初始化检测服务 ===
        self._init_detectors()

        # === ROS2 Service ===
        self._detect_srv = self.create_service(
            GraspDetect,
            '~/detect',
            self._detect_service_callback
        )
        self.get_logger().info('服务: ~/detect')

        # === 结果发布器 ===
        self._result_pub = self.create_publisher(
            GraspObjectArray,
            '~/result',
            LATCHED_QOS
        )
        self.get_logger().info('发布结果: ~/result')

        # === 深度图发布器 ===
        self._depth_pub = self.create_publisher(
            Image,
            '~/depth',
            LATCHED_QOS
        )
        self.get_logger().info('发布深度: ~/depth')

        # === 订阅 prompt 更新 ===
        self._prompt_sub = self.create_subscription(
            String,
            self._prompt_topic,
            self._prompt_update_callback,
            1
        )

        # === 实时检测定时器 ===
        if self._realtime_enabled and self._realtime_rate > 0:
            period = 1.0 / self._realtime_rate
            self._realtime_timer = self.create_timer(period, self._realtime_timer_callback)
            self.get_logger().info(f'实时检测: {self._realtime_rate} Hz')
        else:
            self._realtime_timer = None

        # === 预热 ===
        if self._warmup_enabled:
            self._warmup()

        self.get_logger().info('节点初始化完成')

    def _load_parameters(self):
        """声明和加载参数"""
        # 相机配置
        self.declare_parameter('camera.rgb_topic', '/camera/hand/color/image_raw')
        self.declare_parameter('camera.depth_topic', '/camera/hand/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera.camera_info_topic', '/camera/hand/color/camera_info')

        self._rgb_topic = self.get_parameter('camera.rgb_topic').value
        self._depth_topic = self.get_parameter('camera.depth_topic').value
        self._camera_info_topic = self.get_parameter('camera.camera_info_topic').value

        # 触发模式
        self.declare_parameter('trigger.realtime_mode.enabled', True)
        self.declare_parameter('trigger.realtime_mode.rate', 1.0)
        self.declare_parameter('trigger.realtime_mode.default_prompt', 'pen.box.phone.bottle.toy')
        self.declare_parameter('trigger.realtime_mode.prompt_topic', '/usr/prompt/grasp')

        self._realtime_enabled = self.get_parameter('trigger.realtime_mode.enabled').value
        self._realtime_rate = self.get_parameter('trigger.realtime_mode.rate').value
        self._realtime_prompt = self.get_parameter('trigger.realtime_mode.default_prompt').value
        self._prompt_topic = self.get_parameter('trigger.realtime_mode.prompt_topic').value

        # 服务配置
        self.declare_parameter('services.grasp.server_list', _get_default_grasp_server_list())
        self.declare_parameter('services.grasp.model_name', 'full')
        self.declare_parameter('services.grasp.warmup', True)

        self.declare_parameter('services.dinox.url', 'http://192.168.112.14:10086')
        self.declare_parameter('services.cdm.url', 'http://192.168.112.14:8086')
        self.declare_parameter('services.cdm.enabled', True)

        self._grasp_server_list = self.get_parameter('services.grasp.server_list').value
        self._grasp_model_name = self.get_parameter('services.grasp.model_name').value
        self._warmup_enabled = self.get_parameter('services.grasp.warmup').value
        self._dinox_url = self.get_parameter('services.dinox.url').value
        self._cdm_url = self.get_parameter('services.cdm.url').value
        self._cdm_enabled = self.get_parameter('services.cdm.enabled').value

        # 深度配置
        self.declare_parameter('depth.min_depth', 0.20)
        self.declare_parameter('depth.max_depth', 2.0)

        self._min_depth = self.get_parameter('depth.min_depth').value
        self._max_depth = self.get_parameter('depth.max_depth').value

        # 过滤配置
        self.declare_parameter('filter.min_grasp_score', 0.3)
        self.declare_parameter('filter.min_object_score', 0.25)
        self.declare_parameter('filter.max_gripper_width', 0.090)
        self.declare_parameter('filter.bbox_size_filter.enabled', True)
        self.declare_parameter('filter.bbox_size_filter.size_ratio', 2.0)

        self._min_grasp_score = self.get_parameter('filter.min_grasp_score').value
        self._min_object_score = self.get_parameter('filter.min_object_score').value
        self._max_gripper_width = self.get_parameter('filter.max_gripper_width').value
        self._bbox_filter_enabled = self.get_parameter('filter.bbox_size_filter.enabled').value
        self._bbox_size_ratio = self.get_parameter('filter.bbox_size_filter.size_ratio').value

        # Chosen 策略
        self.declare_parameter('chosen_policy.grasp_score_weight', 0.2)
        self.declare_parameter('chosen_policy.distance_weight', 0.8)
        self.declare_parameter('chosen_policy.distance_mode', 'reciprocal')
        self.declare_parameter('chosen_policy.normalize_distance', True)
        self.declare_parameter('chosen_policy.debug', False)

        self._grasp_weight = self.get_parameter('chosen_policy.grasp_score_weight').value
        self._dist_weight = self.get_parameter('chosen_policy.distance_weight').value
        self._dist_mode = self.get_parameter('chosen_policy.distance_mode').value
        self._normalize_dist = self.get_parameter('chosen_policy.normalize_distance').value
        self._debug_chosen = self.get_parameter('chosen_policy.debug').value

        # 坐标系
        self.declare_parameter('frame_id', 'camera/hand_color_optical_frame')
        self._frame_id = self.get_parameter('frame_id').value

    def _init_detectors(self):
        """初始化检测服务"""
        # GraspAnythingOnline
        if self._grasp_server_list:
            grasp_config = SimpleConfig(
                server_list=self._grasp_server_list,
                model_name=self._grasp_model_name,
                resize=(1280, 720),
                warmup=0
            )
            self.grasp_detector = GraspAnythingOnline(grasp_config)
            self.get_logger().info(f'GraspAnything: {self._grasp_server_list}')
        else:
            self.grasp_detector = None
            self.get_logger().warn('GraspAnything 未配置')

        # DinoXDetectorOnline
        dinox_config = SimpleConfig(
            url=self._dinox_url,
            min_score=self._min_object_score
        )
        self.object_detector = DinoXDetectorOnline(dinox_config)
        self.get_logger().info(f'DinoX: {self._dinox_url}')

        # DepthOptimizerOnline
        cdm_config = SimpleConfig(
            url=self._cdm_url,
            chosen_policy='dn',
            warmup=0
        )
        self.depth_optimizer = DepthOptimizerOnline(cdm_config)
        self.get_logger().info(f'CDM: {self._cdm_url}')

    def _warmup(self):
        """预热 GraspAnything"""
        if self.grasp_detector is None:
            return
        self.get_logger().info('预热 GraspAnything...')
        try:
            dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            self.grasp_detector.forward(dummy_rgb)
            self.get_logger().info('预热完成')
        except Exception as e:
            self.get_logger().warn(f'预热失败: {e}')

    def _camera_info_callback(self, msg: CameraInfo):
        """相机内参回调"""
        if self.intrinsics is None:
            self.intrinsics = {
                'width': msg.width,
                'height': msg.height,
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
            }
            self.get_logger().info(f'内参: {msg.width}x{msg.height}')

    def _sync_callback(self, rgb_msg: Image, depth_msg: Image):
        """同步回调"""
        try:
            rgb_raw = self._bridge.imgmsg_to_cv2(rgb_msg, 'passthrough')
            if rgb_msg.encoding == 'rgb8':
                rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
            elif rgb_msg.encoding == 'bgr8':
                rgb = rgb_raw
            else:
                rgb = rgb_raw

            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth = depth_mm.astype(np.float32) / 1000.0

            with self._data_lock:
                self.rgb = rgb
                self.depth = depth
        except Exception as e:
            self.get_logger().error(f'图像解码失败: {e}')

    def _prompt_update_callback(self, msg: String):
        """更新检测 prompt"""
        new_prompt = msg.data.strip()
        if new_prompt and new_prompt != self._realtime_prompt:
            self.get_logger().info(f'Prompt 更新: {self._realtime_prompt} -> {new_prompt}')
            self._realtime_prompt = new_prompt

    def _realtime_timer_callback(self):
        """实时检测回调"""
        if self.rgb is None or self.depth is None:
            return
        self.detect(self._realtime_prompt, enable_cdm=self._cdm_enabled, wait_for_lock=False)

    def _detect_service_callback(self, request: GraspDetect.Request, response: GraspDetect.Response):
        """服务回调"""
        result = self.detect(request.prompt, request.enable_cdm, wait_for_lock=True)

        response.success = result['success']
        if result['success']:
            response.point3d = result['point3d']
            response.width3d = result['width3d']
            response.angle = result['angle']
            response.category = result['category']
            response.score = float(result.get('score', 0.0))
            response.center_uv = [float(x) for x in result.get('center_uv', [0.0, 0.0])]
            response.depth = float(result.get('depth_value', 0.0))
        else:
            response.error_message = result.get('error_message', 'Detection failed')
        response.detection_time_ms = result.get('detection_time_ms', 0.0)
        return response

    def detect(self, prompt: str, enable_cdm: bool = True, wait_for_lock: bool = True) -> Dict[str, Any]:
        """执行检测"""
        # 并发控制
        if wait_for_lock:
            acquired = self._detect_lock.acquire(blocking=True, timeout=5.0)
            if not acquired:
                return {'success': False, 'error_message': 'Detection timeout'}
        else:
            if not self._detect_lock.acquire(blocking=False):
                return {'success': False, 'error_message': 'Detection in progress'}

        try:
            self._detecting = True
            t0 = time.time()

            # 获取当前帧
            with self._data_lock:
                rgb = self.rgb.copy() if self.rgb is not None else None
                depth = self.depth.copy() if self.depth is not None else None

            if rgb is None or depth is None:
                return {'success': False, 'error_message': 'No image available'}
            if self.intrinsics is None:
                return {'success': False, 'error_message': 'Camera intrinsics not ready'}

            img_h, img_w = rgb.shape[:2]
            depth_mm = (depth * 1000).astype(np.uint16)

            # 确保数组连续
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            if not depth_mm.flags['C_CONTIGUOUS']:
                depth_mm = np.ascontiguousarray(depth_mm)

            # 并行执行服务
            results = {}
            futures = {}

            if self.grasp_detector:
                futures['grasp'] = self._executor.submit(self.grasp_detector.forward, rgb)
            futures['dinox'] = self._executor.submit(self.object_detector.forward, prompt, rgb)
            if enable_cdm:
                futures['cdm'] = self._executor.submit(self.depth_optimizer.forward, rgb, depth_mm, 'dn')

            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=10)
                except Exception as e:
                    self.get_logger().error(f'{name}: 调用失败: {e}')
                    results[name] = None

            # 处理 CDM 结果
            if enable_cdm and results.get('cdm') and results['cdm'].get('success'):
                depth_optimized = results['cdm']['depth'].astype(np.float32) / 1000.0
                depth_optimized_mm = results['cdm']['depth']
            else:
                depth_optimized = depth
                depth_optimized_mm = depth_mm

            # 处理 GraspAnything 结果
            grasp_result = results.get('grasp')
            if grasp_result is not None:
                affs, _ = grasp_result
            else:
                affs = [[]]

            # 解析 DinoX 结果
            dinox_result = results.get('dinox')
            if dinox_result and hasattr(dinox_result, 'result'):
                targets = dinox_result.result.get('objects', [])
            else:
                targets = []

            self.get_logger().info(f'DETECT: GraspAnything={len(affs[0]) if affs and affs[0] else 0}, DinoX={len(targets)}')

            # 构建 GraspObjectArray
            grasp_objects, chosen_idx = self._build_grasp_objects(
                rgb, depth_optimized, affs, targets, img_w, img_h
            )

            detection_time_ms = (time.time() - t0) * 1000

            # 发布结果消息
            result_msg = GraspObjectArray()
            result_msg.header.stamp = self.get_clock().now().to_msg()
            result_msg.frame_id = self._frame_id
            result_msg.prompt = prompt
            result_msg.objects = grasp_objects
            result_msg.chosen_index = chosen_idx
            result_msg.detection_time_ms = detection_time_ms
            self._result_pub.publish(result_msg)

            # 发布优化深度图
            try:
                depth_msg = self._bridge.cv2_to_imgmsg(depth_optimized_mm, '16UC1')
                depth_msg.header.stamp = result_msg.header.stamp
                depth_msg.header.frame_id = self._frame_id
                self._depth_pub.publish(depth_msg)
            except Exception as e:
                self.get_logger().warn(f'发布深度失败: {e}')

            # 构建返回结果
            if 0 <= chosen_idx < len(grasp_objects):
                chosen = grasp_objects[chosen_idx]
                result = {
                    'success': True,
                    'point3d': [chosen.position.x, chosen.position.y, chosen.position.z],
                    'width3d': chosen.grasp_width3d,
                    'angle': math.degrees(chosen.grasp_angle),
                    'category': chosen.category,
                    'score': chosen.grasp_score,
                    'center_uv': list(chosen.grasp_center),
                    'depth_value': chosen.depth,
                    'detection_time_ms': detection_time_ms
                }
                self.get_logger().info(
                    f'DETECT: 成功 category={chosen.category}, '
                    f'point3d={[round(x*1000) for x in result["point3d"]]}mm, '
                    f'time={detection_time_ms:.0f}ms'
                )
            else:
                result = {
                    'success': False,
                    'error_message': 'No valid grasp found',
                    'detection_time_ms': detection_time_ms
                }
                self.get_logger().warn(f'DETECT: 失败 - 无有效抓取点')

            return result

        finally:
            self._detecting = False
            self._detect_lock.release()

    def _build_grasp_objects(self, rgb, depth, affs, targets, img_w, img_h):
        """构建 GraspObject 列表"""
        grasp_objects = []
        chosen_idx = -1
        best_score = -1

        kernel_erode = np.ones((5, 5), np.uint8)
        mask_cache = {}

        # 预解码所有 mask
        for target_idx, target in enumerate(targets):
            target_mask = target.get('mask', None)
            if target_mask is not None and HAS_COCO:
                try:
                    if isinstance(target_mask, dict):
                        mask_arr = coco_mask.decode(target_mask).astype(np.uint8)
                    elif isinstance(target_mask, np.ndarray):
                        mask_arr = target_mask.astype(np.uint8)
                    else:
                        mask_arr = None

                    if mask_arr is not None:
                        if mask_arr.shape[:2] != (img_h, img_w):
                            mask_arr = cv2.resize(mask_arr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        mask_eroded = cv2.erode(mask_arr, kernel_erode, iterations=1)
                        if np.count_nonzero(mask_eroded) < 50:
                            mask_eroded = mask_arr
                        mask_cache[target_idx] = {'original': mask_arr, 'eroded': mask_eroded}
                except Exception as e:
                    self.get_logger().warn(f'Mask 解码失败: {e}')

        # 为每个 target 找最佳抓取点
        for target_idx, target in enumerate(targets):
            target_bbox = target.get('bbox', [])
            target_category = target.get('category', 'unknown')
            target_score = target.get('score', 0.0)

            if len(target_bbox) < 4:
                continue

            x1, y1, x2, y2 = target_bbox[:4]

            # bbox 尺寸过滤
            cached_mask = mask_cache.get(target_idx)
            if self._bbox_filter_enabled:
                if self._should_filter_by_bbox_size(cached_mask, target_bbox, depth, img_w, img_h):
                    continue

            # 在 bbox 内找最佳抓取点
            best_grasp = None
            best_grasp_score = -1

            if affs and len(affs) > 0 and affs[0]:
                for obj in affs[0]:
                    if not isinstance(obj, dict):
                        continue

                    scores_obj = obj.get('scores', obj.get('confidence', []))
                    affs_obj = obj.get('affs', [])
                    tps = obj.get('touching_points', [])

                    for i, aff in enumerate(affs_obj):
                        cx, cy = aff[0], aff[1]

                        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                            continue

                        score = scores_obj[i] if i < len(scores_obj) else 0.5
                        if score < self._min_grasp_score:
                            continue

                        if cx >= img_w or cy >= img_h:
                            continue

                        # 检查深度
                        cx_int, cy_int = int(cx), int(cy)
                        if 0 <= cy_int < depth.shape[0] and 0 <= cx_int < depth.shape[1]:
                            depth_at_point = depth[cy_int, cx_int]
                            if not (self._min_depth < depth_at_point < self._max_depth):
                                continue

                        if score > best_grasp_score:
                            best_grasp_score = score
                            best_grasp = {
                                'aff': aff,
                                'score': score,
                                'touching_points': tps[i] if i < len(tps) else []
                            }

            # 构建 GraspObject
            obj_msg = GraspObject()
            obj_msg.object_id = f'{target_category}_{target_idx}'
            obj_msg.category = target_category
            obj_msg.detection_score = float(target_score)
            obj_msg.bbox = [float(x) for x in target_bbox[:4]]

            # Mask RLE
            target_mask = target.get('mask', None)
            if target_mask is not None and HAS_COCO:
                try:
                    if isinstance(target_mask, dict):
                        counts = target_mask.get('counts', '')
                        size = target_mask.get('size', [img_h, img_w])
                        obj_msg.mask_rle = counts.decode('utf-8') if isinstance(counts, bytes) else str(counts)
                        obj_msg.mask_size = [int(s) for s in size]
                except Exception:
                    obj_msg.mask_rle = ''
                    obj_msg.mask_size = [0, 0]
            else:
                obj_msg.mask_rle = ''
                obj_msg.mask_size = [0, 0]

            # 抓取信息
            if best_grasp is not None:
                aff = best_grasp['aff']
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                obj_msg.grasp_score = best_grasp['score']
                obj_msg.grasp_center = [float(cx), float(cy)]
                obj_msg.grasp_angle = float(aff[4])
                obj_msg.grasp_width = float(aff[2])

                # 3D 位置
                depth_val = self._get_robust_depth(depth, cached_mask, target_bbox)
                if self._min_depth < depth_val < self._max_depth:
                    point3d = self._deproject_pixel_to_point([cx, cy], depth_val)
                    obj_msg.position = Point(x=point3d[0], y=point3d[1], z=point3d[2])
                    obj_msg.depth = float(depth_val)

                    # 3D 宽度
                    tps = best_grasp.get('touching_points', [])
                    if tps and len(tps) >= 2:
                        points_3d = self._deproject_pixels_batch(tps[:2], depth_val)
                        obj_msg.grasp_width3d = float(np.linalg.norm(points_3d[1] - points_3d[0]))
                    else:
                        obj_msg.grasp_width3d = 0.0

                    # 检查夹爪宽度
                    width_valid = (obj_msg.grasp_width3d == 0.0 or obj_msg.grasp_width3d <= self._max_gripper_width)
                    if width_valid:
                        combined_score = self._compute_combined_score(best_grasp['score'], obj_msg.depth)
                        if combined_score > best_score:
                            best_score = combined_score
                            chosen_idx = len(grasp_objects)
                else:
                    obj_msg.position = Point(x=0.0, y=0.0, z=0.0)
                    obj_msg.depth = 0.0
                    obj_msg.grasp_width3d = 0.0
            else:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                obj_msg.grasp_score = 0.0
                obj_msg.grasp_center = [float(cx), float(cy)]
                obj_msg.grasp_angle = 0.0
                obj_msg.grasp_width = 0.0
                obj_msg.grasp_width3d = 0.0

                depth_val = self._get_robust_depth(depth, cached_mask, target_bbox)
                if self._min_depth < depth_val < self._max_depth:
                    point3d = self._deproject_pixel_to_point([cx, cy], depth_val)
                    obj_msg.position = Point(x=point3d[0], y=point3d[1], z=point3d[2])
                    obj_msg.depth = float(depth_val)
                else:
                    obj_msg.position = Point(x=0.0, y=0.0, z=0.0)
                    obj_msg.depth = 0.0

            grasp_objects.append(obj_msg)

        return grasp_objects, chosen_idx

    def _compute_combined_score(self, grasp_score: float, depth: float) -> float:
        """计算综合分数"""
        if self._dist_weight == 0.0:
            return grasp_score

        if depth <= 0 or depth < self._min_depth or depth > self._max_depth:
            return grasp_score

        if self._dist_mode == 'reciprocal':
            raw_score = self._min_depth / depth
            if self._normalize_dist:
                min_score = self._min_depth / self._max_depth
                distance_score = (raw_score - min_score) / (1.0 - min_score)
            else:
                distance_score = raw_score
        else:
            distance_score = 1.0 - (depth - self._min_depth) / (self._max_depth - self._min_depth)

        w_sum = self._grasp_weight + self._dist_weight
        if w_sum > 0:
            combined = (self._grasp_weight * grasp_score + self._dist_weight * distance_score) / w_sum
        else:
            combined = grasp_score

        if self._debug_chosen:
            self.get_logger().info(
                f'SCORE: grasp={grasp_score:.3f}, depth={depth:.3f}m, '
                f'dist_score={distance_score:.3f}, combined={combined:.3f}'
            )

        return combined

    def _should_filter_by_bbox_size(self, mask_cache, bbox, depth, img_w, img_h) -> bool:
        """基于 bbox 短边物理尺寸过滤"""
        short_edge_px = None
        cx, cy = None, None

        if mask_cache is not None:
            try:
                mask_arr = mask_cache['original']
                contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:
                        rotated_rect = cv2.minAreaRect(largest_contour)
                        (cx, cy), (w, h), _ = rotated_rect
                        short_edge_px = min(w, h)
            except Exception:
                pass

        if short_edge_px is None:
            if len(bbox) < 4:
                return False
            x1, y1, x2, y2 = bbox[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            short_edge_px = min(x2 - x1, y2 - y1)

        depth_val = self._get_robust_depth(depth, mask_cache, bbox)
        if not (self._min_depth < depth_val < self._max_depth):
            return False

        fx = self.intrinsics['fx']
        short_edge_m = short_edge_px * depth_val / fx
        threshold_m = self._max_gripper_width * self._bbox_size_ratio

        return short_edge_m > threshold_m

    def _get_robust_depth(self, depth, mask_cache, bbox) -> float:
        """获取鲁棒深度值"""
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

        if x1 >= x2 or y1 >= y2:
            return 0.0

        if mask_cache is not None:
            mask_eroded = mask_cache['eroded']
        else:
            mask_eroded = np.zeros((h, w), dtype=np.uint8)
            mask_eroded[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1

        depth_masked = depth[mask_eroded > 0]
        valid_mask = (depth_masked > self._min_depth) & (depth_masked < self._max_depth)
        depth_values = depth_masked[valid_mask]

        if len(depth_values) < 10:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx, cy = max(0, min(w - 1, cx)), max(0, min(h - 1, cy))
            fallback = float(depth[cy, cx])
            return fallback if self._min_depth < fallback < self._max_depth else 0.0

        q1, q3 = np.percentile(depth_values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        depth_filtered = depth_values[(depth_values >= lower) & (depth_values <= upper)]

        if len(depth_filtered) > 0:
            return float(np.median(depth_filtered))
        elif len(depth_values) > 0:
            return float(np.median(depth_values))
        return 0.0

    def _deproject_pixel_to_point(self, pixel, depth_val) -> list:
        """2D -> 3D 反投影"""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        u, v = pixel
        x = (u - cx) * depth_val / fx
        y = (v - cy) * depth_val / fy
        z = depth_val
        return [x, y, z]

    def _deproject_pixels_batch(self, pixels, depth_val) -> np.ndarray:
        """批量反投影"""
        pixels = np.array(pixels)
        u, v = pixels[:, 0], pixels[:, 1]

        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        x = (u - cx) * depth_val / fx
        y = (v - cy) * depth_val / fy
        z = np.full_like(x, depth_val)

        return np.stack([x, y, z], axis=-1)


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionGraspNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
