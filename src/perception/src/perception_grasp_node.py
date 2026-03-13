#!/usr/bin/env python3
"""
Perception Grasp Node - 基于手部相机的抓取感知节点

功能:
- 订阅手部相机 RGB + Depth + IR 双目
- 三服务并行: GraspAnything + DinoX/SAM3 + CDM 深度优化
- 输出相机光学坐标系下的 3D 抓取位姿
- 发布 GraspObjectArray (所有检测物体) 和优化后深度图

参数:
- depth_source: 'cdm'（默认，RealSense + CDM srv）、'raw'（RealSense 原始）或 'foundation_stereo'（IR 双目 FS）

发布 Topics:
- ~result (GraspObjectArray): 所有检测物体 + chosen_index
- ~depth (sensor_msgs/Image): CDM 优化后深度图 (16UC1, mm)

Service:
- ~detect (GraspDetect): 按需检测服务
"""

import os
import sys
import time
import math
import json
import threading
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import message_filters
from ament_index_python.packages import get_package_share_directory
from perception.utils import CvBridgeNumPy2 as CvBridge, ThrottledLogger, LATCHED_QOS, SENSOR_QOS
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Point

# 导入自定义消息和服务
from perception.msg import GraspResult, GraspObject, GraspObjectArray
from perception.srv import GraspDetect

# 导入检测服务类
from perception.percept import GraspAnythingOnline, DinoXDetectorOnline, SAM3Online, DepthOptimizerOnline

# pycocotools for mask RLE encoding
try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[PerceptGrasp] pycocotools not installed, mask RLE encoding unavailable")


class SimpleConfig:
    """简单的配置类，用于初始化检测服务"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


class PerceptionGraspNode(Node):
    """抓取感知 ROS 节点"""

    def __init__(self):
        super().__init__('perception_grasp_node',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.log = ThrottledLogger(self.get_logger())
        self.log.info("Initializing...")

        # 加载配置 (从 ROS2 参数系统重建嵌套 dict)
        self.config = self._build_config_from_params()
        if not self.config:
            self.log.error("Config is empty, check launch file")
            raise RuntimeError("Config empty")

        # 声明动态参数（可能不在 YAML 中）
        if not self.has_parameter('debug_filter'):
            self.declare_parameter('debug_filter', False)
        if not self.has_parameter('log_level'):
            self.declare_parameter('log_level', 'INFO')

        # 深度源: 'cdm'（默认，RealSense + CDM srv）、'raw'（RealSense 原始）或 'foundation_stereo'（IR 双目）
        if not self.has_parameter('depth_source'):
            self.declare_parameter('depth_source', 'cdm')
        self.depth_source = self.get_parameter('depth_source').value

        # === 并发控制 ===
        self._detect_lock = threading.Lock()
        self._detecting = False

        # === 持久化线程池 (避免每次检测创建新线程的开销) ===
        self._executor = ThreadPoolExecutor(max_workers=3)

        # === 图像缓存 ===
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.bridge = CvBridge()
        self._data_lock = threading.Lock()

        # === 订阅 CameraInfo (一次性获取内参) ===
        camera_cfg = self.config.get('camera', {})
        info_topic = camera_cfg.get('camera_info_topic', '/camera/hand/color/camera_info')
        self.create_subscription(CameraInfo, info_topic, self._camera_info_callback, SENSOR_QOS)
        self.log.info(f"Subscribed CameraInfo: {info_topic}")

        # === 使用 message_filters 同步订阅 RGB + Depth ===
        rgb_topic = camera_cfg.get('rgb_topic', '/camera/hand/color/image_raw')
        depth_topic = camera_cfg.get('depth_topic', '/camera/hand/aligned_depth_to_color/image_raw')

        self._rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=SENSOR_QOS)
        self._depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=SENSOR_QOS)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=5,
            slop=0.1  # 时间容差 100ms
        )
        self._sync.registerCallback(self._sync_callback)
        self.log.info(f"Sync subscribe: {rgb_topic}, {depth_topic}")

        # === IR 双目订阅 (仅 FoundationStereo 模式) ===
        if self.depth_source == 'foundation_stereo':
            self.ir_left = None
            self.ir_right = None
            self.ir_intrinsics = None
            self._ir_lock = threading.Lock()
            ir_left_topic  = camera_cfg.get('ir_left_topic',  '/camera/hand/infra1/image_rect_raw')
            ir_right_topic = camera_cfg.get('ir_right_topic', '/camera/hand/infra2/image_rect_raw')
            ir_info_topic  = camera_cfg.get('ir_info_topic',  '/camera/hand/infra2/camera_info')
            self._ir_left_sub  = message_filters.Subscriber(self, Image, ir_left_topic,  qos_profile=SENSOR_QOS)
            self._ir_right_sub = message_filters.Subscriber(self, Image, ir_right_topic, qos_profile=SENSOR_QOS)
            self._ir_sync = message_filters.ApproximateTimeSynchronizer(
                [self._ir_left_sub, self._ir_right_sub], queue_size=5, slop=0.1)
            self._ir_sync.registerCallback(self._ir_sync_callback)
            self.create_subscription(CameraInfo, ir_info_topic, self._ir_info_callback, SENSOR_QOS)
            self.log.info(f"IR subscribe: {ir_left_topic}, {ir_right_topic}")

        # === 初始化检测服务 ===
        self._init_detectors()

        # === ROS Service ===
        service_name = self.config.get('trigger', {}).get('service_name', '~/detect')
        self._detect_srv = self.create_service(GraspDetect, service_name, self._detect_service_callback)
        self.log.info(f"Service: {service_name}")

        # === 实时模式配置 ===
        realtime_cfg = self.config.get('trigger', {}).get('realtime_mode', {})
        self._realtime_enabled = realtime_cfg.get('enabled', False)
        self._realtime_rate = realtime_cfg.get('rate', 1.0)  # 默认1Hz以降低计算负载
        self._realtime_prompt = realtime_cfg.get('default_prompt', 'pen.box.phone.bottle.toy')
        self._realtime_enable_cdm = self.config.get('services', {}).get('cdm', {}).get('enabled', True)

        # === 结果发布器 ===
        result_topic = realtime_cfg.get('result_topic', '~/result')
        self._result_pub = self.create_publisher(GraspObjectArray, result_topic, LATCHED_QOS)
        self.log.info(f"Publish result: {result_topic}")

        # === 深度图发布器 (CDM优化后) ===
        depth_pub_topic = '~/depth'
        self._depth_pub = self.create_publisher(Image, depth_pub_topic, LATCHED_QOS)
        self.log.info(f"Publish depth: {depth_pub_topic}")

        # === 坐标系 ===
        self._frame_id = self.config.get('frame_id', 'hand_camera_color_optical_frame')

        # === Chosen Object 选择策略 ===
        self._chosen_policy = self.config.get('chosen_policy', {
            'grasp_score_weight': 1.0,  # 默认纯分数（向后兼容）
            'distance_weight': 0.0,
            'distance_mode': 'reciprocal',
            'debug': False
        })
        policy_str = (f"grasp_weight={self._chosen_policy.get('grasp_score_weight', 1.0):.2f}, "
                     f"dist_weight={self._chosen_policy.get('distance_weight', 0.0):.2f}, "
                     f"mode={self._chosen_policy.get('distance_mode', 'reciprocal')}")
        self.log.info(f"Chosen policy: {policy_str}")

        # === 发布 default_prompt (single source of truth) ===
        self._default_prompt_pub = self.create_publisher(String, '/piper/default_prompt', LATCHED_QOS)
        prompt_msg = String()
        prompt_msg.data = self._realtime_prompt
        self._default_prompt_pub.publish(prompt_msg)
        self.log.info(f"Published default_prompt: '{self._realtime_prompt}'")

        # === 可选: 订阅外部 prompt 覆盖默认值 ===
        prompt_topic = realtime_cfg.get('prompt_topic', '/usr/prompt/grasp')
        self._prompt_sub = self.create_subscription(String, prompt_topic, self._prompt_update_callback, 1)

        self.log.info(f"Realtime mode: enabled={self._realtime_enabled}, rate={self._realtime_rate}Hz")

        # === 预热 GraspAnything ===
        if self.config.get('services', {}).get('grasp', {}).get('warmup', True):
            self._warmup()

        # === 启动实时检测定时器 ===
        if self._realtime_enabled:
            period = 1.0 / self._realtime_rate
            self._realtime_timer = self.create_timer(period, self._realtime_timer_callback)
            self.log.info(f"Realtime timer started: {self._realtime_rate}Hz")

        self.log.info("Initialization complete")

    def _build_config_from_params(self):
        """将 ROS2 扁平参数 (a.b.c=v) 重建为嵌套 dict {a: {b: {c: v}}}"""
        config = {}
        for param_name in list(self._parameters.keys()):
            if param_name == 'use_sim_time':
                continue
            value = self._parameters[param_name].value
            keys = param_name.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        return config

    def _init_detectors(self):
        """初始化检测服务"""
        services_cfg = self.config.get('services', {})
        pkg_path = get_package_share_directory('perception')

        # 1. GraspAnythingOnline
        grasp_cfg = services_cfg.get('grasp', {})
        server_list_path = os.path.join(pkg_path, grasp_cfg.get('server_list', 'config/server_grasp.json'))
        grasp_config = SimpleConfig(
            server_list=server_list_path,
            model_name=grasp_cfg.get('model_name', 'full'),
            resize=(1280, 720),
            warmup=0  # 手动预热
        )
        self.grasp_detector = GraspAnythingOnline(grasp_config)
        self.log.info(f"GraspAnything initialized: {server_list_path}")

        # 2. Object detector (SAM3 or DinoX)
        detector_type = services_cfg.get('detector_type', 'sam3')
        min_score = self.config.get('filter', {}).get('min_object_score', 0.25)

        if detector_type == 'sam3':
            sam3_cfg = services_cfg.get('sam3', {})
            det_config = SimpleConfig(
                url=sam3_cfg.get('url', 'http://192.168.112.14:8081'),
                min_score=min_score
            )
            self.object_detector = SAM3Online(det_config)
            self.log.info(f"SAM3 initialized: {sam3_cfg.get('url')}")
        else:
            dinox_cfg = services_cfg.get('dinox', {})
            det_config = SimpleConfig(
                url=dinox_cfg.get('url', 'http://192.168.112.14:10086'),
                min_score=min_score
            )
            self.object_detector = DinoXDetectorOnline(det_config)
            self.log.info(f"DinoX initialized: {dinox_cfg.get('url')}")

        # 3. DepthOptimizerOnline (CDM 或 FoundationStereo)
        if self.depth_source == 'foundation_stereo':
            fs_cfg = services_cfg.get('foundation_stereo', {})
            depth_url = fs_cfg.get('url', 'http://192.168.112.14:8084')
            self.log.info(f"FoundationStereo depth optimizer: {depth_url}")
            depth_config = SimpleConfig(url=depth_url, warmup=0)
        else:
            cdm_cfg = services_cfg.get('cdm', {})
            depth_url = cdm_cfg.get('url', 'http://192.168.112.14:8082')
            depth_config = SimpleConfig(
                url=depth_url,
                chosen_policy=cdm_cfg.get('chosen_policy', 'dn'),
                warmup=0
            )
            self.log.info(f"CDM depth optimizer: {depth_url}")
        self.depth_optimizer = DepthOptimizerOnline(depth_config)

    def _warmup(self):
        """预热 GraspAnything 服务"""
        self.log.info("Warming up GraspAnything...")
        try:
            dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            self.grasp_detector.forward(dummy_rgb)
            self.log.info("Warmup complete")
        except Exception as e:
            self.log.warn(f"Warmup failed: {e}")

    def _camera_info_callback(self, msg):
        """从 camera_info 解析内参"""
        if self.intrinsics is None:
            self.intrinsics = {
                'width': msg.width,
                'height': msg.height,
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
            }
            self.log.info(f"Camera intrinsics loaded: {msg.width}x{msg.height}")

    def _ir_info_callback(self, msg):
        """IR infra2 camera_info 回调 — 提取双目内参（一次性）"""
        if self.ir_intrinsics is not None:
            return
        fx = msg.p[0]
        tx = msg.p[3]  # -fx * baseline for right camera
        baseline = -tx / fx if fx > 0 and tx != 0.0 else 0.054  # D435 default ~54mm
        self.ir_intrinsics = {
            'fx': fx, 'fy': msg.p[5], 'cx': msg.p[2], 'cy': msg.p[6],
            'width': msg.width, 'height': msg.height,
            'baseline': baseline,
        }
        self.log.info(f"IR intrinsics loaded: {msg.width}x{msg.height}, baseline={baseline:.4f}m")

    def _ir_sync_callback(self, ir1_msg, ir2_msg):
        """IR 双目同步回调 — 缓存最新帧"""
        ir_left  = self.bridge.imgmsg_to_cv2(ir1_msg, 'passthrough')
        ir_right = self.bridge.imgmsg_to_cv2(ir2_msg, 'passthrough')
        if ir_left.dtype != np.uint8:
            ir_left  = (ir_left  / ir_left.max()  * 255).astype(np.uint8) if ir_left.max()  > 0 else ir_left.astype(np.uint8)
        if ir_right.dtype != np.uint8:
            ir_right = (ir_right / ir_right.max() * 255).astype(np.uint8) if ir_right.max() > 0 else ir_right.astype(np.uint8)
        with self._ir_lock:
            self.ir_left  = ir_left
            self.ir_right = ir_right

    def _sync_callback(self, rgb_msg, depth_msg):
        """同步回调 - RGB 和 Depth 时间戳对齐"""
        try:
            # 使用 passthrough 避免 cv_bridge 内部转换问题
            rgb_raw = self.bridge.imgmsg_to_cv2(rgb_msg, 'passthrough')
            if rgb_msg.encoding == 'rgb8':
                rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)  # RGB→BGR
            elif rgb_msg.encoding == 'bgr8':
                rgb = rgb_raw
            else:
                rgb = rgb_raw  # 其他编码原样返回

            depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth = depth_mm.astype(np.float32) / 1000.0  # mm -> m

            with self._data_lock:
                self.rgb = rgb  # BGR 格式
                self.depth = depth
        except Exception as e:
            self.log.error(f"Image decode failed: {e}")

    def _prompt_update_callback(self, msg):
        """实时模式: 更新检测 prompt (不立即触发检测)"""
        new_prompt = msg.data.strip()
        if new_prompt and new_prompt != self._realtime_prompt:
            self.log.info(f"Prompt updated: '{self._realtime_prompt}' -> '{new_prompt}'")
            self._realtime_prompt = new_prompt
            # Re-publish so rviz panel and piper_grasp_node pick up the change
            pub_msg = String()
            pub_msg.data = new_prompt
            self._default_prompt_pub.publish(pub_msg)

    def _realtime_timer_callback(self):
        """实时模式: 定时检测回调"""
        # 检查是否有图像数据
        if self.rgb is None or self.depth is None:
            return

        # 调用检测 (不等待锁，如果正在检测则跳过; verbose=False 抑制例行日志)
        self.detect(self._realtime_prompt, enable_cdm=self._realtime_enable_cdm,
                    wait_for_lock=False, verbose=False)

    def _detect_service_callback(self, request, response):
        """Service 回调 (等待锁，确保服务调用能完成)"""
        result = self.detect(request.prompt, request.enable_cdm, wait_for_lock=True)

        response.success = result['success']
        if result['success']:
            response.point3d = result['point3d']
            response.width3d = result['width3d']
            response.angle = result['angle']
            response.category = result['category']
            response.score = result.get('score', 0.0)
            response.center_uv = result.get('center_uv', [0.0, 0.0])
            response.depth = result.get('depth_value', 0.0)
        else:
            response.error_message = result.get('error_message', 'Detection failed')
        response.detection_time_ms = result.get('detection_time_ms', 0.0)
        return response

    def detect(self, prompt, enable_cdm=True, wait_for_lock=True, verbose=True):
        """
        三服务完全并行调用:
        - GraspAnything: forward(rgb) 内部已包含 post_process
        - DinoX: forward(prompt, rgb) 检测目标物体
        - CDM: forward(rgb, depth_mm) 深度优化

        Args:
            prompt: 检测提示词 (如 "bottle.cup" 或 "box")
            enable_cdm: 是否启用 CDM 深度优化
            wait_for_lock: True=等待锁(Service调用), False=跳过(Timer调用)
            verbose: True=输出详细日志(Service调用), False=仅输出关键日志(实时模式)

        Returns:
            dict: 检测结果
        """
        # === 并发控制 ===
        if wait_for_lock:
            # Service 调用: 等待锁 (最多5秒)
            acquired = self._detect_lock.acquire(blocking=True, timeout=5.0)
            if not acquired:
                self.log.warn("DETECT: Wait timeout, detection in progress")
                return {'success': False, 'error_message': 'Detection timeout'}
        else:
            # Timer 调用: 不等待，如果锁被占用则跳过
            if not self._detect_lock.acquire(blocking=False):
                return {'success': False, 'error_message': 'Detection in progress'}

        try:
            self._detecting = True
            t0 = time.time()
            timing = {}  # 时间分布统计

            if verbose:
                self.log.info(f"DETECT: Start detection: prompt='{prompt}', enable_cdm={enable_cdm}")

            # === 获取当前帧 (加锁保护) ===
            with self._data_lock:
                rgb = self.rgb.copy() if self.rgb is not None else None
                depth = self.depth.copy() if self.depth is not None else None

            t1 = time.time()
            timing['get_frame'] = (t1 - t0) * 1000

            if rgb is None or depth is None:
                self.log.error("DETECT: No image data")
                return {'success': False, 'error_message': 'No image available'}
            if self.intrinsics is None:
                self.log.error("DETECT: Camera intrinsics not ready")
                return {'success': False, 'error_message': 'Camera intrinsics not ready'}

            img_h, img_w = rgb.shape[:2]

            t1_5 = time.time()
            depth_mm = (depth * 1000).astype(np.uint16)  # CDM 需要 uint16 mm
            timing['depth_convert'] = (time.time() - t1_5) * 1000

            # ============================================================
            # 服务执行 (并行)
            # ============================================================
            service_times = {}
            detail_timing = {}

            def timed_call(name, func, *args, **kwargs):
                """带计时的服务调用"""
                ts = time.time()
                try:
                    result = func(*args, **kwargs)
                    service_times[name] = (time.time() - ts) * 1000
                    return result
                except Exception as e:
                    service_times[name] = (time.time() - ts) * 1000
                    raise e

            # 确保数组是连续存储的
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            if not depth_mm.flags['C_CONTIGUOUS']:
                depth_mm = np.ascontiguousarray(depth_mm)

            results = {}

            # === 并行执行 ===
            t_submit = time.time()
            futures = {}
            futures['grasp'] = self._executor.submit(
                timed_call, 'grasp', self.grasp_detector.forward, rgb, None, _timing=detail_timing)
            futures['dinox'] = self._executor.submit(
                timed_call, 'dinox', self.object_detector.forward, prompt, rgb, _timing=detail_timing)
            if enable_cdm:
                if self.depth_source == 'foundation_stereo':
                    with self._ir_lock:
                        ir_left  = self.ir_left.copy()  if self.ir_left  is not None else None
                        ir_right = self.ir_right.copy() if self.ir_right is not None else None
                    ir_intr = self.ir_intrinsics
                    if ir_left is not None and ir_right is not None and ir_intr is not None:
                        futures['cdm'] = self._executor.submit(
                            timed_call, 'cdm', self.depth_optimizer.forward_stereo,
                            ir_left, ir_right, ir_intr, _timing=detail_timing)
                    else:
                        self.log.warn("DETECT: IR frames not ready, skip stereo depth")
                else:
                    # CDM 模式: RealSense 原始深度 + CDM srv
                    futures['cdm'] = self._executor.submit(
                        timed_call, 'cdm', self.depth_optimizer.forward,
                        rgb, depth_mm, _timing=detail_timing)
            timing['submit'] = (time.time() - t_submit) * 1000

            t_wait = time.time()
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=10)
                except Exception as e:
                    self.log.error(f"{name}: Call failed: {e}")
                    results[name] = None
            timing['wait'] = (time.time() - t_wait) * 1000

            timing.update(detail_timing)

            t2 = time.time()
            timing['services_parallel'] = (t2 - t1) * 1000
            timing.update({f'svc_{k}': v for k, v in service_times.items()})

            # === 处理 CDM 结果 ===
            if enable_cdm and results.get('cdm') and results['cdm'].get('success'):
                depth_optimized = results['cdm']['depth'].astype(np.float32) / 1000.0
                depth_optimized_mm = results['cdm']['depth']  # 保留 mm 格式用于发布
                if verbose:
                    self.log.info("CDM: Using optimized depth")

            else:
                depth_optimized = depth
                depth_optimized_mm = depth_mm
                if enable_cdm and verbose:
                    self.log.warn("CDM: Depth optimization failed, using raw depth")

            # === 处理 GraspAnything 结果 ===
            grasp_result = results.get('grasp')
            if grasp_result is not None:
                affs, _ = grasp_result  # 解包元组
            else:
                affs = [[]]

            # === 解析 DinoX 结果 ===
            dinox_result = results.get('dinox')
            if dinox_result and hasattr(dinox_result, 'result'):
                targets = dinox_result.result.get('objects', [])
            else:
                targets = []

            if verbose:
                self.log.info(f"DETECT: GraspAnything: {len(affs[0]) if affs and affs[0] else 0} objects, "
                              f"DinoX: {len(targets)} targets")

            t3 = time.time()
            timing['parse_results'] = (t3 - t2) * 1000

            # === 构建 GraspObjectArray ===
            grasp_objects, chosen_idx = self._build_grasp_objects(
                rgb, depth_optimized, affs, targets, img_w, img_h, verbose=verbose)

            t4 = time.time()
            timing['build_objects'] = (t4 - t3) * 1000

            # === 构建返回结果 ===
            detection_time_ms = (time.time() - t0) * 1000

            # 构建 GraspObjectArray 消息
            result_msg = GraspObjectArray()
            result_msg.header.stamp = self.get_clock().now().to_msg()
            result_msg.frame_id = self._frame_id
            result_msg.prompt = prompt
            result_msg.objects = grasp_objects
            result_msg.chosen_index = chosen_idx
            result_msg.detection_time_ms = detection_time_ms

            # 发布结果
            self._result_pub.publish(result_msg)

            # 发布优化后的深度图 (16UC1, mm)
            try:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_optimized_mm, '16UC1')
                depth_msg.header.stamp = result_msg.header.stamp
                depth_msg.header.frame_id = self._frame_id
                self._depth_pub.publish(depth_msg)
            except Exception as e:
                self.log.warn(f"DETECT: Publish depth failed: {e}")

            t5 = time.time()
            timing['publish'] = (t5 - t4) * 1000
            timing['total'] = (t5 - t0) * 1000

            # === 输出时间分布 ===
            if verbose:
                timing_str = ' | '.join([f"{k}:{v:.0f}" for k, v in timing.items()])
                self.log.info(f"TIMING: {timing_str}")

            # 构建返回值 (兼容 Service 响应)
            if chosen_idx >= 0 and chosen_idx < len(grasp_objects):
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
                if verbose:
                    self.log.info(f"DETECT: Success: category={chosen.category}, "
                                  f"point3d={[round(x*1000) for x in result['point3d']]}mm, "
                                  f"score={chosen.grasp_score:.2f}, time={detection_time_ms:.0f}ms")
            else:
                result = {
                    'success': False,
                    'error_message': 'No valid grasp found',
                    'detection_time_ms': detection_time_ms
                }
                if verbose:
                    self.log.warn(f"DETECT: Failed: No valid grasp, time={detection_time_ms:.0f}ms")

            return result

        finally:
            self._detecting = False
            self._detect_lock.release()

    def _compute_combined_score(self, grasp_score, depth):
        """
        计算综合分数 = 抓取质量 + 距离偏好

        Args:
            grasp_score: GraspAnything 分数 [0, 1]
            depth: 深度值（米）

        Returns:
            float: 综合分数（越高越好）
        """
        policy = self._chosen_policy

        # 权重
        w_grasp = policy.get('grasp_score_weight', 1.0)
        w_dist = policy.get('distance_weight', 0.0)

        # 快速路径: 纯分数模式（向后兼容）
        if w_dist == 0.0:
            return grasp_score

        # 边界检查：深度无效时回退到纯分数
        min_depth = self.config.get('depth', {}).get('min_depth', 0.001)
        max_depth = self.config.get('depth', {}).get('max_depth', 2.0)
        if depth <= 0 or depth < min_depth or depth > max_depth:
            return grasp_score

        # === 计算距离分数 ===
        distance_mode = policy.get('distance_mode', 'reciprocal')

        if distance_mode == 'reciprocal':
            # 简单倒数（越近越好）
            raw_score = min_depth / depth

            # 归一化到 [0, 1] 使其与 grasp_score 对齐
            # 当 depth=min_depth 时 → 1.0 (最近)
            # 当 depth=max_depth 时 → 0.0 (最远)
            normalize = policy.get('normalize_distance', True)
            if normalize:
                min_score = min_depth / max_depth
                max_score = 1.0
                distance_score = (raw_score - min_score) / (max_score - min_score)
            else:
                distance_score = raw_score  # 原始倒数 [0.1, 1.0]

        elif distance_mode == 'gaussian':
            # 高斯衰减（以最佳距离为中心）
            optimal_depth = policy.get('optimal_depth', 0.5)
            sigma = policy.get('gaussian_sigma', 0.3)
            distance_score = np.exp(-((depth - optimal_depth) ** 2) / (2 * sigma ** 2))

        else:  # 'linear' (默认)
            # 线性距离（以最佳距离为中心）
            optimal_depth = policy.get('optimal_depth', 0.5)
            distance_range = policy.get('distance_range', max_depth - min_depth)

            deviation = abs(depth - optimal_depth)
            distance_score = max(0.0, 1.0 - deviation / distance_range)

        # === 加权组合 ===
        # 归一化权重（让结果在 [0, 1]）
        w_sum = w_grasp + w_dist
        if w_sum > 0:
            w_grasp_norm = w_grasp / w_sum
            w_dist_norm = w_dist / w_sum
        else:
            # 权重配置错误，回退到纯分数
            return grasp_score

        combined = w_grasp_norm * grasp_score + w_dist_norm * distance_score

        # 调试输出
        if policy.get('debug', False):
            self.log.info(f"SCORE_DEBUG: grasp={grasp_score:.3f}, depth={depth:.3f}m, "
                         f"dist_score={distance_score:.3f}, combined={combined:.3f}")

        return combined

    def _build_grasp_objects(self, rgb, depth, affs, targets, img_w, img_h, verbose=True):
        """
        构建 GraspObject 列表

        Returns:
            tuple: (grasp_objects, chosen_index)
        """
        grasp_objects = []
        chosen_idx = -1
        best_score = -1

        min_grasp_score = self.config.get('filter', {}).get('min_grasp_score', 0.3)
        depth_cfg = self.config.get('depth', {})
        min_depth = depth_cfg.get('min_depth', 0.001)
        max_depth = depth_cfg.get('max_depth', 2.0)

        # 统计所有 GraspAnything 抓取点
        total_grasp_points = 0
        if affs and len(affs) > 0 and affs[0]:
            for obj in affs[0]:
                if isinstance(obj, dict):
                    total_grasp_points += len(obj.get('affs', []))

        if verbose:
            self.log.info(f"BUILD: GraspAnything total grasp points: {total_grasp_points}, DinoX targets: {len(targets)}")

        # ===== 优化：预解码所有 mask（避免重复解码）=====
        mask_cache = {}  # 缓存已解码的 mask
        kernel_erode = np.ones((5, 5), np.uint8)  # 提前创建 kernel，避免重复创建

        # 批量日志收集（减少日志开销）
        build_logs = []
        mask_decode_success = 0  # 统计成功解码的 mask 数量
        bbox_filtered_count = 0  # 统计被 bbox 尺寸过滤的 target 数

        for target_idx, target in enumerate(targets):
            target_mask = target.get('mask', None)
            if target_mask is not None and HAS_COCO:
                try:
                    if isinstance(target_mask, dict):
                        # 解码 RLE mask
                        mask_arr = coco_mask.decode(target_mask).astype(np.uint8)
                    elif isinstance(target_mask, np.ndarray):
                        mask_arr = target_mask.astype(np.uint8)
                    else:
                        mask_arr = None

                    if mask_arr is not None:
                        # 确保尺寸匹配
                        if mask_arr.shape[:2] != (img_h, img_w):
                            mask_arr = cv2.resize(mask_arr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                        # 预腐蚀（用于鲁棒深度计算）
                        mask_eroded = cv2.erode(mask_arr, kernel_erode, iterations=1)
                        if np.count_nonzero(mask_eroded) < 50:
                            mask_eroded = mask_arr

                        # 缓存：原始 + 腐蚀后
                        mask_cache[target_idx] = {
                            'original': mask_arr,
                            'eroded': mask_eroded
                        }
                        mask_decode_success += 1
                except Exception as e:
                    if verbose:
                        self.log.warn(f"BUILD: Mask decode failed for target {target_idx} ({targets[target_idx].get('category', 'unknown')}): {e}")

        # 输出 mask 解码统计
        if verbose and len(targets) > 0:
            self.log.info(f"BUILD: Masks decoded: {mask_decode_success}/{len(targets)}")

        # 为每个 target 找到最佳抓取点
        for target_idx, target in enumerate(targets):
            target_bbox = target.get('bbox', [])
            target_category = target.get('category', 'unknown')
            target_score = target.get('score', 0.0)
            target_mask = target.get('mask', None)

            if len(target_bbox) < 4:
                continue

            x1, y1, x2, y2 = target_bbox[:4]

            # ============================================================
            # 规则2：bbox 短边尺寸过滤（提前过滤整个 target）
            # ============================================================
            cached_mask = mask_cache.get(target_idx)
            if self._should_filter_by_bbox_size(cached_mask, target_bbox, depth, img_w, img_h):
                bbox_filtered_count += 1
                if self.get_parameter('debug_filter').value:
                    self.log.debug(f"BUILD: Target {target_idx} ({target_category}): "
                                  f"Filtered by bbox size (too large for gripper)")
                continue  # 跳过此 target，不构建 GraspObject

            # 在此 target 的 bbox 内找最佳抓取点
            best_grasp = None
            best_grasp_score = -1
            matched_count = 0  # 统计匹配到的抓取点数
            score_filtered_count = 0  # 统计因分数过滤的抓取点数
            depth_filtered_count = 0  # 统计因深度被过滤的抓取点数
            mask_overlap_filtered_count = 0  # 统计因 mask 重叠被过滤的抓取点数
            boundary_filtered_count = 0  # 统计因边界过滤的抓取点数
            total_affs_in_bbox = 0  # bbox内的总抓取点数

            if affs and len(affs) > 0 and affs[0]:
                for obj_idx, obj in enumerate(affs[0]):
                    if not isinstance(obj, dict):
                        continue

                    # 尝试多种可能的 score 字段名
                    scores_obj = []
                    if 'scores' in obj:
                        scores_obj = obj['scores']
                    elif 'confidence' in obj:
                        scores_obj = obj['confidence']
                    elif 'conf' in obj:
                        scores_obj = obj['conf']
                    elif 'score' in obj:
                        # 如果是单个 score 值，转换为列表
                        score_val = obj['score']
                        if isinstance(score_val, list):
                            scores_obj = score_val
                        else:
                            scores_obj = [score_val]

                    affs_obj = obj.get('affs', [])
                    tps = obj.get('touching_points', [])

                    for i, aff in enumerate(affs_obj):
                        cx, cy = aff[0], aff[1]

                        # 检查是否在 bbox 内
                        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                            continue

                        total_affs_in_bbox += 1

                        # 检查分数
                        # 如果 scores_obj 为空（服务器不返回 scores），使用默认分数 0.5
                        if len(scores_obj) == 0:
                            score = 0.5  # 默认中等置信度，允许通过过滤
                        else:
                            score = scores_obj[i] if i < len(scores_obj) else 0

                        if score < min_grasp_score:
                            score_filtered_count += 1
                            continue

                        # 检查是否超出图像范围
                        if cx >= img_w or cy >= img_h:
                            boundary_filtered_count += 1
                            continue

                        matched_count += 1

                        # 提前检查深度（过滤夹爪等近距离物体）
                        cx_int, cy_int = int(cx), int(cy)
                        if 0 <= cy_int < depth.shape[0] and 0 <= cx_int < depth.shape[1]:
                            depth_at_point = depth[cy_int, cx_int]
                            if not (min_depth < depth_at_point < max_depth):
                                if depth_filtered_count == 0:
                                    self.log.debug(f"DEPTH_REJECT T{target_idx}({target_category}): "
                                                   f"pixel=({cx_int},{cy_int}), depth={depth_at_point:.3f}m, "
                                                   f"range=[{min_depth},{max_depth}]")
                                depth_filtered_count += 1
                                continue  # 深度不在有效范围，跳过此 affordance

                        # 检查 affordance 旋转矩形与 mask 的重叠
                        overlap_config = self.config.get('filter', {}).get('affordance_mask_overlap', {})
                        current_tps = tps[i] if i < len(tps) else []
                        cached_mask = mask_cache.get(target_idx)
                        overlap_result = self._check_affordance_mask_overlap(aff, current_tps, cached_mask, img_h, img_w, overlap_config)
                        if overlap_result:
                            mask_overlap_filtered_count += 1
                            # 可选：详细日志（调试用）
                            if self.get_parameter('debug_filter').value:
                                self.log.debug(f"BUILD: Target {target_idx} ({target_category}): Affordance filtered - "
                                              f"center=({cx:.0f},{cy:.0f}), width={aff[2]:.0f}px, score={score:.2f}")
                            continue  # mask 重叠过高，跳过此 affordance

                        if score > best_grasp_score:
                            best_grasp_score = score
                            best_grasp = {
                                'aff': aff,
                                'score': score,
                                'touching_points': tps[i] if i < len(tps) else []
                            }

            # 收集过滤统计
            filter_details = []
            if total_affs_in_bbox > 0:
                filter_details.append(f"total={total_affs_in_bbox}")
            if score_filtered_count > 0:
                filter_details.append(f"score<{min_grasp_score}={score_filtered_count}")
            if boundary_filtered_count > 0:
                filter_details.append(f"boundary={boundary_filtered_count}")
            if depth_filtered_count > 0:
                filter_details.append(f"depth={depth_filtered_count}")
            if mask_overlap_filtered_count > 0:
                filter_details.append(f"mask={mask_overlap_filtered_count}")

            # 如果没有找到有效抓取点，打印详细信息
            if total_affs_in_bbox > 0 and best_grasp is None and verbose:
                self.log.warn(f"BUILD: T{target_idx}({target_category}): All {total_affs_in_bbox} affs filtered! "
                             f"{', '.join(filter_details)}")
            elif filter_details:
                build_logs.append(f"T{target_idx}({target_category}: {', '.join(filter_details)} → found={matched_count})")

            # 构建 GraspObject
            obj_msg = GraspObject()
            obj_msg.object_id = f"{target_category}_{target_idx}"
            obj_msg.category = target_category
            obj_msg.detection_score = target_score
            obj_msg.bbox = target_bbox[:4]

            # Mask RLE 编码
            if target_mask is not None and HAS_COCO:
                try:
                    if isinstance(target_mask, dict):
                        # DinoX 返回的 mask 已经是 RLE dict 格式
                        rle = target_mask
                        counts = rle.get('counts', '')
                        size = rle.get('size', [img_h, img_w])
                        obj_msg.mask_rle = counts.decode('utf-8') if isinstance(counts, bytes) else str(counts)
                        obj_msg.mask_size = list(size)
                    elif isinstance(target_mask, np.ndarray):
                        # numpy 数组格式，需要编码为 RLE（优化：避免不必要的类型转换）
                        if target_mask.dtype != np.uint8:
                            mask_uint8 = target_mask.astype(np.uint8)
                        else:
                            mask_uint8 = target_mask
                        if mask_uint8.shape[:2] != (img_h, img_w):
                            mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        # 确保 C-contiguous 后再转 Fortran（pycocotools 要求）
                        if not mask_uint8.flags['C_CONTIGUOUS']:
                            mask_uint8 = np.ascontiguousarray(mask_uint8)
                        mask_fortran = np.asfortranarray(mask_uint8)
                        rle = coco_mask.encode(mask_fortran)
                        obj_msg.mask_rle = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                        obj_msg.mask_size = [img_h, img_w]
                    else:
                        obj_msg.mask_rle = ""
                        obj_msg.mask_size = [0, 0]
                except Exception as e:
                    if verbose:
                        self.log.warn(f"DETECT: Mask RLE encode failed: {e}")
                    obj_msg.mask_rle = ""
                    obj_msg.mask_size = [0, 0]
            else:
                obj_msg.mask_rle = ""
                obj_msg.mask_size = [0, 0]

            # 抓取信息
            if best_grasp is not None:
                aff = best_grasp['aff']
                # 原: 使用 aff 中心
                # cx, cy = int(aff[0]), int(aff[1])
                # obj_msg.grasp_center = [float(aff[0]), float(aff[1])]
                # 改: 使用 bbox 中心，angle 保持 aff 的
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                obj_msg.grasp_score = best_grasp['score']
                obj_msg.grasp_center = [float(cx), float(cy)]
                obj_msg.grasp_angle = float(aff[4])  # 弧度 (保持 aff 的 angle)
                obj_msg.grasp_width = float(aff[2])  # 像素宽度

                # 3D 位置 - 使用 mask 区域的鲁棒深度
                # 原: depth_val = depth[cy, cx] if 0 <= cy < img_h and 0 <= cx < img_w else 0
                cached_mask = mask_cache.get(target_idx)
                depth_val = self._get_robust_depth(depth, cached_mask, target_bbox, min_depth, max_depth)
                if not (min_depth < depth_val < max_depth):
                    self.log.debug(f"ROBUST_DEPTH_REJECT T{target_idx}({target_category}): "
                                  f"depth={depth_val:.3f}m, range=[{min_depth},{max_depth}]")
                if min_depth < depth_val < max_depth:
                    point3d = self._deproject_pixel_to_point([cx, cy], depth_val)
                    obj_msg.position = Point(x=point3d[0], y=point3d[1], z=point3d[2])
                    obj_msg.depth = float(depth_val)

                    # 计算 3D 夹爪宽度（优化：批量反投影）
                    tps = best_grasp.get('touching_points', [])
                    if tps and len(tps) >= 2:
                        points_3d = self._deproject_pixels_batch(tps[:2], depth_val)
                        obj_msg.grasp_width3d = float(np.linalg.norm(points_3d[1] - points_3d[0]))
                    else:
                        obj_msg.grasp_width3d = 0.0

                    # 检查夹爪宽度限制
                    max_gripper_width = self.config.get('filter', {}).get('max_gripper_width', 0.15)
                    width_valid = (obj_msg.grasp_width3d == 0.0 or obj_msg.grasp_width3d <= max_gripper_width)

                    if not width_valid:
                        tp_px_dist = float(np.linalg.norm(
                            np.array(tps[1]) - np.array(tps[0]))) if tps and len(tps) >= 2 else 0.0
                        log_fn = self.log.warn if verbose else self.log.debug
                        log_fn(f"WIDTH_REJECT T{target_idx}({target_category}): "
                               f"width={obj_msg.grasp_width3d*1000:.1f}mm > max={max_gripper_width*1000:.1f}mm "
                               f"[tp_px={tp_px_dist:.1f}px depth={depth_val:.3f}m "
                               f"tp=({tps[0][0]:.0f},{tps[0][1]:.0f})->({tps[1][0]:.0f},{tps[1][1]:.0f})]")

                    # 更新最佳选择（只有宽度有效时才更新）
                    if width_valid:
                        # 计算综合分数（考虑抓取质量 + 距离）
                        combined_score = self._compute_combined_score(
                            best_grasp['score'],
                            obj_msg.depth
                        )

                        if combined_score > best_score:
                            best_score = combined_score
                            chosen_idx = len(grasp_objects)

                            # 调试: 记录选择原因
                            if self._chosen_policy.get('debug', False):
                                self.log.info(f"CHOSEN_UPDATE: idx={chosen_idx}, category={target_category}, "
                                             f"grasp_score={best_grasp['score']:.3f}, depth={obj_msg.depth:.3f}m, "
                                             f"combined={combined_score:.3f}")
                else:
                    obj_msg.position = Point(x=0.0, y=0.0, z=0.0)
                    obj_msg.depth = 0.0
                    obj_msg.grasp_width3d = 0.0
            else:
                # 无有效抓取点，使用 bbox 中心
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                obj_msg.grasp_score = 0.0
                obj_msg.grasp_center = [float(cx), float(cy)]
                obj_msg.grasp_angle = 0.0
                obj_msg.grasp_width = 0.0
                obj_msg.grasp_width3d = 0.0

                # 原: depth_val = depth[int(cy), int(cx)] if 0 <= int(cy) < img_h and 0 <= int(cx) < img_w else 0
                cached_mask = mask_cache.get(target_idx)
                depth_val = self._get_robust_depth(depth, cached_mask, target_bbox, min_depth, max_depth)
                if min_depth < depth_val < max_depth:
                    point3d = self._deproject_pixel_to_point([cx, cy], depth_val)
                    obj_msg.position = Point(x=point3d[0], y=point3d[1], z=point3d[2])
                    obj_msg.depth = float(depth_val)
                else:
                    obj_msg.position = Point(x=0.0, y=0.0, z=0.0)
                    obj_msg.depth = 0.0

            grasp_objects.append(obj_msg)

        # 批量输出 bbox 尺寸过滤统计
        if verbose and bbox_filtered_count > 0:
            self.log.info(f"BUILD: Bbox size filter: {bbox_filtered_count} targets filtered (too large for gripper)")

        # 批量输出过滤日志摘要（优化：减少日志调用次数）
        if verbose and build_logs:
            self.log.info(f"BUILD: Filters applied: {' | '.join(build_logs)}")

        return grasp_objects, chosen_idx

    def _get_robust_depth(self, depth, mask_cache, bbox, min_depth=0.001, max_depth=2.0):
        """
        从 mask 区域计算鲁棒深度值（优化版：使用预解码的 mask）

        Args:
            depth: 深度图 (H, W), 单位米
            mask_cache: 预解码的 mask 字典 {'original': mask_arr, 'eroded': mask_eroded} 或 None
            bbox: [x1, y1, x2, y2]
            min_depth, max_depth: 有效深度范围

        Returns:
            float: 鲁棒深度值，失败返回 0
        """
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

        # 检查 bbox 有效性
        if x1 >= x2 or y1 >= y2:
            return 0.0

        # 使用缓存的腐蚀 mask（已经预处理好）
        if mask_cache is not None:
            mask_eroded = mask_cache['eroded']
        else:
            # 回退：使用 bbox 区域
            mask_eroded = np.zeros((h, w), dtype=np.uint8)
            mask_eroded[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 1

        # 3. 向量化提取 mask 内深度值并过滤
        depth_masked = depth[mask_eroded > 0]
        valid_mask = (depth_masked > min_depth) & (depth_masked < max_depth)
        depth_values = depth_masked[valid_mask]

        if len(depth_values) < 10:
            # 像素太少，回退到 bbox 中心
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx, cy = max(0, min(w-1, cx)), max(0, min(h-1, cy))
            fallback = float(depth[cy, cx])
            # 校验回退值有效性
            return fallback if min_depth < fallback < max_depth else 0.0

        # 4. IQR 剔除异常值 - 向量化
        q1, q3 = np.percentile(depth_values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        depth_filtered = depth_values[(depth_values >= lower) & (depth_values <= upper)]

        # 5. 返回中值
        if len(depth_filtered) > 0:
            return float(np.median(depth_filtered))
        elif len(depth_values) > 0:
            return float(np.median(depth_values))
        else:
            return 0.0

    def _check_affordance_mask_overlap(self, aff, touching_points, mask_cache, img_h, img_w, config):
        """
        检查 affordance 旋转矩形与 mask 的重叠率（优化版：ROI-based）

        Args:
            aff: affordance 数组 [cx, cy, width, ?, angle, ...]
            touching_points: 两个接触点 [[x1, y1], [x2, y2]]
            mask_cache: 预解码的 mask 字典 {'original': mask_arr, 'eroded': mask_eroded} 或 None
            img_h, img_w: 图像尺寸
            config: affordance_mask_overlap 配置

        Returns:
            bool: True 表示应该过滤（重叠过高），False 表示保留
        """
        if not config.get('enabled', False):
            return False  # 未启用，保留所有

        if mask_cache is None:
            return False  # 无 mask，保留

        # 使用缓存的原始 mask（已经预解码和尺寸匹配）
        mask_arr = mask_cache['original']

        # 2. 计算旋转矩形参数
        cx, cy = aff[0], aff[1]

        # 从 touching_points 计算宽度和角度
        if touching_points and len(touching_points) >= 2:
            tp1, tp2 = touching_points[0], touching_points[1]
            # 宽度 = 两点距离
            width = np.sqrt((tp2[0] - tp1[0])**2 + (tp2[1] - tp1[1])**2)
            # 角度 = 两点连线角度
            angle = np.arctan2(tp2[1] - tp1[1], tp2[0] - tp1[0])
        else:
            # 回退：使用 aff 中的信息
            width = aff[2] if len(aff) > 2 else 50.0
            angle = aff[4] if len(aff) > 4 else 0.0

        # 高度 = 宽度 * 比例
        height_ratio = config.get('rect_height_ratio', 0.4)
        height = width * height_ratio

        # 3. 扩展旋转矩形
        expand_ratio = config.get('expand_ratio', 1.5)
        width_expanded = width * expand_ratio
        height_expanded = height * expand_ratio

        # 4. 优化：只在 ROI 区域计算（避免全图 mask 创建）
        angle_deg = np.degrees(angle)
        rect = ((cx, cy), (width_expanded, height_expanded), angle_deg)
        box = cv2.boxPoints(rect)  # 获取4个顶点
        box = np.int0(box)

        # 获取旋转矩形的外接矩形（ROI）
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return False

        # 只在 ROI 区域创建小 mask（而不是全图）
        roi_w, roi_h = x_max - x_min, y_max - y_min
        rotated_rect_mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # 调整 box 坐标到 ROI 空间
        box_roi = box - np.array([x_min, y_min])
        cv2.fillPoly(rotated_rect_mask_roi, [box_roi], 1)

        # 提取 mask 的 ROI
        mask_roi = mask_arr[y_min:y_max, x_min:x_max]

        # 5. 计算重叠率（只在小区域）
        overlap_pixels = np.sum((mask_roi > 0) & (rotated_rect_mask_roi > 0))
        rotated_rect_pixels = np.sum(rotated_rect_mask_roi > 0)

        if rotated_rect_pixels == 0:
            return False  # 矩形无效，保留

        overlap_ratio = overlap_pixels / rotated_rect_pixels

        # 6. 判断是否过滤
        threshold = config.get('overlap_threshold', 0.9)
        should_filter = overlap_ratio > threshold

        # 调试输出：显示重叠率详情
        if self.get_parameter('debug_filter').value:
            self.log.info(f"OVERLAP_CHECK: cx={cx:.0f}, cy={cy:.0f}, "
                         f"rect={width_expanded:.0f}x{height_expanded:.0f}, "
                         f"overlap={overlap_pixels}/{rotated_rect_pixels}={overlap_ratio:.1%}, "
                         f"threshold={threshold:.1%}, filter={should_filter}")

        return should_filter

    def _should_filter_by_bbox_size(self, mask_cache, bbox, depth, img_w, img_h):
        """
        规则2: 基于 mask rotated bbox 的短边物理尺寸过滤

        原理：
        1. 从 mask 计算最小外接旋转矩形 (rotated bbox)
        2. 取短边像素长度，转换为物理尺寸（米）
        3. 与阈值比较（夹爪宽度 * size_ratio）

        Args:
            mask_cache: 预解码的 mask 字典 {'original': mask_arr, 'eroded': ...} 或 None
            bbox: [x1, y1, x2, y2] 备用（mask 失效时回退）
            depth: (H, W) 深度图，单位米
            img_w, img_h: 图像尺寸

        Returns:
            bool: True=应该过滤, False=保留
        """
        bbox_filter_cfg = self.config.get('filter', {}).get('bbox_size_filter', {})
        if not bbox_filter_cfg.get('enabled', False):
            return False  # 未启用

        # === 1. 计算短边像素长度 ===
        short_edge_px = None
        cx, cy = None, None
        method = "unknown"

        # 优先：从 mask 计算 rotated bbox
        if mask_cache is not None:
            try:
                mask_arr = mask_cache['original']
                # 找轮廓（外轮廓，简化链）
                contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # 取最大轮廓（处理多个分离区域）
                    largest_contour = max(contours, key=cv2.contourArea)

                    # minAreaRect 需要至少 5 个点
                    if len(largest_contour) >= 5:
                        # 计算最小外接旋转矩形
                        rotated_rect = cv2.minAreaRect(largest_contour)
                        (cx, cy), (w, h), angle = rotated_rect
                        short_edge_px = min(w, h)
                        method = "rotated_bbox"
            except Exception as e:
                self.log.warn(f"BBOX_FILTER: Rotated bbox failed: {e}, fallback to axis-aligned")

        # 回退：使用 axis-aligned bbox
        if short_edge_px is None:
            if len(bbox) < 4:
                return False  # 无效 bbox，保守策略：不过滤

            x1, y1, x2, y2 = bbox[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            short_edge_px = min(x2 - x1, y2 - y1)
            method = "axis_aligned_bbox"

        # === 2. 获取鲁棒深度值（使用 mask 区域的中值深度）===
        min_depth = self.config.get('depth', {}).get('min_depth', 0.001)
        max_depth = self.config.get('depth', {}).get('max_depth', 2.0)

        # 使用鲁棒深度计算（复用现有逻辑）
        depth_val = self._get_robust_depth(depth, mask_cache, bbox, min_depth, max_depth)

        # 深度有效性检查
        if not (min_depth < depth_val < max_depth):
            return False  # 深度无效，保守策略：不过滤

        # === 3. 像素 → 物理尺寸转换 ===
        fx = self.intrinsics['fx']
        short_edge_m = short_edge_px * depth_val / fx

        # === 4. 阈值判断 ===
        max_gripper_width = self.config.get('filter', {}).get('max_gripper_width', 0.090)  # 默认 90mm
        size_ratio = bbox_filter_cfg.get('size_ratio', 2.0)  # 默认 2 倍
        threshold_m = max_gripper_width * size_ratio

        should_filter = short_edge_m > threshold_m

        # === 5. 调试日志 ===
        if self.get_parameter('debug_filter').value:
            self.log.info(f"BBOX_FILTER: method={method}, "
                         f"short_edge_px={short_edge_px:.1f}, "
                         f"depth={depth_val:.3f}m, "
                         f"short_edge_m={short_edge_m*1000:.1f}mm, "
                         f"threshold={threshold_m*1000:.1f}mm, "
                         f"filter={should_filter}")

        return should_filter

    def _deproject_pixel_to_point(self, pixel, depth_val):
        """
        纯 Python 实现 2D→3D 反投影

        Args:
            pixel: [u, v] 像素坐标
            depth_val: 深度值 (米)

        Returns:
            [x, y, z] 相机光学坐标系 3D 点 (Z向前, X右, Y下)
        """
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        u, v = pixel
        x = (u - cx) * depth_val / fx
        y = (v - cy) * depth_val / fy
        z = depth_val
        return [x, y, z]

    def _deproject_pixels_batch(self, pixels, depth_val):
        """
        批量反投影（优化版：避免循环）

        Args:
            pixels: [[u1, v1], [u2, v2], ...] 像素坐标列表
            depth_val: 深度值 (米)

        Returns:
            numpy array shape (N, 3): [[x1, y1, z1], [x2, y2, z2], ...]
        """
        pixels = np.array(pixels)  # shape: (N, 2)
        u, v = pixels[:, 0], pixels[:, 1]

        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        x = (u - cx) * depth_val / fx
        y = (v - cy) * depth_val / fy
        z = np.full_like(x, depth_val)

        return np.stack([x, y, z], axis=-1)  # shape: (N, 3)

def main():
    rclpy.init()
    try:
        node = PerceptionGraspNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
