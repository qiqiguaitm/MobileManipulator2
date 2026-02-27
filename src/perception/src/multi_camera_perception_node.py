#!/usr/bin/env python3
"""
Multi-Camera Perception 3D - ROS2 节点 (事件驱动 Worker 架构)

双相机（Top + Chassis）联合感知节点，支持：
1. 双相机并行检测（后台 Worker 线程，事件驱动）
2. 独立发布各相机结果 + 匈牙利融合去重
3. 统一的服务接口

架构:
    - 每相机一个 Worker 线程，轮询 sensor.get_latest_data()
    - 帧去重（stamp 比较）→ 检测 + 批量 3D 测量 → 发布结果
    - 无 Timer / 无 Queue，消除投递-消费速度不匹配的延迟

发布:
    ~/top/objects_3d (Object3DArray) - Top相机检测结果
    ~/chassis/objects_3d (Object3DArray) - Chassis相机检测结果
    ~/fused/objects_3d (Object3DArray) - 融合结果（匈牙利匹配 + ByteTracker3D 跟踪）

服务:
    ~/detect (DetectObjects) - 支持指定相机的检测服务

Usage:
    ros2 run perception multi_camera_perception_node
    ros2 launch perception multi_camera_perception.launch.py
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import TransformStamped

from perception.utils import CvBridgeNumPy2 as CvBridge
from pycocotools import mask as coco_mask

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from perception.msg import Object3D, Object3DArray, PerceptionConfig, PerceptionStatus
from perception.srv import DetectObjects

from perception.coordinate_transformer import CoordinateTransformer
from perception.scene_perception_core import ScenePerceptionCore, SimpleConfig
from perception.percept import DinoXDetectorOnline, SAM3Online, DepthOptimizerOnline
from perception.intelligent_fusion import fuse_dual_camera_positions
from perception.dual_camera_matcher import DualCameraMatcher, DetectionWithDepth
from perception.byte_tracker_3d import ByteTracker3D, TrackerConfig

from perception.synced_sensor_subscriber import SyncedSensorSubscriber
from perception.utils import SENSOR_QOS, LATCHED_QOS, ThrottledLogger, stamp_to_sec


@dataclass
class CameraConfig:
    """相机配置"""
    name: str
    color_topic: str
    depth_topic: str
    info_topic: str
    enable: bool = True


@dataclass
class CameraData:
    """相机同步数据容器"""
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: Dict
    timestamp: float
    frame_id: str
    valid: bool = True


@dataclass
class DetectionTask:
    """检测任务（用于队列传递）"""
    camera_name: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: Dict
    prompt: str
    # 预计算的变换（在主线程查询，避免后台线程TF竞争）
    optical_to_base_matrix: Optional[np.ndarray] = None
    # 元数据
    timestamp_sec: float = 0.0
    frame_id: str = ''


class MultiCameraPerceptionNode(Node):
    """多相机 3D 感知 ROS2 节点（后台线程架构）"""

    def __init__(self):
        super().__init__('multi_camera_perception')
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
        self._last_results = {'top': None, 'chassis': None, 'fused': None}
        self._results_lock = threading.Lock()  # 保护 _last_results 和 fusion 的线程安全

        # CV Bridge
        self._bridge = CvBridge()

        # 发布器 - 按相机分开发布
        self._init_publishers()

        # 订阅配置
        self._config_sub = self.create_subscription(
            PerceptionConfig, '/perception/config',
            self._config_callback, 1
        )

        # 服务
        self._detect_srv = self.create_service(
            DetectObjects, '~/detect', self.handle_detect
        )

        # 状态发布
        self.pub_status = self.create_publisher(
            PerceptionStatus, '/perception/status', 1
        )

        # ========== 事件驱动 Worker 架构 ==========
        # 关闭标志
        self._shutdown = False
        # 工作线程（每相机一个）
        self._worker_threads: Dict[str, threading.Thread] = {}
        # 共享线程池（检测 + CDM 并行）
        self._detection_executor = ThreadPoolExecutor(max_workers=4)
        # 启动工作线程
        self._init_worker_threads()

        self.get_logger().info('='*60)
        self.get_logger().info('Multi-Camera Perception Node 启动完成')
        self.get_logger().info(f'  检测器: {self.detector_type.upper()}')
        self.get_logger().info(f'  Top相机: {"启用" if self.enable_top else "禁用"}')
        self.get_logger().info(f'  Chassis相机: {"启用" if self.enable_chassis else "禁用"}')
        self.get_logger().info(f'  LiDAR: {"启用" if self.enable_lidar else "禁用"}')
        self.get_logger().info(f'  Target frame: {self.target_frame}')
        self.get_logger().info(f'  自动检测: {self.auto_detect_rate} Hz')
        self.get_logger().info('='*60)

    def _load_config(self):
        """声明和加载 ROS2 参数"""
        # 相机启用开关
        self.declare_parameter('enable_top', True)
        self.declare_parameter('enable_chassis', True)
        self.enable_top = self.get_parameter('enable_top').value
        self.enable_chassis = self.get_parameter('enable_chassis').value

        # 相机话题配置
        self.declare_parameter('top_color_topic', '/camera/top/color/image_raw')
        self.declare_parameter('top_depth_topic', '/camera/top/aligned_depth_to_color/image_raw')
        self.declare_parameter('top_info_topic', '/camera/top/color/camera_info')
        self.declare_parameter('chassis_color_topic', '/camera/chassis/color/image_raw')
        self.declare_parameter('chassis_depth_topic', '/camera/chassis/aligned_depth_to_color/image_raw')
        self.declare_parameter('chassis_info_topic', '/camera/chassis/color/camera_info')

        # LiDAR 配置
        self.declare_parameter('lidar_topic', '/rslidar_points')
        self.declare_parameter('enable_lidar', False)
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.enable_lidar = self.get_parameter('enable_lidar').value

        # Prompt 配置
        self.declare_parameter('prompt_source', 'default')
        self.declare_parameter('default_prompt', 'bottle.cup.box.barrel.toy')
        self.default_prompt = self.get_parameter('default_prompt').value

        # 检测服务
        self.declare_parameter('detector_type', 'dinox')  # dinox | sam3
        self.declare_parameter('detector_url', 'http://192.168.112.14:10086')
        self.declare_parameter('sam3_url', 'http://192.168.112.14:8080')
        self.declare_parameter('detector_timeout', 3.0)
        self.declare_parameter('min_score', 0.25)
        self.declare_parameter('iou_threshold', 0.5)
        self.detector_type = self.get_parameter('detector_type').value
        self.detector_url = self.get_parameter('detector_url').value
        self.sam3_url = self.get_parameter('sam3_url').value
        self.detector_timeout = self.get_parameter('detector_timeout').value
        self.min_score = self.get_parameter('min_score').value
        self.iou_threshold = self.get_parameter('iou_threshold').value

        # 深度优化服务
        self.declare_parameter('depth_optimizer_url', 'http://192.168.112.14:8081')
        self.declare_parameter('enable_depth_optimizer', True)
        self.depth_optimizer_url = self.get_parameter('depth_optimizer_url').value
        self.enable_depth_optimizer = self.get_parameter('enable_depth_optimizer').value

        # 坐标系配置
        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').value

        # 距离补偿
        self.declare_parameter('distance_offset', 0.0)
        self.distance_offset_default = self.get_parameter('distance_offset').value

        # 自动检测配置
        self.declare_parameter('auto_detect_rate', 10.0)  # 默认 10Hz
        self.auto_detect_rate = self.get_parameter('auto_detect_rate').value

        # 时间同步参数
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('data_max_age', 2.0)  # 增加到2秒，避免chassis相机因延迟被丢弃
        self.declare_parameter('strict_data_freshness', False)  # 是否严格检查数据新鲜度
        self.sync_slop = self.get_parameter('sync_slop').value
        self.data_max_age = self.get_parameter('data_max_age').value
        self.strict_data_freshness = self.get_parameter('strict_data_freshness').value

        # 外参配置目录
        self.declare_parameter('extrinsics_dir', '')
        self.extrinsics_dir = self.get_parameter('extrinsics_dir').value
        # 外参文件后缀（用于测试新标定，如 '_new'）
        self.declare_parameter('extrinsics_suffix', '')
        self.extrinsics_suffix = self.get_parameter('extrinsics_suffix').value

        # 融合配置（匈牙利匹配器参数）
        self.declare_parameter('fusion_distance_threshold', 0.2)  # 20cm 距离硬门控
        self.declare_parameter('fusion_max_cost', 0.7)  # 最大匹配代价阈值
        self.declare_parameter('fusion_weight_distance', 0.6)  # 距离权重（无特征时主导）
        self.declare_parameter('fusion_weight_category', 0.4)  # 类别权重

        self.fusion_threshold = self.get_parameter('fusion_distance_threshold').value
        fusion_max_cost = self.get_parameter('fusion_max_cost').value
        fusion_weight_distance = self.get_parameter('fusion_weight_distance').value
        fusion_weight_category = self.get_parameter('fusion_weight_category').value

        # 初始化匈牙利匹配器（无视觉特征时，特征权重设为0）
        self._dual_camera_matcher = DualCameraMatcher(
            weight_distance=fusion_weight_distance,
            weight_feature=0.0,  # 当前不使用视觉特征
            weight_category=fusion_weight_category,
            distance_threshold=self.fusion_threshold,
            max_cost_threshold=fusion_max_cost,
        )
        self.get_logger().info(
            f"[FUSION] 匈牙利匹配器已初始化: "
            f"dist_thresh={self.fusion_threshold}m, max_cost={fusion_max_cost}, "
            f"weights=[dist:{fusion_weight_distance}, cat:{fusion_weight_category}]"
        )

        # 初始化 ByteTracker3D 跟踪器（跟踪融合后的结果）
        # 根据检测率动态计算参数
        detect_rate = max(self.auto_detect_rate, 1.0)  # 至少 1Hz
        track_buffer = int(detect_rate * 3.0)  # 3秒最大丢失时间
        confirm_frames = max(2, int(detect_rate * 0.3))  # 0.3秒确认时间，至少2帧

        tracker_cfg = TrackerConfig(
            match_thresh=0.15,       # 第一阶段: 15cm 匹配阈值
            second_thresh=0.25,      # 第二阶段: 25cm (ByteTrack低置信度恢复)
            track_buffer=track_buffer,
            confirm_frames=confirm_frames,
        )
        self._tracker = ByteTracker3D(tracker_cfg, self.get_logger())
        self.get_logger().info(
            f"[TRACKER] ByteTracker3D 已初始化: "
            f"match_thresh={tracker_cfg.match_thresh}m, "
            f"second_thresh={tracker_cfg.second_thresh}m, "
            f"buffer={track_buffer} ({detect_rate}Hz×3s), "
            f"confirm={confirm_frames}"
        )

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

        # 根据 detector_type 检查对应服务
        if self.detector_type == 'sam3':
            if not check_service(self.sam3_url, timeout=5.0):
                self.get_logger().error(f'SAM3 服务不可用: {self.sam3_url}')
                raise RuntimeError('SAM3 服务不可用')
            self.get_logger().info('  SAM3 service: OK')
        else:
            # 默认检查 DINO-X
            if not check_service(self.detector_url, timeout=5.0):
                self.get_logger().error(f'DINO-X 服务不可用: {self.detector_url}')
                raise RuntimeError('DINO-X 服务不可用')
            self.get_logger().info('  DINO-X service: OK')

        # CDM 服务可选
        if self.enable_depth_optimizer:
            if not check_service(self.depth_optimizer_url, timeout=3.0):
                self.get_logger().warn('CDM 服务不可用，禁用深度优化')
                self.enable_depth_optimizer = False
            else:
                self.get_logger().info('  CDM service: OK')

    def _init_components(self):
        """初始化双相机组件"""
        self.cameras = {}
        self.transformers = {}
        self.cores = {}

        # 相机配置
        camera_configs = []
        if self.enable_top:
            camera_configs.append(CameraConfig(
                name='top',
                color_topic=self.get_parameter('top_color_topic').value,
                depth_topic=self.get_parameter('top_depth_topic').value,
                info_topic=self.get_parameter('top_info_topic').value,
            ))
        if self.enable_chassis:
            camera_configs.append(CameraConfig(
                name='chassis',
                color_topic=self.get_parameter('chassis_color_topic').value,
                depth_topic=self.get_parameter('chassis_depth_topic').value,
                info_topic=self.get_parameter('chassis_info_topic').value,
            ))

        # 为每个相机初始化传感器和变换器
        for cfg in camera_configs:
            self.get_logger().info(f'初始化 {cfg.name} 相机...')
            
            # 同步传感器订阅器
            sensor = SyncedSensorSubscriber(
                node=self,
                color_topic=cfg.color_topic,
                depth_topic=cfg.depth_topic,
                info_topic=cfg.info_topic,
                lidar_topic=self.lidar_topic if cfg.name == 'top' else '',  # LiDAR 只给 top 用
                enable_lidar=self.enable_lidar and cfg.name == 'top',
                sync_slop=self.sync_slop,
            )

            if not sensor.connect(timeout=30.0):
                self.get_logger().error(f'{cfg.name} 相机连接失败')
                continue

            self.cameras[cfg.name] = sensor

            # 坐标变换器
            transformer = CoordinateTransformer()
            if self.extrinsics_dir:
                transformer.set_config_dir(self.extrinsics_dir)
            # 只对 chassis 相机应用外参后缀（用于测试新标定）
            suffix = self.extrinsics_suffix if cfg.name == 'chassis' else ''
            transformer.load_all_extrinsics(camera_name=cfg.name, suffix=suffix)
            self.transformers[cfg.name] = transformer

            # 感知核心
            self.cores[cfg.name] = ScenePerceptionCore(
                transformer=transformer,
                intrinsics=sensor.intrinsics,
                target_frame=self.target_frame,
                depth_optimizer=None,  # 稍后初始化共享的优化器
            )

            self.get_logger().info(f'  {cfg.name} 相机初始化完成')

        if not self.cameras:
            raise RuntimeError('没有可用的相机')

        # 共享的检测器（根据 detector_type 选择）
        if self.detector_type == 'sam3':
            detector_cfg = SimpleConfig(
                url=self.sam3_url,
                min_score=self.min_score,  # SAM3Online 会映射为 confidence
                resize=(1280, 720),
                return_mask=True,
            )
            self.detector = SAM3Online(detector_cfg)
            self.get_logger().info(f'使用 SAM3 检测器: {self.sam3_url}')
        else:
            detector_cfg = SimpleConfig(
                url=self.detector_url,
                min_score=self.min_score,
                resize=(1280, 720),
            )
            self.detector = DinoXDetectorOnline(detector_cfg)
            self.get_logger().info(f'使用 DINO-X 检测器: {self.detector_url}')

        # 共享的深度优化器
        self.depth_optimizer = None
        if self.enable_depth_optimizer:
            optimizer_cfg = SimpleConfig(
                url=self.depth_optimizer_url,
                chosen_policy='dn',
            )
            self.depth_optimizer = DepthOptimizerOnline(optimizer_cfg)
            # 给每个 core 设置共享的优化器
            for core in self.cores.values():
                core.depth_optimizer = self.depth_optimizer

    def _init_publishers(self):
        """初始化发布器"""
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10  # 增加深度，避免高频发布时消息被覆盖
        )

        # 每个相机的独立发布器
        self.pubs = {}
        if self.enable_top:
            self.pubs['top'] = {
                'objects': self.create_publisher(Object3DArray, '~/top/objects_3d', latched_qos),
                'depth': self.create_publisher(Image, '~/top/optimized_depth', latched_qos),
            }
        if self.enable_chassis:
            self.pubs['chassis'] = {
                'objects': self.create_publisher(Object3DArray, '~/chassis/objects_3d', latched_qos),
                'depth': self.create_publisher(Image, '~/chassis/optimized_depth', latched_qos),
            }

        # 融合结果发布器
        self.pub_fused = self.create_publisher(Object3DArray, '~/fused/objects_3d', latched_qos)

    def _init_worker_threads(self):
        """初始化事件驱动 Worker 线程（每相机一个，无队列）"""
        if not self.cameras or self.auto_detect_rate <= 0:
            return

        self.get_logger().info(f'启动 {len(self.cameras)} 个事件驱动 Worker...')

        for camera_name in self.cameras.keys():
            t = threading.Thread(
                target=self._detection_worker,
                args=(camera_name,),
                name=f'DetectionWorker-{camera_name}',
                daemon=True
            )
            t.start()
            self._worker_threads[camera_name] = t

        self.get_logger().info(f'Worker 已启动: {list(self._worker_threads.keys())}')

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
            f'min_score={self._current_min_score:.2f}'
        )

    def _detection_worker(self, camera_name: str):
        """
        事件驱动 Worker 线程（每相机独立，无队列）

        轮询 sensor.get_latest_data()，帧去重后执行检测 + 发布。
        """
        self.get_logger().info(f'[Worker-{camera_name}] 启动（事件驱动）')

        sensor = self.cameras.get(camera_name)
        transformer = self.transformers.get(camera_name)
        if not sensor:
            self.get_logger().error(f'[Worker-{camera_name}] sensor 不存在，退出')
            return

        last_stamp = 0.0
        optical_frame = f'{camera_name}_camera_optical_frame'

        while not self._shutdown and rclpy.ok():
            try:
                # 轮询最新数据
                data = sensor.get_latest_data()
                if data is None:
                    time.sleep(0.015)
                    continue

                # 数据过旧检查
                if data.get('age', 0) > 3.0:
                    self._log.warn(f'[{camera_name}] 数据过旧: {data["age"]:.1f}s', period=10.0)
                    time.sleep(0.015)
                    continue

                # 帧去重
                ts = data.get('timestamp')
                if ts is not None:
                    stamp = stamp_to_sec(ts) if hasattr(ts, 'sec') else float(ts)
                else:
                    stamp = time.time()

                if stamp <= last_stamp:
                    time.sleep(0.010)
                    continue
                last_stamp = stamp

                _tw0 = time.time()

                # TF 查询（使用正确的 API）
                optical_to_base = None
                if transformer:
                    try:
                        optical_to_base = transformer.get_transform('optical_to_base')
                    except Exception:
                        optical_to_base = None

                # 构造检测任务（数据已深拷贝于 get_latest_data，无需额外 copy）
                task = DetectionTask(
                    camera_name=camera_name,
                    rgb=data['rgb'],
                    depth=data['depth'],
                    intrinsics=sensor.intrinsics,
                    prompt=self._current_prompt,
                    optical_to_base_matrix=optical_to_base,
                    timestamp_sec=_tw0,
                    frame_id=data.get('frame_id', optical_frame)
                )

                # 执行检测
                result = self._run_detection_task(task)

                _tw1 = time.time()

                # 发布结果
                if result and camera_name in self.pubs:
                    self.pubs[camera_name]['objects'].publish(result)

                    _tw2 = time.time()

                    # 线程安全地更新结果并尝试融合
                    with self._results_lock:
                        self._last_results[camera_name] = result
                        self._try_publish_fusion()

                    _tw3 = time.time()
                    self._publish_status()

                    self.get_logger().info(
                        f'[{camera_name}] WORKER: '
                        f'detect={(_tw1-_tw0)*1000:.0f}ms '
                        f'pub={(_tw2-_tw1)*1000:.0f}ms '
                        f'fusion={(_tw3-_tw2)*1000:.0f}ms '
                        f'total={(_tw3-_tw0)*1000:.0f}ms')

            except Exception as e:
                self._log.warn(f'[Worker-{camera_name}] 检测失败: {e}', period=5.0)
                import traceback
                traceback.print_exc()

        self.get_logger().info(f'[Worker-{camera_name}] 停止')

    def _run_detection_task(self, task: DetectionTask) -> Optional[Object3DArray]:
        """
        执行单个检测任务（在后台线程）
        
        与 _run_single_camera_detection 类似，但：
        - 数据已预加载，无需等待
        - TF 已预查询（可选）
        """
        camera_name = task.camera_name
        core = self.cores.get(camera_name)
        
        if not core:
            return None
        
        result = Object3DArray()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = self.target_frame
        
        rgb = task.rgb
        depth = task.depth

        _t0 = time.time()

        # 1. 并行调用检测 和 CDM 深度优化（带 API 粒度计时）
        _det_timing = {}  # sam3_http, mask_decode
        _cdm_timing = {}  # cdm_http

        def _timed_detection():
            _dt0 = time.time()
            # SAM3/DINOX forward (HTTP call)
            _timing_inner = {}
            raw_result = self.detector.forward(
                text=task.prompt, rgb=rgb,
                min_score=self._current_min_score,
                iou_threshold=self._current_iou_threshold,
                _timing=_timing_inner
            )
            _dt1 = time.time()
            _det_timing['api_http'] = _dt1 - _dt0
            _det_timing.update(_timing_inner)

            # Mask decode
            objects = raw_result.result.get('objects', [])
            if not objects:
                _det_timing['mask_decode'] = 0
                _det_timing['n_objects'] = 0
                return []

            import pycocotools.mask as coco_mask
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

            decoded_masks = {}
            if rle_masks:
                masks_array = coco_mask.decode(rle_masks)
                if masks_array.ndim == 2:
                    masks_array = masks_array[:, :, np.newaxis]
                for idx, obj_idx in enumerate(obj_indices):
                    decoded_masks[obj_idx] = masks_array[:, :, idx].astype(np.uint8)

            for obj_idx, bbox in bbox_masks:
                binary_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, bbox)
                binary_mask[y1:y2, x1:x2] = 1
                decoded_masks[obj_idx] = binary_mask

            detections = []
            category_counts = {}
            for i, obj in enumerate(objects):
                binary_mask = decoded_masks.get(i)
                if binary_mask is None:
                    continue
                if binary_mask.sum() < 300:
                    continue
                category = obj.get('category', task.prompt)
                score = obj.get('score', 0)
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
                obj_id = f"{category}_{category_counts[category]}"
                detections.append({
                    'object_id': obj_id,
                    'category': category,
                    'score': score,
                    'bbox': obj.get('bbox', [0, 0, 0, 0]),
                    'mask': binary_mask,
                })

            _dt2 = time.time()
            _det_timing['mask_decode'] = _dt2 - _dt1
            _det_timing['n_objects'] = len(detections)
            return detections

        def _timed_depth_optimize():
            _ct_start = time.time()
            _cdm_inner = {}
            if core.depth_optimizer:
                # 必须转换为 uint16 mm（CDM 服务要求的格式）
                depth_mm = (depth * 1000).astype(np.uint16) if depth.dtype != np.uint16 else depth
                result = core.depth_optimizer.forward(
                    rgb, depth_mm, chosen_policy='dn', _timing=_cdm_inner
                )
            else:
                result = None
            _ct_end = time.time()
            _cdm_timing['api_total'] = _ct_end - _ct_start
            _cdm_timing.update(_cdm_inner)
            if result and result.get('success') and 'depth' in result:
                return result['depth'].astype(np.float32) / 1000.0
            return depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth

        # 使用共享线程池
        _t_submit = time.time()
        future_detection = self._detection_executor.submit(_timed_detection)
        future_depth = self._detection_executor.submit(_timed_depth_optimize)

        detections = future_detection.result(timeout=10.0)
        optimized_depth = future_depth.result(timeout=10.0)

        _t1 = time.time()

        # 发布优化后的深度图
        try:
            depth_mm = (optimized_depth * 1000).astype(np.uint16)
            depth_msg = self._bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = f'{camera_name}_camera_optical_frame'
            if camera_name in self.pubs:
                self.pubs[camera_name]['depth'].publish(depth_msg)
        except Exception:
            pass

        # 格式化 timing 字符串
        det_http_ms = _det_timing.get('api_http', 0) * 1000
        mask_ms = _det_timing.get('mask_decode', 0) * 1000
        cdm_total_ms = _cdm_timing.get('api_total', 0) * 1000
        n_obj = _det_timing.get('n_objects', 0)
        # SAM3 内部 timing
        sam3_pre = _det_timing.get('sam3_preprocess', 0)
        sam3_enc = _det_timing.get('sam3_encode', 0)
        sam3_http = _det_timing.get('sam3_http', 0)
        # CDM 内部 timing
        cdm_pre = _cdm_timing.get('cdm_preprocess', 0)
        cdm_enc = _cdm_timing.get('cdm_encode', 0)
        cdm_http = _cdm_timing.get('cdm_http', 0)
        cdm_post = _cdm_timing.get('cdm_postprocess', 0)
        # 并行等待时间
        parallel_ms = (_t1 - _t_submit) * 1000

        if not detections:
            self.get_logger().info(
                f'[{camera_name}] PERF: '
                f'sam3=[enc:{sam3_enc:.0f}+http:{sam3_http:.0f}={det_http_ms:.0f}ms] '
                f'cdm=[pre:{cdm_pre:.0f}+enc:{cdm_enc:.0f}+http:{cdm_http:.0f}+post:{cdm_post:.0f}={cdm_total_ms:.0f}ms] '
                f'parallel={parallel_ms:.0f}ms wall={(_t1-_t0)*1000:.0f}ms | 0 objs')
            return result

        _t2 = time.time()

        # 2. 批量 bbox-scoped 3D 测量
        measurement_results = core.batch_camera_3d_percept(
            optimized_depth, detections, skip_transform=True
        )

        # 过滤无效测量
        valid_items = []
        optical_centroids = []
        for det, meas in zip(detections, measurement_results):
            if meas.valid:
                valid_items.append((det, meas))
                optical_centroids.append(meas.centroid_optical)

        if not valid_items:
            self.get_logger().info(
                f'[{camera_name}] PERF: '
                f'sam3=[enc:{sam3_enc:.0f}+http:{sam3_http:.0f}={det_http_ms:.0f}ms] '
                f'mask={mask_ms:.0f}ms '
                f'cdm=[pre:{cdm_pre:.0f}+enc:{cdm_enc:.0f}+http:{cdm_http:.0f}+post:{cdm_post:.0f}={cdm_total_ms:.0f}ms] '
                f'parallel={parallel_ms:.0f}ms '
                f'wall={(_t2-_t0)*1000:.0f}ms | {n_obj} det, 0 valid')
            return result

        # 3. 批量坐标变换到 base_link
        if task.optical_to_base_matrix is not None:
            # 使用预计算的变换矩阵
            optical_points = np.array(optical_centroids)
            # 齐次坐标变换
            points_h = np.hstack([optical_points, np.ones((len(optical_points), 1))])
            target_centroids = (task.optical_to_base_matrix @ points_h.T).T[:, :3]
        else:
            # 回退：使用 transformer（可能在线程中不安全，但尽量尝试）
            transformer = self.transformers.get(camera_name)
            if transformer:
                optical_points = np.array(optical_centroids)
                target_centroids = transformer.optical_to_target(optical_points, self.target_frame)
            else:
                target_centroids = np.array(optical_centroids)  # 无变换

        # 4. 构建 Object3D 消息
        distance_offset = self.distance_offset_default

        for i, (det, camera_result) in enumerate(valid_items):
            obj = Object3D()
            obj.object_id = f"{camera_name}_{det['object_id']}"
            obj.category = det['category']
            obj.score = float(det['score'])
            obj.bbox = [float(x) for x in det['bbox']]

            # 应用距离补偿
            target_centroid = target_centroids[i]
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
            obj.source_camera = camera_name

            result.objects.append(obj)

        _t3 = time.time()
        self.get_logger().info(
            f'[{camera_name}] PERF: '
            f'sam3=[enc:{sam3_enc:.0f}+http:{sam3_http:.0f}={det_http_ms:.0f}ms] '
            f'mask={mask_ms:.0f}ms '
            f'cdm=[pre:{cdm_pre:.0f}+enc:{cdm_enc:.0f}+http:{cdm_http:.0f}+post:{cdm_post:.0f}={cdm_total_ms:.0f}ms] '
            f'parallel={parallel_ms:.0f}ms '
            f'meas={(_t3-_t2)*1000:.0f}ms '
            f'wall={(_t3-_t0)*1000:.0f}ms | {len(result.objects)} objs')
        return result

    def handle_detect(self, request, response):
        """处理检测服务请求 - 支持指定相机（简化版：无融合）"""
        response.success = True

        try:
            # 解析相机选择（默认全部）
            cameras = []
            if request.camera_id:
                # camera_id 可以是 "top", "chassis", 或 "all"
                if request.camera_id == 'all':
                    cameras = list(self.cameras.keys())
                elif request.camera_id in self.cameras:
                    cameras = [request.camera_id]
                else:
                    response.success = False
                    response.error_message = f'未知相机: {request.camera_id}'
                    return response
            else:
                cameras = list(self.cameras.keys())

            # 执行检测
            results = self._run_multi_camera_detection(
                prompt=request.prompt,
                cameras=cameras
            )

            # 返回指定相机或第一个相机的结果
            if results:
                if request.camera_id and request.camera_id != 'all':
                    # 返回指定相机结果
                    response.result = results.get(request.camera_id)
                else:
                    # 返回第一个相机结果
                    primary_camera = cameras[0]
                    response.result = results[primary_camera]

                # 发布所有结果（不做融合）
                for camera_name, result in results.items():
                    if camera_name in self.pubs and result:
                        self.pubs[camera_name]['objects'].publish(result)

                # 智能融合发布（聚类+融合）
                if len(results) >= 2:
                    fused = self._fuse_results(results)
                    if fused:
                        self.pub_fused.publish(fused)

            self._publish_status()

        except Exception as e:
            response.success = False
            response.error_message = f'检测失败: {e}'
            self.get_logger().error(response.error_message)
            import traceback
            traceback.print_exc()

        return response

    def _run_multi_camera_detection(self, prompt: str, cameras: List[str]) -> Dict[str, Object3DArray]:
        """
        并行执行多相机检测
        
        Returns:
            Dict[str, Object3DArray]: 每个相机的检测结果
        """
        results = {}

        def detect_single_camera(camera_name: str) -> Tuple[str, Optional[Object3DArray]]:
            """单相机检测函数"""
            try:
                result = self._run_single_camera_detection(camera_name, prompt)
                return camera_name, result
            except Exception as e:
                self.get_logger().warn(f'{camera_name} 相机检测失败: {e}')
                return camera_name, None

        # 并行执行所有相机检测
        with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            futures = {executor.submit(detect_single_camera, name): name for name in cameras}
            
            for future in as_completed(futures):
                camera_name, result = future.result()
                if result:
                    results[camera_name] = result

        return results

    def _run_single_camera_detection(self, camera_name: str, prompt: str) -> Optional[Object3DArray]:
        """执行单个相机的检测流程"""
        sensor = self.cameras.get(camera_name)
        transformer = self.transformers.get(camera_name)
        core = self.cores.get(camera_name)

        if not sensor or not transformer or not core:
            return None

        result = Object3DArray()
        result.header.stamp = self.get_clock().now().to_msg()
        result.frame_id = self.target_frame

        # 1. 采集同步数据
        try:
            data = sensor.get_synced_data(
                timeout=2.0,
                max_age=self.data_max_age,
                strict=self.strict_data_freshness
            )
        except TimeoutError as e:
            self.get_logger().warn(f'[{camera_name}] 获取数据超时: {e}')
            return None

        if not data:
            return None

        rgb = data['rgb']
        depth = data['depth']

        # 2. 并行调用 DINO-X 检测 和 CDM 深度优化
        def _timed_detection():
            return self._detect_objects(prompt, rgb)

        def _timed_depth_optimize():
            return core.optimize_depth(rgb, depth)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_detection = executor.submit(_timed_detection)
            future_depth = executor.submit(_timed_depth_optimize)

            detections = future_detection.result()
            optimized_depth = future_depth.result()

        # 发布优化后的深度图
        try:
            depth_mm = (optimized_depth * 1000).astype(np.uint16)
            depth_msg = self._bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
            depth_msg.header.stamp = data['timestamp']
            depth_msg.header.frame_id = f'{camera_name}_camera_optical_frame'
            if camera_name in self.pubs:
                self.pubs[camera_name]['depth'].publish(depth_msg)
        except Exception as e:
            pass

        if not detections:
            return result

        self.get_logger().info(f'[{camera_name}] DETECT: Found {len(detections)} objects')

        # 3. 批量 bbox-scoped 3D 测量
        measurement_results = core.batch_camera_3d_percept(
            optimized_depth, detections, skip_transform=True
        )

        # 过滤无效测量
        valid_items = []
        optical_centroids = []
        for det, meas in zip(detections, measurement_results):
            if meas.valid:
                valid_items.append((det, meas))
                optical_centroids.append(meas.centroid_optical)

        if not valid_items:
            return result

        # 4. 批量坐标变换到 base_link
        optical_points = np.array(optical_centroids)
        target_centroids = transformer.optical_to_target(optical_points, self.target_frame)

        # 5. 构建 Object3D 消息
        distance_offset = self.distance_offset_default

        for i, (det, camera_result) in enumerate(valid_items):
            obj = Object3D()
            obj.object_id = f"{camera_name}_{det['object_id']}"
            obj.category = det['category']
            obj.score = float(det['score'])
            obj.bbox = [float(x) for x in det['bbox']]

            # 应用距离补偿
            target_centroid = target_centroids[i]
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
            obj.source_camera = camera_name

            result.objects.append(obj)

        self.get_logger().info(f'[{camera_name}] DETECT: Measured {len(result.objects)} objects')
        return result

    def _simple_merge_results(self, results: Dict[str, Object3DArray]) -> Object3DArray:
        """
        简单合并双相机检测结果（仅拼接，不去重）
        
        策略：
        直接将所有相机的检测结果拼接在一起，添加相机前缀区分
        """
        merged = Object3DArray()
        merged.header.stamp = self.get_clock().now().to_msg()
        merged.header.frame_id = self.target_frame

        total_count = 0
        for camera_name, result in results.items():
            for obj in result.objects:
                # 添加相机前缀到 object_id
                obj_copy = Object3D()
                obj_copy.object_id = obj.object_id  # 已有相机前缀，不重复添加
                obj_copy.category = obj.category
                obj_copy.score = obj.score
                obj_copy.bbox = obj.bbox
                obj_copy.position = obj.position
                obj_copy.distance = obj.distance
                obj_copy.confidence = obj.confidence
                # 复制新增的调试字段
                obj_copy.position_optical = obj.position_optical
                obj_copy.depth = obj.depth
                obj_copy.source_camera = obj.source_camera
                merged.objects.append(obj_copy)
                total_count += 1

        self.get_logger().debug(f'[MERGE] 合并 {len(results)} 个相机结果，共 {total_count} 个物体')
        return merged

    def _try_publish_fusion(self):
        """
        尝试融合并发布多相机结果（用于auto_detect模式）

        流程：
        1. 收集各相机最新结果
        2. 匈牙利融合去重
        3. ByteTracker3D 跟踪（分配持久 track_id）
        4. 发布跟踪后的结果
        """
        try:
            # 收集有效的最新结果（只考虑相机，排除 'fused' 键）
            valid_results = {}
            for camera_name in ['top', 'chassis']:
                result = self._last_results.get(camera_name)
                if result and len(result.objects) > 0:
                    valid_results[camera_name] = result

            # 如果有任意相机的结果，进行融合
            if len(valid_results) >= 1:
                fused = self._fuse_results(valid_results)
                if fused and len(fused.objects) > 0:
                    # === ByteTracker3D 跟踪 ===
                    tracked_result = self._apply_tracking(fused)

                    # 发布跟踪后的结果
                    self.pub_fused.publish(tracked_result)
                    self._last_results['fused'] = tracked_result
                    self.get_logger().info(
                        f'[FUSION] 发布 fused 话题: {len(tracked_result.objects)} objects (tracked)'
                    )
                else:
                    self.get_logger().warn(
                        f'[FUSION] fused 结果为空或无效: '
                        f'fused={fused is not None}, count={len(fused.objects) if fused else 0}'
                    )
        except Exception as e:
            import traceback
            self.get_logger().error(f'[AUTO] 融合发布失败: {e}')
            self.get_logger().error(traceback.format_exc())

    def _apply_tracking(self, fused: Object3DArray) -> Object3DArray:
        """
        应用 ByteTracker3D 跟踪算法

        ByteTrack 核心逻辑：
        - 高置信度检测（source='fused'）优先匹配
        - 低置信度检测（source='*_only'）用于恢复丢失轨迹
        - 返回带有持久 track_id 的结果
        """
        # 将 Object3D 转换为 tracker 期望的格式
        fused_objects = []
        for obj in fused.objects:
            # 安全处理 bbox（避免 numpy 数组真值判断问题）
            bbox = obj.bbox
            if bbox is not None and len(bbox) >= 4:
                bbox_list = list(bbox[:4])
            else:
                bbox_list = [0, 0, 0, 0]

            fused_objects.append({
                'object_id': obj.object_id,
                'category': obj.category,
                'score': obj.score,
                'position': np.array([obj.position.x, obj.position.y, obj.position.z]),
                'source': obj.source_camera,  # 'fused', 'top', 'chassis'
                'confidence': obj.confidence,
                'bbox': bbox_list,
                # 保留原始对象用于后续构建消息
                '_original': obj,
            })

        # 调用 ByteTracker3D
        tracked_objects = self._tracker.update(fused_objects)

        # 构建跟踪后的 Object3DArray
        result = Object3DArray()
        result.header = fused.header

        for track in tracked_objects:
            original_obj = track.get('_original')
            if not original_obj:
                continue

            obj = Object3D()

            # 根据是否有 track_id 决定 object_id
            track_id = track.get('track_id')
            if track_id is not None:
                # 已确认轨迹：使用持久 track_id
                obj.object_id = f"track_{track_id}"
            else:
                # 未确认检测：保持原始 object_id
                obj.object_id = original_obj.object_id

            obj.category = track['category']
            obj.score = track.get('score', original_obj.score)
            obj.bbox = original_obj.bbox
            obj.position = Point(
                x=float(track['position'][0]),
                y=float(track['position'][1]),
                z=float(track['position'][2])
            )
            obj.distance = float(np.linalg.norm(track['position']))
            obj.confidence = track.get('confidence', original_obj.confidence)
            obj.position_optical = original_obj.position_optical
            obj.depth = original_obj.depth
            obj.source_camera = track.get('source', original_obj.source_camera)
            result.objects.append(obj)

        return result

    def _fuse_results(self, results: Dict[str, Object3DArray]) -> Object3DArray:
        """
        融合双相机检测结果（使用匈牙利匹配算法）

        策略：
        1. 将 Object3D 转换为 DetectionWithDepth
        2. 使用匈牙利算法进行全局最优匹配（考虑距离+类别）
        3. 匹配成功的进行位置融合
        4. 未匹配的直接保留
        """
        fused = Object3DArray()
        fused.header.stamp = self.get_clock().now().to_msg()
        fused.header.frame_id = self.target_frame

        # 分离两个相机的检测结果
        chassis_objects = list(results.get('chassis', Object3DArray()).objects)
        top_objects = list(results.get('top', Object3DArray()).objects)

        # 如果只有一个相机有结果，直接返回
        if not chassis_objects and not top_objects:
            return fused
        if not chassis_objects:
            fused.objects.extend(top_objects)
            self.get_logger().info(f'[FUSION] 仅top相机: {len(top_objects)} objects')
            return fused
        if not top_objects:
            fused.objects.extend(chassis_objects)
            self.get_logger().info(f'[FUSION] 仅chassis相机: {len(chassis_objects)} objects')
            return fused

        # 转换为 DetectionWithDepth
        chassis_dets = [self._object3d_to_detection(obj, 'chassis') for obj in chassis_objects]
        top_dets = [self._object3d_to_detection(obj, 'top') for obj in top_objects]

        # 使用匈牙利匹配器（直接调用 matcher.match 获取索引）
        try:
            matches, unmatched_c_idx, unmatched_t_idx = self._dual_camera_matcher.matcher.match(
                chassis_dets, top_dets
            )

            # 处理匹配成功的对（先融合 MatchPair）
            for match in matches:
                det_c = chassis_dets[match.chassis_idx]
                det_t = top_dets[match.top_idx]
                fused_match = self._dual_camera_matcher.fuser.fuse_match(match, det_c, det_t)

                obj_chassis = chassis_objects[match.chassis_idx]
                obj_top = top_objects[match.top_idx]
                fused_obj = self._fuse_matched_pair(obj_chassis, obj_top, fused_match)
                if fused_obj:
                    fused.objects.append(fused_obj)

            # 添加未匹配的 chassis 检测（直接使用索引，避免 index() 的 numpy 比较问题）
            for idx in unmatched_c_idx:
                fused.objects.append(chassis_objects[idx])

            # 添加未匹配的 top 检测
            for idx in unmatched_t_idx:
                fused.objects.append(top_objects[idx])

            total_input = len(chassis_objects) + len(top_objects)
            self.get_logger().info(
                f'[FUSION] 匈牙利匹配: {total_input} objects -> {len(fused.objects)} unique '
                f'(matched:{len(matches)}, unmatched_c:{len(unmatched_c_idx)}, unmatched_t:{len(unmatched_t_idx)})'
            )

        except Exception as e:
            import traceback
            self.get_logger().error(f'[FUSION] 匈牙利匹配失败: {e}')
            self.get_logger().error(traceback.format_exc())
            # 回退：简单合并所有结果
            fused.objects.extend(chassis_objects)
            fused.objects.extend(top_objects)

        return fused

    def _object3d_to_detection(self, obj: Object3D, camera_name: str) -> DetectionWithDepth:
        """将 Object3D 转换为 DetectionWithDepth"""
        # 安全地处理 bbox（避免 numpy 数组的布尔值判断问题）
        bbox = obj.bbox
        if bbox is not None and len(bbox) >= 4:
            bbox_arr = np.array(bbox[:4], dtype=np.float32)
        else:
            bbox_arr = np.array([0, 0, 0, 0], dtype=np.float32)

        return DetectionWithDepth(
            bbox=bbox_arr,
            category=obj.category,
            detection_score=obj.score,
            position_3d=np.array([obj.position.x, obj.position.y, obj.position.z]),
            distance=obj.distance,
            visual_features=None,  # 当前不使用视觉特征
            camera_source=camera_name,
        )

    def _fuse_matched_pair(self, obj_chassis: Object3D, obj_top: Object3D, match) -> Optional[Object3D]:
        """融合一对匹配的检测结果"""
        try:
            pos_chassis = np.array([obj_chassis.position.x, obj_chassis.position.y, obj_chassis.position.z])
            pos_top = np.array([obj_top.position.x, obj_top.position.y, obj_top.position.z])

            # 使用匹配结果中的融合位置
            fused_pos = match.fused_position

            # 构建融合后的 Object3D
            fused_obj = Object3D()
            fused_obj.object_id = f"fused_{obj_chassis.category}_{id(fused_obj) % 10000}"
            fused_obj.category = match.fused_category if match.fused_category else obj_chassis.category
            fused_obj.score = max(obj_chassis.score, obj_top.score)
            fused_obj.bbox = obj_chassis.bbox  # 使用 chassis 的 bbox
            fused_obj.source_camera = "fused"

            # 融合位置
            fused_obj.position = Point(
                x=float(fused_pos[0]),
                y=float(fused_pos[1]),
                z=float(fused_pos[2])
            )
            fused_obj.distance = float(np.linalg.norm(fused_pos))
            fused_obj.depth = fused_obj.distance

            # 融合置信度
            fused_obj.confidence = match.fused_confidence if match.fused_confidence > 0 else \
                (obj_chassis.confidence + obj_top.confidence) / 2.0

            self.get_logger().debug(
                f"[FUSE] {fused_obj.category}: quality={match.quality.value}, "
                f"dist_3d={match.distance_3d:.3f}m, conf={fused_obj.confidence:.2f}"
            )

            return fused_obj

        except Exception as e:
            self.get_logger().warn(f"[FUSE] 融合匹配对失败: {e}")
            # 回退：返回置信度高的
            return obj_chassis if obj_chassis.confidence >= obj_top.confidence else obj_top

    def _detect_objects(self, prompt: str, rgb: np.ndarray) -> list:
        """检测物体并解码 mask（复用原代码逻辑）"""
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

        decoded_masks = {}
        if rle_masks:
            masks_array = coco_mask.decode(rle_masks)
            if masks_array.ndim == 2:
                masks_array = masks_array[:, :, np.newaxis]
            for idx, obj_idx in enumerate(obj_indices):
                decoded_masks[obj_idx] = masks_array[:, :, idx].astype(np.uint8)

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

    def _publish_status(self):
        """发布当前状态"""
        status = PerceptionStatus()
        status.prompt = self._current_prompt
        status.min_score = float(self._current_min_score)
        status.object_count = sum(
            len(r.objects) for r in self._last_results.values() if r
        )
        self.pub_status.publish(status)


    def destroy_node(self):
        """清理资源"""
        self.get_logger().info('正在停止 Worker 线程...')
        self._shutdown = True

        # 等待所有工作线程结束（最多5秒）
        for t in self._worker_threads.values():
            t.join(timeout=5.0)

        # 关闭共享线程池
        self._detection_executor.shutdown(wait=False)

        self.get_logger().info('Worker 线程已停止')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
