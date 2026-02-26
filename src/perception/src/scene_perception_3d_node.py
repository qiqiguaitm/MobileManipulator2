#!/usr/bin/python3
"""
Scene Perception 3D - ROS 节点

3D 场景感知节点，集成相机和 LiDAR 数据进行物体 3D 定位。
"""

import os
import sys

# 禁用代理 (避免内网服务被代理拦截)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# 添加 src 目录到 Python 路径 (用于导入同目录下的模块)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 添加 ROS devel Python 路径 (用于导入 perception.msg/srv)
_devel_python_path = os.path.realpath(os.path.join(
    os.path.dirname(_src_dir), '..', '..', 'devel', 'lib', 'python3', 'dist-packages'
))
if os.path.isdir(_devel_python_path) and _devel_python_path not in sys.path:
    sys.path.insert(0, _devel_python_path)

# 添加 ROS noetic Python 路径
_ros_python_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_python_path) and _ros_python_path not in sys.path:
    sys.path.insert(0, _ros_python_path)

import rospy
import requests
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pycocotools import mask as coco_mask
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from synced_sensor_subscriber import SyncedSensorSubscriber
from coordinate_transformer import CoordinateTransformer
from scene_perception_core import ScenePerceptionCore, MeasurementResult, SimpleConfig
from percept import DinoXDetectorOnline, DepthOptimizerOnline
from perception_visualizer import PerceptionVisualizer

# Logging
from common.logger import get_logger

# ROS 消息
from perception.msg import Object3D, Object3DArray, PerceptionConfig, PerceptionStatus
from perception.srv import DetectObjects, DetectObjectsResponse
from geometry_msgs.msg import Point


class ScenePerception3DNode:
    """场景 3D 感知 ROS 节点"""

    def __init__(self):
        rospy.init_node('scene_perception_3d')
        self.log = get_logger("Percept3D")

        # 加载配置
        self._load_config()

        # 启动时健康检查
        self._health_check()

        # 初始化组件
        self._init_components()

        # Prompt 订阅 (可选)
        self._current_prompt = self.default_prompt
        self._current_min_score = self.min_score
        self._current_iou_threshold = self.iou_threshold
        self._last_detect_time = 0.0
        self._last_object_count = 0
        self._last_categories = []

        if self.prompt_source == 'topic':
            self._prompt_sub = rospy.Subscriber(
                self.prompt_topic, String, self._prompt_callback
            )
            self.log.info(f" Prompt 订阅: {self.prompt_topic}")

        # 配置订阅 (PerceptPanel 用)
        self._config_sub = rospy.Subscriber(
            '/perception/config', PerceptionConfig, self._config_callback
        )
        self.log.info(" 配置订阅: /perception/config")

        # ROS 接口
        self.pub = rospy.Publisher('~objects_3d', Object3DArray, queue_size=1, latch=True)
        self.pub_optimized_depth = rospy.Publisher('~optimized_depth', Image, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher('/perception/status', PerceptionStatus, queue_size=1)
        self.srv = rospy.Service('~detect', DetectObjects, self.handle_detect)

        # 自动检测定时器
        self._auto_detect_timer = None
        if self.auto_detect_rate > 0:
            self._auto_detect_timer = rospy.Timer(
                rospy.Duration(1.0 / self.auto_detect_rate),
                self._auto_detect_callback
            )
            self.log.info(f" 自动检测已启用: {self.auto_detect_rate} Hz")

        self.log.info(f" 节点启动完成")
        self.log.info(f"  Camera: {self.camera_name}")
        self.log.info(f"  LiDAR: {'启用' if self.enable_lidar else '禁用'}")
        self.log.info(f"  Target frame: {self.target_frame}")
        self.log.info(f"  Service: ~detect")
        self.log.info(f"  Topic: ~objects_3d")
        self.log.info(f"  Auto detect: {'%.1f Hz' % self.auto_detect_rate if self.auto_detect_rate > 0 else '禁用'}")

    def _load_config(self):
        """加载 ROS 参数"""
        ns = '~'

        # 相机配置
        self.camera_name = rospy.get_param(f'{ns}camera_name', 'top')

        # LiDAR 配置
        self.lidar_topic = rospy.get_param(f'{ns}lidar_topic', '/lidar/chassis/point_cloud')
        self.enable_lidar = rospy.get_param(f'{ns}enable_lidar', True)

        # Prompt 配置
        self.prompt_source = rospy.get_param(f'{ns}prompt_source', 'service')
        self.prompt_topic = rospy.get_param(f'{ns}prompt_topic', '/user/prompt')
        self.default_prompt = rospy.get_param(f'{ns}default_prompt', 'bottle.cup.box')

        # 检测服务
        self.detector_url = rospy.get_param(f'{ns}detector_url', 'http://192.168.112.14:10086')
        self.detector_timeout = rospy.get_param(f'{ns}detector_timeout', 3.0)
        self.min_score = rospy.get_param(f'{ns}min_score', 0.25)
        self.iou_threshold = rospy.get_param(f'{ns}iou_threshold', 0.5)
        self.detector_warmup = rospy.get_param(f'{ns}detector_warmup', 0)

        # 深度优化服务
        self.depth_optimizer_url = rospy.get_param(f'{ns}depth_optimizer_url', 'http://192.168.112.14:8086')
        self.enable_depth_optimizer = rospy.get_param(f'{ns}enable_depth_optimizer', True)
        self.depth_optimizer_warmup = rospy.get_param(f'{ns}depth_optimizer_warmup', 0)

        # 坐标系配置
        self.target_frame = rospy.get_param(f'{ns}target_frame', 'base_link')

        # 距离补偿参数（相机测距系统性偏差修正）
        # 默认值，运行时会动态读取最新值
        self.distance_offset_default = 0.30  # 默认 +30cm

        # 功能开关
        self.enable_accuracy_log = rospy.get_param(f'{ns}enable_accuracy_log', False)
        self.enable_visualization = rospy.get_param(f'{ns}enable_visualization', False)
        self.results_dir = rospy.get_param(f'{ns}results_dir', '/home/agilex/MobileManipulator/src/perception/results')

        # 自动检测配置
        self.auto_detect_rate = rospy.get_param(f'{ns}auto_detect_rate', 1.0)  # Hz, 0=禁用

        # 时间同步参数
        self.sync_slop = rospy.get_param(f'{ns}sync_slop', 0.1)
        self.data_max_age = rospy.get_param(f'{ns}data_max_age', 0.5)

    def _health_check(self):
        """启动时健康检查"""
        self.log.info(" 健康检查...")

        # 检查 DINO-X 服务
        if not self._check_service(self.detector_url, timeout=5.0):
            self.log.error(f" DINO-X 服务不可用: {self.detector_url}")
            raise RuntimeError("DINO-X 服务不可用")
        self.log.info("  DINO-X service: OK")

        # CDM 服务可选，失败时自动禁用
        if self.enable_depth_optimizer:
            if not self._check_service(self.depth_optimizer_url, timeout=3.0):
                self.log.warn(f" CDM 服务不可用，禁用深度优化")
                self.enable_depth_optimizer = False
            else:
                self.log.info("  CDM service: OK")

    def _check_service(self, url: str, timeout: float = 3.0) -> bool:
        """检查 HTTP 服务是否可用"""
        try:
            # 显式禁用代理
            resp = requests.get(url, timeout=timeout, proxies={'http': None, 'https': None})
            return resp.status_code in (200, 404, 405)  # 服务在线
        except Exception:
            return False

    def _init_components(self):
        """初始化组件"""
        # ROS-CV Bridge
        self.bridge = CvBridge()

        # 构建相机 topic
        camera_topics = self._build_camera_topics(self.camera_name)

        # 同步传感器订阅器
        self.sensor = SyncedSensorSubscriber(
            color_topic=camera_topics['color'],
            depth_topic=camera_topics['depth'],
            info_topic=camera_topics['info'],
            lidar_topic=self.lidar_topic,
            enable_lidar=self.enable_lidar,
            sync_slop=self.sync_slop
        )

        if not self.sensor.connect(timeout=10.0):
            raise RuntimeError("传感器连接失败")

        # 坐标变换器
        self.transformer = CoordinateTransformer()
        self.transformer.load_all_extrinsics()

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

        # 可视化器 (可选)
        self.visualizer = None
        if self.enable_visualization:
            self.visualizer = PerceptionVisualizer(
                results_dir=self.results_dir,
                transformer=self.transformer,
                intrinsics=self.sensor.intrinsics,
            )
            self.log.info(f" 可视化已启用, 结果目录: {self.results_dir}")

    def _build_camera_topics(self, camera_name: str) -> dict:
        """根据相机名称构建 topic"""
        return {
            'color': f'/camera/{camera_name}/color/image_raw',
            'depth': f'/camera/{camera_name}/aligned_depth_to_color/image_raw',
            'info': f'/camera/{camera_name}/color/camera_info',
        }

    def _prompt_callback(self, msg: String):
        """Prompt topic 回调"""
        self._current_prompt = msg.data
        self.log.info(f" 更新 prompt: '{msg.data}'")

    def _config_callback(self, msg: PerceptionConfig):
        """配置 topic 回调 (PerceptPanel 用)"""
        if msg.prompt:
            self._current_prompt = msg.prompt
        if msg.min_score > 0:
            self._current_min_score = msg.min_score
        if msg.iou_threshold > 0:
            self._current_iou_threshold = msg.iou_threshold
        self.log.info(f" 更新配置: prompt='{self._current_prompt}', "
                      f"min_score={self._current_min_score:.2f}, iou={self._current_iou_threshold:.2f}")

    def _publish_status(self):
        """发布当前状态"""
        status = PerceptionStatus()
        status.prompt = self._current_prompt
        status.min_score = self._current_min_score
        status.iou_threshold = self._current_iou_threshold
        status.object_count = self._last_object_count
        status.categories = self._last_categories
        status.last_detect_time = self._last_detect_time
        self.pub_status.publish(status)

    def _get_prompt(self, req_prompt: str = None) -> str:
        """获取检测 prompt

        优先级: service 请求 > topic 订阅 > 默认配置
        """
        if self.prompt_source == 'service' and req_prompt:
            return req_prompt
        elif self.prompt_source == 'topic':
            return self._current_prompt
        else:
            return self.default_prompt

    def _get_distance_offset(self) -> float:
        """动态获取距离补偿参数（支持在线调整）

        Returns:
            float: 距离补偿值（米），实时从参数服务器读取
        """
        return rospy.get_param('~distance_offset', self.distance_offset_default)

    def _auto_detect_callback(self, event):
        """自动检测定时器回调"""
        try:
            # 使用当前 prompt 和配置的 LiDAR 设置
            result, timings = self._run_detection(
                prompt=self._current_prompt,
                enable_lidar=self.enable_lidar
            )

            # 发布到 topic
            if result:
                self.pub.publish(result)

            # 记录性能统计（throttle）
            self.log.debug_throttle(10.0, f"Detection time: {timings.get('total', 0):.3f}s")

        except Exception as e:
            self.log.warn_throttle(5.0, f" 自动检测失败: {e}")

    def handle_detect(self, req) -> DetectObjectsResponse:
        """处理检测请求"""
        response = DetectObjectsResponse()
        response.success = True

        try:
            result, timings = self._run_detection(
                prompt=req.prompt,
                enable_lidar=req.enable_lidar
            )

            if result:
                response.result = result
                # 发布到 topic
                self.pub.publish(result)

            # 记录性能统计
            self._log_timings(timings, req.enable_lidar)

        except TimeoutError as e:
            response.success = False
            response.error_message = f"数据采集超时: {e}"
            self.log.warn(f" {response.error_message}")
        except Exception as e:
            response.success = False
            response.error_message = f"检测失败: {e}"
            self.log.error(f" {response.error_message}")
            import traceback
            traceback.print_exc()

        return response

    def _log_timings(self, timings: dict, enable_lidar: bool):
        """记录性能统计"""
        total = timings.get('total', 0)
        self.log.info(f" ===== 性能统计 =====")
        self.log.info(f"  总耗时: {total*1000:.0f} ms")
        self.log.info(f"  - 数据同步: {timings.get('data_sync', 0)*1000:.0f} ms ({timings.get('data_sync', 0)/total*100:.1f}%)")

        # 显示并行执行信息
        if 'parallel_time' in timings:
            detection_t = timings.get('detection', 0)
            depth_opt_t = timings.get('depth_optimize', 0)
            parallel_t = timings.get('parallel_time', 0)
            saved_t = (detection_t + depth_opt_t) - parallel_t
            self.log.info(f"  - 并行执行 (DINO-X + CDM): {parallel_t*1000:.0f} ms ({parallel_t/total*100:.1f}%)")
            self.log.info(f"    * DINO-X检测: {detection_t*1000:.0f} ms")
            self.log.info(f"    * 深度优化: {depth_opt_t*1000:.0f} ms")
            self.log.info(f"    * 并行节省: {saved_t*1000:.0f} ms")
        else:
            self.log.info(f"  - DINO-X检测: {timings.get('detection', 0)*1000:.0f} ms ({timings.get('detection', 0)/total*100:.1f}%)")
            self.log.info(f"  - 深度优化: {timings.get('depth_optimize', 0)*1000:.0f} ms ({timings.get('depth_optimize', 0)/total*100:.1f}%)")

        self.log.info(f"  - 3D测量: {timings.get('measurement_total', 0)*1000:.0f} ms ({timings.get('measurement_total', 0)/total*100:.1f}%)")
        if enable_lidar:
            self.log.info(f"    * 相机: {timings.get('measurement_camera', 0)*1000:.0f} ms")
            self.log.info(f"    * LiDAR: {timings.get('measurement_lidar', 0)*1000:.0f} ms")
        if 'visualization' in timings:
            self.log.info(f"  - 可视化: {timings.get('visualization', 0)*1000:.0f} ms ({timings.get('visualization', 0)/total*100:.1f}%)")

    def _run_detection(self, prompt: str, enable_lidar: bool) -> tuple:
        """执行检测流程，返回 (Object3DArray, 性能统计字典)"""
        import time

        result = Object3DArray()
        result.header.stamp = rospy.Time.now()
        result.frame_id = self.target_frame

        # 性能统计
        timings = {}
        total_start = time.time()

        # 1. 采集同步数据 (带新鲜度检查)
        t0 = time.time()
        data = self.sensor.get_synced_data(
            timeout=2.0,
            max_age=self.data_max_age
        )
        timings['data_sync'] = time.time() - t0
        self.log.debug(f"Data sync OK, age={data['age']:.3f}s")

        # 2. 获取 prompt
        final_prompt = self._get_prompt(prompt)

        # 3. 并行调用 DINO-X 检测 和 CDM 深度优化
        t_parallel_start = time.time()

        # 创建包装函数来记录各自的执行时间
        def _timed_detection():
            t0 = time.time()
            result = self._detect_objects(final_prompt, data['rgb'])
            return result, time.time() - t0

        def _timed_depth_optimize():
            t0 = time.time()
            result = self.core.optimize_depth(data['rgb'], data['depth'])
            return result, time.time() - t0

        with ThreadPoolExecutor(max_workers=2) as executor:
            # 同时提交两个任务
            future_detection = executor.submit(_timed_detection)
            future_depth = executor.submit(_timed_depth_optimize)

            # 等待两个任务完成
            detections, detection_time = future_detection.result()
            depth, depth_optimize_time = future_depth.result()

            # 记录各自的执行时间
            timings['detection'] = detection_time
            timings['depth_optimize'] = depth_optimize_time

        # 记录并行执行的总时间（实际墙上时间，应该约等于 max(detection, depth_optimize)）
        timings['parallel_time'] = time.time() - t_parallel_start

        # 发布优化后的深度图 (for RViz visualization)
        if self.pub_optimized_depth.get_num_connections() > 0:
            try:
                depth_mm = (depth * 1000).astype(np.uint16)  # m -> mm
                depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding='16UC1')
                depth_msg.header.stamp = data['timestamp']
                depth_msg.header.frame_id = f"{self.camera_name}_camera_optical_frame"
                self.pub_optimized_depth.publish(depth_msg)
            except Exception as e:
                self.log.warn_throttle(10.0, f" 发布优化深度图失败: {e}")

        if not detections:
            self.log.info(f"DETECT: No objects found for '{final_prompt}'")
            timings['total'] = time.time() - total_start
            return result, timings

        self.log.info(f"DETECT: Found {len(detections)} objects")

        # 5. 对每个物体进行 3D 测量（并行化优化）
        lidar_points = data.get('lidar_points') if enable_lidar else None

        t0 = time.time()

        # 5.1 并行执行相机 3D 测量（不做坐标变换）
        t_camera_start = time.time()

        def _measure_camera(det):
            """相机测量（optical 坐标系）"""
            camera_result = self.core.camera_3d_percept(depth, det['mask'], skip_transform=True)
            return det, camera_result

        # 并行测量所有物体
        with ThreadPoolExecutor(max_workers=4) as executor:
            camera_results = list(executor.map(_measure_camera, detections))

        # 过滤无效测量，收集 optical 坐标
        valid_items = []
        optical_centroids = []

        for det, camera_result in camera_results:
            if camera_result.valid:
                valid_items.append((det, camera_result))
                optical_centroids.append(camera_result.centroid_optical)

        timings['measurement_camera'] = time.time() - t_camera_start

        if not valid_items:
            self.log.warn("DETECT: All camera measurements failed")
            timings['measurement_total'] = time.time() - t0
            timings['total'] = time.time() - total_start
            return result, timings

        # 5.2 批量坐标变换（optical -> target_frame）
        t_transform_start = time.time()
        optical_points = np.array(optical_centroids)
        target_centroids = self.transformer.optical_to_target(optical_points, self.target_frame)
        timings['coordinate_transform'] = time.time() - t_transform_start

        # 5.3 构建 Object3D 消息
        valid_detections = []
        object_results = []

        for i, (det, camera_result) in enumerate(valid_items):
            obj = Object3D()
            obj.object_id = det['object_id']
            obj.category = det['category']
            obj.score = det['score']
            obj.bbox = det['bbox']

            # 使用批量变换后的坐标
            target_centroid = target_centroids[i]

            # 应用距离补偿（沿视线方向调整位置，动态读取参数支持在线调整）
            distance_offset = self._get_distance_offset()
            original_distance = np.linalg.norm(target_centroid)
            if original_distance > 0.01:  # 避免除以零
                # 沿视线方向调整：new_position = position * (1 + offset/distance)
                scale_factor = (original_distance + distance_offset) / original_distance
                compensated_centroid = target_centroid * scale_factor
            else:
                compensated_centroid = target_centroid

            obj.position = Point(
                x=compensated_centroid[0],
                y=compensated_centroid[1],
                z=compensated_centroid[2]
            )
            obj.distance = np.linalg.norm(compensated_centroid)
            obj.confidence = camera_result.confidence

            # 保存相机光学坐标系下的原始位置 (用于调试)
            optical_centroid = camera_result.centroid_optical
            if optical_centroid is not None:
                obj.position_optical = Point(
                    x=optical_centroid[0],
                    y=optical_centroid[1],
                    z=optical_centroid[2]
                )
                obj.depth = optical_centroid[2]  # z 值即为深度
            obj.source_camera = self.camera_name

            # 测量结果（用于可视化）
            measure_result = {
                'camera': {'distance': obj.distance}
            }

            # LiDAR 3D 感知 (可选，串行处理)
            lidar_result = None  # 初始化为 None
            if lidar_points is not None and len(lidar_points) > 0:
                camera_depth = camera_result.stats.get('depth_median', None)
                lidar_result = self.core.lidar_3d_percept(
                    lidar_points, det['bbox'], camera_depth
                )

                if lidar_result.valid:
                    obj.position_lidar = Point(
                        x=lidar_result.centroid[0],
                        y=lidar_result.centroid[1],
                        z=lidar_result.centroid[2]
                    )
                    obj.distance_lidar = lidar_result.distance
                    obj.confidence_lidar = lidar_result.confidence
                    measure_result['lidar'] = {'distance': lidar_result.distance}

            # 精度日志 (如果启用)
            if self.enable_accuracy_log:
                # 临时更新 camera_result 的 centroid 为变换后的坐标
                camera_result.centroid = target_centroid
                camera_result.distance = obj.distance
                # 直接传递 lidar_result 对象而不是字典
                self._log_accuracy(obj, camera_result, lidar_result)

            result.objects.append(obj)
            valid_detections.append(det)
            object_results.append(measure_result)

        timings['measurement_total'] = time.time() - t0
        self.log.info(f"DETECT: Measured {len(result.objects)} objects successfully")

        # 6. 可视化 (可选)
        if self.visualizer and len(valid_detections) > 0:
            try:
                t0 = time.time()
                vis_path = self.visualizer.save_overview(
                    rgb=data['rgb'],
                    depth=depth,
                    detections=valid_detections,
                    object_results=object_results,
                    lidar_points=lidar_points,
                    prompt=final_prompt
                )
                timings['visualization'] = time.time() - t0
                self.log.info(f" 可视化保存: {vis_path}")
            except Exception as e:
                import traceback
                self.log.warn(f" 可视化保存失败: {e}")
                traceback.print_exc()

        timings['total'] = time.time() - total_start

        # 输出简洁的 TIMING 日志
        timing_str = ' | '.join([f"{k}:{int(v*1000)}" for k, v in timings.items()])
        self.log.info(f"TIMING: {timing_str}")

        # 更新状态并发布
        self._last_detect_time = timings['total']
        self._last_object_count = len(result.objects) if result else 0
        self._last_categories = list(set(obj.category for obj in result.objects)) if result else []
        self._publish_status()

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

        # 批量解码 masks (优化: 一次性解码所有 masks)
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
            # pycocotools 批量解码: 返回 (N, H, W) 数组
            masks_array = coco_mask.decode(rle_masks)
            if masks_array.ndim == 2:  # 单个 mask 的情况
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

            # 检查 mask 面积
            if binary_mask.sum() < 300:
                continue

            category = obj.get('category', prompt)
            score = obj.get('score', 0)

            # 分配 object_id
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

    def _process_detection(self, det: dict, depth: np.ndarray,
                           lidar_points: np.ndarray) -> tuple:
        """处理单个检测结果，返回 (Object3D, 测量结果字典)"""
        obj = Object3D()
        obj.object_id = det['object_id']
        obj.category = det['category']
        obj.score = det['score']
        obj.bbox = det['bbox']

        # 测量结果字典 (用于可视化)
        measure_result = {'camera': None, 'lidar': None}

        # 相机 3D 感知
        camera_result = self.core.camera_3d_percept(depth, det['mask'])
        if not camera_result.valid:
            self.log.warn(f" {det['object_id']} 相机测量失败: {camera_result.error_msg}")
            return None, measure_result

        # 应用距离补偿（沿视线方向调整位置，动态读取参数支持在线调整）
        distance_offset = self._get_distance_offset()
        original_distance = camera_result.distance
        if original_distance > 0.01:  # 避免除以零
            # 沿视线方向调整：new_position = position * (1 + offset/distance)
            scale_factor = (original_distance + distance_offset) / original_distance
            compensated_centroid = camera_result.centroid * scale_factor
        else:
            compensated_centroid = camera_result.centroid

        obj.position = Point(
            x=compensated_centroid[0],
            y=compensated_centroid[1],
            z=compensated_centroid[2]
        )
        obj.distance = np.linalg.norm(compensated_centroid)
        obj.confidence = camera_result.confidence
        measure_result['camera'] = {'distance': obj.distance}

        # LiDAR 3D 感知 (可选)
        lidar_result = None
        if lidar_points is not None and len(lidar_points) > 0:
            camera_depth = camera_result.stats.get('depth_median', None)
            lidar_result = self.core.lidar_3d_percept(
                lidar_points, det['bbox'], camera_depth
            )

            if lidar_result.valid:
                obj.position_lidar = Point(
                    x=lidar_result.centroid[0],
                    y=lidar_result.centroid[1],
                    z=lidar_result.centroid[2]
                )
                obj.distance_lidar = lidar_result.distance
                obj.confidence_lidar = lidar_result.confidence
                measure_result['lidar'] = {'distance': lidar_result.distance}

        # 精度日志
        if self.enable_accuracy_log:
            self._log_accuracy(obj, camera_result, lidar_result)

        return obj, measure_result

    def _log_accuracy(self, obj: Object3D, camera_result: MeasurementResult,
                      lidar_result: MeasurementResult = None):
        """打印精度日志"""
        self.log.info(f"=== 检测结果: {obj.object_id} ===")
        self.log.info(f"  坐标系: {self.target_frame}")
        self.log.info(f"  相机测量:")
        self.log.info(f"    位置: ({obj.position.x:.3f}, {obj.position.y:.3f}, {obj.position.z:.3f}) m")
        self.log.info(f"    距离: {obj.distance:.3f} m")
        self.log.info(f"    置信度: {obj.confidence:.2f}")
        self.log.info(f"    深度点数: {camera_result.stats.get('num_points', 'N/A')}")

        if lidar_result and lidar_result.valid:
            self.log.info(f"  LiDAR测量:")
            self.log.info(f"    位置: ({obj.position_lidar.x:.3f}, {obj.position_lidar.y:.3f}, {obj.position_lidar.z:.3f}) m")
            self.log.info(f"    距离: {obj.distance_lidar:.3f} m")
            self.log.info(f"    置信度: {obj.confidence_lidar:.2f}")
            self.log.info(f"    LiDAR点数: {lidar_result.stats.get('num_points', 'N/A')}")

            # 误差分析
            err_dist = (obj.distance - obj.distance_lidar) * 1000
            self.log.info(f"  误差分析:")
            self.log.info(f"    距离差: {err_dist:+.1f} mm ({err_dist/obj.distance_lidar*100:+.2f}%)")

    def run(self):
        """运行节点"""
        rospy.spin()

    def shutdown(self):
        """关闭节点"""
        self.log.info(" 节点关闭")
        if hasattr(self, 'sensor'):
            self.sensor.disconnect()


def main():
    try:
        node = ScenePerception3DNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[Percept3D] Startup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
