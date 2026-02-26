# Perception Grasp 设计方案

## 1. 概述

### 1.1 目标
基于手部相机实现**纯感知**功能：
- 使用 `GraspAnythingOnline` 检测抓取位姿
- 使用 `DinoXDetectorOnline` 检测目标物体
- 使用 `DepthOptimizerOnline` (CDM) 深度图去噪优化
- **三个服务并行执行**，提高效率
- 通过 bbox 内检测 + 置信度排序 关联抓取位姿与目标物体
- **输出相机光学坐标系下的 3D 数据** (Z向前, X右, Y下)
- 提供文件保存和 ROS Topic 两种可视化输出

### 1.2 设计原则
```
┌─────────────────────────────────────────────────────────────┐
│                      职责分离                                │
├─────────────────────────────────────────────────────────────┤
│  perception_grasp (本模块)     │  机械臂控制 (demo.py)       │
│  ─────────────────────────────│─────────────────────────────│
│  • RGB-D 图像采集              │  • 获取机械臂末端位姿        │
│  • GraspAnything 检测          │  • 相机→基座坐标变换        │
│  • DinoX 物体检测              │  • 工作范围检查             │
│  • 过滤关联                    │  • 运动规划与执行           │
│  • 2D→3D (相机坐标系)          │                            │
│                               │                            │
│  输出: point3d (相机系)        │  输入: point3d (相机系)     │
│        width3d                │  输出: offset (基座系)      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 依赖
- 手部相机 Topic (由 camera_driver 提供):
  - RGB: `/camera/hand/color/image_raw`
  - Depth: `/camera/hand/aligned_depth_to_color/image_raw`
  - CameraInfo: `/camera/hand/color/camera_info`
- 分辨率: 1280x720
- 检测服务:
  - GraspAnything: http://192.168.112.14:12086
  - DinoX: http://192.168.112.14:10086
  - CDM (深度优化): http://192.168.112.14:8086
- 相机内参: 从 `camera_info` Topic 自动获取

### 1.4 参考实现
- `arm_robot/src/demo.py:detect()` (438-642行) - 完整检测流程
- `arm_robot/src/camera.py:unprj()` (332-374行) - 2D→3D反投影
- `arm_robot/src/percept.py:DepthOptimizerOnline` - CDM深度优化服务

---

## 2. 架构设计

### 2.1 系统架构 (Topic 订阅模式)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              camera_driver                                       │
│   ┌─────────────────────┐                                                       │
│   │    Hand Camera      │                                                       │
│   │   1280x720 RGB+D    │                                                       │
│   └──────────┬──────────┘                                                       │
│              │                                                                  │
│   ┌──────────┼──────────────────────────────────┐                               │
│   │          │                                  │                               │
│   ▼          ▼                                  ▼                               │
│ ~hand/color  ~hand/aligned_depth    ~hand/color/camera_info                     │
│ /image_raw   _to_color/image_raw                                                │
└─────┬────────────────┬──────────────────────────┬───────────────────────────────┘
      │                │                          │
      │    订阅 Topic  │                          │
      ▼                ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        perception_grasp_node (服务端)                            │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  图像缓存 (最新一帧)          内参缓存 (从 camera_info 解析)              │   │
│   │  self.rgb, self.depth        self.intrinsics                            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                      │                                                          │
│   ┌──────────────────▼──────────────────────────────────────────────────────┐   │
│   │              ThreadPoolExecutor (max_workers=3)                         │   │
│   │                        三服务完全并行                                     │   │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │   │
│   │  │ GraspAnything   │  │     DinoX       │  │      CDM        │         │   │
│   │  │  forward(rgb)   │  │  forward(prompt,│  │ forward(rgb,    │         │   │
│   │  │  内部含postproc │  │         rgb)    │  │    depth_mm)    │         │   │
│   │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │   │
│   │           │ affs               │ targets            │ depth_optimized   │   │
│   └───────────┼────────────────────┼────────────────────┼───────────────────┘   │
│               │                    │                    │                       │
│               └────────────────────┼────────────────────┘                       │
│                                    │                                            │
│               ┌────────────────────▼────────────────────┐                       │
│               │  _choose() 过滤关联 (affs + targets)     │                       │
│               └────────────────────┬────────────────────┘                       │
│                                    ▼                                            │
│               ┌─────────────────────────────────────────┐                       │
│               │  _unprj() 2D→3D (使用 CDM 优化后深度)    │                       │
│               └────────────────────┬────────────────────┘                       │
│                                    ▼                                            │
│               ┌─────────────────────┐     ┌─────────────────────┐               │
│               │    Visualization    │─────►│  ~vis_rgb (Topic)   │               │
│               │                     │     │  ~vis_depth         │               │
│               └──────────┬──────────┘     │  ~vis_combined      │               │
│                          │                └─────────────────────┘               │
│                          │                ┌─────────────────────┐               │
│                          └────────────────►│  File Output        │               │
│                                           └─────────────────────┘               │
└─────────────────────────────────────────────┬───────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  ~detect (Service)   │      │  /usr/prompt/grasp   │      │  ~result (Topic)     │
│  同步返回结果         │      │  触发检测            │      │  实时模式输出         │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

### 2.2 数据流时序 (三服务完全并行)

```
User              PerceptionGrasp      CDM          GraspAnything      DinoX
  │                     │               │                │               │
  │ call detect(prompt) │               │                │               │
  │────────────────────►│               │                │               │
  │                     │               │                │               │
  │                     │ ══════════════╪════════════════╪═══════════════╪══╗
  │                     │ ║          三服务完全并行                        ║
  │                     │ ╠═════════════╪════════════════╪═══════════════╪══╣
  │                     │ ║             │                │               │ ║
  │                     │──forward(rgb)─────────────────────────────────►│ ║
  │                     │ ║             │      (内部含 post_process)      │ ║
  │                     │ ║             │                │               │ ║
  │                     │──forward(prompt, rgb)─────────────────────────►│ ║
  │                     │ ║             │                │               │ ║
  │                     │──forward(rgb, depth_mm)──────►│               │ ║
  │                     │ ║             │                │               │ ║
  │                     │ ╚═════════════╪════════════════╪═══════════════╪══╝
  │                     │               │                │               │
  │                     │◄──depth_optimized──│           │               │
  │                     │               │                │               │
  │                     │◄───────────────────────────────────────────affs│
  │                     │               │                │               │
  │                     │◄──────────────────────────────────────targets──│
  │                     │               │                │               │
  │                     │ _choose (关联 affs + targets)  │               │
  │                     │─────────┐     │                │               │
  │                     │◄────────┘     │                │               │
  │                     │               │                │               │
  │                     │ _unprj (用 CDM 优化后深度)      │               │
  │                     │─────────┐     │                │               │
  │                     │◄────────┘     │                │               │
  │                     │               │                │               │
  │                     │ visualize (file + topic)       │               │
  │                     │─────────┐     │                │               │
  │                     │◄────────┘     │                │               │
  │                     │               │                │               │
  │◄────────────────────│ GraspDetectionResult           │               │
  │                     │               │                │               │
```

**时序说明**:
- **三服务完全并行**: GraspAnything + CDM + DinoX 同时执行，无依赖
- **GraspAnything**: `forward(rgb)` 内部已包含 `post_process`，不依赖深度
- **CDM 深度优化**: 仅用于后续 `_unprj()` 计算 3D 坐标

**GraspAnything 调用方式**:
```python
# forward 返回元组 (ret, padded_img)
# ret 格式: [[obj1, obj2, ...]] - 双层列表 (batch_size=1)
ret, padded_img = grasp_detector.forward(rgb)
affs = ret  # affs[0] 是物体列表
```

**延迟估算**:
```
并行阶段: max(GraspAnything ~200ms, CDM ~200ms, DinoX ~200ms) ≈ 200ms
后处理: _choose + _unprj ~10ms
Total: ~210ms
```

---

## 3. 核心算法

### 3.0 初始化与预热

```python
import threading
import os
import time
import math
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

# 简单配置类 (用于初始化检测服务)
class SimpleConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class PerceptionGrasp:
    def __init__(self, config):
        """
        Args:
            config: 从 perception_grasp.yaml 加载的配置字典
        """
        self.config = config

        # === 并发控制 ===
        self._detect_lock = threading.Lock()
        self._detecting = False

        # === 图像缓存 ===
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.bridge = CvBridge()
        self._data_lock = threading.Lock()

        # === 订阅 CameraInfo (一次性获取内参) ===
        rospy.Subscriber(config['camera']['camera_info_topic'],
                         CameraInfo, self._camera_info_callback, queue_size=1)

        # === 使用 message_filters 同步订阅 RGB + Depth ===
        self._rgb_sub = message_filters.Subscriber(
            config['camera']['rgb_topic'], Image)
        self._depth_sub = message_filters.Subscriber(
            config['camera']['depth_topic'], Image)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=5,
            slop=0.1  # 时间容差 100ms
        )
        self._sync.registerCallback(self._sync_callback)

        # === 初始化检测服务 (使用 cfg 对象) ===
        # 1. GraspAnythingOnline - 需要 server_list 文件和 model_name
        #    server_list 路径相对于当前工作目录，使用 rospkg 获取包路径
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('perception')
        server_list_path = os.path.join(pkg_path,
            config['services']['grasp'].get('server_list', 'config/server_grasp.json'))

        grasp_cfg = SimpleConfig(
            server_list=server_list_path,
            model_name=config['services']['grasp'].get('model_name', 'full'),
            resize=(1280, 720),
            warmup=0  # 手动预热
        )
        self.grasp_detector = GraspAnythingOnline(grasp_cfg)

        # 2. DinoXDetectorOnline
        dinox_cfg = SimpleConfig(
            url=config['services']['dinox']['url'],
            min_score=config['filter'].get('min_object_score', 0.25)
        )
        self.object_detector = DinoXDetectorOnline(dinox_cfg)

        # 3. DepthOptimizerOnline (CDM)
        cdm_cfg = SimpleConfig(
            url=config['services']['cdm']['url'],
            chosen_policy=config['services']['cdm'].get('chosen_policy', 'dn')
        )
        self.depth_optimizer = DepthOptimizerOnline(cdm_cfg)

        # === 实时模式: 订阅 prompt Topic ===
        if config['trigger']['realtime_mode'].get('enabled', False):
            self._prompt_sub = rospy.Subscriber(
                config['trigger']['realtime_mode']['prompt_topic'],
                String, self._prompt_callback, queue_size=1
            )
            self._result_pub = rospy.Publisher(
                config['trigger']['realtime_mode']['result_topic'],
                GraspResult, queue_size=1
            )

        # === 可视化 Topic 发布器 ===
        self._vis_rgb_pub = rospy.Publisher('~vis_rgb', Image, queue_size=1)
        self._vis_depth_pub = rospy.Publisher('~vis_depth', Image, queue_size=1)
        self._vis_combined_pub = rospy.Publisher('~vis_combined', Image, queue_size=1)

        # === 预热 GraspAnything ===
        self._warmup()

    def _prompt_callback(self, msg):
        """实时模式: 收到 prompt 后触发检测"""
        prompt = msg.data
        enable_cdm = self.config['services']['cdm'].get('enabled', True)
        result = self.detect(prompt, enable_cdm=enable_cdm)

        # 发布结果到 ~result Topic
        result_msg = GraspResult()
        result_msg.header.stamp = rospy.Time.now()
        result_msg.success = result['success']
        if result['success']:
            result_msg.point3d = result['point3d']
            result_msg.width3d = result['width3d']
            result_msg.angle = result['angle']
            result_msg.category = result['category']
            result_msg.score = result.get('score', 0.0)  # 使用 get 避免 KeyError
            result_msg.center_uv = result.get('center_uv', [0, 0])
            result_msg.depth = result.get('depth_value', 0.0)
        else:
            result_msg.error_message = result.get('error_message', 'Detection failed')
        result_msg.detection_time_ms = result.get('detection_time_ms', 0)
        self._result_pub.publish(result_msg)

    def _sync_callback(self, rgb_msg, depth_msg):
        """同步回调 - RGB 和 Depth 时间戳对齐

        参考 synced_sensor_subscriber.py:160-179
        - camera_driver 发布 RGB8 编码
        - 转换为 BGR 存储 (所有检测服务实际都使用 BGR)
        """
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
            self.rgb = rgb  # BGR 格式，所有服务统一使用
            self.depth = depth

    def _camera_info_callback(self, msg):
        """从 camera_info 解析内参"""
        if self.intrinsics is None:
            self.intrinsics = {
                'width': msg.width,
                'height': msg.height,
                'fx': msg.K[0],
                'fy': msg.K[4],
                'cx': msg.K[2],  # ppx
                'cy': msg.K[5],  # ppy
            }
            rospy.loginfo(f"[PerceptionGrasp] 相机内参已加载: {msg.width}x{msg.height}")

    def _warmup(self):
        """预热 GraspAnything 服务，减少首次调用延迟"""
        rospy.loginfo("[PerceptionGrasp] 预热 GraspAnything...")
        try:
            # 使用小图预热
            dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            self.grasp_detector.forward(dummy_rgb)
            rospy.loginfo("[PerceptionGrasp] 预热完成")
        except Exception as e:
            rospy.logwarn(f"[PerceptionGrasp] 预热失败: {e}")
```

### 3.1 detect() 主流程 (三服务完全并行)

```python
def detect(self, prompt: str, enable_cdm: bool = True) -> dict:
    """
    三服务完全并行调用:
    - GraspAnything: forward(rgb) 内部已包含 post_process，不依赖深度
    - DinoX: forward(prompt, rgb) 检测目标物体
    - CDM: forward(rgb, depth_mm) 深度优化，仅用于后续 _unprj()

    Args:
        prompt: 检测提示词 (如 "bottle.cup")
        enable_cdm: 是否启用 CDM 深度优化

    Returns:
        dict: 检测结果
    """
    # === 并发控制: 防止重复检测 ===
    if not self._detect_lock.acquire(blocking=False):
        rospy.logwarn("[DETECT] 检测进行中，忽略新请求")
        return {'success': False, 'error_message': 'Detection in progress'}

    try:
        self._detecting = True
        start_time = time.time()

        # === 获取当前帧 (加锁保护) ===
        with self._data_lock:
            rgb = self.rgb.copy() if self.rgb is not None else None
            depth = self.depth.copy() if self.depth is not None else None

        if rgb is None or depth is None:
            return {'success': False, 'error_message': 'No image available'}
        if self.intrinsics is None:
            return {'success': False, 'error_message': 'Camera intrinsics not ready'}

        img_h, img_w = rgb.shape[:2]
        depth_mm = (depth * 1000).astype(np.uint16)  # CDM 需要 uint16 mm

        # ============================================================
        # 三服务完全并行执行
        # ============================================================
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            # 1. GraspAnything forward(rgb) - 内部已包含 post_process
            #    - 使用 partial 避免 lambda 闭包捕获问题
            futures['grasp'] = executor.submit(
                partial(self.grasp_detector.forward, rgb)
            )

            # 2. DinoX (prompt, rgb)
            futures['dinox'] = executor.submit(
                self.object_detector.forward, prompt, rgb
            )

            # 3. CDM 深度优化 (仅用于后续 _unprj)
            if enable_cdm:
                futures['cdm'] = executor.submit(
                    self.depth_optimizer.forward, rgb, depth_mm, 'dn'
                )

            # 收集结果
            results = {}
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=10)
                except Exception as e:
                    rospy.logerr(f"[{name}] 调用失败: {e}")
                    results[name] = None

        # === 处理 CDM 结果 (仅用于 _unprj 计算 3D 坐标) ===
        if enable_cdm and results.get('cdm') and results['cdm'].get('success'):
            depth_optimized = results['cdm']['depth'].astype(np.float32) / 1000.0
            rospy.loginfo("[CDM] 使用优化后深度图")
        else:
            depth_optimized = depth
            if enable_cdm:
                rospy.logwarn("[CDM] 深度优化失败，使用原始深度")

        # === 处理 GraspAnything 结果 ===
        # forward() 返回 (ret, padded_img) 元组
        grasp_result = results.get('grasp')
        if grasp_result is not None:
            affs, _ = grasp_result  # 解包元组，丢弃 padded_img
        else:
            affs = [[]]  # 空结果: [[]] 保持双层列表结构

        # === 解析 DinoX 结果 ===
        dinox_result = results.get('dinox')
        if dinox_result and hasattr(dinox_result, 'result'):
            targets = dinox_result.result.get('objects', [])
        else:
            targets = []

        fts = {'targets': targets}

        # === choose: 关联、过滤、排序选择 ===
        chosen_aff = self._choose(rgb, depth_optimized, affs, fts)

        # === unprj: 2D→3D 反投影 ===
        point3d = None
        width3d = None
        angle = None
        category = None
        center_uv = None
        depth_value = None
        score = None

        if chosen_aff is not None:
            aff = chosen_aff['aff']
            cx, cy = aff[0], aff[1]
            if cx >= img_w or cy >= img_h:
                rospy.logwarn(f'[DETECT] 抓取点超出图像范围: ({cx}, {cy})')
                chosen_aff = None
            else:
                data3d = self._unprj(chosen_aff, depth_optimized)
                if data3d['status']:
                    point3d = data3d['point3d']
                    width3d = data3d['width3d']
                    angle = aff[4] * 180 / math.pi  # rad → deg
                    category = chosen_aff.get('category', 'unknown')
                    score = chosen_aff.get('score', 0.0)
                    center_uv = [cx, cy]
                    depth_value = depth_optimized[int(cy), int(cx)]
                else:
                    rospy.logwarn('[DETECT] 深度值无效')
                    chosen_aff = None

        # === 构建返回结果 ===
        detection_time_ms = (time.time() - start_time) * 1000

        if chosen_aff is not None:
            result = {
                'success': True,
                'chosen_aff': chosen_aff,
                'point3d': point3d,
                'width3d': width3d,
                'angle': angle,
                'category': category,
                'score': score,
                'center_uv': center_uv,
                'depth_value': depth_value,
                'affs': affs,
                'targets': targets,
                'fts': fts,
                'depth_optimized': depth_optimized,
                'detection_time_ms': detection_time_ms
            }
        else:
            result = {
                'success': False,
                'chosen_aff': None,
                'point3d': None,
                'width3d': None,
                'angle': None,
                'category': None,
                'score': None,
                'center_uv': None,
                'depth_value': None,
                'affs': affs,
                'targets': targets,
                'fts': fts,
                'depth_optimized': depth_optimized,
                'detection_time_ms': detection_time_ms
            }

        # === Step 7: 生成可视化 ===
        state = {
            'affs': affs,
            'fts': fts,
            'chosen_aff': chosen_aff,
            'point3d': point3d,
            'width3d': width3d
        }
        vis_rgb, vis_depth, vis_combined = self._visualize(rgb.copy(), depth_optimized, state)
        result['vis_rgb'] = vis_rgb
        result['vis_depth'] = vis_depth
        result['vis_combined'] = vis_combined

        # === Step 8: 发布可视化 Topic ===
        if self.config['visualization'].get('publish_to_topic', True):
            self._vis_rgb_pub.publish(self.bridge.cv2_to_imgmsg(vis_rgb, 'bgr8'))
            self._vis_depth_pub.publish(self.bridge.cv2_to_imgmsg(vis_depth, 'bgr8'))
            self._vis_combined_pub.publish(self.bridge.cv2_to_imgmsg(vis_combined, 'bgr8'))

        # === Step 9: 保存文件 ===
        if self.config['visualization'].get('save_to_file', False):
            self._save_result(rgb, depth_optimized, result)

        return result

    finally:
        # === 确保释放锁 ===
        self._detecting = False
        self._detect_lock.release()


def _choose(self, rgb, depth, affs, fts):
    """
    选择最优抓取点 - 完全参考 demo.py:choose() (324-429行)

    流程:
    1. 检查 targets 是否存在 (必须匹配 prompt)
    2. 遍历所有检测到的物体的抓取点
    3. 过滤超出图像范围的点
    4. 只保留中心点落在 targets bbox 内的抓取
    5. 每个物体选择 score 最高的抓取点
    6. 所有物体中选择 score 最高的作为最终结果

    Args:
        rgb: BGR图像
        depth: 深度图
        affs: GraspAnything 检测结果
        fts: 过滤条件，包含 'targets' (DinoX bbox列表)

    Returns:
        dict: {aff, score, touching_points, category} 或 None
    """
    result_bbox = []
    score_li = []
    tps_li = []
    category_li = []
    img_h, img_w = rgb.shape[:2]

    # === 必须有 targets (DinoX 结果) 才能匹配 ===
    if 'targets' not in fts or not fts['targets']:
        rospy.logwarn("[CHOOSE] 无 DinoX targets，prompt 未匹配到物体")
        return None

    # 检查 affs 格式
    if not affs or len(affs) == 0 or not affs[0]:
        return None

    # 遍历所有检测到的物体
    for obj in affs[0]:
        if not isinstance(obj, dict):
            continue

        scores_obj = obj.get('scores', [])
        affs_obj = obj.get('affs', [])
        tps = obj.get('touching_points', [])

        if not affs_obj or not scores_obj:
            continue

        # === 过滤超出图像范围的抓取点 ===
        tmp1, tmp2, tmp3 = [], [], []
        for i in range(len(affs_obj)):
            if affs_obj[i][0] > img_w or affs_obj[i][1] > img_h:
                continue
            tmp1.append(affs_obj[i])
            tmp2.append(scores_obj[i])
            if i < len(tps):
                tmp3.append(tps[i])

        if len(tmp1) < 1:
            continue

        affs_obj = tmp1
        scores_obj = tmp2
        tps = tmp3

        # === 根据 targets 过滤 (抓取中心点必须在 bbox 内) ===
        # 注意: targets 已在函数开头检查，此处必定存在
        chosen_category = 'unknown'
        targets = fts['targets']
        tmp1, tmp2, tmp3, tmp_categories = [], [], [], []

        for i in range(len(affs_obj)):
            bbox_center = affs_obj[i]  # [cx, cy, w, h, angle]
            cx, cy = bbox_center[0], bbox_center[1]

            for target in targets:
                target_bbox = target.get('bbox', [])
                target_category = target.get('category', 'unknown')

                if len(target_bbox) >= 4:
                    x1, y1, x2, y2 = target_bbox[:4]
                    # 检查抓取中心是否在 target bbox 内
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        tmp1.append(affs_obj[i])
                        tmp2.append(scores_obj[i])
                        if i < len(tps):
                            tmp3.append(tps[i])
                        tmp_categories.append(target_category)
                        break

        if len(tmp1) > 0:
            affs_obj = tmp1
            scores_obj = tmp2
            tps = tmp3
            # 选最高分对应的类别
            k = np.argmax(scores_obj)
            chosen_category = tmp_categories[k] if tmp_categories else 'unknown'
        else:
            continue  # 该物体无有效抓取点

        # === 该物体选择 score 最高的抓取点 ===
        k = np.argmax(scores_obj)
        result_bbox.append(affs_obj[k])
        score_li.append(scores_obj[k])
        if k < len(tps):
            tps_li.append(tps[k])
        else:
            tps_li.append([])
        category_li.append(chosen_category)

    # === 所有物体中选择 score 最高的 ===
    if len(score_li) == 0:
        return None

    max_index, max_score = max(enumerate(score_li), key=lambda x: x[1])

    return {
        'aff': result_bbox[max_index],
        'score': max_score,
        'touching_points': tps_li[max_index],
        'category': category_li[max_index]
    }


def _deproject_pixel_to_point(self, pixel, depth):
    """
    纯 Python 实现 2D→3D 反投影 (替代 pyrealsense2)

    Args:
        pixel: [u, v] 像素坐标
        depth: 深度值 (米)

    Returns:
        [x, y, z] 相机坐标系 3D 点
    """
    fx = self.intrinsics['fx']
    fy = self.intrinsics['fy']
    cx = self.intrinsics['cx']
    cy = self.intrinsics['cy']

    u, v = pixel
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return [x, y, z]


def _unprj(self, chosen_aff, depth):
    """
    2D→3D 反投影 - 参考 camera.py:unprj()

    Args:
        chosen_aff: 选中的抓取信息 {aff, touching_points, ...}
        depth: 深度图 (H, W), 单位: 米

    Returns:
        dict: {status, point3d, width3d}
    """
    aff5 = chosen_aff['aff']  # [cx, cy, w, h, angle]
    tps = chosen_aff.get('touching_points', [])
    cx, cy = aff5[0], aff5[1]

    height, width = depth.shape

    # 边界检查
    if cx >= width or cy >= height:
        return {'status': False, 'point3d': [0, 0, 0], 'width3d': 0}

    # 获取中心点深度
    center_depth = depth[int(cy), int(cx)]

    # 深度有效性检查
    if center_depth < 0.001 or center_depth > 2.0:
        rospy.logwarn(f'[UNPRJ] 深度值无效: {center_depth:.3f}m at ({cx:.1f}, {cy:.1f})')
        return {'status': False, 'point3d': [0, 0, 0], 'width3d': 0}

    # 像素→相机坐标系 3D 点 (纯 Python 实现)
    point3d = self._deproject_pixel_to_point([cx, cy], center_depth)

    # 计算夹爪宽度 (两个 touching_points 的 3D 距离)
    width3d = 0
    if tps and len(tps) >= 2:
        p1 = self._deproject_pixel_to_point(tps[0], center_depth)
        p2 = self._deproject_pixel_to_point(tps[1], center_depth)
        width3d = np.linalg.norm(np.array(p1) - np.array(p2))

    return {
        'status': True,
        'point3d': point3d,
        'width3d': width3d
    }
```

---

## 4. 可视化 (_visualize)

```python
def _visualize(self, rgb, depth_optimized, state):
    """
    可视化检测结果，参考 demo.py:vis_frame

    Args:
        rgb: BGR 图像 (会被修改，输入已是 BGR 格式)
        depth_optimized: CDM 优化后的深度图 (用于可视化和深度值显示)
        state: 检测结果状态

    Returns:
        tuple: (vis_rgb, vis_depth, vis_combined) - 均为 BGR 格式，用于 OpenCV/ROS

    在图像上叠加:
    - 所有抓取点 (红点)
    - 目标 bbox (红框)
    - 选中抓取框 (绿色旋转矩形)
    - 3D 信息文字
    """
    # rgb 已经是 BGR 格式，无需转换

    affs = state['affs']
    fts = state['fts']
    chosen_aff = state['chosen_aff']
    point3d = state['point3d']        # 相机坐标系 3D 点
    img_h, img_w = rgb.shape[:2]
    depth = depth_optimized           # 使用 CDM 优化后的深度图

    # === 深度图可视化 (TURBO colormap, 使用优化后深度) ===
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_TURBO)

    # === 绘制所有抓取点 (红点) ===
    # affs 格式: [[obj1, obj2, ...]] - 双层列表
    if affs and len(affs) > 0 and len(affs[0]) > 0:
        try:
            tmp = [obj.get('affs', []) for obj in affs[0] if isinstance(obj, dict)]
            ps = [x for p in tmp for x in p]
            ps = [(int(x[0]), int(x[1])) for x in ps]
            ps = [x for x in ps if x[0] <= img_w and x[1] <= img_h]
            if ps:
                ps = [[x, x] for x in ps]
                ps = np.array(ps, dtype=np.int32)
                cv2.polylines(rgb, ps, isClosed=False, color=[0, 0, 255], thickness=7)
        except (KeyError, TypeError, IndexError) as e:
            rospy.logwarn(f"[VIS] 解析 affs 失败: {e}")

    # === 绘制目标 bbox (红框) ===
    if 'targets' in fts:
        for target in fts['targets']:
            bbox = target['bbox']
            bbox = [int(x) for x in bbox]
            cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0, 0, 255], 2)

    # === 绘制选中的抓取框和3D信息 ===
    text_offset_u = 100
    text_offset_v = 20
    delta_v = 30

    if chosen_aff is not None:
        # 绘制旋转抓取框 (绿色)
        rect = chosen_aff['aff']
        rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4] * 180 / math.pi)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(rgb, [box], 0, (0, 255, 0), 2)

        # 绘制抓取中心点 (绿点)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.circle(rgb, (cx, cy), 3, [0, 255, 0], 3)

        # === 叠加 3D 信息文字 ===
        # 1. 相机坐标系 3D 位置
        if point3d is not None:
            cv2.putText(rgb,
                f'3D in Cam: {[round(x * 1000) for x in point3d]} mm',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_offset_v += delta_v

        # 2. 像素坐标
        cv2.putText(rgb,
            f'UV in RGB: ({cx}, {cy}) px',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_offset_v += delta_v

        # 3. 深度值
        depth_value = depth[cy, cx] if 0 <= cy < img_h and 0 <= cx < img_w else 0
        cv2.putText(rgb,
            f'depth: {depth_value*1000:.1f} mm',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_offset_v += delta_v

        # 4. 抓取置信度
        cv2.putText(rgb,
            f'aff score: {chosen_aff["score"]:.2f}',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_offset_v += delta_v

        # 5. 物体类别
        category = chosen_aff.get('category', 'unknown')
        cv2.putText(rgb,
            f'category: {category}',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_offset_v += delta_v

        # 6. 夹爪宽度
        width3d = state.get('width3d')
        if width3d is not None:
            cv2.putText(rgb,
                f'gripper width: {width3d*1000:.1f} mm',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_offset_v += delta_v

        # 7. 抓取角度
        angle_deg = chosen_aff['aff'][4] * 180 / math.pi
        cv2.putText(rgb,
            f'angle: {angle_deg:.1f} deg',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        # 无有效抓取
        cv2.putText(rgb,
            'no valid affordance',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === RGB + Depth 上下堆叠 ===
    vis_combined = np.vstack((rgb, depth_colored))

    return rgb, depth_colored, vis_combined


def _save_result(self, rgb, depth, result):
    """
    保存检测结果到文件

    Args:
        rgb: RGB 图像 (原始, RGB 格式)
        depth: 深度图 (CDM 优化后, float32 米)
        result: detect() 返回的完整结果
    """
    import json
    import os
    from datetime import datetime

    result_dir = self.config['visualization'].get('result_dir', 'result/perception_grasp')

    # 创建时间戳目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(result_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # rgb 已经是 BGR 格式，可直接保存
    # 1. 保存原始图像
    cv2.imwrite(os.path.join(save_dir, 'rgb.jpg'), rgb)

    # 2. 保存深度图 (16bit, mm)
    depth_mm = (depth * 1000).astype(np.uint16)
    cv2.imwrite(os.path.join(save_dir, 'depth.png'), depth_mm)

    # 3. 保存可视化图像
    if 'vis_rgb' in result:
        cv2.imwrite(os.path.join(save_dir, 'vis_detection.jpg'), result['vis_rgb'])
    if 'vis_depth' in result:
        cv2.imwrite(os.path.join(save_dir, 'vis_depth.jpg'), result['vis_depth'])
    if 'vis_combined' in result:
        cv2.imwrite(os.path.join(save_dir, 'vis_combined.jpg'), result['vis_combined'])

    # 4. 保存结果 JSON
    json_data = {
        'timestamp': time.time(),
        'success': result['success'],
        'detection_time_ms': result.get('detection_time_ms', 0),
    }

    if result['success']:
        json_data.update({
            'point3d': result['point3d'],
            'width3d': result['width3d'],
            'angle': result['angle'],
            'category': result['category'],
            'score': result['score'],
            'center_uv': result['center_uv'],
            'depth_value': result['depth_value'],
        })

        # 保存 chosen_aff 详情
        if result.get('chosen_aff'):
            json_data['chosen_aff'] = {
                'aff': list(result['chosen_aff']['aff']),
                'score': result['chosen_aff']['score'],
                'category': result['chosen_aff'].get('category', 'unknown'),
                'touching_points': result['chosen_aff'].get('touching_points', [])
            }

    # 保存 targets (DinoX 结果)
    if result.get('targets'):
        json_data['targets'] = result['targets']

    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 5. 更新 latest 软链接
    latest_link = os.path.join(result_dir, 'latest')
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(timestamp, latest_link)

    rospy.loginfo(f"[PerceptionGrasp] 结果已保存到 {save_dir}")
```

---

## 5. ROS 接口设计

### 5.1 两种触发模式

| 模式 | 触发方式 | 适用场景 |
|------|----------|----------|
| **Service 模式** | 调用 `~detect` 服务 | demo.py 按需调用，需要返回值 |
| **实时检测模式** | 发布到 `/usr/prompt/grasp` | 持续监控，结果通过 Topic 输出 |

```
                        ┌─────────────────────────────────────┐
                        │       perception_grasp_node         │
                        │                                     │
  /usr/prompt/grasp ───►│  实时模式: 订阅 prompt 触发检测      │
  (std_msgs/String)     │                                     │
                        │                                     │───► ~vis_rgb
  ~detect (Service) ───►│  Service模式: 按需调用返回结果       │───► ~vis_depth
                        │                                     │───► ~vis_combined
                        │                                     │───► ~result (实时模式)
                        └─────────────────────────────────────┘
```

### 5.2 Service 接口

**srv/GraspDetect.srv** - 简洁接口，与 demo.py 兼容

```
# === Request ===
string prompt                    # 检测提示词 (如 "bottle.cup")
bool enable_cdm                  # 是否启用 CDM 深度优化 (默认 true)

---

# === Response ===
bool success                     # 检测是否成功
string error_message             # 错误信息 (失败时)

# 核心 3D 输出 (相机光学坐标系: Z向前, X右, Y下)
float64[] point3d                # [x, y, z] 米 (长度=3)
float64 width3d                  # 夹爪宽度 米
float64 angle                    # 抓取角度 度 (非弧度)
string category                  # 物体类别
float64 score                    # 抓取置信度

# 2D 信息 (辅助)
float64[] center_uv              # [u, v] 像素坐标 (长度=2)
float64 depth                    # 中心点深度值 米

# 统计
float64 detection_time_ms        # 检测耗时 ms
```

### 5.3 实时检测模式 (Topic 触发)

**输入 Topic**:
| Topic | 类型 | 说明 |
|-------|------|------|
| `/usr/prompt/grasp` | std_msgs/String | 检测提示词，发布后触发一次检测 |

**输出 Topic**:
| Topic | 类型 | 说明 |
|-------|------|------|
| `~result` | perception/GraspResult | 检测结果 (实时模式专用) |

**msg/GraspResult.msg** - 实时模式结果消息

```
std_msgs/Header header
bool success
string error_message

# 核心 3D 输出
float64[] point3d                # [x, y, z] 米
float64 width3d                  # 夹爪宽度 米
float64 angle                    # 抓取角度 度
string category                  # 物体类别
float64 score                    # 置信度

# 2D 信息
float64[] center_uv              # [u, v] 像素
float64 depth                    # 深度值 米

float64 detection_time_ms
```

### 5.4 调用示例

```python
# Service 调用
from perception.srv import GraspDetect
grasp_detect = rospy.ServiceProxy('/perception_grasp_node/detect', GraspDetect)
resp = grasp_detect(prompt='bottle', enable_cdm=True)
if resp.success:
    point3d = list(resp.point3d)  # 相机坐标系 3D 点
```

```bash
# Topic 触发
rostopic pub /usr/prompt/grasp std_msgs/String "bottle"
rostopic echo /perception_grasp_node/result
```

---

## 6. 配置文件

### 6.1 perception_grasp.yaml

```yaml
perception_grasp:
  # 相机 Topic 配置 (订阅 camera_driver)
  camera:
    rgb_topic: '/camera/hand/color/image_raw'
    depth_topic: '/camera/hand/aligned_depth_to_color/image_raw'
    camera_info_topic: '/camera/hand/color/camera_info'
    width: 1280
    height: 720

  # 触发模式配置
  trigger:
    # Service 模式 (始终启用)
    service_name: "~detect"

    # 实时检测模式
    realtime_mode:
      enabled: true                          # 是否启用实时模式
      prompt_topic: "/usr/prompt/grasp"      # 输入: 检测提示词
      result_topic: "~result"                # 输出: 检测结果

  # 检测服务 (三个服务并行调用)
  services:
    grasp:
      server_list: 'config/server_grasp.json'  # 服务器列表配置文件
      model_name: 'full'                       # 模型名称: 'full' 或 'lite'
      timeout: 10
      warmup: true               # 启动时预热

    dinox:
      url: http://192.168.112.14:10086
      timeout: 10

    cdm:  # 深度优化服务
      url: http://192.168.112.14:8086
      timeout: 10
      enabled: true              # 是否启用 CDM 深度优化 (实时模式使用此配置)
      chosen_policy: 'dn'        # 'dn' 仅去噪, 'dn,vis' 去噪+可视化

  # 深度有效范围 (米)
  depth:
    min_depth: 0.001
    max_depth: 2.0

  # 过滤参数
  filter:
    min_grasp_score: 0.3
    min_object_score: 0.25

  # 可视化配置
  visualization:
    save_to_file: true
    result_dir: "result/perception_grasp"
    publish_to_topic: true
```

---

## 7. 验证方法

```bash
# 编译
cd ~/MobileManipulator && catkin_make

# 启动相机和感知节点
roslaunch camera_driver camera_driver.launch hand_enable:=true
roslaunch perception perception_grasp.launch

# 调用检测服务
rosservice call /perception_grasp_node/detect "{prompt: 'bottle', enable_cdm: true}"
```

---

## 8. 变更记录

| 版本 | 主要变更 |
|------|----------|
| v1.0 | 服务类初始化、实时模式、文件保存 |
| v1.1 | **三服务完全并行**，移除两阶段设计 |
| v1.2 | **最终审查**: 修复返回值解包、移除 pyrealsense2 依赖、路径处理 |
| v1.3 | **二次审查**: Topic 名称修正、端口修正 (12086)、BGR 统一、targets 必须匹配 |
| v1.4 | **坐标系澄清**: 输出为相机光学坐标系 (Z向前, X右, Y下)，移除 iou_weight |
