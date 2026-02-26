# Scene Perception 3D 模块设计文档

> 版本: v1.4
> 日期: 2026-01-16
> 更新: Phase 1-3 完成，核心算法方法重命名为 camera_3d_percept / lidar_3d_percept

## 目标

1. 从 ROS1 topic 读取 camera (RGB + Depth) 和 LiDAR 数据（替代 RealSense SDK 直接调用）
2. 输出深度相机检测到的 objects 的 3D 坐标到 topic
3. 可选输出 LiDAR 检测距离到对应 topic
4. 误差精度报表打印到日志（可通过参数控制开关）

---

## 组件状态

| 文件 | 位置 | 状态 |
|------|------|------|
| `ros_lidar.py` | `src/perception/src/` | ✅ 直接复用 |
| `coordinate_transformer.py` | `src/perception/src/` | ✅ 已更新 (支持 base_link 变换) |
| `synced_sensor_subscriber.py` | `src/perception/src/` | ✅ 已实现 (P0/P1 修复) |
| `scene_perception_core.py` | `src/perception/src/` | ✅ 已实现 (核心算法) |
| `percept.py` (DinoXDetectorOnline) | `src/perception/src/` | ✅ 直接复用 |
| `msg/Object3D.msg` | `src/perception/msg/` | ✅ 已创建 |
| `msg/Object3DArray.msg` | `src/perception/msg/` | ✅ 已创建 |
| `srv/DetectObjects.srv` | `src/perception/srv/` | ✅ 已创建 |

---

## ROS Topics 定义

**输入 - 传感器 (camera_driver)**:
- `/camera/{camera_name}/color/image_raw` - sensor_msgs/Image (bgr8, 640x480@15fps)
- `/camera/{camera_name}/aligned_depth_to_color/image_raw` - sensor_msgs/Image (16UC1, mm)
- `/camera/{camera_name}/color/camera_info` - sensor_msgs/CameraInfo
- `/lidar/chassis/point_cloud` - sensor_msgs/PointCloud2 **(10Hz)**

**输入 - Prompt (可选)**:
- `/user/prompt` - std_msgs/String (持续监听检测提示词)

**输出**:
- `/scene_perception_3d/objects_3d` - perception/Object3DArray

**Service**:
- `/scene_perception_3d/detect` - perception/DetectObjects
- 预期响应时间: **~500ms** (含 DINO-X 检测 + CDM 优化)

**camera_name 可选值**: `top` (默认) | `chassis` | `hand`

---

## 文件结构

```
src/perception/
├── CMakeLists.txt                    # [重新创建] ROS 构建配置
├── package.xml                       # [重新创建] ROS 包依赖
├── setup.py                          # [重新创建] Python 包安装
├── msg/
│   ├── Object3D.msg                  # [新增] 单个 3D 物体
│   └── Object3DArray.msg             # [新增] 3D 物体数组
├── srv/
│   └── DetectObjects.srv             # [新增] 检测服务
├── config/
│   ├── scene_perception_3d.yaml      # [新增] 节点配置
│   └── extrinsics_*.yaml             # [已有] 外参
├── launch/
│   └── scene_perception_3d.launch    # [新增] 启动文件
└── src/
    ├── synced_sensor_subscriber.py   # [新增] 同步订阅器 (RGB+Depth+LiDAR)
    ├── ros_lidar.py                  # [已有] 可选单独使用
    ├── scene_perception_core.py      # [新增] 核心测量算法 (从 analyzer 提取)
    ├── scene_perception_3d_node.py   # [新增] 主感知 ROS 节点
    ├── coordinate_transformer.py     # [已有] 坐标变换
    └── percept.py                    # [已有] 检测服务
```

---

## 消息定义

### msg/Object3D.msg
```
string object_id                   # 物体ID (category_index, 如 "bottle_1")
string category                    # 类别名称
float64 score                      # 检测置信度
float64[4] bbox                    # 2D bbox [x1, y1, x2, y2]

# 相机测量 (坐标系由 target_frame 配置决定，默认 base_link)
geometry_msgs/Point position       # 3D 位置
float64 distance                   # 到原点距离 (m)
float64 confidence                 # 测量置信度

# LiDAR 测量 (可选, distance_lidar=0 表示无效或禁用)
float64 distance_lidar             # LiDAR 测距 (m)
geometry_msgs/Point position_lidar # LiDAR 质心 (同一坐标系)
float64 confidence_lidar           # LiDAR 置信度
```

### msg/Object3DArray.msg
```
std_msgs/Header header
string frame_id                    # 坐标系 (arm_base_link)
perception/Object3D[] objects
```

### srv/DetectObjects.srv
```
string prompt                      # 检测提示词 (如 "bottle.cup.box")
bool enable_lidar                  # 是否启用 LiDAR 测量
---
bool success
string error_message
perception/Object3DArray result
```

---

## 核心类设计

### 1. SyncedSensorSubscriber (synced_sensor_subscriber.py)

**统一订阅器** - 同步 RGB + Depth，LiDAR 可选

```python
class SyncedSensorSubscriber:
    """同步订阅相机和LiDAR数据

    特性:
    - RGB + Depth 始终同步 (相机 15fps)
    - LiDAR 可选同步 (10Hz)
    - 数据新鲜度检查，避免使用过时数据
    """

    def __init__(self,
                 color_topic='/camera/top/color/image_raw',
                 depth_topic='/camera/top/aligned_depth_to_color/image_raw',
                 info_topic='/camera/top/color/camera_info',
                 lidar_topic='/lidar/chassis/point_cloud',
                 enable_lidar=True,
                 sync_slop=0.1):  # 同步容差 100ms
        self._bridge = CvBridge()
        self._latest_data = None
        self._data_lock = threading.Lock()
        self.enable_lidar = enable_lidar

    def connect(self) -> bool:
        """启动同步订阅 - 分层同步策略"""
        # 订阅 CameraInfo (一次性获取内参)
        self._info_sub = rospy.Subscriber(info_topic, CameraInfo, self._info_cb)

        # 相机同步订阅 (始终需要)
        self._color_sub = message_filters.Subscriber(color_topic, Image)
        self._depth_sub = message_filters.Subscriber(depth_topic, Image)

        if self.enable_lidar:
            # 三路同步: RGB + Depth + LiDAR
            self._lidar_sub = message_filters.Subscriber(lidar_topic, PointCloud2)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub, self._lidar_sub],
                queue_size=10,
                slop=self.sync_slop
            )
            self._sync.registerCallback(self._sync_callback_with_lidar)
        else:
            # 两路同步: RGB + Depth only
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub],
                queue_size=10,
                slop=self.sync_slop
            )
            self._sync.registerCallback(self._sync_callback_camera_only)

    def _sync_callback_with_lidar(self, color_msg, depth_msg, lidar_msg):
        """三路同步回调"""
        rgb = self._bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = depth_mm.astype(np.float32) / 1000.0  # mm -> m
        lidar_points = self._parse_pointcloud(lidar_msg)

        with self._data_lock:
            self._latest_data = {
                'rgb': rgb,
                'depth': depth_m,
                'lidar_points': lidar_points,
                'timestamp': color_msg.header.stamp
            }

    def _sync_callback_camera_only(self, color_msg, depth_msg):
        """两路同步回调 (无 LiDAR)"""
        rgb = self._bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = depth_mm.astype(np.float32) / 1000.0

        with self._data_lock:
            self._latest_data = {
                'rgb': rgb,
                'depth': depth_m,
                'lidar_points': None,  # LiDAR 禁用
                'timestamp': color_msg.header.stamp
            }

    @property
    def intrinsics(self) -> dict:  # {fx, fy, cx, cy, width, height}

    def get_synced_data(self, timeout=5.0, max_age=0.5) -> dict:
        """获取同步数据，带新鲜度检查

        Args:
            timeout: 等待超时 (秒)
            max_age: 数据最大允许年龄 (秒)，默认 500ms

        Returns:
            {'rgb', 'depth', 'lidar_points', 'timestamp'}

        Raises:
            TimeoutError: 无法获取新鲜数据
        """
        start = time.time()
        while time.time() - start < timeout:
            with self._data_lock:
                if self._latest_data is not None:
                    age = (rospy.Time.now() - self._latest_data['timestamp']).to_sec()
                    if age < max_age:
                        return self._latest_data.copy()
            time.sleep(0.01)
        raise TimeoutError(f"无法获取新鲜数据 (max_age={max_age}s, timeout={timeout}s)")
```

### 2. ScenePerceptionCore (scene_perception_core.py) ✅ 已实现

从 `depth_accuracy_analyzer.py` 提取核心算法：

```python
class ScenePerceptionCore:
    """3D 场景感知核心算法"""

    DEPTH_MIN = 0.3
    DEPTH_MAX = 10.0
    MASK_ERODE_KERNEL = 5
    MIN_MASK_AREA = 300
    MIN_DEPTH_POINTS = 10
    MIN_LIDAR_POINTS = 3

    def __init__(self, transformer: CoordinateTransformer, intrinsics: dict,
                 target_frame: str = 'base_link',
                 depth_optimizer: DepthOptimizerOnline = None):
        self.target_frame = target_frame  # 目标坐标系
        self.depth_optimizer = depth_optimizer  # CDM 深度优化 (可选)

    def optimize_depth(self, rgb, depth) -> np.ndarray:
        """CDM 深度去噪优化"""

    def camera_3d_percept(self, depth, mask) -> MeasurementResult:
        """深度相机 3D 感知 (基于 Mask)"""

    def lidar_3d_percept(self, lidar_points, bbox, camera_depth=None) -> MeasurementResult:
        """LiDAR 3D 感知 (相机引导模式)"""
```

### 3. ScenePerception3DNode (scene_perception_3d_node.py)

```python
class ScenePerception3DNode:
    """场景3D感知 ROS 节点

    特性:
    - 支持多相机配置 (top/chassis/hand)
    - 支持多种 prompt 输入方式 (service/topic/default)
    - 启动时健康检查外部服务
    - 检测失败返回空数组 (success=true)
    """

    def __init__(self):
        rospy.init_node('scene_perception_3d')

        # 加载配置
        self._load_config()

        # 启动时健康检查
        self._health_check()

        # 构建相机 topic (根据 camera_name 配置)
        camera_topics = self._build_camera_topics(self.camera_name)

        # 组件初始化
        self.sensor = SyncedSensorSubscriber(
            color_topic=camera_topics['color'],
            depth_topic=camera_topics['depth'],
            info_topic=camera_topics['info'],
            enable_lidar=self.enable_lidar,
            sync_slop=self.sync_slop
        )
        self.transformer = CoordinateTransformer()
        self.transformer.load_all_extrinsics()
        self.detector = DinoXDetectorOnline(...)
        self.core = ScenePerceptionCore(...)

        # Prompt 订阅 (可选)
        self._current_prompt = self.default_prompt
        if self.prompt_source == 'topic':
            self._prompt_sub = rospy.Subscriber(
                self.prompt_topic, String, self._prompt_callback
            )

        # ROS 接口
        self.pub = rospy.Publisher('~objects_3d', Object3DArray, queue_size=1)
        self.srv = rospy.Service('~detect', DetectObjects, self.handle_detect)

    def _build_camera_topics(self, camera_name):
        """根据相机名称构建 topic"""
        return {
            'color': f'/camera/{camera_name}/color/image_raw',
            'depth': f'/camera/{camera_name}/aligned_depth_to_color/image_raw',
            'info': f'/camera/{camera_name}/color/camera_info',
        }

    def _prompt_callback(self, msg):
        """Prompt topic 回调"""
        self._current_prompt = msg.data
        rospy.loginfo(f"更新 prompt: '{msg.data}'")

    def _get_prompt(self, req_prompt=None):
        """获取检测 prompt

        优先级: service 请求 > topic 订阅 > 默认配置
        """
        if self.prompt_source == 'service' and req_prompt:
            return req_prompt
        elif self.prompt_source == 'topic':
            return self._current_prompt
        else:
            return self.default_prompt

    def _health_check(self):
        """启动时健康检查"""
        # 检查 DINO-X 服务
        if not self._check_service(self.detector_url, timeout=5.0):
            rospy.logerr(f"DINO-X 服务不可用: {self.detector_url}")
            raise RuntimeError("DINO-X 服务不可用")

        # CDM 服务可选，失败时自动禁用
        if self.enable_depth_optimizer:
            if not self._check_service(self.cdm_url, timeout=3.0):
                rospy.logwarn(f"CDM 服务不可用，禁用深度优化: {self.cdm_url}")
                self.enable_depth_optimizer = False

    def handle_detect(self, req) -> DetectObjectsResponse:
        """处理检测请求

        错误处理策略:
        - 数据采集失败: success=false, error_message 说明原因
        - 检测到 0 个物体: success=true, objects=[]
        - DINO-X 失败: success=false
        """
        response = DetectObjectsResponse()
        response.success = True
        response.result.frame_id = self.target_frame

        try:
            # 1. 采集同步数据 (带新鲜度检查)
            data = self.sensor.get_synced_data(timeout=2.0, max_age=0.5)

            # 2. 获取 prompt
            prompt = self._get_prompt(req.prompt)

            # 3. 检测目标 (DINO-X)
            detections = self.detector.detect(prompt, data['rgb'])
            if not detections:
                rospy.loginfo(f"未检测到目标: '{prompt}'")
                return response

            # 4. CDM 深度优化 (可选)
            depth = self.core.optimize_depth(data['rgb'], data['depth'])

            # 5. 对每个物体进行 3D 测量
            for det in detections:
                obj = self._measure_object(det, depth, data['lidar_points'])
                if obj:
                    response.result.objects.append(obj)

        except TimeoutError as e:
            response.success = False
            response.error_message = f"数据采集超时: {e}"
        except Exception as e:
            response.success = False
            response.error_message = f"检测失败: {e}"

        # 6. 发布到 topic
        self.pub.publish(response.result)
        return response
```

---

## CDM 深度优化集成

### 现有实现位置
- **服务类**: `src/perception/src/percept.py:928` - `DepthOptimizerOnline`
- **调用示例**: `src/perception/src/depth_accuracy_analyzer.py:311-336` - `optimize_depth()`

### 数据流
```
RGB (BGR, uint8)  ─────┐
                       ├──► CDM Service ──► 优化后深度图
Depth (m, float32) ────┘    (http://192.168.112.14:8086)
      │
      ▼ 转换
Depth (mm, uint16)  ──► API 请求 ──► 响应 (mm, uint16) ──► 转回 (m, float32)
```

### 集成代码 (直接复用)

```python
# 来自 depth_accuracy_analyzer.py:311-336
def optimize_depth(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """使用 CDM 服务优化深度图

    Args:
        rgb: BGR 图像 (H, W, 3) uint8
        depth: 深度图 (H, W) float32, 单位: 米

    Returns:
        优化后的深度图 (H, W) float32, 单位: 米
    """
    if self.depth_optimizer is None:
        return depth

    # 转换为 mm uint16 (CDM 服务要求的格式)
    depth_mm = (depth * 1000).astype(np.uint16)

    # 调用 CDM 服务
    result = self.depth_optimizer.forward(rgb, depth_mm)

    if result.get('success') and 'depth' in result:
        # 转回 m float32
        optimized = result['depth'].astype(np.float32) / 1000.0
        return optimized
    else:
        rospy.logwarn(f"深度优化失败: {result.get('error', 'unknown')}")
        return depth  # 失败时返回原始深度
```

### DepthOptimizerOnline 初始化

```python
# 在 PerceptionNode 中初始化
from percept import DepthOptimizerOnline

class SimpleConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# 初始化 CDM
if self.enable_depth_optimizer:
    cdm_cfg = SimpleConfig(
        url='http://192.168.112.14:8086',
        chosen_policy='dn',  # 仅去噪，不需要可视化
        warmup=0
    )
    self.depth_optimizer = DepthOptimizerOnline(cdm_cfg)
else:
    self.depth_optimizer = None
```

### 关键注意事项

1. **单位转换**: ROS topic 深度是 mm (uint16)，需要先转为 m (float32)，CDM 输入要求 mm (uint16)
2. **失败降级**: CDM 服务不可用时，返回原始深度继续处理
3. **性能**: CDM 处理约 50-100ms/帧，对实时性有轻微影响

---

## 数据流

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              SyncedSensorSubscriber                     │
                        │         (分层同步，带新鲜度检查)                          │
                        │                                                         │
/camera/top/color/image_raw ────┐                                                │
/camera/top/aligned_depth... ───┼──► RGB + Depth 同步 (必须)                     │
/camera/top/color/camera_info ──┘                                                │
                                                                                 │
/lidar/chassis/point_cloud ─────────► LiDAR 同步 (可选，enable_lidar=true)       │
                        └─────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼ get_synced_data(max_age=0.5s)
                                          ┌──────────────┐
                                          │ synced_data  │
                                          │ {rgb, depth, │
                                          │  lidar_pts}  │
                                          └──────┬───────┘
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        │                        │                        │
                        ▼                        ▼                        ▼
                 DinoXDetector            CDM 深度优化               LiDAR 点云
                 (RGB → bbox+mask)     (RGB+Depth → depth')         (可选)
                        │                        │                        │
                        └────────────────────────┼────────────────────────┘
                                                 │
                                                 ▼
                                          PerceptionCore
                                    ┌─────────────────────────┐
                                    │ measure_camera(depth, mask)
                                    │     → 相机 3D 位置 (optical 坐标系)
                                    │
                                    │ measure_lidar(lidar_pts, bbox)
                                    │     → LiDAR 3D 位置 (rslidar 坐标系)
                                    └─────────────────────────┘
                                                 │
                                                 ▼
                                       CoordinateTransformer
                                    ┌─────────────────────────┐
                                    │ 相机: optical → target_frame
                                    │ LiDAR: rslidar → target_frame (直接)
                                    └─────────────────────────┘
                                                 │
                                                 ▼
                                          Object3DArray
                                       (target_frame 坐标系)
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             /perception/objects_3d                         DetectObjects Response
                   (topic)                                      (service)
```

---

## 配置文件

### config/scene_perception_3d.yaml
```yaml
scene_perception_3d:
  # 相机配置 (top | chassis | hand)
  camera_name: top
  # 自动拼接为:
  #   /camera/{camera_name}/color/image_raw
  #   /camera/{camera_name}/aligned_depth_to_color/image_raw
  #   /camera/{camera_name}/color/camera_info

  # LiDAR 配置
  lidar_topic: /lidar/chassis/point_cloud
  enable_lidar: true             # 启用 LiDAR (10Hz)

  # Prompt 配置
  prompt_source: service         # service | topic | default
  prompt_topic: /user/prompt     # prompt_source=topic 时使用
  default_prompt: "bottle.cup.box"  # 默认检测提示词

  # 检测服务 (DINO-X)
  detector_url: http://192.168.112.14:10086
  detector_timeout: 3.0          # 检测超时 (秒)
  min_score: 0.25

  # 深度优化服务 (CDM)
  depth_optimizer_url: http://192.168.112.14:8086
  enable_depth_optimizer: true

  # 坐标系配置
  target_frame: base_link        # 输出坐标系: base_link (默认) 或 arm_base_link

  # 功能开关
  enable_accuracy_log: false

  # 时间同步参数
  sync_slop: 0.1                 # 同步容差 100ms (相机 15fps=66ms, LiDAR 10Hz=100ms)
  data_max_age: 0.5              # 数据新鲜度阈值 500ms

  # 算法参数
  depth_min: 0.3
  depth_max: 10.0
```

### launch/scene_perception_3d.launch
```xml
<launch>
  <arg name="camera_name" default="top"/>
  <arg name="enable_lidar" default="true"/>
  <arg name="prompt_source" default="service"/>
  <arg name="enable_accuracy_log" default="false"/>

  <node name="scene_perception_3d" pkg="perception" type="scene_perception_3d_node.py" output="screen">
    <rosparam file="$(find perception)/config/scene_perception_3d.yaml"/>
    <param name="camera_name" value="$(arg camera_name)"/>
    <param name="enable_lidar" value="$(arg enable_lidar)"/>
    <param name="prompt_source" value="$(arg prompt_source)"/>
    <param name="enable_accuracy_log" value="$(arg enable_accuracy_log)"/>
  </node>
</launch>
```

---

## 实现步骤

### Phase 1: ROS 包基础设施 ✅ 已完成
1. [x] 重新创建 `CMakeLists.txt` 添加 msg/srv 生成
2. [x] 重新创建 `package.xml` 添加依赖
3. [x] 创建 `msg/Object3D.msg` 和 `msg/Object3DArray.msg`
4. [x] 创建 `srv/DetectObjects.srv`
5. [x] `catkin build` 编译验证

### Phase 2: 数据订阅层 ✅ 已完成
6. [x] 实现 `synced_sensor_subscriber.py` (message_filters 同步订阅)
7. [x] 更新 `coordinate_transformer.py` (支持 base_link)
8. [x] 单独测试数据采集

### Phase 3: 核心算法提取 ✅ 已完成
9. [x] 创建 `scene_perception_core.py`，包含:
   - `camera_3d_percept()` - 深度相机 3D 感知
   - `lidar_3d_percept()` - LiDAR 3D 感知
   - `optimize_depth()` - CDM 深度优化
   - `MeasurementResult` 数据类

### Phase 4: 节点集成
10. [ ] 实现 `scene_perception_3d_node.py`
11. [ ] 创建 `config/scene_perception_3d.yaml`
12. [ ] 创建 `launch/scene_perception_3d.launch`

### Phase 5: 测试验证
13. [ ] 启动 camera_driver 和 rslidar
14. [ ] 启动 scene_perception_3d 节点
15. [ ] 调用 service 验证检测结果

---

## 验证方法

```bash
# 1. 启动依赖
roslaunch camera_driver camera_driver.launch top_enable:=true
roslaunch rslidar_sdk start.launch

# 2. 启动感知节点 (使用 top camera)
roslaunch perception scene_perception_3d.launch camera_name:=top enable_accuracy_log:=true

# 3. 调用检测服务
rosservice call /scene_perception_3d/detect "{prompt: 'bottle', enable_lidar: true}"

# 4. 查看输出 topic
rostopic echo /scene_perception_3d/objects_3d

# --- 其他使用方式 ---

# 使用 chassis camera
roslaunch perception scene_perception_3d.launch camera_name:=chassis

# 使用 topic 方式输入 prompt
roslaunch perception scene_perception_3d.launch prompt_source:=topic
rostopic pub /user/prompt std_msgs/String "data: 'person.chair'"
```

---

## 设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| CDM 深度优化 | ✅ 集成 | http://192.168.112.14:8086，启动时检查，不可用自动禁用 |
| 触发模式 | Service | `/perception_node/detect` 按需调用，响应时间 ~500ms |
| 相机支持 | top camera | 后续可扩展 hand/chassis |
| 坐标系输出 | 可配置 | `base_link` (默认) 或 `arm_base_link` |
| 时间同步 | message_filters | 分层同步：RGB+Depth 必须，LiDAR 可选 |
| 数据新鲜度 | max_age=0.5s | 超过 500ms 的数据视为过时 |
| LiDAR 变换 | rslidar→target | 直接变换，不经过 optical |
| 检测失败 | 空数组 | success=true + objects=[] |
| 包结构 | 重新创建 | CMakeLists.txt, package.xml, setup.py |

### 已修复的架构问题

| 问题 | 级别 | 修复方案 | 状态 |
|------|------|----------|------|
| CoordinateTransformer 不支持 base_link | P0 | 新增 optical_to_base, rslidar_to_base 变换加载 | ✅ 已实现 |
| 数据新鲜度检查缺失 | P0 | get_synced_data() 增加 max_age 参数 | ✅ 已实现 |
| enable_lidar=false 会卡死 | P1 | 分层同步策略，LiDAR 可选 | ✅ 已实现 |
| 无服务健康检查 | P1 | 启动时检查 DINO-X/CDM 服务 | 待实现 (节点) |
| 消息定义与配置矛盾 | P1 | 修正注释为 "由 target_frame 配置决定" | ✅ 已修正 |

---

## 误差精度报表 (单次检测)

### 日志输出格式

当 `enable_accuracy_log: true` 时，每次检测输出：

```
[PerceptionNode] === 检测结果 ===
  物体: bottle_1 (score=0.85)
  坐标系: base_link

  相机测量:
    位置: (1.234, 0.156, 0.892) m
    距离: 1.523 m
    置信度: 0.82
    深度点数: 156

  LiDAR测量:
    位置: (1.245, 0.148, 0.901) m
    距离: 1.534 m
    置信度: 0.78
    LiDAR点数: 23

  误差分析:
    距离差: +11 mm (0.72%)
    位置偏差: [+11, -8, +9] mm
```

### 实现方式

```python
def _log_accuracy_report(self, obj_id, camera_result, lidar_result, target_frame):
    """打印单次检测的精度报表"""
    if not self.enable_accuracy_log:
        return

    rospy.loginfo(f"=== 检测结果 ===")
    rospy.loginfo(f"  物体: {obj_id}")
    rospy.loginfo(f"  坐标系: {target_frame}")
    # ... 格式化输出
```

---

## 坐标系配置

### 支持的变换路径

**相机测量变换链**: optical → target
```
top_camera_optical_frame
         │
         ├──► base_link      (默认，底盘中心)
         │
         └──► arm_base_link  (可选，机械臂基座)
```

**LiDAR 测量变换链**: rslidar → target (直接变换，不经过 optical)
```
rslidar
    │
    ├──► base_link      (默认)
    │
    └──► arm_base_link  (可选)
```

### 外参文件对应关系

| 变换 | 外参文件 |
|------|----------|
| optical → base_link | `extrinsics_top_camera_optical_frame_to_base_link.yaml` |
| optical → arm_base_link | `extrinsics_top_camera_optical_frame_to_arm_base_link.yaml` |
| rslidar → base_link | `extrinsics_rslidar_to_base_link.yaml` |
| rslidar → arm_base_link | `extrinsics_rslidar_to_arm_base_link.yaml` |
| rslidar → optical | `extrinsics_rslidar_to_top_camera_optical_frame.yaml` |

### CoordinateTransformer 更新 (P0 修复) ✅ 已实现

`coordinate_transformer.py` 已更新，新增以下便捷方法:

```python
# 直接变换到目标坐标系
transformer.optical_to_target(points, 'base_link')     # optical → base_link
transformer.optical_to_target(points, 'arm_base_link') # optical → arm_base_link
transformer.rslidar_to_target(points, 'base_link')     # rslidar → base_link
transformer.rslidar_to_target(points, 'arm_base_link') # rslidar → arm_base_link
```

已加载的变换矩阵:
- `rslidar_to_optical`, `optical_to_arm`, `rslidar_to_arm` (原有)
- `optical_to_base`, `rslidar_to_base` (新增)
- `base_to_optical`, `base_to_rslidar` (自动计算逆变换)

---

## 关键文件引用

- `src/perception/src/depth_accuracy_analyzer.py:338-674` - 核心测量算法
- `src/perception/src/ros_lidar.py` - LiDAR 订阅器参考
- `src/perception/src/coordinate_transformer.py` - 坐标变换
- `src/perception/src/percept.py:100-300` - DinoXDetectorOnline
