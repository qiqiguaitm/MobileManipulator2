# Pick Navigator 设计文档

> ROS2 自主拣取系统（导航 + 抓取）
>
> 版本: 5.0
> 日期: 2026-02-26

---

## 1. 概述

### 1.1 目标

机器人自主拣取室内所有可见物品：
- **顶部相机**：发现物体、提供导航目标
- **导航模块**：底盘移动到物体附近
- **手部相机**：精确定位抓取点
- **机械臂**：执行抓取

### 1.2 核心流程

```
导航到目标附近 → 扫描工作区 → 批量抓取区内物品 → 导航到下一个目标 → 循环
```

### 1.3 关键设计

| 问题 | 解决方案 |
|------|----------|
| 导航精度不足 | 三阶段导航（Nav2 → 对齐 → 精确接近） |
| 抓取需要精确位姿 | 双相机协作（顶部粗定位 + 手部精定位） |
| 已抓物体如何删除 | SAM2跟踪器自动处理（看不到就消失） |
| 减少导航次数 | 批量抓取（一次导航抓取工作区内所有物品） |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION (ROS2 ✅已实现)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ObjectTrackerNode                    PerceptionGraspNode               │
│  (顶部相机 5Hz)                       (手部相机 按需)                    │
│  ├─ SAM2 跟踪                         ├─ DINO-X + GraspAnything         │
│  ├─ 双相机融合 (Top+Chassis)          ├─ 精确位置 ±5mm                  │
│  ├─ 3D 位置 ±5cm                      ├─ 抓取角度                       │
│  └─ track_id (持续跟踪)               └─ 夹爪宽度                       │
│                                                                         │
│  发布: /object_tracker_node/tracked_objects (TrackedObject3DArray)     │
│  服务: /perception_grasp_node/detect (GraspDetect.srv)                 │
└────────────┬────────────────────────────────┬───────────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PICK MISSION MANAGER (❌待实现)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   状态机: IDLE → PLANNING → NAVIGATING → SCANNING → PICKING → 循环     │
│                                                                         │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│   │  TargetPool  │   │ Approach     │   │ Piper        │               │
│   │  (目标管理)   │   │ Navigator    │   │ GraspNode    │               │
│   │              │   │ (✅已实现)    │   │ (✅已迁移)    │               │
│   └──────────────┘   └──────────────┘   └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           HARDWARE LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│   Nav2 Stack              DepthSensor             PiperAPI V2           │
│   (路径规划)               (深度点云处理)          (CAN总线控制)          │
│   - BasicNavigator        - RANSAC去地面          - 分段MoveJ下降       │
│   - goToPose()            - 聚类找最近点          - IK轨迹规划          │
│                           - 25Hz感知线程          - 黑名单系统          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块实现状态

| 模块 | 状态 | 包名 | 说明 |
|------|------|------|------|
| ObjectTrackerNode | ✅ 已实现 | `perception_nodes` | 顶部相机跟踪，SAM2 Online |
| PerceptionGraspNode | ✅ 已实现 | `perception_nodes` | 手部相机检测，GraspDetect.srv |
| ApproachNavigator | ✅ 已实现 | `approach_navigator` | 三阶段精确导航 |
| PiperAPI V2 | ✅ 已实现 | `piper_driver` | SDK层，无ROS依赖 |
| PiperGraspNode | ✅ 已迁移 | `piper_grasp` | ROS2抓取节点 |
| **PickMissionManager** | ❌ **待实现** | `pick_navigator` | 任务调度状态机 |

### 依赖关系

```
ObjectTrackerNode ──┐
                    ├──► PickMissionManager (待实现)
ApproachNavigator ──┤
                    │
PiperGraspNode ─────┘
```

---

## 4. 各模块 ROS2 接口

### 4.1 ObjectTrackerNode（感知-跟踪）

**位置**: `src/perception2/perception_nodes/`

**发布话题**:
```yaml
~/tracked_objects:
  类型: TrackedObject3DArray
  频率: 5Hz
  内容:
    - track_id: int32          # 跟踪ID（持续不变）
    - category: string         # 类别名称
    - position: Point          # base_link 坐标系，单位 m
    - distance: float64        # 到原点距离，单位 m
    - track_score: float64     # 跟踪置信度
    - position_confidence: float64  # 3D位置置信度
```

**特性**:
- SAM2 Online 跟踪（HTTP 远程服务）
- 双相机融合（Top + Chassis，匈牙利算法匹配）
- 目标丢失后自动从列表消失（无历史记忆）

### 4.2 PerceptionGraspNode（感知-抓取检测）

**位置**: `src/perception2/perception_nodes/`

**服务**:
```yaml
/perception_grasp_node/detect:
  类型: GraspDetect.srv
  请求:
    prompt: string            # 检测提示词，如 "bottle.cup"
    enable_cdm: bool          # 启用 CDM 深度优化
  响应:
    success: bool
    point3d: float64[3]       # [x,y,z] 单位 m（相机光学坐标系）
    width3d: float64          # 夹爪宽度，单位 m
    angle: float64            # 抓取角度，单位 度
    category: string
    score: float64
```

### 4.3 ApproachNavigator（导航-三阶段接近）

**位置**: `src/navigation/approach_navigator/`

**核心类**:
```python
class ApproachNavigator(Node):
    def approach_to_target(
        self,
        target_position: Point,           # map 坐标系
        status_callback: Callable = None
    ) -> ApproachResult
```

**三阶段流程**:
| 阶段 | 方法 | 精度 | 说明 |
|------|------|------|------|
| Stage1 | Nav2 goToPose | ±5cm | 全局规划，停在距目标 45cm |
| Stage2 | PD 原地旋转 | ±3° | 对齐目标方向 |
| Stage3 | 深度闭环前进 | ±2cm | 底盘前边缘距目标 5cm |

**返回结果**:
```python
@dataclass
class ApproachResult:
    success: bool
    error_code: str      # NAV_FAILED, ALIGN_FAILED, APPROACH_FAILED, NAV_CANCELLED
    final_distance: float
```

### 4.4 PiperGraspNode（机械臂控制）

**位置**: `src/piper_grasp/`

#### 服务接口

```yaml
/piper/enable:
  类型: EnableEnhanced.srv
  动作: ACTION_ENABLE=1, ACTION_DISABLE=2, ACTION_FORCE_ENABLE=3,
        ACTION_CLEAN_ERROR=4, ACTION_RECONNECT=5, ACTION_SHUTDOWN=6

/piper/go_ready:
  类型: GoReady.srv
  请求:
    speed: int32              # 1-100%，默认 30
    open_gripper: bool        # 默认 false（闭合更安全）
  响应:
    success: bool
    position: float32[6]      # [x,y,z,r,p,y] mm/度

/piper/observe:
  类型: Observe.srv
  请求:
    prompt: string            # 检测提示词
    enable_cdm: bool          # 默认 true
  响应:
    success: bool
    category: string
    point3d_camera: float32[3]  # 相机坐标系 [x,y,z] mm
    point3d_base: float32[3]    # 基础坐标系 [x,y,z] mm
    offset: float32[3]          # 相对当前位置偏移 mm
    angle_base: float32         # 目标 yaw 角，度
    gripper_width: float32      # 夹爪宽度 mm
    score: float32

/piper/in_working_area:
  类型: InWorkingArea.srv
  请求:
    offset: float32[]         # [x,y,z] mm
    point_in_base: float32[]  # 基础坐标系点 mm
  响应:
    in_area: bool
```

#### 动作接口

```yaml
/piper/pick:
  类型: PiperPick.action
  目标:
    use_last_observe: bool    # 使用上次 observe 结果
    target_offset: float32[3] # 手动偏移 mm
    speed: int32              # 1-100%，默认 30
    gripper_width: float32    # 夹爪宽度 mm
    lift_height: float32      # 提升高度 mm，默认 200
    return_to_ready: bool     # 完成后返回 ready
  反馈:
    state: uint8              # 0-9 状态码
    step_name: string         # CHECKING, APPROACHING, OPENING, DESCENDING,
                              # CLOSING, VERIFYING, LIFTING, RETURNING, DONE
    progress: float32         # 0.0-1.0
  结果:
    success: bool
    category: string
    grasp_position: float32[3]  # mm

/piper/place:
  类型: PiperPlace.action
  目标:
    use_default_place: bool   # 使用默认放置位置
    place_position: float32[6]  # [x,y,z,r,p,y] mm/度
    speed: int32
    return_to_ready: bool
  结果:
    success: bool
    final_position: float32[6]
```

---

## 5. 完整抓取流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      批量抓取流程 (1:M)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] PLANNING - 选择导航目标                                            │
│      ├─ 订阅: /object_tracker_node/tracked_objects                      │
│      ├─ 过滤: track_score≥0.3, position_confidence≥0.3, 0.3m<距离<3m   │
│      ├─ 排除: 已抓过的 track_id, 尝试次数≥3 的                          │
│      └─ 选择: 距离最近的作为导航目标                                     │
│                                                                         │
│  [2] NAVIGATING - 三阶段导航                                            │
│      ├─ 调用: ApproachNavigator.approach_to_target(target_position)    │
│      ├─ Stage1: Nav2 到接近点（距目标 45cm）                            │
│      ├─ Stage2: PD 原地对齐（朝向目标 ±3°）                             │
│      └─ Stage3: 深度闭环前进（底盘前边缘距目标 5cm）                     │
│                                                                         │
│  [3] SCANNING - 工作区扫描                                              │
│      ├─ 调用: /piper/go_ready（机械臂到观察位 x=400,y=0,z=150）         │
│      ├─ 等待 0.5s 让跟踪器稳定                                          │
│      ├─ 从 /object_tracker_node/tracked_objects 获取当前目标            │
│      ├─ 调用: /piper/in_working_area 筛选工作区内物品                   │
│      └─ 按距离排序，生成抓取队列                                         │
│                                                                         │
│  [4] PICKING - 批量抓取（循环）                                         │
│      ├─ 从队列取下一个目标的 category                                   │
│      ├─ 调用: /piper/observe (prompt=category)                         │
│      │   └─ 返回: point3d_base, angle_base, gripper_width (mm)         │
│      ├─ 调用: /piper/pick (use_last_observe=true)                      │
│      │   └─ 状态机: CHECKING→APPROACHING→OPENING→DESCENDING→           │
│      │              CLOSING→VERIFYING→LIFTING→RETURNING                │
│      ├─ 调用: /piper/place (use_default_place=true)                    │
│      ├─ 成功: mark_picked(track_id)                                    │
│      ├─ 失败: mark_failed(track_id), 加入黑名单                         │
│      └─ 队列非空则继续，否则回到 [1]                                     │
│                                                                         │
│  [5] 循环 [1]-[4]，直到没有可抓目标 → COMPLETED                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 工作区定义

### 6.1 机械臂工作范围

基于 Piper 机械臂实际可达范围（gripper center，base_link 坐标系）：

```
观察位姿: x=400, y=0, z=150, roll=180, pitch=30, yaw=180 (手部相机俯视地面)
抓取范围: 从观察位下降到地面，沿 X/Y 方向微调

┌─────────────────────────────────────────────────────────────┐
│                      工作区 (侧视图)                          │
│                                                             │
│    Z                                                        │
│    ▲     观察位 (z=150mm)                                   │
│  150├─────●────────────────                                 │
│    │      ↓                                                 │
│    │      ↓ 手部相机视野                                     │
│    │      ↓                                                 │
│    0├─────┼────────────────  地面                           │
│    │      ↓                                                 │
│ -150├─────▼────────────────  最低抓取高度                    │
│    │                                                        │
│    └──────┼──────┼──────┼───────► X                        │
│         250    400    500  (mm)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      工作区 (俯视图)                          │
│                                                             │
│                    X: 250 ~ 500mm (前方)                    │
│                                                             │
│                      ┌───────────┐                          │
│                      │           │                          │
│      Y: -200mm ──────│  可抓取    │────── Y: +200mm         │
│                      │   区域     │                          │
│                      │           │                          │
│                      └───────────┘                          │
│                           ▲                                 │
│                      base_link                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 关键参数（来自 piper_grasp_node.yaml）

| 参数 | 值 | 来源 |
|------|-----|------|
| 观察位 X | 400 mm | `positions.ready.x` |
| 观察位 Z | 150 mm | `positions.ready.z` |
| 最低 Z | -150 mm | `grasp.min_z` |
| 最高 Z | 300 mm | `grasp.max_z` |
| 夹爪偏移 | 135.03 mm | `gripper.offset_mm` |
| 夹爪最大开度 | 90 mm | `gripper.max_mm` |
| 提升高度 | 200 mm | `grasp.lift_height` |

### 6.3 工作区判断（使用 Piper 服务）

```python
# 方法1: 调用 /piper/in_working_area 服务
req = InWorkingArea.Request()
req.point_in_base = [x_mm, y_mm, z_mm]
resp = in_working_area_client.call(req)
if resp.in_area:
    # 可抓取

# 方法2: 本地判断（与 Piper 配置一致）
def is_in_workspace(position_base_m: Point) -> bool:
    x_mm = position_base_m.x * 1000
    y_mm = position_base_m.y * 1000
    z_mm = position_base_m.z * 1000

    # 基于 piper_grasp_node.yaml working_area 配置
    return (-200 <= x_mm - 400 <= 300 and   # offset_xmin/xmax
            -250 <= y_mm <= 250 and          # offset_ymin/ymax
            -250 <= z_mm <= 300)             # offset_zmin/zmax
```

---

## 7. 状态机设计

### 7.1 状态定义

```python
class MissionState(IntEnum):
    IDLE = 0        # 等待启动
    PLANNING = 1    # 选择导航目标
    NAVIGATING = 2  # 底盘导航
    SCANNING = 3    # 工作区扫描
    PICKING = 4     # 批量抓取
    PLACING = 5     # 放置物品
    COMPLETED = 6   # 全部完成
    ERROR = 7       # 出错
```

### 7.2 状态转移

```
         start()
            │
            ▼
┌───────► IDLE
│           │ /pick_mission/start 服务
│           ▼
│       PLANNING ◄─────────────────────────┐
│           │                              │
│           │ 有目标                       │ 工作区队列空
│           ▼                              │
│       NAVIGATING                         │
│           │                              │
│           │ ApproachResult.success       │
│           ▼                              │
│       SCANNING                           │
│           │                              │
│           │ 扫描完成                      │
│           ▼                              │
│       PICKING ◄──────┐                   │
│           │          │                   │
│           │ pick成功  │                   │
│           ▼          │                   │
│       PLACING        │                   │
│           │          │                   │
│           │ place完成 │ 队列非空          │
│           ├──────────┘                   │
│           │                              │
│           │ 队列空 ──────────────────────┘
│
│       无目标
│           │
│           ▼
│       COMPLETED
│
└─────── ERROR ◄── 连续失败超过阈值
```

---

## 8. 目标管理

### 8.1 核心原则

**基于 map 位置标识物体**：
- 物体是静止的 → map 坐标位置不变
- track_id 会在跟踪丢失后重新分配 → 不可靠
- 用 map 位置（距离阈值 < 0.15m）判断是否同一物体

```
TrackedObject3D (base_link) ──TF变换──► map 位置 ──► 唯一标识
```

### 8.2 位置匹配逻辑

```python
MATCH_THRESHOLD = 0.08  # 8cm 内认为是同一物体

# 阈值选择依据:
# - 顶部相机定位误差: ±5cm → 同一物体漂移通常 < 5cm
# - 小物品直径: ~7cm (瓶子/杯子)
# - 两物品紧挨中心距: ~10cm+
# - 8cm 阈值: 能容忍误差，又能区分相邻物品

def positions_match(pos1: Point, pos2: Point) -> bool:
    """判断两个 map 位置是否指向同一物体"""
    dist = math.sqrt(
        (pos1.x - pos2.x) ** 2 +
        (pos1.y - pos2.y) ** 2
    )
    return dist < MATCH_THRESHOLD
```

### 8.3 TargetPool 实现

```python
@dataclass
class TargetRecord:
    """目标记录（基于 map 位置）"""
    position_map: Point       # map 坐标系位置
    category: str             # 类别
    attempt_count: int = 0    # 尝试次数
    picked: bool = False      # 是否已抓取
    last_seen: float = 0.0    # 最后一次看到的时间戳


class TargetPool:
    """目标池 - 基于 map 位置管理，而非 track_id"""

    MATCH_THRESHOLD = 0.08    # 位置匹配阈值 (m)，8cm
    MAX_ATTEMPTS = 3          # 最大尝试次数

    def __init__(self, tf_buffer: tf2_ros.Buffer):
        self.tf_buffer = tf_buffer
        self.targets: List[TargetRecord] = []

    def update_from_tracker(self, tracked_objects: List[TrackedObject3D]) -> None:
        """从跟踪结果更新目标池"""
        now = time.time()

        for obj in tracked_objects:
            # 1. base_link → map 坐标变换
            pos_map = self._transform_to_map(obj.position)
            if pos_map is None:
                continue

            # 2. 查找是否已存在
            existing = self._find_by_position(pos_map)

            if existing:
                # 更新已有记录
                existing.last_seen = now
                existing.category = obj.category  # 可能更新类别
            else:
                # 添加新目标
                self.targets.append(TargetRecord(
                    position_map=pos_map,
                    category=obj.category,
                    last_seen=now
                ))

    def get_nav_target(self) -> Optional[TargetRecord]:
        """选择导航目标（最近的可抓物品）"""
        candidates = [
            t for t in self.targets
            if not t.picked
            and t.attempt_count < self.MAX_ATTEMPTS
        ]
        if not candidates:
            return None

        # 获取机器人当前位置
        robot_pos = self._get_robot_position()
        if robot_pos is None:
            return candidates[0]  # fallback

        # 选最近的
        return min(candidates, key=lambda t: self._distance(t.position_map, robot_pos))

    def get_workspace_targets(
        self,
        in_working_area_client
    ) -> List[TargetRecord]:
        """获取工作区内的可抓目标"""
        robot_pos = self._get_robot_position()
        candidates = []

        for t in self.targets:
            if t.picked or t.attempt_count >= self.MAX_ATTEMPTS:
                continue

            # 转换到 base_link 检查工作区
            pos_base = self._transform_to_base(t.position_map)
            if pos_base is None:
                continue

            req = InWorkingArea.Request()
            req.point_in_base = [
                pos_base.x * 1000,
                pos_base.y * 1000,
                pos_base.z * 1000
            ]
            resp = in_working_area_client.call(req)
            if resp.in_area:
                candidates.append(t)

        # 按距离排序
        if robot_pos:
            candidates.sort(key=lambda t: self._distance(t.position_map, robot_pos))

        return candidates

    def mark_picked(self, position_map: Point) -> None:
        """标记位置已抓取"""
        target = self._find_by_position(position_map)
        if target:
            target.picked = True

    def mark_failed(self, position_map: Point) -> None:
        """标记抓取失败"""
        target = self._find_by_position(position_map)
        if target:
            target.attempt_count += 1

    def _find_by_position(self, pos: Point) -> Optional[TargetRecord]:
        """根据位置查找目标"""
        for t in self.targets:
            if self._distance(t.position_map, pos) < self.MATCH_THRESHOLD:
                return t
        return None

    def _distance(self, p1: Point, p2: Point) -> float:
        """计算 2D 距离"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _transform_to_map(self, pos_base: Point) -> Optional[Point]:
        """base_link → map 坐标变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            # 应用变换...
            return transformed_point
        except:
            return None

    def _transform_to_base(self, pos_map: Point) -> Optional[Point]:
        """map → base_link 坐标变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'map', rclpy.time.Time()
            )
            # 应用变换...
            return transformed_point
        except:
            return None

    def _get_robot_position(self) -> Optional[Point]:
        """获取机器人在 map 中的位置"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            return Point(
                x=transform.transform.translation.x,
                y=transform.transform.translation.y,
                z=0.0
            )
        except:
            return None
```

### 8.4 优势

| 对比 | 旧方案 (track_id) | 新方案 (map 位置) |
|------|------------------|------------------|
| 跟踪丢失 | track_id 丢失，无法识别 | 位置不变，仍可识别 |
| 重新出现 | 新 track_id，误认为新物体 | 位置匹配，认出是同一个 |
| 机器人转头 | 所有目标"消失" | 记录保留，转回来可继续 |
| 已抓物体 | 依赖 SAM2 删除 | 明确标记 picked=True |

---

## 9. 三阶段导航

### 9.1 为什么需要三阶段

| 阶段 | 方法 | 精度 | 解决的问题 |
|------|------|------|------------|
| Stage1 | Nav2 全局规划 | ±5cm | 避障、路径规划 |
| Stage2 | PD 原地旋转 | ±3° | 朝向目标 |
| Stage3 | 深度闭环前进 | ±2cm | 精确停位 |

### 9.2 关键参数（来自 approach_navigator）

```yaml
approach_navigator:
  approach_distance: 0.45      # Stage1 停止距离（必须 > costmap inflation 35cm）
  final_distance: 0.05         # Stage3 最终距离（底盘前边缘到目标）
  align_tolerance: 0.052       # Stage2 角度容差（rad，约3°）

  # 深度传感器
  depth_topic: /camera/chassis/aligned_depth_to_color/image_raw
  depth_min_valid: 0.10        # 最小有效深度 m
  depth_max_valid: 2.0         # 最大有效深度 m

  # 机器人参数
  robot_front_offset: 0.40     # base_link 到机器人前边缘距离 m
  camera_forward_offset: 0.394 # 相机光学中心前向偏移 m
```

### 9.3 使用方式

```python
from approach_navigator import ApproachNavigator, ApproachConfig

# 创建导航器
config = ApproachConfig()
navigator = ApproachNavigator(config)

# 执行导航
result = navigator.approach_to_target(
    target_position=Point(x=1.5, y=0.3, z=0.0),  # map 坐标系
    status_callback=lambda stage, msg: print(f"[{stage.name}] {msg}")
)

if result.success:
    print(f"到达目标，最终距离: {result.final_distance:.3f}m")
else:
    print(f"导航失败: {result.error_code} - {result.error_message}")
```

---

## 10. 双相机协作

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          双相机分工                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   顶部相机 (ObjectTracker)           手部相机 (Piper Observe)            │
│   ─────────────────────────          ─────────────────────────          │
│   视角: 俯视 1-3m                    视角: 近距离 0.3-0.5m               │
│   精度: ±5cm                         精度: ±5mm                          │
│   输出: track_id, position (m)       输出: point3d_base, angle (mm/度)  │
│   用途: "去哪里" + "工作区有什么"     用途: "怎么抓"                      │
│                                                                         │
│   话题: ~/tracked_objects            服务: /piper/observe               │
│   频率: 5Hz 持续发布                 频率: 按需调用                       │
│                                                                         │
│   [PLANNING + SCANNING 阶段]         [PICKING 阶段]                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Piper 抓取流程详解

### 11.1 Observe → Pick 流程

```python
# 1. 确保在 ready 位置
go_ready_client.call(GoReady.Request(speed=30, open_gripper=False))

# 2. 观察目标
observe_req = Observe.Request(prompt="bottle", enable_cdm=True)
observe_resp = observe_client.call(observe_req)

if observe_resp.success:
    print(f"检测到: {observe_resp.category}")
    print(f"位置(base): {observe_resp.point3d_base} mm")
    print(f"角度: {observe_resp.angle_base}°")
    print(f"夹爪宽度: {observe_resp.gripper_width} mm")

# 3. 执行抓取（使用 observe 结果）
pick_goal = PiperPick.Goal(
    use_last_observe=True,      # 使用刚才的 observe 结果
    speed=30,
    lift_height=200.0,
    return_to_ready=True
)
pick_result = await pick_client.send_goal_async(pick_goal)

# 4. 放置
place_goal = PiperPlace.Goal(
    use_default_place=True,
    return_to_ready=True
)
place_result = await place_client.send_goal_async(place_goal)
```

### 11.2 Pick 内部状态机

```
CHECKING (1)     验证 observe 结果有效且未过期（30秒）
    ↓
APPROACHING (2)  移动到目标上方 100mm，调整 yaw 角
    ↓
OPENING (3)      打开夹爪（宽度 + 安全裕度）
    ↓
DESCENDING (4)   分段 MoveJ 下降（60mm/段，锁定 XY）
    ↓
CLOSING (5)      闭合夹爪
    ↓
VERIFYING (6)    验证夹爪宽度 > 5mm（确认抓住物体）
    ↓
LIFTING (7)      提升 200mm
    ↓
RETURNING (8)    返回 ready 位置
    ↓
DONE (9)         完成
```

### 11.3 错误恢复

Piper 内置黑名单系统，抓取失败的位置会被记录：
- 空间黑名单：精确位置匹配（半径 30mm）
- 物体ID黑名单：跟踪特定物体（半径 60mm）
- 黑名单过期时间：300 秒

---

## 12. 文件结构

```
src/
├── perception2/
│   ├── perception_interfaces/      # ✅ 消息定义
│   │   ├── msg/
│   │   │   ├── TrackedObject3D.msg
│   │   │   ├── TrackedObject3DArray.msg
│   │   │   └── Object3D.msg
│   │   └── srv/
│   │       ├── GraspDetect.srv
│   │       └── DetectObjects.srv
│   └── perception_nodes/           # ✅ 感知节点
│       └── object_tracker_node.py
│
├── navigation/
│   ├── approach_navigator/         # ✅ 三阶段导航
│   │   ├── navigator.py
│   │   ├── depth_sensor.py
│   │   ├── config.py
│   │   └── nav_types.py
│   └── pick_navigator/             # ❌ 待实现
│       ├── pick_navigator/
│       │   ├── __init__.py
│       │   ├── mission_manager.py
│       │   └── target_pool.py
│       ├── config/
│       │   └── pick_navigator.yaml
│       └── launch/
│           └── pick_navigator.launch.py
│
├── piper_msgs/                     # ✅ Piper 消息定义
│   ├── srv/
│   │   ├── Observe.srv
│   │   ├── GoReady.srv
│   │   ├── EnableEnhanced.srv
│   │   ├── SetPosition.srv
│   │   ├── SetGripper.srv
│   │   └── InWorkingArea.srv
│   └── action/
│       ├── PiperPick.action
│       └── PiperPlace.action
│
├── piper_driver/                   # ✅ SDK 层
│   └── scripts/
│       └── piper_api_v2.py
│
└── piper_grasp/                    # ✅ 抓取节点
    ├── scripts/
    │   ├── piper_grasp_node.py
    │   ├── coordinate_transformer.py
    │   └── grasp_blacklist.py
    └── config/
        └── piper_grasp_node.yaml
```

---

## 13. 实现计划

### Phase 1: Piper 迁移 ✅ 已完成

- [x] 创建 piper_msgs（消息定义）
- [x] 复制 piper_driver（SDK）
- [x] 迁移 PiperGraspNode（ROS2）
- [x] 单独测试抓取功能

### Phase 2: Pick Navigator（待实现）

- [ ] 创建 pick_navigator 包
- [ ] 实现 TargetPool（目标过滤 + 状态记录）
- [ ] 实现 PickMissionManager 状态机
- [ ] 集成 ApproachNavigator
- [ ] 集成 PiperGraspNode
- [ ] 集成测试

---

## 14. 配置参数

```yaml
pick_mission_manager:
  ros__parameters:
    # 目标过滤
    min_track_score: 0.3
    min_position_confidence: 0.3
    min_distance: 0.3
    max_distance: 3.0
    max_attempts: 3

    # 批量抓取
    max_picks_per_nav: 10        # 单次导航最多抓取数

    # 话题
    tracked_objects_topic: "/object_tracker_node/tracked_objects"

    # 抓取参数（透传给 Piper）
    pick_speed: 30               # 抓取速度 (1-100%)
    lift_height: 200.0           # 抬升高度 (mm)
    observe_prompt: "bottle.cup.box"  # 默认检测提示词

    # 超时
    observe_timeout: 10.0        # observe 服务超时 (秒)
    pick_timeout: 60.0           # pick 动作超时 (秒)
    place_timeout: 30.0          # place 动作超时 (秒)
    nav_timeout: 120.0           # 导航超时 (秒)

    # 错误处理
    max_consecutive_failures: 3  # 连续失败次数阈值
```

---

## 15. 坐标系与单位

### 坐标系

```
map
 └── odom
      └── base_link
           ├── top_camera_link        (顶部相机)
           ├── chassis_camera_link    (底盘相机)
           └── piper_link_base
                └── hand_camera_link  (手部相机)
```

### 单位约定

| 模块 | 长度 | 角度 |
|------|------|------|
| Nav2 / ApproachNavigator | m | rad |
| ObjectTracker (话题) | m | - |
| Piper 所有接口 | **mm** | **degrees** |
| PerceptionGrasp (服务) | m | degrees |

**注意**: PerceptionGrasp 返回米，Piper 接口使用毫米，需要转换！

---

## 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| 5.0 | 2026-02-26 | 更新 Piper 迁移完成状态，详细记录各模块 ROS2 接口 |
| 4.1 | 2025-02-26 | 加入批量抓取(1:M)和工作区扫描策略 |
| 4.0 | 2025-02-26 | 简化状态机，简化目标选取，明确实现进度 |
| 3.0 | 2025-02-26 | 整合 Piper 迁移方案 |
