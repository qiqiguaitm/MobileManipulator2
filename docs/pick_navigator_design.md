# Cleaner Manager 设计文档

> ROS2 自主机器人任务管理系统 — 当前任务: 自主拣取（Pick）
>
> 版本: 9.1
> 日期: 2026-03-27

---

## 1. 概述

### 1.1 系统目标

统一调度三大核心模块，实现机器人自主拣取室内所有可见物品：

| 模块 | 包名 | 职责 |
|------|------|------|
| **SLAM** | `slam` | 定位(HDL+FastLIO)、里程计融合、Nav2导航、TF树 |
| **Perception** | `perception` | 双相机3D感知、目标跟踪(ByteTracker3D)、抓取检测 |
| **Manipulation** | `piper_grasp` | 机械臂控制(PiperAPI)、observe/pick/place、碰撞检测 |

### 1.2 设计原则

**模块解耦**：三大模块（SLAM、Perception、Manipulation）均已独立测试完毕，是稳定的黑盒。CleanerManager 作为**薄编排层**，仅通过各模块的 ROS2 接口（Topic/Service/Action）进行协调，不深入了解各模块内部实现细节。

**可拓展性**：包名 `cleaner_manager` 不绑定特定任务类型。当前实现自主拣取（Pick）任务，未来可扩展巡逻、探索等任务类型，复用相同的模块接口和基础设施（TargetPool、状态反馈、健康检查等）。

### 1.3 核心流程

```
感知发现目标 → 导航到目标附近 → 收臂安全导航 → 展臂扫描工作区 → 批量抓取 → 循环
```

### 1.4 关键设计决策

| 问题 | 解决方案 | 依赖模块 |
|------|----------|----------|
| 导航精度不足 | 三阶段导航（Nav2 → 对齐 → 深度闭环接近） | SLAM |
| 导航目标可能超出臂工作区 | 导航(base_link)和臂工作区(piper_link_base)是不同坐标系；SCANNING 阶段通过 `/piper/in_working_area` 服务判断可达性 | Manager + Manipulation |
| 抓取需精确位姿 | 三相机协作（顶部粗定位 ±5cm + 手部精定位 ±5mm） | Perception |
| 已抓物体删除 | 双机制：ByteTracker3D 丢失 + TargetPool 标记 picked | Perception + Manager |
| 减少导航次数 | 批量抓取（一次导航抓取工作区内所有物品） | Manager |
| 导航中机械臂安全 | NAVIGATING 前收臂到安全位，SCANNING 前展开到观察位 | Manager |
| 定位漂移 | HDL全局纠正 + FastLIO+轮式里程计融合(50Hz) | SLAM |

---

## 2. 系统架构

### 2.1 模块分层

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERCEPTION (✅ 已实现)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MultiCameraPerceptionNode          PerceptionGraspNode                     │
│  (Top+Chassis 双相机融合)            (手部相机 按需)                          │
│  ├─ SAM3 检测(自然语言)             ├─ SAM3 + GraspAnything               │
│  ├─ IntelligentFusion 融合           ├─ CDM 深度优化                        │
│  ├─ ByteTracker3D 跟踪              ├─ 精确位置 ±5mm                        │
│  └─ 3D 位置 ±5cm                    └─ 抓取角度 + 夹爪宽度                  │
│                                                                             │
│  ObjectTrackerNode (独立流水线)                                              │
│  ├─ 订阅 MultiCamera 检测作为初始化种子                                      │
│  ├─ 自有 Top 相机 RGB+Depth 订阅                                            │
│  ├─ SAM2 在线跟踪 + 独立深度测量                                             │
│  └─ 输出带 track_id 的持续跟踪结果                                           │
│                                                                             │
└──────────┬─────────────────────────────┬────────────────────────────────────┘
           │                             │
           ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CLEARNER MANAGER (✅ 已实现)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  状态机: IDLE → PLANNING → STOWING → NAVIGATING → DEPLOYING →              │
│          SCANNING → PICKING → 循环                                          │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │  TargetPool  │   │ Approach     │   │ Piper        │                    │
│  │  (目标管理)   │   │ Navigator    │   │ GraspNode    │                    │
│  │  (✅已实现)   │   │ (✅已实现)    │   │ (✅已实现)    │                    │
│  └──────────────┘   └──────────────┘   └──────────────┘                    │
│                                                                             │
└──────────┬─────────────────────────────┬────────────────────────────────────┘
           │                             │
           ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SLAM (✅ 已实现)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  里程计层                        定位层                   导航层              │
│  ├─ FastLIO (10Hz)              ├─ HDL NDT (10Hz)       ├─ Nav2             │
│  ├─ 轮式里程计 (50Hz)           ├─ map→odom TF          │  ├─ Smac2D 规划   │
│  └─ OdomFusion (50Hz合成)       └─ TF平滑发布(100Hz)    │  ├─ MPPI 控制      │
│                                                          │  └─ 行为树        │
│  TF树: map → odom → base_link → [传感器/机械臂]          │                   │
│                                                                             │
└──────────┬─────────────────────────────┬────────────────────────────────────┘
           │                             │
           ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HARDWARE DRIVERS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  rslidar_sdk       hipnuc_imu       tracer_base       PiperSDK (CAN)       │
│  (3D LiDAR)        (IMU)            (底盘)             (机械臂)              │
│  camera_driver (RealSense D455 × 3: top/chassis/hand)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 感知 Topic 链路（精确映射）

```
MultiCameraPerceptionNode
  发布 → ~/fused/objects_3d (Object3DArray, base_link 坐标, m)
           │
           │ [launch 中 remap 为 /perception_3d/objects]  ← 必须在 launch 配置!
           ▼
ObjectTrackerNode
  订阅 ← /perception_3d/objects (检测种子)
  订阅 ← /top_camera/color/image_raw (自有 RGB)
  订阅 ← /top_camera/aligned_depth_to_color/image_raw (自有 Depth)
  发布 → ~/tracked_objects (TrackedObject3DArray, base_link 坐标, m)
           │
           ▼
CleanerManager
  订阅 ← /object_tracker_node/tracked_objects  ← 唯一输入源
```

**注意**：ObjectTrackerNode 是独立的感知流水线，它用 MultiCamera 的检测结果作为 SAM2 跟踪的初始化种子，但 RGB/Depth 读取和 3D 深度测量都是独立运行的（仅用 Top 相机）。所以同一物体经过两条流水线的 3D 位置可能略有差异（±2cm）。

---

## 3. 模块实现状态

| 模块 | 状态 | 包名 | 说明 |
|------|------|------|------|
| FastLIO + OdomFusion | ✅ 已实现 | `slam` | 里程计融合，50Hz 平滑输出 |
| HDL Localization | ✅ 已实现 | `slam` | NDT 全局定位 + TF平滑发布 |
| Nav2 Stack | ✅ 已实现 | `slam` | Smac2D + MPPI，完整导航栈 |
| MultiCameraPerceptionNode | ✅ 已实现 | `perception` | 双相机融合 + ByteTracker3D |
| ObjectTrackerNode | ✅ 已实现 | `perception` | 独立 Top 相机 SAM2 跟踪 |
| PerceptionGraspNode | ✅ 已实现 | `perception` | 手部相机抓取检测 |
| ApproachNavigator | ✅ 已实现 | `approach_navigator` | 三阶段精确导航 |
| PiperGraspNode | ✅ 已实现 | `piper_grasp` | 机械臂控制 + observe/pick/place |
| PiperAPI V2 | ✅ 已实现 | `piper_grasp` | CAN 总线通信，无ROS依赖 |
| **CleanerManager** | ✅ 已实现 | `cleaner_manager` | 任务编排 + 状态机 + 目标管理 |

### 依赖关系

```
Perception
├─ MultiCameraPerceptionNode ──remap──► ObjectTrackerNode ──┐
│                                                            │
SLAM                                                        ├──► CleanerManager
├─ Nav2 (via ApproachNavigator) ────────────────────────────┤
├─ TF树 (map→odom→base_link) ──────────────────────────────┤
│                                                            │
Manipulation                                                │
└─ PiperGraspNode ─────────────────────────────────────────┘

### 依赖关系图（全部已实现）

所有模块均已开发完毕并通过独立测试。Cleaner Manager 作为薄编排层将三大模块串联为完整的自主拣取系统。
```

---

## 4. SLAM 模块

### 4.1 数据流

```
传感器 (rslidar, IMU, 轮式里程计)
    ↓
[FastLIO] → /fastlio/odom (camera_init→body, 10Hz)
    ↓
[OdomFusion] → /odom/fused (odom→base_link, 50Hz)
    ↓
[HDL Localization] → map→base_link (NDT匹配, 10Hz)
    ↓
[hdl_odom_to_tf + tf_republisher] → /tf: map→odom (平滑, 100Hz)
    ↓
Nav2: 路径规划(Smac2D) + 轨迹跟踪(MPPI) + 行为树
    ↓
/cmd_vel → 底盘控制
```

### 4.2 TF树

```
map                                    (全局，HDL定位)
 └── odom                              (局部，里程计)
      └── base_link                    (机器人底盘)
           ├── rslidar                 (3D LiDAR)
           ├── camera/top_optical_frame
           ├── camera/chassis_optical_frame
           └── piper_link_base
                └── camera/hand_optical_frame
```

### 4.3 Nav2 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 全局规划器 | Smac2D (A*) | 适合2D地图 |
| 局部控制器 | MPPI | 模型预测路径积分控制 |
| 底盘尺寸 | 0.809m × 0.62m | 矩形足迹 |
| 最大线速度 | 0.3 m/s | |
| 最大角速度 | 0.5 rad/s | |
| 膨胀半径 | 0.20m | costmap |

---

## 5. Perception 模块

### 5.1 架构

```
┌───────────────────────────────────────────────────────────────────┐
│              在线服务层（HTTP / GPU 推理）                          │
│  SAM3 (自然语言检测) │ SAM3 (分割) │ GraspAnything │ CDM (深度优化) │
└───────────────────────┬───────────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────────┐
│              核心算法层                                             │
│  ScenePerceptionCore │ IntelligentFusion │ ByteTracker3D │ 坐标变换 │
└───────────────────────┬───────────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────────┐
│              ROS2 节点层                                           │
│  MultiCameraPerceptionNode    (双相机融合, ~/fused/objects_3d)     │
│  ObjectTrackerNode            (独立跟踪, ~/tracked_objects)       │
│  PerceptionGraspNode          (手部抓取, ~/detect 服务)           │
└───────────────────────────────────────────────────────────────────┘
```

### 5.2 MultiCameraPerceptionNode

**发布话题** (Object3DArray):
- `~/top/objects_3d` — Top相机独立结果
- `~/chassis/objects_3d` — Chassis相机独立结果
- `~/fused/objects_3d` — 融合+去重结果（**需在 launch 中 remap 为 `/perception_3d/objects`**）

### 5.3 ObjectTrackerNode（CleanerManager 唯一感知输入）

**注意**：ObjectTrackerNode 是**独立的感知流水线**，不是 MultiCamera 的简单包装。它：
1. 接收 MultiCamera 的 `/perception_3d/objects` 作为 SAM2 跟踪的初始化种子
2. 订阅自己的 Top 相机 RGB/Depth 话题
3. 运行独立的深度测量流水线
4. 通过 SAM2 在线跟踪维持持续 track_id

**发布话题**:
```yaml
~/tracked_objects:
  类型: TrackedObject3DArray
  频率: 5Hz
  内容:
    - track_id: int32              # 持续不变的跟踪ID
    - category: string             # 类别名称
    - position: Point              # base_link 坐标系，单位 m
    - distance: float64            # 到 base_link 原点距离，单位 m
    - track_score: float64         # 跟踪置信度 (0-1)
    - position_confidence: float64 # 3D位置置信度 (0-1)
```

### 5.4 PerceptionGraspNode（手部相机抓取检测）

**服务**:
```yaml
~/detect:
  类型: GraspDetect.srv
  请求:
    prompt: string              # 检测提示词，如 "bottle.cup"
    enable_cdm: bool            # 启用 CDM 深度优化
  响应:
    success: bool
    point3d: float64[3]         # [x,y,z] 单位 m（手部相机光学坐标系）
    width3d: float64            # 夹爪宽度，单位 m
    angle: float64              # 抓取角度，单位 度
    category: string
    score: float64
```

---

## 6. Manipulation 模块（PiperGraspNode）

### 6.1 服务接口

```yaml
/piper/enable:        # EnableEnhanced.srv — 使能/关闭/清错/重连
/piper/go_ready:      # GoReady.srv — 到观察位
/piper/observe:       # Observe.srv — 调用感知 + 坐标变换
/piper/in_working_area: # InWorkingArea.srv — 工作区判断
/piper/get_status:    # GetStatus.srv — 状态查询
/piper/set_position:  # SetPosition.srv — 关节运动
/piper/set_gripper:   # SetGripper.srv — 夹爪控制
```

### 6.2 动作接口

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
    state: uint8              # 1-9 状态码
    step_name: string         # CHECKING → ... → DONE
    progress: float32         # 0.0-1.0
  结果:
    success: bool
    category: string
    grasp_position: float32[3]
    execution_time_ms: float32
    error_message: string

/piper/place:
  类型: PiperPlace.action
  目标:
    use_default_place: bool
    place_position: float32[6]
    speed: int32
    return_to_ready: bool
  结果:
    success: bool
    final_position: float32[6]
```

### 6.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 观察位 | x=317, y=15, z=248 mm | 手部相机俯视地面 |
| 放置位 | x=-86, y=-150, z=300 mm | 默认放置 |
| 夹爪偏移 | 135.03 mm | 法兰到夹爪中心 |
| 夹爪最大开度 | 90 mm | 物理限制 |
| 下降方式 | 分段 MoveJ | 60mm/段，XY锁定 |
| 黑名单(空间层) | 半径 30mm, 过期 600s | 精确位置匹配 |
| 黑名单(物体层) | 半径 60mm, 过期 300s | 类别+位置匹配 |

---

## 7. ApproachNavigator（三阶段导航）

### 7.1 核心接口

```python
class ApproachNavigator(Node):
    def approach_to_target(
        self,
        target_position: Point,           # map 坐标系
        status_callback: Callable = None
    ) -> ApproachResult

    def cancel(self) -> None

@dataclass
class ApproachResult:
    success: bool
    error_code: str      # NAV_FAILED, ALIGN_FAILED, APPROACH_FAILED, NAV_CANCELLED
    error_message: str
    final_distance: float
```

### 7.2 三阶段流程

| 阶段 | 方法 | 精度 | 说明 |
|------|------|------|------|
| Stage1 | Nav2 goToPose | ±5cm | 全局规划，停在距目标 0.45m |
| Stage2 | PD 原地旋转 | ±5° | 对齐目标方向 |
| Stage3 | 深度闭环前进 | ±2cm | 底盘前边缘距最近障碍 0.08m |

### 7.3 关键参数（来自实际配置）

```yaml
approach_navigator:
  approach_distance: 0.45         # Stage1 Nav2停止距离 (m)
  final_approach_distance: 0.08   # Stage3 最终距离 (m)，前边缘到最近障碍
  align_tolerance: 0.087          # Stage2 角度容差 (rad, 约5°)
  robot_front_offset: 0.434       # base_link到前边缘 (m)
  camera_forward_offset: 0.394    # 相机光学中心前向偏移 (m)
  depth_topic: /camera/chassis/aligned_depth_to_color/image_raw
```

### 7.4 导航与工作区的坐标系关系

**关键事实：导航坐标系和机械臂工作区坐标系是不同的坐标系。**

```
导航坐标系:
  ApproachNavigator 的参数（robot_front_offset, final_approach_distance）
  基于 base_link 坐标系

机械臂工作区坐标系:
  PiperGraspNode 的 working_area 参数（offset_xmin/xmax/ymin/ymax/zmin/zmax）
  基于 piper_link_base (arm_base_link) 坐标系
  两者之间存在固定 TF 变换

不能直接比较两个坐标系中的数值!
```

**模块解耦原则**：CleanerManager 不需要理解机械臂的几何细节。导航模块负责"把机器人送到目标附近"，机械臂模块通过 `/piper/in_working_area` 服务自行判断目标是否可达（该服务内部处理坐标系转换）。

**设计决策**：
- 导航只负责"到附近"，不保证目标在工作区内
- SCANNING 阶段调用 `/piper/in_working_area` 筛选可达物体（服务内部处理 base_link → piper_link_base 的坐标变换）
- 若工作区内有物体则批量抓取，若无目标则回 PLANNING 选择下一个导航目标

---

## 8. Cleaner Manager 状态机（Pick 任务）

### 8.1 状态定义

```python
class PickState(IntEnum):
    IDLE = 0          # 等待启动
    PLANNING = 1      # 选择导航目标
    STOWING = 2       # 收臂到安全位（导航前）
    NAVIGATING = 3    # 底盘导航（三阶段）
    DEPLOYING = 4     # 展臂到观察位（导航后）
    SCANNING = 5      # 工作区扫描
    PICKING = 6       # 批量抓取
    PLACING = 7       # 放置物品
    ERROR = 8         # 出错（可恢复）
    COMPLETED = 9     # 全部完成
```

### 8.2 状态转移

```
         start()
            │
            ▼
┌───────► IDLE
│           │ /cleaner_manager/start 服务
│           ▼
│       PLANNING ◄──────────────────────────┐
│           │                               │
│           │ 有目标           无工作区目标   │
│           ▼                               │
│       STOWING                             │
│           │ go_ready(close gripper)        │
│           ▼                               │
│       NAVIGATING                          │
│           │ ApproachResult.success        │
│           ▼                               │
│       DEPLOYING                           │
│           │ go_ready 到观察位              │
│           ▼                               │
│       SCANNING ───────────────────────────┘
│           │                     (工作区内无目标 → 回 PLANNING)
│           │ 有工作区目标
│           ▼
│       PICKING ◄──────┐
│           │          │
│           │ pick成功  │
│           ▼          │
│       PLACING        │
│           │          │
│           │ place完成 │ 队列非空
│           ├──────────┘
│           │
│           │ 队列空 ──► PLANNING
│
│       无目标
│           │
│           ▼
│       COMPLETED ◄── 全局无可抓目标
│
│       ERROR ◄── 连续失败超过阈值
│           │
│           │ cooldown (5s) + 自动恢复
│           ▼
└─────── PLANNING (重试)
│
│       /cleaner_manager/abort 服务 (任何状态均可触发)
│           │
│           ├─ cancel ApproachNavigator
│           ├─ cancel Piper pick/place action
│           ├─ 收臂到安全位
│           └─ → IDLE
```

### 8.3 关键设计：导航过程中的机械臂安全

| 阶段 | 臂状态 | 原因 |
|------|--------|------|
| STOWING | 收到 zero 位或安全位 | Nav2 recovery behavior (backup/spin) 可能让伸出的臂撞到障碍物 |
| NAVIGATING | 保持收拢 | 底盘在移动，臂不能伸出 |
| DEPLOYING | go_ready 到观察位 | 导航完成后展臂准备扫描 |
| PICKING/PLACING | 主动控制 | PiperGraspNode 管理 |

### 8.4 进程与线程模型

> 源码: `src/cleaner_manager/src/cleaner_manager_node.py`

```
┌─────────────── 进程: cleaner_manager_node ───────────────┐
│                                                         │
│  MultiThreadedExecutor (4 threads)                      │
│  ├─ CleanerManagerNode                                    │
│  │   ├─ Subscription: tracked_objects (5Hz)             │
│  │   ├─ Timer: status publisher (1Hz)                   │
│  │   ├─ Service: ~/start, ~/abort                       │
│  │   ├─ Service clients: observe, go_ready,             │
│  │   │   in_working_area, get_status                    │
│  │   ├─ Action clients: pick, place                     │
│  │   └─ TF buffer + listener                            │
│  └─ ApproachNavigator (独立 Node，共享 executor)        │
│                                                         │
│  Worker Thread (daemon):                                │
│  └─ PickStateMachine.run(abort_event)                   │
│      阻塞式状态机主循环                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**线程分工**:

| 线程 | 职责 |
|------|------|
| Executor threads (1-4) | ROS2 回调: timer, subscription, service, ApproachNavigator 回调 |
| Worker thread (daemon) | 状态机主循环: 阻塞操作 (nav, pick, place) 在此运行 |

**Service/Action 异步调用模式**:

Worker thread 不能调用 `spin_until_future_complete()`（只有 executor 线程可以），因此使用 `call_async()` + 轮询：

```python
def _wait_future(self, future, timeout, abort_event):
    deadline = time.time() + timeout
    while not future.done():
        if abort_event and abort_event.is_set():
            return None
        if time.time() > deadline:
            return None
        time.sleep(0.05)
    return future.result()
```

Executor 线程持续 spin 处理 future 回调，所以 future 会正常 resolve。

**Abort 机制**:

1. `~/abort` service callback（executor 线程）→ 设置 `threading.Event`
2. 同时调 `navigator.cancel()` + cancel 活跃的 pick/place `goal_handle`
3. Worker thread 下次检查 `abort_event.is_set()` 时退出循环
4. 退出前 best-effort stow arm → 状态回 IDLE

### 8.5 状态处理器表（handler 驱动，消除 switch/case）

> 源码: `src/cleaner_manager/cleaner_manager/pick_state_machine.py`

```python
self._handlers = {
    PickState.PLANNING:   self._do_planning,
    PickState.STOWING:    self._do_stowing,
    PickState.NAVIGATING: self._do_navigating,
    PickState.DEPLOYING:  self._do_deploying,
    PickState.SCANNING:   self._do_scanning,
    PickState.PICKING:    self._do_picking,
    PickState.PLACING:    self._do_placing,
    PickState.ERROR:      self._do_error,
}
```

每个 handler 签名统一：`(abort_event) -> PickState`，返回下一状态。主循环：

```python
while not abort_event.is_set():
    state = self.state
    if state in (PickState.IDLE, PickState.COMPLETED):
        break
    next_state = self._handlers[state](abort_event)
    self._set_state(next_state)
```

### 8.6 状态转移详表

| 状态 | 动作 | 成功 → | 失败 → |
|------|------|--------|--------|
| PLANNING | `pool.get_nav_target()` | STOWING | COMPLETED (无目标) |
| STOWING | `/piper/go_ready(open_gripper=False)` 收臂 | NAVIGATING | ERROR |
| NAVIGATING | `navigator.approach_to_target()` + pool.pause | DEPLOYING | ERROR (连续失败≥3) 或 PLANNING (单次失败) |
| DEPLOYING | `/piper/go_ready(open_gripper=True)` 展臂 | SCANNING | ERROR |
| SCANNING | pool.resume + 等待稳定 + `pool.get_workspace_targets()` | PICKING | PLANNING (无工作区目标) |
| PICKING | `/piper/observe` → `/piper/pick` | PLACING | ERROR (连续失败≥3) 或 PICKING (跳过当前) |
| PLACING | `/piper/place` → mark_picked/failed | PICKING (队列非空) 或 PLANNING (队列空) | PICKING (继续下一个) |
| ERROR | cooldown 5s → 重试 | PLANNING | COMPLETED (重试≥3次) |

---

## 9. 完整拣取流程（Pick 任务）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      批量抓取流程 (1:M)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] PLANNING - 选择导航目标                                            │
│      ├─ 订阅: /object_tracker_node/tracked_objects (唯一输入)           │
│      ├─ base_link→map 坐标变换，写入 TargetPool                        │
│      ├─ 过滤: track_score≥0.3, position_confidence≥0.3, 0.3m<距离<3m  │
│      ├─ 排除: picked=True, attempt_count≥3                            │
│      └─ 选择: 距机器人最近的作为导航目标                                 │
│                                                                         │
│             │
│                                                                         │
│  [2] NAVIGATING - 三阶段导航                                            │
│      ├─ 暂停 TargetPool 更新（避免导航中位置漂移污染）                    │
│      ├─ 调用: ApproachNavigator.approach_to_target(target.position_map)│
│      ├─ Stage1: Nav2 → 距目标 0.45m                                   │
│      ├─ Stage2: PD 旋转 → 朝向目标 ±5°                                │
│      └─ Stage3: 深度闭环前进 → 前边缘距障碍 0.08m                      │
│                                                                         │
│  [3] DEPLOYING - 展臂                                                  │
│      └─ 调用: /piper/go_ready（到观察位 x=317, y=15, z=248）           │
│                                                                         │
│  [4] SCANNING - 工作区扫描                                              │
│      ├─ 恢复 TargetPool 更新                                           │
│      ├─ 等待感知稳定（连续 3 帧物体数量不变，最长 3s 超时）              │
│      ├─ 从 /object_tracker_node/tracked_objects 获取当前目标            │
│      ├─ base_link→map 变换后更新 TargetPool                            │
│      ├─ 调用: /piper/in_working_area 筛选工作区内物品                   │
│      ├─ 按距离排序，生成抓取队列                                         │
│      └─ 若工作区无目标 → 回 PLANNING（选下一个导航目标）                 │
│                                                                         │
│  [5] PICKING - 批量抓取（循环）                                         │
│      ├─ 从队列取下一个目标的 category                                   │
│      ├─ 调用: /piper/observe (prompt=category)                         │
│      │   └─ 返回: point3d_base, angle_base, gripper_width (mm)         │
│      ├─ 调用: /piper/pick (use_last_observe=true)                      │
│      │   └─ 状态: CHECKING→APPROACHING→OPENING→DESCENDING→             │
│      │            CLOSING→VERIFYING→LIFTING→RETURNING                  │
│      ├─ 调用: /piper/place (use_default_place=true)                    │
│      ├─ 成功: target_pool.mark_picked(position_map)                    │
│      ├─ 失败: target_pool.mark_failed(position_map)                    │
│      └─ 队列非空则继续，否则回到 PLANNING
|  [6] STOWING - 收臂                                                     │
│      └─ 调用: /piper/go_ready(speed=30, open_gripper=False)             │
│                                                                         │
│  [7] 循环 [1]-[6]，直到没有可抓目标 → COMPLETED                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 目标管理（TargetPool）

### 10.1 核心原则

**基于 map 位置标识物体**：
- 物体静止 → map 坐标不变
- track_id 跟踪丢失后重新分配 → 不可靠
- 用 map 位置匹配（阈值 8cm）判断同一物体

**更新时机**：
- PLANNING 阶段：持续更新
- NAVIGATING 阶段：**暂停更新**（机器人移动中 TF 变化 + 感知误差会导致同一物体的 map 位置抖动，超过 8cm 阈值会被误识为两个目标）
- SCANNING 阶段：恢复更新

### 10.2 位置匹配

```python
MATCH_THRESHOLD = 0.08  # 8cm

# 阈值依据:
# - 顶部相机定位误差: ±5cm → 同一物体漂移 < 5cm
# - 小物品直径: ~7cm (瓶子/杯子)
# - 两物品紧挨中心距: ~10cm+
# - 8cm: 容忍误差，区分相邻物品
```

### 10.3 TargetPool 实现

> 源码: `src/cleaner_manager/cleaner_manager/target_pool.py`

```python
class TargetStatus(IntEnum):
    ACTIVE = 0
    PICKED = 1
    FAILED = 2

@dataclass
class TargetRecord:
    track_id: int
    category: str
    position_map: Point          # map 坐标系，单位 m
    track_score: float
    position_confidence: float
    status: TargetStatus = TargetStatus.ACTIVE
    attempts: int = 0
    last_seen: float = field(default_factory=time.time)
    fail_reason: str = ""


class TargetPool:
    """线程安全的目标池。单把 threading.Lock 保护 _targets 和 _update_enabled。"""

    MATCH_THRESHOLD = 0.08  # 8cm

    def __init__(self, tf_buffer, logger, min_track_score, min_position_confidence,
                 min_distance, max_distance, max_attempts):
        self._lock = threading.Lock()
        self._targets: List[TargetRecord] = []
        self._update_enabled = True

    def update_from_tracker(self, msg: TrackedObject3DArray) -> None:
        """Subscription callback。TF lookup 在锁外执行，更新在锁内。"""
        # 1. 检查 update_enabled（锁内）
        # 2. lookup_transform('map', 'base_link', msg.header.stamp)（锁外）
        # 3. 遍历 msg.objects，过滤 + 匹配 + 指数移动平均更新位置（锁内）

    def get_nav_target(self, robot_pos_map: Point) -> Optional[TargetRecord]:
        """返回距机器人最近的 ACTIVE 且未超限目标。"""

    def get_workspace_targets(self, in_working_area_fn, robot_pos_map) -> List[TargetRecord]:
        """接受回调函数判断工作区，不直接持有 ROS client。按距离排序。"""

    def pause(self) / resume(self):
        """NAVIGATING 期间暂停更新，SCANNING 恢复。"""

    def mark_picked(self, pos) / mark_failed(self, pos, reason):
        """基于最近距离匹配，更新状态或累计 attempts。"""
```

**关键改进（相对设计稿的变化）**：
- 使用 `TargetStatus` 枚举替代 `picked: bool`，状态更清晰
- 位置更新采用指数移动平均（0.7/0.3 权重），减少单帧噪声
- `get_workspace_targets()` 接受回调函数而非直接持有 service client，保持 TargetPool 为纯逻辑层
- TF 查询在锁外执行，避免锁内阻塞

---

## 11. 双相机协作

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          三相机分工                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Top+Chassis 双相机融合              手部相机 (Piper Observe)            │
│  ───────────────────────────         ─────────────────────────          │
│  节点: MultiCamera → ObjectTracker  节点: PerceptionGraspNode           │
│  视角: 俯视 0.5-3m                  视角: 近距离 0.15-0.5m              │
│  精度: ±5cm (融合后)                精度: ±5mm                          │
│  输出: track_id, position (m)       输出: point3d_base, angle (mm/度)  │
│  用途: "去哪里" + "工作区有什么"     用途: "怎么抓"                      │
│                                                                         │
│  话题: ~/tracked_objects            服务: /piper/observe               │
│  频率: 5Hz 持续发布                 频率: 按需调用                       │
│                                                                         │
│  [PLANNING + SCANNING 阶段]         [PICKING 阶段]                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. 工作区定义

### 12.1 机械臂工作范围（实际配置值）

基于 `piper_grasp_node.yaml` 中的 `working_area` 配置。

**注意**：以下所有数值均在 **piper_link_base（arm_base_link）坐标系** 中，不是 base_link 坐标系。两者之间存在固定 TF 变换。

```
┌─────────────────────────────────────────────────────────────┐
│           工作区 (侧视图, piper_link_base 坐标系)              │
│                                                             │
│    Z (mm)                                                   │
│    ▲     观察位 (z=248mm)                                   │
│  248├─────●────────────────                                 │
│    │      ↓                                                 │
│    │      ↓ 手部相机视野                                     │
│    │      ↓                                                 │
│    0├─────┼────────────────  地面                           │
│    │                                                        │
│ -150├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  offset_zmax (抓取区上界)       │
│    │      │ 可抓取区       │                                │
│ -400├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  offset_zmin (抓取区下界)       │
│    │                                                        │
│    └──────┼──────┼──────┼───────► X (mm)                   │
│         267    317    467                                   │
│        (X_min) (ready) (X_max)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│           工作区 (俯视图, piper_link_base 坐标系)              │
│                                                             │
│                    X: 267 ~ 467mm (前方)                    │
│                    (ready.x±offset)                         │
│                                                             │
│                      ┌───────────┐                          │
│                      │           │                          │
│      Y: -150mm ──────│  可抓取    │────── Y: +150mm         │
│                      │   区域     │                          │
│                      │           │                          │
│                      └───────────┘                          │
│                           ▲                                 │
│                    piper_link_base                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 关键参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 观察位 | x=317, y=15, z=248 mm | `positions.ready` |
| X 范围 | 267 ~ 467 mm | `ready.x + [offset_xmin, offset_xmax]` (piper_link_base 坐标系) |
| Y 范围 | -150 ~ +150 mm | `[offset_ymin, offset_ymax]` (piper_link_base 坐标系) |
| Z 范围 | -400 ~ -150 mm | `[offset_zmin, offset_zmax]` (piper_link_base 坐标系) |
| Yaw 范围 | 90° ~ 270° | `[yaw_min, yaw_max]` |
| 夹爪偏移 | 135.03 mm | `gripper.offset_mm` |
| 夹爪最大开度 | 90 mm | `gripper.max_mm` |
| 提升高度 | 200 mm | `grasp.lift_height` |

### 12.3 坐标系与模块解耦说明

**核心原则**：导航参数（base_link 坐标系）和机械臂工作区参数（piper_link_base 坐标系）属于不同坐标系，不能直接做数值比较。

| 参数 | 坐标系 | 所属模块 |
|------|--------|----------|
| `robot_front_offset` (434mm) | base_link | ApproachNavigator |
| `final_approach_distance` (80mm) | base_link | ApproachNavigator |
| `working_area` X/Y/Z 范围 | piper_link_base | PiperGraspNode |

CleanerManager 作为编排层，遵循以下解耦原则：
- **不关心**两个坐标系之间的 TF 变换细节
- **不关心**机械臂的具体几何参数
- 通过调用 `/piper/in_working_area` 服务判断目标可达性（服务内部完成坐标变换）
- 导航目标不一定落在工作区内，**SCANNING 阶段的服务调用过滤是必需的**

---

## 13. Pick 内部流水线（PiperGraspNode）

### 13.1 Observe → Pick → Place

```python
# 1. 确保在 ready 位置
go_ready_client.call(GoReady.Request(speed=30, open_gripper=False))

# 2. 观察目标
observe_resp = observe_client.call(
    Observe.Request(prompt="bottle", enable_cdm=True)
)

# 3. 执行抓取
pick_result = await pick_client.send_goal_async(
    PiperPick.Goal(use_last_observe=True, speed=30,
                   lift_height=200.0, return_to_ready=True)
)

# 4. 放置
place_result = await place_client.send_goal_async(
    PiperPlace.Goal(use_default_place=True, return_to_ready=True)
)
```

### 13.2 Pick 状态流水线

```
CHECKING (1)     验证 observe 结果有效且未过期（30秒），检查黑名单
    ↓
APPROACHING (2)  MoveJ 到目标上方 100mm，调整 yaw，pitch=10°
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

---

## 14. 单位转换规则

### 14.1 各模块单位

| 模块 | 长度 | 角度 |
|------|------|------|
| SLAM / Nav2 / ApproachNavigator | m | rad |
| Perception 话题 (tracked_objects) | m | — |
| Perception 服务 (GraspDetect) | m | degrees |
| Piper 所有接口 | **mm** | **degrees** |

### 14.2 转换边界

**CleanerManager 内部统一使用 SI 单位（m）**，仅在一个地方做 m→mm 转换：

> 源码: `pick_state_machine.py:_check_in_working_area()`

```python
# 唯一的 m→mm 转换点
def _check_in_working_area(self, pos_map: Point) -> bool:
    # 1. TF: map → base_link
    pos_base = _transform_point(pos_map, tf_map_to_base)
    # 2. m → mm（唯一转换点）
    req.point_in_base = [pos_base.x * 1000.0, pos_base.y * 1000.0, pos_base.z * 1000.0]
    # 3. 调用 /piper/in_working_area
    return resp.in_area
```

`lift_height` (config, 200.0mm) 和 `pick_speed` 直接透传给 Piper action，不经过转换。

### 14.3 坐标系

```
map                                    (全局，HDL定位)
 └── odom                              (局部，里程计)
      └── base_link                    (机器人底盘 — 导航参数在此坐标系)
           ├── rslidar                 (3D LiDAR)
           ├── camera/top_optical_frame
           ├── camera/chassis_optical_frame
           └── piper_link_base         (机械臂基座 — 工作区参数在此坐标系)
                └── camera/hand_optical_frame
```

**CleanerManager 中的坐标系使用**：
- TargetPool 中物体位置存储为 `map` 坐标（静态参考）
- 调用 `/piper/in_working_area` 时传入 `base_link` 坐标（m→mm 转换后），服务内部处理到 `piper_link_base` 的变换
- 不需要在 CleanerManager 中手动做 base_link ↔ piper_link_base 变换

---

## 15. 监控与诊断

### 15.1 状态反馈话题

> 消息定义: `src/cleaner_manager/msg/CleanerManagerStatus.msg`

```yaml
~/status:                           # 实际话题: /cleaner_manager_node/status
  类型: cleaner_manager/CleanerManagerStatus
  频率: 1Hz
  内容:
    header: std_msgs/Header
    state: uint8                    # 当前状态枚举 (0-9)
    state_name: string              # "IDLE", "PLANNING", "NAVIGATING", ...
    targets_total: uint32           # TargetPool 中总目标数
    targets_picked: uint32          # 已抓取数
    targets_failed: uint32          # 失败数
    targets_remaining: uint32       # 剩余可抓数 (ACTIVE 状态)
    current_target_category: string # 当前目标类别
    consecutive_failures: uint8     # max(nav_failures, pick_failures)
    last_nav_time_ms: float32       # 最近导航耗时
    last_pick_time_ms: float32      # 最近抓取耗时
    error_message: string           # 错误信息（ERROR 状态时）
```

### 15.2 启动健康检查

> 源码: `cleaner_manager_node.py:_startup_check()`

调用 `~/start` 服务时执行，失败则拒绝启动并返回错误信息：

```python
def _startup_check(self):
    """返回 (ok: bool, message: str)"""
    # 1. TF map→base_link 可用（5s 超时）
    # 2. /piper/observe, /piper/go_ready, /piper/in_working_area 服务可用（2s 超时）
    # 3. /piper/pick, /piper/place action server 可用（2s 超时）
    # 4. ApproachNavigator 已设置
```

### 15.3 中止机制

```yaml
~/abort:                              # 实际话题: /cleaner_manager_node/abort
  类型: std_srvs/Trigger
  行为:
    1. 设置 abort_event (threading.Event)
    2. Cancel ApproachNavigator (navigator.cancel())
    3. Cancel 活跃的 Piper pick/place goal_handle
    4. Worker thread 检测到 abort → best-effort stow arm
    5. 状态 → IDLE
```

---

## 16. 文件结构

```
src/
├── slam/                              # ✅ SLAM 模块
│   ├── launch/
│   │   ├── hdl_navigation_launch.py   # HDL定位 + Nav2 导航
│   │   └── scpgo_mapping_launch.py    # FastLIO + SC-PGO 建图
│   ├── config/
│   │   └── nav2_minimal_params.yaml   # Nav2 完整配置
│   └── scripts/
│       ├── odom_fusion_sync.py        # FastLIO+轮式融合(50Hz)
│       ├── tf_republisher.py          # map→odom 平滑发布
│       ├── hdl_odom_to_tf.py          # HDL→TF 转换
│       └── globalmap_publisher.py     # PCD 地图加载
│
├── perception/                        # ✅ 感知模块
│   ├── msg/
│   │   ├── Object3D.msg / Object3DArray.msg
│   │   ├── TrackedObject3D.msg / TrackedObject3DArray.msg
│   │   └── GraspObject.msg / GraspResult.msg
│   ├── srv/
│   │   ├── GraspDetect.srv
│   │   └── DetectObjects.srv
│   ├── src/
│   │   ├── multi_camera_perception_node.py  # 双相机融合
│   │   ├── object_tracker_node.py           # 独立 SAM2 跟踪
│   │   ├── perception_grasp_node.py         # 手部抓取检测
│   │   ├── scene_perception_core.py         # Mask→3D 核心算法
│   │   ├── intelligent_fusion.py            # 融合策略
│   │   ├── byte_tracker_3d.py               # 3D ByteTrack
│   │   └── percept.py                       # 在线服务客户端
│   └── config/
│       └── extrinsics_*.yaml                # 外参标定
│
├── navigation/
│   └── approach_navigator/            # ✅ 三阶段导航
│       ├── approach_navigator/
│       │   ├── navigator.py           # 核心导航器(530行)
│       │   ├── depth_sensor.py        # 深度传感器(556行)
│       │   └── config.py             # 配置管理
│       └── config/
│           └── approach_navigator.yaml
│
├── piper_msgs/                        # ✅ Piper 消息定义
│   ├── srv/  (Observe, GoReady, EnableEnhanced, InWorkingArea, ...)
│   └── action/  (PiperPick, PiperPlace)
│
├── piper_grasp/                       # ✅ 机械臂抓取
│   ├── scripts/
│   │   ├── piper_grasp_node.py        # 主节点(2100行)
│   │   ├── piper_api_v2.py            # CAN通信(850行)
│   │   ├── collision_checker.py       # 碰撞检测
│   │   ├── grasp_blacklist.py         # 双层黑名单
│   │   └── safe_trajectory_planner.py # 碰撞感知规划
│   └── config/
│       ├── piper_grasp_node.yaml
│       └── collision_config.yaml
│
└── cleaner_manager/                     # ✅ 已实现
    ├── cleaner_manager/                 # Python 模块（安装到 site-packages）
    │   ├── __init__.py
    │   ├── target_pool.py             # TargetPool + TargetRecord (~180行)
    │   └── pick_state_machine.py      # PickStateMachine + PickState (~300行)
    ├── src/
    │   └── cleaner_manager_node.py      # ROS2 节点入口 + main() (~230行)
    ├── msg/
    │   └── CleanerManagerStatus.msg     # 状态反馈消息
    ├── config/
    │   └── cleaner_manager.yaml         # 20+ 参数
    ├── launch/
    │   └── cleaner_manager.launch.py
    ├── CMakeLists.txt                 # ament_cmake + rosidl_generate_interfaces
    └── package.xml
```

---

## 17. 实现计划

### Phase 1: 基础模块 ✅ 已完成

- [x] SLAM: HDL定位 + Nav2导航 + 里程计融合
- [x] Perception: 双相机融合 + ByteTracker3D + 抓取检测
- [x] Manipulation: PiperGraspNode + observe/pick/place
- [x] Navigation: ApproachNavigator 三阶段导航
- [x] 消息定义: perception msg/srv + piper_msgs

### Phase 2: Cleaner Manager + Pick 任务 ✅ 已完成

- [x] 创建 cleaner_manager 包 + CleanerManagerStatus.msg
- [x] 实现 TargetPool（线程安全、map坐标目标管理、暂停/恢复、指数移动平均位置更新）
- [x] 实现 PickStateMachine（10 个状态、handler 表驱动、Piper service/action 异步调用封装）
- [x] 实现 CleanerManagerNode（MultiThreadedExecutor、daemon worker thread、ReentrantCallbackGroup）
- [x] 实现启动健康检查（TF + services + actions + navigator）
- [x] 实现 abort 服务（cancel goal + cancel nav + stow arm）
- [x] 集成 ApproachNavigator（MultiThreadedExecutor 共享）
- [x] 集成 PiperGraspNode（service/action clients）
- [x] 集成 /object_tracker_node/tracked_objects（唯一感知输入）

### Phase 3: 端到端集成测试 ✅ 已完成

- [x] 启动完整系统（SLAM + Perception + PiperGrasp + CleanerManager）
- [x] 验证健康检查通过
- [x] 验证完整 pick 流程（PLANNING → ... → COMPLETED）
- [x] 验证 abort 中断恢复
- [x] 验证连续失败自动重试和退出
- [x] 长时间运行稳定性测试

---

## 18. 配置参数

```yaml
cleaner_manager_node:
  ros__parameters:
    # --- 感知输入（唯一来源）---
    tracked_objects_topic: "/object_tracker_node/tracked_objects"

    # --- 目标过滤 ---
    min_track_score: 0.3
    min_position_confidence: 0.3
    min_distance: 0.3              # m
    max_distance: 3.0              # m
    max_attempts: 3

    # --- 批量抓取 ---
    max_picks_per_nav: 10

    # --- 感知稳定化 ---
    scan_stable_frames: 3          # 连续 N 帧物体数不变视为稳定
    scan_stable_timeout: 3.0       # 最长等待时间 (秒)

    # --- 抓取参数（透传给 Piper）---
    pick_speed: 30                 # 1-100%
    lift_height: 200.0             # mm
    observe_prompt: "bottle.cup.box"

    # --- 超时 ---
    observe_timeout: 10.0          # 秒
    pick_timeout: 60.0             # 秒
    place_timeout: 30.0            # 秒
    nav_timeout: 120.0             # 秒

    # --- 错误处理 ---
    max_consecutive_failures: 3    # 连续失败 → ERROR
    error_cooldown: 5.0            # ERROR 状态冷却时间 (秒)，之后自动重试
    max_error_retries: 3           # 最大自动重试次数，超过则终止

    # --- 状态反馈 ---
    status_rate: 1.0               # Hz
```

---

## 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| 9.1 | 2026-03-27 | 包名 robot_manager→cleaner_manager 同步更新；Nav2 控制器 TEB→MPPI；感知管线 DinoX→SAM3+FS；Phase 3 标记完成 |
| 9.0 | 2026-03-02 | **robot_manager 包实现完毕**。更新全文"待实现"→"已实现"；新增 §8.4 进程/线程模型、§8.5 handler 表驱动、§8.6 状态转移详表；修正 PickState 枚举顺序(ERROR=8,COMPLETED=9)；TargetPool 更新为线程安全实现(threading.Lock + 指数移动平均)；STOWING 改用 go_ready 替代 go_zero；status msg 字段名与实际 .msg 对齐；Phase 2 标记完成，新增 Phase 3 集成测试 |
| 8.0 | 2026-03-02 | 包名从 pick_mission 恢复为 robot_manager，为后续任务类型拓展留空间；ROS2 命名空间统一为 /robot_manager/*；节点/消息/配置文件同步更名；新增可拓展性设计原则说明 |
| 7.1 | 2026-03-02 | 修正坐标系错误：导航参数(base_link)和机械臂工作区(piper_link_base)是不同坐标系，移除错误的跨坐标系数值对比；强化模块解耦原则；工作区图表标注正确坐标系 |
| 7.0 | 2026-03-02 | 深度审查修订：修复感知链路描述(Topic remap)；新增 STOWING/DEPLOYING 状态；TargetPool 暂停/恢复；ObjectTrackerNode 独立流水线；统一感知输入；ERROR 自动恢复；abort 机制和健康检查；监控诊断；帧计数稳定化；集中单位转换 |
| 6.0 | 2026-03-02 | 重构为 Robot Manager；SLAM 提升为独立模块；更新双相机融合架构 |
| 5.0 | 2026-02-26 | Piper 迁移完成，详细记录 ROS2 接口 |
| 4.1 | 2025-02-26 | 批量抓取(1:M)和工作区扫描策略 |
| 4.0 | 2025-02-26 | 简化状态机，明确实现进度 |
| 3.0 | 2025-02-26 | 整合 Piper 迁移方案 |
