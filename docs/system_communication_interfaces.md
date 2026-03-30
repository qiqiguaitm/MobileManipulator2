# 系统通信接口与信息流文档

> 模块：perception / piper_grasp / slam / cleaner_manager
>
> 更新日期：2026-03-27

---

## 1. 模块角色定位

| 模块 | 层级 | 核心职责 |
|------|------|---------|
| **perception** | 感知层 | 多相机 3D 目标检测、跟踪、手部抓取检测 |
| **piper_grasp** | 执行层 | 机械臂控制（CAN 总线）、抓取/放置动作执行 |
| **slam** | 定位导航层 | LiDAR SLAM、里程计融合、Nav2 导航栈 |
| **cleaner_manager** | 编排层 | 状态机编排，串联 感知→导航→抓取→放置 |

架构分层关系：

```
┌──────────────────────────────────────────┐
│            cleaner_manager (编排层)         │
│  状态机驱动，调度下层三个模块协同工作       │
└────┬──────────────┬──────────────┬───────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐  ┌───────────┐  ┌──────────┐
│perception│  │piper_grasp│  │   slam   │
│ (感知层) │  │ (执行层)  │  │(导航层)  │
└─────────┘  └───────────┘  └──────────┘
```

---

## 2. 模块间直接通信接口

### 2.1 perception → cleaner_manager（Topic）

```
perception::ObjectTrackerNode
    │
    │  /object_tracker_node/tracked_objects
    │  类型: perception/TrackedObject3DArray
    │  频率: 5Hz (跟踪器输出频率)
    │  坐标系: base_link
    │  内容: track_id, category, position, distance, track_score, position_confidence
    │
    ▼
cleaner_manager::TargetPool
    → TF 变换到 map 坐标 → 目标池管理 → 供状态机调度
```

**唯一直接数据通道。** cleaner_manager 被动接收场景跟踪结果，通过 TF 转到 map 坐标系后存入全局目标池，不主动触发场景感知。

### 2.2 perception ← piper_grasp（Service）

```
piper_grasp::piper_grasp_node         perception::perception_grasp_node
        │                                          │
        │  调用 /perception_grasp_node/detect       │
        │  类型: perception/srv/GraspDetect         │
        │  请求: prompt, enable_cdm                 │
        │  响应: point3d(米,相机系), width3d,        │
        │        angle, category, score,            │
        │        center_uv, depth                   │
        │                                          │
        └──────────── client ──────────► server ───┘
```

piper_grasp 是 client，perception_grasp 是 server。此调用封装在 piper_grasp 的 `/piper/observe` 服务内部——cleaner_manager 调 observe，observe 内部再调 GraspDetect。

### 2.3 cleaner_manager → piper_grasp（Service + Action）

**Services（同步短时命令）：**

| 服务名 | 类型 | 调用阶段 | 说明 |
|--------|------|---------|------|
| `/piper/observe` | `piper_msgs/srv/Observe` | SCANNING | 触发手部感知，返回目标 3D 位姿 |
| `/piper/go_ready` | `piper_msgs/srv/GoReady` | STOWING / DEPLOYING | 收臂(导航前) / 展臂(抓取前) |
| `/piper/in_working_area` | `piper_msgs/srv/InWorkingArea` | SCANNING | 判断 map 坐标点是否在臂可达范围 |
| `/piper/get_status` | `piper_msgs/srv/GetStatus` | 启动检查 | 确认机械臂连接和使能状态 |

**Actions（异步长时操作）：**

| Action 名 | 类型 | 调用阶段 | 说明 |
|-----------|------|---------|------|
| `/piper/pick` | `piper_msgs/action/PiperPick` | PICKING | 接近→张开→下降→夹取→验证→抬起 |
| `/piper/place` | `piper_msgs/action/PiperPlace` | PLACING | 抬起→移动→下降→释放→回 ready |

### 2.4 cleaner_manager → slam / Nav2（Action + Topic）

```
cleaner_manager::ApproachNavigator
    │
    │  阶段1: nav2_msgs/action/NavigateToPose (Nav2 全局导航)
    │  阶段2-3: 发布 /cmd_vel (geometry_msgs/Twist, 精确接近控制)
    │
    ▼
slam 启动的 Nav2 栈 (planner_server, controller_server, bt_navigator)
    → 路径规划 → 局部控制 → 底盘运动
```

### 2.5 slam → 全模块（TF 坐标链）

```
map ──(hdl_odom_to_tf)──► odom ──(Fast-LIO)──► base_link
                                                   │
                                        静态 TF: body, lidar_link,
                                               arm_base, camera frames...
```

所有模块的坐标依赖：

| 模块 | 依赖的 TF | 用途 |
|------|----------|------|
| perception | base_link 为目标坐标系 | 3D 检测结果统一到 base_link |
| piper_grasp | arm_base → 手眼标定 → camera_hand_optical | 相机坐标转机械臂坐标 |
| cleaner_manager | map ↔ base_link 双向 | 目标池坐标转换、导航目标生成 |
| Nav2 | map → odom → base_link 完整链 | 全局定位 + 路径规划 |

---

## 3. 通信接口形式分类

| 通信形式 | 特点 | 使用场景 |
|---------|------|---------|
| **Topic** | 异步、持续、多对多 | 高频数据流：跟踪结果、odom、点云、TF |
| **Service** | 同步、请求-响应 | 短时命令/查询：observe、go_ready、GraspDetect |
| **Action** | 异步、带反馈、可取消 | 长时操作：pick、place、navigate_to_pose |
| **TF** | 全局坐标变换树 | 基础设施：所有模块共享的坐标系关系 |

---

## 4. 各模块详细接口清单

### 4.1 perception

#### 自定义消息

| 消息 | 说明 |
|------|------|
| `Object3D` | 单个 3D 检测结果（相机+LiDAR 位置） |
| `Object3DArray` | Object3D 数组，含 header 和 frame_id |
| `TrackedObject3D` | 带 track_id 的跟踪目标 |
| `TrackedObject3DArray` | TrackedObject3D 数组 |
| `GraspObject` | 可抓取目标（2D 检测 + mask + 抓取信息 + 3D 位置） |
| `GraspObjectArray` | GraspObject 数组，含 chosen_index |
| `GraspResult` | 紧凑型抓取结果 |
| `PerceptionConfig` | 动态检测配置（prompt, min_score, iou_threshold） |
| `PerceptionStatus` | 当前配置 + 检测统计 |

#### 自定义服务

| 服务 | 说明 |
|------|------|
| `DetectObjects` | 请求: prompt, enable_lidar, camera_id → 响应: Object3DArray |
| `GraspDetect` | 请求: prompt, enable_cdm → 响应: point3d, width3d, angle, category, score |

#### 主要节点接口

**scene_perception_3d_node** — 单相机 3D 场景感知

| 方向 | 接口 | 类型 |
|------|------|------|
| Sub | `/camera/{name}/color/image_raw` | sensor_msgs/Image |
| Sub | `/camera/{name}/aligned_depth_to_color/image_raw` | sensor_msgs/Image |
| Sub | `/camera/{name}/color/camera_info` | sensor_msgs/CameraInfo |
| Sub | `/lidar/chassis/point_cloud` | sensor_msgs/PointCloud2 |
| Sub | `/perception/config` | perception/PerceptionConfig |
| Pub | `~/objects_3d` | perception/Object3DArray |
| Pub | `~/optimized_depth` | sensor_msgs/Image |
| Pub | `/perception/status` | perception/PerceptionStatus |
| Srv | `~/detect` | perception/DetectObjects |

**multi_camera_perception_node** — 双相机并行检测 + 融合

| 方向 | 接口 | 类型 |
|------|------|------|
| Sub | `/camera/top/*`, `/camera/chassis/*` | Image, CameraInfo |
| Sub | `/rslidar_points` | sensor_msgs/PointCloud2 |
| Sub | `/perception/config` | perception/PerceptionConfig |
| Pub | `~/top/objects_3d`, `~/chassis/objects_3d` | perception/Object3DArray |
| Pub | `~/fused/objects_3d` | perception/Object3DArray |
| Pub | `~/top/optimized_depth`, `~/chassis/optimized_depth` | sensor_msgs/Image |
| Srv | `~/detect` | perception/DetectObjects |

**object_tracker_node** — SAM2 跟踪

| 方向 | 接口 | 类型 |
|------|------|------|
| Sub | RGB + Depth + CameraInfo | sensor_msgs/Image, CameraInfo |
| Sub | 检测结果 topic | perception/Object3DArray |
| Pub | `~/tracked_objects` | perception/TrackedObject3DArray |

**perception_grasp_node** — 手部抓取感知

| 方向 | 接口 | 类型 |
|------|------|------|
| Sub | `/camera/hand/color/image_raw` | sensor_msgs/Image |
| Sub | `/camera/hand/aligned_depth_to_color/image_raw` | sensor_msgs/Image |
| Sub | `/camera/hand/color/camera_info` | sensor_msgs/CameraInfo |
| Sub | `/usr/prompt/grasp` | std_msgs/String |
| Pub | `~/result` | perception/GraspObjectArray |
| Pub | `~/depth` | sensor_msgs/Image |
| Srv | `~/detect` | perception/GraspDetect |

### 4.2 piper_grasp

#### 依赖消息包：piper_msgs

**Messages:** `PiperStatus`, `PiperStatusMsg`, `PosCmd`

**Services:** `Enable`, `Gripper`, `GoZero`, `EnableEnhanced`, `SetPosition`, `SetGripper`, `GetStatus`, `GetPosition`, `GoReady`, `Observe`, `InWorkingArea`

**Actions:** `PiperPick` (10 状态反馈), `PiperPlace` (8 状态反馈)

#### piper_grasp_node 接口

| 方向 | 接口 | 类型 | 说明 |
|------|------|------|------|
| Pub | `/joint_states` | sensor_msgs/JointState | 8 关节状态, 200Hz |
| Pub | `/piper/status` | piper_msgs/PiperStatus | 综合状态 |
| Pub | `/piper/gripper_center` | geometry_msgs/PoseStamped | 夹爪中心位姿 (arm_base 系) |
| Pub | `/piper/error_state` | std_msgs/String | 错误状态 |
| Pub | `/piper/connection_state` | std_msgs/UInt8 | 连接状态码 |
| Pub | `/arm_status` | piper_msgs/PiperStatusMsg | 底层状态 |
| Pub | `/end_pose_euler` | piper_msgs/PosCmd | 末端位姿 |
| Srv | `/piper/enable` | piper_msgs/EnableEnhanced | 使能/禁用/重连 |
| Srv | `/piper/set_position` | piper_msgs/SetPosition | 设置臂位置 |
| Srv | `/piper/set_gripper` | piper_msgs/SetGripper | 设置夹爪 |
| Srv | `/piper/get_status` | piper_msgs/GetStatus | 获取状态 |
| Srv | `/piper/get_position` | piper_msgs/GetPosition | 获取位置 |
| Srv | `/piper/go_ready` | piper_msgs/GoReady | 回 ready 位 |
| Srv | `/piper/observe` | piper_msgs/Observe | 目标检测定位 |
| Srv | `/piper/in_working_area` | piper_msgs/InWorkingArea | 可达性检查 |
| Act | `/piper/pick` | piper_msgs/PiperPick | 拾取动作 |
| Act | `/piper/place` | piper_msgs/PiperPlace | 放置动作 |
| Call | `/perception_grasp_node/detect` | perception/GraspDetect | 调用感知抓取 |

### 4.3 slam

**无自定义消息/服务/动作定义。** 编排外部组件 (Fast-LIO, HDL, Nav2)。

#### 核心节点接口

| 节点 | 订阅 | 发布 | 说明 |
|------|------|------|------|
| odom_frame_converter | `/fastlio/odom` | `/odom` | camera_init/body → odom/base_link |
| odom_fusion_sync | `/fastlio/odom` + `/wheel_odom` | `/odom/fused` | 融合 LiDAR(10Hz) + 轮式(50Hz) |
| odom_relay | `/odom` | `/odom/fused` | 简单转发（无融合时） |
| delayed_cloud_relay | `/rslidar_points` | `/rslidar_points_delayed` | 延迟点云供 HDL 使用 |
| globalmap_publisher | — | `/globalmap` + `/globalmap_visual` | 加载 PCD 地图 |
| hdl_odom_to_tf | `/hdl_localization/odom_out` + `/odom/fused` | TF: map→odom | HDL 定位输出转 TF |
| tf_republisher | `/tf` (过滤 map→odom) | TF: map→odom (100Hz) | 高频重发防 extrapolation |
| emergency_stop | — | `/cmd_vel` (零速度) | 急停 |

#### 服务

| 服务名 | 类型 | 节点 |
|--------|------|------|
| `/emergency_stop/stop` | std_srvs/Trigger | emergency_stop (daemon) |
| `/emergency_stop/release` | std_srvs/Trigger | emergency_stop (daemon) |

#### 启动的 Nav2 组件（通过 launch）

planner_server (Smac2D), controller_server (MPPI), behavior_server, bt_navigator, map_server, lifecycle_manager — 提供标准 Nav2 Action 接口。

### 4.4 cleaner_manager

#### 自定义消息

| 消息 | 字段 |
|------|------|
| `RobotManagerStatus` | state, state_name, targets_total/picked/failed/remaining, current_target_category, consecutive_failures, last_nav/pick_time_ms, error_message |

#### cleaner_manager_node 接口

| 方向 | 接口 | 类型 | 说明 |
|------|------|------|------|
| Sub | `/object_tracker_node/tracked_objects` | perception/TrackedObject3DArray | 感知跟踪结果 |
| Pub | `~/status` | cleaner_manager/RobotManagerStatus | 状态机状态 |
| Srv | `~/start` | std_srvs/Trigger | 启动自主任务 |
| Srv | `~/abort` | std_srvs/Trigger | 中止任务 |
| Call | `/piper/observe` | piper_msgs/Observe | 触发感知 |
| Call | `/piper/go_ready` | piper_msgs/GoReady | 收/展臂 |
| Call | `/piper/in_working_area` | piper_msgs/InWorkingArea | 可达判断 |
| Call | `/piper/get_status` | piper_msgs/GetStatus | 臂状态查询 |
| Act Client | `/piper/pick` | piper_msgs/PiperPick | 拾取 |
| Act Client | `/piper/place` | piper_msgs/PiperPlace | 放置 |
| Nav | NavigateToPose + `/cmd_vel` | nav2_msgs / geometry_msgs | 导航接近 |
| TF | map ↔ base_link | — | 坐标转换 |

---

## 5. 完整信息流：自主抓取任务

```
用户调用 /cleaner_manager_node/start (Trigger)
│
▼
┌─── cleaner_manager 状态机循环 ──────────────────────────────────────────────────┐
│                                                                               │
│  ① PLANNING                                                                  │
│     TargetPool.get_nav_target()                                              │
│     └─ 数据来源: perception/TrackedObject3DArray (持续 topic 更新)             │
│     └─ 选最近的、未超最大尝试次数的 ACTIVE 目标                                 │
│                                                                               │
│  ② STOWING                                                                   │
│     调 /piper/go_ready (open_gripper=False)                                  │
│     └─ 机械臂收回安全位，避免导航中碰撞                                        │
│                                                                               │
│  ③ NAVIGATING                                                                │
│     ApproachNavigator.navigate_to(target_pose)                               │
│     └─ 阶段1: Nav2 NavigateToPose Action → slam/Nav2 全局路径规划+执行         │
│     └─ 阶段2-3: /cmd_vel 直接控制精确接近                                     │
│     └─ 依赖: slam 提供的 TF (map→odom→base_link) + costmap (/scan)           │
│                                                                               │
│  ④ DEPLOYING                                                                 │
│     调 /piper/go_ready (open_gripper=True)                                   │
│     └─ 展臂到观测位，张开夹爪准备抓取                                          │
│                                                                               │
│  ⑤ SCANNING                                                                  │
│     调 /piper/in_working_area (检查目标可达性)                                 │
│     └─ 将 map 坐标 TF 变换到 base_link → 乘 1000 转 mm → 调服务               │
│     └─ 等待 scan_stable_frames 帧稳定                                         │
│                                                                               │
│  ⑥ PICKING                                                                   │
│     调 /piper/observe (prompt="bottle.cup.box")                              │
│       └─ piper 内部调 /perception_grasp_node/detect (GraspDetect)            │
│         └─ perception 手部相机: SAM3 + GraspAnything + CDM                    │
│         └─ 返回 point3d(相机系,米) → piper 手眼标定 → base_link(mm)           │
│     调 /piper/pick Action                                                    │
│       └─ 反馈: CHECKING→APPROACHING→OPENING→DESCENDING→CLOSING→              │
│              VERIFYING→LIFTING→RETURNING→DONE                                │
│                                                                               │
│  ⑦ PLACING                                                                   │
│     调 /piper/place Action                                                   │
│       └─ 反馈: MOVING→DESCENDING→OPENING→LIFTING→RETURNING→DONE              │
│                                                                               │
│  └── 回到 ① 继续下一个目标，直到目标池清空或达到最大失败次数                     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键数据流转路径

### 6.1 场景感知数据流（全局目标发现）

```
相机(top/chassis)
  → multi_camera_perception_node (SAM3 检测 + FoundationStereo 深度)
    → Object3DArray (per-camera)
      → 匈牙利匹配 + ByteTracker3D
        → fused/objects_3d
          → object_tracker_node (SAM2 跟踪)
            → TrackedObject3DArray (base_link 坐标)
              → cleaner_manager::TargetPool (TF 转 map 坐标)
```

### 6.2 抓取感知数据流（精确抓取定位）

```
cleaner_manager 调 /piper/observe
  → piper_grasp_node 调 /perception_grasp_node/detect
    → perception_grasp_node: 手部相机拍照
      → SAM3 检测 + GraspAnything 抓取点 + CDM 深度优化
        → point3d (camera_hand_optical 坐标系, 米)
          → piper_grasp_node: 手眼标定 (T_cam2flan, T_gripper2flan)
            → base_link 坐标 (mm)
              → 返回给 cleaner_manager (Observe 响应)
```

### 6.3 定位导航数据流

```
LiDAR (/rslidar_points)
  ├→ Fast-LIO → /fastlio/odom (camera_init→body)
  │    ├→ odom_frame_converter → /odom (odom→base_link)
  │    └→ odom_fusion_sync + /wheel_odom → /odom/fused
  │
  └→ delayed_cloud_relay → /rslidar_points_delayed
       → hdl_localization → /hdl_localization/odom_out
            → hdl_odom_to_tf → TF: map→odom
                 → tf_republisher (100Hz 高频重发)

/odom/fused → Nav2 controller_server
/aligned_points → pointcloud_to_laserscan → /scan → Nav2 costmaps
GlobalMap.pcd → globalmap_publisher → hdl_localization (初始定位)
```

---

## 7. 模块间依赖关系图

```
                    ┌────────────┐
                    │   slam     │
                    │ TF + Nav2  │
                    └─────┬──────┘
                          │ TF: map→odom→base_link
                          │ Nav2 Action + /cmd_vel
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌────────────┐  ┌─────────────┐  ┌───────────┐
   │ perception │  │cleaner_manager│  │piper_grasp│
   └──────┬─────┘  └──┬──────┬──┘  └─────┬─────┘
          │            │      │           │
          │  Topic     │      │  Srv/Act  │
          └───────────►│      └──────────►│
                       │                  │
                       │   (observe 内部) │
                       │      ┌───────────┘
                       │      │ Srv: GraspDetect
                       │      ▼
                       │  ┌──────────┐
                       │  │perception│
                       │  │(grasp)   │
                       │  └──────────┘
                       │
                       │  Pub: ~/status (RobotManagerStatus)
                       ▼
                    [用户/监控]
```

**依赖方向总结：**
- cleaner_manager 依赖全部三个模块
- piper_grasp 依赖 perception（GraspDetect 服务）
- perception 和 slam 彼此独立，无直接通信
- slam 是纯基础设施，被全部模块隐式依赖（TF）
