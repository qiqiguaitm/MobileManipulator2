<div align="center">

# MobileManipulator2

**ROS2 Humble 自主移动操作系统 · NVIDIA Jetson Orin**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Platform](https://img.shields.io/badge/平台-Jetson%20Orin-green?logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![Architecture](https://img.shields.io/badge/架构-ARM64-orange)](https://developer.nvidia.com/embedded/jetson-agx-orin)
[![License](https://img.shields.io/badge/许可证-MIT-yellow)](LICENSE)

*融合三层激光雷达定位、视觉语言感知与六自由度抓取的生产级自主移动操作系统，运行于 NVIDIA Jetson Orin。*

[English](README.md) · [架构文档](docs/navigation_slam_architecture.md) · [抓取设计](docs/pick_navigator_design.md)

</div>

---

## 系统简介

MobileManipulator2 是面向非结构化室内环境的完整自主拣取系统。差速移动底盘搭载六自由度机械臂，结合多模态传感器与分层软件栈，能够根据自然语言提示自主检测、导航、接近并抓取目标物体。

```
"bottle.cup.box"  →  检测  →  导航  →  精确接近  →  抓取  →  放置
```

全系统在 NVIDIA Jetson Orin（ARM64）上独立运行，无需外部算力。

---

## 核心能力

| 模块 | 能力描述 |
|------|---------|
| **定位** | 三层定位：FastLIO2（激光惯性里程计）+ HDL-NDT 地图匹配 + ScanContext 自动重定位 |
| **感知** | 视觉语言目标检测（DinoX / SAM3 二选一）、双相机三维融合、ByteTracker3D 多目标跟踪 |
| **抓取** | GraspAnything 抓取姿态估计、CDM 深度增强、完整九阶段 Pick 动作流水线 |
| **导航** | 三阶段精确接近：Nav2 全局规划 → 航向 PD 对齐 → 深度相机闭环最终接近 |
| **编排** | 十状态自主状态机，含自动恢复、目标池管理与批量抓取 |
| **硬件** | Tracer2 底盘 + Piper 6-DOF 机械臂 + RoundScan LiDAR + 3× RealSense D4xx |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          传感器层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐  │
│  │RoundScan Qx30│  │  HiPNUC IMU │  │D455 顶部  │  │D435 手部  │  │
│  │  16线 10Hz   │  │   100 Hz    │  │D435 底盘  │  │（夹爪）   │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘  │
└─────────┼─────────────────┼───────────────┼──────────────┼────────┘
          │                 │               │              │
┌─────────▼─────────────────▼───────┐  ┌───▼──────────────▼────────┐
│            SLAM 层                │  │         感知层              │
│                                   │  │                            │
│  FastLIO2 ──► 里程计 (10 Hz)     │  │  DinoX / SAM3              │
│      +                            │  │  （自然语言驱动检测）        │
│  轮式里程计 ──► 融合 (50 Hz)      │  │       │                    │
│      │                            │  │  GraspAnything             │
│  HDL-NDT ──► map→odom TF         │  │  （抓取姿态估计）            │
│      │                            │  │       │                    │
│  ScanContext ──► 自动重定位        │  │  ByteTracker3D             │
│                                   │  │  （多目标三维跟踪）          │
│  Nav2（Smac2D + MPPI/TEB）        │  │       │                    │
└───────────────────┬───────────────┘  └───────┬────────────────────┘
                    │                           │
          ┌─────────▼───────────────────────────▼──────────┐
          │                  编排层                          │
          │                                                  │
          │   RobotManagerNode（cleaner_manager）            │
          │   ┌──────────────────────────────────────┐     │
          │   │           PickStateMachine            │     │
          │   │                                      │     │
          │   │  IDLE → PLANNING → STOWING →         │     │
          │   │  NAVIGATING → DEPLOYING → SCANNING → │     │
          │   │  PICKING → PLACING → [ERROR] →       │     │
          │   │  COMPLETED                           │     │
          │   └──────────────────────────────────────┘     │
          │                                                  │
          │   TargetPool（目标池）  │  ApproachNavigator     │
          └───────┬────────────────────────┬────────────────┘
                  │                        │
     ┌────────────▼───────┐  ┌────────────▼────────────────┐
     │      操控层         │  │        导航执行层            │
     │                    │  │                             │
     │  PiperGraspNode    │  │  阶段1：Nav2 全局规划        │
     │  ├─ Observe 服务   │  │  阶段2：PD 航向对齐          │
     │  ├─ Pick 动作      │  │  阶段3：深度相机闭环         │
     │  ├─ Place 动作     │  │         精确接近             │
     │  └─ GoReady 服务   │  └─────────────────────────────┘
     │                    │
     │  PiperDriver（CAN）│
     └────────────────────┘
```

---

## 硬件配置

| 组件 | 型号 | 通信接口 | 规格 |
|------|------|---------|------|
| **移动底盘** | 松灵 Tracer2 | CAN1 · 500 kbaud | 差速驱动 · 协议 V2 |
| **机械臂** | 松灵 Piper | CAN · 协议 V2 | 6-DOF · 负载 2 kg · 臂展 580 mm |
| **激光雷达** | RoundScan Qx30（Helios 16P） | 以太网 | 16 线 · 10 Hz · 10 m 量程 |
| **IMU** | HiPNUC CH110 | USB 串口 | 9 轴 · 100 Hz |
| **相机（顶部）** | Intel RealSense D455 | USB3 | RGBD · 30 Hz · 大视角 |
| **相机（底盘）** | Intel RealSense D435 | USB3 | RGBD · 30 Hz |
| **相机（手部）** | Intel RealSense D435 | USB3 | 夹爪安装 RGBD |
| **算力平台** | NVIDIA Jetson Orin | — | ARM64 · JetPack 5.x |

---

## 软件架构

```
┌─────────────────────────────────────────────────────────┐
│                  ROS2 Humble（27 个软件包）              │
├──────────────┬──────────────┬──────────────┬───────────┤
│  SLAM / 导航  │     感知      │    操控       │  系统     │
├──────────────┼──────────────┼──────────────┼───────────┤
│ fast_lio     │ perception   │ piper_grasp  │ cleaner_  │
│ hdl_local.   │  ├─ DinoX/   │  ├─ Observe  │ manager   │
│ hdl_global_  │  │  SAM3(或) │  ├─ Pick     │           │
│   local.     │  ├─ Grasp    │  ├─ Place    │ approach_ │
│ sc_pgo       │  │  Anything │  └─ GoReady  │ navigator │
│ nav2         │  └─ CDM 深度  │              │           │
│ ndt_omp      │ ByteTrack3D  │ piper_driver │ slam      │
│ fast_gicp    │ camera_driver│ piper_msgs   │（launch）  │
├──────────────┴──────────────┴──────────────┴───────────┤
│      tracer_base · lidar_driver · hipnuc_imu           │
│      mobile_manipulator2_description（URDF）            │
└─────────────────────────────────────────────────────────┘
```

---

## 自主拣取系统详解

### 十状态状态机

系统核心是一个**十状态有限状态机**，统一协调所有子系统：

```
IDLE（待机）
 │ 调用 /robot_manager_node/start
 ▼
PLANNING（规划）──── 无可抓目标 ─────────────────► COMPLETED（完成）
 │ 选中目标
 ▼
STOWING（收臂）      ← GoReady(open_gripper=False)，机械臂收至安全位
 │
 ▼
NAVIGATING（导航）   ← 三阶段 ApproachNavigator
 │  阶段1  Nav2 全局规划 → 距目标 0.45 m 停止
 │  阶段2  PD 控制原地旋转对齐目标方向（容差 ±5°）
 │  阶段3  深度相机闭环精确接近 → 前边缘距目标 0.08 m
 ▼
DEPLOYING（展臂）    ← GoReady(open_gripper=True)，展至观察位
 │
 ▼
SCANNING（扫描）     ← 等待感知稳定（连续3帧）→ 建立工作区抓取队列
 │ 工作区有目标
 ▼
PICKING（拣取）◄─────────────────────────────────┐
 │  1. Observe（DinoX / SAM3 + CDM，手部相机）    │
 │  2. Pick 动作（九阶段机械臂流水线）             │
 │     CHECKING → APPROACHING → OPENING →       │
 │     DESCENDING → CLOSING → VERIFYING →       │
 │     LIFTING → RETURNING → DONE               │
 │                                              │
 ├── 成功 → PLACING（放置）─────────────────────┘
 │              （放置完成后返回 ready 位）
 │
 ├── 连续失败 ≥3 次 → ERROR → 5 s 冷却 → PLANNING
 │
 └── 队列用尽 → PLANNING（导航至下一目标）
```

### 抓取感知流水线

#### Observe 服务（手部相机 RealSense D435）

```
手部相机 RGBD
    │
    ├─► DinoX 检测  或  SAM3 分割（配置选其一）
    │       → 目标 Bounding Box / 实例 Mask
    │
    ├─► CDM 深度增强（Conditional Diffusion Model）
    │       → 优化深度图（消除边缘噪声）
    │
    ├─► GraspAnything 抓取姿态估计
    │       → 抓取点 + 夹爪建议宽度
    │
    └─► 坐标变换（相机坐标系 → arm_base_link）
            → point3d_base [x, y, z] mm
            → angle_base（度）
            → gripper_width（mm）
```

#### Pick 动作九阶段流水线

| 阶段 | 动作 | 说明 |
|------|------|------|
| **CHECKING** | 验证感知结果 | 检查 Observe 结果有效（<30 s），检查黑名单 |
| **APPROACHING** | 移至目标上方 | MoveJ 到目标上方 100 mm，调整 yaw 朝向目标 |
| **OPENING** | 张开夹爪 | 开至物体宽度 + 安全裕量 |
| **DESCENDING** | 分段下降 | 分段 MoveJ（60 mm/段），仅动 Z 轴 |
| **CLOSING** | 夹紧物体 | 闭合夹爪（速度 500 mm/s） |
| **VERIFYING** | 验证抓握 | 检查夹爪宽度 > 5 mm（成功） |
| **LIFTING** | 提升 | 提升 `lift_height`（默认 200 mm） |
| **RETURNING** | 返回观察位 | MoveJ 回 ready 姿态 |
| **DONE** | 完成 | — |

### 三阶段精确接近导航

| 阶段 | 方法 | 精度 | 目标位置 |
|------|------|------|---------|
| **阶段1（全局导航）** | Nav2 Smac2D + MPPI/TEB | ±5 cm | 距目标 0.45 m |
| **阶段2（航向对齐）** | PD 旋转控制 | ±5°（0.087 rad） | 正面朝向目标 |
| **阶段3（精确接近）** | 深度相机闭环 | ±2 cm | 前边缘距目标 0.08 m |

```
阶段3 深度相机闭环控制逻辑：

loop:
    current_depth = 底盘相机测距（ROI 中值）
    distance_to_go = current_depth - 0.08 m

    if distance_to_go ≤ 0.03 m:  # 紧急刹车
        stop()
        break

    if distance_to_go < tolerance:
        break

    cmd_vel.linear.x = f(distance_to_go)  # 正比控制
    publish(cmd_vel)
```

### 目标池（TargetPool）

目标池以 **map 坐标系**统一管理所有检测到的物体：

| 机制 | 实现 | 说明 |
|------|------|------|
| **位置匹配** | 8 cm 阈值 + 同类别 | 区分相邻物体，容忍感知误差 |
| **位置更新** | 指数移动平均（α=0.3） | 稳定跟踪，抑制抖动 |
| **暂停/恢复** | 导航中暂停更新 | 防止底盘移动时 TF 漂移导致误识重复目标 |
| **状态管理** | ACTIVE → PICKED / FAILED | 记录每个目标的拣取状态和失败原因 |
| **黑名单** | 失败位置封锁 600 s | 半径 30 mm，避免反复尝试不可抓目标 |

### 安全机制

```
1. 启动健康检查
   ├─ TF map→base_link 可用（5 s 超时）
   ├─ /piper/observe 服务可用
   ├─ /piper/pick、/piper/place action 可用
   └─ ApproachNavigator 已配置

2. 导航安全
   └─ 导航前强制收臂（STOWING）
      避免 Nav2 恢复行为（倒车/旋转）撞到伸出的机械臂

3. 连续失败恢复
   ├─ 导航失败 ≥3 次 → ERROR → 5 s 冷却 → 重新规划
   ├─ 抓取失败 ≥3 次 → ERROR → 5 s 冷却 → 重新规划
   └─ ERROR 重试 ≥3 次 → COMPLETED（停止）

4. 优雅中断（/abort）
   └─ 取消导航 + 取消 Pick/Place + 收臂 → IDLE

5. Observe 缓存有效期
   └─ CHECKING 阶段验证 Observe 结果 < 30 s
```

---

## 快速开始

### 环境要求

- ROS2 Humble（Ubuntu 20.04，Jetson JetPack 5.x）
- `colcon`、`rosdep`
- CAN 接口 `can1`，配置为 500 kbaud
- Intel RealSense SDK
- RoundScan rslidar SDK

### 编译

```bash
cd /data/workspace/MobileManipulator2
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

或使用提供的构建脚本：

```bash
./scripts/build_navigation.sh     # SLAM + Nav2 栈
./scripts/build_perception.sh     # 感知 + 抓取栈
./scripts/build_tracer2.sh        # 底盘驱动
```

### CAN 初始化

```bash
make can-bringup    # 初始化 CAN1（500 kbaud）
make health         # 硬件健康检查
```

### 建图

```bash
# 启动 SLAM 建图模式
ros2 launch slam fastlio_odom_launch.py

# 手动驱动机器人覆盖目标区域
# 建图完成后保存地图
ros2 service call /sc_pgo/save_map std_srvs/srv/Empty

# 构建 ScanContext 数据库（全局重定位所需，每张地图只需执行一次）
ros2 run hdl_global_localization build_sc_database \
  --input /home/didi/workspace/MobileManipulator2/maps/sc_pgo/latest
```

### 定位 + 导航

```bash
make navi           # 启动导航栈（HDL 定位 + Nav2）
make navi-fusion    # 启动含轮式里程计融合的导航栈
```

### 仅感知

```bash
make percept-full   # 双相机感知 + 目标跟踪
```

### 全系统自主运行

```bash
# 启动完整系统：SLAM + 感知 + 机械臂 + 编排
make cleaner-manager

# 启动自主拣取
ros2 service call /robot_manager_node/start std_srvs/srv/Empty

# 监控状态（1 Hz）
ros2 topic echo /robot_manager_node/status

# 中断停止
ros2 service call /robot_manager_node/abort std_srvs/srv/Empty
```

---

## 配置说明

### cleaner_manager.yaml 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `observe_prompt` | `"bottle.cup.box"` | 目标检测文本提示（点号分隔） |
| `min_track_score` | `0.3` | 最低跟踪置信度门限 |
| `min_distance` | `0.3 m` | 最近感知距离（去噪） |
| `max_distance` | `3.0 m` | 最远感知距离 |
| `max_attempts` | `3` | 单个目标最大抓取尝试次数 |
| `max_picks_per_nav` | `10` | 单次导航停靠最多批量抓取数 |
| `pick_speed` | `30 %` | 机械臂运动速度（1–100%） |
| `lift_height` | `200 mm` | 抓取后提升高度 |
| `observe_timeout` | `10 s` | Observe 服务超时 |
| `pick_timeout` | `60 s` | Pick 动作超时 |
| `nav_timeout` | `120 s` | 导航超时 |
| `error_cooldown` | `5 s` | 自动恢复冷却时间 |
| `max_error_retries` | `3` | 最大自动恢复次数 |

### Piper 机械臂 Ready 位姿

| 轴 | 值 | 坐标系 |
|---|---|------|
| X | 317 mm | `piper_link_base` |
| Y | 15 mm | `piper_link_base` |
| Z | 248 mm | `piper_link_base` |
| Roll | 180° | — |
| Pitch | 30° | — |
| Yaw | 180° | — |

### 三阶段导航参数

```yaml
approach_navigator:
  ros__parameters:
    approach_distance: 0.45            # 阶段1 停止距离 (m)
    final_approach_distance: 0.08      # 阶段3 前边缘目标 (m)
    align_tolerance: 0.087             # 阶段2 角度容差 (rad, ~5°)
    robot_front_offset: 0.434          # base_link 到前边缘 (m)
    camera_forward_offset: 0.394       # 底盘相机前向偏移 (m)
```

---

## ROS2 接口速查

### 主要话题

| 话题 | 消息类型 | 频率 | 说明 |
|------|---------|------|------|
| `/object_tracker_node/tracked_objects` | `TrackedObject3DArray` | 5 Hz | 三维跟踪目标（编排层输入） |
| `/robot_manager_node/status` | `RobotManagerStatus` | 1 Hz | 状态机实时状态 |
| `/piper/joint_states` | `sensor_msgs/JointState` | 50 Hz | 机械臂关节状态 |

### 主要服务

| 服务 | 类型 | 说明 |
|------|------|------|
| `/robot_manager_node/start` | `std_srvs/Empty` | 启动自主拣取 |
| `/robot_manager_node/abort` | `std_srvs/Empty` | 中止并收臂 |
| `/piper/observe` | `Observe` | 感知 + 计算抓取姿态 |
| `/piper/go_ready` | `GoReady` | 机械臂移动至 Ready 位 |

### 主要动作

| 动作 | 类型 | 说明 |
|------|------|------|
| `/piper/pick` | `PiperPick` | 完整九阶段抓取执行 |
| `/piper/place` | `PiperPlace` | 放置物体至默认位置 |

### RobotManagerStatus 状态消息

```
/robot_manager_node/status 包含：
├─ state: uint8              当前状态枚举值
├─ state_name: string        "PLANNING" / "NAVIGATING" / ...
├─ targets_total: uint32     检测总目标数
├─ targets_picked: uint32    已成功拣取数
├─ targets_failed: uint32    失败目标数
├─ targets_remaining: uint32 剩余可抓目标数
├─ current_target_category   当前目标类别
├─ consecutive_failures      连续失败次数
├─ last_nav_time_ms          最后一次导航耗时
├─ last_pick_time_ms         最后一次拣取耗时
└─ error_message             ERROR 状态下的错误信息
```

---

## Makefile 快捷命令

```bash
make navi              # 启动导航（HDL 定位 + Nav2）
make navi-fusion       # 启动含轮式里程计融合的导航
make percept-full      # 双相机感知 + 目标跟踪
make cleaner-manager   # 完整系统（SLAM + 感知 + 机械臂 + 编排）
make can-bringup       # 初始化 CAN1 接口
make health            # 硬件健康检查
make cam-clean         # 清理残留相机进程
```

---

## 标定

标定结果存储于 `calibration_data/`。

| 工具包 | 用途 |
|-------|------|
| `cam_lidar_calibration` | 相机–激光雷达外参标定 |
| `handeye_calibration` | 夹爪相机手眼标定 |
| `multi_eye_calibration` | 多相机外参联合标定 |

传感器重新安装后需重新标定。

---

## 项目结构

```
MobileManipulator2/
├── src/
│   ├── cleaner_manager/        # 编排状态机
│   ├── perception/             # 视觉语言感知（v2.4.0）
│   ├── piper_grasp/            # 抓取控制 + RViz 面板
│   ├── piper_driver/           # Piper 机械臂 CAN 驱动
│   ├── approach_navigator/     # 三阶段精确接近导航
│   ├── slam/                   # SLAM 启动 + Nav2 配置
│   ├── hdl_localization/       # HDL-NDT 地图匹配
│   ├── hdl_global_localization/# ScanContext 全局定位
│   ├── sc_pgo/                 # 位姿图优化（GTSAM）
│   ├── fast_lio/               # FastLIO2 激光惯性里程计
│   ├── tracer_base/            # Tracer2 底盘驱动
│   ├── camera_driver/          # 多路 RealSense 驱动
│   ├── lidar_driver/           # RoundScan Qx30 驱动
│   ├── hipnuc_imu/             # HiPNUC IMU 驱动
│   ├── mobile_manipulator2_description/ # 完整 URDF
│   └── ...                     # （共 27 个软件包）
├── docs/
│   ├── navigation_slam_architecture.md  # SLAM 架构详解
│   ├── pick_navigator_design.md         # 抓取系统设计
│   └── system_communication_interfaces.md # ROS2 接口速查
├── calibration_data/           # 标定结果
├── maps/sc_pgo/                # 保存的 SLAM 地图
├── scripts/                    # 诊断与测试脚本（_cc_* 前缀）
├── config/                     # FastDDS 配置
└── Makefile                    # 快捷命令
```

---

## 技术文档

| 文档 | 内容 |
|------|------|
| [导航与 SLAM 架构](docs/navigation_slam_architecture.md) | 三层定位设计、TF 树、传感器融合 |
| [拣取系统设计](docs/pick_navigator_design.md) | 状态机、目标池、抓取流水线 |
| [系统通信接口](docs/system_communication_interfaces.md) | 完整 ROS2 话题/服务/动作参考 |

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

---

<div align="center">
<sub>基于 ROS2 Humble · 运行于 NVIDIA Jetson Orin · ARM64</sub>
</div>
