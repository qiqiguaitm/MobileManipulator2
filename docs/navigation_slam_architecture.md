# MobileManipulator2 导航与 SLAM 系统架构

> 版本: 2026-03 | 平台: Jetson Orin (12核 ARM v8, 29GB RAM) | ROS2 Humble

---

## 1. 系统总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Mission / RViz Goal                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Nav2 导航栈                                                        │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────────┐ │
│  │ Smac2D 全局  │→ │ MPPI 局部控制 │→ │ /cmd_vel → Tracer 底盘   │ │
│  │ 路径规划      │  │ (备选: TEB)   │  │                           │ │
│  └──────────────┘  └───────────────┘  └───────────────────────────┘ │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │ Global Costmap (map)│  │ Local Costmap (odom) │  ← /scan         │
│  └─────────────────────┘  └─────────────────────┘                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ 依赖 TF: map→odom→base_link
┌──────────────────────────────┴───────────────────────────────────────┐
│  三层定位系统                                                         │
│  ┌─────────────────┐ ┌──────────────────┐ ┌───────────────────────┐ │
│  │ L1: Fast-LIO2   │ │ L2: HDL-NDT      │ │ L3: 全局定位          │ │
│  │ 里程计 (10Hz)   │ │ 地图匹配 (10Hz)  │ │ ScanContext 重定位    │ │
│  │ odom→base_link  │ │ map→odom         │ │ 初始位姿/丢失恢复     │ │
│  └─────────────────┘ └──────────────────┘ └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                               ▲
┌──────────────────────────────┴───────────────────────────────────────┐
│  传感器层                                                            │
│  LiDAR (Helios16P) │ IMU (HiPNUC) │ 底盘 (Tracer CAN) │ 双相机    │
└──────────────────────────────────────────────────────────────────────┘
```

**机器人规格**: 差速驱动, 足迹 0.809m × 0.62m, 前边缘 0.434m, 后边缘 -0.375m, 搭载 Piper 6DOF 机械臂

---

## 2. 传感器与驱动

| 传感器 | 型号 | 驱动包 | 输出话题 | 频率 |
|--------|------|--------|---------|------|
| 3D LiDAR | Helios 16P (16线) | `rslidar_sdk` | `/rslidar_points` | 10Hz |
| IMU | HiPNUC | `hipnuc_imu` | `/imu/data` | 100Hz |
| 底盘 | Tracer (CAN1) | `tracer_base` | `/wheel_odom` | 50Hz |
| 顶部相机 | RGBD | `camera_driver` | `~/top/color`, `~/top/depth` | 30Hz |
| 底盘相机 | RGBD | `camera_driver` | `~/chassis/color`, `~/chassis/depth` | 30Hz |

**串口注意**: IMU 使用 `/dev/ttyUSB0` (115200 baud), 用户需在 `dialout` 组中。

---

## 3. TF 变换树

### 3.1 完整 TF 链

```
map                                              ← 全局地图坐标系
 └──[map → odom]─── odom                         ← HDL 定位发布, tf_republisher 100Hz 重发
                      └──[odom → camera_init]─── camera_init    ← 静态 identity TF
                                                   └──[camera_init → body]─── body   ← Fast-LIO 动态 TF
                                                                                └──[body → base_link]─── base_link  ← 静态 identity TF
                                                                                                           ├── lidar_link
                                                                                                           │    ├── rslidar        (LiDAR 传感器帧)
                                                                                                           │    └── chassis_camera_link
                                                                                                           └── box_link
                                                                                                                ├── arm_base → link1..link8
                                                                                                                └── lifting_link
                                                                                                                     └── top_camera_link
```

### 3.2 各段 TF 来源

| TF 变换 | 类型 | 发布者 | 频率 | 说明 |
|---------|------|--------|------|------|
| `map → odom` | 动态 | `hdl_localization_node` → `tf_republisher` | 100Hz | NDT 地图匹配结果, EMA 平滑 (α=0.3) |
| `odom → camera_init` | 静态 | `fastlio_odom_launch.py` | latched | identity, 帧名桥接 |
| `camera_init → body` | 动态 | `fastlio_mapping` (Fast-LIO2) | 10Hz | LIO 里程计, 依赖 LiDAR + IMU |
| `body → base_link` | 静态 | `hdl_navigation_launch.py` | latched | identity, 帧名桥接 |
| `base_link → lidar_link` | 静态 | `robot_state_publisher` (URDF) | latched | xyz=(0.273, 0.0, 0.032) |
| `lidar_link → rslidar` | 静态 | `hdl_navigation_launch.py` | latched | z=0.0635m + 微小旋转校准 |
| `base_link → box_link → arm_base` | 静态 | `robot_state_publisher` (URDF) | latched | 机械臂基座 |
| `arm_base → link1..link8` | 动态 | `joint_state_static` + `robot_state_publisher` | 50Hz | 机械臂关节 (固定姿态) |

### 3.3 启动依赖链

```
LiDAR ─┐
       ├→ Fast-LIO ─→ camera_init→body TF ─→ odom→base_link 链完整
IMU ──┘                                        │
                                                ↓
                              HDL canTransform("odom","base_link") 成功
                                                │
                                                ↓
                              HDL 计算并发布精确的 map→odom TF
                                                │
                                                ↓
                              tf_republisher 以 100Hz + 50ms 超前重发
                                                │
                                                ↓
                              Nav2 costmap/controller 正常工作
```

**关键设计**: 当 `odom→base_link` 不可用时, HDL 回退发布 `map→odom` (近似值), 而非 `map→base_link`, 避免 `base_link` 出现两个父节点导致 TF 树断裂。

### 3.4 TF 时间同步策略

| 组件 | 时间戳策略 | 原因 |
|------|-----------|------|
| Fast-LIO | `now()` 系统时间 | 避免传感器时间与系统时间偏差 |
| delayed_cloud_relay | 保持原始传感器时间 | 维持与 Fast-LIO TF 的时序一致性 |
| HDL localization | 点云原始时间 | TF 查询需匹配传感器时间 |
| tf_republisher | `now() + 50ms` | 超前发布避免 Nav2 extrapolation 报错 |

---

## 4. 三层定位系统

### 4.1 L1: Fast-LIO2 里程计

**角色**: 高频相对定位, 提供 `odom → base_link` 链路

```
输入: /rslidar_points (16线 LiDAR) + /imu/data
输出: /fastlio/odom (camera_init → body)
TF:   camera_init → body (动态)
```

| 参数 | 值 | 说明 |
|------|---|------|
| `feature_extract_enable` | false | 直接用原始点, 室内更鲁棒 |
| `point_filter_num` | 1 | 不降采样, 保留全部点 |
| `max_iteration` | 8 | IEKF 最大迭代 |
| `filter_size_surf` | 0.1m | 表面特征体素 |
| `filter_size_map` | 0.1m | 局部地图体素 |
| `cube_side_length` | 500.0m | 局部地图范围 |

### 4.2 L2: HDL-NDT 地图匹配定位

**角色**: 绝对定位, 消除里程计漂移, 提供 `map → odom`

```
输入: /rslidar_points_delayed (延迟50ms的点云)
      /globalmap (参考地图 PCD)
      /odom/fused (里程计预测)
输出: /pose (map → base_link, Odometry)
      /aligned_points (对齐后点云)
TF:   map → odom (动态, 通过 tf_republisher 100Hz 重发)
```

#### 配准算法: NDT-OMP

| 参数 | 值 | 说明 |
|------|---|------|
| `reg_method` | NDT_OMP | 多线程 NDT |
| `ndt_resolution` | 0.5m | 体素分辨率 |
| `ndt_max_iterations` | 50 | 确保收敛 |
| `ndt_neighbor_search_method` | DIRECT7 | 7 邻域搜索 |
| `ndt_neighbor_search_radius` | 2.0m | 搜索半径 |
| `downsample_resolution` | 0.15m | 输入点云下采样 |

#### UKF 状态估计

- **状态向量**: 16 维 (位置3 + 速度3 + 四元数4 + 加速度偏置3 + 陀螺仪偏置3)
- **预测模式**: 里程计预测 (`enable_robot_odometry_prediction: true`)
  - 查询 `odom → base_link` TF 增量
  - 结合 NDT 校正更新位姿
- **cool_time**: 2.0s (NDT 校正后 2 秒内不做预测)

### 4.3 L3: 全局定位 (hdl_global_localization)

**角色**: 初始位姿估计 / 丢失后重定位

```
服务: /set_global_map (接收参考地图)
      /query (输入当前扫描, 返回候选位姿)
```

#### 三种引擎 (可切换)

| 引擎 | 方法 | 精度 | 速度 | 适用场景 |
|------|------|------|------|---------|
| **SCANCONTEXT** (默认) | 回环描述子匹配 | ±0.3m | 快 (<100ms) | 大场景重访 |
| BBS | 分支定界 3D 形状匹配 | ±0.5m | 慢 (~2s) | 任意拓扑 |
| FPFH_RANSAC | 3D 特征匹配 + RANSAC | ±1.0m | 中等 | 特征丰富场景 |

#### ScanContext 配置 (`general_config.yaml`)

```yaml
global_localization_engine: SCANCONTEXT
sc_database_path: maps/sc_pgo/sc_database.bin
sc_dist_threshold: 0.4              # 相似度阈值 (0~1, 越低越严格)
sc_num_candidates: 3                # 返回 Top-N 候选
sc_max_radius: 20.0m                # 搜索半径
```

#### 自动重定位调度器 (HDL 内置)

```
globalmap 收到 → 等待 3s → 采集当前扫描 → 调用 /query 服务
  ├── SC 距离 < 0.4 → 接受位姿, 初始化 pose_estimator
  └── SC 距离 ≥ 0.4 → 拒绝, 等待手动设置或重试
```

| 参数 | 值 |
|------|---|
| `auto_relocalization` | true |
| `auto_reloc_delay_ms` | 3000 |
| `auto_reloc_conf_threshold` | 0.4 |

---

## 5. 里程计融合

### 融合节点: `odom_fusion_sync.py`

```
/fastlio/odom (10Hz, 绝对位姿) ──┐
                                  ├→ /odom/fused (50Hz, 融合输出)
/wheel_odom   (50Hz, 相对增量) ──┘
```

**算法**: 以轮式里程计频率 (50Hz) 为基础, Fast-LIO 每次更新时校准绝对位姿, EMA 滤波 twist:

| 参数 | 值 | 说明 |
|------|---|------|
| `twist_alpha` | 0.3 | 线速度 EMA: 30% 新值 + 70% 旧值 |
| `wz_alpha` | 0.7 | 角速度 EMA: 70% 新值 + 30% 旧值 |
| `wz_deadzone` | 0.03 rad/s | 小角速度死区 |
| `yaw_calibration_alpha` | 0.9 | 航向漂移校准系数 |

**回退**: 若 `use_odom_fusion=false`, `odom_relay.py` 直接将 `/odom` 转发为 `/odom/fused`。

### 帧转换: `odom_frame_converter.py`

Fast-LIO 输出 `camera_init/body` 帧名, Nav2 需要 `odom/base_link`:

```
/fastlio/odom (camera_init → body) → /odom (odom → base_link)
```

纯帧名替换, 不改变位姿/速度数据。

---

## 6. 点云处理链

### 延迟中继: `delayed_cloud_relay.py`

```
/rslidar_points ──[延迟 50ms]──→ /rslidar_points_delayed ──→ HDL / LaserScan
```

**目的**: 确保 Fast-LIO 的 TF (stamp=T) 在 HDL 处理同时间戳点云前已发布。

| 参数 | 值 |
|------|---|
| `delay` | 0.05s (50ms) |
| `timestamp_offset` | 0.0 (保持原始时间戳) |
| `use_system_time` | false |

### LaserScan 生成: `pointcloud_to_laserscan`

```
/rslidar_points_delayed ──→ /scan (前方 180° 激光扫描)
```

| 参数 | 值 | 说明 |
|------|---|------|
| `min_height` | -0.08m | 地面以下 |
| `max_height` | 1.36m | 检测人/桌椅 |
| `angle_min/max` | ±90° | 前方 180°, 屏蔽后方金属反射 |
| `range_min/max` | 0.2m ~ 10.0m | 有效检测范围 |

### 全局地图发布: `globalmap_publisher.py`

```
GlobalMap_hdl.pcd ──→ /globalmap (XYZI, 全部点)
                  ──→ /globalmap_visual (XYZRGB, 高度着色, 去地面)
```

- 自适应地面分离: 0.5m 网格, 取最低 10% 为地面, 3×3 高斯平滑
- Transient-Local QoS (类似 ROS1 latch)

---

## 7. Nav2 导航栈

### 7.1 全局规划器: Smac 2D

```yaml
plugin: "nav2_smac_planner/SmacPlanner2D"    # 优化 A*
tolerance: 0.25m
max_planning_time: 2.0s
allow_unknown: true
```

### 7.2 局部控制器: MPPI (当前) / TEB (备选)

#### MPPI 配置

```yaml
plugin: "nav2_mppi_controller::MPPIController"
controller_frequency: 20.0 Hz              # period = model_dt = 0.05s
time_steps: 56                              # 规划视野 2.8s
batch_size: 1000                            # 采样轨迹数
motion_model: "DiffDrive"
temperature: 0.3                            # 探索度
vx_max: 0.3 m/s | vx_min: -0.15 m/s | wz_max: 0.5 rad/s
```

**9 个评价函数 (Critics)**:

| Critic | 权重 | 作用 |
|--------|------|------|
| ConstraintCritic | 4.0 | 运动学硬约束 |
| ObstaclesCritic | repulsion=1.5, critical=20.0 | 障碍物斥力 + 矩形足迹碰撞检测 |
| CostCritic | 3.81 | costmap 代价感知 |
| GoalCritic | 5.0 | 趋向目标 (<1.4m 激活) |
| GoalAngleCritic | 3.0 | 到达朝向 (<0.5m 激活) |
| PathFollowCritic | 5.0 | 跟随全局路径 |
| PathAlignCritic | 14.0 | 轨迹-路径对齐 |
| PathAngleCritic | 2.0 | 航向与路径方向一致 |
| PreferForwardCritic | 5.0 | 优先前进 (差速关键) |

**vs TEB 的改进**: 消除了震荡恢复、视野缩短、homotopy 等补丁机制; 足迹配置从 3 处同步降为 2 处。

#### TEB 备选配置

注释保留在 `nav2_minimal_params.yaml` 中, 切换方法:
```
注释 MPPI FollowPath 块 → 取消注释 TEB FollowPath 块
```

### 7.3 Costmap

| 层级 | 坐标系 | 尺寸 | 分辨率 | 膨胀半径 |
|------|--------|------|--------|---------|
| Local | odom (rolling) | 4m × 4m | 0.05m | 0.40m |
| Global | map (static) | 全图 | 0.05m | 0.45m |

两层均使用 obstacle_layer (`/scan`) + inflation_layer, global 额外有 static_layer。

### 7.4 恢复行为

```yaml
behavior_plugins: ["spin", "backup", "wait"]
```

BT: `navigate_to_pose_w_replanning_and_recovery.xml` (Nav2 内置)

---

## 8. 建图系统: SC-PGO

**用途**: 离线建图, 生成导航所需的全部地图文件

### 算法: ScanContext + GTSAM ISAM2

```
FastLIO 里程计 ──→ 关键帧提取 ──→ ScanContext 回环检测 ──→ ICP 精配准
                                                              │
                                                              ▼
                                                    GTSAM ISAM2 位姿图优化
                                                              │
                                                              ▼
                                                    输出: 优化后地图 + 位姿
```

**多线程架构**:

| 线程 | 功能 |
|------|------|
| processPG | 关键帧提取 (间距 0.5m / 10°) |
| processLCD | ScanContext 回环检测 (阈值 0.15) |
| processICP | ICP 点云精配准 (fitness 阈值 0.3) |
| processISAM | GTSAM ISAM2 增量优化 |
| processVizMap/Path | 可视化发布 |

### 输出文件

```
maps/sc_pgo/
├── GlobalMap.pcd              # 完整点云地图 (~5MB)
├── GlobalMap_hdl.pcd          # HDL 用下采样地图 (0.2m 体素, ~218KB)
├── map.pgm + map.yaml         # 2D 占据栅格 (Nav2 用)
├── Scans/                     # 关键帧 PCD 文件集
├── optimized_poses.txt        # KITTI 格式优化位姿
├── times.txt                  # 关键帧时间戳
└── sc_database.bin            # ScanContext 描述子数据库 (~430KB)
```

SC 数据库需建图后单独生成: `ros2 run hdl_global_localization build_sc_database`

---

## 9. 感知系统

### 双相机感知: `multi_camera_perception_node.py`

```
顶部相机 (color+depth) ──┐
                          ├→ 匈牙利匹配融合 → ByteTracker3D → /fused/objects_3d
底盘相机 (color+depth) ──┘
```

**检测流水线**: DinoX 检测 → SAM3 分割 → 深度优化 → 3D 包围盒

**ByteTracker3D 参数**:

| 参数 | 值 | 说明 |
|------|---|------|
| `match_thresh` | 0.15m | 一级匹配距离阈值 |
| `second_thresh` | 0.25m | 二级恢复阈值 |
| `track_buffer` | 15 帧 | 丢失缓冲 |
| `confirm_frames` | 2 | 确认激活帧数 |

---

## 10. 话题总览

### 核心数据流

| 话题 | 类型 | 来源 | 消费者 | 频率 |
|------|------|------|--------|------|
| `/rslidar_points` | PointCloud2 | LiDAR 驱动 | delayed_relay | 10Hz |
| `/rslidar_points_delayed` | PointCloud2 | delayed_relay | HDL, laserscan | 10Hz |
| `/imu/data` | Imu | hipnuc_imu | Fast-LIO (可选) | 100Hz |
| `/fastlio/odom` | Odometry | Fast-LIO | frame_converter, fusion | 10Hz |
| `/odom` | Odometry | frame_converter | fusion | 10Hz |
| `/wheel_odom` | Odometry | Tracer 底盘 | fusion | 50Hz |
| `/odom/fused` | Odometry | fusion | Nav2 controller/BT | 50Hz |
| `/globalmap` | PointCloud2 | globalmap_publisher | HDL, global_loc | 1× latch |
| `/scan` | LaserScan | pointcloud_to_laserscan | Nav2 costmap | 10Hz |
| `/aligned_points` | PointCloud2 | HDL | RViz / 备选 laserscan | 10Hz |
| `/cmd_vel` | Twist | Nav2 controller | Tracer 底盘 | 20Hz |

---

## 11. 启动结构

### 主 launch: `hdl_navigation_launch.py`

```
ros2 launch slam hdl_navigation_launch.py [map_dir:=...] [use_odom_fusion:=true]
```

**可选参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `map_dir` | `maps/sc_pgo/` | 地图目录 |
| `launch_chassis` | true | 启动底盘驱动 |
| `use_odom_fusion` | true | 融合轮式里程计 |
| `use_raw_laserscan` | true | 用原始点云生成 laserscan |
| `cloud_delay` | 0.05s | 点云延迟 |
| `enable_rviz` | true | 启动 RViz |

### 启动顺序

```
1. 传感器: rslidar_sdk, hipnuc_imu, tracer_base (可选)
2. SLAM:   fastlio_mapping + 静态 TF (odom→camera_init)
3. 转换:   odom_frame_converter, odom_fusion_sync, delayed_cloud_relay
4. 模型:   robot_state_publisher, joint_state_static, body→base_link TF
5. 地图:   globalmap_publisher → hdl_global_localization → hdl_localization
6. 导航:   tf_republisher, laserscan, map_server, Nav2 (5s 延迟启动)
7. 可视化: RViz
```

---

## 12. 关键文件路径

| 组件 | 路径 |
|------|------|
| 主 Launch | `src/slam/launch/hdl_navigation_launch.py` |
| Fast-LIO Launch | `src/slam/launch/fastlio_odom_launch.py` |
| Nav2 配置 | `src/slam/config/nav2_minimal_params.yaml` |
| 全局定位配置 | `src/hdl_global_localization/config/general_config.yaml` |
| SC-PGO 配置 | `src/slam/config/sc_pgo_params.yaml` |
| HDL 定位节点 | `src/hdl_localization/src/hdl_localization_node.cpp` |
| 位姿估计器 | `src/hdl_localization/src/hdl_localization/pose_estimator.cpp` |
| 全局定位节点 | `src/hdl_global_localization/src/hdl_global_localization_node.cpp` |
| SC 引擎 | `src/hdl_global_localization/src/hdl_global_localization/engines/global_localization_sc.cpp` |
| 里程计融合 | `src/slam/scripts/odom_fusion_sync.py` |
| TF 重发布 | `src/slam/scripts/tf_republisher.py` |
| URDF | `src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_description.urdf` |
| 地图目录 | `/home/didi/workspace/MobileManipulator2/maps/sc_pgo/` |

---

## 13. 构建与调试

### 构建

```bash
# 全量构建
colcon build

# 单包构建 (改配置后)
colcon build --packages-select slam --symlink-install

# 改 C++ 后 (HDL 内存敏感, 单线程编译)
colcon build --packages-select hdl_localization --parallel-workers 1
```

**构建顺序**: `hdl_global_localization` → `hdl_localization` (服务依赖)

### 调试命令

```bash
# 查看 TF 树
ros2 run tf2_ros view_frames

# 检查具体 TF
ros2 run tf2_ros tf2_echo map base_link

# 话题频率
ros2 topic hz /fastlio/odom

# 定位状态
ros2 topic echo /hdl_localization_node/status
```

### 建图流程

```bash
# 1. 启动建图
ros2 launch slam sc_pgo_mapping_launch.py

# 2. 遥控/手动驱动机器人遍历场景

# 3. 建图完成后, 生成 SC 数据库
ros2 run hdl_global_localization build_sc_database

# 4. 用 hdl_navigation_launch.py 启动导航
ros2 launch slam hdl_navigation_launch.py map_dir:=/path/to/maps/sc_pgo
```
