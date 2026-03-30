# CPU 资源分析报告

**日期**: 2026-03-17 (更新)
**平台**: Jetson Orin (12 核 ARM A78AE, 30.7GB RAM)
**数据源**: 3 次采集 — 优化前 `make full-manual`、优化后 `make full-manual`、优化后 `make full-auto`

---

## 1. 优化效果总览

| 指标 | 优化前 (full-manual) | 优化后 (full-auto) | 变化 |
|------|--------------------:|-------------------:|-----:|
| **CPU avg** | 95% | **86%** | **-9%** |
| **CPU TOTAL** | 911% | **821%** | **-90%** |
| **RAM** | 5394 MB | **4517 MB** | **-877 MB** |
| **进程数** | 41 | 37 | -4 |
| 温度 | 75°C | 75°C | - |

### 核心改进

| 改动 | 节省 CPU | 节省 RAM |
|------|--------:|--------:|
| 模式切换: 消除冗余控制节点 | **~94%** | **~370 MB** |
| DepthSensor 按需激活 | **~22%** | - |
| 50Hz 轮询→callback + 2线程 + 2Hz status | (被 DepthSensor 掩盖) | - |
| RViz QoS 修复 (costmap/globalmap 可见) | 0 | 0 |

---

## 2. 已完成优化 (P0)

### P0-1: 模式切换 — 消除冗余控制节点 ✅

**问题**: `make full-manual` 同时启动 ManualControlNode 和 CleanerManagerNode，各自创建独立的 ApproachNavigator + DepthSensor，但同一时刻只有一个在工作。

**修复**:
- launch 文件新增 `mode` 参数 (`manual` / `auto`)
- `make full-manual` → 只启动 ManualControlNode + ObjectSelectorNode
- `make full` / `make full-auto` → 只启动 CleanerManagerNode
- 文件: `manual_control_full.launch.py`, `Makefile`

| 消除的进程 | 原 CPU% | 原 RAM |
|-----------|--------:|-------:|
| manual_control_node (auto 模式) | 91% | 290 MB |
| object_selector_node (auto 模式) | 3% | 79 MB |

### P0-2: DepthSensor 感知线程按需激活 ✅

**问题**: `DepthSensor._perception_loop()` 以 30Hz 无条件运行 RANSAC + DBSCAN + cKDTree 点云处理，即使导航空闲也在计算。

**修复**:
- 感知线程用 `threading.Event` 门控，默认暂停
- 仅在阶段3 (final approach) 由 `ApproachNavigator` 调用 `activate()` 唤醒
- 阶段3结束后 `deactivate()` 暂停
- 文件: `depth_sensor.py`, `navigator.py`

| 状态 | CPU 开销 |
|------|---------|
| 修复前 (idle) | ~91% / 实例 (30Hz RANSAC+DBSCAN) |
| 修复后 (idle) | ~0% (`Event.wait` 阻塞) |
| 修复后 (阶段3) | ~30% (正常工作) |

### P0-3: DepthSensor 深度订阅按需 ✅

**问题**: 修复 P0-2 后 CleanerMgr 仍 69% CPU — 深度图订阅 `_depth_callback` 始终接收每帧图像做 `cv_bridge` 解码 + float32 转换。

**修复**:
- `deactivate()` 销毁深度图和 camera_info 订阅
- `activate()` 重建订阅
- idle 时零 ROS 消息处理开销
- 文件: `depth_sensor.py`

### P0-4: 控制节点代码优化 ✅

| 修复 | 文件 |
|------|------|
| 50Hz 轮询→`future.add_done_callback()` | `manual_control_node.py` |
| 4 线程 executor→2 线程 | `manual_control_node.py`, `cleaner_manager_node.py` |
| status 10Hz→2Hz (manual) / 1Hz (cleaner) | 同上 |
| nice 优先级: 控制 +5, RViz +15 | `manual_control_full.launch.py` (prefix) |

### P0-5: RViz QoS 不匹配修复 ✅

**问题**: RViz 配置中 costmap 和 globalmap 的 Durability Policy 为 `Volatile`，但 Nav2 costmap 和 globalmap_publisher 使用 `TRANSIENT_LOCAL` 发布 → RViz 永远收不到初始数据。

**修复**: `cleaner_manager.rviz` 中三个 display 的 Durability Policy 改为 `Transient Local`:
- GlobalCostmap (`/global_costmap/costmap`)
- LocalCostmap (`/local_costmap/costmap`)
- GlobalMap (`/globalmap_visual`)

---

## 3. 当前模块 CPU 占用 (优化后, `make full-auto`)

| 排名 | 模块 | CPU% | RAM(MB) | 进程数 | 优先级 |
|------|------|-----:|--------:|-------:|--------|
| 1 | **Localization** | **166%** | 743 | 8 | ni=0 |
| 2 | **Percept-Viz** | **110%** | 415 | 2 | ni=0 |
| 3 | Percept-Multi | 77% | 347 | 1 | ni=0 |
| 4 | CleanerMgr | 69% | 270 | 1 | ni=+5 |
| 5 | Arm-Grasp | 67% | 113 | 1 | ni=0 |
| 6 | RViz | 66% | 639 | 1 | ni=+15 |
| 7 | Nav2 | 53% | 281 | 6 | ni=0 |
| 8 | Cam-Chassis | 39% | 247 | 1 | ni=0 |
| 9 | Cam-Hand | 39% | 243 | 1 | ni=0 |
| 10 | Arm-TF | 39% | 214 | 4 | ni=0 |
| 11 | Percept-Grasp | 27% | 167 | 1 | ni=0 |
| 12 | Cam-Top | 26% | 243 | 1 | ni=0 |
| 13 | Lidar | 16% | 115 | 3 | ni=0 |
| 14 | IMU | 9% | 69 | 2 | ni=0 |
| 15 | Chassis | 5% | 38 | 1 | ni=0 |
| 16 | Other ROS | 11% | 373 | 6 | ni=0 |
| | **TOTAL** | **~821%** | **4517** | **37** | |

> **注**: CleanerMgr 69% 为深度订阅按需优化前的数据。优化后预计降至 ~30-40%。

---

## 4. 待优化项

### P1 — 应尽快处理

| 问题 | 原因 | 建议 | 预期节省 |
|------|------|------|----------|
| **Percept-Viz 110%** | multi_camera_rviz 88% + perception_grasp_rviz 22% | 仅调试时启动，或迁移远程 | **-110% CPU** |
| **RViz 66%** | 不该跑在 Jetson 上 | 迁移到远程机器 | **-66% CPU, -639MB** |
| **tf_republisher 30%** | 100Hz 重发 map→odom TF | 降到 50Hz | **-15% CPU** |

**迁移 RViz + 关闭 Percept-Viz 后，理论节省 ~191% CPU (约 1.6 核)。**

### P2 — 中期优化

| 问题 | 原因 | 建议 | 预期节省 |
|------|------|------|----------|
| **piper_grasp_node 67%** | Python SDK CAN 通信轮询 | 检查 busy-wait | -30% |
| **Arm-TF 39%** | joint_state_relay 23% + joint_state_static 9% | 降低发布频率 | -15% |
| **Cam-Hand 39%** | D435 全流启动 | 仅在 observe 阶段启用 | -30% |
| **Percept-Grasp 27%** | 手部相机感知 idle 时仍在跑 | 按需启动 | -20% |

### P3 — 架构级优化

| 方向 | 说明 |
|------|------|
| **GPU 利用率仅 11-21%** | 所有 ROS 节点 CPU 计算，GPU 几乎空闲。考虑 NDT 匹配 / 点云降采样 CUDA 加速 |
| **Percept-Multi 77%** | 双相机感知最大 CPU 消费者。HTTP+序列化在 CPU，考虑本地 GPU 推理 |
| **Python GIL 瓶颈** | CleanerMgr/odom_fusion 等 Python 节点受 GIL 限制，关键路径考虑 C++ |
| **控制节点合并** | ManualCtrl 和 CleanerMgr 90% 执行逻辑相同，仅决策层不同，可合并为单节点+模式切换 |

---

## 5. 优先级配置

通过 launch prefix + `make renice` 实现:

```
nice -15  hdl_localization_node, fastlio_mapping         (定位核心, make renice)
nice -10  odom_fusion, odom_frame_converter              (里程计, make renice)
nice -10  controller_server, planner_server              (Nav2 路径, make renice)
nice   0  perception, camera, lidar, arm, nav2 其他       (默认)
nice  +5  manual_control_node / cleaner_manager_node      (控制, launch prefix)
nice +15  rviz2                                           (可视化, launch prefix)
```

> **注意**: launch prefix 的 nice 需要以 root 运行才生效。当前监控数据显示 NI=0，
> 说明 `make renice` 未在启动后执行。启动后需手动 `make renice`。

---

## 6. CPU 预算表

| 模块 | 优化前 | 当前 | 进一步目标 | 手段 |
|------|-------:|-----:|----------:|------|
| Localization | 231% | 166% | 136% | tf_republisher 降频 |
| Percept-Multi | 120% | 77% | 77% | 已达标 |
| Nav2 | 101% | 53% | 53% | 正常 |
| Percept-Viz | 78% | 110% | 0% | 迁移远程/按需 |
| ManualCtrl | 56% | 0% | 0% | 模式切换消除 |
| CleanerMgr | 57% | 69%→~35% | ~35% | 深度订阅按需 |
| Arm-TF | 55% | 39% | 25% | 降频 |
| Arm-Grasp | 50% | 67% | 35% | 排查轮询 |
| RViz | 46% | 66% | 0% | 迁移远程 |
| Cam (3台) | 81% | 104% | 75% | Hand 按需启停 |
| Percept-Grasp | 0.4% | 27% | 5% | 按需启动 |
| Lidar | 23% | 16% | 16% | 正常 |
| IMU+Chassis | 7% | 14% | 14% | 正常 |
| Other ROS | 6% | 11% | 11% | 正常 |
| **TOTAL** | **911%** | **821%** | **~482%** | |

**目标 CPU 利用率: 482%/1200% ≈ 40%，留出充足余量。**

---

## 7. 监控工具

```bash
# 实时 dashboard
make monitor       # 或: python3 scripts/_cc_resource_monitor.py

# 带日志记录
python3 scripts/_cc_resource_monitor.py --log

# 调整优先级 (启动后执行)
make renice

# 启动命令
make full-manual    # 手动模式 (ManualControlNode)
make full           # 自主模式 (CleanerManagerNode)
make full-auto      # 同上
```

---

## 8. 修改文件索引

| 文件 | 改动 |
|------|------|
| `src/navigation/approach_navigator/approach_navigator/depth_sensor.py` | 感知线程 Event 门控 + 深度订阅按需创建/销毁 |
| `src/navigation/approach_navigator/approach_navigator/navigator.py` | `_do_final_approach` activate/deactivate |
| `src/cleaner_manager/launch/manual_control_full.launch.py` | `mode` 参数, 条件启动控制节点 |
| `src/cleaner_manager/src/manual_control_node.py` | callback 替代轮询, 2线程, 2Hz status |
| `src/cleaner_manager/src/cleaner_manager_node.py` | 2线程 executor |
| `src/cleaner_manager/config/cleaner_manager.rviz` | costmap/globalmap QoS → Transient Local |
| `src/slam/config/nav2_minimal_params.yaml` | static_layer 显式 map_topic |
| `Makefile` | `make full` / `make full-auto` 新增 |
