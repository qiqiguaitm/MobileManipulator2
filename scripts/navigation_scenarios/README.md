# 导航场景采集

在 robot02 上一键采集（需要机器人开机，遥控由人操作）：

```bash
cd /home/didi/junbin.zhang/MobileManipulator2
bash scripts/navigation_scenarios/capture.sh first_scene --duration 20
```

入口会检查地图文件和磁盘、检查/启用 500 kbit/s 的 can2，使用现有
`hdl_navigation_launch.py` 启动 LiDAR、IMU、FastLIO、HDL 定位、机器人 TF、
底盘轮速与融合里程计、地图和 Nav2。明确传入 2026-10-13 新地图目录。
CAN 启用使用 `sudo -n`，系统未授权时明确失败，不保存密码。
不会下发导航目标或运动指令；等待 `LOCALIZATION READY` 和 `READY` 后再遥控。
`--duration` 从定位检查通过、正式建包时开始计算；Ctrl-C 提前结束。
结束后停止本脚本启动的节点并自动离线验包，CAN 保持已启用状态。

已有硬件进程时默认拒绝重复启动（包括其他 ROS 域）；先确认是本次使用的完整链路，
再使用 `--existing-stack`。该模式不启停既有节点，也不改 CAN。

```bash
bash scripts/navigation_scenarios/capture.sh straight_stop --duration 60 --existing-stack
```

录制门槛：`/status` ScanMatchingStatus 收敛且内点比例 ≥ 0.80（同项目 RuntimeHealth），
地图逐格匹配指定 map.yaml/PGM，scan/odom/TF 新鲜、scan 时间戳可变换到 map，
连续正常 3 秒才创建 bag。等待默认最多 180 秒，失败不创建正式场景。
期间若定位或数据检查持续失败超过 1 秒，停止录制并标记 incomplete。
数值门槛不能排除错误地点匹配；开录前仍应在 RViz 确认位置和点云对齐。
地图直接录入，不做离线建图；不运行闭环仿真或自主任务。

下述为仅检查/仅录制的低层入口（不会启停其他节点）：

```bash
cd /home/didi/junbin.zhang/MobileManipulator2
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=144 ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/didi/junbin.zhang/MobileManipulator2/log/rviz_cyclonedds.xml
/usr/bin/python3 scripts/navigation_scenarios/record.py --check \
  --expected-map maps/auto_explore/remap_20261013_1349/navigation/map.yaml
```

低层录制同样等待定位就绪，再创建 bag（默认 120 秒超时）：

```bash
/usr/bin/python3 scripts/navigation_scenarios/record.py straight_stop \
  --description '空旷直线行驶并停车'
/usr/bin/python3 scripts/navigation_scenarios/record.py obstacle_turn \
  --description '绕过静态障碍物或通过直角转弯'
```

每次生成独立的 `data/navigation_scenarios/<name>_<UTC>/`，包含 `bag/` 和
`scenario.json`。加 `--duration 60` 可定时结束。定位使用 `/odom/fused`，没有融合里程计时
显式使用 `--odom-topic /odom`。`--goal X Y YAW` 指定地图坐标系中的评测目标（米、弧度），
会写入 `/scenario/goal`；未指定时目标为空，不能把遥控终点直接视为任务目标。

原始话题在 bag 内统一加 `/reference` 前缀：`/map`、`/scan`、`/tf`、`/tf_static`、
所选里程计、`/cmd_vel`、`/goal_pose`、`/initialpose`、`/status`。不会把前缀话题发布到在线系统。
扫描和 TF 保留原时间戳。`/reference/trajectory` 是约 20Hz 的 PoseStamped 流，
由 map→base_link 最新公共 TF 时间提取，不积分 cmd_vel；同一 TF 时间不会重复写入。
首个轨迹位姿另写为 `/scenario/start`。
开录时补写就绪检查期间最近 2 秒 TF 和缓存的地图/全部静态变换，保留消息内部原时间戳，
保证第一批带延迟的扫描可以查询变换；这些预备 TF 的 bag 接收时间为开录时刻。

地图和静态 TF 使用 reliable/transient-local 订阅，能够接收开始录制前已发布且仍由
存活发布者缓存的数据；若现场发布者不提供这些 QoS，检查会失败，需检查发布链路。
结果 bag 为 sqlite3，不依赖额外 MCAP 插件；保留各话题 QoS 元数据供以后回放使用。

结束后检查：

```bash
ros2 bag info data/navigation_scenarios/<name>_<UTC>/bag
cat data/navigation_scenarios/<name>_<UTC>/scenario.json
/usr/bin/python3 scripts/navigation_scenarios/verify.py data/navigation_scenarios/<name>_<UTC> \
  --expected-map maps/auto_explore/remap_20261013_1349/navigation/map.yaml
```

必须有地图、scan、TF、静态 TF、里程计和轨迹；缺数据返回码为 2，状态为 `incomplete`。
`captured_needs_review` 只表示基本输入存在、至少一次扫描可按其时间戳变换到 map，
不代表全段 TF 完整、地图覆盖充分或已通过导航评测。还需检查实际运动、数据断流和
定位跳变。静止链路试录不能当作行驶场景。cmd_vel 在遥控器直接走 CAN 时可能为空，
不会伪造控制指令。扫描和轨迹使用同一定位系的前提也需核对。

第一批建议：每个场景约 30–90 秒，开头/结尾静止数秒，先录静态障碍物，必要时
从多角度观察；保存失败案例。地图外和未观测区域不视为可通行。
验包会离线读取独立 TF Buffer，检查各扫描时刻 TF 覆盖率、消息数量、时间倒退，
并逐格核对指定二维地图；输出 verification.json。不会向在线系统发布回放数据。
该工具仅负责采集和数据检查；控制器闭环仿真、目标后补和导航评测器尚未实现。

## 巡检导航任务录制

GC/NC 巡检使用 robot_tasks session 的速度保护和独占底盘驱动。不要用默认 capture.sh
同时启动另一个底盘驱动。导航应使用 `launch_chassis:=false use_odom_fusion:=false
use_odom_fused:=false`，定位就绪后由 `ros2 run robot_tasks session start --skip-grasp`
管理相机/底盘/GC/NC。该模式为检测和接近，不能标记为实机抓取。

在已有健康 session 中，低层 record.py 支持 `--extra-topic /话题`（可重复），
按发现的实际消息类型录入 `/reference/话题`，指定话题无法发现会报错。
`--task-description 'inspect_objects:垃圾桶;skip_grasp=true'` 只将意图写进
`/scenario/task`，不执行任务。可录 `/gc/run_task/_action/feedback`、
`/gc/run_task/_action/status`、`/nc/approach/_action/feedback`、`/plan`、`/local_plan`、
`/board_approach/safe_cmd_vel`，并将任务 CLI 结果和感知证据归档到场景目录。
实际任务需要在录包 READY 后单独提交：`ros2 run robot_tasks task inspect_objects "垃圾桶" --execute`。

## 可视化回放控制窗口

在 robot02 桌面运行 `scripts/navigation_scenarios/view_bag.sh [场景目录或 bag 目录]`。
窗口支持打开文件夹或 `metadata.yaml`，播放、暂停、回到 0 秒，以及前进约 1 秒。
打开后默认暂停，点击播放后 RViz 显示地图、激光、RGB、位姿与累积轨迹。
前进按钮依据 `/clock` 停止，存在调度延迟，并非精确逐帧定位。

默认使用独立 ROS 域 147；可用 `--domain` 改为其他空闲域，拒绝实机域 144。
只回放地图、TF、激光、轨迹和 RGB 白名单，绝不回放控制指令或导航任务。
同时支持本工具的 `/reference/*` topics 和相应无前缀 topics。
关闭控制窗口会关闭其启动的 RViz、轨迹显示节点和 bag 播放器，不关闭实机进程。
回到 0 秒会重新打开 RViz，清除上一段图像和轨迹缓存。
日志在 `~/.ros/navigation_bag_viewer/`。依赖 ROS 2 Humble、RViz2、PyQt5、PyYAML。

裁剪：`python3 scripts/navigation_scenarios/trim.py <场景目录>` 按保存的确认接收时间
保留正式区间，另存 `_trimmed`。不修改原 bag；补齐地图、静态 TF、相机标定及
2 秒动态 TF 上下文。确认接收时间不是 UI 点击的精确时间，原始数据始终保留。

## 测试集清单

代码库中的 `scripts/navigation_scenarios/test_set.yaml` 是测试集成员的唯一清单。
只有 `scenes` 下 `enabled: true` 的条目已正式收录；pending/deferred 条目不参与测试集。
新录制完成后由用户选择是否加入，再修改对应 enabled 和 inclusion_decision 字段。
暂不加入不会删除 bag。试录数据和 `simulation_runs` 结果不会因出现在数据盘而自动入集。

`scene_dir`、`original_scene_dir`、`baseline_result_dir` 相对于 `data_root`；
`map_config` 相对于代码库根目录。`scene_dir/bag` 是实际 rosbag 目录。
迁移数据盘时修改 data_root 即可，稳定的场景 id 不变。大文件 bag 不随代码提交。
目前单次测试脚本仍接收显式场景路径；此清单维护成员关系，不会自动启动批量测试。

## 首版近似闭环测试

`prepare_test.py <scene> <output>` 使用 bag 中同步 RGB-D、内参、原始 TF 和
robot02 当前相机外参，通过真实千问检测与项目已有深度定位生成物体位置和候选停靠点。
外参副本、检测图像和响应摘要保存在 output，不保存 API key。

`test_navigation.py <scene> <output>` 必须在 `ROS_DOMAIN_ID=149` 中运行。
output 需要包含 perception.json 及只读导出的 controller_server.yaml；可额外提供
local_costmap__local_costmap.yaml 和 global_costmap__global_costmap.yaml。
程序只启动 NavFn、TEB 和 lifecycle manager，不启动硬件驱动、定位、GC 或机械臂。
在可达候选中选择最短路径，通过 FollowPath 执行一次接近；本版不运行整个巡检循环、
行为树恢复或沿途重复检测。结果保存 result.json、trajectory.csv、result_bag 和节点日志。

模型使用理想差速运动学、0.4 秒指令超时。静态场景近似观测来自原始 scan 经录制 TF
转换的地图点，按当前仿真位姿匹配参考扫描，再投影到虚拟激光坐标；空射线保持未知。
不能生成新视角的遮挡信息，也不能代表动态障碍。距匹配参考位姿超过 0.5 米或
朝向差超过 45 度时停止并标记观测近似失效；这些是首版工程限制，不是已验证安全边界。
碰撞检查使用矩形足迹与地图占用栅格中心、匹配激光点，结果只表示该近似模型内的情况。
轨迹误差按空间最近点计算，不能替代任务语义与终点朝向评价。
