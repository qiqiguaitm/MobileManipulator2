# 导航场景采集

## 数据目录

机器人数据根目录为 `/data/junbin.zhang/navigation_scenarios`：

```text
scenes/<场景ID>/                 正式场景，含 bag/ 和 scenario.json
pending/<名称_时间>/              新录制，等待决定是否入集
simulation_runs/<场景ID>/<运行>/  仿真结果
playback/                        回放配置与日志
archive/                         原始长录制、历史试录、验证数据和备份
remote_mcap/                     当前网页查看入口
```

新录制默认写入 pending/。可用 `NAVIGATION_SCENARIOS_ROOT` 指定数据根目录，
或用 `--output` 明确指定录制父目录；没有机器人数据盘时，回退到代码库的
`data/navigation_scenarios/pending/`。入集后移动目录并更新 test_set.yaml，不能仅凭目录自动入集。
2026-10-16 按用户要求清理历史数据，目前仅保留正式 trimmed 场景与最新第 06 次仿真结果。
pending/、playback/、archive/ 按需重新创建，不再保留旧试录和迁移备份。
remote_mcap 中的当前文件使用相对软链接，不复制数据；移动目标后需更新软链接。
当前网页入口为 `bin_approach_20261015_01.mcap`（正式场景）
和 `bin_approach_20261015_01_20261015T113004Z.mcap`（最新结果）。

在 robot02 上一键采集（需要机器人开机，遥控由人操作）：

```bash
cd /home/didi/junbin.zhang/MobileManipulator2
bash scripts/navigation_scenarios/capture.sh first_scene
```

入口会检查地图文件和磁盘、检查/启用 500 kbit/s 的 can2，使用现有
`hdl_navigation_launch.py` 启动 LiDAR、IMU、FastLIO、HDL 定位、机器人 TF、
底盘轮速与融合里程计、地图和 Nav2。明确传入 2026-10-13 新地图目录。
CAN 启用使用 `sudo -n`，系统未授权时明确失败，不保存密码。
不会下发导航目标或运动指令；等待 `LOCALIZATION READY` 和 `READY` 后再遥控。
默认检查通过后输出 `AWAITING START` 并等待，不创建 bag。开始按钮确认后，
向日志中的 `recorder_pid` 发送 `kill -USR1 <recorder_pid>`，等待录制 `READY` 后遥控。
结束按钮发送 `kill -INT <recorder_pid>`，关闭并校验 bag；不需要裁剪。
默认持续录制直到结束确认。可选 `--duration` 从正式建包开始计时；
`--auto-start` 仅用于无需交互确认的自动测试。
结束后停止本脚本启动的节点并自动离线验包，CAN 保持已启用状态。

已有硬件进程时默认拒绝重复启动（包括其他 ROS 域）；先确认是本次使用的完整链路，
再使用 `--existing-stack`。该模式不启停既有节点，也不改 CAN。

```bash
bash scripts/navigation_scenarios/capture.sh straight_stop --duration 60 --existing-stack
```

录制门槛：`/status` ScanMatchingStatus 收敛且内点比例 ≥ 0.80（同项目 RuntimeHealth），
地图逐格匹配指定 map.yaml/PGM，scan/odom/TF 新鲜、scan 时间戳可变换到 map，
连续正常 3 秒后允许开始录制。就绪检查默认最多 180 秒，失败不创建正式场景；
等待人工开始确认不受此超时限制，就绪状态持续复查。
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

每次生成独立的 `data/navigation_scenarios/pending/<name>_<UTC>/`，包含 `bag/` 和
`scenario.json`。加 `--duration 60` 可定时结束。定位使用 `/odom/fused`，没有融合里程计时
显式使用 `--odom-topic /odom`。`--goal X Y YAW` 指定地图坐标系中的评测目标（米、弧度），
仅保存到 scenario.json 的 goal 字段；未指定时目标为空，不能把遥控终点直接视为任务目标。

录制保留原始话题名，不添加 `/reference` 前缀：`/map`、`/scan`、`/tf`、`/tf_static`、
所选里程计、`/cmd_vel`、`/goal_pose`、`/initialpose`、`/status`。录制器不向在线系统发布消息。
扫描和 TF 保留原时间戳。`/current_pose` 是约 20Hz 的 PoseStamped 流，
由 map→base_link 最新公共 TF 时间提取，不积分 cmd_vel；同一 TF 时间不会重复写入。
首个位姿保存到 scenario.json 的 start 字段，不另建起点话题。
开录时补写就绪检查期间最近 2 秒 TF 和缓存的地图/全部静态变换，保留消息内部原时间戳，
保证第一批带延迟的扫描可以查询变换；这些预备 TF 的 bag 接收时间为开录时刻。

地图和静态 TF 使用 reliable/transient-local 订阅，能够接收开始录制前已发布且仍由
存活发布者缓存的数据；若现场发布者不提供这些 QoS，检查会失败，需检查发布链路。
录制、裁剪和仿真输出统一为 MCAP，保留话题 QoS、消息定义与索引。
依赖 ROS 插件 ros-humble-rosbag2-storage-mcap；缺失时明确报错，不回退到 SQLite。
读取通过 bag_io.py 自动识别存储格式，兼容旧 SQLite bag。

结束后检查：

```bash
ros2 bag info data/navigation_scenarios/pending/<name>_<UTC>/bag
cat data/navigation_scenarios/pending/<name>_<UTC>/scenario.json
/usr/bin/python3 scripts/navigation_scenarios/verify.py data/navigation_scenarios/pending/<name>_<UTC> \
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
采集和验包工具不执行导航；离线闭环测试使用 prepare_test.py 和 test_navigation.py。

## 巡检导航任务录制

GC/NC 巡检使用 robot_tasks session 的速度保护和独占底盘驱动。不要用默认 capture.sh
同时启动另一个底盘驱动。导航应使用 `launch_chassis:=false use_odom_fusion:=false
use_odom_fused:=false`，定位就绪后由 `ros2 run robot_tasks session start --skip-grasp`
管理相机/底盘/GC/NC。该模式为检测和接近，不能标记为实机抓取。

在已有健康 session 中，低层 record.py 支持 `--extra-topic /话题`（可重复），
按发现的实际消息类型、原话题名录入，指定话题无法发现会报错。
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

历史数据修复（不属于新录制流程）：`python3 scripts/navigation_scenarios/trim.py <场景目录>` 按保存的确认接收时间
保留正式区间，另存 `_trimmed`。不修改原 bag；补齐地图、静态 TF、相机标定及
2 秒动态 TF 上下文。确认接收时间不是 UI 点击的精确时间，原始数据始终保留。

## 测试集清单

代码库中的 `scripts/navigation_scenarios/test_set.yaml` 是测试集成员的唯一清单。
只有 `scenes` 下 `enabled: true` 的条目已正式收录；pending/deferred 条目不参与测试集。
新录制完成后由用户选择是否加入，再修改对应 enabled 和 inclusion_decision 字段。
暂不加入不会删除 bag。试录数据和 `simulation_runs` 结果不会因出现在数据盘而自动入集。

`scene_dir`、可选的 `original_scene_dir` 相对于 `data_root`；
`map_config` 相对于代码库根目录。`scene_dir/bag` 是实际 rosbag 目录。
共享清单现在以 `/data` 为根，同时收录 junbin.zhang 和 wangyibo 的场景；
垃圾桶场景解析后的绝对路径不变，不移动同事数据，也不改变采集的默认目录。
迁移数据盘时修改 data_root 即可，稳定的场景 id 不变。大文件 bag 不随代码提交。
目前单次测试脚本仍接收显式场景路径；此清单维护成员关系，不会自动启动批量测试。
清单不绑定某次仿真结果；结果保存在 simulation_runs 下，网页软链接独立维护。

红色灭火器场景的共享位置、校验及复用命令见
[灭火器场景说明](fire_extinguisher_20261016.md)。Git 仅提交清单、代码和说明，
完整录像保留在 robot02 的 wangyibo 数据目录；其他机器需另行获取该场景目录。

## 首版近似闭环测试

依赖机器人工作区的 `robot_tasks.geometry` 和 `robot_tasks.vlm`，需先构建并 source
对应工作区。检测类别由 `prepare_test.py --target` 指定，默认仍为“垃圾桶”；
灭火器场景需明确传 `--target 灭火器`，检测失败不回退其他类别。
`--snapshot-offset` 选择录像中的时刻（秒，默认 0），优先使用录制时保存的相机外参。

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
转换的地图点。仍按当前仿真位姿匹配参考扫描，但直接保留原始 ranges、角度和量程，
不再生成 sim_scan 或做角度分桶。回放 TF 为 map → reference/base_link → reference/scan，
使用录制时的完整底盘位姿和真实雷达外参，与仿真 map → odom → base_link 分离。
/scan 只更新时间戳和 frame_id，配套变换写入 /tf。仿真结果不重复保存原始
扫描和 TF；如需追溯原始时间戳及坐标系，读取原始场景 bag。
局部和全局 costmap 的该扫描源均 marking=true、clearing=false；静态障碍标记可累积，
不会将历史射线误用为仿真当前位置的自由空间证据。只修改本轮 test_params.yaml。
不能生成新视角的遮挡信息，也不能代表动态障碍。距匹配参考位姿超过 0.5 米或
朝向差超过 45 度时停止并标记观测近似失效；这些是首版工程限制，不是已验证安全边界。
碰撞检查使用矩形足迹与地图占用栅格中心、匹配激光点，结果只表示该近似模型内的情况。
轨迹误差按空间最近点计算，不能替代任务语义与终点朝向评价。

## Foxglove 远程文件与时间轴

`export_viewer_mcap.py <result_bag> <output.mcap>` 将已关闭的 MCAP 或旧 SQLite 仿真结果
转换为带消息定义、分块索引的 MCAP，保留源消息字节和录制时间。额外生成
缺少的 `/path`（每帧累积轨迹）、`/robot_box`，不再复制规划路径和目标。
新录制和仿真结果已包含显示数据，无需此导出步骤。要求 ROS Humble 环境、
`mcap`、`rosbags`；不支持缺少消息定义的自定义类型。

机器人当前依赖安装在数据盘 `.viewer_deps`，不修改系统 ROS Python 环境：

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/data/junbin.zhang/navigation_scenarios/.viewer_deps:$PYTHONPATH
python3 scripts/navigation_scenarios/export_viewer_mcap.py \
  /data/junbin.zhang/navigation_scenarios/simulation_runs/bin_approach_20261015_01/bin_20261015_03_reference_scan/result_bag \
  /data/junbin.zhang/navigation_scenarios/remote_mcap/bin_20261015_03_reference_scan.mcap
python3 scripts/navigation_scenarios/serve_mcap.py \
  /data/junbin.zhang/navigation_scenarios/remote_mcap --bind 192.168.112.106 --port 8770 \
  --allowed-root /data/junbin.zhang/navigation_scenarios \
  --public-base-url http://127.0.0.1:18770
```

转换目标必须不存在，原 bag 不变。文件服务只展示指定目录顶层的 `.mcap` 入口，拒绝子路径。
支持软链接，但真实目标必须是 `--allowed-root` 内的 `.mcap` 文件；未指定时只允许服务目录内的目标。
失效、循环、越界链接不会展示或提供下载。支持 HTTP Range、HEAD、OPTIONS 和 Foxglove 来源的 CORS。
访问 `http://192.168.112.106:8770/` 选择文件进入 Foxglove Remote file 模式。
此时底部时间轴直接暂停、播放、拖动，不使用 WebSocket 的服务调用按钮，也不需要
机器人端 ros2 bag play。服务仅用于可访问该机器人 IP 的受信任局域网。
浏览器端若要求 Foxglove 的本地网络访问权限，需由用户按浏览器提示允许。

### Mac 浏览器访问机器人文件

HTTPS Foxglove 直接请求机器人 HTTP 地址可能被 Mixed Content 拦截。
在 Mac 建立 SSH 转发，数据仍保存在机器人上：

```bash
ssh -fNT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -L 127.0.0.1:18770:192.168.112.106:8770 robot02
```

转发已运行时不重复执行。Mac 重启或隧道退出后重新运行。
机器人文件服务启动时加 `--public-base-url http://127.0.0.1:18770`，
使文件列表生成的 Foxglove 链接也使用转发地址，不再指向机器人 HTTP IP。
在 Mac 打开 `http://127.0.0.1:18770/` 选择文件；若 Chrome 请求本地网络权限，允许此站点访问。


## MCAP 存储与旧数据迁移

所有新录制的 `bag/`、仿真输出 `result_bag/` 内均为 `.mcap`，
同时保留 ROS 标准 `metadata.yaml`。仿真直接读取 MCAP，不需要中间 SQLite 文件。
测试集目录不变，清单无需因存储转换而改变场景 ID 或重新决定入集。

机器人未提供免密 sudo，因此使用官方 ROS deb 的独立解包安装：
`~/.local/opt/rosbag2_mcap` 指向数据盘的插件前缀。`bag_io.configure_storage()`
自动加载该前缀，也支持 `NAVIGATION_MCAP_PREFIX` 指定其他位置。
其他机器可正常安装 `ros-humble-rosbag2-storage-mcap`，无须独立前缀。
独立前缀环境下直接使用 `ros2 bag play` 前，需将该路径加入 `AMENT_PREFIX_PATH`，
并将其 `lib` 加入 `LD_LIBRARY_PATH`；本仓库查看器会自动配置子进程环境。

`python3 scripts/navigation_scenarios/migrate_mcap.py <bag目录> ...`
只处理已关闭且没有正在回放的 bag。转换完成后逐话题核对消息字节、时间戳、数量
和 QoS，再将旧目录保留为 `bag_sqlite3_backup`（或 `result_bag_sqlite3_backup`），
原目录切换为 MCAP。失败时不替换原始 bag；pending 目录用于排查，不视为成功数据。
迁移只改变存储格式，不把原本失败/不完整的场景标记为合格。


## 直接录入显示话题

新录制和新仿真都写入 `/path`（截至当前帧的累积
nav_msgs/Path）和 `/robot_box`（底盘轮廓 Marker）。只写入 bag，不向实机
发布显示消息。box 与位姿使用相同时间戳，尺寸沿用导航足迹，显示高度不代表机械臂。
参考轨迹直接用 `/reference/path`，规划路径用 `/plan`，导航终点用 `/goal_pose`，不再保存重复显示副本。
因此新原生 MCAP 可直接供 Foxglove 使用，无需 export_viewer_mcap.py 生成显示副本；
该导出工具只用于缺少这些话题的历史结果。裁剪时重新累计显示路径，避免预录轨迹混入。

仿真结果还直接保存检测选中的物体位置：`/detected_target/position`
（geometry_msgs/PointStamped，map 坐标系）和 `/detected_target`
（紫色球形 Marker，Foxglove 可直接勾选）。位置来自 perception.json 的
selected.xyz_map，不来自遥控终点。检测没有给出物体朝向，因此不伪造目标 Pose。
每帧保存同一个检测位置以支持回放显示，不代表每帧重新检测。
没有运行检测的原始遥控 bag 不含目标位置；仿真检测后将它写入结果 MCAP。

结果 bag 命名统一：`/current_pose` 为当前位姿（逐帧保存），`/path` 为累积实际轨迹，
`/cmd_vel` 为控制指令；`/plan` 为规划路径，`/goal_pose` 为导航终点。
只有原始录制参考数据使用 `/reference/*`。新结果不使用 `/simulation/*` 或 `/replay/*`。

仿真加载器 `scene_topics.py` 只接受保留原话题名的输入 bag；发现 `/reference/*` 输入时明确报错，不自动去前缀。
地图、扫描是算法输入，使用 `/map`、`/scan`；扫描的坐标系为 `reference/scan`，
原始 TF 只用于重建观测位置，禁止直接混入仿真机器人 TF 树。
参考回放采用 REFERENCE_TOPICS 白名单，原始位姿、规划、目标、控制指令、检测结果
只发布到 `/reference/*`，不驱动算法。原始参考轨迹显示为 `/reference/path`。
结果 writer 直接记录确定后的话题名，不进行重命名或复制显示副本。

## 维护与提交

### 录制后的轻量可视化文件

正常录制结束后自动生成 `visualization_bag/`（MCAP）及 `visualization.json`。
`bag/` 始终保留完整数据并用于仿真；可视化版不能替代仿真输入。
每个相机保留开头、中间、结尾最多 3 帧 RGB，深度等同相机图像按消息时间戳
匹配（最大差 50 ms），找不到匹配时不伪造或重复帧。其他话题完整保留，
消息内容、时间戳、话题名和 QoS 不变。画面在采样点之间会保持或为空，不代表连续视频。
生成失败不会覆盖原始录制；已有可视化目录不会被自动覆盖。

已有场景可手动生成：

```bash
python3 scripts/navigation_scenarios/preview.py <场景目录> --samples 3
```

capture.py 在验包通过后生成；直接运行 record.py 时在正常关闭 bag 后生成，
可用 `--no-preview` 跳过。生成期间不需要继续遥控。

| 文件 | 职责 |
|---|---|
| capture.sh / capture.py | 启动采集依赖、等待确认、录制后验包 |
| record.py / readiness.py | 原名录制、定位和数据就绪检查 |
| bag_io.py / scene_topics.py | MCAP 读写、输入名称检查、参考话题白名单 |
| display_messages.py | 轨迹、底盘方框、检测目标显示消息 |
| preview.py | 生成稀疏相机帧的轻量可视化 MCAP |
| prepare_test.py / test_navigation.py | RGB-D 检测准备、近似闭环测试 |
| verify.py / test_set.yaml | 数据完整性检查、正式测试集成员 |
| bag_viewer.py / trajectory_view.py / replay.rviz | RViz 回放与显示 |
| serve_mcap.py / replay_web_control.py | 文件服务、旧 WebSocket 回放控制 |
| trim.py / migrate_mcap.py / export_viewer_mcap.py | 历史数据修复和迁移 |

提交代码和测试集清单，不提交 bag、output/、tmp/ 或生成的报告。以下检查不依赖 ROS：

```bash
python3 -m unittest discover -s scripts/navigation_scenarios/tests
python3 -m compileall -q scripts/navigation_scenarios
bash -n scripts/navigation_scenarios/capture.sh scripts/navigation_scenarios/view_bag.sh
```

完整仿真在隔离 ROS 域中运行，运行前确认该域没有其他测试。

网页输入软链接命名为 `<场景目录名>.mcap`。仿真运行目录与输出软链接使用
`<场景目录名>_<YYYYMMDDTHHMMSSZ>`，时间取该次运行的 UTC 时间，Z 表示 UTC。
