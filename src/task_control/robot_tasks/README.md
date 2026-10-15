# GC / NC 任务控制

本包提供任务控制（GC）、目标接近（NC）、RGB-D 感知和 robot02 常驻运行环境。
GC/NC 不调用旧 `board_approach` 的目标检测、法向估计或舱门拟合；常驻运行环境
复用其底盘保护和 Nav2 参数配置工具，不修改旧模块源码。

## 职责与接口

| 模块 | ROS 接口 | 职责 |
| --- | --- | --- |
| GC | `/gc/run_task`，`RunTask` action | 编排检测、接近、操作与多目标循环 |
| NC | `/nc/approach`，`Approach` action | 查询目标或接收位置、选择停靠点、调用 Nav2、检查到位 |
| 感知 | `/gc/detect_scene`，`DetectScene` service | VLM 检测、双目测距、转换到 map 坐标系 |
| 操作适配器 | `/manipulation/operate`，`OperateTarget` action | 预留洗衣机操作与抓取接口，尚未接入实机实现 |

GC 支持两个任务：

- `operate_washer`：查询洗衣机 → 正面对准 → 调用洗衣机操作适配器。
- `inspect_objects`：按完整 `description` 检测 → 双目测距 → 选择未处理目标 → 接近 → 抓取 → 重新观察。描述必填，例如“地上的水桶”；不拆关键词、不限定固定类别。暂不前往下一巡检点。

巡检检测失败最多尝试三次，不把错误当作空场景；连续两次成功观测均无可处理目标时结束。
按名称及 map 三维位置关联处理记录，默认关联半径 12 cm，单目标失败两次后跳过，
有失败时结果为 `PARTIAL`。紧邻、移动物体和深度波动较大的物体仍可能关联不准，
当前记录不等同于永久语义地图。

NC 有两种模式：

- `front_align`：输入物体名称。`SemanticMapMock.lookup_front_pose(name)` 模拟按名称查询 Semantic Map，返回 `[x,y,z,outward_yaw]`。这是物体正面参考点，+X 朝外。目标停靠位置沿外法向偏移“车头间距＋车头前伸”，车头朝向物体。后续替换查询实现即可接入 Semantic Mapper。
- `grasp_ready`：输入带采集时间的 map 坐标 `PointStamped`，默认最长 45 秒；无需图像或物体朝向。采样五个面向物体的停靠点，用 `ComputePathToPose` 检查并选择最短可达路径，再调用 `NavigateToPose`。

`grasp_ready` 保留接口名称，目前表示**导航停靠条件满足**，不是机械臂可达性验证。
默认 `grasp_front_clearance=0.5 m`，从物理车头算起；robot02 的
`front_extent=0.434 m`，因此 base_link 到目标的距离是 **0.934 m**。
Nav2 成功后，NC 用新鲜 TF 检查位置（5 cm）、朝向（约 1°）和目标在底盘坐标中的停靠范围。
物体位置来自可见表面深度，不是物体质心；真实抓取仍需近距离定位和工作空间验证。

## 构建与环境

在工作空间根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select robot_task_interfaces robot_tasks
source install/local_setup.bash
```

依赖现有 Nav2、定位、`board_approach`、`tracer_base`，以及相机启动脚本和标定文件。
常驻会话用于 robot02 Linux 环境，不会自动启动 Nav2、定位或 RViz，也不配置开机自启。
实机终端须沿用机器人一致的 ROS 配置（当前 domain 为 144），不要混入下面的 mock 域。

## 常驻运行与重复任务

先启动定位和 Nav2、确认定位有效，再启动一次会话：

```bash
ros2 run robot_tasks session start --skip-grasp
```

会话管理 GC、NC、感知、顶置相机、底盘 driver、CAN2 和速度保护节点；重复启动不会重启节点。
默认工作空间为 `~/junbin.zhang/MobileManipulator2`，其他路径使用
`session start --workspace <工作空间绝对路径> --skip-grasp`。
会话使用该工作空间内的 `scripts/run_top_camera_rsusb.sh` 启动相机。

之后每次只提交任务：

```bash
# 巡检导航测试：完成导航后跳过机械臂操作
ros2 run robot_tasks task inspect_objects "垃圾桶" --execute
ros2 run robot_tasks task inspect_objects "饮水机" --execute

# 洗衣机仅接近对准，不调用尚未实现的操作后端
ros2 run robot_tasks approach 洗衣机 --execute
```

省略 `--execute` 只规划、不发送导航或操作目标，结果为 `PREVIEW`，完成数量为零。
巡检预览仍调用真实 VLM API。`--skip-grasp` 仅影响巡检，成功码为
`NAVIGATION_ONLY_COMPLETED`，到位数量不表示抓取成功。
完整 `task operate_washer --execute` 仍要求真实操作适配器。

```bash
ros2 run robot_tasks session status
ros2 run robot_tasks session hold   # 锁住运动，保留节点，当前导航取消
ros2 run robot_tasks session reset  # 故障消除且导航结束后复位，不启动任务
ros2 run robot_tasks session stop   # 关闭本会话及 CAN2，保留 Nav2/定位/RViz
```

任务完成、普通失败和取消均保留常驻节点；切换 `skip_grasp` 需显式停止后重新启动。
会话拒绝接管已经被其他进程占用的底盘、相机或任务控制器。
会话占用 CAN2 期间不要同时运行旧底盘控制脚本。

## 配置分层

- `config/controllers.yaml`：通用控制参数；默认不允许执行，洗衣机为示例 pose，禁止真实执行示例目标。
- `config/robot02.yaml`：robot02 的实测洗衣机 pose、车头尺寸、相机话题/标定路径和联调深度阈值。常驻会话及默认 launch 在通用参数之后加载它，保持原有 robot02 配置值。
- `config/washer_pose_evidence.json`：实测 pose 的来源、离散范围及地图哈希，路径相对于仓库根目录。

洗衣机 pose 为 `[-0.2352384763, 2.3744471805, 0.2588673834, -1.7791760731]`，
来自 `20261013_140943_123577` 的 12 帧有效观测中位数。它仅对应
`maps/auto_explore/remap_20261013_1349/navigation`，并非测量真值。
程序尚未自动校验地图哈希；换地图或移动洗衣机后必须更新记录。NC 不在线拟合法向。

相机标定路径相对于显式传入的 `workspace`，也支持绝对路径。
robot02 使用 `rslidar <- top_color_optical_frame` 外参合成采集时的 map←camera 变换，
不修改全局 TF 树。更换相机或安装位置后应重新标定。

### 只启动控制器（已有外部运行环境时）

`controllers.launch.py` 只启动 GC/NC/感知，不启动硬件、CAN、Nav2 或定位；
不要与常驻 session 同时启动。独立预览需要已有相机、定位和 Nav2：

```bash
# 在工作空间根目录执行；默认加载 robot02 配置
ros2 launch robot_tasks controllers.launch.py workspace:="$PWD"
```

另一个同域终端可运行不带 `--execute` 的任务。若使用其他平台，可设置
`profile:=<配置文件绝对路径>`；`profile:=''` 禁用 robot02 覆盖，仅加载通用配置。
`camera_extrinsics_file:=<路径>` 可显式覆盖 profile 中的标定路径。
独立执行还需 `allow_execution:=true`、外部受保护的底盘链路及操作适配器，
巡检导航测试可加 `skip_grasp:=true`。通常优先使用常驻 session。

## 感知与深度

先取得 RGB、对齐 RGB 的双目深度、内参和采集时的 TF 快照，再调用 DeepSeek。
计算三维位置始终使用这份快照，不使用模型返回时的图像或机器人位姿。
检测完整物体框，不要求正面；模型按整段任务描述返回多个实例。

密钥读取自 `DEEPSEEK_API_KEY` 或 `~/.config/robot-object-approach/deepseek.json`（权限 600），
不要把密钥文件放入仓库。证据保存在 `~/.ros/robot_tasks/perception/`，
包括 RGB、深度、框、内参、TF 和结果 JSON。

双目深度范围为 0.2–10 m，使用框中央 ROI，检查有效比例、离群值与混合深度。
无有效深度时返回错误，不猜测距离、不回退 LiDAR。`depth_max_spread_m` 是
ROI 内 P90−P10 深度差上限：通用配置为 0.12 m，robot02 联调覆盖为 **1.0 m**。
后者仅用于跑通流程，不代表测距精度达到要求。常驻节点可即时调整：

```bash
ros2 param set /scene_perception depth_max_spread_m 1.0
```

每帧 `quality.max_spread_m` 记录实际阈值；参数的重启默认值来自配置文件。
接近期间假定目标静止，尚未进行运动中的视觉跟踪。

## 运行保护、取消与日志

NC 导航时持续发送运动许可，超过 0.5 秒未收到许可即停止放行。
空闲输出零速度，driver 指令看门狗为 0.4 秒。会话临时设置 Nav2 低速和到位容差，
停止时尝试恢复原参数；NC 本身不写 Nav2 参数。子进程异常退出会关闭会话拥有的硬件。

定位要求收敛、数值有效、内点比例至少 80%，并检查定位、里程计、扫描数据的新鲜度。
可用 `ros2 param set /board_approach_io localization_min_inlier_fraction 0.80` 调整；
这不会清除已触发的停车故障，也不修改旧 board_approach 的默认门槛。

CAN 超过 0.4 秒无新鲜反馈时立即输出零速度，暂时保留 Nav2 任务；
连续三个正常帧（间隔不超过 0.3 秒）后恢复。等待恢复超过 2 秒则终止任务，
急停、故障码或异常控制模式立即终止。恢复前清除缓存指令，不重置行驶距离预算。
CAN 接收使用内核时间戳，避免将队列中的旧帧误认为新反馈。

CLI 的 Ctrl+C 向下游传播取消；若无法确认停止，控制器进入锁定状态。
每个控制器只接受一个任务，其他导航入口仍需上层统一仲裁，当前检查不是跨客户端原子锁。
保护故障需确认原因消除且导航结束后执行 `session reset`。

日志位于 `~/.ros/robot_tasks/`：

- `<时间>.json`：任务目标、重试、结果与事件。
- `session/`：会话状态、各节点日志、CAN 帧记录与故障证据；`fault.json` 为最近一次故障快照。
- `perception/`：感知证据。
- `planning/`：规划候选、路径、costmap 和 footprint 快照。

## RViz

使用 `config/nc_navigation.rviz`：包含 `/map` 原始占用栅格、定位、路径、costmap
及 `/nc/planning_debug`。规划 Marker 使用 Transient Local、每秒重发、lifetime=0。

各候选点以 `candidate_N` namespace 区分：青色为起点/选中点，绿色为可达，
红色为失败，粉色为目标物体；轮廓使用 Nav2 全局 costmap 的 footprint 和 padding。
红色栅格是 cost=254 障碍物，橙色是 253 内切膨胀区，蓝灰色是 255 未知区。
`grid/*` 可单独关闭；这些是最近一次规划快照，实时障碍请看 Nav2 costmap 图层。
可视化失败不改变规划或控制。

## 操作适配器契约

`OperateTarget` 的 `operation=operate_washer` 用于洗衣机操作，`operation=grasp`
用于指定实例的近距离确认、抓取及放置/释放，收臂后才返回成功。
适配器必须支持取消；只有机械臂已停止且收好、可以继续导航时，才能返回可跳过的
`GRASP_FAILED`。其他操作故障应使用其他错误码，使 GC 停止整项任务。

目前只有接口和隔离 mock，未接入真实机械臂操作，也未复用旧“抓取上次 Observe 队列”接口。
GC 在导航前检查操作接口，`skip_grasp` 巡检导航测试除外。

## 隔离演示与验证

两个终端都 source 工作空间并设置隔离域，绝不能在机器人 domain 144 启动 mock：

```bash
export ROS_DOMAIN_ID=166 ROS_LOCALHOST_ONLY=1
ros2 launch robot_tasks demo.launch.py
```

另一个同域终端：

```bash
ros2 run robot_tasks task operate_washer --execute
ros2 run robot_tasks task inspect_objects "地上的水桶" --execute
```

此处 `--execute` 只驱动假后端，不启动底盘、机械臂或付费 VLM。mock 强制限定域 166。
测试命令（工作空间根目录，ROS 环境已加载）：

```bash
ROS_DOMAIN_ID=166 ROS_LOCALHOST_ONLY=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m pytest src/task_control/robot_tasks/test -q
```

测试覆盖几何、深度拒绝、模型结果解析、目标记录、ROS action 循环/取消、运行保护、
配置分层及相机快照一致性。mock 成功不等于实机抓取成功。
目前车头 0.5 m 停靠已通过规划预览，尚未完成该配置下的实机到达验收；
CAN 暂停恢复已做隔离测试，尚未通过实机主动制造反馈中断来验收。

## 巡检视觉模型

默认使用千问 `qwen3-vl-plus`，读取 `~/.config/robot-object-approach/qwen.json`
中的 `api_key`、`model`、`region`（beijing 或 singapore）；配置权限须为 600。
可用 `DASHSCOPE_API_KEY`、`QWEN_MODEL` 覆盖。若需恢复 DeepSeek，
在启动会话前设置 `ROBOT_TASKS_VLM_PROVIDER=deepseek`。切换代码后重启会话生效。
识别框仍使用 0–1000 归一化坐标；截断、异常响应不会当成无目标。
