# 实机复测

## 使用入口

仅使用项目 `scripts/go_to_board.sh`。包目录不保留重复启动脚本，运行不依赖 `_cc_` 临时脚本、历史检测图或过去运行的目标文件。

Mac 上检查，不行驶：

```bash
ssh -t didi@192.168.112.106 'bash /home/didi/junbin.zhang/MobileManipulator2/scripts/go_to_board.sh --check'
```

Mac 上实际执行：

```bash
ssh -t didi@192.168.112.106 'bash /home/didi/junbin.zhang/MobileManipulator2/scripts/go_to_board.sh'
```

实际执行需定位与 Nav2 就绪、相机能看到目标板、现场有人看护。运行账户可能要求输入 sudo 密码。
已有底盘驱动或 CAN 被其他程序占用时拒绝接管。Ctrl+C 结束任务；紧急情况使用实体急停。

## 本次要验收的流程

1. 从现场选择的起点运行一次命令，中间不手工框选、发目标或重启程序。
2. 远处编号不可读时，观察程序是否自动进行板型/深度粗检。首次搜索约 30 秒，期间停车。
3. 粗接近在路径允许时每段最多 1.5 米，停稳后重新观察；读取编号后自动进入精对正。
4. 最终由三个有效 RGB-D 观测确认：距离误差不超过 10 厘米、横向误差不超过 10 厘米、朝向误差不超过 5°。
5. 结果应为 `arrived`；起点本来已经到位时应为 `already_arrived`，且不启用底盘。
6. 结束后底盘停止、导航目标清理、本程序启用的 CAN 关闭。

目标不明确、深度或定位异常、路径不通时，正确结果是 `stopped` 和非零退出码；不要反复强制执行。
`--check` 返回 `checked` 只表示观测和规划检查完成，不能代表已经到位。

## 查结果

命令开始时打印本次报告目录：`log/board_approach/<时间>/`。

- `result.json`：最终状态、失败原因或到位测量。
- `stage_*/observation.png`、`rgb_*.png`：每段观察画面。
- `stage_*/observations.json`、`goal.json`：实时测量及本次规划目标。
- `stage_*/motion.json`、`motion.driver.log`：运动状态和底盘日志。
  `motion.json` 的 `events` 含转向/直行进度和传感器暂停/恢复记录；持续超时会记录具体失败原因。

报告中的地图坐标是本次实时测量转换后的诊断记录，不作为后续运行的目标输入。
没有固定板子地图坐标配置；下次运行重新从相机检测。

## 开发验证（不会运动）

在 robot02 项目目录中：

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash
/usr/bin/python3 -m pytest src/navigation/board_approach/test -q
g++ -std=c++11 -I src/tracer2/tracer_base/include src/navigation/board_approach/test/test_command_watchdog.cpp -o /tmp/board-command-watchdog-test
/tmp/board-command-watchdog-test
```

硬件动作和分段控制已有先前实测记录；新版无人工衔接的远距离完整流程仍待上述复测。不要把离线测试或 `--check` 当作完整实机验收。
