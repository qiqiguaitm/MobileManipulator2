# 红色灭火器测试场景

用户于 2026-10-16 确认收录 `fire_extinguisher_approach_20261016_01`。
这是当天 11:04（北京时间）的人工遥控 `scene_02`，不是尚未录成的 `scene_03`。
录制约 34.55 秒、5.45 米。数据完整性通过，但此前接近目标的仿真规划失败；
用途是感知测试和规划失败回归，不是导航成功基准。

## 数据在哪里

数据保留在 robot02（192.168.112.106），不随 Git 提交：

```text
/data/workspace/wangyibo/navigation_scenarios_mcap/scenes/fire_extinguisher_approach_20261016_01/
  bag/                   完整 MCAP，仿真必须用它（约 2.39 GB）
  visualization_bag/     轻量 MCAP（约 17 MB），每路相机仅开头/中间/结尾三帧
  visualization.json     预览采样及字节校验报告
  scenario.json          采集元数据，goal 仍为 null
  verification.json      地图、消息数量和激光 TF 校验
  dataset_registration.json  入集决定、原数据位置、话题映射及 SHA-256
  capture_assets/        录制时地图、相机外参等副本
  capture_provenance.json 原录制版本及配置来源
```

同一机器人上的同事可直接读取，无须复制进自己的仓库。其他机器仅拉代码不会获得录像，
需另行传输完整场景目录；预览数据不能替代仿真输入。代码中没有密码或 API key。
清单的 `data_root` 为 `/data`，原垃圾桶条目的绝对数据位置保持不变。

完整 MCAP 的 SHA-256：

```text
10555ebdae3db09242d39090cfb9ed5ae2d11c4f13ee949f0a3aa30c8f09d741
```

原始 SQLite 场景位于
`/data/workspace/wangyibo/navigation_scenarios/scene_02_20261016T030409_571423Z`，未删除或改写。
它由旧版 `a07e0ee` 采集。本次只在衍生副本里去除 `/reference` 话题前缀、
将 `/reference/trajectory` 改为 `/current_pose`，以满足新版仿真输入规则。
全部 8503 条消息的序列化内容、接收时间、消息头时间、TF frame 和 QoS 均校验保留；
旧版额外的 `/scenario/start` 也保留，当前工具不依赖它。
`scenario.json` 更新了话题映射、计数键和任务标注，原始采集版本、起点、终点不改。
`capture_provenance.json` 中的 pending 是历史采集状态，当前入集决定以 `test_set.yaml`
和 `dataset_registration.json` 为准。

## 快速验证和查看

在自己的仓库根目录，先 source 已构建的 ROS Humble 工作区环境，然后：

```bash
SCENE=/data/workspace/wangyibo/navigation_scenarios_mcap/scenes/fire_extinguisher_approach_20261016_01
python3 scripts/navigation_scenarios/verify.py "$SCENE" \
  --expected-map "$SCENE/capture_assets/map.yaml"
```

预期 `passed_data_checks`，地图匹配、`scan_tf_coverage: 1.0`。
包含 RGB/对齐深度/两路内参各 518 帧，激光 346 帧，轨迹 347 帧。
没有预设导航目标的警告是预期情况；不能把遥控终点当作导航输入。

Foxglove 可打开 `visualization_bag/` 下的 `.mcap` 做轻量预览；
要看连续相机视频，选择 `bag/bag_0.mcap`。两者均支持文件模式的播放、暂停和拖动。
远程 HTTP Range 服务沿用 README 的 `serve_mcap.py`，无需先把完整文件下载到 Mac。
不要向实机 ROS 域 144 回放 bag 或运行仿真。

## 检测与仿真

`prepare_test.py` 已支持显式类别，灭火器要用 `--target 灭火器`。
未指定时为兼容原垃圾桶 demo 仍默认垃圾桶；不会因识别失败改找空气净化器。
此步骤会调用配置好的视觉服务并上传所选图像，需使用者具备相应服务访问权限和上传授权；
密钥只从运行环境配置获取，不写进 Git。

```bash
SCENE=/data/workspace/wangyibo/navigation_scenarios_mcap/scenes/fire_extinguisher_approach_20261016_01
RUN_DIR=$(mktemp -d /data/workspace/wangyibo/navigation_scenarios_mcap/simulation_runs/fire_extinguisher_XXXXXX)
ROS_DOMAIN_ID=149 python3 scripts/navigation_scenarios/prepare_test.py \
  "$SCENE" "$RUN_DIR" --target 灭火器 --snapshot-offset 0
```

默认采用场景保存的外参和 RGB-D 同步帧。后续按 README 为 `RUN_DIR` 准备 Nav2 参数，
再在空闲的隔离域 149 执行 `test_navigation.py "$SCENE" "$RUN_DIR"`。
新输出请放新目录，不覆盖既有感知/仿真记录。

已有感知与仿真结果供排查（无需再次调用 API）：
`/data/workspace/wangyibo/navigation_scenarios_mcap/simulation_runs/scene_02_20261016T030409_571423Z/fire_offset0`。
其中检测到了灭火器，五个停靠候选均规划失败，`result.json` 为
`No reachable stopping pose`。这是既有失败记录，不是本次重新跑出的成功结果；
地图/定位一致性问题仍需排查。近似闭环仿真不能生成新视角，也不能代表实机安全验证。
