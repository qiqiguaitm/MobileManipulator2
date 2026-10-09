# 2026-09-23 验证记录

部署位置：robot02 `/home/didi/junbin.zhang/MobileManipulator2`。

- `tracer_base`、`board_approach` 编译成功。原有灯光回调存在未初始化值编译警告，未修改该功能。
- Python 离线测试：24 项通过。
- C++ 单调时钟指令看门狗测试通过，验证首次指令前、0.4 秒边界、刷新和禁用情况。
- Python 编译、入口脚本语法及底盘代码 diff 空白检查通过。
- 保存的真实 RGB-D：检测到 ID 0–16 共 17 个标记；板中心相机深度 1.22770 米，平面拟合 RMSE 4.42 毫米。
- 实时 `--check`：定位和图像输入检查通过；当前画面不含目标板，识别失败后返回退出码 2，未启动底盘。
- 实时报告：`log/board_approach/20260923_094940_334753/result.json`，原图位于其 `stage_0/rgb_0.png`。
- 最终 CAN0–3 均 DOWN，无底盘驱动运行。

以上为当天首次静止验证，当时受用户离场限制，未运行 `--execute`。

## 用户到场授权后的实机验证

- 远距离使用人工确认的图像区域及对齐深度做分段辅助接近；该阶段不是全自动板子识别。
- 自动识别成功后，一键程序生成目标并执行近距离分段对正。
- 修复启动时等待 TF 链路、相机/定位时间配对、子进程 sudo 终端继承问题。
- 新增 `--segmented`：原地转向、经规划检查的短直行、最后原地朝向目标。
- 实测发现原地转向错误触发前进停车范围；改为前进检查前方范围、转向检查径向净空，保留全部传感器、超时和倒车保护。
- 分段对正经历保护停车后单独恢复最后转向，因此不是一次无中断的完整端到端运行。
- 离线测试扩展至 30 项，全部通过，包含控制序列、偏离路径、过冲、转向净空与前进保护。
- 最终报告：`log/board_approach/20260923_102800_587625/result.json`。
- 三帧测量验收：底盘参考点到板面 0.99163 米，横向偏差 -0.05172 米，朝向误差 -0.000778 弧度（约 -0.045 度）；ID 0–16 全部识别。满足 1 米目标的 10 厘米/10 厘米/5 度验收标准。
- 最终 CAN0–3 均 DOWN，机器人停止。

## 无预存目标位置的单命令版

- 入口：`bash scripts/go_to_board.sh`（实际运动）；追加 `--check` 只检查。
- 不保存或读取板子固定地图坐标。曾试作的坐标配置已删除，源码不依赖该配置。
- 每次由全画面编号识别或板型外观匹配获取目标，用实时对齐深度拟合板面；粗接近每段最多 0.4 米，之后重新观测，最终必须通过编号与几何验收。
- 39 项离线测试通过，包括中心编号遮挡、目标歧义、深度无效、粗检不得报告到位、路径分段和运动保护。
- 真实历史 RGB-D 回放：远处约 2.85 米相机深度的板子被自动找到（外观分数 0.717，平面残差约 7.9 毫米）；斜视近处板子被找到（分数 0.644，残差约 5.1 毫米）；无板子的办公区画面被拒绝。
- `go_to_board.sh --check` 实时验收成功，报告 `log/board_approach/20260923_114936_817395/result.json`：距离 0.9917 米，横向偏差 -0.0450 米，朝向误差 0.00633 弧度，判定已经到位。
- 本版未为了重复测试而移走机器人；无人工衔接的远距离完整自动执行仍待实机验收。此前分段行走成功不等同于新版整个流程已完成一次无中断实测。

## 实机复测前的整理

- 正式脚本统一位于项目 `scripts/`，移除包目录中重复的两个入口副本。
- 补充 `TESTING.md`：运行前提、一次命令验收流程、退出状态和每段日志位置。
- `--help` 不请求 sudo 或连接 ROS；互相冲突的执行/检查参数在入口拒绝。
- 正式运行代码不引用 `_cc_` 临时脚本、历史目标文件、手工 ROI 或固定板子地图坐标。
- 重跑 39 项 Python 测试和 C++ 指令看门狗测试均通过；Shell 语法检查通过。
- 包重新构建，本文档、复测说明和使用说明随 ROS 包安装。
- 本轮只做软件整理和离线检查，没有发送行走目标。

## 修复用户复测失败及粗接近分段过多

- 用户运行 `20260923_120318_897035`：前两段完成，第三段报 `scan message timestamp is stale`，不是目标识别失败。
- 运动进程取消 RGB/深度/相机内参订阅，传感器 QoS 队列只保留最新消息，减少不必要处理和积压。
- 新鲜度阈值保持原值：过期立即发零速度，2 秒内连续取得至少三个时间戳递增的新鲜扫描且稳定 0.2 秒才恢复；持续失联、定位异常、CAN 异常仍锁存停车。
- 粗接近单段上限改为 1.5 米，原 0.4 米限制取消；速度仍为最高 0.05 米/秒，必须通过原路径和障碍检查。失败记录的起点几何回归测试确认可基本一段到达粗观察位置。
- 运动结果增加阶段与暂停/恢复事件；顶层失败结果直接包含底层原因。
- 底盘驱动改为复用 executor，并在 ROS 关闭后退出循环，修复退出时反复构建 executor 引发的 guard condition 异常。
- 静止订阅监测确认运动节点无相机数据，扫描消息最大时延 0.474 秒、位姿最大时延 0.581 秒，监测期间无健康检查错误。
- 本轮没有启用 CAN 或发送运动命令；修改后的长段接近及短暂停车恢复仍待实机复测。

## 2026-09-23 generic object-centre mode

Replaced the proposed checkerboard fallback with a DINO-X object-box + aligned depth interface.
Target yaw and stopping point follow the ray to the visible object centre; no front-plane requirement.
56 tests passed on robot02, including generic geometry, invalid/mixed depth, ambiguous targets,
three-observation verification and disabled marker decoding. No physical motion was performed.
The configured detector at 192.168.112.14:10086 is currently unreachable. Automatic detection
and end-to-end object approach are NOT validated; the code must stop on service failure.

## GPT API integration

OpenAI Responses image input + strict JSON output added. Default detector is GPT; ArUco and DINO-X remain options.
60 tests passed; GPT protocol, response validation, failure handling and depth fusion tested using mocked HTTP only.
No real GPT request completed: no API key configured and robot02 direct HTTPS connection to api.openai.com timed out.
Missing-key --check must fail before camera startup/movement. No motion performed during integration.
# LiDAR geometry integration — 2026-09-24

## Feedback-expiry follow-up

User run `20260924_151559_513383` stopped on expired asynchronous feedback. Five successive image/plane estimates were valid; later capture reported an odometry speed/jump rejection, and the independent motion guard stopped when feedback exceeded six seconds. The old failure bundle did not retain the offending odometry samples, so its precise jump cannot be reconstructed.

Inspection found that `odom_frame_converter.py` overwrites FAST-LIO timestamps with callback receipt times and relabels frames. Deskew now subscribes to `/fastlio/odom` with original timestamps and applies the actual child-to-base transform. Chassis initialization is followed by a disabled-motion wait for feedback no older than three seconds before submitting a Nav2 goal. Six-second running expiry and all obstacle/CAN guards are unchanged. Per-frame timing and capture-error history with raw odometry samples were added. The shell entry now caps OpenBLAS/OpenMP/MKL/NumExpr threads at one, matching the validation environment; the previous default OpenBLAS pool had 12 threads.

Two longer read-only diagnostics still observed an occasional feedback expiry: one timed frame spent 5.17 seconds in plane processing plus evidence output despite a 1.60-second detector request. Replaying the same frame offline took 0.18 seconds without evidence and 0.28 seconds with evidence; the exact transient stall was not reproduced by profiling, and should not be claimed fully eliminated. Subsequent profiled 35-second run `async_readonly_20260924_170418` passed with 16 valid updates and maximum observed feedback age 4.80 seconds. Installed entry `20260924_170609_587038` then passed static observation, planning and three asynchronous updates with the new timestamp source and bounded numerical threads. CAN2 remained DOWN during these diagnostics. Complete physical docking still needs supervised verification.

The expanded 102-test suite covers retimed-message false speed jumps, nonidentity body-to-base transforms, original-topic routing and disabled-motion startup waits for fresh feedback, in addition to the earlier checks.

The default no-ArUco path now separates image-only target detection from metric geometry. It uses current RS16 scans, saved camera/LiDAR extrinsics and live RGB intrinsics/distortion. Raw point timestamps are recovered with the deployed publisher's `header + time - 0.1` convention and deskewed to the RGB timestamp using bracketed `/odom` SE(3) interpolation. No camera depth values are consulted in LiDAR geometry mode; invalid scans do not silently fall back to camera depth.

97 tests passed on robot02, including padded point decoding, sensor synchronization without a camera depth stream, translation/rotation deskew, stale history and odometry reset rejection, independent left/right plane checks, sparse/collinear/ground/mixed surface rejection, failure artifacts, original motion guards and optional ArUco behavior. The live pillar regression verifies that vertical crop changes do not invalidate horizontal docking, while horizontal displacement still fails at the existing threshold. Tests do not start motors.

Candidate live `--check` completed at `log/board_approach/20260924_150603_483824`: three independent automatic detections and LiDAR fits; roughly 425 candidates/409 inliers on the last frame, 7 scan rings, 5.25 mm plane residual; roughly 2.06 m coarse Nav2 plan. Three subsequent asynchronous observations also passed (408–425 candidates, 4.93–5.35 mm residual). The ROI was detected by DeepSeek, not manually specified. An earlier narrow ROI produced insufficient scan coverage, so the default description now asks for a larger fully visible mid/lower face region. Cropped detections are rejected and retried within the bounded observation budget.

This is stationary sensing/planning/feedback validation, not completed physical docking. No motion commands were issued. CAN2 remained DOWN. Short stationary residuals are not absolute metric accuracy. Camera/LiDAR extrinsics have only been checked by visual overlay; changes to mounting require recalibration. Near-range scan coverage, moving field of view and the complete coarse-to-fine maneuver still require supervised physical testing.

The existing coarse/precise phase count, speed limits, obstacle checks, watchdogs and emergency-stop handling are retained. `--check` now validates three asynchronous feedback updates as well as the static observations and plan. Failures retain raw LiDAR data and projected inlier/outlier images. No target map coordinates are stored for reuse.

Final installed one-command check passed at `log/board_approach/20260924_151259_591500`, through `scripts/go_to_board.sh --check --no-aruco --detector deepseek --geometry-source lidar --standoff 1.0`: three static observations, 1.69 cm horizontal centre spread, 0.35 degree heading span, 409/423 plane inliers, 7 rings, 5.08 mm plane residual, 2.071 m coarse path and three valid asynchronous updates. Build completed with `colcon build --packages-select board_approach --symlink-install`. An earlier installed check correctly exposed an irrelevant vertical-centre spread; LiDAR static and asynchronous consistency now both use map XY, preserving the 6 cm static / 25 cm asynchronous horizontal limits. Final check did not execute motion.

## 2026-09-24 reverse-command failure

Run `20260924_171355_458131` stopped with `Reverse/lateral motion is disabled`.
The old report did not retain vx/vy, so the exact rejected command is unknown.
Live TEB allowed 0.15 m/s backwards while the motion guard disallowed reversing.
The approach task now snapshots, temporarily applies, reads back, and restores
TEB speed limits and a stronger forward-drive penalty. TEB requires a positive
backwards limit (zero disables its limiter); use 0.001 m/s with penalty epsilon
0.0001. The existing independent guard clamps every negative output to zero and
still rejects reverse commands below -0.005 m/s or lateral commands above 0.001.
Forward speed is configured in the planner as well as limited in the guard.
Failure events now retain vx/vy/wz and command age.

Validation: 105 tests passed on robot02. With CAN2 down and no active Nav2 goal,
a parameter-only probe applied/read back all coarse/fine settings, then restored
and verified every original value. No CAN socket, goal, or velocity was sent.
Physical tracking remains unverified; forward-only infeasible paths may still
fail safely rather than reverse. Prior intermittent perception latency remains
an independent limitation.

## 2026-09-24 endpoint-gated visual feedback

User requested temporary loss of visual target during a planned leg to be allowed,
with failure only near that leg's endpoint. Coarse and precise Nav2 legs now
require fresh feedback within 0.40 m of the active endpoint (latched once entered),
and unconditionally before declaring the leg successful. Fresh means the existing
6-second capture-age limit, with the original target-identity validation.
Startup still requires fresh validated detection. Away from the endpoint, missing
or expired feedback keeps the last validated navigation goal; it cannot correct
that goal. Detection continues asynchronously; newly detected identity changes
still fail. Old inference results are retained as evidence but not published.
Once the near-goal gate is latched, goal corrections are frozen. Sensor freshness,
localization, obstacle, CAN, speed and time/distance guards are unchanged.

109 tests passed on robot02, including mid-leg expiry continuing at bounded speed,
near-endpoint expiry stopping, fresh reacquisition permitting progress, latched
near-goal requirements, identity mismatch, completion check and delayed inference.
No physical motion was commanded during validation. This change does not guarantee
reacquisition if the camera is still facing away near the endpoint.

## 2026-09-24 unified asynchronous geometry regression

Run 20260924_175643_049173 passed three stationary observations and path planning
but all six async frames failed `Object centre outside approach range` in the old
RGB-D centre-only helper. LiDAR async now uses the same surface geometry as static
observation. Shared limits.py centralizes 6 m LiDAR and 4 m RGB-D ranges. Failed or
old results never refresh feedback. Status reports retain underlying failure
counts; startup timeout includes the actual cause rather than only missing feedback.
Static and async artifacts now record the camera transform used for replay.
No point/plane quality or motor safety thresholds were relaxed.

118 tests passed, including full VisualMonitor.infer at 4.23, 5.5, 6 and 6.1 m and
comparison with stationary surface geometry. Offline replay of 19 saved frames
from 174832 and 175643 passed all geometry/feedback construction checks; detector
boxes were replayed, not re-inferred. Older data used the configured static
extrinsics (matching the previously captured base<-rslidar transform).
Official installed --check passed at 180424_553890: target about 4.23 m, planned
path 2.272 m, three valid asynchronous updates. This is stationary verification,
not an executed docking test. No robot movement was commanded.

## 2026-09-24 capture-time TF comparison

Snapshot consistency previously compared a historical localization pose to latest
map<-base_link TF. It now requests TF at the selected localization pose's exact
nanosecond stamp (pose remains subject to the existing image synchronization
window). No latest-transform fallback is allowed if history is unavailable.
Original 0.10 m / 0.12 rad mismatch limits remain; errors now include measured
residuals and both timestamps. Static extrinsic queries still use latest TF.
124 offline tests passed on robot02, including motion-induced latest-TF mismatch,
pose-vs-image timestamp distinction, missing history, genuine same-time mismatch,
and exact nanoseconds reaching the TF buffer. Deployed ros_io.py. No paid detector
API calls, robot movement or physical end-to-end retest performed for this fix.

## 2026-09-24 near-target LiDAR vertical field of view

Final verification in 183215_177909 rejected ten frames because inlier height
covered only ~24.6% of the full image ROI. Both navigation legs had succeeded.
Coverage now uses the intersection of the fitted face, full-scan beam elevation
envelope, image ROI and existing base z>0.15 m filter. Beam extrema are obtained
from all 16 raw RS16 rings before deskew, independently of target-selected returns.
If complete, finite beam calibration/transform data is unavailable, the original
full-ROI coverage requirement remains. Horizontal coverage, 35% vertical support,
3 rings, physical 2-D extent, both half-plane fits and plane quality checks remain.

128 tests passed, including a nearby tall face, insufficient ring calibration,
single-ring/degenerate support and a small fragment in a larger observable face.
Offline replay of all ten saved final frames passed geometry and unchanged
arrival criteria: normal distance 1.0185–1.0197 m, lateral offset 0.0160–0.0296 m,
heading error 3.00–3.77 degrees. Observable vertical coverage 91.9–95.2%.
Historical scans were stationary; beam elevations were recovered from the full
saved deskewed scan using the recorded base<-lidar transform, not target points.
These are sensor-derived offline measurements, not independent ground truth.
No paid API, navigation goal or motor command used. Deployed lidar_geometry.py
and lidar_snapshot.py; no physical rerun requested for this verification.
