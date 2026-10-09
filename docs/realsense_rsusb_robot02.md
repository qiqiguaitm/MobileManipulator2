# robot02 D455 RSUSB 启动说明

2026-09-22 修复。设备：D455 318122302992，固件 5.17.0.10。
ASIC/固件更新编号 327743062831 是同一台相机的内部编号。

## 原因和修复

系统 librealsense 2.56.5 的 V4L2 后端在这台 Jetson 5.15.148-tegra 上，
初始化时报告 No HID info provided，随后 bad optional access。
独立构建官方 librealsense v2.56.4，启用 FORCE_RSUSB_BACKEND，
安装到 ~/.local/opt/realsense-rsusb-2.56.4/install。
未替换系统库、未修改固件、未修改内核。
切换后取流曾出现 RS2_USB_STATUS_PIPE，相机 hardware_reset 后恢复。

## 启动

先确认没有其他程序占用相机，然后在本项目根目录运行：

```bash
ROS_DOMAIN_ID=144 ROS_LOCALHOST_ONLY=1 bash scripts/run_top_camera_rsusb.sh
```

此脚本使用现有 camera_driver.launch.py，只启动 top D455，
默认 RGB 和深度为 1280x720@15，启用对齐深度。
默认不启动自定义 camera TF 发布器，避免与现有机器人 TF 重复；
需要该发布器时显式追加 enable_camera_tf:=true，并先检查现有 TF。

ROS_LOCALHOST_ONLY 必须与定位和订阅端一致。
脚本保留调用方显式设置的 ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY；
未设置时默认 144 / 1。
REALSENSE_RSUSB_PREFIX 可覆盖 SDK 安装位置。

## 数据接口

- /camera/top/color/image_raw
- /camera/top/depth/image_rect_raw
- /camera/top/aligned_depth_to_color/image_raw
- /camera/top/color/camera_info

通过 ROS 订阅图像，无须使用 V4L2 单独占用设备。
系统 /usr/bin/rs-enumerate-devices 和原有 pip pyrealsense2 不受此启动脚本影响，
不要把系统默认 SDK 的结果当成该 ROS 进程使用的 SDK。

独立 SDK 枚举命令：

```bash
sdk="$HOME/.local/opt/realsense-rsusb-2.56.4/install"
LD_LIBRARY_PATH="$sdk/lib" "$sdk/bin/rs-enumerate-devices" -s
```

此次启动日志 /tmp/junbin-top-camera.log。
脚本仅启动相机，不启动底盘、不发送运动指令，也未配置开机自启动。

## 实测验证

15 秒采集：深度 214 帧、RGB 211 帧、对齐深度 210 帧，约 14 Hz；
RGB 和对齐深度时间戳一致，分辨率 1280x720，深度编码 16UC1。
对齐深度非零像素比例 86.2%，同期定位内点比例 99.94%。
启动初期存在短暂 USB 控制请求警告，此后采集正常。
详细验证记录：/tmp/junbin-rgbd-verification.json。
