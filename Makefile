# MobileManipulator2 Quick Commands (ROS2)
# Usage: make <target>

SHELL := /bin/bash
WS_DIR := /home/didi/workspace/MobileManipulator2
# 使用独立的 ROS_DOMAIN_ID 避免与局域网其他 ROS2 节点冲突
ROS_DOMAIN_ID := 42
# FastDDS 配置：增大点云等大消息的缓冲区，解决 "sequence size exceeds remaining buffer" 问题
FASTDDS_PROFILE := $(WS_DIR)/config/fastdds_profile.xml
# ROS_SETUP: 如果 FastDDS 配置存在则使用，否则跳过
ROS_SETUP := export ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) && \
	(test -f $(FASTDDS_PROFILE) && export FASTRTPS_DEFAULT_PROFILES_FILE=$(FASTDDS_PROFILE) || true) && \
	source /opt/ros/humble/setup.bash && \
	source $(WS_DIR)/install/setup.bash

.PHONY: help \
        navigation navi nav navi-fusion nav-fusion navi-approach \
        percept-nav \
        cleaner-manager cleaner-manager-no-rviz cleaner-manager-node \
        full-manual \
        build build-all build-drivers build-slam build-nav build-perception build-cleaner-manager \
        can-bringup can-bringup-auto can-reset can-status \
        cam-clean cam-top cam-hand cam-chassis cam-dual cam-status \
        percept-check percept-3d percept-multi percept-3d-rviz percept-multi-rviz percept-full percept-full-sam3 percept-full-sam3-stereo-hand percept-full-sam3-stereo-hand-nocdm percept-stop \
        status stop clean kill-ros health

help:
	@echo "=============================================="
	@echo "  MobileManipulator2 Quick Commands (ROS2)"
	@echo "=============================================="
	@echo ""
	@echo "相机命令:"
	@echo "  make cam-top        - 启动顶部相机 (D455)"
	@echo "  make cam-hand       - 启动手臂相机 (D435)"
	@echo "  make cam-chassis    - 启动底盘相机 (D435)"
	@echo "  make cam-dual       - 启动双相机 (Top + Chassis)"
	@echo "  make cam-clean      - 清理相机进程"
	@echo "  make cam-status     - 查看相机状态"
	@echo ""
	@echo "感知命令:"
	@echo "  make percept-3d       - 3D场景感知 (需先启动相机)"
	@echo "  make percept-multi    - 双相机融合感知"
	@echo "  make percept-3d-rviz  - 3D感知 + RViz"
	@echo "  make percept-multi-rviz - 双相机感知 + RViz"
	@echo "  make percept-full     - 一键启动 (相机+感知+RViz, DINOX)"
	@echo "  make percept-full-sam3- 一键启动 (相机+感知+RViz, SAM3)"
	@echo "  make percept-check    - 检查感知依赖"
	@echo "  make percept-stop     - 停止感知节点"
	@echo ""
	@echo "导航命令:"
	@echo "  make navi           - 启动导航 (HDL定位+Nav2, 不含底盘)"
	@echo "  make navi-fusion    - 启动导航 (含底盘+轮速融合)"
	@echo "  make navi-approach  - 启动导航+底盘相机 (接近导航用)"
	@echo "  make navi-no-fusion - 启动导航 (无轮速融合)"
	@echo "  make navi-no-rviz   - 启动导航 (无RViz)"
	@echo ""
	@echo "感知导航命令:"
	@echo "  make percept-nav      - 一键启动 (导航+相机+感知+接近导航)"
	@echo ""
	@echo "全系统命令 (Cleaner Manager):"
	@echo "  make cleaner-manager         - 全系统启动 (导航+相机+机械臂+感知+管理+RViz)"
	@echo "  make cleaner-manager-no-rviz - 全系统启动 (无RViz)"
	@echo "  make cleaner-manager-node    - 仅启动管理节点 (需其他模块已运行)"
	@echo "  make full-manual            - 感知导航抓取全流程 (仿ROS1 start_amr.sh full)"
	@echo "  make build-cleaner-manager  - 构建 cleaner_manager 包"
	@echo ""
	@echo "构建命令:"
	@echo "  make build          - 构建所有包"
	@echo "  make build-drivers  - 构建硬件驱动 (tracer, rslidar, imu)"
	@echo "  make build-slam     - 构建 SLAM 相关"
	@echo "  make build-nav      - 构建导航相关"
	@echo "  make build-perception - 构建感知相关"
	@echo "  make build-perception-clean - 清理感知构建"
	@echo "  make build-perception-verify - 验证感知构建"
	@echo "  make build PKG=xxx  - 构建指定包"
	@echo ""
	@echo "CAN 设备命令:"
	@echo "  make can-bringup      - CAN 初始化 (手动模式)"
	@echo "  make can-bringup-auto - CAN 初始化 (自动检测)"
	@echo "  make can-reset        - 重置 CAN 接口"
	@echo "  make can-status       - 查看 CAN 状态"
	@echo ""
	@echo "维护命令:"
	@echo "  make status         - 查看节点状态"
	@echo "  make stop           - 停止所有节点"
	@echo "  make clean          - 清理构建文件"
	@echo "  make health         - 硬件健康检测"
	@echo "=============================================="

# ============================================
# 导航命令
# ============================================

# 导航: HDL定位 + Nav2 (不含底盘)
navigation navi nav:
	@echo "[NAV] 启动导航 (HDL+Nav2, 不含底盘)..."
	@$(ROS_SETUP) && ros2 launch slam hdl_navigation_launch.py \
		use_odom_fusion:=false \
		launch_chassis:=false \
		enable_rviz:=true

# 带轮速融合 (默认使用HDL校正后的scan，更稳定)
navi-fusion nav-fusion:
	@echo "[NAV] 启动导航 (轮速融合 + HDL校正scan)..."
	@$(ROS_SETUP) && ros2 launch slam hdl_navigation_launch.py \
		use_odom_fusion:=true \
		launch_chassis:=true \
		use_raw_laserscan:=false \
		enable_rviz:=true

# 无轮速融合
navi-no-fusion nav-no-fusion:
	@echo "[NAV] 启动导航 (无轮速融合)..."
	@$(ROS_SETUP) && ros2 launch slam hdl_navigation_launch.py \
		use_odom_fusion:=false \
		launch_chassis:=false \
		enable_rviz:=true

# 无RViz
navi-no-rviz nav-no-rviz:
	@echo "[NAV] 启动导航 (无RViz)..."
	@$(ROS_SETUP) && ros2 launch slam hdl_navigation_launch.py \
		use_odom_fusion:=true \
		launch_chassis:=true \
		enable_rviz:=false

# 导航 + 底盘相机 (接近导航用)
navi-approach:
	@echo "[NAV] 启动导航 + 底盘相机..."
	@echo "[1/2] 后台启动底盘相机..."
	@$(ROS_SETUP) && ros2 launch camera_driver camera_driver.launch.py \
		top_enable:=false hand_enable:=false chassis_enable:=true &
	@sleep 3
	@echo "[2/2] 启动导航..."
	@$(ROS_SETUP) && ros2 launch slam hdl_navigation_launch.py \
		use_odom_fusion:=true \
		launch_chassis:=true \
		enable_rviz:=true

# ============================================
# 感知导航命令 (Perception + Navigation)
# ============================================

# 一键启动感知导航 (导航 + 相机 + 感知 + 接近导航)
percept-nav:
	@echo "[PERCEPT-NAV] 一键启动感知导航..."
	@bash $(SCRIPTS_DIR)/start_percept_nav.sh

# ============================================
# 全系统命令 (Cleaner Manager)
# ============================================

# 全系统启动: 导航 + 相机 + 机械臂 + 感知 + Cleaner Manager + RViz
cleaner-manager:
	@echo "[CLEANER-MANAGER] 全系统启动..."
	@$(ROS_SETUP) && ros2 launch cleaner_manager cleaner_manager_full.launch.py \
		rviz:=true \
		use_odom_fusion:=true \
		launch_chassis:=true \
		detector_type:=sam3 \
		extrinsics_suffix:=_stereo_hand \
		enable_depth_optimizer:=true

# 全系统启动 (无RViz)
cleaner-manager-no-rviz:
	@echo "[CLEANER-MANAGER] 全系统启动 (无RViz)..."
	@$(ROS_SETUP) && ros2 launch cleaner_manager cleaner_manager_full.launch.py \
		rviz:=false \
		use_odom_fusion:=true \
		launch_chassis:=true \
		detector_type:=sam3 \
		extrinsics_suffix:=_stereo_hand \
		enable_depth_optimizer:=true

# 仅启动 cleaner_manager_node (需其他模块已运行)
cleaner-manager-node:
	@echo "[CLEANER-MANAGER] 启动管理节点..."
	@$(ROS_SETUP) && ros2 launch cleaner_manager cleaner_manager.launch.py

# 构建 cleaner_manager 包
build-cleaner-manager:
	@echo "[BUILD] 构建 cleaner_manager..."
	@cd $(WS_DIR) && $(ROS_SETUP) && colcon build --packages-select cleaner_manager
	@echo "[OK] cleaner_manager 构建完成"

# 全手动模式: 感知 + 导航 + 抓取全流程 (仿 ROS1 start_amr.sh full)
full-manual:
	@echo "[FULL-MANUAL] 启动全流程 (导航+相机+感知+机械臂+抓取)..."
	@$(ROS_SETUP) && ros2 launch cleaner_manager cleaner_manager_full.launch.py \
		rviz:=true \
		use_odom_fusion:=true \
		launch_chassis:=true \
		detector_type:=sam3 \
		extrinsics_suffix:=_stereo_hand \
		enable_depth_optimizer:=true

# ============================================
# 构建命令
# ============================================

# 构建所有包
build:
ifdef PKG
	@echo "[BUILD] 构建包: $(PKG)..."
	@cd $(WS_DIR) && $(ROS_SETUP) && colcon build --packages-select $(PKG)
else
	@echo "[BUILD] 构建所有包..."
	@cd $(WS_DIR) && $(ROS_SETUP) && colcon build
endif
	@echo "[OK] 构建完成"

# 完整构建
build-all: build-drivers build-slam build-nav
	@echo ""
	@echo "=============================================="
	@echo "[OK] 完整构建完成!"
	@echo "=============================================="

# 构建硬件驱动
build-drivers:
	@echo "[BUILD] 硬件驱动包..."
	@cd $(WS_DIR) && $(ROS_SETUP) && colcon build --packages-select \
		ugv_sdk tracer_base rslidar_sdk hipnuc_imu
	@echo "[OK] 驱动构建完成"

# 构建 SLAM 相关包 (ndt + hdl_loc + fast_lio + slam)
build-slam:
	@bash $(WS_DIR)/scripts/build_navigation.sh slam

# 构建导航相关包 (ndt + hdl_loc + slam)
build-nav:
	@bash $(WS_DIR)/scripts/build_navigation.sh nav

# 构建建图模块 (fast_lio + sc_pgo)
build-mapping:
	@bash $(WS_DIR)/scripts/build_navigation.sh mapping

# 构建机器人描述包
build-desc:
	@echo "[BUILD] 机器人描述包..."
	@cd $(WS_DIR) && $(ROS_SETUP) && colcon build --packages-select \
		mobile_manipulator2_description tracer2_description
	@echo "[OK] 描述包构建完成"

# 清理构建文件
build-clean:
	@echo "[BUILD] 清理构建文件..."
	@cd $(WS_DIR) && rm -rf build install log
	@echo "[OK] 清理完成"

# ============================================
# CAN 设备命令
# ============================================

CAN_TOOLS := $(WS_DIR)/scripts/can_tools

# CAN 初始化 (手动模式)
can-bringup:
	@echo "[CAN] 初始化 (手动模式)..."
	@bash $(CAN_TOOLS)/bringup.sh manual

# CAN 初始化 (自动检测)
can-bringup-auto:
	@echo "[CAN] 初始化 (自动检测)..."
	@bash $(CAN_TOOLS)/bringup.sh auto

# 重置 CAN 接口
can-reset:
	@echo "[CAN] 重置接口..."
	@bash $(CAN_TOOLS)/reset_can.sh

# 查看 CAN 状态
can-status:
	@echo "=============================================="
	@echo "  CAN 接口状态"
	@echo "=============================================="
	@ip -br link show type can 2>/dev/null || echo "  无 CAN 接口"
	@echo ""
	@echo "  can0 (1000kbps) -> Piper 机械臂"
	@echo "  can1 (500kbps)  -> Tracer 底盘"
	@echo "=============================================="

# ============================================
# 维护命令
# ============================================

# 查看节点状态
status:
	@echo "=============================================="
	@echo "  ROS2 节点状态"
	@echo "=============================================="
	@$(ROS_SETUP) && ros2 node list 2>/dev/null || echo "  无节点运行"
	@echo ""
	@echo "=== TF 帧 ==="
	@$(ROS_SETUP) && ros2 run tf2_ros tf2_echo map base_link --timeout 1 2>&1 | head -5 || echo "  无 TF"
	@echo "=============================================="

# 停止所有节点（导航+感知+相机）
stop:
	@echo "[STOP] 停止所有节点..."
	@bash $(WS_DIR)/scripts/stop_nodes.sh || true
	@echo "[STOP] 停止感知节点..."
	@pkill -15 -f "[s]cene_perception_3d_node" 2>/dev/null || true
	@pkill -15 -f "[m]ulti_camera_perception_node" 2>/dev/null || true
	@pkill -15 -f "[m]ulti_camera_rviz_node" 2>/dev/null || true
	@pkill -15 -f "[p]erception_viz_node" 2>/dev/null || true
	@killall -15 rviz2 2>/dev/null || true
	@echo "[STOP] 停止相机节点..."
	@pkill -15 -f "ros2 launch camera_driver" 2>/dev/null || true
	@pkill -15 -f "realsense2_camera_node" 2>/dev/null || true
	@pkill -15 -f "camera_tf_publisher" 2>/dev/null || true
	@sleep 1
	@pkill -9 -f "ros2 launch camera_driver" 2>/dev/null || true
	@pkill -9 -f "realsense2_camera_node" 2>/dev/null || true
	@pkill -9 -f "camera_tf_publisher" 2>/dev/null || true
	@echo "[OK] 停止完成"

# 强制清理（调用 stop + 额外强制清理）
clean: stop
	@echo "[CLEAN] 强制清理残留进程..."
	@pkill -9 -f "[s]cene_perception_3d_node" 2>/dev/null || true
	@pkill -9 -f "[m]ulti_camera_perception_node" 2>/dev/null || true
	@killall -9 rviz2 2>/dev/null || true
	@killall -9 realsense2_camera_node 2>/dev/null || true
	@killall -9 camera_tf_publisher 2>/dev/null || true
	@echo "[OK] 清理完成"

# 杀死 ROS daemon
kill-ros:
	@echo "[KILL] 停止 ROS2 daemon..."
	@ros2 daemon stop 2>/dev/null || true
	@$(MAKE) stop
	@echo "[OK] 完成"

# ============================================
# 硬件健康检测
# ============================================

# 硬件健康检测
health:
	@echo "=============================================="
	@echo "  硬件健康检测"
	@echo "=============================================="
	@echo ""
	@echo "=== CAN 接口 ==="
	@ip -br link show type can 2>/dev/null || echo "  无 CAN 接口"
	@echo ""
	@echo "=== USB 设备 ==="
	@lsusb | grep -iE "lidar|realsense|camera|can|serial" || echo "  无特殊 USB 设备"
	@echo ""
	@echo "=== 串口设备 ==="
	@ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  无串口设备"
	@echo ""
	@echo "=== 相机设备 ==="
	@ls -la /dev/video* 2>/dev/null || echo "  无相机设备"
	@echo ""
	@echo "=============================================="

# ============================================
# 相机命令 (Camera)
# ============================================

# 清理相机进程
cam-clean:
	@echo "[CAM] 清理相机进程..."
	@killall -15 realsense2_camera_node 2>/dev/null || true
	@killall -15 camera_tf_publisher 2>/dev/null || true
	@sleep 1
	@killall -9 realsense2_camera_node 2>/dev/null || true
	@killall -9 camera_tf_publisher 2>/dev/null || true
	@sleep 1
	@echo "[OK] 相机已清理"

# 启动顶部相机 (D455)
cam-top: cam-clean
	@echo "[CAM] 启动顶部相机 (D455)..."
	@$(ROS_SETUP) && ros2 launch camera_driver camera_driver.launch.py \
		top_enable:=true hand_enable:=false chassis_enable:=false

# 启动手臂相机 (D435)
cam-hand: cam-clean
	@echo "[CAM] 启动手臂相机 (D435)..."
	@$(ROS_SETUP) && ros2 launch camera_driver camera_driver.launch.py \
		top_enable:=false hand_enable:=true chassis_enable:=false

# 启动底盘相机 (D435)
cam-chassis: cam-clean
	@echo "[CAM] 启动底盘相机 (D435)..."
	@$(ROS_SETUP) && ros2 launch camera_driver camera_driver.launch.py \
		top_enable:=false hand_enable:=false chassis_enable:=true

# 启动双相机 (Top + Chassis)
cam-dual: cam-clean
	@echo "[CAM] 启动双相机 (Top + Chassis)..."
	@$(ROS_SETUP) && ros2 launch camera_driver camera_driver.launch.py \
		top_enable:=true hand_enable:=false chassis_enable:=true

# 查看相机状态
cam-status:
	@echo "=============================================="
	@echo "  相机状态"
	@echo "=============================================="
	@echo ""
	@echo "=== USB 设备 ==="
	@lsusb | grep -i "intel" || echo "  无 RealSense 设备"
	@echo ""
	@echo "=== 相机话题 ==="
	@$(ROS_SETUP) && ros2 topic list 2>/dev/null | grep -E "^/camera" | head -20 || echo "  无相机话题"
	@echo "=============================================="

# ============================================
# 感知命令 (Perception)
# ============================================

SCRIPTS_DIR := $(WS_DIR)/scripts

# 检查感知系统依赖（调用脚本）
percept-check:
	@bash $(SCRIPTS_DIR)/build_perception.sh deps

# 3D 场景感知（调用脚本）
percept-3d:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=top --skip-camera

# 双相机融合感知（调用脚本）
percept-multi:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --skip-camera

# 3D 感知 + RViz（调用脚本）
percept-3d-rviz:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=top --skip-camera --rviz

# 双相机感知 + RViz（调用脚本）
percept-multi-rviz:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --skip-camera --rviz

# 一键启动 (相机 + 感知 + RViz) - DINOX 检测器
percept-full:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz

# 一键启动 - 使用 SAM3 检测器, CDM已禁用
percept-full-sam3:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --detector=sam3

# 一键启动 - 使用新标定外参（测试用）
percept-full-new:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --new-extrinsics

# 测试新标定外参 (2系数固定内参, 2026-02-28), CDM已启用
percept-full-sam3-new:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --detector=sam3 --extrinsics-suffix=_new

percept-full-sam3-new-nocdm:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --detector=sam3 --extrinsics-suffix=_new --no-cdm

# 立体手链外参 (chassis→hand→flange→base, 2026-03-02), CDM已启用
percept-full-sam3-stereo-hand:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --detector=sam3 --extrinsics-suffix=_stereo_hand

# 同上（保留别名，兼容旧脚本）
percept-full-sam3-stereo-hand-nocdm:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --detector=sam3 --extrinsics-suffix=_stereo_hand --no-cdm

# 临时测试 - 使用最早的旧外参 (_back), CDM已启用
percept-full-sam3-back:
	@bash $(SCRIPTS_DIR)/start_perception_3d.sh --camera=dual --rviz --detector=sam3 --extrinsics-suffix=_back

# 停止感知节点（Python节点用pkill，[x]技巧避免自杀）
percept-stop:
	@echo "[PERCEPT] 停止感知节点..."
	@pkill -15 -f "[s]cene_perception_3d_node" 2>/dev/null || true
	@pkill -15 -f "[m]ulti_camera_perception_node" 2>/dev/null || true
	@pkill -15 -f "[p]erception_viz_node" 2>/dev/null || true
	@killall -15 rviz2 2>/dev/null || true
	@sleep 1
	@pkill -9 -f "[s]cene_perception_3d_node" 2>/dev/null || true
	@pkill -9 -f "[m]ulti_camera_perception_node" 2>/dev/null || true
	@killall -9 rviz2 2>/dev/null || true
	@echo "[OK] 感知已停止"

# ============================================
# 构建感知包
# ============================================

# 构建感知相关包（调用脚本）
build-perception:
	@bash $(SCRIPTS_DIR)/build_perception.sh build

# 清理感知构建
build-perception-clean:
	@bash $(SCRIPTS_DIR)/build_perception.sh clean

# 验证感知构建
build-perception-verify:
	@bash $(SCRIPTS_DIR)/build_perception.sh verify
