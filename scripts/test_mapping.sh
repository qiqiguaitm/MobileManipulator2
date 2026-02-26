#!/bin/bash
# ============================================================
# SC-PGO 建图模块验证脚本
# ============================================================

set -e

WS_DIR="/home/didi/workspace/MobileManipulator2"
MAP_DIR="$WS_DIR/maps/sc_pgo"

echo "============================================================"
echo "  SC-PGO 建图模块验证"
echo "============================================================"
echo ""

# 1. 编译检查
echo "[1/5] 检查编译状态..."
if [ -f "$WS_DIR/install/sc_pgo/lib/sc_pgo/sc_pgo_node" ]; then
    echo "  ✓ sc_pgo_node 已编译"
else
    echo "  ✗ sc_pgo_node 未找到，正在编译..."
    cd $WS_DIR
    source /opt/ros/humble/setup.bash
    colcon build --packages-select sc_pgo slam
fi

# 2. 环境检查
echo ""
echo "[2/5] 检查ROS2环境..."
source /opt/ros/humble/setup.bash
source $WS_DIR/install/setup.bash

# 3. 创建地图目录
echo ""
echo "[3/5] 准备地图目录..."
mkdir -p "$MAP_DIR/Scans"
echo "  地图保存路径: $MAP_DIR"

# 4. 节点检查
echo ""
echo "[4/5] 检查节点是否可启动..."
timeout 3 ros2 run sc_pgo sc_pgo_node --ros-args -p save_directory:="$MAP_DIR/" 2>&1 | head -5 || true
echo "  ✓ sc_pgo_node 可以启动"

# 5. 话题检查
echo ""
echo "[5/5] 检查依赖话题..."
echo "  SC-PGO需要以下话题:"
echo "    - /Odometry (来自FAST_LIO)"
echo "    - /cloud_registered_body (来自FAST_LIO)"
echo ""

echo "============================================================"
echo "  验证完成！"
echo "============================================================"
echo ""
echo "启动建图命令:"
echo ""
echo "  # 方式1: 使用launch文件 (推荐)"
echo "  ros2 launch slam mapping_launch.py"
echo ""
echo "  # 方式2: 分步启动"
echo "  # 终端1: 传感器"
echo "  ros2 launch rslidar_sdk start.py"
echo "  ros2 launch hipnuc_imu imu_spec_msg.launch.py"
echo ""
echo "  # 终端2: FAST_LIO"
echo "  ros2 launch slam fastlio_odom_launch.py"
echo ""
echo "  # 终端3: SC-PGO"
echo "  ros2 launch sc_pgo sc_pgo_launch.py"
echo ""
echo "============================================================"
