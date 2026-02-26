#!/bin/bash
# ROS2 节点清理脚本
# 功能：优雅地停止所有导航相关节点

set +e  # 忽略错误继续执行
trap '' TERM INT  # 忽略终止信号，避免自己被杀死

echo "[NAV] 停止所有节点..."

# Source ROS2 环境
WS_DIR="/home/didi/workspace/MobileManipulator2"
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$WS_DIR/install/setup.bash" 2>/dev/null || true

# 显示当前进程状态（调试用）
echo "[DEBUG] 当前 ROS2 相关进程:"
ps aux | grep -E "(ros2|fastlio|hdl|tracer|rslidar)" | grep -v grep | head -10 || echo "  无进程"

# ============================================
# 1. 停止 lifecycle 管理的节点
# ============================================
echo "[1/5] 停止 lifecycle 节点..."
LIFECYCLE_NODES="map_server planner_server controller_server bt_navigator"
for node in $LIFECYCLE_NODES; do
    timeout 2 ros2 lifecycle set /$node shutdown 2>/dev/null || true
done
sleep 0.5

# ============================================
# 2. 获取所有运行的节点并逐个停止
# ============================================
echo "[2/5] 停止 ROS2 节点..."
NODES=$(timeout 3 ros2 node list 2>/dev/null | grep -v "^$" || true)
if [ -n "$NODES" ]; then
    echo "  发现节点:"
    echo "$NODES" | sed 's/^/    /'

    # ROS2 没有 rosnode kill，使用 pkill 按节点名停止
    while IFS= read -r node; do
        # 去掉开头的 /
        node_name="${node#/}"
        if [ -n "$node_name" ]; then
            # 尝试 pkill 匹配节点名
            pkill -15 -f "$node_name" 2>/dev/null || true
        fi
    done <<< "$NODES"
fi
sleep 1

# ============================================
# 3. 停止 ros2 launch 进程
# ============================================
echo "[3/5] 停止 ros2 launch..."
pkill -15 -f "ros2 launch" 2>/dev/null || true
sleep 2
pkill -9 -f "ros2 launch" 2>/dev/null || true

# ============================================
# 4. 清理残留进程
# ============================================
echo "[4/5] 清理残留进程..."

# 按可执行文件名清理
EXECUTABLES=(
    "rviz2"
    "fastlio_mapping"
    "hdl_localization_node"
    "hdl_global_localization_node"
    "planner_server"
    "controller_server"
    "bt_navigator"
    "behavior_server"
    "map_server"
    "lifecycle_manager"
    "rslidar_sdk_node"
    "tracer_base_node"
    "robot_state_publisher"
    "static_transform_publisher"
    "pointcloud_to_laserscan_node"
    "realsense2_camera_node"
    "realsense_node"
)

# Python 脚本清理
PYTHON_SCRIPTS=(
    "tf_republisher.py"
    "odom_fusion_sync.py"
    "odom_frame_converter.py"
    "fastlio_odom_to_tf.py"
    "delayed_cloud_relay.py"
    "globalmap_publisher.py"
    "joint_state_static.py"
    "odom_relay.py"
)

for script in "${PYTHON_SCRIPTS[@]}"; do
    pkill -15 -f "$script" 2>/dev/null || true
done
sleep 0.5
for script in "${PYTHON_SCRIPTS[@]}"; do
    pkill -9 -f "$script" 2>/dev/null || true
done

# IMU 驱动清理 (talker 和 listener)
pkill -9 -f "hipnuc_imu/talker" 2>/dev/null || true
pkill -9 -f "hipnuc_imu/listener" 2>/dev/null || true
pkill -9 -f "hipnuc_imu" 2>/dev/null || true

# 按节点名清理残留（覆盖 ROS2 节点名与进程名不匹配的情况）
# tracer_base 的节点名是 /tracer，但进程名是 tracer_base_node
pkill -9 -f "tracer_base" 2>/dev/null || true
# rslidar_sdk 的内部节点
pkill -9 -f "rslidar_sdk" 2>/dev/null || true

for exe in "${EXECUTABLES[@]}"; do
    pkill -15 -f "$exe" 2>/dev/null || true
done
sleep 1

# 强制清理
for exe in "${EXECUTABLES[@]}"; do
    pkill -9 -f "$exe" 2>/dev/null || true
done

# ============================================
# 4.5 额外清理：按命令行模式匹配
# ============================================
echo "[4.5/5] 额外清理..."

# 清理所有包含 slam 的 python 进程（但排除当前脚本）
pkill -9 -f "python3.*slam" 2>/dev/null || true

# 清理 ros2 run 命令
pkill -9 -f "ros2 run" 2>/dev/null || true

# 清理 component_container
pkill -9 -f "component_container" 2>/dev/null || true

# 清理 nav2 相关
pkill -9 -f "nav2" 2>/dev/null || true
pkill -9 -f "amcl" 2>/dev/null || true
pkill -9 -f "waypoint_follower" 2>/dev/null || true
pkill -9 -f "velocity_smoother" 2>/dev/null || true
pkill -9 -f "collision_monitor" 2>/dev/null || true

# 清理 fast_lio 相关（不同命名形式）
pkill -9 -f "fast_lio" 2>/dev/null || true
pkill -9 -f "FAST_LIO" 2>/dev/null || true

# 清理 hdl 相关
pkill -9 -f "hdl_" 2>/dev/null || true

# ============================================
# 5. 验证清理结果
# ============================================
echo "[5/5] 验证清理结果..."
sleep 1

# 再次显示残留进程
echo "[DEBUG] 残留 ROS2 相关进程:"
ps aux | grep -E "(ros2|fastlio|hdl|tracer|rslidar|slam)" | grep -v grep | grep -v "stop_nodes" | head -10 || echo "  无进程"

REMAINING=$(timeout 3 ros2 node list 2>/dev/null | grep -v "^$" | wc -l || echo "0")
if [ "$REMAINING" -eq 0 ] || [ "$REMAINING" = "0" ]; then
    echo "  ✓ 所有节点已停止"
else
    echo "  ⚠ 残留 $REMAINING 个节点:"
    timeout 3 ros2 node list 2>/dev/null | sed 's/^/    /'
fi

echo ""
echo "============================================"
echo "[NAV] 清理完成"
echo "============================================"

exit 0
