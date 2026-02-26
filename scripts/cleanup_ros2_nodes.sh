#!/bin/bash
# ============================================================
# ROS2 残余节点清理脚本
# 用于启动新 launch 前清理可能冲突的节点
#
# Usage:
#   ./scripts/cleanup_ros2_nodes.sh          # 交互式，先列出再确认
#   ./scripts/cleanup_ros2_nodes.sh -y        # 直接执行，不确认
#   ./scripts/cleanup_ros2_nodes.sh -l        # 仅列出当前节点，不清理
#   ./scripts/cleanup_ros2_nodes.sh -y -a    # 激进模式：清理所有 ros2 相关进程
# ============================================================

set -e

AUTO_YES=false
LIST_ONLY=false
AGGRESSIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes) AUTO_YES=true; shift ;;
        -l|--list) LIST_ONLY=true; shift ;;
        -a|--all) AGGRESSIVE=true; shift ;;
        -h|--help)
            head -18 "$0" | tail -14
            exit 0
            ;;
        *) shift ;;
    esac
done

# 常见 ROS2 节点进程名（按包/可执行文件名）
NODES=(
    "hdl_localization_node"
    "hdl_global_localization"
    "fastlio_mapping"
    "fastlio_odometry"
    "tracer_base_node"
    "rviz2"
    "robot_state_publisher"
    "joint_state_static"
    "rslidar_sdk_node"
    "delayed_cloud_relay"
    "hdl_odom_to_tf"
    "tf_republisher"
    "pointcloud_to_laserscan"
    "map_server"
    "controller_server"
    "planner_server"
    "bt_navigator"
    "lifecycle_manager"
    "amcl"
    "sc_pgo_node"
    "odom_fusion_sync"
    "odom_frame_converter"
    "odom_relay"
    "globalmap_loader"
    "load_global_map"
)

list_nodes() {
    echo "=============================================="
    echo "  当前运行的 ROS2 相关进程"
    echo "=============================================="
    if command -v ros2 &>/dev/null; then
        echo ""
        echo "[ros2 node list]"
        (source /opt/ros/humble/setup.bash 2>/dev/null; ros2 node list 2>/dev/null) || echo "  (无 / ros2 不可用)"
    fi
    echo ""
    echo "[按进程名匹配的 PID]"
    for name in "${NODES[@]}"; do
        pids=$(pgrep -f "$name" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            for pid in $pids; do
                cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 80)
                echo "  $pid  $name  $cmd"
            done
        fi
    done
    echo ""
}

kill_nodes() {
    local killed=0
    for name in "${NODES[@]}"; do
        pids=$(pgrep -f "$name" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            for pid in $pids; do
                echo "  kill $pid ($name)"
                kill "$pid" 2>/dev/null || true
                ((killed++)) || true
            done
        fi
    done
    return $killed
}

kill_aggressive() {
    echo "  激进模式: 终止所有 ros2 启动的 Python 节点..."
    pkill -f "ros2 launch" 2>/dev/null || true
    pkill -f "ros2 run" 2>/dev/null || true
    for name in "${NODES[@]}"; do
        pkill -f "$name" 2>/dev/null || true
    done
    # 额外：可能由 launch 派生的进程
    pkill -f "joint_state_static.py" 2>/dev/null || true
    pkill -f "hdl_odom_to_tf" 2>/dev/null || true
    pkill -f "delayed_cloud_relay" 2>/dev/null || true
    echo "  完成"
}

main() {
    if [[ "$LIST_ONLY" == true ]]; then
        list_nodes
        exit 0
    fi

    list_nodes

    if [[ "$AGGRESSIVE" == true ]]; then
        if [[ "$AUTO_YES" != true ]]; then
            echo "即将执行激进清理（包括 ros2 launch/run），确认? [y/N]"
            read -r ans
            [[ "$ans" =~ ^[yY]$ ]] || exit 0
        fi
        kill_aggressive
    else
        if [[ "$AUTO_YES" != true ]]; then
            echo "即将清理上述匹配的节点进程，确认? [y/N]"
            read -r ans
            [[ "$ans" =~ ^[yY]$ ]] || exit 0
        fi
        kill_nodes || true
    fi

    echo ""
    echo "停止 ros2 daemon..."
    (source /opt/ros/humble/setup.bash 2>/dev/null; ros2 daemon stop 2>/dev/null) || true
    echo "清理完成"
}

main
