#!/bin/bash
# ============================================================================
# ROS2 共享环境配置 - 单一真相源
# 所有感知相关脚本 source 此文件，确保环境一致
# ============================================================================

# 路径配置
export WS_DIR="/home/didi/workspace/MobileManipulator2"
export ROS_DOMAIN_ID=42
export SCRIPTS_DIR="$WS_DIR/scripts"

# 颜色定义
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export CYAN='\033[0;36m'
export RED='\033[0;31m'
export NC='\033[0m'

# 外部服务地址
export DINOX_URL="http://192.168.112.14:10086"
export SAM3_URL="http://192.168.112.14:8080"
export CDM_URL="http://192.168.112.14:8086"
export TRACKER_URL="http://192.168.112.14:11086"

# ============================================================================
# 环境设置函数
# ============================================================================
setup_ros2_env() {
    # 使用系统 Python（避免 miniconda 冲突）
    export PATH="/usr/bin:$PATH"

    # 清理 miniconda 路径（避免库冲突导致段错误）
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')

    # 禁用代理（内网服务访问）
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export no_proxy="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
    export NO_PROXY="$no_proxy"

    # ROS2 环境
    source /opt/ros/humble/setup.bash
    if [ -f "$WS_DIR/install/setup.bash" ]; then
        source "$WS_DIR/install/setup.bash"
    fi

    # FastDDS 配置（如果存在）
    if [ -f "$WS_DIR/config/fastdds_profile.xml" ]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="$WS_DIR/config/fastdds_profile.xml"
    fi
}

# ============================================================================
# 进程清理函数
# ============================================================================
cleanup_camera() {
    killall -15 realsense2_camera_node 2>/dev/null || true
    killall -15 camera_tf_publisher 2>/dev/null || true
    sleep 1
    killall -9 realsense2_camera_node 2>/dev/null || true
    killall -9 camera_tf_publisher 2>/dev/null || true
    sleep 1
}

cleanup_perception() {
    # 使用 [x] 技巧避免 pkill 匹配自身
    pkill -15 -f "[s]cene_perception_3d_node" 2>/dev/null || true
    pkill -15 -f "[o]bject_tracker_node" 2>/dev/null || true
    pkill -15 -f "[p]erception_viz_node" 2>/dev/null || true
    pkill -15 -f "[m]ulti_camera_perception_node" 2>/dev/null || true
    sleep 1
    pkill -9 -f "[s]cene_perception_3d_node" 2>/dev/null || true
    pkill -9 -f "[o]bject_tracker_node" 2>/dev/null || true
    pkill -9 -f "[p]erception_viz_node" 2>/dev/null || true
    pkill -9 -f "[m]ulti_camera_perception_node" 2>/dev/null || true
}

cleanup_rviz() {
    pkill -15 -f "rviz2" 2>/dev/null || true
    sleep 1
    pkill -9 -f "rviz2" 2>/dev/null || true
}

cleanup_all() {
    cleanup_perception
    cleanup_camera
    cleanup_rviz
}

# ============================================================================
# 话题等待函数
# ============================================================================
wait_for_topic() {
    local topic=$1
    local timeout=${2:-30}
    local count=0

    while [ $count -lt $timeout ]; do
        if ros2 topic list 2>/dev/null | grep -q "^${topic}$"; then
            return 0
        fi
        sleep 1
        ((count++))
    done
    return 1
}

# ============================================================================
# 服务检查函数
# ============================================================================
check_service() {
    local url=$1
    local name=$2
    local timeout=${3:-2}

    if curl -s --noproxy '*' --connect-timeout $timeout --max-time 3 "$url" >/dev/null 2>&1; then
        echo -e "${GREEN}   ✓ $name${NC}"
        return 0
    else
        echo -e "${YELLOW}   ⚠ $name 不可达 ($url)${NC}"
        return 1
    fi
}

# ============================================================================
# USB 设备检查
# ============================================================================
check_realsense_usb() {
    local count=$(lsusb | grep -ci "intel" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
        echo -e "${GREEN}   ✓ 检测到 $count 个 RealSense 设备${NC}"
        return 0
    else
        echo -e "${RED}   ✗ 未检测到 RealSense USB 设备${NC}"
        return 1
    fi
}
