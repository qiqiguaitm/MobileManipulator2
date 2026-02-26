#!/bin/bash
# ============================================================================
# Perception + Navigation 一键启动脚本 (ROS2 Humble)
# ============================================================================
# Usage: ./start_percept_nav.sh [options]
#
# Options:
#   --no-rviz           不启动 RViz
#   --no-fusion         不使用轮速融合
#   --skip-nav          跳过导航栈启动（假设已运行）
#   --skip-camera       跳过相机启动（假设已运行）
#   --skip-perception   跳过感知启动（假设已运行）
#
# Examples:
#   ./start_percept_nav.sh                    # 完整启动
#   ./start_percept_nav.sh --no-rviz          # 不启动 RViz
#   ./start_percept_nav.sh --skip-nav         # 仅启动感知和导航节点
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ros2_env.sh"

# ============================================================================
# 默认参数
# ============================================================================
ENABLE_RVIZ="true"
USE_FUSION="true"
SKIP_NAV="false"
SKIP_CAMERA="false"
SKIP_PERCEPTION="false"

# ============================================================================
# 解析参数
# ============================================================================
for arg in "$@"; do
    case $arg in
        --no-rviz)
            ENABLE_RVIZ="false"
            ;;
        --no-fusion)
            USE_FUSION="false"
            ;;
        --skip-nav)
            SKIP_NAV="true"
            ;;
        --skip-camera)
            SKIP_CAMERA="true"
            ;;
        --skip-perception)
            SKIP_PERCEPTION="true"
            ;;
        -h|--help)
            head -18 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "未知参数: $arg"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 清理函数
# ============================================================================
NAV_PID=""
CAMERA_PID=""
PERCEPTION_PID=""
APPROACH_PID=""

cleanup() {
    echo ""
    echo -e "${YELLOW}正在清理...${NC}"
    [ -n "$APPROACH_PID" ] && kill $APPROACH_PID 2>/dev/null || true
    [ -n "$PERCEPTION_PID" ] && kill $PERCEPTION_PID 2>/dev/null || true
    [ -n "$CAMERA_PID" ] && kill $CAMERA_PID 2>/dev/null || true
    [ -n "$NAV_PID" ] && kill $NAV_PID 2>/dev/null || true

    # 清理 approach_navigator
    pkill -15 -f "[a]pproach_navigator" 2>/dev/null || true
    cleanup_perception
    sleep 1
    pkill -9 -f "[a]pproach_navigator" 2>/dev/null || true
    echo -e "${GREEN}清理完成${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 显示配置
# ============================================================================
echo "========================================"
echo "Perception + Navigation 一键启动"
echo "========================================"
echo "RViz:           $ENABLE_RVIZ"
echo "轮速融合:       $USE_FUSION"
echo "跳过导航栈:     $SKIP_NAV"
echo "跳过相机:       $SKIP_CAMERA"
echo "跳过感知:       $SKIP_PERCEPTION"
echo "========================================"

# ============================================================================
# 环境配置
# ============================================================================
echo "[1/6] 配置环境..."
setup_ros2_env
echo -e "${GREEN}   ✓ 环境已配置${NC}"

# ============================================================================
# 检查外部服务
# ============================================================================
echo "[2/6] 检查外部服务..."
DINOX_OK=false
check_service "$DINOX_URL" "DINO-X 服务" && DINOX_OK=true
check_service "$CDM_URL" "CDM 服务" || true

if [ "$DINOX_OK" = false ] && [ "$SKIP_PERCEPTION" = "false" ]; then
    echo -e "${RED}   ✗ DINO-X 服务必须可用${NC}"
    exit 1
fi

# ============================================================================
# 启动导航栈 (如果需要)
# ============================================================================
if [ "$SKIP_NAV" = "false" ]; then
    echo "[3/6] 启动导航栈..."

    NAV_ARGS="enable_rviz:=$ENABLE_RVIZ"
    if [ "$USE_FUSION" = "true" ]; then
        NAV_ARGS="$NAV_ARGS use_odom_fusion:=true launch_chassis:=true"
    else
        NAV_ARGS="$NAV_ARGS use_odom_fusion:=false launch_chassis:=false"
    fi

    echo "   启动: ros2 launch slam hdl_navigation_launch.py $NAV_ARGS"
    ros2 launch slam hdl_navigation_launch.py $NAV_ARGS > /tmp/navigation.log 2>&1 &
    NAV_PID=$!

    # 等待导航栈启动
    echo "   等待导航栈启动..."
    if wait_for_topic "/map" 30; then
        echo -e "${GREEN}   ✓ 导航栈启动成功${NC}"
    else
        echo -e "${YELLOW}   ⚠ 导航栈等待超时，继续启动...${NC}"
    fi
    sleep 3
else
    echo "[3/6] 跳过导航栈启动"
fi

# ============================================================================
# 启动相机 (如果需要)
# ============================================================================
if [ "$SKIP_CAMERA" = "false" ]; then
    echo "[4/6] 启动相机驱动..."

    # 检查 USB 设备
    check_realsense_usb || {
        echo -e "${YELLOW}   ⚠ 未检测到 RealSense，跳过相机启动${NC}"
        SKIP_CAMERA="true"
    }

    if [ "$SKIP_CAMERA" = "false" ]; then
        # 启动双相机 (Top + Chassis)
        CAM_ARGS="top_enable:=true hand_enable:=false chassis_enable:=true"
        echo "   启动: ros2 launch camera_driver camera_driver.launch.py $CAM_ARGS"
        ros2 launch camera_driver camera_driver.launch.py $CAM_ARGS > /tmp/camera_driver.log 2>&1 &
        CAMERA_PID=$!

        # 等待相机启动
        echo "   等待相机启动..."
        if wait_for_topic "/camera/top/color/image_raw" 25; then
            echo -e "${GREEN}   ✓ 相机启动成功${NC}"
        else
            echo -e "${YELLOW}   ⚠ 相机等待超时，继续启动...${NC}"
        fi
    fi
else
    echo "[4/6] 跳过相机启动"
fi

# ============================================================================
# 启动感知节点 (如果需要)
# ============================================================================
if [ "$SKIP_PERCEPTION" = "false" ]; then
    echo "[5/6] 启动感知节点..."

    # 使用双相机感知
    LAUNCH_FILE="multi_camera_perception.launch.py"
    PERCEPTION_ARGS="use_camera_driver:=false auto_detect_rate:=1.0"

    echo "   启动: ros2 launch perception_bringup $LAUNCH_FILE $PERCEPTION_ARGS"
    ros2 launch perception_bringup $LAUNCH_FILE $PERCEPTION_ARGS > /tmp/perception.log 2>&1 &
    PERCEPTION_PID=$!

    # 等待感知节点启动
    echo "   等待感知节点启动..."
    sleep 5
    if wait_for_topic "/multi_camera_perception/fused/objects_3d" 15; then
        echo -e "${GREEN}   ✓ 感知节点启动成功${NC}"
    else
        echo -e "${YELLOW}   ⚠ 感知节点等待超时，继续启动...${NC}"
    fi
else
    echo "[5/6] 跳过感知启动"
fi

# ============================================================================
# 启动 Approach Navigator
# ============================================================================
echo "[6/6] 启动 Approach Navigator..."

echo "   启动: ros2 launch approach_navigator approach_navigator.launch.py"
ros2 launch approach_navigator approach_navigator.launch.py > /tmp/approach_navigator.log 2>&1 &
APPROACH_PID=$!

sleep 3

# 检查进程是否存活
if kill -0 $APPROACH_PID 2>/dev/null; then
    echo -e "${GREEN}   ✓ Approach Navigator 启动成功${NC}"
else
    echo -e "${RED}   ✗ Approach Navigator 启动失败${NC}"
    tail -20 /tmp/approach_navigator.log
    cleanup
    exit 1
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================"
echo -e "${GREEN}所有模块启动完成！${NC}"
echo "========================================"
echo ""
echo "可用话题:"
echo "  - /multi_camera_perception/fused/objects_3d  (感知结果)"
echo "  - /approach_navigator/test_approach          (测试服务)"
echo ""
echo "测试三阶段导航:"
echo "  ros2 run approach_navigator test_approach_nav.py"
echo ""
echo "按 Ctrl+C 停止所有节点"
echo "========================================"

# 前台等待
wait $APPROACH_PID
