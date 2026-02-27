#!/bin/bash
# ============================================================================
# Scene Perception 3D 启动脚本 (ROS2 Humble)
# ============================================================================
# Usage: ./start_perception_3d.sh [options]
#
# Options:
#   --camera=NAME       相机名称 (top | chassis | hand | dual | none), 默认: top
#   --detector=TYPE     检测器类型 (dinox | sam3), 默认: dinox
#   --rviz              启动 RViz 3D 可视化
#   --skip-camera       跳过相机启动（假设相机已运行）
#   --test              启动后自动测试服务
#   --prompt=TEXT       自定义检测提示词
#   --new-extrinsics    使用新标定的外参 (_new 后缀)
#
# Examples:
#   ./start_perception_3d.sh                    # Top相机 + 感知
#   ./start_perception_3d.sh --rviz             # Top相机 + 感知 + RViz
#   ./start_perception_3d.sh --camera=dual      # 双相机 + 感知
#   ./start_perception_3d.sh --skip-camera      # 仅启动感知（相机已运行）
#   ./start_perception_3d.sh --rviz --test      # 启动并测试
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ros2_env.sh"

# ============================================================================
# 默认参数
# ============================================================================
CAMERA_NAME="top"
DETECTOR_TYPE="dinox"
ENABLE_RVIZ="false"
SKIP_CAMERA="false"
RUN_TEST="false"
CUSTOM_PROMPT=""
EXTRINSICS_SUFFIX=""

# ============================================================================
# 解析参数
# ============================================================================
for arg in "$@"; do
    case $arg in
        --camera=*)
            CAMERA_NAME="${arg#*=}"
            ;;
        --detector=*)
            DETECTOR_TYPE="${arg#*=}"
            ;;
        --rviz)
            ENABLE_RVIZ="true"
            ;;
        --skip-camera)
            SKIP_CAMERA="true"
            ;;
        --test)
            RUN_TEST="true"
            ;;
        --prompt=*)
            CUSTOM_PROMPT="${arg#*=}"
            ;;
        --new-extrinsics)
            EXTRINSICS_SUFFIX="_new"
            ;;
        -h|--help)
            head -25 "$0" | tail -23
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
CAMERA_PID=""
LAUNCH_PID=""

cleanup() {
    echo ""
    echo -e "${YELLOW}正在清理...${NC}"
    [ -n "$CAMERA_PID" ] && kill $CAMERA_PID 2>/dev/null || true
    [ -n "$LAUNCH_PID" ] && kill $LAUNCH_PID 2>/dev/null || true
    cleanup_perception
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 显示配置
# ============================================================================
echo "========================================"
echo "Scene Perception 3D (ROS2)"
echo "========================================"
echo "Camera:     $CAMERA_NAME"
echo "Detector:   $DETECTOR_TYPE"
echo "RViz:       $ENABLE_RVIZ"
echo "SkipCamera: $SKIP_CAMERA"
echo "Test:       $RUN_TEST"
echo "========================================"

# ============================================================================
# 环境配置
# ============================================================================
echo "[1/5] 配置环境..."
setup_ros2_env
echo -e "${GREEN}   ✓ 环境已配置${NC}"

# ============================================================================
# 检查外部服务
# ============================================================================
echo "[2/5] 检查外部服务..."
DETECTOR_OK=false

if [ "$DETECTOR_TYPE" = "sam3" ]; then
    check_service "$SAM3_URL" "SAM3 服务" && DETECTOR_OK=true
    if [ "$DETECTOR_OK" = false ]; then
        echo -e "${RED}   ✗ SAM3 服务必须可用${NC}"
        exit 1
    fi
else
    check_service "$DINOX_URL" "DINO-X 服务" && DETECTOR_OK=true
    if [ "$DETECTOR_OK" = false ]; then
        echo -e "${RED}   ✗ DINO-X 服务必须可用${NC}"
        exit 1
    fi
fi

check_service "$CDM_URL" "CDM 服务" || true

# ============================================================================
# 清理旧进程
# ============================================================================
echo "[3/5] 清理旧进程..."
cleanup_perception
if [ "$SKIP_CAMERA" = "false" ]; then
    cleanup_camera
fi
echo -e "${GREEN}   ✓ 清理完成${NC}"

# ============================================================================
# 启动相机（如果需要）
# ============================================================================
if [ "$SKIP_CAMERA" = "false" ] && [ "$CAMERA_NAME" != "none" ]; then
    echo "[4/5] 启动相机驱动..."

    # 检查 USB 设备
    check_realsense_usb || {
        echo -e "${RED}   ✗ 无法继续${NC}"
        exit 1
    }

    # 根据相机类型设置参数
    case $CAMERA_NAME in
        top)
            CAM_ARGS="top_enable:=true hand_enable:=false chassis_enable:=false"
            WAIT_TOPIC="/camera/top/color/image_raw"
            ;;
        hand)
            CAM_ARGS="top_enable:=false hand_enable:=true chassis_enable:=false"
            WAIT_TOPIC="/camera/hand/color/image_raw"
            ;;
        chassis)
            CAM_ARGS="top_enable:=false hand_enable:=false chassis_enable:=true"
            WAIT_TOPIC="/camera/chassis/color/image_raw"
            ;;
        dual)
            CAM_ARGS="top_enable:=true hand_enable:=false chassis_enable:=true"
            WAIT_TOPIC="/camera/top/color/image_raw"
            ;;
        *)
            echo -e "${RED}   ✗ 未知相机: $CAMERA_NAME${NC}"
            exit 1
            ;;
    esac

    echo "   启动: ros2 launch camera_driver camera_driver.launch.py $CAM_ARGS"
    ros2 launch camera_driver camera_driver.launch.py $CAM_ARGS > /tmp/camera_driver.log 2>&1 &
    CAMERA_PID=$!

    # 等待相机启动
    echo "   等待相机启动..."
    if wait_for_topic "$WAIT_TOPIC" 25; then
        echo -e "${GREEN}   ✓ 相机启动成功${NC}"
    else
        echo -e "${YELLOW}   ⚠ 相机话题等待超时，继续启动...${NC}"
    fi
else
    echo "[4/5] 跳过相机启动"
fi

# ============================================================================
# 启动感知节点
# ============================================================================
echo "[5/5] 启动感知节点..."

# 选择 launch 文件
if [ "$CAMERA_NAME" = "dual" ]; then
    LAUNCH_FILE="multi_camera_3d_rviz.launch.py"
else
    LAUNCH_FILE="perception_3d_rviz.launch.py"
fi

RVIZ_ARG="rviz:=$ENABLE_RVIZ"
DETECTOR_ARG="detector_type:=$DETECTOR_TYPE"
EXTRINSICS_ARG=""
if [ -n "$EXTRINSICS_SUFFIX" ]; then
    EXTRINSICS_ARG="extrinsics_suffix:=$EXTRINSICS_SUFFIX"
    echo -e "${YELLOW}   使用新外参: $EXTRINSICS_SUFFIX${NC}"
fi

echo "   启动: ros2 launch perception $LAUNCH_FILE $RVIZ_ARG $DETECTOR_ARG $EXTRINSICS_ARG"
echo ""

# 如果需要测试，后台启动
if [ "$RUN_TEST" = "true" ]; then
    ros2 launch perception $LAUNCH_FILE $RVIZ_ARG $DETECTOR_ARG $EXTRINSICS_ARG > /tmp/perception_launch.log 2>&1 &
    LAUNCH_PID=$!

    # 等待节点启动
    echo "等待感知节点启动..."
    sleep 8

    # 检查进程是否存活
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then
        echo -e "${RED}节点启动失败，查看日志:${NC}"
        tail -30 /tmp/perception_launch.log
        exit 1
    fi

    echo -e "${GREEN}节点启动成功！${NC}"
    echo ""

    # 执行测试
    echo "========================================"
    echo -e "${CYAN}测试检测服务...${NC}"
    echo "========================================"

    TEST_PROMPT="${CUSTOM_PROMPT:-bottle.cup.box.pen.phone}"
    echo "Prompt: $TEST_PROMPT"
    echo ""

    # ROS2 服务调用
    RESULT=$(ros2 service call /scene_perception_3d/detect perception/srv/DetectObjects "{prompt: '$TEST_PROMPT'}" 2>&1 || echo "服务调用失败")

    echo "$RESULT" | head -50
    echo ""

    if echo "$RESULT" | grep -q "success=True\|success: true"; then
        echo -e "${GREEN}========================================"
        echo "测试成功！"
        echo "========================================${NC}"
    else
        echo -e "${YELLOW}========================================"
        echo "未检测到物体或发生错误"
        echo "========================================${NC}"
    fi

    echo ""
    echo "节点继续运行中... (Ctrl+C 停止)"
    wait $LAUNCH_PID
else
    # 前台启动
    exec ros2 launch perception $LAUNCH_FILE $RVIZ_ARG $DETECTOR_ARG $EXTRINSICS_ARG
fi
