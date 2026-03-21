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
DETECTOR_TYPE="sam3"
ENABLE_RVIZ="false"
SKIP_CAMERA="false"
RUN_TEST="false"
CUSTOM_PROMPT=""
EXTRINSICS_SUFFIX=""
ENABLE_CDM="false"
DEPTH_MODE="combined"   # combined | fs | raw

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
        --extrinsics-suffix=*)
            EXTRINSICS_SUFFIX="${arg#*=}"
            ;;
        --no-cdm)
            ENABLE_CDM="false"
            ;;
        --combined)
            DEPTH_MODE="combined"
            ;;
        --foundation-stereo|--fs-only)
            DEPTH_MODE="fs"
            ;;
        --raw-depth)
            DEPTH_MODE="raw"
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

# Combined 模式: SAM3+FS 捆绑在 combined_server，不需要 standalone SAM3
if [ "$DEPTH_MODE" = "combined" ]; then
    COMBINED_OK=true
    check_service "$COMBINED_GPU0_URL" "Combined GPU0" || COMBINED_OK=false
    check_service "$COMBINED_GPU1_URL" "Combined GPU1" || COMBINED_OK=false
    if [ "$COMBINED_OK" = false ]; then
        echo -e "${YELLOW}   Combined 服务不可用，回退到 FS + standalone SAM3${NC}"
        DEPTH_MODE="fs"
    fi
fi

# 非 combined 模式: 需要 standalone 检测器
if [ "$DEPTH_MODE" != "combined" ]; then
    DETECTOR_OK=false
    if [ "$DETECTOR_TYPE" = "sam3" ]; then
        check_service "$SAM3_URL" "SAM3 服务" && DETECTOR_OK=true
    else
        check_service "$DINOX_URL" "DINO-X 服务" && DETECTOR_OK=true
    fi
    if [ "$DETECTOR_OK" = false ]; then
        echo -e "${RED}   ✗ 检测器服务必须可用${NC}"
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

    # FoundationStereo 需要启用相机的 IR 流（关闭投影仪）
    INFRA_ARGS=""
    if [ "$DEPTH_MODE" = "combined" ] || [ "$DEPTH_MODE" = "fs" ]; then
        INFRA_ARGS="top_enable_infra:=true chassis_enable_infra:=true"
    fi

    # 根据相机类型设置参数
    case $CAMERA_NAME in
        top)
            CAM_ARGS="top_enable:=true hand_enable:=false chassis_enable:=false $INFRA_ARGS"
            WAIT_TOPIC="/camera/top/color/image_raw"
            ;;
        hand)
            CAM_ARGS="top_enable:=false hand_enable:=true chassis_enable:=false"
            WAIT_TOPIC="/camera/hand/color/image_raw"
            ;;
        chassis)
            CAM_ARGS="top_enable:=false hand_enable:=false chassis_enable:=true $INFRA_ARGS"
            WAIT_TOPIC="/camera/chassis/color/image_raw"
            ;;
        dual)
            CAM_ARGS="top_enable:=true hand_enable:=false chassis_enable:=true $INFRA_ARGS"
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

    # Dual 模式: 等 Top 深度稳定后验证 chassis depth，失败则自动重启 chassis 节点
    # 根本原因: Top D455 硬件 reset (~7s) 期间 chassis 启动会导致 USB interface busy
    # 修复策略: Top 稳定后重启 chassis，此时无 USB 竞争
    if [ "$CAMERA_NAME" = "dual" ]; then
        echo "   等待 Top 深度流稳定..."
        if wait_for_topic "/camera/top/depth/image_rect_raw" 20; then
            sleep 3  # USB 总线完全静默
            if ! ros2 topic list 2>/dev/null | grep -q "/camera/chassis/depth/image_rect_raw"; then
                echo -e "${YELLOW}   ⚠ Chassis 深度流未启动（USB 竞争），自动恢复中...${NC}"
                CHASSIS_CAM_PID=$(ps aux | grep "realsense2_camera_node.*__node:=chassis" | grep -v grep | awk '{print $2}' | head -1)
                CHASSIS_PARAMS=$(grep -rl "337122071540" /tmp/launch_params_* 2>/dev/null | head -1)
                if [ -n "$CHASSIS_CAM_PID" ] && [ -n "$CHASSIS_PARAMS" ]; then
                    kill -9 "$CHASSIS_CAM_PID" 2>/dev/null
                    sleep 3
                    ros2 run realsense2_camera realsense2_camera_node \
                        --ros-args \
                        -r __node:=chassis \
                        -r __ns:=/camera \
                        --params-file "$CHASSIS_PARAMS" \
                        >> /tmp/camera_driver.log 2>&1 &
                    echo "   Chassis 节点已重启 (PID=$!)"
                    if wait_for_topic "/camera/chassis/depth/image_rect_raw" 20; then
                        echo -e "${GREEN}   ✓ Chassis 深度流恢复成功${NC}"
                    else
                        echo -e "${YELLOW}   ⚠ Chassis 深度流恢复失败，继续启动...${NC}"
                    fi
                else
                    echo -e "${YELLOW}   ⚠ 未找到 chassis 节点或参数文件，跳过恢复${NC}"
                fi
            else
                echo -e "${GREEN}   ✓ Chassis 深度流正常${NC}"
            fi
        fi
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

CDM_ARG=""
if [ "$ENABLE_CDM" = "false" ]; then
    CDM_ARG="enable_depth_optimizer:=false"
    echo -e "${YELLOW}   CDM 深度优化: 已禁用 (使用原始深度)${NC}"
fi

DEPTH_ARG=""
if [ "$DEPTH_MODE" = "combined" ]; then
    DEPTH_ARG="depth_source:=combined top_enable_infra:=true chassis_enable_infra:=true"
    echo -e "${YELLOW}   Combined SAM3+FS: 已启用 (单次HTTP，双GPU)${NC}"
elif [ "$DEPTH_MODE" = "fs" ]; then
    DEPTH_ARG="depth_source:=foundation_stereo top_enable_infra:=true chassis_enable_infra:=true"
    echo -e "${YELLOW}   FoundationStereo: 已启用 (被动双目深度估计)${NC}"
else
    echo -e "${YELLOW}   深度: 原始 RealSense 深度${NC}"
fi

echo "   启动: ros2 launch perception $LAUNCH_FILE $RVIZ_ARG $DETECTOR_ARG $EXTRINSICS_ARG $CDM_ARG $DEPTH_ARG"
echo ""

# 如果需要测试，后台启动
if [ "$RUN_TEST" = "true" ]; then
    ros2 launch perception $LAUNCH_FILE $RVIZ_ARG $DETECTOR_ARG $EXTRINSICS_ARG $CDM_ARG $DEPTH_ARG > /tmp/perception_launch.log 2>&1 &
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
    exec ros2 launch perception $LAUNCH_FILE $RVIZ_ARG $DETECTOR_ARG $EXTRINSICS_ARG $CDM_ARG $DEPTH_ARG
fi
