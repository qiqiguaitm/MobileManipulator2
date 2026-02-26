#!/bin/bash
# ============================================================================
# Perception 模块构建脚本 (ROS2 Humble)
# 用法: ./build_perception.sh [build|clean|verify|deps]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ros2_env.sh"

# 感知包列表
PERCEPTION_PACKAGES=(
    "perception_interfaces"
    "perception_core"
    "perception_nodes"
    "perception_bringup"
)

# 相机驱动包
CAMERA_PACKAGES=(
    "camera_driver"
)

echo "========================================"
echo "Perception 模块构建 (ROS2)"
echo "========================================"

# ============================================================================
# 检查依赖
# ============================================================================
check_deps() {
    echo -e "${CYAN}[依赖] 检查构建依赖...${NC}"

    local MISSING_DEPS=0

    # 1. 检查 ROS2
    if [ ! -f "/opt/ros/humble/setup.bash" ]; then
        echo -e "${RED}   ✗ ROS2 Humble 未安装${NC}"
        MISSING_DEPS=1
    else
        echo -e "${GREEN}   ✓ ROS2 Humble${NC}"
    fi

    # 2. 检查 Python 依赖
    export PATH="/usr/bin:$PATH"

    python3 -c "import cv2" 2>/dev/null && echo -e "${GREEN}   ✓ OpenCV (Python)${NC}" || {
        echo -e "${RED}   ✗ OpenCV Python 未安装${NC}"
        MISSING_DEPS=1
    }

    python3 -c "import numpy" 2>/dev/null && echo -e "${GREEN}   ✓ NumPy${NC}" || {
        echo -e "${RED}   ✗ NumPy 未安装${NC}"
        MISSING_DEPS=1
    }

    python3 -c "import requests" 2>/dev/null && echo -e "${GREEN}   ✓ Requests${NC}" || {
        echo -e "${RED}   ✗ Requests 未安装: pip install requests${NC}"
        MISSING_DEPS=1
    }

    python3 -c "from pycocotools import mask" 2>/dev/null && echo -e "${GREEN}   ✓ pycocotools${NC}" || {
        echo -e "${YELLOW}   ⚠ pycocotools 未安装: pip install pycocotools${NC}"
    }

    # 3. 检查外部服务（运行时需要，构建时可选）
    echo -e "${CYAN}[依赖] 检查外部服务...${NC}"
    check_service "$DINOX_URL" "DINO-X 服务" || true
    check_service "$CDM_URL" "CDM 服务" || true

    # 4. 检查 USB 设备
    echo -e "${CYAN}[依赖] 检查硬件...${NC}"
    check_realsense_usb || true

    echo ""
    if [ $MISSING_DEPS -eq 1 ]; then
        echo -e "${RED}存在缺失依赖，请先安装。${NC}"
        return 1
    fi
    return 0
}

# ============================================================================
# 构建
# ============================================================================
build_perception() {
    echo -e "${CYAN}[构建] 编译 Perception ROS2 包...${NC}"

    cd "$WS_DIR"
    setup_ros2_env

    # 构建感知包（按依赖顺序）
    colcon build \
        --packages-up-to ${PERCEPTION_PACKAGES[*]} ${CAMERA_PACKAGES[*]} \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers $(nproc)

    # 重新 source
    source "$WS_DIR/install/setup.bash"

    echo -e "${GREEN}   ✓ Perception 构建完成${NC}"
}

# ============================================================================
# 验证构建
# ============================================================================
verify_build() {
    echo -e "${CYAN}[验证] 检查构建结果...${NC}"

    setup_ros2_env

    local ALL_OK=true

    # 检查包是否存在
    for pkg in "${PERCEPTION_PACKAGES[@]}" "${CAMERA_PACKAGES[@]}"; do
        if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
            echo -e "${GREEN}   ✓ $pkg${NC}"
        else
            echo -e "${RED}   ✗ $pkg 未找到${NC}"
            ALL_OK=false
        fi
    done

    # 检查 launch 文件
    echo ""
    echo "  检查 launch 文件..."
    local LAUNCH_DIR="$WS_DIR/install/perception_bringup/share/perception_bringup/launch"

    for launch in perception_3d_rviz.launch.py multi_camera_3d_rviz.launch.py; do
        if [ -f "$LAUNCH_DIR/$launch" ]; then
            echo -e "${GREEN}   ✓ $launch${NC}"
        else
            echo -e "${YELLOW}   ⚠ $launch 未找到${NC}"
        fi
    done

    echo ""
    if [ "$ALL_OK" = true ]; then
        echo -e "${GREEN}验证完成！${NC}"
        return 0
    else
        echo -e "${RED}部分包缺失${NC}"
        return 1
    fi
}

# ============================================================================
# 清理
# ============================================================================
clean_perception() {
    echo -e "${YELLOW}[清理] 清理 Perception 构建...${NC}"

    cd "$WS_DIR"

    for pkg in "${PERCEPTION_PACKAGES[@]}" "${CAMERA_PACKAGES[@]}"; do
        rm -rf "build/$pkg" "install/$pkg" 2>/dev/null || true
        echo "   已清理: $pkg"
    done

    echo -e "${GREEN}   ✓ 清理完成${NC}"
}

# ============================================================================
# 主逻辑
# ============================================================================
case "${1:-build}" in
    deps|check)
        check_deps
        ;;
    build)
        check_deps && build_perception && verify_build
        ;;
    clean)
        clean_perception
        ;;
    verify)
        verify_build
        ;;
    *)
        echo "用法: $0 [deps|build|clean|verify]"
        echo "  deps   - 检查依赖"
        echo "  build  - 构建 Perception 包 (默认)"
        echo "  clean  - 清理构建"
        echo "  verify - 验证构建结果"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}完成!${NC}"
echo -e "${GREEN}========================================${NC}"
