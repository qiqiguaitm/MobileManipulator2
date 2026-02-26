#!/bin/bash
# ============================================================================
# 导航系统构建脚本 (ROS2 版本)
# 用法: ./build_navigation.sh [all|ndt|loc|slam|mapping|full|clean|verify]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS="$(dirname "$SCRIPT_DIR")"
NPROC=$(nproc)

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "导航系统构建 (ROS2)"
echo "========================================"

# ============================================================================
# 环境配置
# ============================================================================
setup_env() {
    echo -e "${CYAN}[ENV] 配置构建环境...${NC}"

    source /opt/ros/humble/setup.bash
    [ -f "$ROS_WS/install/setup.bash" ] && source "$ROS_WS/install/setup.bash"

    export PATH="/usr/bin:$PATH"
    export ROS_DOMAIN_ID=42

    echo "  ROS_WS: $ROS_WS"
    echo "  NPROC: $NPROC"
}

# ============================================================================
# 构建点云配准库 (ndt_omp)
# ============================================================================
build_ndt() {
    echo -e "${CYAN}[NDT] 构建点云配准库...${NC}"

    cd "$ROS_WS"
    colcon build --packages-select ndt_omp --parallel-workers $NPROC

    echo -e "${GREEN}   ✓ ndt_omp 构建完成${NC}"
}

# ============================================================================
# 构建定位包 (hdl_global_localization, hdl_localization)
# ============================================================================
build_localization() {
    echo -e "${CYAN}[LOC] 构建定位包...${NC}"

    cd "$ROS_WS"
    colcon build --packages-select hdl_global_localization hdl_localization --parallel-workers $NPROC

    echo -e "${GREEN}   ✓ 定位包构建完成${NC}"
}

# ============================================================================
# 构建 SLAM 包 (launch 和配置文件)
# ============================================================================
build_slam_pkg() {
    echo -e "${CYAN}[SLAM] 构建 SLAM 包...${NC}"

    cd "$ROS_WS"
    colcon build --packages-select slam --parallel-workers $NPROC

    echo -e "${GREEN}   ✓ SLAM 包构建完成${NC}"
}

# ============================================================================
# 构建建图模块 (fast_lio + sc_pgo)
# ============================================================================
build_mapping() {
    echo -e "${CYAN}[MAPPING] 构建建图模块...${NC}"

    cd "$ROS_WS"

    echo -e "${CYAN}   构建 FAST-LIO...${NC}"
    colcon build --packages-select fast_lio --parallel-workers $NPROC

    # sc_pgo 如果存在才构建
    if [ -d "$ROS_WS/src/sc_pgo" ]; then
        echo -e "${CYAN}   构建 SC-PGO...${NC}"
        colcon build --packages-select sc_pgo --parallel-workers $NPROC
    fi

    echo -e "${GREEN}   ✓ 建图模块构建完成${NC}"
}

# ============================================================================
# 构建导航相关 (ndt + loc + slam)
# ============================================================================
build_nav() {
    echo -e "${CYAN}[NAV] 构建导航相关包...${NC}"

    cd "$ROS_WS"
    colcon build --packages-select \
        ndt_omp \
        hdl_global_localization \
        hdl_localization \
        slam \
        --parallel-workers $NPROC

    echo -e "${GREEN}   ✓ 导航相关包构建完成${NC}"
}

# ============================================================================
# 构建 SLAM 相关 (ndt + loc + fastlio + slam)
# ============================================================================
build_slam() {
    echo -e "${CYAN}[SLAM] 构建 SLAM 相关包...${NC}"

    cd "$ROS_WS"

    local packages="ndt_omp hdl_global_localization hdl_localization fast_lio slam"

    colcon build --packages-select $packages --parallel-workers $NPROC

    echo -e "${GREEN}   ✓ SLAM 相关包构建完成${NC}"
}

# ============================================================================
# 清理
# ============================================================================
clean_all() {
    echo -e "${YELLOW}清理构建...${NC}"
    cd "$ROS_WS"
    rm -rf build install log
    echo -e "${GREEN}清理完成${NC}"
}

# ============================================================================
# 验证构建
# ============================================================================
verify_build() {
    echo -e "${CYAN}[验证] 检查构建结果...${NC}"

    source "$ROS_WS/install/setup.bash"

    local all_ok=true
    local packages="ndt_omp hdl_global_localization hdl_localization slam"

    for pkg in $packages; do
        if ros2 pkg prefix $pkg &>/dev/null; then
            echo -e "${GREEN}   ✓ $pkg 包存在${NC}"
        else
            echo -e "${RED}   ✗ $pkg 包未找到${NC}"
            all_ok=false
        fi
    done

    if [ "$all_ok" = true ]; then
        echo -e "${GREEN}   所有组件验证通过！${NC}"
        return 0
    else
        echo -e "${RED}   部分组件验证失败${NC}"
        return 1
    fi
}

# ============================================================================
# 主逻辑
# ============================================================================
setup_env

case "${1:-nav}" in
    ndt)
        build_ndt
        ;;
    loc)
        build_localization
        ;;
    slam-pkg)
        build_slam_pkg
        ;;
    mapping)
        build_mapping
        ;;
    nav)
        build_nav
        ;;
    slam)
        build_slam
        ;;
    full)
        build_slam
        build_mapping
        verify_build
        ;;
    clean)
        clean_all
        exit 0
        ;;
    verify)
        verify_build
        ;;
    *)
        echo "用法: $0 [nav|slam|mapping|full|clean|verify]"
        echo ""
        echo "主要命令:"
        echo "  nav     - 构建导航相关 (ndt + hdl_loc + slam) [默认]"
        echo "  slam    - 构建 SLAM 相关 (nav + fast_lio)"
        echo "  mapping - 构建建图模块 (fast_lio + sc_pgo)"
        echo "  full    - 完整构建 (slam + mapping)"
        echo ""
        echo "单独构建:"
        echo "  ndt      - 仅构建 ndt_omp"
        echo "  loc      - 仅构建定位包"
        echo "  slam-pkg - 仅构建 slam 包"
        echo ""
        echo "其他:"
        echo "  clean   - 清理构建"
        echo "  verify  - 验证构建结果"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================"
echo "构建完成!"
echo "========================================${NC}"
