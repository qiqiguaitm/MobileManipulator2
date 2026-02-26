#!/bin/bash
# ============================================================================
# 三阶段导航测试脚本
# ============================================================================
# Usage: ./run_approach_nav_test.sh [options]
#
# Options:
#   --prompt=TEXT   检测提示词 (如 "bottle.cup.box")
#   --auto          自动选择最近的物体
#
# Examples:
#   ./run_approach_nav_test.sh                    # 交互式选择
#   ./run_approach_nav_test.sh --prompt=bottle    # 指定检测类别
#   ./run_approach_nav_test.sh --auto             # 自动选择最近物体
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ros2_env.sh"

# 配置环境
setup_ros2_env

# 运行测试脚本
exec /usr/bin/python3 "$SCRIPT_DIR/test_approach_nav.py" "$@"
