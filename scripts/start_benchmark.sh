#!/bin/bash
# ============================================================================
# Perception Benchmark 启动脚本
# ============================================================================
# Usage: ./start_benchmark.sh [options]
#
# Options:
#   --num-runs=N        每组数据测试次数 (default: 10)
#   --warmup=N          预热次数 (default: 3)
#   --vis               运行可视化测试（输出到 /tmp/perception_benchmark/）
#   --benchmark         运行性能 benchmark（默认）
#   --samples-dir=PATH  样本数据目录
#
# Examples:
#   ./start_benchmark.sh                        # 默认 benchmark
#   ./start_benchmark.sh --num-runs=20          # 20 轮测试
#   ./start_benchmark.sh --vis                  # 可视化模式
#   ./start_benchmark.sh --vis --benchmark      # 可视化 + benchmark
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_ros2_env.sh"

# ============================================================================
# 默认参数
# ============================================================================
NUM_RUNS=10
WARMUP=3
MODE_ARGS="--benchmark"
SAMPLES_DIR=""
EXTRA_ARGS=""

# ============================================================================
# 解析参数
# ============================================================================
for arg in "$@"; do
    case $arg in
        --num-runs=*)
            NUM_RUNS="${arg#*=}"
            ;;
        --warmup=*)
            WARMUP="${arg#*=}"
            ;;
        --vis)
            MODE_ARGS="--vis $MODE_ARGS"
            ;;
        --benchmark)
            # 已是默认
            ;;
        --samples-dir=*)
            SAMPLES_DIR="${arg#*=}"
            ;;
        -h|--help)
            head -16 "$0" | tail -14
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# ============================================================================
# 环境配置
# ============================================================================
echo "========================================"
echo "Perception Benchmark"
echo "========================================"
echo "Runs:    $NUM_RUNS"
echo "Warmup:  $WARMUP"
echo "Mode:    $MODE_ARGS"
echo "========================================"

setup_ros2_env

# ============================================================================
# 检查外部服务
# ============================================================================
echo ""
echo "检查服务可用性..."
check_service "$DINOX_URL" "DinoX" || true
check_service "$SAM3_URL" "SAM3" || true
check_service "$CDM_URL" "CDM" || true
echo ""

# ============================================================================
# 构建命令参数
# ============================================================================
CMD_ARGS="$MODE_ARGS --num-runs $NUM_RUNS --warmup $WARMUP"

if [ -n "$SAMPLES_DIR" ]; then
    CMD_ARGS="$CMD_ARGS --samples-dir $SAMPLES_DIR"
fi

CMD_ARGS="$CMD_ARGS $EXTRA_ARGS"

# ============================================================================
# 运行 benchmark
# ============================================================================
BENCHMARK_SCRIPT="$WS_DIR/src/perception/src/percept_benchmark.py"

if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo -e "${RED}找不到 benchmark 脚本: $BENCHMARK_SCRIPT${NC}"
    exit 1
fi

echo "执行: python3 $BENCHMARK_SCRIPT $CMD_ARGS"
echo ""
exec python3 "$BENCHMARK_SCRIPT" $CMD_ARGS
