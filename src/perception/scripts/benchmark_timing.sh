#!/bin/bash
# 感知节点性能压测 - 带时间分布统计

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERCEPTION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PERCEPTION_ROOT/../.." && pwd)"

echo "======================================"
echo "感知节点性能压测 (带时间分布)"
echo "======================================"
echo ""

# 检查参数
MODE="${1:-with-lidar}"
if [ "$MODE" == "no-lidar" ]; then
    LIDAR_FLAG="--no-lidar"
    echo "模式: 禁用 LiDAR"
else
    LIDAR_FLAG=""
    echo "模式: 启用 LiDAR"
fi
echo ""

# 1. 检查依赖服务
echo "1. 检查外部服务..."
if ! curl -s --noproxy '*' --connect-timeout 2 --max-time 3 http://192.168.112.14:10086 > /dev/null 2>&1; then
    echo "✗ DINO-X 服务不可用"
    exit 1
fi
echo "  ✓ DINO-X 服务: OK"

if ! curl -s --noproxy '*' --connect-timeout 2 --max-time 3 http://192.168.112.14:8086 > /dev/null 2>&1; then
    echo "  ⚠ CDM 服务不可用"
else
    echo "  ✓ CDM 服务: OK"
fi
echo ""

# 2. 编译
echo "2. 编译 perception 包..."
cd "$PROJECT_ROOT"
source /opt/ros/noetic/setup.bash
catkin build perception > /dev/null 2>&1
echo "  ✓ 编译成功"
echo ""

# 3. 启动节点
echo "3. 启动 scene_perception_3d 节点..."
source "$PROJECT_ROOT/devel/setup.bash"

# 杀死已有节点
if rosnode list 2>/dev/null | grep -q "scene_perception_3d"; then
    rosnode kill /scene_perception_3d > /dev/null 2>&1
    sleep 2
fi

# 启动节点 (禁用自动检测, 禁用精度日志和可视化)
roslaunch perception scene_perception_3d.launch \
    auto_detect_rate:=0 \
    enable_accuracy_log:=false \
    enable_visualization:=false \
    > /tmp/perception_benchmark_timing.log 2>&1 &
NODE_PID=$!

echo "  等待节点初始化..."
sleep 5

if ! rosnode list 2>/dev/null | grep -q "scene_perception_3d"; then
    echo "  ✗ 节点启动失败"
    kill $NODE_PID 2>/dev/null || true
    exit 1
fi
echo "  ✓ 节点启动成功"
echo ""

# 4. 运行压测
echo "4. 运行压测 (10 次调用)..."
echo "======================================"
rosrun perception benchmark_with_timing.py $LIDAR_FLAG

# 5. 清理
echo ""
echo "5. 清理..."
rosnode kill /scene_perception_3d > /dev/null 2>&1 || true
kill $NODE_PID 2>/dev/null || true
echo "  ✓ 节点已停止"
echo ""

echo "======================================"
echo "压测完成"
echo "======================================"
