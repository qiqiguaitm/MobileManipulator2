#!/bin/bash
# 感知节点性能压测 - 禁用 LiDAR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERCEPTION_ROOT="$PROJECT_ROOT/src/perception"

echo "======================================"
echo "感知节点性能压测 (禁用 LiDAR)"
echo "======================================"
echo ""

# 1. 检查依赖服务
echo "1. 检查外部服务..."
if ! curl -s --noproxy '*' --connect-timeout 2 --max-time 3 http://192.168.112.14:10086 > /dev/null 2>&1; then
    echo "✗ DINO-X 服务不可用: http://192.168.112.14:10086"
    exit 1
fi
echo "  ✓ DINO-X 服务: OK"

if ! curl -s --noproxy '*' --connect-timeout 2 --max-time 3 http://192.168.112.14:8086 > /dev/null 2>&1; then
    echo "  ⚠ CDM 服务不可用，将禁用深度优化"
else
    echo "  ✓ CDM 服务: OK"
fi
echo ""

# 2. 编译包
echo "2. 编译 perception 包..."
cd "$PROJECT_ROOT"
source /opt/ros/noetic/setup.bash
catkin build perception > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ 编译成功"
else
    echo "  ✗ 编译失败"
    exit 1
fi
echo ""

# 3. 启动节点
echo "3. 启动 scene_perception_3d 节点..."
source "$PROJECT_ROOT/devel/setup.bash"

# 杀死已有节点
if rosnode list 2>/dev/null | grep -q "scene_perception_3d"; then
    echo "  发现已有节点，正在停止..."
    rosnode kill /scene_perception_3d > /dev/null 2>&1
    sleep 2
fi

# 启动节点（禁用自动检测模式，仅测试 service）
roslaunch perception scene_perception_3d.launch auto_detect_rate:=0 > /tmp/perception_benchmark_node.log 2>&1 &
NODE_PID=$!
echo "  节点 PID: $NODE_PID"

# 等待节点启动
echo "  等待节点初始化..."
sleep 5

# 检查节点是否运行
if ! rosnode list 2>/dev/null | grep -q "scene_perception_3d"; then
    echo "  ✗ 节点启动失败"
    echo "  日志: /tmp/perception_benchmark_node.log"
    kill $NODE_PID 2>/dev/null || true
    exit 1
fi
echo "  ✓ 节点启动成功"
echo ""

# 4. 运行压测 (禁用 LiDAR)
echo "4. 运行压测 (20 次调用, 禁用 LiDAR)..."
echo "======================================"
rosrun perception perception_benchmark.py --no-lidar

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
echo ""
echo "对比有 LiDAR vs 无 LiDAR 的性能差异"
