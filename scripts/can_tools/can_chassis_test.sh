#!/bin/bash
# ============================================================================
# CAN1 底盘控制测试脚本
# ============================================================================
# 测试底盘前进 10cm，验证 CAN1 通信正常
# ============================================================================

echo ""
echo "========================================"
echo "   ⚠️  CAN1 底盘控制测试 ⚠️"
echo "========================================"
echo ""

# 1. 检查/启动 roscore
echo "[1/6] 检查 roscore..."
if ! rostopic list &>/dev/null; then
    echo "  roscore 未运行，启动中..."
    roscore &
    sleep 3
fi
echo "  roscore 运行中 ✓"

# 2. 清理残留节点
echo "[2/6] 清理残留节点..."
rosnode kill /tracer_base /tracer_base_node 2>/dev/null || true
pkill -9 -f tracer_base 2>/dev/null || true
sleep 2

# 3. 重置 CAN1 (清除缓冲区)
echo "[3/6] 重置 CAN1..."
echo "didi" | sudo -S ip link set can1 down 2>/dev/null
echo "didi" | sudo -S ip link set can1 type can bitrate 500000 2>/dev/null
echo "didi" | sudo -S ip link set can1 up 2>/dev/null
sleep 1

CAN_STATE=$(ip -details link show can1 2>/dev/null | grep "can state" | awk '{print $3}')
CAN_BITRATE=$(ip -details link show can1 2>/dev/null | grep bitrate | awk '{print $2}')
echo "  状态: $CAN_STATE, 波特率: $CAN_BITRATE"

if [[ "$CAN_STATE" != "ERROR-ACTIVE" ]]; then
    echo "  [错误] CAN1 状态异常!"
    exit 1
fi
echo "  CAN1 正常 ✓"

# 4. 强制重启底盘节点
echo "[4/6] 启动 tracer_base_node..."
source /home/agilex/MobileManipulator/devel/setup.bash

# 强制杀掉已有节点，确保使用新的 CAN 连接
rosnode kill /tracer_base /tracer_base_node 2>/dev/null || true
pkill -9 -f tracer_base 2>/dev/null || true
sleep 1

rosrun tracer_base tracer_base_node _port_name:=can1 &
NODE_PID=$!
sleep 3

if ! rosnode list 2>/dev/null | grep -q tracer_base; then
    echo "  [错误] tracer_base_node 启动失败"
    exit 1
fi
echo "  tracer_base_node 已启动 ✓"

# 5. 安全告警
echo ""
echo "[5/6] ========================================="
echo "      ⚠️⚠️⚠️  安全警告 ⚠️⚠️⚠️"
echo "      ========================================="
echo ""
echo "      底盘即将前进 10cm (0.1米)"
echo "      速度: 0.1 m/s"
echo ""
echo "      请确保:"
echo "        1. 底盘前方无障碍物"
echo "        2. 底盘前方无人员"
echo "        3. 急停按钮可触及"
echo ""
echo "      ========================================="

sleep 1

# 6. 移动约 10cm
# 实测: 0.15m/s × 1.5s ≈ 10cm+
echo "[6/6] 执行前进 10cm..."

rostopic pub -r 50 /cmd_vel geometry_msgs/Twist "linear: {x: 0.15, y: 0.0, z: 0.0}
angular: {x: 0.0, y: 0.0, z: 0.0}" &
PUB_PID=$!

sleep 1.5

# 停止
kill $PUB_PID 2>/dev/null
wait $PUB_PID 2>/dev/null
rostopic pub -1 /cmd_vel geometry_msgs/Twist "linear: {x: 0.0}" >/dev/null 2>&1

echo "  移动完成 ✓"

# 清理
echo ""
echo "清理..."
kill $NODE_PID 2>/dev/null
wait $NODE_PID 2>/dev/null

echo ""
echo "========================================"
echo "   测试完成!"
echo "========================================"
echo ""
echo "   预期移动: 10cm (±2cm)"
echo ""
echo "   结果判定:"
echo "     ✓ 底盘移动约 10cm = CAN1 正常"
echo "     ✗ 底盘未移动 = 检查底盘电源/急停"
echo "     ✗ 移动距离偏差大 = 检查轮子/编码器"
echo ""
echo "========================================"
