#!/bin/bash
# ============================================================
# CAN 总线优化配置脚本
# 解决 AsyncCAN bad_weak_ptr 问题
# ============================================================

CAN_IF="${1:-can1}"
BITRATE="${2:-500000}"

echo "配置 CAN 接口: $CAN_IF"

# 关闭 CAN 接口
sudo ip link set $CAN_IF down 2>/dev/null

# 配置 CAN 参数
# - bitrate: 500kbps (Tracer 标准)
# - restart-ms: 100ms (发生错误后自动重启)
# - txqueuelen: 100 (增加发送队列)
sudo ip link set $CAN_IF type can bitrate $BITRATE restart-ms 100

# 增加接收队列长度 (默认10，增加到100)
sudo ip link set $CAN_IF txqueuelen 100

# 启动 CAN 接口
sudo ip link set $CAN_IF up

# 验证配置
echo ""
echo "=== CAN 配置验证 ==="
ip -details link show $CAN_IF | grep -E "(state|bitrate|restart-ms|qlen)"

echo ""
echo "✅ CAN 接口 $CAN_IF 配置完成"
