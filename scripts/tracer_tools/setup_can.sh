#!/bin/bash
# CAN 接口初始化脚本
# 默认使用 can1

CAN_IF=${1:-can1}
BITRATE=${2:-500000}

echo "配置 CAN 接口: $CAN_IF, 波特率: $BITRATE"

# 关闭接口（如果已启动）
sudo ip link set $CAN_IF down 2>/dev/null

# 配置 CAN 接口
sudo ip link set $CAN_IF type can bitrate $BITRATE

# 启动接口
sudo ip link set $CAN_IF up

# 检查状态
if ip link show $CAN_IF | grep -q "UP"; then
    echo "[OK] $CAN_IF 已成功启动"
    ip link show $CAN_IF
else
    echo "[FAIL] $CAN_IF 启动失败"
    exit 1
fi
