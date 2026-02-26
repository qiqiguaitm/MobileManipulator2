#!/bin/bash
# 交换两个 CAN 接口的名称
# 用法: ./can_name_swap.sh can0 can1

set -e

if [ $# -ne 2 ]; then
    echo "用法: $0 <接口1> <接口2>"
    echo "示例: $0 can0 can1"
    exit 1
fi

CAN_A="$1"
CAN_B="$2"
TMP_NAME="can_tmp"

# 检查接口是否存在
if ! ip link show "$CAN_A" &>/dev/null; then
    echo "错误: 接口 $CAN_A 不存在"
    exit 1
fi

if ! ip link show "$CAN_B" &>/dev/null; then
    echo "错误: 接口 $CAN_B 不存在"
    exit 1
fi

echo "交换前状态:"
ip -br link show dev "$CAN_A"
ip -br link show dev "$CAN_B"

# 关闭接口 -> 重命名 -> 启动
sudo ip link set "$CAN_A" down
sudo ip link set "$CAN_B" down

sudo ip link set "$CAN_A" name "$TMP_NAME"
sudo ip link set "$CAN_B" name "$CAN_A"
sudo ip link set "$TMP_NAME" name "$CAN_B"

sudo ip link set "$CAN_A" up
sudo ip link set "$CAN_B" up

echo "交换后状态:"
ip -br link show dev "$CAN_A"
ip -br link show dev "$CAN_B"
echo "完成"
