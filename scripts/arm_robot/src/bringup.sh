#!/bin/bash
# ============================================================================
# Bringup Script - CAN 设备初始化
# ============================================================================
# 支持两种模式：
#   1. 自动检测模式（默认）: ./bringup.sh 或 ./bringup.sh auto
#   2. 手动指定模式:        ./bringup.sh manual
#
# 自动检测会通过 CAN 协议特征识别底盘和机械臂
# ============================================================================

MODE=${1:-auto}

cd "$(dirname "$0")"

echo "========================================"
echo "CAN 设备初始化"
echo "========================================"

# 显示当前 CAN 接口状态
bash ./find_all_can_port.sh

echo ""

if [ "$MODE" == "auto" ]; then
    echo "模式: 自动检测"
    echo "----------------------------------------"
    bash ./can_auto_detect.sh
elif [ "$MODE" == "manual" ]; then
    echo "模式: 手动指定 (硬编码 USB 地址)"
    echo "----------------------------------------"
    # 备用的硬编码方式，当自动检测失败时使用
    sudo ip link set can0 down
    sudo ip link set can1 down
    sudo ip link set can0 name can0_t
    sudo ip link set can1 name can1_t
    bash ./can_activate.sh can0 1000000 1-4.1.4:1.0  # for piper
    bash ./can_activate.sh can1 500000  1-4.1.2:1.0  # for chassis
else
    echo "用法: ./bringup.sh [auto|manual]"
    echo "  auto   - 自动检测 CAN 设备类型（默认）"
    echo "  manual - 使用硬编码 USB 地址"
    exit 1
fi

echo ""
echo "========================================"
echo "最终 CAN 接口状态:"
echo "========================================"
ip -br link show type can
echo ""
echo "  can0 (1000kbps) -> Piper 机械臂"
echo "  can1 (500kbps)  -> Tracer 底盘"
echo "========================================"
