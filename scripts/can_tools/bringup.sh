#!/bin/bash
# ============================================================================
# Bringup Script - CAN 设备初始化
# ============================================================================
# 支持两种模式：
#   1. 自动检测模式: ./bringup.sh auto
#   2. 手动指定模式（默认）: ./bringup.sh 或 ./bringup.sh manual
#
# 手动模式通过 USB 地址识别设备（需确认硬件连接）
# ============================================================================

# 预先缓存 sudo 凭据
echo "didi" | sudo -S true 2>/dev/null

MODE=${1:-manual}

cd "$(dirname "$0")"

echo "========================================"
echo "CAN 设备初始化"
echo "========================================"

# 显示当前 CAN 接口状态
bash ./find_all_can_port.sh

echo ""

# ---------------------------------------------------------------------------
# 清理函数：删除所有临时接口名，确保干净的起点
# ---------------------------------------------------------------------------
cleanup_can_interfaces() {
    echo "清理残留的 CAN 接口..."

    # 关闭并删除所有 _t 结尾的临时接口
    for iface in $(ip -br link show type can | awk '{print $1}' | grep '_t$'); do
        sudo ip link set "$iface" down 2>/dev/null
        sudo ip link delete "$iface" 2>/dev/null || true
    done

    # 关闭现有的 can0 和 can1
    sudo ip link set can0 down 2>/dev/null || true
    sudo ip link set can1 down 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 根据 USB 地址配置 CAN 接口
# 参数: $1=目标名称 $2=比特率 $3=USB地址
# ---------------------------------------------------------------------------
configure_can_by_usb() {
    local TARGET_NAME="$1"
    local BITRATE="$2"
    local USB_ADDR="$3"

    echo "配置 $TARGET_NAME (${BITRATE}bps) @ USB $USB_ADDR"

    # 查找 USB 地址对应的接口
    local FOUND_IFACE=""
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        local bus=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | awk '{print $2}')
        if [ "$bus" == "$USB_ADDR" ]; then
            FOUND_IFACE="$iface"
            break
        fi
    done

    if [ -z "$FOUND_IFACE" ]; then
        echo "  [错误] 未找到 USB 地址 $USB_ADDR 对应的接口"
        return 1
    fi

    echo "  找到接口: $FOUND_IFACE"

    # 如果接口名已经是目标名，直接配置
    if [ "$FOUND_IFACE" == "$TARGET_NAME" ]; then
        sudo ip link set "$TARGET_NAME" down
        sudo ip link set "$TARGET_NAME" type can bitrate "$BITRATE"
        sudo ip link set "$TARGET_NAME" up
        echo "  [完成] $TARGET_NAME 已配置"
        return 0
    fi

    # 如果目标名被占用，先把它改成临时名
    if ip link show "$TARGET_NAME" &>/dev/null; then
        local TEMP_NAME="${TARGET_NAME}_tmp_$$"
        sudo ip link set "$TARGET_NAME" down
        sudo ip link set "$TARGET_NAME" name "$TEMP_NAME"
        echo "  将占用的 $TARGET_NAME 临时改名为 $TEMP_NAME"
    fi

    # 配置找到的接口
    sudo ip link set "$FOUND_IFACE" down
    sudo ip link set "$FOUND_IFACE" type can bitrate "$BITRATE"
    sudo ip link set "$FOUND_IFACE" name "$TARGET_NAME"
    sudo ip link set "$TARGET_NAME" up

    if [ $? -eq 0 ]; then
        echo "  [完成] $TARGET_NAME 已配置并启动"
        return 0
    else
        echo "  [错误] 配置失败"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

if [ "$MODE" == "auto" ]; then
    echo "模式: 自动检测"
    echo "----------------------------------------"
    bash ./can_auto_detect.sh

elif [ "$MODE" == "manual" ]; then
    echo "模式: 手动指定 (硬编码 USB 地址)"
    echo "----------------------------------------"
    echo ""
    echo "USB 地址映射 (硬编码):"
    echo "  1-4.2:1.0 -> Piper 机械臂 (can0, 1000kbps)"
    echo "  1-4.1:1.0 -> Tracer 底盘 (can1, 500kbps)"
    echo ""

    # 清理残留接口
    cleanup_can_interfaces

    # 配置 Piper 机械臂 (can0)
    configure_can_by_usb "can0" "1000000" "1-4.2:1.0"
    PIPER_OK=$?

    # 配置 Tracer 底盘 (can1)
    configure_can_by_usb "can1" "500000" "1-4.1:1.0"
    CHASSIS_OK=$?

    # 清理临时接口
    for iface in $(ip -br link show type can | awk '{print $1}' | grep '_tmp_'); do
        sudo ip link delete "$iface" 2>/dev/null || true
    done

else
    echo "用法: ./bringup.sh [auto|manual]"
    echo "  auto   - 自动检测 CAN 设备类型"
    echo "  manual - 使用硬编码 USB 地址（默认）"
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

# 验证
echo ""
echo "验证连接:"
if ip link show can0 2>/dev/null | grep -q "UP"; then
    echo "  can0: ✓ UP"
else
    echo "  can0: ✗ DOWN - 请检查机械臂电源和连接"
fi

if ip link show can1 2>/dev/null | grep -q "UP"; then
    echo "  can1: ✓ UP"
else
    echo "  can1: ✗ DOWN - 请检查底盘电源和连接"
fi
