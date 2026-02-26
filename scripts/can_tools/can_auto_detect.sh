#!/bin/bash
# ============================================================================
# CAN Auto-Detect Script
# ============================================================================
# 自动识别 CAN 接口连接的是 Tracer 底盘还是 Piper 机械臂
# 根据 CAN 协议特征自动配置正确的波特率和接口名称
#
# Usage: ./can_auto_detect.sh
#
# 识别原理：
#   - Tracer 底盘 (500kbps): 主动广播 0x211, 0x221, 0x241, 0x311
#   - Piper 机械臂 (1000kbps): 主动广播 0x2A1-0x2A8
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
CHASSIS_CAN_NAME="can1"
CHASSIS_BITRATE=500000
PIPER_CAN_NAME="can0"
PIPER_BITRATE=1000000
DETECT_TIMEOUT=1.0  # 秒

# Tracer 底盘特征 CAN ID (十六进制) - 仅使用独有 ID
# 注意: 251-254 也被 Piper 使用，不能作为 Tracer 特征
TRACER_IDS="211 221 241 311"

# Piper 机械臂特征 CAN ID (十六进制) - 关节反馈 0x2A1-0x2A8
PIPER_IDS="2A1 2A2 2A3 2A4 2A5 2A6 2A7 2A8"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   CAN 设备自动识别工具${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查依赖
check_dependencies() {
    local missing=0

    if ! command -v candump &> /dev/null; then
        echo -e "${RED}错误: 未安装 can-utils${NC}"
        echo "请运行: sudo apt install can-utils"
        missing=1
    fi

    if ! command -v ethtool &> /dev/null; then
        echo -e "${RED}错误: 未安装 ethtool${NC}"
        echo "请运行: sudo apt install ethtool"
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

# 获取所有 CAN 接口
get_can_interfaces() {
    ip -br link show type can 2>/dev/null | awk '{print $1}' || true
}

# 设置 CAN 接口波特率
setup_can_interface() {
    local iface=$1
    local bitrate=$2

    sudo ip link set "$iface" down 2>/dev/null || true
    sudo ip link set "$iface" type can bitrate "$bitrate" 2>/dev/null
    sudo ip link set "$iface" up 2>/dev/null
}

# 检测 CAN 接口上的设备类型
# 返回: "tracer", "piper", 或 "unknown"
detect_device_on_interface() {
    local iface=$1
    local bitrate=$2
    local timeout=$3

    # 设置接口
    setup_can_interface "$iface" "$bitrate"
    sleep 0.2

    # 捕获 CAN 消息
    local dump_file="/tmp/candump_${iface}_$$.txt"
    timeout "$timeout" candump "$iface" > "$dump_file" 2>/dev/null || true

    # 检查是否有数据
    if [ ! -s "$dump_file" ]; then
        rm -f "$dump_file"
        return 1
    fi

    # 提取 CAN ID（格式: can0 123 [8] ...）
    local can_ids=$(awk '{print $2}' "$dump_file" | sort -u)
    rm -f "$dump_file"

    # 检查 Tracer 特征
    for id in $TRACER_IDS; do
        if echo "$can_ids" | grep -qi "^${id}$"; then
            echo "tracer"
            return 0
        fi
    done

    # 检查 Piper 特征
    for id in $PIPER_IDS; do
        if echo "$can_ids" | grep -qi "^${id}$"; then
            echo "piper"
            return 0
        fi
    done

    echo "unknown"
    return 0
}

# 重命名 CAN 接口
rename_can_interface() {
    local old_name=$1
    local new_name=$2

    if [ "$old_name" == "$new_name" ]; then
        return 0
    fi

    # 如果目标名称已存在，先临时重命名
    if ip link show "$new_name" &>/dev/null; then
        local temp_name="can_temp_$$"
        sudo ip link set "$new_name" down 2>/dev/null || true
        sudo ip link set "$new_name" name "$temp_name" 2>/dev/null || true
    fi

    sudo ip link set "$old_name" down 2>/dev/null || true
    sudo ip link set "$old_name" name "$new_name" 2>/dev/null
}

# 主逻辑
main() {
    check_dependencies

    # 获取所有 CAN 接口
    local interfaces=$(get_can_interfaces)

    if [ -z "$interfaces" ]; then
        echo -e "${RED}错误: 未检测到任何 CAN 接口${NC}"
        echo "请检查 CAN 适配器是否连接"
        exit 1
    fi

    echo -e "${YELLOW}检测到 CAN 接口:${NC}"
    for iface in $interfaces; do
        local bus_info=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | awk '{print $2}')
        echo "  - $iface (USB: $bus_info)"
    done
    echo ""

    # 存储检测结果
    declare -A detected_devices
    declare -A detected_interfaces

    # 对每个接口进行检测
    for iface in $interfaces; do
        echo -e "${YELLOW}正在检测 $iface ...${NC}"

        # 先尝试底盘波特率 (500kbps)
        echo -n "  尝试 500kbps (底盘): "
        local result=$(detect_device_on_interface "$iface" $CHASSIS_BITRATE $DETECT_TIMEOUT)

        if [ "$result" == "tracer" ]; then
            echo -e "${GREEN}检测到 Tracer 底盘!${NC}"
            detected_devices["$iface"]="tracer"
            detected_interfaces["tracer"]="$iface"
            continue
        elif [ "$result" == "piper" ]; then
            # 不太可能，但处理一下
            echo -e "${GREEN}检测到 Piper (异常波特率)${NC}"
            detected_devices["$iface"]="piper"
            detected_interfaces["piper"]="$iface"
            continue
        else
            echo "无响应"
        fi

        # 尝试机械臂波特率 (1000kbps)
        echo -n "  尝试 1000kbps (机械臂): "
        result=$(detect_device_on_interface "$iface" $PIPER_BITRATE $DETECT_TIMEOUT)

        if [ "$result" == "piper" ]; then
            echo -e "${GREEN}检测到 Piper 机械臂!${NC}"
            detected_devices["$iface"]="piper"
            detected_interfaces["piper"]="$iface"
            continue
        elif [ "$result" == "tracer" ]; then
            echo -e "${GREEN}检测到 Tracer (异常波特率)${NC}"
            detected_devices["$iface"]="tracer"
            detected_interfaces["tracer"]="$iface"
            continue
        else
            echo "无响应"
        fi

        echo -e "  ${RED}未能识别设备${NC}"
        detected_devices["$iface"]="unknown"
    done

    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}   检测结果${NC}"
    echo -e "${BLUE}============================================${NC}"

    # 显示结果
    local chassis_iface="${detected_interfaces[tracer]:-}"
    local piper_iface="${detected_interfaces[piper]:-}"

    if [ -n "$chassis_iface" ]; then
        echo -e "${GREEN}Tracer 底盘: $chassis_iface -> $CHASSIS_CAN_NAME @ ${CHASSIS_BITRATE}bps${NC}"
    else
        echo -e "${RED}Tracer 底盘: 未检测到${NC}"
    fi

    if [ -n "$piper_iface" ]; then
        echo -e "${GREEN}Piper 机械臂: $piper_iface -> $PIPER_CAN_NAME @ ${PIPER_BITRATE}bps${NC}"
    else
        echo -e "${RED}Piper 机械臂: 未检测到${NC}"
    fi

    echo ""

    # 询问是否应用配置
    if [ -z "$chassis_iface" ] && [ -z "$piper_iface" ]; then
        echo -e "${RED}未检测到任何已知设备，请检查硬件连接${NC}"
        exit 1
    fi

    read -p "是否应用配置? [Y/n] " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo -e "${YELLOW}正在应用配置...${NC}"

        # 先关闭所有接口
        for iface in $interfaces; do
            sudo ip link set "$iface" down 2>/dev/null || true
        done

        # 配置底盘
        if [ -n "$chassis_iface" ]; then
            echo "  配置底盘: $chassis_iface -> $CHASSIS_CAN_NAME @ ${CHASSIS_BITRATE}bps"
            rename_can_interface "$chassis_iface" "$CHASSIS_CAN_NAME"
            setup_can_interface "$CHASSIS_CAN_NAME" $CHASSIS_BITRATE
        fi

        # 配置机械臂
        if [ -n "$piper_iface" ]; then
            # 如果底盘已占用目标名称，需要处理
            if [ "$chassis_iface" == "$PIPER_CAN_NAME" ]; then
                # 底盘原来叫 can0，现在改成 can1 了，piper 可以用 can0
                :
            fi
            echo "  配置机械臂: $piper_iface -> $PIPER_CAN_NAME @ ${PIPER_BITRATE}bps"
            # 如果 piper_iface 还是原名（没被底盘重命名覆盖）
            if ip link show "$piper_iface" &>/dev/null; then
                rename_can_interface "$piper_iface" "$PIPER_CAN_NAME"
            fi
            setup_can_interface "$PIPER_CAN_NAME" $PIPER_BITRATE
        fi

        echo ""
        echo -e "${GREEN}配置完成!${NC}"
        echo ""
        echo "当前 CAN 接口状态:"
        ip -br link show type can
    else
        echo "已取消"
    fi
}

main "$@"
