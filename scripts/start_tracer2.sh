#!/bin/bash
#
# Tracer ROS2 Driver Start Script
# Usage: ./start_tracer2.sh [mini]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default parameters
CAN_PORT="can1"
IS_MINI="false"

# Parse arguments
if [ "$1" == "mini" ]; then
    IS_MINI="true"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Tracer ROS2 Driver Start Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "CAN Port: $CAN_PORT"
echo "Is Mini: $IS_MINI"
echo ""

# Source ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo -e "${RED}[ERROR]${NC} ROS2 Humble not found"
    exit 1
fi

# Source workspace
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
else
    echo -e "${RED}[ERROR]${NC} Workspace not built. Run build_tracer2.sh first"
    exit 1
fi

# Check CAN interface
if ! ip link show "$CAN_PORT" &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} $CAN_PORT not found, trying to bring up..."
    sudo modprobe gs_usb 2>/dev/null || true
    sudo ip link set "$CAN_PORT" up type can bitrate 500000
fi

CAN_STATE=$(ip link show "$CAN_PORT" 2>/dev/null | grep -o "state [A-Z]*" | awk '{print $2}')
if [ "$CAN_STATE" != "UP" ]; then
    echo -e "${YELLOW}[WARN]${NC} $CAN_PORT is $CAN_STATE, bringing up..."
    sudo ip link set "$CAN_PORT" up type can bitrate 500000
fi
echo -e "${GREEN}[OK]${NC} CAN interface $CAN_PORT is ready"

# Kill existing tracer nodes if any
rosnode_name="/tracer_base_node"
if ros2 node list 2>/dev/null | grep -q "$rosnode_name"; then
    echo -e "${YELLOW}[INFO]${NC} Killing existing tracer_base_node..."
    pkill -f "tracer_base_node" || true
    sleep 1
fi

# Launch
echo ""
echo -e "${GREEN}Launching tracer_base...${NC}"
echo ""

if [ "$IS_MINI" == "true" ]; then
    ros2 launch tracer_base tracer_mini_base.launch.py port_name:="$CAN_PORT"
else
    ros2 launch tracer_base tracer_base.launch.py port_name:="$CAN_PORT"
fi
