#!/bin/bash
#
# Tracer ROS2 Driver Build Script
# Usage: ./build_tracer2.sh [clean]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Tracer ROS2 Driver Build Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Workspace: $WS_DIR"

# Source ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}[OK]${NC} ROS2 Humble sourced"
else
    echo -e "${RED}[ERROR]${NC} ROS2 Humble not found"
    exit 1
fi

# Check dependencies
echo ""
echo "Checking dependencies..."

if ! dpkg -s libasio-dev &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} libasio-dev not installed, installing..."
    sudo apt-get update && sudo apt-get install -y libasio-dev
fi

if ! dpkg -s can-utils &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} can-utils not installed, installing..."
    sudo apt-get update && sudo apt-get install -y can-utils
fi

echo -e "${GREEN}[OK]${NC} Dependencies satisfied"

cd "$WS_DIR"

# Clean build if requested
if [ "$1" == "clean" ]; then
    echo ""
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    rm -rf build/ugv_sdk build/tracer_msgs build/tracer_base
    rm -rf install/ugv_sdk install/tracer_msgs install/tracer_base
    echo -e "${GREEN}[OK]${NC} Clean completed"
fi

# Build packages in order
echo ""
echo -e "${GREEN}Building packages...${NC}"
echo ""

echo "Step 1/3: Building ugv_sdk..."
colcon build --packages-select ugv_sdk --cmake-args -DCMAKE_BUILD_TYPE=Release
echo -e "${GREEN}[OK]${NC} ugv_sdk built"

echo ""
echo "Step 2/3: Building tracer_msgs..."
source install/setup.bash
colcon build --packages-select tracer_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release
echo -e "${GREEN}[OK]${NC} tracer_msgs built"

echo ""
echo "Step 3/3: Building tracer_base..."
source install/setup.bash
colcon build --packages-select tracer_base --cmake-args -DCMAKE_BUILD_TYPE=Release
echo -e "${GREEN}[OK]${NC} tracer_base built"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To use the driver:"
echo "  source $WS_DIR/install/setup.bash"
echo "  ros2 launch tracer_base tracer_base.launch.py port_name:=can1"
echo ""
