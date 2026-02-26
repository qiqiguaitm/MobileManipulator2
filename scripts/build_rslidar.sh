#!/bin/bash
#
# RSLidar Driver Build Script for ROS2 Humble
# Usage: ./build_rslidar.sh [clean]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RSLidar Driver Build Script (ROS2)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Workspace: $WS_DIR"

# Source ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}[OK]${NC} ROS2 Humble sourced"
else
    echo -e "${RED}[ERROR]${NC} ROS2 Humble not found at /opt/ros/humble"
    exit 1
fi

# Check dependencies
echo ""
echo "Checking dependencies..."

if ! dpkg -s libpcap-dev &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} libpcap-dev not installed, installing..."
    sudo apt-get update && sudo apt-get install -y libpcap-dev
fi

if ! dpkg -s libyaml-cpp-dev &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} libyaml-cpp-dev not installed, installing..."
    sudo apt-get update && sudo apt-get install -y libyaml-cpp-dev
fi

echo -e "${GREEN}[OK]${NC} Dependencies satisfied"

# Navigate to workspace
cd "$WS_DIR"

# Ensure rslidar_sdk symlink exists (needed for colcon to find nested package)
if [ ! -L "$WS_DIR/src/rslidar_sdk" ]; then
    echo ""
    echo "Creating rslidar_sdk symlink..."
    ln -sf robot_drivers/lidar_driver/rslidar_sdk "$WS_DIR/src/rslidar_sdk"
    echo -e "${GREEN}[OK]${NC} Symlink created"
fi

# Clean build if requested
if [ "$1" == "clean" ]; then
    echo ""
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    rm -rf build/rslidar_msg build/rslidar_sdk build/lidar_driver
    rm -rf install/rslidar_msg install/rslidar_sdk install/lidar_driver
    rm -rf log/
    echo -e "${GREEN}[OK]${NC} Clean completed"
fi

# Build packages in order
echo ""
echo -e "${GREEN}Building packages...${NC}"
echo ""

echo "Step 1/3: Building rslidar_msg..."
colcon build --packages-select rslidar_msg --cmake-args -DCMAKE_BUILD_TYPE=Release
echo -e "${GREEN}[OK]${NC} rslidar_msg built"

echo ""
echo "Step 2/3: Building rslidar_sdk..."
source install/setup.bash
colcon build --packages-select rslidar_sdk --cmake-args -DCMAKE_BUILD_TYPE=Release
echo -e "${GREEN}[OK]${NC} rslidar_sdk built"

echo ""
echo "Step 3/3: Building lidar_driver..."
source install/setup.bash
colcon build --packages-select lidar_driver --cmake-args -DCMAKE_BUILD_TYPE=Release
echo -e "${GREEN}[OK]${NC} lidar_driver built"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To use the driver:"
echo "  source $WS_DIR/install/setup.bash"
echo "  ros2 launch lidar_driver start_lidar.launch.py"
echo ""
