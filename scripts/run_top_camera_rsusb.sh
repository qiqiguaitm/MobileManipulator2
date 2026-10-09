#!/usr/bin/env bash
# D455 on Jetson: use the isolated RSUSB SDK rather than the system V4L2 backend.
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_prefix="${REALSENSE_RSUSB_PREFIX:-$HOME/.local/opt/realsense-rsusb-2.56.4/install}"
if [[ ! -f "$sdk_prefix/lib/librealsense2.so.2.56" ]]; then
  echo "Missing RSUSB SDK: $sdk_prefix/lib/librealsense2.so.2.56" >&2
  exit 1
fi
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$project_root/install/local_setup.bash"
export LD_LIBRARY_PATH="$sdk_prefix/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-144}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
exec ros2 launch camera_driver camera_driver.launch.py \
  top_enable:=true hand_enable:=false chassis_enable:=false \
  top_depth_profile:=1280,720,15 top_color_profile:=1280,720,15 \
  align_depth_enable:=true enable_camera_tf:=false "$@"
