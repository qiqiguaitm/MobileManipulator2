#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$root/install/local_setup.bash"
export MM2_PROJECT_ROOT="$root" ROS_DOMAIN_ID="${BOARD_ROS_DOMAIN_ID:-144}" ROS_LOCALHOST_ONLY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /usr/bin/python3 -m auto_explore.supervisor "$@"
