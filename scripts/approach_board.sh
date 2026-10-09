#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$project_root/install/local_setup.bash"
export ROS_DOMAIN_ID="${BOARD_ROS_DOMAIN_ID:-144}"
export ROS_LOCALHOST_ONLY="${BOARD_ROS_LOCALHOST_ONLY:-1}"
export MM2_PROJECT_ROOT="$project_root"
# Small geometry matrices should not fan out across Jetson cores and starve ROS.
# Match the bounded numerical-thread settings used by the validation runs.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Optional per-application proxy; keeps ROS/LAN networking unchanged.
proxy_config="$HOME/.config/robot-object-approach/proxy.env"
if [[ -f "$proxy_config" ]]; then
  source "$proxy_config"
fi
for argument in "$@"; do
  if [[ "$argument" == "--execute" ]]; then
    echo '实际行走模式：请留在机器人旁，保持急停可用。'
    break
  fi
done
exec /usr/bin/python3 -m board_approach.runner "$@"
