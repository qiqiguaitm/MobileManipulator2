#!/usr/bin/env bash
set -euo pipefail
viewer_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
viewer_workspace="$(cd "$viewer_dir/../.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
source "$viewer_workspace/install/local_setup.bash"
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [[ -f "$viewer_workspace/log/rviz_cyclonedds.xml" ]]; then
  export CYCLONEDDS_URI="file://$viewer_workspace/log/rviz_cyclonedds.xml"
fi
exec python3 "$viewer_dir/bag_viewer.py" "$@"
