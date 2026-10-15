#!/usr/bin/env bash
# Run on robot02. Starts capture dependencies, waits for localization, then records.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec /usr/bin/python3 "$ROOT/scripts/navigation_scenarios/capture.py" --help
fi
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin
source /opt/ros/humble/setup.bash
source "$ROOT/install/local_setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-144}" ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$ROOT/log/rviz_cyclonedds.xml"
test -f "$ROOT/log/rviz_cyclonedds.xml" || { echo "Missing CycloneDDS configuration"; exit 2; }
cd "$ROOT"
exec /usr/bin/python3 scripts/navigation_scenarios/capture.py "$@"
