#!/bin/bash
# ros_renice.sh — Adjust ROS2 process priorities for localization-first scheduling.
#
# Requires: /etc/sudoers.d/ros-priority (NOPASSWD for nice/renice)
# Usage: ./scripts/ros_renice.sh          (after launch is up)
#        make renice                       (Makefile target)
#
# Priority tiers:
#   nice -15  Critical localization (hdl_localization, fastlio_mapping)
#   nice -10  Localization support  (odom_fusion, odom_frame_converter)
#   nice   0  Normal               (perception, camera, nav2)
#   nice   5  Low                  (cleaner_manager, manual_control)
#   nice  15  Lowest               (rviz2)

set -euo pipefail

renice_by_name() {
    local nice_val="$1"
    local pattern="$2"
    local label="$3"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "  [ ] $label — not running"
        return
    fi
    for pid in $pids; do
        sudo -n renice "$nice_val" -p "$pid" >/dev/null 2>&1 && \
            echo "  [OK] $label (PID $pid) → nice $nice_val" || \
            echo "  [FAIL] $label (PID $pid)"
    done
}

echo "=== ROS2 Process Priority Adjustment ==="
echo ""
echo "--- Critical (nice -15) ---"
renice_by_name -15 "hdl_localization_node" "hdl_localization"
renice_by_name -15 "fastlio_mapping"       "fast_lio"

echo ""
echo "--- High (nice -10) ---"
renice_by_name -10 "tf_republisher"        "tf_republisher"
renice_by_name -10 "fastlio_odom_to_tf"    "fastlio_odom_to_tf"
renice_by_name -10 "delayed_cloud_relay"   "delayed_cloud_relay"
renice_by_name -10 "odom_fusion_sync"      "odom_fusion"
renice_by_name -10 "odom_frame_converter"  "odom_frame_converter"
renice_by_name -10 "controller_server"     "nav2_controller"
renice_by_name -10 "planner_server"        "nav2_planner"

echo ""
echo "--- Low (nice 5) ---"
renice_by_name  5  "cleaner_manager_node"  "cleaner_manager"
renice_by_name  5  "manual_control_node"   "manual_control"

echo ""
echo "--- Lowest (nice 15) ---"
renice_by_name 15  "rviz2"                 "rviz2"

echo ""
echo "=== Done ==="
