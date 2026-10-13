# Vendored m-explore-ros2 packages

Source: https://github.com/robo-friends/m-explore-ros2
Commit: 326cf8a0b487c34246bb8f3326afbcd69576dc60

Only explore/ and explore_lite_msgs/ are vendored here.
Root upstream license is retained as LICENSE. No map_merge package is used.

Local diagnostic patch (2026-10-12): explore publishes its exact NavigateToPose
target and action state on `/explore/current_goal` as latched, periodically
refreshed markers. Exploration and return goals are distinguished. This does not
change frontier selection or motion control. See auto_explore README for recording
and replay support.

Local frontier/navigation patch (2026-10-12): search traverses known cost cells
below INSCRIBED_INFLATED_OBSTACLE, including inflation costs; blocked/unknown
start cells do not jump to unrelated free islands. Goals use reachable known
neighbors of frontier contours instead of potentially unknown centroids. Initial
frontier cells participate in point/count/centroid calculations. Nav2 footprint
and trajectory rejection are unchanged. Active goals are retained during boundary
jitter. Rejected/aborted/no-progress goals are blacklisted and trigger a deferred
100 ms replan; generation checks ignore obsolete action callbacks.

Local restart fix (2026-10-12): the full costmap subscription uses reliable,
transient-local QoS (depth 1), matching Nav2's latched full-grid publisher.
A restarted explorer previously waited indefinitely because its volatile
subscription missed the full grid and listened only for a new publication.
Incremental update subscriptions are unchanged. The initial map wait now also
checks rclcpp::ok() so shutdown is not ignored while waiting for a map.
