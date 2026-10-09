"""Shared perception ranges and bounded navigation budgets."""
LIDAR_TARGET_DEPTH_M = 6.0
DEPTH_CAMERA_TARGET_RANGE_M = 4.0
LIDAR_POINT_DEPTH_MARGIN_M = 1.0


def target_range_limit(observation):
    return (LIDAR_TARGET_DEPTH_M
            if (observation.plane_fit or {}).get('geometry_source') == 'lidar'
            else DEPTH_CAMERA_TARGET_RANGE_M)

# A target up to 6 m away needs roughly 4 m to the 2 m staging point.
MAX_STAGE_PATH_M = 4.5
MAX_TOTAL_PATH_M = 6.5
MAX_STAGE_TRAVEL_M = 5.5
MAX_STAGE_DISPLACEMENT_M = 4.5
MOTION_TIMEOUT_S = 210.
MOTION_PROCESS_TIMEOUT_S = 240.

# Final perception verification; controller tolerances are tighter than these.
FINAL_LATERAL_TOLERANCE_M = .05
FINAL_HEADING_TOLERANCE_DEG = 1.0
FINE_XY_GOAL_TOLERANCE_M = .015
FINE_YAW_GOAL_TOLERANCE_RAD = .008726646259971648  # 0.5 degrees
