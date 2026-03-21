# -*- coding: utf-8 -*-
"""
Camera-LiDAR Calibration Library

Pure Python implementation based on acfr/cam_lidar_calibration algorithm.
No ROS dependency required.
"""

# Modular imports
from .io_utils import (
    load_extrinsics,
    load_intrinsics,
    load_pcd,
    save_extrinsics,
)

from .transforms import (
    invert_tf_extrinsics,
    tf_to_point_transform,
    point_transform_to_tf,
    transform_link_to_optical,
    project_camera_center_to_lidar,
    transform_points_lidar_to_camera,
    transform_points_camera_to_lidar,
    get_camera_optical_to_link_transform,
    convert_optical_to_link_extrinsics,
)

from .geometry import (
    fit_rectangle_known_size,
    get_board_corners_3d,
    compute_lid_center_from_camera,
    validate_projected_rectangle,
    project_points,
    calculate_reprojection_error,
    draw_projection,
)

from .plane_fitting import (
    fit_ground_plane,
    fit_plane_ransac,
)

from .quality import (
    evaluate_quality,
    print_quality_summary_table,
    print_frame_quality,
    evaluate_calibration_result,
)

from .chessboard import (
    detect_chessboard,
    draw_chessboard,
)

from .calibration_solver import (
    solve_rotation_kabsch,
    solve_translation_plane_constraint,
    check_normal_diversity,
    optimize_extrinsics_robust,
    optimize_extrinsics_v2,
)

__all__ = [
    # IO
    'load_extrinsics',
    'load_intrinsics',
    'load_pcd',
    'save_extrinsics',
    # Transforms
    'invert_tf_extrinsics',
    'tf_to_point_transform',
    'point_transform_to_tf',
    'transform_link_to_optical',
    'project_camera_center_to_lidar',
    'transform_points_lidar_to_camera',
    'transform_points_camera_to_lidar',
    'get_camera_optical_to_link_transform',
    'convert_optical_to_link_extrinsics',
    # Geometry
    'fit_rectangle_known_size',
    'get_board_corners_3d',
    'compute_lid_center_from_camera',
    'validate_projected_rectangle',
    'project_points',
    'calculate_reprojection_error',
    'draw_projection',
    # Plane fitting
    'fit_ground_plane',
    'fit_plane_ransac',
    # Quality
    'evaluate_quality',
    'print_quality_summary_table',
    'print_frame_quality',
    'evaluate_calibration_result',
    # Chessboard
    'detect_chessboard',
    'draw_chessboard',
    # Calibration solver
    'solve_rotation_kabsch',
    'solve_translation_plane_constraint',
    'check_normal_diversity',
    'optimize_extrinsics_robust',
    'optimize_extrinsics_v2',
]
