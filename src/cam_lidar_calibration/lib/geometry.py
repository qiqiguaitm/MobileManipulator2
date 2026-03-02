# -*- coding: utf-8 -*-
"""
Geometric computation utilities for camera-LiDAR calibration.
"""

import cv2
import numpy as np
from scipy.optimize import minimize


def fit_rectangle_known_size(coords_2d, board_width, board_height, verbose=True):
    """Fit a known-size rectangle to 2D point cloud.

    Optimizes rectangle position and rotation to best cover the point cloud.

    Args:
        coords_2d: 2D coordinates (N, 2), columns are (h, v)
        board_width: rectangle width (h direction)
        board_height: rectangle height (v direction)
        verbose: whether to print debug info

    Returns:
        center_h: rectangle center h coordinate
        center_v: rectangle center v coordinate
        theta: rotation angle (radians)
        success: whether fitting succeeded
    """
    if len(coords_2d) < 10:
        if verbose:
            print(f"  [矩形拟合] 点数太少({len(coords_2d)})，跳过")
        return None, None, None, False

    # Initial estimate from point cloud centroid
    init_cx = np.mean(coords_2d[:, 0])
    init_cy = np.mean(coords_2d[:, 1])
    init_theta = 0.0

    half_w = board_width / 2
    half_h = board_height / 2

    def point_to_rect_distance(px, py, cx, cy, theta, half_w, half_h):
        """Compute signed distance from point to rectangle."""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = px - cx
        dy = py - cy
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy

        dist_x = abs(lx) - half_w
        dist_y = abs(ly) - half_h

        if dist_x <= 0 and dist_y <= 0:
            return max(dist_x, dist_y)
        elif dist_x > 0 and dist_y <= 0:
            return dist_x
        elif dist_x <= 0 and dist_y > 0:
            return dist_y
        else:
            return np.sqrt(dist_x**2 + dist_y**2)

    def cost_function(params):
        cx, cy, theta = params
        total_cost = 0.0
        outside_count = 0

        for pt in coords_2d:
            dist = point_to_rect_distance(pt[0], pt[1], cx, cy, theta, half_w, half_h)
            if dist > 0:
                total_cost += dist ** 2 * 10
                outside_count += 1
            else:
                total_cost += dist ** 2 * 0.1

        outside_ratio = outside_count / len(coords_2d)
        if outside_ratio > 0.1:
            total_cost += outside_ratio * 100

        return total_cost

    # Optimize
    x0 = [init_cx, init_cy, init_theta]
    bounds = [
        (init_cx - board_width, init_cx + board_width),
        (init_cy - board_height, init_cy + board_height),
        (-np.pi/6, np.pi/6)
    ]

    result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds)

    cx, cy, theta = result.x

    # Validate
    inside_count = 0
    outside_dists = []
    for pt in coords_2d:
        dist = point_to_rect_distance(pt[0], pt[1], cx, cy, theta, half_w, half_h)
        if dist <= 0:
            inside_count += 1
        else:
            outside_dists.append(dist)

    inside_ratio = inside_count / len(coords_2d)

    if verbose:
        print(f"  [矩形拟合] 中心: ({cx:.3f}, {cy:.3f})m, 旋转: {np.degrees(theta):.1f}°")
        print(f"  [矩形拟合] 矩形内点: {inside_count}/{len(coords_2d)} ({inside_ratio*100:.1f}%)")
        if outside_dists:
            print(f"  [矩形拟合] 外部点最大距离: {max(outside_dists)*1000:.1f}mm")

    if inside_ratio < 0.7:
        if verbose:
            print(f"  [矩形拟合] 内点比例过低，拟合可能不可靠")
        return cx, cy, theta, False

    return cx, cy, theta, True


def get_board_corners_3d(rvec, tvec, pattern_size, square_size, board_size):
    """Compute calibration board corner positions in camera frame.

    Args:
        rvec: rotation vector from PnP
        tvec: translation vector from PnP
        pattern_size: inner corner count (cols, rows)
        square_size: square edge length (meters)
        board_size: physical board size (width, height) (meters)

    Returns:
        board_corners_cam: 4 boundary corners in camera frame (4, 3)
                          order: top-left, top-right, bottom-right, bottom-left
    """
    cols, rows = pattern_size
    board_w, board_h = board_size

    pattern_w = (cols - 1) * square_size
    pattern_h = (rows - 1) * square_size

    margin_x = (board_w - pattern_w) / 2
    margin_y = (board_h - pattern_h) / 2

    board_corners_obj = np.array([
        [-pattern_w/2 - margin_x, -pattern_h/2 - margin_y, 0],
        [ pattern_w/2 + margin_x, -pattern_h/2 - margin_y, 0],
        [ pattern_w/2 + margin_x,  pattern_h/2 + margin_y, 0],
        [-pattern_w/2 - margin_x,  pattern_h/2 + margin_y, 0],
    ], dtype=np.float64)

    R_board, _ = cv2.Rodrigues(rvec)
    board_corners_cam = (R_board @ board_corners_obj.T).T + tvec.flatten()

    return board_corners_cam


def compute_lid_center_from_camera(cam_corners_3d, cam_center, R_c2l, t_c2l,
                                    lid_plane_normal, lid_plane_point):
    """Compute LiDAR center from camera corner projections (Method C).

    1. Project camera corners to LiDAR frame
    2. Constrain projected points to LiDAR-fitted plane
    3. Compute center from constrained corners

    Args:
        cam_corners_3d: 4 boundary corners in camera frame (4, 3)
        cam_center: board center in camera frame (3,)
        R_c2l: rotation matrix (point transform form)
        t_c2l: translation vector
        lid_plane_normal: LiDAR plane normal (3,)
        lid_plane_point: a point on LiDAR plane (3,)

    Returns:
        lid_center: board center in LiDAR frame (3,)
        corners_on_plane: 4 corners projected to plane (4, 3)
        projection_errors: distance from each corner to plane (4,)
    """
    R_inv = R_c2l.T
    t_inv = -R_c2l.T @ t_c2l

    corners_lid = []
    for corner_cam in cam_corners_3d:
        corner_lid = R_inv @ corner_cam + t_inv
        corners_lid.append(corner_lid)
    corners_lid = np.array(corners_lid)

    corners_on_plane = []
    projection_errors = []
    for corner in corners_lid:
        d = np.dot(corner - lid_plane_point, lid_plane_normal)
        corner_proj = corner - d * lid_plane_normal
        corners_on_plane.append(corner_proj)
        projection_errors.append(abs(d))
    corners_on_plane = np.array(corners_on_plane)
    projection_errors = np.array(projection_errors)

    lid_center = np.mean(corners_on_plane, axis=0)

    return lid_center, corners_on_plane, projection_errors


def validate_projected_rectangle(corners_on_plane, board_size, tolerance=0.15):
    """Validate that projected corners form a reasonable rectangle.

    Args:
        corners_on_plane: 4 corners on plane (4, 3)
                         order: top-left, top-right, bottom-right, bottom-left
        board_size: physical board size (width, height) (meters)
        tolerance: allowed relative error

    Returns:
        is_valid: whether validation passed
        metrics: validation metrics dict
    """
    board_w, board_h = board_size

    edge_top = np.linalg.norm(corners_on_plane[1] - corners_on_plane[0])
    edge_bottom = np.linalg.norm(corners_on_plane[2] - corners_on_plane[3])
    edge_left = np.linalg.norm(corners_on_plane[3] - corners_on_plane[0])
    edge_right = np.linalg.norm(corners_on_plane[2] - corners_on_plane[1])

    diag1 = np.linalg.norm(corners_on_plane[2] - corners_on_plane[0])
    diag2 = np.linalg.norm(corners_on_plane[3] - corners_on_plane[1])
    expected_diag = np.sqrt(board_w**2 + board_h**2)

    width_avg = (edge_top + edge_bottom) / 2
    height_avg = (edge_left + edge_right) / 2
    width_error = abs(width_avg - board_w) / board_w
    height_error = abs(height_avg - board_h) / board_h
    diag_error = abs((diag1 + diag2) / 2 - expected_diag) / expected_diag

    parallel_h = abs(edge_top - edge_bottom) / max(edge_top, edge_bottom)
    parallel_v = abs(edge_left - edge_right) / max(edge_left, edge_right)

    metrics = {
        'width_avg': width_avg,
        'height_avg': height_avg,
        'width_error': width_error,
        'height_error': height_error,
        'diag_error': diag_error,
        'parallel_h': parallel_h,
        'parallel_v': parallel_v,
        'expected_width': board_w,
        'expected_height': board_h
    }

    is_valid = (width_error < tolerance and
                height_error < tolerance and
                diag_error < tolerance and
                parallel_h < tolerance and
                parallel_v < tolerance)

    return is_valid, metrics


def project_points(points_3d, R, t, K, D):
    """Project 3D points to image plane.

    Args:
        points_3d: 3D points in LiDAR frame (N, 3)
        R: rotation matrix (point transform form)
        t: translation vector
        K: camera intrinsic matrix
        D: distortion coefficients

    Returns:
        points_2d: projected 2D points
        points_cam: points in camera frame
        points_3d_valid: valid input points
    """
    valid_mask = np.isfinite(points_3d).all(axis=1)
    if not valid_mask.any():
        return np.array([]), np.array([]), np.array([])

    points_3d = points_3d[valid_mask]

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        points_cam = (R @ points_3d.T).T + t

    valid_mask2 = np.isfinite(points_cam).all(axis=1)
    if not valid_mask2.any():
        return np.array([]), np.array([]), np.array([])

    points_cam = points_cam[valid_mask2]
    points_3d = points_3d[valid_mask2]

    valid = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid]
    points_3d_valid = points_3d[valid]

    if len(points_cam) == 0:
        return np.array([]), np.array([]), np.array([])

    points_2d, _ = cv2.projectPoints(
        points_cam, np.zeros(3), np.zeros(3), K, D
    )
    points_2d = points_2d.reshape(-1, 2)

    return points_2d, points_cam, points_3d_valid


def calculate_reprojection_error(objp, corners_2d, rvec, tvec, K, D):
    """Calculate reprojection error for detected corners.

    Args:
        objp: object points (N, 3)
        corners_2d: detected corners (N, 2)
        rvec: rotation vector
        tvec: translation vector
        K: camera matrix
        D: distortion coefficients

    Returns:
        rms: RMS error
        mean: mean error
        max_err: max error
        errors: per-point errors
    """
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)

    errors = np.linalg.norm(projected - corners_2d, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    mean = np.mean(errors)
    max_err = np.max(errors)

    return rms, mean, max_err, errors


def draw_projection(image, points_2d, points_cam):
    """在图像上绘制投影点"""
    vis = image.copy()
    h, w = vis.shape[:2]

    for i, (pt2d, pt_cam) in enumerate(zip(points_2d, points_cam)):
        x, y = int(pt2d[0]), int(pt2d[1])
        if 0 <= x < w and 0 <= y < h:
            # 根据深度着色
            depth = pt_cam[2]
            color_val = int(255 * min(depth / 5.0, 1.0))
            color = (255 - color_val, color_val, 0)  # 近蓝远绿
            cv2.circle(vis, (x, y), 2, color, -1)

    return vis
