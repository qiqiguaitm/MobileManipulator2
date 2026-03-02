# -*- coding: utf-8 -*-
"""
Chessboard detection utilities for camera-LiDAR calibration.
"""

import cv2
import numpy as np


def detect_chessboard(image, pattern_size=(7, 6), square_size=0.072):
    """Detect chessboard corners with multiple preprocessing strategies.

    Supports complex lighting conditions like backlight.

    Args:
        image: input BGR image
        pattern_size: inner corner count (cols, rows)
        square_size: square edge length in meters

    Returns:
        corners: detected corners (N, 2) or None
        objp: 3D object points (N, 3) or None
        pattern_size: pattern size tuple or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Multiple preprocessing strategies
    preprocess_strategies = [
        ("原图", gray),
        ("CLAHE", None),
        ("CLAHE强化", None),
        ("直方图均衡", cv2.equalizeHist(gray)),
    ]

    flags_fast = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                  cv2.CALIB_CB_NORMALIZE_IMAGE |
                  cv2.CALIB_CB_FAST_CHECK)

    flags_full = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                  cv2.CALIB_CB_NORMALIZE_IMAGE)

    corners = None
    best_gray = gray

    for strategy_name, processed in preprocess_strategies:
        if strategy_name == "CLAHE" and processed is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
        elif strategy_name == "CLAHE强化" and processed is None:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            processed = clahe.apply(gray)

        if processed is None:
            continue

        ret, corners = cv2.findChessboardCorners(processed, pattern_size, flags_fast)

        if not ret:
            ret, corners = cv2.findChessboardCorners(processed, pattern_size, flags_full)

        if ret:
            best_gray = processed
            break

    if corners is None:
        return None, None, None

    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(best_gray, corners, (11, 11), (-1, -1), criteria)
    corners = corners.reshape(-1, 2)

    # Generate 3D object points (centered at pattern center)
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    for i in range(rows):
        for j in range(cols):
            objp[i * cols + j] = [
                (j - (cols - 1) / 2.0) * square_size,
                (i - (rows - 1) / 2.0) * square_size,
                0
            ]

    return corners, objp, pattern_size


def draw_chessboard(image, corners, pattern_size):
    """Draw detected chessboard corners on image.

    Args:
        image: input BGR image
        corners: detected corners (N, 2)
        pattern_size: inner corner count (cols, rows)

    Returns:
        vis: visualization image with corners drawn
    """
    vis = image.copy()
    if corners is not None and len(corners) > 0:
        corners_draw = corners.reshape(-1, 1, 2).astype(np.float32)
        cv2.drawChessboardCorners(vis, pattern_size, corners_draw, True)
    return vis
