# -*- coding: utf-8 -*-
"""
Visualizer Module

Visualization utilities for camera-lidar calibration.
"""

import numpy as np
import cv2


class Visualizer:
    """
    Visualization tools for calibration results.
    """

    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        Initialize visualizer.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients (optional)
        """
        self.K = np.array(camera_matrix).reshape(3, 3)
        self.D = np.array(dist_coeffs).flatten() if dist_coeffs is not None else np.zeros(5)

    def project_points(self, points_3d, R, t):
        """
        Project 3D points (in lidar frame) to image coordinates.

        Args:
            points_3d: (N, 3) array of 3D points in lidar frame
            R: 3x3 rotation matrix (T_lidar_to_camera)
            t: 3-element translation vector

        Returns:
            (N, 2) array of 2D image coordinates
        """
        # Transform to camera frame
        points_cam = (R @ points_3d.T).T + t

        # Filter points behind camera
        valid = points_cam[:, 2] > 0.1
        points_cam = points_cam[valid]

        if len(points_cam) == 0:
            return np.array([]), np.array([])

        # Project to image
        u = self.K[0, 0] * points_cam[:, 0] / points_cam[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * points_cam[:, 1] / points_cam[:, 2] + self.K[1, 2]

        points_2d = np.column_stack([u, v])
        depths = points_cam[:, 2]

        return points_2d, depths

    def draw_projection(self, image, points_3d, R, t, color_by_depth=True,
                        point_size=2, max_depth=10.0):
        """
        Draw projected point cloud on image.

        Args:
            image: BGR image
            points_3d: (N, 3) array of 3D points in lidar frame
            R: 3x3 rotation matrix
            t: 3-element translation vector
            color_by_depth: if True, color points by depth
            point_size: radius of points
            max_depth: maximum depth for color scaling

        Returns:
            Annotated image
        """
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        points_2d, depths = self.project_points(points_3d, R, t)

        if len(points_2d) == 0:
            return vis_image

        for i, (pt, depth) in enumerate(zip(points_2d, depths)):
            x, y = int(pt[0]), int(pt[1])

            # Skip out-of-bounds points
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            if color_by_depth:
                # Color by depth (blue=near, red=far)
                t_val = min(1.0, depth / max_depth)
                color = self._depth_to_color(t_val)
            else:
                color = (0, 255, 0)

            cv2.circle(vis_image, (x, y), point_size, color, -1)

        return vis_image

    def _depth_to_color(self, t):
        """Convert depth ratio to BGR color (jet colormap)."""
        # Simple jet-like colormap
        if t < 0.25:
            r, g, b = 0, int(255 * t / 0.25), 255
        elif t < 0.5:
            r, g, b = 0, 255, int(255 * (0.5 - t) / 0.25)
        elif t < 0.75:
            r, g, b = int(255 * (t - 0.5) / 0.25), 255, 0
        else:
            r, g, b = 255, int(255 * (1.0 - t) / 0.25), 0

        return (b, g, r)

    def draw_board_overlay(self, image, features, R, t):
        """
        Draw calibration board overlay showing alignment quality.

        Args:
            image: BGR image
            features: FrameFeatures object
            R: 3x3 rotation matrix
            t: 3-element translation vector

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        # Draw detected corners (green)
        if features.corners_2d is not None:
            for corner in features.corners_2d:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), -1)

        # Draw projected lidar board corners (red)
        if features.lidar_corners is not None:
            corners_2d, _ = self.project_points(features.lidar_corners, R, t)
            for corner in corners_2d:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(vis_image, (x, y), 8, (0, 0, 255), 2)

            # Draw board outline
            if len(corners_2d) >= 4:
                pts = corners_2d[:4].astype(np.int32)
                cv2.polylines(vis_image, [pts], True, (0, 0, 255), 2)

        # Draw projected board center
        if features.lidar_centre is not None:
            centre_2d, _ = self.project_points(features.lidar_centre.reshape(1, 3), R, t)
            if len(centre_2d) > 0:
                x, y = int(centre_2d[0, 0]), int(centre_2d[0, 1])
                cv2.drawMarker(vis_image, (x, y), (0, 0, 255),
                              cv2.MARKER_CROSS, 20, 2)

        return vis_image

    def create_error_plot(self, errors, title="Reprojection Error"):
        """
        Create a bar plot of per-frame errors.

        Args:
            errors: List of error values
            title: Plot title

        Returns:
            Plot image (numpy array)
        """
        # Simple bar plot using OpenCV
        n = len(errors)
        if n == 0:
            return np.zeros((200, 400, 3), dtype=np.uint8)

        plot_width = max(400, n * 40)
        plot_height = 200
        margin = 40
        bar_width = max(10, (plot_width - 2 * margin) // n - 5)

        plot = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

        max_error = max(errors) * 1.2 if max(errors) > 0 else 1.0
        scale = (plot_height - 2 * margin) / max_error

        for i, err in enumerate(errors):
            x = margin + i * (bar_width + 5)
            h = int(err * scale)
            y_top = plot_height - margin - h
            y_bottom = plot_height - margin

            # Color based on error magnitude
            if err < 1.0:
                color = (0, 200, 0)  # Green
            elif err < 2.0:
                color = (0, 200, 200)  # Yellow
            else:
                color = (0, 0, 200)  # Red

            cv2.rectangle(plot, (x, y_top), (x + bar_width, y_bottom), color, -1)
            cv2.rectangle(plot, (x, y_top), (x + bar_width, y_bottom), (0, 0, 0), 1)

            # Frame number
            cv2.putText(plot, str(i), (x, plot_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Title
        cv2.putText(plot, title, (margin, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Y-axis label
        cv2.putText(plot, "%.1f px" % max_error, (5, margin),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return plot

    def save_result_visualization(self, image, points_3d, R, t, output_path,
                                   features=None, title=None):
        """
        Save a comprehensive visualization of calibration result.

        Args:
            image: BGR image
            points_3d: (N, 3) array of 3D points
            R: 3x3 rotation matrix
            t: 3-element translation vector
            output_path: Path to save the image
            features: Optional FrameFeatures for additional overlay
            title: Optional title text
        """
        # Draw projection
        vis = self.draw_projection(image, points_3d, R, t,
                                   color_by_depth=True, point_size=2)

        # Add board overlay if features provided
        if features is not None:
            vis = self.draw_board_overlay(vis, features, R, t)

        # Add title
        if title:
            cv2.putText(vis, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Add calibration info
        euler = self._rotation_to_euler(R)
        info_text = "T: [%.3f, %.3f, %.3f] m" % (t[0], t[1], t[2])
        cv2.putText(vis, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        rot_text = "R: [%.1f, %.1f, %.1f] deg" % tuple(np.degrees(euler))
        cv2.putText(vis, rot_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imwrite(output_path, vis)
        print("[Visualizer] Saved: %s" % output_path)

    def _rotation_to_euler(self, R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        return rot.as_euler('xyz')
