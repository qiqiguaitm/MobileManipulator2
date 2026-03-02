# -*- coding: utf-8 -*-
"""
Feature Extractor Module

Extracts calibration features from images (chessboard corners) and
point clouds (plane fitting, edge detection).
"""

import numpy as np
import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARNING] open3d not installed. Point cloud features will be limited.")


class FrameFeatures:
    """Container for extracted features from a single frame."""

    def __init__(self):
        # Image features
        self.corners_2d = None          # (N, 2) chessboard corners in image
        self.corners_3d_cam = None      # (N, 3) corners in camera frame
        self.camera_normal = None       # (3,) board normal in camera frame
        self.camera_centre = None       # (3,) board center in camera frame
        self.rvec = None                # (3,) rotation vector from solvePnP
        self.tvec = None                # (3,) translation vector from solvePnP

        # Point cloud features
        self.lidar_normal = None        # (3,) board normal in lidar frame
        self.lidar_centre = None        # (3,) board center in lidar frame
        self.lidar_corners = None       # (4, 3) estimated corners in lidar frame
        self.plane_inliers = None       # indices of plane inliers
        self.board_points = None        # (M, 3) points on the board

        # Quality metrics
        self.board_dimension_error = None  # difference from expected board size
        self.valid = False


class FeatureExtractor:
    """
    Extract calibration features from image and point cloud pairs.

    Based on acfr/cam_lidar_calibration algorithm.
    Uses camera-guided filtering to isolate calibration board in sparse point clouds.
    """

    def __init__(self, pattern_size, square_size, camera_matrix, dist_coeffs,
                 board_width=0.80, board_height=0.60):
        """
        Initialize feature extractor.

        Args:
            pattern_size: (cols, rows) inner corner count, e.g., (7, 6)
            square_size: chessboard square size in meters
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients [k1, k2, p1, p2, k3]
            board_width: physical board width in meters
            board_height: physical board height in meters
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.K = np.array(camera_matrix).reshape(3, 3)
        self.D = np.array(dist_coeffs).flatten()
        self.board_width = board_width
        self.board_height = board_height

        # Generate 3D object points for chessboard
        self.objp = self._generate_object_points()

        # Point cloud filtering parameters
        # Relaxed bounds - camera-guided filter will do the fine filtering
        self.pc_bounds = {
            'x_min': 0.5, 'x_max': 5.0,
            'y_min': -2.5, 'y_max': 2.5,
            'z_min': -0.5, 'z_max': 2.0  # Include points below LiDAR
        }

        # Initial extrinsics for camera-guided filtering (T_lidar_to_camera_optical)
        # Using idealized rotation based on coordinate system relationship:
        # Camera optical: X-right, Y-down, Z-forward
        # LiDAR: X-forward, Y-left, Z-up
        # Translation from config/init_extrinsics_lidar_to_top_camera.yaml
        self.init_R = np.array([
            [0, -1, 0],   # Camera X = -LiDAR Y (right = -left)
            [0, 0, -1],   # Camera Y = -LiDAR Z (down = -up)
            [1, 0, 0]     # Camera Z = LiDAR X (forward = forward)
        ], dtype=float)
        # Simplified translation estimate for camera-guided filtering
        # Camera is approximately: in front of LiDAR (-0.43m), above LiDAR (~0.1m)
        # In optical frame (X-right, Y-down, Z-forward):
        # - LiDAR is behind camera (Z negative)
        # - LiDAR is slightly below camera (Y slightly positive)
        self.init_t = np.array([0.0, 0.1, -0.43])  # Adjusted for reasonable board Z
        self.use_camera_guided_filter = True
        self.guide_filter_margin = 0.6  # meters margin around estimated board

    def set_initial_extrinsics(self, R, t, is_tf_convention=False):
        """
        Set initial extrinsics estimate for camera-guided filtering.

        Args:
            R: 3x3 rotation matrix
            t: 3-element translation vector
            is_tf_convention: If True, R and t are in TF convention where:
                - t is camera origin in lidar frame
                - R columns are camera axes in lidar frame
                Need to invert for point transformation: p_cam = R^T @ p_lid - R^T @ t
        """
        R = np.array(R).reshape(3, 3)
        t = np.array(t).flatten()

        if is_tf_convention:
            # Convert TF convention to point transform convention
            # TF: t = camera position in lidar frame
            # Point transform: p_cam = R @ p_lid + t
            self.init_R = R.T
            self.init_t = -R.T @ t
            print("  [Extrinsics] Converted from TF convention")
        else:
            self.init_R = R
            self.init_t = t

    def _generate_object_points(self):
        """Generate 3D coordinates of chessboard corners in board frame."""
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), np.float32)

        for i in range(rows):
            for j in range(cols):
                objp[i * cols + j] = [
                    (j - (cols - 1) / 2.0) * self.square_size,
                    (i - (rows - 1) / 2.0) * self.square_size,
                    0
                ]
        return objp

    def set_pointcloud_bounds(self, x_min=None, x_max=None,
                               y_min=None, y_max=None,
                               z_min=None, z_max=None):
        """Set point cloud filtering bounds."""
        if x_min is not None:
            self.pc_bounds['x_min'] = x_min
        if x_max is not None:
            self.pc_bounds['x_max'] = x_max
        if y_min is not None:
            self.pc_bounds['y_min'] = y_min
        if y_max is not None:
            self.pc_bounds['y_max'] = y_max
        if z_min is not None:
            self.pc_bounds['z_min'] = z_min
        if z_max is not None:
            self.pc_bounds['z_max'] = z_max

    def extract(self, image, pcd_or_points):
        """
        Extract features from an image and point cloud pair.

        Args:
            image: BGR image (numpy array)
            pcd_or_points: Open3D PointCloud or (N, 3) numpy array

        Returns:
            FrameFeatures object
        """
        features = FrameFeatures()

        # Extract image features first (needed for camera-guided filtering)
        img_ok = self._extract_image_features(image, features)
        if not img_ok:
            return features

        # Extract point cloud features using camera-guided filtering
        pc_ok = self._extract_pointcloud_features(pcd_or_points, features)
        if not pc_ok:
            return features

        features.valid = True
        return features

    def _camera_guided_filter(self, points, features):
        """
        Filter point cloud using camera detection result to isolate board region.

        Uses the camera-detected board pose to estimate where the board should be
        in the LiDAR frame, then filters points to that region.

        Args:
            points: (N, 3) point cloud in LiDAR frame
            features: FrameFeatures with camera detection results

        Returns:
            Filtered points near the estimated board location
        """
        if features.camera_centre is None:
            return points

        # Board center in camera frame
        board_center_cam = features.camera_centre

        # Inverse transform to get rough board position in LiDAR frame
        # T_lidar_to_camera: p_cam = R @ p_lid + t
        # So: p_lid = R^T @ (p_cam - t)
        R_inv = self.init_R.T
        t_inv = -R_inv @ self.init_t

        # Estimate board center in LiDAR frame
        board_center_lidar = R_inv @ board_center_cam + t_inv

        # Also get the board's 4 corners in camera frame and transform
        # Board corners in board frame (half-widths)
        hw, hh = self.board_width / 2, self.board_height / 2
        board_corners_local = np.array([
            [-hw, -hh, 0],
            [hw, -hh, 0],
            [hw, hh, 0],
            [-hw, hh, 0]
        ])

        # Transform to camera frame
        R_board, _ = cv2.Rodrigues(features.rvec)
        board_corners_cam = (R_board @ board_corners_local.T).T + features.tvec

        # Transform to LiDAR frame
        board_corners_lidar = (R_inv @ board_corners_cam.T).T + t_inv

        # Compute bounding box with margin
        margin = self.guide_filter_margin
        x_min = board_corners_lidar[:, 0].min() - margin
        x_max = board_corners_lidar[:, 0].max() + margin
        y_min = board_corners_lidar[:, 1].min() - margin
        y_max = board_corners_lidar[:, 1].max() + margin
        z_min = board_corners_lidar[:, 2].min() - margin
        z_max = board_corners_lidar[:, 2].max() + margin

        # Filter points within the bounding box
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )

        filtered = points[mask]

        print("    [Camera-guided] Estimated board at LiDAR [%.2f, %.2f, %.2f]" %
              tuple(board_center_lidar))
        print("    [Camera-guided] Filter box: X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]" %
              (x_min, x_max, y_min, y_max, z_min, z_max))
        print("    [Camera-guided] Points: %d -> %d" % (len(points), len(filtered)))

        return filtered

    def _extract_image_features(self, image, features):
        """
        Detect chessboard and compute camera-frame features.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)

        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if not ret:
            print("  [WARN] Chessboard not detected in image")
            return False

        # Refine to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        features.corners_2d = corners.reshape(-1, 2)

        # Solve PnP to get board pose in camera frame
        ret, rvec, tvec = cv2.solvePnP(
            self.objp, corners, self.K, self.D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ret:
            print("  [WARN] solvePnP failed")
            return False

        features.rvec = rvec.flatten()
        features.tvec = tvec.flatten()

        # Convert to rotation matrix
        R_cam, _ = cv2.Rodrigues(rvec)

        # Board center in camera frame (board origin is at center)
        features.camera_centre = tvec.flatten()

        # Board normal in camera frame (Z-axis of board frame)
        # The board's Z-axis points out of the board surface
        features.camera_normal = R_cam[:, 2]

        # Ensure normal points towards camera (negative Z in camera frame)
        if features.camera_normal[2] > 0:
            features.camera_normal = -features.camera_normal

        # Transform object points to camera frame
        features.corners_3d_cam = (R_cam @ self.objp.T).T + tvec.flatten()

        return True

    def _extract_pointcloud_features(self, pcd_or_points, features):
        """
        Extract plane and edge features from point cloud.
        Uses camera-guided filtering to isolate the calibration board.
        """
        # Convert to numpy array if Open3D point cloud
        if HAS_OPEN3D and isinstance(pcd_or_points, o3d.geometry.PointCloud):
            points = np.asarray(pcd_or_points.points)
        else:
            points = np.array(pcd_or_points)

        # Filter out NaN values
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

        if len(points) == 0:
            print("  [WARN] Empty point cloud")
            return False

        # Stage 1: Basic bounding box filter
        mask = (
            (points[:, 0] >= self.pc_bounds['x_min']) &
            (points[:, 0] <= self.pc_bounds['x_max']) &
            (points[:, 1] >= self.pc_bounds['y_min']) &
            (points[:, 1] <= self.pc_bounds['y_max']) &
            (points[:, 2] >= self.pc_bounds['z_min']) &
            (points[:, 2] <= self.pc_bounds['z_max'])
        )
        filtered_points = points[mask]

        # Stage 2: Camera-guided filtering (if enabled and camera features available)
        if self.use_camera_guided_filter and features.camera_centre is not None:
            filtered_points = self._camera_guided_filter(filtered_points, features)

        if len(filtered_points) < 5:
            print("  [WARN] Too few points after filtering: %d" % len(filtered_points))
            return False

        # Fit plane using RANSAC
        plane_normal, plane_center, inlier_points = self._fit_plane_ransac(filtered_points)

        if plane_normal is None:
            print("  [WARN] Plane fitting failed")
            return False

        features.lidar_normal = plane_normal
        features.lidar_centre = plane_center
        features.board_points = inlier_points

        # Estimate board corners from edge points
        corners = self._estimate_board_corners(inlier_points, plane_normal)
        if corners is not None:
            features.lidar_corners = corners

            # Compute board dimension error
            width = np.linalg.norm(corners[1] - corners[0])
            height = np.linalg.norm(corners[3] - corners[0])
            expected_diag = np.sqrt(self.board_width**2 + self.board_height**2)
            actual_diag = np.sqrt(width**2 + height**2)
            features.board_dimension_error = abs(actual_diag - expected_diag)

        return True

    def _fit_plane_ransac(self, points, distance_threshold=0.02, max_iterations=1000):
        """
        Fit a plane to points using RANSAC.

        Returns:
            (normal, center, inlier_points) or (None, None, None) if failed
        """
        if HAS_OPEN3D:
            return self._fit_plane_ransac_open3d(points, distance_threshold, max_iterations)
        else:
            return self._fit_plane_ransac_numpy(points, distance_threshold, max_iterations)

    def _fit_plane_ransac_open3d(self, points, distance_threshold, max_iterations):
        """Use Open3D for RANSAC plane fitting."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Try multiple RANSAC iterations to find the best vertical-ish plane
        best_plane = None
        best_inliers = []
        best_score = -1

        for attempt in range(3):
            try:
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=3,
                    num_iterations=max_iterations
                )
            except Exception as e:
                continue

            if len(inliers) < 5:
                continue

            # Extract normal
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)

            # Score: prefer tilted planes (some X component, but NOT perfectly [1,0,0])
            # Wall has normal ~ [1, 0, 0], calibration board should have some tilt
            x_comp = abs(normal[0])
            y_comp = abs(normal[1])
            z_comp = abs(normal[2])

            # Penalize horizontal planes (ground) strongly - but don't skip entirely
            ground_penalty = 0.0
            if z_comp > 0.7:
                ground_penalty = 0.9  # Heavy penalty but still consider

            # Penalize perfectly frontal planes (wall) - prefer tilted boards
            # Wall: x_comp ~1, y_comp ~0, z_comp ~0
            # Tilted board: x_comp ~0.6-0.9, y_comp or z_comp ~0.3-0.6
            wall_penalty = 1.0 if (x_comp > 0.98 and y_comp < 0.15 and z_comp < 0.15) else 0.0
            tilt_bonus = y_comp + z_comp  # Reward tilted planes

            score = len(inliers) * (0.5 + 0.5 * x_comp) * (1.0 + 0.5 * tilt_bonus) * (1.0 - 0.5 * wall_penalty) * (1.0 - ground_penalty)

            if score > best_score:
                best_score = score
                best_plane = plane_model
                best_inliers = inliers

            # Remove inliers and try again
            remaining_idx = list(set(range(len(points))) - set(inliers))
            if len(remaining_idx) < 10:
                break
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[remaining_idx])

        if best_plane is None:
            print("  [WARN] No valid plane found")
            return None, None, None

        # Extract normal
        normal = np.array(best_plane[:3])
        normal = normal / np.linalg.norm(normal)

        # Check for horizontal plane (ground) - should be rejected
        if abs(normal[2]) > 0.5:
            print("  [WARN] Detected horizontal plane (normal_z=%.3f), likely ground" % abs(normal[2]))
            return None, None, None

        # Ensure normal points towards LiDAR sensor (negative X direction)
        # The board faces the LiDAR, so its normal should point back towards the sensor
        if normal[0] > 0:
            normal = -normal

        inlier_points = points[best_inliers]
        center = np.mean(inlier_points, axis=0)

        return normal, center, inlier_points

    def _fit_plane_ransac_numpy(self, points, distance_threshold, max_iterations):
        """Pure numpy RANSAC plane fitting fallback."""
        best_inliers = []
        best_normal = None
        n_points = len(points)

        for _ in range(max_iterations):
            # Random sample 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # Count inliers
            d = -np.dot(normal, p1)
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_normal = normal

        if len(best_inliers) < 5:
            return None, None, None

        # Refine normal using all inliers (SVD)
        inlier_points = points[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # Last row is the normal

        # Check for horizontal plane (ground)
        if abs(normal[2]) > 0.5:
            print("  [WARN] Detected horizontal plane (normal_z=%.3f), likely ground" % abs(normal[2]))
            return None, None, None

        # Ensure normal points towards LiDAR sensor (negative X direction)
        if normal[0] > 0:
            normal = -normal

        return normal, centroid, inlier_points

    def _estimate_board_corners(self, inlier_points, plane_normal):
        """
        Estimate the 4 corners of the calibration board from edge points.

        Uses PCA to find principal axes, then finds extrema.
        """
        if len(inlier_points) < 4:
            return None

        # Project points onto the plane (remove normal component)
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid

        # Find principal axes in the plane
        # Remove the normal direction
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # First two eigenvectors span the plane
        axis1 = eigenvectors[:, 0]
        axis2 = eigenvectors[:, 1]

        # Project points onto 2D plane coordinates
        coords_2d = np.column_stack([
            np.dot(centered, axis1),
            np.dot(centered, axis2)
        ])

        # Find convex hull or extrema
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords_2d)
            hull_points = coords_2d[hull.vertices]

            # Find 4 corners as extrema in rotated coordinates
            corners_2d = self._find_rectangle_corners(hull_points)
        except:
            # Fallback: use min/max
            corners_2d = np.array([
                [coords_2d[:, 0].min(), coords_2d[:, 1].min()],
                [coords_2d[:, 0].max(), coords_2d[:, 1].min()],
                [coords_2d[:, 0].max(), coords_2d[:, 1].max()],
                [coords_2d[:, 0].min(), coords_2d[:, 1].max()],
            ])

        # Transform back to 3D
        corners_3d = np.zeros((4, 3))
        for i, (u, v) in enumerate(corners_2d):
            corners_3d[i] = centroid + u * axis1 + v * axis2

        return corners_3d

    def _find_rectangle_corners(self, hull_points):
        """Find 4 corners that best represent a rectangle from convex hull."""
        n = len(hull_points)
        if n < 4:
            return hull_points

        # Find the 4 points with maximum sum of distances to others
        # This is a heuristic to find corner-like points
        distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i] += np.linalg.norm(hull_points[i] - hull_points[j])

        # Get indices of 4 points with highest distances
        corner_idx = np.argsort(distances)[-4:]

        # Sort corners in consistent order (clockwise from top-left)
        corners = hull_points[corner_idx]
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        corners = corners[sorted_idx]

        return corners

    def visualize_detection(self, image, features, output_path=None):
        """
        Draw detected chessboard corners on image.

        Args:
            image: input BGR image
            features: FrameFeatures object
            output_path: optional path to save the visualization

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        if features.corners_2d is not None:
            # Draw corners
            corners = features.corners_2d.reshape(-1, 1, 2).astype(np.float32)
            cv2.drawChessboardCorners(vis_image, self.pattern_size, corners, True)

            # Draw board center
            if features.camera_centre is not None:
                center_2d, _ = cv2.projectPoints(
                    np.array([[0, 0, 0]], dtype=np.float32),
                    features.rvec, features.tvec, self.K, self.D
                )
                center_px = tuple(center_2d[0, 0].astype(int))
                cv2.circle(vis_image, center_px, 10, (0, 255, 0), -1)

            # Draw coordinate axes
            axis_points = np.float32([
                [0.1, 0, 0],   # X axis (red)
                [0, 0.1, 0],   # Y axis (green)
                [0, 0, 0.1]    # Z axis (blue)
            ])
            imgpts, _ = cv2.projectPoints(axis_points, features.rvec, features.tvec, self.K, self.D)
            origin = tuple(center_2d[0, 0].astype(int))

            cv2.line(vis_image, origin, tuple(imgpts[0, 0].astype(int)), (0, 0, 255), 3)  # X - red
            cv2.line(vis_image, origin, tuple(imgpts[1, 0].astype(int)), (0, 255, 0), 3)  # Y - green
            cv2.line(vis_image, origin, tuple(imgpts[2, 0].astype(int)), (255, 0, 0), 3)  # Z - blue

        if output_path:
            cv2.imwrite(output_path, vis_image)

        return vis_image
