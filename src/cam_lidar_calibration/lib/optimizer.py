# -*- coding: utf-8 -*-
"""
Optimizer Module

Two-stage optimization for camera-lidar extrinsic calibration.
Based on acfr/cam_lidar_calibration algorithm.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class Optimizer:
    """
    Two-stage optimizer for camera-lidar extrinsic calibration.

    Stage 1: Optimize rotation only (using normal alignment)
    Stage 2: Jointly optimize rotation and translation
    """

    def __init__(self, camera_matrix):
        """
        Initialize optimizer.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.K = np.array(camera_matrix).reshape(3, 3)

        # Cost function weights
        self.weights = {
            'perpendicular': 1.0,
            'normal_align': 1.0,
            'centre_mean': 1.0,
            'centre_std': 0.5,
            'reprojection': 0.1
        }

    def solve(self, features_list):
        """
        Solve for extrinsic calibration using two-stage optimization.

        Args:
            features_list: List of FrameFeatures objects (valid ones only)

        Returns:
            (R, t): Rotation matrix (3x3) and translation vector (3,)
                    Represents T_lidar_to_camera
        """
        # Filter valid features
        valid_features = [f for f in features_list if f.valid]

        if len(valid_features) < 3:
            raise ValueError("Need at least 3 valid frames for calibration, got %d" % len(valid_features))

        print("[Optimizer] Using %d valid frames" % len(valid_features))

        # Stage 1: Rotation only optimization
        print("[Stage 1] Optimizing rotation...")
        R_init = self._compute_initial_rotation(valid_features)
        euler_init = Rotation.from_matrix(R_init).as_euler('xyz')

        result1 = minimize(
            self._rotation_cost,
            euler_init,
            args=(valid_features,),
            method='BFGS',
            options={'maxiter': 500, 'disp': False}
        )

        R_opt = Rotation.from_euler('xyz', result1.x).as_matrix()
        print("  Initial rotation cost: %.6f" % self._rotation_cost(euler_init, valid_features))
        print("  Optimized rotation cost: %.6f" % result1.fun)

        # Stage 2: Joint rotation + translation optimization
        print("[Stage 2] Optimizing rotation + translation...")
        t_init = self._compute_initial_translation(R_opt, valid_features)

        params_init = np.concatenate([result1.x, t_init])

        result2 = minimize(
            self._full_cost,
            params_init,
            args=(valid_features,),
            method='BFGS',
            options={'maxiter': 1000, 'disp': False}
        )

        R_final = Rotation.from_euler('xyz', result2.x[:3]).as_matrix()
        t_final = result2.x[3:6]

        print("  Initial full cost: %.6f" % self._full_cost(params_init, valid_features))
        print("  Optimized full cost: %.6f" % result2.fun)

        # Compute and display final errors
        self._print_error_statistics(R_final, t_final, valid_features)

        return R_final, t_final

    def _compute_initial_rotation(self, features):
        """
        Compute initial rotation estimate using SVD.

        T_lidar_to_camera means: p_camera = R @ p_lidar + t
        So: camera_normal = R @ lidar_normal
        We solve for R such that R @ lidar_normals ≈ camera_normals
        """
        cam_normals = np.array([f.camera_normal for f in features])
        lid_normals = np.array([f.lidar_normal for f in features])

        # Solve for R such that R @ lid_normals.T ≈ cam_normals.T
        # Using SVD: R = V @ U.T where USV = cam_normals.T @ lid_normals

        H = cam_normals.T @ lid_normals
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R

    def _compute_initial_translation(self, R, features):
        """
        Compute initial translation estimate.

        T_lidar_to_camera: p_camera = R @ p_lidar + t
        So: camera_centre = R @ lidar_centre + t
        Thus: t = camera_centre - R @ lidar_centre
        """
        translations = []
        for f in features:
            t = f.camera_centre - R @ f.lidar_centre
            translations.append(t)

        return np.mean(translations, axis=0)

    def _rotation_cost(self, euler, features):
        """
        Cost function for rotation-only optimization.

        T_lidar_to_camera: camera_normal = R @ lidar_normal
        Minimize: 1 - |dot(R @ lidar_normal, camera_normal)|
        """
        R = Rotation.from_euler('xyz', euler).as_matrix()

        cost = 0.0

        for f in features:
            # Transform lidar normal to camera frame
            lid_normal_transformed = R @ f.lidar_normal

            # Normal alignment cost: 1 - |dot(R @ lidar_normal, camera_normal)|
            dot_product = np.dot(lid_normal_transformed, f.camera_normal)
            normal_cost = 1.0 - abs(dot_product)
            cost += self.weights['normal_align'] * normal_cost

        return cost / len(features)

    def _full_cost(self, params, features):
        """
        Full cost function for joint optimization.

        T_lidar_to_camera: p_camera = R @ p_lidar + t
        - camera_normal = R @ lidar_normal
        - camera_centre = R @ lidar_centre + t
        """
        euler = params[:3]
        t = params[3:6]
        R = Rotation.from_euler('xyz', euler).as_matrix()

        normal_cost = 0.0
        centre_errors = []

        for f in features:
            # Normal alignment: R @ lidar_normal should equal camera_normal
            lid_normal_transformed = R @ f.lidar_normal
            dot_product = np.dot(lid_normal_transformed, f.camera_normal)
            normal_cost += 1.0 - abs(dot_product)

            # Centre alignment: R @ lidar_centre + t should equal camera_centre
            lid_centre_transformed = R @ f.lidar_centre + t
            centre_error = np.linalg.norm(lid_centre_transformed - f.camera_centre)
            centre_errors.append(centre_error)

        n = len(features)
        normal_cost = self.weights['normal_align'] * normal_cost / n

        centre_mean = np.mean(centre_errors)
        centre_std = np.std(centre_errors)
        centre_cost = (self.weights['centre_mean'] * centre_mean +
                       self.weights['centre_std'] * centre_std)

        total_cost = normal_cost + centre_cost

        return total_cost

    def _print_error_statistics(self, R, t, features):
        """Print calibration error statistics."""
        normal_errors = []
        centre_errors = []

        for f in features:
            # Normal alignment error (degrees)
            lid_normal_transformed = R @ f.lidar_normal
            dot_product = np.clip(np.dot(lid_normal_transformed, f.camera_normal), -1, 1)
            angle_error = np.degrees(np.arccos(abs(dot_product)))
            normal_errors.append(angle_error)

            # Centre alignment error (meters)
            lid_centre_transformed = R @ f.lidar_centre + t
            centre_error = np.linalg.norm(lid_centre_transformed - f.camera_centre)
            centre_errors.append(centre_error)

        print("\n[Calibration Statistics]")
        print("  Normal alignment error:")
        print("    Mean: %.2f deg, Std: %.2f deg" % (np.mean(normal_errors), np.std(normal_errors)))
        print("  Centre alignment error:")
        print("    Mean: %.3f m, Std: %.3f m" % (np.mean(centre_errors), np.std(centre_errors)))

    def compute_reprojection_error(self, R, t, features, image_size=None):
        """
        Compute reprojection error for verification.

        Projects lidar board points to image and compares with detected corners.

        Args:
            R: 3x3 rotation matrix
            t: 3-element translation vector
            features: List of FrameFeatures
            image_size: (width, height) for bounds checking

        Returns:
            List of per-frame mean reprojection errors in pixels
        """
        errors = []

        for f in features:
            if f.lidar_corners is None or f.corners_2d is None:
                continue

            # Transform lidar corners to camera frame
            # T_lidar_to_camera: p_cam = R @ p_lid + t
            corners_cam = (R @ f.lidar_corners.T).T + t

            # Project to image
            corners_proj = []
            for p in corners_cam:
                if p[2] <= 0:  # Behind camera
                    continue
                u = self.K[0, 0] * p[0] / p[2] + self.K[0, 2]
                v = self.K[1, 1] * p[1] / p[2] + self.K[1, 2]
                corners_proj.append([u, v])

            if len(corners_proj) < 4:
                continue

            corners_proj = np.array(corners_proj)

            # Compare with detected corners (use closest points)
            # Note: This is approximate since we only have 4 lidar corners
            # but many chessboard corners
            detected_corners = f.corners_2d

            # Use the 4 outer corners of the detected chessboard
            cols, rows = 7, 6  # Assuming this pattern size
            outer_corners = np.array([
                detected_corners[0],                    # Top-left
                detected_corners[cols - 1],             # Top-right
                detected_corners[(rows - 1) * cols],    # Bottom-left
                detected_corners[rows * cols - 1]       # Bottom-right
            ])

            # Compute mean distance
            # Match corners by finding closest pairs
            min_errors = []
            for pc in corners_proj:
                dists = np.linalg.norm(outer_corners - pc, axis=1)
                min_errors.append(np.min(dists))

            mean_error = np.mean(min_errors)
            errors.append(mean_error)

        return errors


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # Returns [x, y, z, w]
    return q


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    rot = Rotation.from_quat(q)
    return rot.as_matrix()


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (in radians) to 3x3 rotation matrix."""
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return rot.as_matrix()
