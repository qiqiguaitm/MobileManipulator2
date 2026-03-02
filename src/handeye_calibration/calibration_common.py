#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_common.py - Hand-Eye Calibration Common Utilities

Provides:
- Math utility functions
- Data preparation and conversion
- Camera intrinsics management

Design: Linus Torvalds "Good Taste" principle
- Eliminate special cases and boundary conditions
- Short and focused functions
- No hard-coding, configuration-driven
"""

import numpy as np
import cv2
import yaml
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Math Utility Functions - Angle normalization and difference
# ============================================================================

def normalize_angle_deg(angle_deg):
    """Normalize angle to [-180, 180] degrees

    Good Taste: Eliminate boundary cases

    Args:
        angle_deg: Input angle in degrees

    Returns:
        float: Normalized angle
    """
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    return angle_deg


def angle_difference_deg(angle1_deg, angle2_deg):
    """Calculate minimum difference between two angles (degrees)

    Correctly handles crossing ±180° boundary

    Args:
        angle1_deg: Angle 1 in degrees
        angle2_deg: Angle 2 in degrees

    Returns:
        float: Absolute angle difference
    """
    angle1 = normalize_angle_deg(angle1_deg)
    angle2 = normalize_angle_deg(angle2_deg)
    diff = abs(angle1 - angle2)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


# ============================================================================
# Data Preparation - Convert collected data to calibration algorithm inputs
# ============================================================================

def prepare_calibration_data(collected_data, camera_matrix, dist_coeffs,
                            board_size, chessboard_size_mm,
                            compute_reprojection_errors=False):
    """Prepare hand-eye calibration pose data

    Extract robot poses and camera-target poses from raw collected data

    Args:
        collected_data: List of collected calibration data, each containing:
            - 'pose': (x,y,z,roll,pitch,yaw) robot end-effector pose
            - 'corners': Detected chessboard corners
            - 'frame_id': Frame number
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Camera distortion coefficients (5,)
        board_size: Chessboard inner corner count (cols, rows)
        chessboard_size_mm: Square size in millimeters
        compute_reprojection_errors: Whether to compute reprojection errors

    Returns:
        tuple: (R_gripper2base_list, t_gripper2base_list, R_target2cam_list,
                t_target2cam_list, frame_ids, reprojection_errors)
                If compute_reprojection_errors=False, reprojection_errors is None
    """
    # Prepare chessboard world coordinates
    square_size = chessboard_size_mm / 1000.0  # Convert to meters
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Initialize data lists
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    frame_ids = []
    reprojection_errors = [] if compute_reprojection_errors else None

    for data in collected_data:
        # Robot pose (end-effector to base)
        x, y, z, roll, pitch, yaw = data['pose']
        t_gripper2base_list.append(np.array([[x], [y], [z]]))
        R_robot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        R_gripper2base_list.append(R_robot)
        frame_ids.append(data['frame_id'])

        # Solve chessboard pose relative to camera
        ret, rvec, tvec = cv2.solvePnP(objp, data['corners'], camera_matrix, dist_coeffs)
        if ret:
            R_target2cam_mat, _ = cv2.Rodrigues(rvec)
            R_target2cam_list.append(R_target2cam_mat)
            t_target2cam_list.append(tvec)

            # Optional: compute reprojection error
            if compute_reprojection_errors:
                projected_points, _ = cv2.projectPoints(objp, rvec, tvec,
                                                       camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                detected_points = data['corners'].reshape(-1, 2)
                error = np.sqrt(np.mean(np.sum((projected_points - detected_points)**2, axis=1)))
                reprojection_errors.append(error)
        else:
            print(f"Warning: Frame {data['frame_id']}: solvePnP failed")

    return (R_gripper2base_list, t_gripper2base_list, R_target2cam_list,
            t_target2cam_list, frame_ids, reprojection_errors)


# ============================================================================
# Camera Intrinsics Manager - Smart loading and saving
# ============================================================================

class CameraIntrinsicsManager:
    """Camera intrinsics manager

    Smart loading strategy (priority descending):
    1. Load from data_dir/realsense_intrinsics.yaml
    2. Load from config_dir/hand_camera_intrinsics.yaml
    3. Load from RealSense camera factory calibration
    """

    @staticmethod
    def load_from_file(file_path, target_resolution=(1280, 720)):
        """Load camera intrinsics from YAML file

        Args:
            file_path: YAML file path
            target_resolution: (width, height) tuple for multi-resolution files

        Returns:
            tuple: (camera_matrix, dist_coeffs) or (None, None) if failed
        """
        if not os.path.exists(file_path):
            return None, None

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # 检查是否是多分辨率格式 (OpenCV calibration format)
            if 'model_type:' in content and 'projection_parameters:' in content:
                return CameraIntrinsicsManager._load_opencv_multi_resolution(
                    content, target_resolution
                )

            # 标准 YAML 格式
            data = yaml.load(content, Loader=yaml.FullLoader)

            # Try two formats
            if 'camera_matrix' in data:
                # realsense_intrinsics.yaml format
                camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
                dist_coeffs = np.array(data['distortion_coefficients'], dtype=np.float32)
            elif 'K' in data:
                # hand_camera_intrinsics.yaml format (ROS style)
                K = np.array(data['K'], dtype=np.float32)
                camera_matrix = K.reshape(3, 3)
                dist_coeffs = np.array(data['D'][:5], dtype=np.float32)
            else:
                print(f"Warning: Unknown intrinsics file format: {file_path}")
                return None, None

            return camera_matrix, dist_coeffs

        except Exception as e:
            print(f"Warning: Failed to read intrinsics file: {e}")
            return None, None

    @staticmethod
    def _load_opencv_multi_resolution(content, target_resolution):
        """Load OpenCV multi-resolution calibration format

        Args:
            content: File content string
            target_resolution: (width, height) tuple

        Returns:
            tuple: (camera_matrix, dist_coeffs) or (None, None)
        """
        import re

        # 分割成多个calibration块
        blocks = content.split('\n\n')

        for block in blocks:
            if not block.strip() or block.startswith('#'):
                continue

            # 提取 image_width 和 image_height
            width_match = re.search(r'image_width:\s*(\d+)', block)
            height_match = re.search(r'image_height:\s*(\d+)', block)

            if width_match and height_match:
                width = int(width_match.group(1))
                height = int(height_match.group(1))

                # 找到匹配的分辨率
                if (width, height) == target_resolution:
                    # 提取参数
                    fx_match = re.search(r'fx:\s*([\d.e+-]+)', block)
                    fy_match = re.search(r'fy:\s*([\d.e+-]+)', block)
                    cx_match = re.search(r'cx:\s*([\d.e+-]+)', block)
                    cy_match = re.search(r'cy:\s*([\d.e+-]+)', block)

                    k1_match = re.search(r'k1:\s*([\d.e+-]+)', block)
                    k2_match = re.search(r'k2:\s*([\d.e+-]+)', block)
                    p1_match = re.search(r'p1:\s*([\d.e+-]+)', block)
                    p2_match = re.search(r'p2:\s*([\d.e+-]+)', block)

                    if all([fx_match, fy_match, cx_match, cy_match,
                           k1_match, k2_match, p1_match, p2_match]):
                        # 构建 camera_matrix
                        fx = float(fx_match.group(1))
                        fy = float(fy_match.group(1))
                        cx = float(cx_match.group(1))
                        cy = float(cy_match.group(1))

                        camera_matrix = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ], dtype=np.float32)

                        # 构建 dist_coeffs (k1, k2, p1, p2, k3)
                        k1 = float(k1_match.group(1))
                        k2 = float(k2_match.group(1))
                        p1 = float(p1_match.group(1))
                        p2 = float(p2_match.group(1))

                        dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float32)

                        print(f"  Loaded {width}×{height} intrinsics from multi-resolution file")
                        return camera_matrix, dist_coeffs

        print(f"  Warning: Resolution {target_resolution} not found in file")
        return None, None

    @staticmethod
    def load_from_realsense(camera_id=None):
        """Load factory intrinsics from RealSense camera

        Args:
            camera_id: Optional, specify camera serial number

        Returns:
            tuple: (camera_matrix, dist_coeffs, intrinsics_info) or (None, None, None)
        """
        try:
            import pyrealsense2 as rs
            print("Attempting to get intrinsics from RealSense camera...")

            # Create temporary pipeline to get intrinsics
            pipeline = rs.pipeline()
            config = rs.config()

            # If camera ID specified, connect to that camera
            if camera_id:
                try:
                    config.enable_device(camera_id)
                    print(f"   Connecting to specified camera: {camera_id}")
                except:
                    print("   Using default camera")

            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

            # Start pipeline to get intrinsics
            profile = pipeline.start(config)
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            # Convert to OpenCV format
            camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array(intrinsics.coeffs[:5], dtype=np.float32)

            pipeline.stop()

            intrinsics_info = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'model': str(intrinsics.model),
                'coeffs': dist_coeffs.tolist()
            }

            print(f"Successfully got intrinsics from RealSense")
            print(f"   Focal: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"   Principal: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

            return camera_matrix, dist_coeffs, intrinsics_info

        except Exception as e:
            print(f"Warning: Cannot get intrinsics from camera: {e}")
            return None, None, None

    @staticmethod
    def save_to_file(camera_matrix, dist_coeffs, file_path, intrinsics_info=None):
        """Save camera intrinsics to YAML file

        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Camera distortion coefficients
            file_path: Save path
            intrinsics_info: Optional, additional intrinsics info
        """
        try:
            data = {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'source': 'RealSense Factory Calibration' if intrinsics_info else 'Manual',
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

            if intrinsics_info:
                data.update(intrinsics_info)

            with open(file_path, 'w') as f:
                yaml.dump(data, f)

            print(f"   Intrinsics saved to: {file_path}")
            return True

        except Exception as e:
            print(f"   Warning: Failed to save intrinsics: {e}")
            return False

    @staticmethod
    def get_camera_intrinsics(config_dir=None,
                             camera_id=None, save_to_dir=None):
        """Smart get camera intrinsics (multi-strategy fallback)

        Loading strategy priority:
        1. config_dir/hand_camera_intrinsics.yaml (unified config dir)
        2. Get from RealSense camera in real-time

        Args:
            data_dir: Optional, data directory path
            config_dir: Optional, config directory path
            default_intrinsics_file: Optional, default intrinsics file path
            camera_id: Optional, RealSense camera serial number
            save_to_dir: Optional, directory to save retrieved intrinsics

        Returns:
            tuple: (camera_matrix, dist_coeffs, source) or (None, None, "Failed")
        """
        # Strategy 1: Use hand_camera_intrinsics.yaml from config_dir
        if config_dir:
            intrinsics_file = os.path.join(config_dir, "hand_camera_intrinsics.yaml")
            if os.path.exists(intrinsics_file):
                camera_matrix, dist_coeffs = CameraIntrinsicsManager.load_from_file(intrinsics_file)
                if camera_matrix is not None:
                    print(f"Loaded camera intrinsics from config dir: {intrinsics_file}")
                    return camera_matrix, dist_coeffs, f"File: {intrinsics_file}"

       
        # Strategy 2: Try to get from RealSense camera
        camera_matrix, dist_coeffs, intrinsics_info = CameraIntrinsicsManager.load_from_realsense(camera_id)
        if camera_matrix is not None:
            # Save retrieved intrinsics
            if save_to_dir:
                save_path = os.path.join(save_to_dir, "realsense_intrinsics.yaml")
                CameraIntrinsicsManager.save_to_file(camera_matrix, dist_coeffs,
                                                    save_path, intrinsics_info)

            return camera_matrix, dist_coeffs, "RealSense Camera (Factory Calibration)"

        # All strategies failed
        print(f"Failed: All intrinsics loading strategies failed")
        return None, None, "Failed"


# ============================================================================
# Helper Functions
# ============================================================================

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (XYZ order)

    Args:
        R: Rotation matrix (3x3)

    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    return r.as_euler('xyz', degrees=False)


def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix (XYZ order)

    Args:
        roll, pitch, yaw: Euler angles in radians

    Returns:
        np.ndarray: Rotation matrix (3x3)
    """
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_matrix()
