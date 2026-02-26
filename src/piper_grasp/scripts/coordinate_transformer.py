#!/usr/bin/env python3
"""
CoordinateTransformer - Hand-eye coordinate transformation module

Transforms points from camera optical frame to robot base frame
using hand-eye calibration parameters.

All internal calculations use meters. Unit conversion (m <-> mm)
is handled at the interface boundary.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class CoordinateTransformer:
    """
    Hand-eye coordinate transformer

    Coordinate frames:
    - Camera optical: Z forward, X right, Y down
    - Robot base: X forward, Y left, Z up
    - Gripper center: Relative to flange, Z offset
    """

    def __init__(self, config):
        """
        Initialize transformer with calibration parameters

        Args:
            config: dict with calibration parameters
                - T_cam2flan: [x, y, z] meters, camera to flange translation
                - q_cam2flan: [qx, qy, qz, qw] quaternion, camera to flange rotation
                - T_gripper2flan: [x, y, z] meters, gripper center to flange offset
        """
        # Load calibration (all in meters)
        self.T_cam2flan = np.array(config['T_cam2flan'])
        self.R_cam2flan = R.from_quat(config['q_cam2flan']).as_matrix()
        self.T_gripper2flan = np.array(config['T_gripper2flan'])

        # Compute camera to gripper center transform
        # gripper_center = flan + T_gripper2flan
        # cam = flan + T_cam2flan
        # So: cam_to_gripper = T_cam2flan - T_gripper2flan
        self.T_cam2gripper = self.T_cam2flan - self.T_gripper2flan
        self.R_cam2gripper = self.R_cam2flan

    def camera_to_base(self, point3d_cam_m, end_pose_gripper_center):
        """
        Transform point from camera frame to base frame

        Args:
            point3d_cam_m: [x, y, z] in meters, camera optical frame
            end_pose_gripper_center: [x, y, z, roll, pitch, yaw]
                                     position in mm, orientation in degrees
                                     (gripper center position)

        Returns:
            [x, y, z] in meters, base frame
        """
        point_cam = np.array(point3d_cam_m)

        # Step 1: Camera -> Gripper center
        point_in_gripper = self.R_cam2gripper @ point_cam + self.T_cam2gripper

        # Step 2: Gripper center -> Base
        # Convert gripper position from mm to meters
        gripper_pos_m = np.array(end_pose_gripper_center[0:3]) * 0.001
        # SDK uses xyz fixed-axis (extrinsic) Euler angles
        gripper_rot = R.from_euler(
            'xyz',  # Extrinsic (fixed-axis), matches Piper SDK convention
            end_pose_gripper_center[3:6],
            degrees=True
        )
        point_in_base = gripper_rot.as_matrix() @ point_in_gripper + gripper_pos_m

        return point_in_base.tolist()

    def compute_grasp_offset(self, point3d_cam_m, end_pose_gripper_center):
        """
        Compute grasp offset from current position

        Args:
            point3d_cam_m: [x, y, z] in meters, camera frame
            end_pose_gripper_center: [x, y, z, r, p, y] mm/degrees

        Returns:
            dict with:
                - point_in_base: [x, y, z] meters
                - offset: [dx, dy, dz] meters (target - current)
        """
        point_in_base = self.camera_to_base(point3d_cam_m, end_pose_gripper_center)

        # Current position in meters
        current_pos_m = np.array(end_pose_gripper_center[0:3]) * 0.001

        # Offset = target - current
        offset = np.array(point_in_base) - current_pos_m

        return {
            'point_in_base': point_in_base,
            'offset': offset.tolist()
        }


def create_transformer_from_ros_config(config_dict):
    """
    Create transformer from ROS parameter config

    Args:
        config_dict: dict from rosparam with calibration section

    Returns:
        CoordinateTransformer instance
    """
    calibration = config_dict.get('calibration', {})
    return CoordinateTransformer(calibration)


if __name__ == '__main__':
    # Test with example config
    test_config = {
        'T_cam2flan': [0.04010927904656854, 0.08283986339504643, 0.0025333968091171308],
        'q_cam2flan': [0.14206540330899609, -0.1423312973808512, 0.6810236905433535, 0.7041064947060541],
        'T_gripper2flan': [0.0, 0.0, 0.13503]
    }

    transformer = CoordinateTransformer(test_config)

    # Test point in camera frame (meters)
    point_cam = [0.15, 0.02, 0.35]

    # Test end pose (mm/degrees)
    end_pose = [400, 0, 150, 180, 30, 180]

    result = transformer.compute_grasp_offset(point_cam, end_pose)

    print("Test CoordinateTransformer:")
    print(f"  Camera point: {point_cam} m")
    print(f"  End pose: {end_pose} mm/deg")
    print(f"  Base point: {result['point_in_base']} m")
    print(f"  Offset: {result['offset']} m")

    # Convert to mm for display
    base_mm = [p * 1000 for p in result['point_in_base']]
    offset_mm = [o * 1000 for o in result['offset']]
    print(f"  Base point: {base_mm} mm")
    print(f"  Offset: {offset_mm} mm")
