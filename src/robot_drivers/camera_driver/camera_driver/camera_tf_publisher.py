#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera TF Publisher for ROS2
Reads extrinsic calibration from YAML files and publishes static TF transforms.
"""

import os
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory


class CameraTFPublisher(Node):
    """
    Camera TF Publisher - Reads extrinsic parameters from config files and publishes static TF transforms.
    """

    def __init__(self):
        super().__init__('camera_tf_publisher')

        # Declare parameters
        default_config_path = os.path.join(
            get_package_share_directory('camera_driver'), 'config'
        )
        self.declare_parameter('config_path', default_config_path)

        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.get_logger().info(f'Config path: {self.config_path}')

        # Create static TF broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # Load and publish all camera TF transforms
        self._load_and_publish_transforms()

        self.get_logger().info('Camera TF Publisher initialized')

    def _load_transform_from_yaml(self, yaml_file, expected_parent=None, expected_child=None):
        """Load transform from YAML file."""
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            transform_data = data['transform']

            # Extract transform information
            translation = transform_data['translation']
            rotation = transform_data['rotation']

            # Frame info may be under header or at root level
            if 'header' in data and 'frame_id' in data['header']:
                parent_frame = data['header']['frame_id']
                child_frame = data['header'].get('child_frame_id') or data.get('child_frame_id')
            else:
                parent_frame = data.get('frame_id') or data.get('parent_frame_id')
                child_frame = data.get('child_frame_id')

            # Optional: validate frame names
            if expected_parent and parent_frame != expected_parent:
                self.get_logger().warning(f'Expected parent {expected_parent}, got {parent_frame}')
            if expected_child and child_frame != expected_child:
                self.get_logger().warning(f'Expected child {expected_child}, got {child_frame}')

            return parent_frame, child_frame, translation, rotation

        except Exception as e:
            self.get_logger().error(f'Failed to load transform from {yaml_file}: {e}')
            return None

    def _create_transform_stamped(self, parent_frame, child_frame, translation, rotation):
        """Create TransformStamped message."""
        t = TransformStamped()
        # Use Time() for static TF (equivalent to ROS1 Time(0))
        t.header.stamp = Time().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Translation
        t.transform.translation.x = float(translation['x'])
        t.transform.translation.y = float(translation['y'])
        t.transform.translation.z = float(translation['z'])

        # Rotation (quaternion)
        t.transform.rotation.x = float(rotation['x'])
        t.transform.rotation.y = float(rotation['y'])
        t.transform.rotation.z = float(rotation['z'])
        t.transform.rotation.w = float(rotation['w'])

        return t

    def _standardize_frame_name(self, original_name):
        """Standardize frame name - avoid conflicts with URDF calibration frames."""
        frame_mapping = {
            'rs16_lidar': 'calibration/lidar',
            'rslidar': 'calibration/lidar',
            'chassis_camera': 'calibration/chassis_camera_physical',
            'top_camera': 'calibration/top_camera_physical',
            'hand_camera': 'calibration/hand_camera_physical',
            'gripper': 'gripper_link',
            'chassis_base': 'base_link',
            'base': 'base_link'
        }

        return frame_mapping.get(original_name, original_name)

    def _load_and_publish_transforms(self):
        """Load config files and publish all TF transforms."""
        transforms_to_publish = []

        # 1. Base to calibrated LiDAR frame
        lidar_config = os.path.join(self.config_path, 'extrinsics_chassis_base_to_rslidar.yaml')
        self.get_logger().info(f'Looking for lidar config: {lidar_config}')
        self.get_logger().info(f'File exists: {os.path.exists(lidar_config)}')

        if os.path.exists(lidar_config):
            result = self._load_transform_from_yaml(lidar_config)
            self.get_logger().info(f'Load result: {result}')
            if result:
                parent, child, trans, rot = result
                parent = self._standardize_frame_name(parent)
                child = self._standardize_frame_name(child)

                self.get_logger().info(f'Lidar config: original parent={result[0]}, child={result[1]}')
                self.get_logger().info(f'Lidar config: mapped parent={parent}, child={child}')

                if parent == 'base_link' and child == 'calibration/lidar':
                    t = self._create_transform_stamped(parent, child, trans, rot)
                    transforms_to_publish.append(t)
                    self.get_logger().info(f'Loaded calibrated: {parent} -> {child}')
                else:
                    self.get_logger().warning(f'Lidar config condition failed: parent={parent}, child={child}')

        # 2. Calibrated LiDAR to chassis camera physical position
        chassis_config = os.path.join(self.config_path, 'extrinsics_lidar_to_chassis_camera.yaml')
        if os.path.exists(chassis_config):
            result = self._load_transform_from_yaml(chassis_config)
            if result:
                parent, child, trans, rot = result
                parent = self._standardize_frame_name(parent)

                if parent == 'calibration/lidar':
                    t = self._create_transform_stamped(
                        parent, 'calibration/chassis_camera_physical', trans, rot
                    )
                    transforms_to_publish.append(t)
                    self.get_logger().info(f'Loaded calibrated: {parent} -> calibration/chassis_camera_physical')

                    # Add optical frame transform
                    optical_t = self._create_camera_to_optical_transform_4calib(
                        'calibration/chassis_camera_physical',
                        'calibration/chassis_camera_optical'
                    )
                    transforms_to_publish.append(optical_t)
                    self.get_logger().info(
                        'Added optical: calibration/chassis_camera_physical -> calibration/chassis_camera_optical'
                    )

        # 3. Calibrated LiDAR to top camera physical position
        top_config = os.path.join(self.config_path, 'extrinsics_lidar_to_top_camera.yaml')
        if os.path.exists(top_config):
            result = self._load_transform_from_yaml(top_config)
            if result:
                parent, child, trans, rot = result
                parent = self._standardize_frame_name(parent)

                if parent == 'calibration/lidar':
                    t = self._create_transform_stamped(
                        parent, 'calibration/top_camera_physical', trans, rot
                    )
                    transforms_to_publish.append(t)
                    self.get_logger().info(f'Loaded calibrated: {parent} -> calibration/top_camera_physical')

                    # Add optical frame transform
                    optical_t = self._create_camera_to_optical_transform_4calib(
                        'calibration/top_camera_physical',
                        'calibration/top_camera_optical'
                    )
                    transforms_to_publish.append(optical_t)
                    self.get_logger().info(
                        'Added optical: calibration/top_camera_physical -> calibration/top_camera_optical'
                    )

        # 4. Connection transforms: URDF camera link -> RealSense camera link
        # Top camera connection
        top_connect_t = TransformStamped()
        top_connect_t.header.stamp = Time().to_msg()
        top_connect_t.header.frame_id = 'top_camera_link'
        top_connect_t.child_frame_id = 'camera/top_link'
        top_connect_t.transform.translation.x = 0.0
        top_connect_t.transform.translation.y = 0.0
        top_connect_t.transform.translation.z = 0.0
        top_connect_t.transform.rotation.x = 0.0
        top_connect_t.transform.rotation.y = 0.0
        top_connect_t.transform.rotation.z = 0.0
        top_connect_t.transform.rotation.w = 1.0
        transforms_to_publish.append(top_connect_t)
        self.get_logger().info('Added connection: top_camera_link -> camera/top_link')

        # Chassis camera connection (identity - chassis_camera_link = camera_link)
        chassis_connect_t = TransformStamped()
        chassis_connect_t.header.stamp = Time().to_msg()
        chassis_connect_t.header.frame_id = 'chassis_camera_link'
        chassis_connect_t.child_frame_id = 'camera/chassis_link'
        chassis_connect_t.transform.translation.x = 0.0
        chassis_connect_t.transform.translation.y = 0.0
        chassis_connect_t.transform.translation.z = 0.0
        chassis_connect_t.transform.rotation.x = 0.0
        chassis_connect_t.transform.rotation.y = 0.0
        chassis_connect_t.transform.rotation.z = 0.0
        chassis_connect_t.transform.rotation.w = 1.0
        transforms_to_publish.append(chassis_connect_t)
        self.get_logger().info('Added connection: chassis_camera_link -> camera/chassis_link')

        # Hand camera connection
        hand_connect_t = TransformStamped()
        hand_connect_t.header.stamp = Time().to_msg()
        hand_connect_t.header.frame_id = 'hand_camera_link'
        hand_connect_t.child_frame_id = 'camera/hand_link'
        hand_connect_t.transform.translation.x = 0.0
        hand_connect_t.transform.translation.y = 0.0
        hand_connect_t.transform.translation.z = 0.0
        hand_connect_t.transform.rotation.x = 0.0
        hand_connect_t.transform.rotation.y = 0.0
        hand_connect_t.transform.rotation.z = 0.0
        hand_connect_t.transform.rotation.w = 1.0
        transforms_to_publish.append(hand_connect_t)
        self.get_logger().info('Added connection: hand_camera_link -> camera/hand_link')

        # Publish all transforms
        if transforms_to_publish:
            self.tf_broadcaster.sendTransform(transforms_to_publish)
            self.get_logger().info(f'Published {len(transforms_to_publish)} camera TF transforms')
        else:
            self.get_logger().warning('No valid TF transforms found to publish')

    def _create_camera_to_optical_transform_4rs(self, physical_frame, optical_frame):
        """Create standard transform from camera physical position to optical center.

        RealSense camera optical coordinate system:
        - Z axis: forward (into the scene)
        - X axis: right
        - Y axis: down
        """
        t = TransformStamped()
        t.header.stamp = Time().to_msg()
        t.header.frame_id = physical_frame
        t.child_frame_id = optical_frame

        # No translation (optical center coincides with physical center)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Rotation: from physical to optical coordinate system
        r = R.from_euler('xyz', [-np.pi / 2, 0, -np.pi / 2])
        quat = r.as_quat()

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        return t

    def _create_camera_to_optical_transform_4calib(self, physical_frame, optical_frame):
        """Create transform from camera physical position to optical center for calibration."""
        t = TransformStamped()
        t.header.stamp = Time().to_msg()
        t.header.frame_id = physical_frame
        t.child_frame_id = optical_frame

        # No translation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Identity rotation for calibration frames
        r = R.from_euler('xyz', [0, 0, 0])
        quat = r.as_quat()

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        return t

    def _invert_transform(self, translation, rotation):
        """Compute inverse of a transform."""
        # Create rotation object
        r = R.from_quat([rotation['x'], rotation['y'], rotation['z'], rotation['w']])

        # Inverse rotation
        r_inv = r.inv()
        quat_inv = r_inv.as_quat()

        # Inverse translation: t_inv = -R^-1 * t
        trans_vec = [translation['x'], translation['y'], translation['z']]
        trans_inv_vec = -r_inv.apply(trans_vec)

        inv_translation = {
            'x': trans_inv_vec[0],
            'y': trans_inv_vec[1],
            'z': trans_inv_vec[2]
        }

        inv_rotation = {
            'x': quat_inv[0],
            'y': quat_inv[1],
            'z': quat_inv[2],
            'w': quat_inv[3]
        }

        return inv_translation, inv_rotation


def main(args=None):
    rclpy.init(args=args)

    node = CameraTFPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
