#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Launch file for camera_driver
Launches multiple RealSense cameras with custom TF publisher.
Cameras are started with delays to avoid USB bandwidth conflicts:
- Top camera: starts immediately
- Chassis camera: starts after 3 seconds (reduced from 6s)
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    camera_driver_dir = get_package_share_directory('camera_driver')
    realsense_dir = get_package_share_directory('realsense2_camera')

    # ========== Launch Arguments ==========
    # Camera enable flags
    top_enable_arg = DeclareLaunchArgument(
        'top_enable', default_value='false',
        description='Enable top camera'
    )
    hand_enable_arg = DeclareLaunchArgument(
        'hand_enable', default_value='false',
        description='Enable hand camera'
    )
    chassis_enable_arg = DeclareLaunchArgument(
        'chassis_enable', default_value='false',
        description='Enable chassis camera'
    )

    # Camera serial numbers (quoted strings required by realsense2_camera)
    # 车顶相机 (D455) - 最高处，大相机
    top_serial_arg = DeclareLaunchArgument(
        'top_serial_no', default_value="'318122302992'",
        description='Top camera serial number (D455, 车顶, 大相机)'
    )
    # 手臂相机 (D435) - 手臂末端，小相机
    hand_serial_arg = DeclareLaunchArgument(
        'hand_serial_no', default_value="'337122071190'",
        description='Hand camera serial number (D435, 手臂)'
    )
    # 底盘相机 (D435) - 小相机
    chassis_serial_arg = DeclareLaunchArgument(
        'chassis_serial_no', default_value="'337122071540'",
        description='Chassis camera serial number (D435, 底盘)'
    )

    # Image configuration (1280x720 matching LiDAR calibration)
    depth_profile_arg = DeclareLaunchArgument(
        'depth_profile', default_value='1280,720,6',
        description='Depth stream profile (width,height,fps)'
    )
    color_profile_arg = DeclareLaunchArgument(
        'color_profile', default_value='1280,720,6',
        description='Color stream profile (width,height,fps)'
    )

    # Feature flags
    enable_depth_arg = DeclareLaunchArgument(
        'enable_depth', default_value='true',
        description='Enable depth stream'
    )
    enable_color_arg = DeclareLaunchArgument(
        'enable_color', default_value='true',
        description='Enable color stream'
    )
    pointcloud_enable_arg = DeclareLaunchArgument(
        'pointcloud_enable', default_value='false',
        description='Enable point cloud'
    )
    enable_sync_arg = DeclareLaunchArgument(
        'enable_sync', default_value='true',
        description='Enable sync mode (required for RTAB-Map)'
    )
    align_depth_enable_arg = DeclareLaunchArgument(
        'align_depth_enable', default_value='true',
        description='Enable depth alignment to color'
    )
    initial_reset_arg = DeclareLaunchArgument(
        'initial_reset', default_value='true',
        description='Reset device on startup'
    )
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf', default_value='true',
        description='Publish TF from RealSense driver'
    )
    enable_camera_tf_arg = DeclareLaunchArgument(
        'enable_camera_tf', default_value='true',
        description='Enable custom camera TF publisher'
    )

    # Startup delay between cameras (seconds)
    camera_delay_arg = DeclareLaunchArgument(
        'camera_delay', default_value='3.0',
        description='Delay between camera startups to avoid USB conflicts'
    )

    # Get launch configurations
    top_enable = LaunchConfiguration('top_enable')
    hand_enable = LaunchConfiguration('hand_enable')
    chassis_enable = LaunchConfiguration('chassis_enable')
    top_serial_no = LaunchConfiguration('top_serial_no')
    hand_serial_no = LaunchConfiguration('hand_serial_no')
    chassis_serial_no = LaunchConfiguration('chassis_serial_no')
    depth_profile = LaunchConfiguration('depth_profile')
    color_profile = LaunchConfiguration('color_profile')
    enable_depth = LaunchConfiguration('enable_depth')
    enable_color = LaunchConfiguration('enable_color')
    pointcloud_enable = LaunchConfiguration('pointcloud_enable')
    enable_sync = LaunchConfiguration('enable_sync')
    align_depth_enable = LaunchConfiguration('align_depth_enable')
    initial_reset = LaunchConfiguration('initial_reset')
    publish_tf = LaunchConfiguration('publish_tf')
    enable_camera_tf = LaunchConfiguration('enable_camera_tf')

    # Common camera arguments
    def get_camera_args(camera_name, camera_namespace, serial_no):
        return {
            'camera_name': camera_name,
            'camera_namespace': camera_namespace,
            'serial_no': serial_no,
            'initial_reset': initial_reset,
            'enable_depth': enable_depth,
            'enable_color': enable_color,
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'depth_module.depth_profile': depth_profile,
            'rgb_camera.color_profile': color_profile,
            'enable_sync': enable_sync,
            'align_depth.enable': align_depth_enable,
            'pointcloud.enable': pointcloud_enable,
            'publish_tf': publish_tf,
        }

    # ========== Top Camera (starts immediately) ==========
    # Topic format: /{camera_namespace}/{camera_name}/... → /camera/top/...
    top_camera_launch = GroupAction(
        condition=IfCondition(top_enable),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(realsense_dir, 'launch', 'rs_launch.py')
                ),
                launch_arguments=get_camera_args('top', 'camera', top_serial_no).items()
            )
        ]
    )

    # ========== Hand & Chassis Cameras (delayed, scope-safe) ==========
    # NOTE: OpaqueFunction is used to resolve LaunchConfiguration values at
    # traversal time. This avoids a scoping bug where TimerAction fires AFTER
    # IncludeLaunchDescription has popped its LaunchConfiguration scope,
    # causing "launch configuration does not exist" errors.
    def _create_delayed_cameras(context, *args, **kwargs):
        """Create delayed camera launches with resolved values."""
        rs_launch = os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        cfg = context.launch_configurations
        actions = []

        def _resolved_camera_args(camera_name, serial_key):
            return {
                'camera_name': camera_name,
                'camera_namespace': 'camera',
                'serial_no': cfg.get(serial_key, ''),
                'initial_reset': cfg.get('initial_reset', 'true'),
                'enable_depth': cfg.get('enable_depth', 'true'),
                'enable_color': cfg.get('enable_color', 'true'),
                'enable_infra1': 'false',
                'enable_infra2': 'false',
                'enable_gyro': 'false',
                'enable_accel': 'false',
                'depth_module.depth_profile': cfg.get('depth_profile', '1280,720,6'),
                'rgb_camera.color_profile': cfg.get('color_profile', '1280,720,6'),
                'enable_sync': cfg.get('enable_sync', 'true'),
                'align_depth.enable': cfg.get('align_depth_enable', 'true'),
                'pointcloud.enable': cfg.get('pointcloud_enable', 'false'),
                'publish_tf': cfg.get('publish_tf', 'true'),
            }

        delay = float(cfg.get('camera_delay', '3.0'))

        if cfg.get('hand_enable', 'false').lower() == 'true':
            actions.append(TimerAction(
                period=delay,
                actions=[IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch),
                    launch_arguments=_resolved_camera_args('hand', 'hand_serial_no').items()
                )]
            ))

        if cfg.get('chassis_enable', 'false').lower() == 'true':
            actions.append(TimerAction(
                period=delay,
                actions=[IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(rs_launch),
                    launch_arguments=_resolved_camera_args('chassis', 'chassis_serial_no').items()
                )]
            ))

        return actions

    delayed_cameras = OpaqueFunction(function=_create_delayed_cameras)

    # ========== Camera TF Publisher ==========
    camera_tf_publisher_node = Node(
        condition=IfCondition(enable_camera_tf),
        package='camera_driver',
        executable='camera_tf_publisher',
        name='camera_tf_publisher',
        output='screen',
        parameters=[{
            'config_path': os.path.join(camera_driver_dir, 'config')
        }]
    )

    return LaunchDescription([
        # Declare all arguments
        top_enable_arg,
        hand_enable_arg,
        chassis_enable_arg,
        top_serial_arg,
        hand_serial_arg,
        chassis_serial_arg,
        depth_profile_arg,
        color_profile_arg,
        enable_depth_arg,
        enable_color_arg,
        pointcloud_enable_arg,
        enable_sync_arg,
        align_depth_enable_arg,
        initial_reset_arg,
        publish_tf_arg,
        enable_camera_tf_arg,
        camera_delay_arg,

        # Launch cameras
        top_camera_launch,      # Starts immediately
        delayed_cameras,        # Hand & chassis cameras (delayed, scope-safe)

        # Launch TF publisher
        camera_tf_publisher_node,
    ])
