#!/usr/bin/env python3
"""
Multi-Camera RViz Launch File

同时可视化 top 和 chassis 两个相机的感知结果

Usage:
    ros2 launch perception_bringup multi_camera_rviz.launch.py
    ros2 launch perception_bringup multi_camera_rviz.launch.py enable_chassis:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception_bringup')
    config_dir = os.path.join(bringup_share, 'config')

    # ==================== Launch Arguments ====================
    enable_top_arg = DeclareLaunchArgument(
        'enable_top', default_value='true',
        description='Enable top camera visualization'
    )
    enable_chassis_arg = DeclareLaunchArgument(
        'enable_chassis', default_value='true',
        description='Enable chassis camera visualization'
    )
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', default_value='2.0',
        description='Visualization publish rate (Hz)'
    )
    cloud_skip_arg = DeclareLaunchArgument(
        'cloud_skip', default_value='2',
        description='Point cloud downsampling factor (larger = fewer points)'
    )
    image_scale_arg = DeclareLaunchArgument(
        'image_scale', default_value='0.5',
        description='Image scale for point cloud generation (0.5 = half resolution)'
    )

    # ==================== RViz Node ====================
    multi_camera_rviz_node = Node(
        package='perception_nodes',
        executable='multi_camera_rviz_node',
        name='multi_camera_rviz',
        output='screen',
        parameters=[{
            'enable_top': LaunchConfiguration('enable_top'),
            'enable_chassis': LaunchConfiguration('enable_chassis'),
            'target_frame': 'base_link',
            'publish_rate': LaunchConfiguration('publish_rate'),
            'cloud_skip': LaunchConfiguration('cloud_skip'),
            'image_scale': LaunchConfiguration('image_scale'),
            'depth_max': 5.0,
            'enable_outlier_filter': True,
            'outlier_radius': 0.05,
            'outlier_min_neighbors': 5,
            'enable_depth_denoise': True,
            'depth_denoise_ksize': 5,
            'lidar_topic': '/rslidar_points',
            'lidar_min_range': 0.0,
            'lidar_max_range': 10.0,
            'lidar_skip': 1,
            'lidar_publish_rate': 10.0,
            'extrinsics_dir': config_dir,
        }],
    )

    return LaunchDescription([
        enable_top_arg,
        enable_chassis_arg,
        publish_rate_arg,
        cloud_skip_arg,
        image_scale_arg,
        multi_camera_rviz_node,
    ])
