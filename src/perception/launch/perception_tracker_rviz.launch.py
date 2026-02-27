#!/usr/bin/env python3
"""
Perception Tracker RViz Launch File (ROS2)

启动完整的感知 + 跟踪 + RViz 可视化管道。
使用统一的 perception_3d.rviz 配置。

使用方法:
    ros2 launch perception perception_tracker_rviz.launch.py
    ros2 launch perception perception_tracker_rviz.launch.py enable_lidar:=true
    ros2 launch perception perception_tracker_rviz.launch.py track_rate:=10.0
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception')
    rviz_dir = os.path.join(bringup_share, 'rviz')

    # ==================== Launch Arguments ====================

    camera_name_arg = DeclareLaunchArgument(
        'camera_name', default_value='top',
        description='Camera name (top, chassis, hand)'
    )
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar', default_value='false',
        description='Enable LiDAR fusion'
    )
    enable_tracker_arg = DeclareLaunchArgument(
        'enable_tracker', default_value='true',
        description='Enable object tracker'
    )
    tracker_url_arg = DeclareLaunchArgument(
        'tracker_url', default_value='http://192.168.112.14:11086',
        description='SAM2 tracker service URL'
    )
    track_rate_arg = DeclareLaunchArgument(
        'track_rate', default_value='5.0',
        description='Tracking rate in Hz'
    )
    target_frame_arg = DeclareLaunchArgument(
        'target_frame', default_value='base_link',
        description='Target coordinate frame'
    )

    # ==================== Included Launches ====================

    # 包含 scene_perception_3d.launch.py (带 tracker 和 viz)
    scene_perception_3d_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'scene_perception_3d.launch.py')
        ),
        launch_arguments={
            'camera_name': LaunchConfiguration('camera_name'),
            'enable_lidar': LaunchConfiguration('enable_lidar'),
            'target_frame': LaunchConfiguration('target_frame'),
            'enable_tracker': LaunchConfiguration('enable_tracker'),
            'tracker_url': LaunchConfiguration('tracker_url'),
            'track_rate': LaunchConfiguration('track_rate'),
            'enable_rviz_viz': 'true',
        }.items(),
    )

    # RViz (统一配置)
    rviz_config = os.path.join(rviz_dir, 'perception_3d.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        camera_name_arg,
        enable_lidar_arg,
        enable_tracker_arg,
        tracker_url_arg,
        track_rate_arg,
        target_frame_arg,

        # Included Launches
        scene_perception_3d_launch,

        # RViz
        rviz_node,
    ])
