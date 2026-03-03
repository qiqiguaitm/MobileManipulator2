#!/usr/bin/env python3
"""Launch robot_manager_node with config."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('robot_manager'),
        'config', 'robot_manager.yaml')

    return LaunchDescription([
        Node(
            package='robot_manager',
            executable='robot_manager_node',
            name='robot_manager_node',
            output='screen',
            parameters=[config],
        ),
    ])
