#!/usr/bin/env python3
"""Launch cleaner_manager_node with config."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('cleaner_manager'),
        'config', 'cleaner_manager.yaml')

    return LaunchDescription([
        Node(
            package='cleaner_manager',
            executable='robot_manager_node.py',
            name='robot_manager_node',
            output='screen',
            parameters=[config],
        ),
    ])
