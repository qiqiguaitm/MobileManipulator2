#!/usr/bin/env python3
"""
Approach Navigator Launch File

Launches the three-stage approach navigator node with configuration.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('approach_navigator')

    # Config file path
    config_file = os.path.join(pkg_dir, 'config', 'approach_navigator.yaml')

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to configuration file'
    )

    # Approach Navigator Node
    approach_navigator_node = Node(
        package='approach_navigator',
        executable='approach_navigator_node.py',
        name='approach_navigator',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        approach_navigator_node,
    ])
