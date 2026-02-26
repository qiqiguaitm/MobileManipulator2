#!/usr/bin/env python3
"""Launch file for Piper grasp node"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('piper_grasp')
    default_config = os.path.join(pkg_share, 'config', 'piper_grasp_node.yaml')

    # Declare arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to configuration file'
    )

    # Node
    piper_grasp_node = Node(
        package='piper_grasp',
        executable='piper_grasp_node.py',
        name='piper_grasp_node',
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('config_file')}]
    )

    return LaunchDescription([
        config_file_arg,
        piper_grasp_node,
    ])
