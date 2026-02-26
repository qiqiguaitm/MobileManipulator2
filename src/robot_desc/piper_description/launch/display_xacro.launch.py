#!/usr/bin/env python3
"""
Launch file for displaying Piper robot from XACRO in RViz2
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('piper_description')

    # Declare arguments
    xacro_file_arg = DeclareLaunchArgument(
        'xacro_file',
        default_value=os.path.join(pkg_share, 'urdf', 'piper_description.xacro'),
        description='Path to XACRO file'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'piper_ctrl.rviz'),
        description='Path to RViz config file'
    )

    # Process XACRO file
    xacro_file = os.path.join(pkg_share, 'urdf', 'piper_description.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = robot_description_config.toxml()

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 100.0
        }]
    )

    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )

    return LaunchDescription([
        xacro_file_arg,
        rviz_config_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz_node
    ])
