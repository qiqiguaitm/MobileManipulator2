#!/usr/bin/env python3
"""
Launch file for dual Piper arm (master/slave) setup
ROS2 version

This launch file starts both left and right arm nodes with proper topic remapping.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='1',
        description='Mode: 0=read both arms, 1=control slave arm'
    )

    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Auto enable arms on startup (only works in mode 1)'
    )

    # Left arm node
    left_arm_node = Node(
        package='piper_driver',
        executable='piper_start_ms_node.py',
        name='piper_left',
        output='screen',
        parameters=[{
            'can_port': 'left_piper',
            'mode': LaunchConfiguration('mode'),
            'auto_enable': LaunchConfiguration('auto_enable'),
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/left_joint_states'),
        ]
    )

    # Right arm node
    right_arm_node = Node(
        package='piper_driver',
        executable='piper_start_ms_node.py',
        name='piper_right',
        output='screen',
        parameters=[{
            'can_port': 'right_piper',
            'mode': LaunchConfiguration('mode'),
            'auto_enable': LaunchConfiguration('auto_enable'),
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/right_joint_states'),
        ]
    )

    return LaunchDescription([
        mode_arg,
        auto_enable_arg,
        left_arm_node,
        right_arm_node,
    ])
