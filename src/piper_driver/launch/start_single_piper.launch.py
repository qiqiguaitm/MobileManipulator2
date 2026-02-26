#!/usr/bin/env python3
"""Launch file for single Piper arm control node"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='arm',
        description='CAN port name for arm communication'
    )

    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Auto-enable arm on startup'
    )

    gripper_val_mutiple_arg = DeclareLaunchArgument(
        'gripper_val_mutiple',
        default_value='1.0',
        description='Gripper value multiplier'
    )

    gripper_exist_arg = DeclareLaunchArgument(
        'gripper_exist',
        default_value='true',
        description='Whether gripper exists'
    )

    # Node
    piper_node = Node(
        package='piper_driver',
        executable='piper_ctrl_single_node.py',
        name='piper_ctrl_single_node',
        output='screen',
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_val_mutiple': LaunchConfiguration('gripper_val_mutiple'),
            'gripper_exist': LaunchConfiguration('gripper_exist'),
        }],
        remappings=[
            ('joint_ctrl_single', '/joint_states'),
            ('joint_states_single', '/joint_states'),
        ]
    )

    return LaunchDescription([
        can_port_arg,
        auto_enable_arg,
        gripper_val_mutiple_arg,
        gripper_exist_arg,
        piper_node,
    ])
