#!/usr/bin/env python3
"""
Launch file for single Piper arm with RViz visualization
ROS2 version
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare arguments
    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='CAN port name'
    )

    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Auto enable arm on startup'
    )

    # Include piper_description display launch (if available)
    # This provides URDF and robot_state_publisher
    piper_description_launch = None
    try:
        piper_description_share = get_package_share_directory('piper_description')
        display_launch_file = os.path.join(
            piper_description_share, 'launch', 'display_xacro.launch.py')
        if os.path.exists(display_launch_file):
            piper_description_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(display_launch_file)
            )
    except Exception:
        pass  # piper_description not available

    # Piper control node
    piper_ctrl_node = Node(
        package='piper_driver',
        executable='piper_ctrl_single_node.py',
        name='piper_ctrl_single_node',
        output='screen',
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_val_mutiple': 2,
            'gripper_exist': True,
        }],
        remappings=[
            ('joint_ctrl_single', '/joint_states'),
        ]
    )

    # Build launch description
    ld = LaunchDescription([
        can_port_arg,
        auto_enable_arg,
    ])

    # Add piper_description launch if available
    if piper_description_launch is not None:
        ld.add_action(piper_description_launch)

    ld.add_action(piper_ctrl_node)

    return ld
