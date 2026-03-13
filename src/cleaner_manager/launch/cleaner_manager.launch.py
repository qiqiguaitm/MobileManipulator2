#!/usr/bin/env python3
"""
cleaner_manager.launch.py — Interactive manual control with RViz2 panels.

Launches ManualControlNode + ObjectSelectorNode + RViz2 (with CleanerPanel,
PerceptionPanel, PiperGraspPanel embedded).

Assumes navigation, cameras, perception, and piper are already running.

Usage:
  ros2 launch cleaner_manager cleaner_manager.launch.py
  ros2 launch cleaner_manager cleaner_manager.launch.py rviz:=false
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    LogInfo,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('cleaner_manager')
    config = os.path.join(pkg_dir, 'config', 'cleaner_manager.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'cleaner_manager.rviz')

    args = [
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz2 with embedded panels'),
    ]

    manual_node = Node(
        package='cleaner_manager',
        executable='manual_control_node.py',
        name='manual_control_node',
        parameters=[config],
        output='screen',
    )

    object_selector = Node(
        package='cleaner_manager',
        executable='object_selector_node.py',
        name='object_selector_node',
        output='log',
    )

    display = os.environ.get('DISPLAY', ':1001')
    rviz_proc = TimerAction(period=2.0, actions=[
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('rviz')),
            cmd=['rviz2', '-d', rviz_config, '--ros-args', '--log-level', 'WARN',
                 '--log-level', 'rviz:=ERROR'],
            additional_env={'DISPLAY': display},
            output='screen',
        ),
    ])

    ld = LaunchDescription(args)
    ld.add_action(LogInfo(msg='[cleaner_manager] 启动手动控制 + RViz2 面板...'))
    ld.add_action(manual_node)
    ld.add_action(object_selector)
    ld.add_action(rviz_proc)
    return ld
