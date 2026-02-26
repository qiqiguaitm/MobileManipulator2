#!/usr/bin/env python3
"""
Perception Grasp Launch File (ROS2)

启动抓取感知节点和可选的可视化节点。

使用方法:
    ros2 launch perception_bringup perception_grasp.launch.py
    ros2 launch perception_bringup perception_grasp.launch.py enable_rviz:=true
    ros2 launch perception_bringup perception_grasp.launch.py realtime_mode:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception_bringup')
    config_dir = os.path.join(bringup_share, 'config')

    # ==================== Launch Arguments ====================

    realtime_mode_arg = DeclareLaunchArgument(
        'realtime_mode', default_value='true',
        description='Enable realtime detection mode'
    )
    rate_arg = DeclareLaunchArgument(
        'rate', default_value='1.0',
        description='Realtime detection rate (Hz)'
    )
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz', default_value='false',
        description='Enable RViz visualization node'
    )
    grasp_server_list_arg = DeclareLaunchArgument(
        'grasp_server_list', default_value='',
        description='Path to grasp server list JSON file'
    )

    # ==================== Nodes ====================

    # Perception Grasp Node
    perception_grasp_node = Node(
        package='perception_nodes',
        executable='perception_grasp_node',
        name='perception_grasp_node',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            os.path.join(config_dir, 'perception_grasp.yaml'),
            {
                'trigger.realtime_mode.enabled': LaunchConfiguration('realtime_mode'),
                'trigger.realtime_mode.rate': LaunchConfiguration('rate'),
                'services.grasp.server_list': LaunchConfiguration('grasp_server_list'),
            }
        ],
    )

    # Perception Grasp RViz Node (可选)
    perception_grasp_rviz_node = Node(
        package='perception_nodes',
        executable='perception_grasp_rviz_node',
        name='perception_grasp_rviz_node',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
        parameters=[
            os.path.join(config_dir, 'perception_grasp.yaml'),
        ],
    )

    return LaunchDescription([
        # Arguments
        realtime_mode_arg,
        rate_arg,
        enable_rviz_arg,
        grasp_server_list_arg,

        # Nodes
        perception_grasp_node,
        perception_grasp_rviz_node,
    ])
