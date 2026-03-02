#!/usr/bin/env python3
"""
双相机棋盘格采集 Launch 文件

使用方法:
    ros2 launch perception stereo_chessboard_capture.launch.py
    
    # 指定输出目录
    ros2 launch perception stereo_chessboard_capture.launch.py output_dir:=/path/to/save
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 参数声明
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='',
        description='输出目录（空则自动生成）'
    )
    
    sync_slop_arg = DeclareLaunchArgument(
        'sync_slop',
        default_value='0.05',
        description='时间同步容差（秒）'
    )
    
    # 采集节点
    capture_node = Node(
        package='perception',
        executable='stereo_chessboard_capture',
        name='stereo_chessboard_capture',
        output='screen',
        parameters=[{
            'top_topic': '/camera/top/color/image_raw',
            'chassis_topic': '/camera/chassis/color/image_raw',
            'output_dir': LaunchConfiguration('output_dir'),
            'sync_slop': LaunchConfiguration('sync_slop'),
        }]
    )
    
    return LaunchDescription([
        output_dir_arg,
        sync_slop_arg,
        capture_node,
    ])
