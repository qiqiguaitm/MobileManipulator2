#!/usr/bin/env python3
"""
Object Tracker Launch File (ROS2)

启动基于 SAM2TrackerOnline 的 2D 跟踪节点，结合深度测量获取 3D 位置。

使用方法:
    ros2 launch perception_bringup object_tracker.launch.py
    ros2 launch perception_bringup object_tracker.launch.py tracker_url:=http://localhost:11086
    ros2 launch perception_bringup object_tracker.launch.py track_rate:=10.0
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception_bringup')
    config_dir = os.path.join(bringup_share, 'config')

    # ==================== Launch Arguments ====================

    track_rate_arg = DeclareLaunchArgument(
        'track_rate', default_value='5.0',
        description='Tracking rate in Hz'
    )
    target_frame_arg = DeclareLaunchArgument(
        'target_frame', default_value='base_link',
        description='Target coordinate frame'
    )

    # SAM2 Tracker 服务地址
    tracker_url_arg = DeclareLaunchArgument(
        'tracker_url', default_value='http://192.168.112.14:11086',
        description='SAM2 tracker service URL'
    )

    # 相机话题
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic', default_value='/camera/top/color/image_raw',
        description='RGB image topic'
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/camera/top/aligned_depth_to_color/image_raw',
        description='Depth image topic (aligned to color)'
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic', default_value='/camera/top/color/camera_info',
        description='Camera info topic'
    )

    # 检测输入
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic', default_value='/scene_perception_3d/objects_3d',
        description='Detection input topic'
    )

    # 输出话题
    output_topic_arg = DeclareLaunchArgument(
        'output_topic', default_value='/object_tracker/tracked_objects',
        description='Tracked objects output topic'
    )

    # ==================== Nodes ====================

    object_tracker_node = Node(
        package='perception_nodes',
        executable='object_tracker_node',
        name='object_tracker',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[{
            'track_rate': LaunchConfiguration('track_rate'),
            'target_frame': LaunchConfiguration('target_frame'),
            'tracker_url': LaunchConfiguration('tracker_url'),
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'detection_topic': LaunchConfiguration('detection_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
        }],
    )

    return LaunchDescription([
        # Arguments
        track_rate_arg,
        target_frame_arg,
        tracker_url_arg,
        rgb_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        detection_topic_arg,
        output_topic_arg,

        # Nodes
        object_tracker_node,
    ])
