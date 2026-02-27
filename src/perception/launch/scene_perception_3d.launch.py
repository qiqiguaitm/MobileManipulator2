#!/usr/bin/env python3
"""
Scene Perception 3D Launch File (ROS2)

启动场景感知节点，支持可选的 Tracker 和可视化。

使用方法:
    ros2 launch perception scene_perception_3d.launch.py
    ros2 launch perception scene_perception_3d.launch.py enable_tracker:=true
    ros2 launch perception scene_perception_3d.launch.py enable_rviz_viz:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception')
    config_dir = os.path.join(bringup_share, 'config')

    # ==================== Launch Arguments ====================

    # 基础参数
    camera_name_arg = DeclareLaunchArgument(
        'camera_name', default_value='top',
        description='Camera name (top, chassis, hand)'
    )
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar', default_value='false',
        description='Enable LiDAR fusion'
    )
    target_frame_arg = DeclareLaunchArgument(
        'target_frame', default_value='base_link',
        description='Target coordinate frame (base_link, arm_base_link)'
    )
    auto_detect_rate_arg = DeclareLaunchArgument(
        'auto_detect_rate', default_value='1.0',
        description='Auto detection rate in Hz (0=disable)'
    )
    enable_depth_optimizer_arg = DeclareLaunchArgument(
        'enable_depth_optimizer', default_value='true',
        description='Enable CDM depth optimization'
    )

    # Tracker 参数
    enable_tracker_arg = DeclareLaunchArgument(
        'enable_tracker', default_value='false',
        description='Enable object tracker'
    )
    tracker_url_arg = DeclareLaunchArgument(
        'tracker_url', default_value='http://192.168.112.14:11086',
        description='SAM2 tracker service URL'
    )
    track_rate_arg = DeclareLaunchArgument(
        'track_rate', default_value='5.0',
        description='Tracking rate in Hz'
    )

    # 可视化参数
    enable_rviz_viz_arg = DeclareLaunchArgument(
        'enable_rviz_viz', default_value='false',
        description='Enable RViz visualization node'
    )

    # 外参目录 - 默认使用 perception/config
    bringup_share = get_package_share_directory('perception')
    default_extrinsics_dir = os.path.join(bringup_share, 'config')
    
    extrinsics_dir_arg = DeclareLaunchArgument(
        'extrinsics_dir', default_value=default_extrinsics_dir,
        description='Extrinsics config directory'
    )

    # ==================== Nodes ====================

    # Scene Perception 3D 节点
    # 使用 namespace 区分 chassis 和 top 实例
    camera_name = LaunchConfiguration('camera_name')
    
    scene_perception_3d_node = Node(
        package='perception',
        executable='scene_perception_3d_node',
        name='scene_perception_3d',
        namespace=camera_name,  # 使用 camera_name 作为 namespace
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            os.path.join(config_dir, 'scene_perception_3d.yaml'),
            {
                'camera_name': camera_name,
                'enable_lidar': LaunchConfiguration('enable_lidar'),
                'target_frame': LaunchConfiguration('target_frame'),
                'auto_detect_rate': LaunchConfiguration('auto_detect_rate'),
                'enable_depth_optimizer': LaunchConfiguration('enable_depth_optimizer'),
                'extrinsics_dir': LaunchConfiguration('extrinsics_dir'),
            }
        ],
    )

    # Object Tracker 节点 (可选)
    object_tracker_node = Node(
        package='perception',
        executable='object_tracker_node',
        name='object_tracker',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(LaunchConfiguration('enable_tracker')),
        parameters=[{
            'track_rate': LaunchConfiguration('track_rate'),
            'tracker_url': LaunchConfiguration('tracker_url'),
            'target_frame': LaunchConfiguration('target_frame'),
            'detection_topic': ['/', camera_name, '/scene_perception_3d/objects_3d'],
            'rgb_topic': ['/camera/', camera_name, '/color/image_raw'],
            'depth_topic': ['/camera/', camera_name, '/aligned_depth_to_color/image_raw'],
            'camera_info_topic': ['/camera/', camera_name, '/color/camera_info'],
            'extrinsics_dir': LaunchConfiguration('extrinsics_dir'),
        }],
    )

    # Perception Viz 节点 (可选)
    perception_viz_node = Node(
        package='perception',
        executable='perception_viz_node',
        name='perception_viz',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(LaunchConfiguration('enable_rviz_viz')),
        parameters=[{
            'camera_name': LaunchConfiguration('camera_name'),
            'target_frame': LaunchConfiguration('target_frame'),
            'detection_topic': ['/', camera_name, '/scene_perception_3d/objects_3d'],
            'tracking_topic': '/object_tracker/tracked_objects',
        }],
    )

    return LaunchDescription([
        # Arguments
        camera_name_arg,
        enable_lidar_arg,
        target_frame_arg,
        auto_detect_rate_arg,
        enable_depth_optimizer_arg,
        enable_tracker_arg,
        tracker_url_arg,
        track_rate_arg,
        enable_rviz_viz_arg,
        extrinsics_dir_arg,

        # Nodes
        scene_perception_3d_node,
        object_tracker_node,
        perception_viz_node,
    ])
