#!/usr/bin/env python3
"""
Perception 3D RViz Launch File (ROS2)

启动场景感知节点 + RViz 可视化节点。
支持性能模式参数控制可视化质量和资源消耗。

使用方法:
    ros2 launch perception perception_3d_rviz.launch.py
    ros2 launch perception perception_3d_rviz.launch.py perf_mode:=3
    ros2 launch perception perception_3d_rviz.launch.py rviz:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception')
    config_dir = os.path.join(bringup_share, 'config')
    rviz_dir = os.path.join(bringup_share, 'rviz')

    # ==================== Launch Arguments ====================

    # 性能模式: 0=禁用, 1=节能模式(1Hz), 3=高性能模式(3Hz)
    perf_mode_arg = DeclareLaunchArgument(
        'perf_mode', default_value='1',
        description='Performance mode: 0=disabled, 1=power_saving(default), 3=high_performance'
    )
    camera_name_arg = DeclareLaunchArgument(
        'camera_name', default_value='top',
        description='Camera name (top, chassis, hand)'
    )
    target_frame_arg = DeclareLaunchArgument(
        'target_frame', default_value='base_link',
        description='Target coordinate frame'
    )
    publish_rgb_cloud_arg = DeclareLaunchArgument(
        'publish_rgb_cloud', default_value='true',
        description='Publish RGB point cloud'
    )
    depth_max_arg = DeclareLaunchArgument(
        'depth_max', default_value='5.0',
        description='Maximum depth (m)'
    )
    depth_alpha_arg = DeclareLaunchArgument(
        'depth_alpha', default_value='0.7',
        description='EMA temporal smoothing alpha (0.7=fast, 0.3=smooth)'
    )

    # 检测参数
    auto_detect_rate_arg = DeclareLaunchArgument(
        'auto_detect_rate', default_value='1.0',
        description='Auto detection rate in Hz (0=disable)'
    )
    enable_depth_optimizer_arg = DeclareLaunchArgument(
        'enable_depth_optimizer', default_value='true',
        description='Enable CDM depth optimization'
    )

    # LiDAR 参数
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic', default_value='/rslidar_points',
        description='LiDAR point cloud topic'
    )
    lidar_min_range_arg = DeclareLaunchArgument(
        'lidar_min_range', default_value='0.0',
        description='LiDAR min range (m) - blue'
    )
    lidar_max_range_arg = DeclareLaunchArgument(
        'lidar_max_range', default_value='10.0',
        description='LiDAR max range (m) - red'
    )

    # RViz
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz'
    )

    # ==================== Nodes ====================

    # Scene Perception 3D 节点 (检测节点 - 必须!)
    scene_perception_3d_node = Node(
        package='perception',
        executable='scene_perception_3d_node',
        name='scene_perception_3d',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            os.path.join(config_dir, 'scene_perception_3d.yaml'),
            {
                'camera_name': LaunchConfiguration('camera_name'),
                'target_frame': LaunchConfiguration('target_frame'),
                'auto_detect_rate': LaunchConfiguration('auto_detect_rate'),
                'enable_depth_optimizer': LaunchConfiguration('enable_depth_optimizer'),
                'extrinsics_dir': config_dir,
            }
        ],
    )

    # Perception RViz 节点 (可视化)
    perception_rviz_node = Node(
        package='perception',
        executable='perception_rviz_node',
        name='perception_rviz_node',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[{
            'camera_name': LaunchConfiguration('camera_name'),
            'target_frame': LaunchConfiguration('target_frame'),
            'extrinsics_dir': config_dir,
            'publish_rgb_cloud': LaunchConfiguration('publish_rgb_cloud'),
            'depth_max': LaunchConfiguration('depth_max'),
            'depth_alpha': LaunchConfiguration('depth_alpha'),
            'lidar_topic': LaunchConfiguration('lidar_topic'),
            'lidar_min_range': LaunchConfiguration('lidar_min_range'),
            'lidar_max_range': LaunchConfiguration('lidar_max_range'),
            # 性能模式自适应参数
            'publish_rate': PythonExpression([
                '0.0 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(1.0 if ', LaunchConfiguration('perf_mode'), '==1 else 2.5)'
            ]),
            'image_scale': PythonExpression([
                '0.1 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(0.25 if ', LaunchConfiguration('perf_mode'), '==1 else 0.4)'
            ]),
            'cloud_skip': PythonExpression([
                '10 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(3 if ', LaunchConfiguration('perf_mode'), '==1 else 1)'
            ]),
            'lidar_skip': PythonExpression([
                '10 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(4 if ', LaunchConfiguration('perf_mode'), '==1 else 2)'
            ]),
            'lidar_publish_rate': PythonExpression([
                '0.0 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(3.0 if ', LaunchConfiguration('perf_mode'), '==1 else 8.0)'
            ]),
        }],
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(rviz_dir, 'perception_3d.rviz')],
        output='screen',
    )

    # LiDAR TF (lidar_link -> rslidar)
    lidar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_link_to_rslidar',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'rslidar'],
    )

    # 基础 TF (确保 base_link 存在)
    base_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_base_link',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
    )

    return LaunchDescription([
        # Arguments
        perf_mode_arg,
        camera_name_arg,
        target_frame_arg,
        publish_rgb_cloud_arg,
        depth_max_arg,
        depth_alpha_arg,
        auto_detect_rate_arg,
        enable_depth_optimizer_arg,
        lidar_topic_arg,
        lidar_min_range_arg,
        lidar_max_range_arg,
        rviz_arg,

        # Nodes
        base_tf_node,              # 基础 TF (map -> base_link)
        lidar_tf_node,             # LiDAR TF (lidar_link -> rslidar)
        scene_perception_3d_node,  # 检测节点
        perception_rviz_node,      # 可视化节点
        rviz_node,
    ])
