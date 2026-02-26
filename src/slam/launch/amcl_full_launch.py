"""
2D AMCL 定位 (完整版): 传感器 + FAST-LIO里程计 + AMCL + RViz

Usage:
  ros2 launch slam amcl_full_launch.py
  ros2 launch slam amcl_full_launch.py rviz:=false
  ros2 launch slam amcl_full_launch.py map:=/path/to/map.yaml
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 参数
    default_map = '/home/didi/workspace/MobileManipulator2/maps/sc_pgo/map.yaml'

    fast_lio_share = FindPackageShare('fast_lio')
    slam_share = FindPackageShare('slam')
    robot_desc_share = FindPackageShare('mobile_manipulator2_description')

    config_path = PathJoinSubstitution([
        fast_lio_share, 'config', 'helios16p.yaml'
    ])

    rviz_config = PathJoinSubstitution([
        slam_share, 'rviz', 'amcl_localization.rviz'
    ])

    urdf_file = PathJoinSubstitution([
        robot_desc_share, 'urdf', 'mobile_manipulator2_description.urdf'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz'),

        # ========== 机器人模型 ==========
        # 使用xacro/urdf加载机器人描述
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            arguments=[urdf_file]
        ),

        # ========== 传感器 ==========
        # 雷达
        Node(
            package='rslidar_sdk',
            executable='rslidar_sdk_node',
            name='rslidar_sdk_node',
            output='screen',
        ),

        # IMU
        Node(
            package='hipnuc_imu',
            executable='talker',
            name='IMU_publisher',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'baud_rate': 115200,
                'frame_id': 'imu_link',
                'imu_topic': '/imu/data',
            }]
        ),

        # ========== FAST-LIO ==========
        Node(
            package='fast_lio',
            executable='fastlio_mapping',
            name='fastlio_mapping',
            output='screen',
            parameters=[config_path],
            remappings=[
                ('/lidar/chassis/point_cloud', '/rslidar_points'),
            ]
        ),

        # TF: odom -> camera_init
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_camera_init',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'odom', 'camera_init']
        ),

        # TF: body -> base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='body_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'body', 'base_link']
        ),

        # ========== RViz ==========
        Node(
            condition=IfCondition(LaunchConfiguration('rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            arguments=['-d', rviz_config]
        ),

        # ========== 定位 (延迟启动) ==========
        TimerAction(
            period=5.0,
            actions=[
                # PointCloud to LaserScan
                Node(
                    package='pointcloud_to_laserscan',
                    executable='pointcloud_to_laserscan_node',
                    name='pointcloud_to_laserscan',
                    output='screen',
                    remappings=[
                        ('cloud_in', '/cloud_registered_body'),
                        ('scan', '/scan')
                    ],
                    parameters=[{
                        'target_frame': 'body',
                        'transform_tolerance': 0.5,
                        'min_height': 0.05,
                        'max_height': 1.0,
                        'angle_min': -3.14159,
                        'angle_max': 3.14159,
                        'angle_increment': 0.00872,
                        'scan_time': 0.1,
                        'range_min': 0.2,
                        'range_max': 20.0,
                        'use_inf': True,
                    }]
                ),

                # Map Server
                Node(
                    package='nav2_map_server',
                    executable='map_server',
                    name='map_server',
                    output='screen',
                    parameters=[{
                        'yaml_filename': LaunchConfiguration('map'),
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'bond_timeout': 10.0,
                    }]
                ),

                # AMCL
                Node(
                    package='nav2_amcl',
                    executable='amcl',
                    name='amcl',
                    output='screen',
                    parameters=[{
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
                        'base_frame_id': 'base_link',
                        'odom_frame_id': 'odom',
                        'global_frame_id': 'map',
                        'scan_topic': '/scan',
                        'min_particles': 500,
                        'max_particles': 2000,
                        'update_min_d': 0.1,
                        'update_min_a': 0.2,
                        'resample_interval': 1,
                        'transform_tolerance': 1.0,
                        'laser_model_type': 'likelihood_field',
                        'laser_max_range': 20.0,
                        'laser_min_range': 0.1,
                        'max_beams': 60,
                        'z_hit': 0.95,
                        'z_rand': 0.05,
                        'sigma_hit': 0.2,
                        'bond_timeout': 10.0,
                    }]
                ),

                # Lifecycle Manager
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_localization',
                    output='screen',
                    parameters=[{
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'autostart': True,
                        'node_names': ['map_server', 'amcl'],
                        'bond_timeout': 10.0,
                        'attempt_respawn_reconnection': True,
                        'bond_respawn_max_duration': 10.0,
                    }]
                ),
            ]
        ),
    ])
