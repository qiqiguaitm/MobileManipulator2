import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 地图文件路径
    default_map = '/home/didi/workspace/MobileManipulator2/maps/sc_pgo/map.yaml'

    map_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # PointCloud to LaserScan - 3D点云转2D激光
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            remappings=[
                ('cloud_in', '/cloud_registered_body'),  # body frame的点云
                ('scan', '/scan')
            ],
            parameters=[{
                'target_frame': '',  # 使用点云自带的frame
                'transform_tolerance': 0.1,
                'min_height': 0.05,
                'max_height': 1.0,
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.00872,  # ~0.5度
                'scan_time': 0.1,
                'range_min': 0.2,
                'range_max': 20.0,
                'use_inf': True,
            }]
        ),

        # Map Server - 发布地图
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_file,
                'use_sim_time': use_sim_time
            }]
        ),

        # AMCL - 2D定位
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
                'base_frame_id': 'body',
                'odom_frame_id': 'odom',
                'global_frame_id': 'map',
                'scan_topic': '/scan',
                'min_particles': 500,
                'max_particles': 2000,
                'update_min_d': 0.1,
                'update_min_a': 0.2,
                'resample_interval': 1,
                'transform_tolerance': 1.0,
                'recovery_alpha_slow': 0.001,
                'recovery_alpha_fast': 0.1,
                'laser_model_type': 'likelihood_field',
                'laser_max_range': 20.0,
                'laser_min_range': 0.1,
                'max_beams': 60,
                'z_hit': 0.95,
                'z_rand': 0.05,
                'sigma_hit': 0.2,
                'initial_pose_x': 0.0,
                'initial_pose_y': 0.0,
                'initial_pose_yaw': 0.0,
            }]
        ),

        # Lifecycle Manager - 管理节点生命周期
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['map_server', 'amcl']
            }]
        ),
    ])
