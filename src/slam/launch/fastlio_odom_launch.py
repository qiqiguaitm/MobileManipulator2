"""FAST-LIO 里程计 launch（严格仿照 ROS1 fastlio_odom.launch）

输出 /fastlio/odom 及 camera_init->body TF；odom->camera_init 由静态 TF 提供。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument('rviz', default_value='false')
    config_file_arg = DeclareLaunchArgument('config_file', default_value='helios16p.yaml')

    fast_lio_share = FindPackageShare('fast_lio')
    config_path = PathJoinSubstitution([fast_lio_share, 'config', LaunchConfiguration('config_file')])
    rviz_config = PathJoinSubstitution([fast_lio_share, 'rviz_cfg', 'loam_livox.rviz'])

    # 与 ROS1 一致：话题重映射 + 室内密集优化参数（与 navigation_3d 中 fastlio_odom.launch 一致）
    fastlio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fastlio_mapping',
        output='screen',
        parameters=[
            config_path,
            {
                'feature_extract_enable': False,
                'point_filter_num': 1,
                'max_iteration': 8,
                'filter_size_surf': 0.1,
                'filter_size_map': 0.1,
                'cube_side_length': 500.0,
                'runtime_pos_log_enable': False,
            },
        ],
        remappings=[
            ('/lidar/chassis/point_cloud', '/rslidar_points'),
            ('/Odometry', '/fastlio/odom'),
            ('/cloud_registered', '/fastlio/cloud_registered'),
            ('/cloud_registered_body', '/fastlio/cloud_body'),
            ('/path', '/fastlio/path'),
        ],
    )

    odom_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='fastlio_odom_frame',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'odom', '--child-frame-id', 'camera_init']
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('rviz')),
        package='rviz2',
        executable='rviz2',
        name='fastlio_rviz',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        rviz_arg, config_file_arg,
        fastlio_node, odom_tf_node, rviz_node,
    ])
