"""Launch the ROS 2 global localization service with the package configuration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('hdl_global_localization'), 'config'
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'config_dir', default_value=config_dir,
            description='Directory containing the engine parameter files',
        ),
        DeclareLaunchArgument(
            'general_config',
            default_value=PathJoinSubstitution([
                LaunchConfiguration('config_dir'), 'general_config.yaml'
            ]),
            description='General parameters, including engine and ScanContext database',
        ),
        Node(
            package='hdl_global_localization',
            executable='hdl_global_localization_node',
            # Match the node key in the existing ROS 2 parameter files.
            name='hdl_global_localization',
            namespace='/',
            output='screen',
            parameters=[
                LaunchConfiguration('general_config'),
                *[
                    PathJoinSubstitution([
                        LaunchConfiguration('config_dir'), filename
                    ])
                    for filename in (
                        'bbs_config.yaml', 'fpfh_config.yaml',
                        'ransac_config.yaml', 'teaser_config.yaml',
                    )
                ],
                {'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool
                )},
            ],
            # The local localization client uses these root-level services.
            remappings=[
                ('set_engine', '/set_engine'),
                ('set_global_map', '/set_global_map'),
                ('query', '/query'),
            ],
        ),
    ])
