from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('rslidar_sdk')
    rviz_config = os.path.join(pkg_share, 'rviz', 'rviz2.rviz')
    config_file = os.path.join(pkg_share, 'config', 'config.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RViz2'),

        Node(
            namespace='rslidar_sdk',
            package='rslidar_sdk',
            executable='rslidar_sdk_node',
            output='screen',
            parameters=[{'config_path': config_file}]
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('rviz')),
            namespace='rviz2',
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config]
        )
    ])
