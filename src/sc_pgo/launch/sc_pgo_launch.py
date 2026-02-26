"""
SC-PGO Launch File for ROS2
Scan Context Pose Graph Optimization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取包路径
    pkg_share = get_package_share_directory('sc_pgo')
    config_file = os.path.join(pkg_share, 'config', 'sc_pgo_params.yaml')

    # 声明参数
    declare_config = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to SC-PGO config file'
    )

    # SC-PGO节点
    sc_pgo_node = Node(
        package='sc_pgo',
        executable='sc_pgo_node',
        name='sc_pgo',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        # 话题已在代码中直接订阅正确的话题名
        # 如需remap可在此添加
    )

    return LaunchDescription([
        declare_config,
        sc_pgo_node,
    ])
