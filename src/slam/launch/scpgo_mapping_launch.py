"""
FAST-LIO2 + SC-PGO + Octomap 建图 Launch File
一键启动完整建图流程

Usage:
  ros2 launch slam scpgo_mapping_launch.py
  ros2 launch slam scpgo_mapping_launch.py enable_rviz:=true
  ros2 launch slam scpgo_mapping_launch.py enable_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ==================== 参数声明 ====================
    # 使用 'enable_rviz' 避免与子launch文件中的 'rviz' 参数冲突
    declare_rviz = DeclareLaunchArgument(
        'enable_rviz', default_value='true',
        description='Launch RViz visualization for mapping'
    )

    declare_save_dir = DeclareLaunchArgument(
        'save_dir',
        default_value='/home/didi/workspace/MobileManipulator2/maps/sc_pgo/',
        description='Directory to save map'
    )

    # ==================== 包路径 ====================
    slam_pkg = get_package_share_directory('slam')
    sc_pgo_pkg = get_package_share_directory('sc_pgo')
    rslidar_pkg = get_package_share_directory('rslidar_sdk')
    hipnuc_pkg = get_package_share_directory('hipnuc_imu')

    # ==================== 传感器驱动 ====================
    # LiDAR
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rslidar_pkg, 'launch', 'start.py')
        ),
        launch_arguments={'rviz': 'false'}.items()
    )

    # IMU
    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hipnuc_pkg, 'launch', 'imu_spec_msg.launch.py')
        )
    )

    # ==================== FAST_LIO ====================
    fastlio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'fastlio_odom_launch.py')
        ),
        launch_arguments={'rviz': 'false'}.items()
    )

    # ==================== SC-PGO ====================
    sc_pgo_config = os.path.join(sc_pgo_pkg, 'config', 'sc_pgo_params.yaml')

    sc_pgo_node = Node(
        package='sc_pgo',
        executable='sc_pgo_node',
        name='sc_pgo',
        output='screen',
        parameters=[sc_pgo_config],
        remappings=[
            ('/Odometry', '/fastlio/odom'),
            ('/cloud_registered_body', '/fastlio/cloud_body'),
        ],
    )

    # ==================== Octomap (如果已安装) ====================
    # 注意: 需要先安装 ros-humble-octomap-server
    octomap_config = os.path.join(slam_pkg, 'config', 'octomap_params.yaml')

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[octomap_config],
        remappings=[
            ('cloud_in', '/aft_pgo_map'),
        ]
    )

    # ==================== RViz ====================
    rviz_config = os.path.join(slam_pkg, 'rviz', 'scpgo_mapping.rviz')

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz_mapping',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # ==================== 返回 ====================
    return LaunchDescription([
        # 参数
        declare_rviz,
        declare_save_dir,

        # 传感器
        lidar_launch,
        imu_launch,

        # SLAM
        fastlio_launch,
        sc_pgo_node,

        # 地图
        # octomap_node,  # 取消注释以启用Octomap

        # 可视化
        rviz_node,
    ])
