#!/usr/bin/env python3
"""
robot_manager_full.launch.py — 全系统启动

启动顺序 (带延时保证依赖就绪):
  [0s]  Navigation (SLAM + Nav2 + 传感器 + TF + 底盘)
  [0s]  Camera driver (Top + Chassis + Hand)
  [8s]  Piper grasp node (CAN arm control)
  [10s] Perception (multi_camera + object_tracker)
  [15s] Robot Manager (编排层)
  [17s] RViz (可选)

前提:
  - CAN 已 bringup (make can-bringup-auto)
  - 外部服务已启动 (DINO-X / SAM3 / CDM)
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _pkg_dir(name):
    try:
        return get_package_share_directory(name)
    except PackageNotFoundError:
        return None


def generate_launch_description():
    # ==================== Directories ====================
    slam_dir = _pkg_dir('slam')
    perception_dir = _pkg_dir('perception')
    piper_grasp_dir = _pkg_dir('piper_grasp')
    camera_driver_dir = _pkg_dir('camera_driver')
    robot_manager_dir = _pkg_dir('robot_manager')

    # ==================== Arguments ====================
    args = [
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch combined RViz'),
        DeclareLaunchArgument('use_odom_fusion', default_value='true',
                              description='Enable wheel odometry fusion'),
        DeclareLaunchArgument('launch_chassis', default_value='true',
                              description='Launch chassis driver'),
        DeclareLaunchArgument('detector_type', default_value='sam3',
                              description='Perception detector: dinox or sam3'),
        DeclareLaunchArgument('extrinsics_suffix', default_value='_stereo_hand',
                              description='Extrinsics calibration suffix'),
        DeclareLaunchArgument('enable_depth_optimizer', default_value='true',
                              description='Enable CDM depth optimization'),
    ]

    # ==================== [0s] Navigation ====================
    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_dir, 'launch', 'hdl_navigation_launch.py')
        ),
        launch_arguments={
            'use_odom_fusion': LaunchConfiguration('use_odom_fusion'),
            'launch_chassis': LaunchConfiguration('launch_chassis'),
            'enable_rviz': 'false',  # 用自己的 rviz
            'use_raw_laserscan': 'false',
        }.items()
    )

    # ==================== [0s] Cameras (Top + Chassis + Hand) ====================
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(camera_driver_dir, 'launch', 'camera_driver.launch.py')
        ),
        launch_arguments={
            'top_enable': 'true',
            'hand_enable': 'true',
            'chassis_enable': 'true',
        }.items()
    )

    # ==================== [8s] Piper Grasp Node ====================
    piper_node = TimerAction(period=8.0, actions=[
        LogInfo(msg='[robot_manager_full] Starting piper_grasp_node...'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(piper_grasp_dir, 'launch', 'piper_grasp.launch.py')
            ),
            launch_arguments={
                'hand_camera': 'false',   # 已在上面启动
                'perception': 'false',    # 用 multi_camera 而非 grasp perception
                'rviz': 'false',          # 用自己的 rviz
                'can_port': 'can0',
            }.items()
        ),
    ])

    # ==================== [10s] Perception (multi_camera + tracker) ====================
    perception_launch = TimerAction(period=10.0, actions=[
        LogInfo(msg='[robot_manager_full] Starting perception...'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(perception_dir, 'launch', 'multi_camera_perception.launch.py')
            ),
            launch_arguments={
                'use_camera_driver': 'false',  # 已在上面启动
                'auto_detect_rate': '1.0',
                'detector_type': LaunchConfiguration('detector_type'),
                'extrinsics_suffix': LaunchConfiguration('extrinsics_suffix'),
                'enable_depth_optimizer': LaunchConfiguration('enable_depth_optimizer'),
            }.items()
        ),
    ])

    # ==================== [15s] Robot Manager ====================
    rm_config = os.path.join(robot_manager_dir, 'config', 'robot_manager.yaml')
    robot_manager_node = TimerAction(period=15.0, actions=[
        LogInfo(msg='[robot_manager_full] Starting robot_manager_node...'),
        Node(
            package='robot_manager',
            executable='robot_manager_node',
            name='robot_manager_node',
            output='screen',
            parameters=[rm_config],
        ),
    ])

    # ==================== [17s] RViz ====================
    rviz_config = os.path.join(robot_manager_dir, 'config', 'robot_manager.rviz')
    rviz_node = TimerAction(period=17.0, actions=[
        Node(
            condition=IfCondition(LaunchConfiguration('rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', rviz_config, '--ros-args', '--log-level', 'rviz:=warn'],
            output='screen',
        ),
    ])

    # ==================== Build ====================
    ld = LaunchDescription(args)
    ld.add_action(LogInfo(msg='[robot_manager_full] Starting full system...'))
    ld.add_action(nav_launch)
    ld.add_action(camera_launch)
    ld.add_action(piper_node)
    ld.add_action(perception_launch)
    ld.add_action(robot_manager_node)
    ld.add_action(rviz_node)
    return ld
