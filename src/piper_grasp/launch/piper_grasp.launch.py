#!/usr/bin/env python3
"""
Piper Grasp Full Test Launch File

One-click launch for testing piper grasp system:
- Hand camera (RealSense D435)
- Piper arm driver
- Perception grasp nodes
- RViz visualization

Usage:
    ros2 launch piper_grasp piper_grasp.launch.py
    ros2 launch piper_grasp piper_grasp.launch.py rviz:=false
    ros2 launch piper_grasp piper_grasp.launch.py can_port:=can1
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _is_package_available(package_name: str) -> bool:
    """Check if ROS2 package is available"""
    try:
        get_package_share_directory(package_name)
        return True
    except PackageNotFoundError:
        return False


def generate_launch_description():
    # ==================== Package Directories ====================
    pkg_share = get_package_share_directory('piper_grasp')
    rviz_config = os.path.join(pkg_share, 'config', 'piper_grasp_ros2.rviz')
    node_config = os.path.join(pkg_share, 'config', 'piper_grasp_node.yaml')

    # ==================== Launch Arguments ====================
    # Arm config
    can_port_arg = DeclareLaunchArgument(
        'can_port', default_value='can0',
        description='CAN port for Piper arm'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable', default_value='true',
        description='Auto enable arm on startup'
    )

    # Camera config
    hand_camera_arg = DeclareLaunchArgument(
        'hand_camera', default_value='true',
        description='Enable hand camera'
    )
    hand_serial_arg = DeclareLaunchArgument(
        'hand_serial_no', default_value="'337122071190'",
        description='Hand camera serial number (D435)'
    )

    # Perception config
    perception_arg = DeclareLaunchArgument(
        'perception', default_value='true',
        description='Enable perception nodes'
    )
    perf_mode_arg = DeclareLaunchArgument(
        'perf_mode', default_value='1',
        description='Performance mode: 0=disabled, 1=power_saving, 3=high_performance'
    )

    # Visualization config
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz'
    )

    # ==================== Hand Camera ====================
    hand_camera_launch = None
    if _is_package_available('realsense2_camera'):
        realsense_dir = get_package_share_directory('realsense2_camera')
        hand_camera_launch = TimerAction(
            period=1.0,  # Small delay for system init
            actions=[
                GroupAction(
                    condition=IfCondition(LaunchConfiguration('hand_camera')),
                    actions=[
                        IncludeLaunchDescription(
                            PythonLaunchDescriptionSource(
                                os.path.join(realsense_dir, 'launch', 'rs_launch.py')
                            ),
                            launch_arguments={
                                'camera_name': 'hand',
                                'camera_namespace': 'camera',
                                'serial_no': LaunchConfiguration('hand_serial_no'),
                                'initial_reset': 'true',
                                'enable_depth': 'true',
                                'enable_color': 'true',
                                'enable_infra1': 'false',
                                'enable_infra2': 'false',
                                'enable_gyro': 'false',
                                'enable_accel': 'false',
                                'depth_module.depth_profile': '640,480,15',
                                'rgb_camera.color_profile': '640,480,15',
                                'enable_sync': 'true',
                                'align_depth.enable': 'true',
                                'pointcloud.enable': 'false',
                                'publish_tf': 'true',
                            }.items()
                        )
                    ]
                )
            ]
        )

    # ==================== Piper Arm Driver ====================
    piper_ctrl_node = None
    piper_description_launch = None
    if _is_package_available('piper_driver'):
        piper_ctrl_node = Node(
            package='piper_driver',
            executable='piper_ctrl_single_node.py',
            name='piper_ctrl_single_node',
            output='screen',
            parameters=[{
                'can_port': LaunchConfiguration('can_port'),
                'auto_enable': LaunchConfiguration('auto_enable'),
                'gripper_val_mutiple': 2,
                'gripper_exist': True,
            }],
            remappings=[
                ('joint_ctrl_single', '/joint_states'),
            ]
        )

        # Include piper_description for URDF
        if _is_package_available('piper_description'):
            piper_description_share = get_package_share_directory('piper_description')
            display_launch_file = os.path.join(
                piper_description_share, 'launch', 'display_xacro.launch.py')
            if os.path.exists(display_launch_file):
                piper_description_launch = IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(display_launch_file)
                )

    # ==================== Perception Nodes ====================
    perception_nodes = []
    if _is_package_available('perception_nodes'):
        bringup_share = get_package_share_directory('perception_bringup')
        perception_config_dir = os.path.join(bringup_share, 'config')

        # Perception Grasp detection node
        perception_grasp_node = Node(
            condition=IfCondition(LaunchConfiguration('perception')),
            package='perception_nodes',
            executable='perception_grasp_node',
            name='perception_grasp_node',
            output='screen',
            respawn=True,
            respawn_delay=3.0,
            parameters=[{
                'trigger.realtime_mode.enabled': True,
                'trigger.realtime_mode.rate': PythonExpression([
                    '0.0 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                    '(1.0 if ', LaunchConfiguration('perf_mode'), '==1 else 3.0)'
                ]),
            }],
        )
        perception_nodes.append(perception_grasp_node)

        # Perception Grasp RViz visualization node
        perception_rviz_config = os.path.join(perception_config_dir, 'perception_grasp_rviz.yaml')
        perception_grasp_rviz_node = Node(
            condition=IfCondition(LaunchConfiguration('perception')),
            package='perception_nodes',
            executable='perception_grasp_rviz_node',
            name='perception_grasp_rviz_node',
            output='screen',
            respawn=True,
            respawn_delay=3.0,
            parameters=[
                perception_rviz_config if os.path.exists(perception_rviz_config) else {},
                {
                    'pointcloud_scale': PythonExpression([
                        '0.1 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                        '(0.25 if ', LaunchConfiguration('perf_mode'), '==1 else 0.4)'
                    ]),
                }
            ],
        )
        perception_nodes.append(perception_grasp_rviz_node)

    # ==================== Piper Grasp Node ====================
    piper_grasp_node = Node(
        package='piper_grasp',
        executable='piper_grasp_node.py',
        name='piper_grasp_node',
        output='screen',
        parameters=[{'config_file': node_config}] if os.path.exists(node_config) else []
    )

    # ==================== RViz ====================
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen',
    )

    # ==================== Build Launch Description ====================
    ld = LaunchDescription([
        # Arguments
        can_port_arg,
        auto_enable_arg,
        hand_camera_arg,
        hand_serial_arg,
        perception_arg,
        perf_mode_arg,
        rviz_arg,
    ])

    # Add piper description (URDF)
    if piper_description_launch is not None:
        ld.add_action(piper_description_launch)

    # Add piper arm driver
    if piper_ctrl_node is not None:
        ld.add_action(piper_ctrl_node)

    # Add hand camera
    if hand_camera_launch is not None:
        ld.add_action(hand_camera_launch)

    # Add perception nodes
    for node in perception_nodes:
        ld.add_action(node)

    # Add piper grasp node
    ld.add_action(piper_grasp_node)

    # Add RViz (delayed to allow other nodes to start first)
    ld.add_action(TimerAction(period=2.0, actions=[rviz_node]))

    return ld
