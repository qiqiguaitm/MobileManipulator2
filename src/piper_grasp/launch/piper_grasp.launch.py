#!/usr/bin/env python3
"""
Piper Grasp Launch File - ROS2 version

Equivalent to ROS1 launch files:
- piper_grasp_node.launch   → perception:=false hand_camera:=false rviz:=false
- piper_grasp_rviz.launch   → perception:=false hand_camera:=false
- piper_grasp_full.launch   → (default, or rviz:=false)
- demo_grasp.launch         → run_demo:=true

Features:
- Hand camera (RealSense D435)
- Perception grasp nodes
- Piper grasp node
- RViz visualization
- Demo client (optional)
- Fake joints for testing without hardware

Usage:
    # Full system (equivalent to piper_grasp_full.launch + rviz)
    ros2 launch piper_grasp piper_grasp.launch.py

    # Node only (equivalent to piper_grasp_node.launch)
    ros2 launch piper_grasp piper_grasp.launch.py perception:=false hand_camera:=false rviz:=false

    # RViz only (equivalent to piper_grasp_rviz.launch)
    ros2 launch piper_grasp piper_grasp.launch.py perception:=false hand_camera:=false

    # RViz with fake joints (no hardware)
    ros2 launch piper_grasp piper_grasp.launch.py use_fake_joints:=true

    # Demo mode (equivalent to demo_grasp.launch run_demo:=true)
    ros2 launch piper_grasp piper_grasp.launch.py run_demo:=true demo_prompt:=bottle.cup
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
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, AndSubstitution, NotSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import xacro


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
    default_rviz_config = os.path.join(pkg_share, 'config', 'piper_grasp_ros2.rviz')
    node_config = os.path.join(pkg_share, 'config', 'piper_grasp_node.yaml')

    # ==================== Launch Arguments ====================
    # Arm config (matching ROS1 piper_grasp_node.launch)
    can_port_arg = DeclareLaunchArgument(
        'can_port', default_value='can0',
        description='CAN port for Piper arm'
    )
    auto_connect_arg = DeclareLaunchArgument(
        'auto_connect', default_value='true',
        description='Auto connect to arm on startup'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable', default_value='true',
        description='Auto enable arm on startup'
    )
    enable_grasp_arg = DeclareLaunchArgument(
        'enable_grasp', default_value='true',
        description='Enable grasp functionality'
    )
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='INFO',
        description='Logging level: DEBUG, INFO, WARN, ERROR, FATAL'
    )

    # Camera config - parameters forwarded to camera_driver.launch.py
    hand_camera_arg = DeclareLaunchArgument(
        'hand_camera', default_value='true',
        description='Enable hand camera (via camera_driver)'
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

    # Visualization config (matching ROS1 piper_grasp_rviz.launch)
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz_config,
        description='RViz config file path'
    )
    use_fake_joints_arg = DeclareLaunchArgument(
        'use_fake_joints', default_value='false',
        description='Use fake joint states for testing without hardware (no real driver)'
    )

    # Demo config (matching ROS1 demo_grasp.launch)
    run_demo_arg = DeclareLaunchArgument(
        'run_demo', default_value='false',
        description='Run demo client for single grasp'
    )
    demo_prompt_arg = DeclareLaunchArgument(
        'demo_prompt', default_value='bottle.cup.box.can.toy.pen.bag',
        description='Demo target object prompt'
    )
    demo_place_after_arg = DeclareLaunchArgument(
        'demo_place_after', default_value='true',
        description='Place object after picking in demo mode'
    )

    # ==================== Hand Camera (via camera_driver) ====================
    # Delegate to camera_driver.launch.py - single source of truth for camera params
    # (resolution, fps, serial_no, etc. all configured in camera_driver)
    hand_camera_launch = None
    if _is_package_available('camera_driver'):
        camera_driver_dir = get_package_share_directory('camera_driver')
        hand_camera_launch = GroupAction(
            condition=IfCondition(LaunchConfiguration('hand_camera')),
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(camera_driver_dir, 'launch', 'camera_driver.launch.py')
                    ),
                    launch_arguments={
                        'hand_enable': 'true',
                    }.items()
                )
            ]
        )

    # ==================== Piper Arm Driver ====================
    # NOTE: piper_ctrl_single_node is NOT needed here because piper_grasp_node
    # already contains PiperAPI which directly controls the arm via CAN.
    # Running both would cause CAN bus conflicts!
    piper_ctrl_node = None

    # ==================== Robot State Publisher (URDF) ====================
    # NOTE: 禁用 robot_state_publisher！因为导航系统 (make navi-fusion) 已经使用
    # mobile_manipulator2_description.urdf 发布了完整的机械臂 TF (arm_base, link1-8)。
    # 如果同时启动两个 RSP，会导致 TF 冲突：arm_base/link1-link8 被竞争发布，
    # 导致 RViz 机器人模型异常、夹爪 TF 错误无法抓取。
    # 夹爪 (gripper_base) 的 TF 单独通过 static_transform_publisher 发布。
    robot_state_publisher_node = None
    joint_state_publisher_node = None
    robot_description = None

    # Gripper base to link8 static transform (fixed joint from piper_description.xacro)
    # <joint name="joint6_to_gripper_base" type="fixed">
    #   <origin xyz="0 0 0" rpy="0 0 0"/>
    #   <parent link="link8"/>
    #   <child link="gripper_base"/>
    # </joint>
    gripper_base_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='link8_to_gripper_base',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'link8', '--child-frame-id', 'gripper_base',
        ],
    )

    # ==================== Perception Nodes ====================
    perception_nodes = []
    if _is_package_available('perception'):
        perception_share = get_package_share_directory('perception')
        perception_config_dir = os.path.join(perception_share, 'config')

        # Perception Grasp detection node
        perception_grasp_config = os.path.join(perception_config_dir, 'perception_grasp.yaml')
        perception_grasp_node = Node(
            condition=IfCondition(LaunchConfiguration('perception')),
            package='perception',
            executable='perception_grasp_node',
            name='perception_grasp_node',
            output='screen',
            respawn=True,
            respawn_delay=3.0,
            parameters=[
                perception_grasp_config if os.path.exists(perception_grasp_config) else {},
                {
                    'trigger.realtime_mode.enabled': False,  # 关闭自动检测，只在 observe 调用时检测
                    'trigger.realtime_mode.rate': 1.0,       # 检测频率 1Hz
                },
            ],
        )
        perception_nodes.append(perception_grasp_node)

        # Perception Grasp RViz visualization node
        perception_rviz_config = os.path.join(perception_config_dir, 'perception_grasp_rviz.yaml')
        perception_grasp_rviz_node = Node(
            condition=IfCondition(LaunchConfiguration('perception')),
            package='perception',
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
    # piper_grasp_node.py loads config from config_file
    # Parameters match ROS1 piper_grasp_node.launch
    # Only launch when NOT using fake joints (real hardware mode)
    piper_grasp_node = Node(
        condition=UnlessCondition(LaunchConfiguration('use_fake_joints')),
        package='piper_grasp',
        executable='piper_grasp_node.py',
        name='piper_grasp_node',
        output='screen',
        parameters=[{
            'config_file': node_config,
            'can_port': LaunchConfiguration('can_port'),
            'auto_connect': LaunchConfiguration('auto_connect'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'enable_grasp': LaunchConfiguration('enable_grasp'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    # ==================== Demo Client (Optional) ====================
    # Equivalent to ROS1 demo_grasp.launch run_demo:=true
    # Delayed start to allow piper_grasp_node to initialize
    demo_client_node = TimerAction(
        period=5.0,  # 5 second delay (matching ROS1 sleep 5)
        actions=[
            Node(
                condition=IfCondition(LaunchConfiguration('run_demo')),
                package='piper_grasp',
                executable='grasp_demo_node.py',
                name='demo_client',
                output='screen',
                parameters=[{
                    'prompt': LaunchConfiguration('demo_prompt'),
                    'place_after': LaunchConfiguration('demo_place_after'),
                }],
            )
        ]
    )

    # ==================== RViz ====================
    # Use configurable rviz_config path (matching ROS1 piper_grasp_rviz.launch)
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', LaunchConfiguration('rviz_config'), '--ros-args', '--log-level', 'rviz:=warn'],
        output='screen',
    )


    # ==================== Build Launch Description ====================
    ld = LaunchDescription([
        # Arm config arguments
        can_port_arg,
        auto_connect_arg,
        auto_enable_arg,
        enable_grasp_arg,
        log_level_arg,
        # Camera config arguments
        hand_camera_arg,
        # Perception config arguments
        perception_arg,
        perf_mode_arg,
        # Visualization config arguments
        rviz_arg,
        rviz_config_arg,
        use_fake_joints_arg,
        # Demo config arguments
        run_demo_arg,
        demo_prompt_arg,
        demo_place_after_arg,
    ])

    # Add robot state publisher (URDF)
    # Add gripper base static transform (link8 -> gripper_base)
    # This is needed because mobile_manipulator2_description.urdf doesn't include gripper
    ld.add_action(gripper_base_tf_node)

    # Add fake joint state publisher (when use_fake_joints:=true)
    if joint_state_publisher_node is not None:
        ld.add_action(joint_state_publisher_node)

    # Add piper arm driver (not used - piper_grasp_node has its own PiperAPI)
    if piper_ctrl_node is not None:
        ld.add_action(piper_ctrl_node)

    # Add hand camera
    if hand_camera_launch is not None:
        ld.add_action(hand_camera_launch)

    # Add perception nodes
    for node in perception_nodes:
        ld.add_action(node)

    # Add piper grasp node (only when NOT using fake joints)
    ld.add_action(piper_grasp_node)

    # Add demo client (when run_demo:=true)
    ld.add_action(demo_client_node)

    # Add RViz (delayed to allow other nodes to start first)
    ld.add_action(TimerAction(period=2.0, actions=[rviz_node]))

    return ld
