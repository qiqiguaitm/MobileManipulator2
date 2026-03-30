#!/usr/bin/env python3
"""
manual_control_full.launch.py — Full system for manual pick-and-place.

Sequential startup with readiness confirmation (mirrors ROS1 start_amr.sh):
  [Stage 0]  Navigation + Camera (immediate)
  [Gate  0]  Wait for /odom + camera topics
  [Stage 1]  Piper arm
  [Gate  1]  Wait for piper_grasp_node
  [Stage 2]  Perception
  [Gate  2]  Wait for multi_camera_perception
  [Stage 3]  Manual control + GUI panel
  [Gate  3]  Wait for manual_control_node
  [Stage 4]  RViz (LAST — all topics ready, immediate data display)

Terminal output policy:
  screen : stage gates, manual_control_node, rviz, ManualPanel
  log    : nav, camera, piper, perception  → ~/.ros/log/

Prerequisites:
  - CAN bringup done (make can-bringup-auto)
  - External services running (SAM3 / CDM)
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
    ExecuteProcess,
    SetEnvironmentVariable,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _pkg_dir(name):
    try:
        return get_package_share_directory(name)
    except PackageNotFoundError:
        return None


def _ready_gate(label, check_cmd, timeout=30):
    """Readiness gate: polls check_cmd each second, exits on success or timeout."""
    return ExecuteProcess(
        cmd=['bash', '-c',
             f'echo "  等待 {label}..."; '
             f'for i in $(seq 1 {timeout}); do '
             f'if {check_cmd}; then '
             f'echo "  [OK] {label}"; exit 0; fi; '
             f'sleep 1; done; '
             f'echo "  [WARN] {label} 超时 ({timeout}s)"; exit 0'],
        output='screen',
    )


def generate_launch_description():
    slam_dir          = _pkg_dir('slam')
    perception_dir    = _pkg_dir('perception')
    piper_grasp_dir   = _pkg_dir('piper_grasp')
    camera_driver_dir = _pkg_dir('camera_driver')
    cleaner_mgr_dir   = _pkg_dir('cleaner_manager')

    # ==================== Arguments ====================
    # --- Mode condition helpers ---
    mode = LaunchConfiguration('mode')
    is_manual = IfCondition(PythonExpression(["'", mode, "' == 'manual'"]))
    is_auto   = IfCondition(PythonExpression(["'", mode, "' == 'auto'"]))

    args = [
        DeclareLaunchArgument('mode', default_value='manual',
                              description="'manual' (手动控制) or 'auto' (自主清扫)"),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('use_odom_fusion', default_value='true'),
        DeclareLaunchArgument('launch_chassis', default_value='true'),
        DeclareLaunchArgument('detector_type', default_value='sam3',
                              description='sam3 or dinox'),
        DeclareLaunchArgument('extrinsics_suffix', default_value='',
                              description='Extrinsics file suffix'),
        DeclareLaunchArgument('depth_source', default_value='combined',
                              description="Depth source: 'combined' | 'foundation_stereo' | 'raw'"),
        DeclareLaunchArgument('perception_rviz', default_value='true',
                              description="启动感知可视化节点 (占~80% CPU, 自动模式可关闭)"),
        DeclareLaunchArgument('top_detect_rate', default_value='5.0',
                              description="顶部相机自动检测频率 (Hz), 0=禁用 (service模式设0)"),
        DeclareLaunchArgument('chassis_detect_rate', default_value='6.0',
                              description="底部相机自动检测频率 (Hz), 0=禁用 (service模式设0)"),
        DeclareLaunchArgument('fusion_publish_rate', default_value='5.0',
                              description="融合发布频率 (Hz), 0=禁用"),
    ]

    # ==================== Node / launch definitions ====================

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_dir, 'launch', 'hdl_navigation_launch.py')
        ),
        launch_arguments={
            'use_odom_fusion': LaunchConfiguration('use_odom_fusion'),
            'launch_chassis':  LaunchConfiguration('launch_chassis'),
            'enable_rviz':     'false',
            'use_raw_laserscan': 'false',
        }.items()
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(camera_driver_dir, 'launch', 'camera_driver.launch.py')
        ),
        launch_arguments={
            'top_enable':     'true',
            'hand_enable':    'true',
            'chassis_enable': 'true',
            'config_file':    "''",
            # initial_reset only affects top camera (hand/chassis hardcode true
            # matching ROS1). Set false here because Makefile usbreset already
            # cleans USB state before launching.
            'initial_reset':  'false',
            'camera_delay':   '8.0',  # hand at 8s, chassis at 16s (staggered, increased for USB stability)
            # Enable IR streams for all cameras (required for FoundationStereo stereo depth).
            'top_enable_infra':     'true',
            'chassis_enable_infra': 'true',
            'hand_enable_infra':    'true',
        }.items()
    )

    piper_config = os.path.join(piper_grasp_dir, 'config', 'piper_grasp_node.yaml')
    piper_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='link8_to_gripper_base',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'link8', '--child-frame-id', 'gripper_base',
        ],
        output='log',
    )
    piper_node = Node(
        package='piper_grasp',
        executable='piper_grasp_node.py',
        name='piper_grasp_node',
        parameters=[{
            'config_file':   piper_config,
            'can_port':      'can0',
            'auto_connect':  True,
            'auto_enable':   True,
            'enable_grasp':  True,
        }],
        output='log',
    )

    perception_config_dir = os.path.join(perception_dir, 'config')
    perception_node = Node(
        package='perception',
        executable='multi_camera_perception_node',
        name='multi_camera_perception',
        parameters=[
            {'enable_top':     True},
            {'enable_chassis': True},
            {'top_color_topic':    '/camera/top/color/image_raw'},
            {'top_depth_topic':    '/camera/top/aligned_depth_to_color/image_raw'},
            {'top_info_topic':     '/camera/top/color/camera_info'},
            {'chassis_color_topic':  '/camera/chassis/color/image_raw'},
            {'chassis_depth_topic':  '/camera/chassis/aligned_depth_to_color/image_raw'},
            {'chassis_info_topic':   '/camera/chassis/color/camera_info'},
            {'top_detect_rate':     LaunchConfiguration('top_detect_rate')},
            {'chassis_detect_rate': LaunchConfiguration('chassis_detect_rate')},
            {'fusion_publish_rate': LaunchConfiguration('fusion_publish_rate')},
            {'default_prompt':      "can.vegetable.bottle.box.food.Rubik's cube.tool.bread.objects"},
            {'detector_type':       LaunchConfiguration('detector_type')},
            {'extrinsics_dir':      perception_config_dir},
            {'extrinsics_suffix':   LaunchConfiguration('extrinsics_suffix')},
            {'depth_source':        LaunchConfiguration('depth_source')},
            {'data_max_age':          2.0},
            {'strict_data_freshness': False},
            {'fusion_distance_threshold': 0.3},
            {'target_frame': 'base_link'},
        ],
        ros_arguments=['--log-level', 'multi_camera_perception:=WARN'],
        output='log',
        respawn=False,
    )

    # Grasp perception node (hand camera, provides /perception_grasp_node/detect for piper_grasp_node)
    perception_grasp_config = os.path.join(perception_dir, 'config', 'perception_grasp.yaml')
    perception_grasp_node = Node(
        package='perception',
        executable='perception_grasp_node',
        name='perception_grasp_node',
        parameters=[
            perception_grasp_config,
        ],
        ros_arguments=['--log-level', 'perception_grasp_node:=WARN'],
        output='log',
        respawn=True,
        respawn_delay=3.0,
    )

    # Grasp RViz node: publishes 2x2 panel (RGB + depth + detections + IR)
    # Topic: /perception_grasp_rviz_node/vis/panel — matches cleaner_manager.rviz HandCamera
    perception_grasp_rviz_config = os.path.join(perception_dir, 'config', 'perception_grasp_rviz.yaml')
    perception_grasp_rviz_node = Node(
        package='perception',
        executable='perception_grasp_rviz_node',
        name='perception_grasp_rviz_node',
        parameters=[perception_grasp_rviz_config],
        ros_arguments=['--log-level', 'perception_grasp_rviz_node:=WARN'],
        output='log',
        respawn=True,
        respawn_delay=3.0,
    )

    # Perception RViz visualization (detection images + 3D markers)
    # 占 ~80% CPU (一个核), 自动模式下可通过 perception_rviz:=false 关闭
    perception_rviz_node = Node(
        package='perception',
        executable='multi_camera_rviz_node',
        name='multi_camera_rviz',
        condition=IfCondition(LaunchConfiguration('perception_rviz')),
        parameters=[{
            'enable_top':       True,
            'enable_chassis':   True,
            'target_frame':     'base_link',
            'publish_rate':     2.0,
            'cloud_skip':       4,
            'image_scale':      0.5,
            'depth_max':        5.0,
            'extrinsics_dir':   perception_config_dir,
            'extrinsics_suffix': LaunchConfiguration('extrinsics_suffix'),
            'enable_fused_viz': False,
        }],
        output='log',
    )

    manual_config = os.path.join(cleaner_mgr_dir, 'config', 'cleaner_manager.yaml')
    manual_node = Node(
        package='cleaner_manager',
        executable='manual_control_node.py',
        name='manual_control_node',
        parameters=[manual_config],
        prefix='nice -n 5 ',
        output='screen',
        condition=is_manual,
    )

    cleaner_manager_node = Node(
        package='cleaner_manager',
        executable='cleaner_manager_node.py',
        name='cleaner_manager_node',
        parameters=[manual_config],
        prefix='nice -n 5 ',
        output='screen',
        condition=is_auto,
    )

    object_selector_node = Node(
        package='cleaner_manager',
        executable='object_selector_node.py',
        name='object_selector_node',
        output='log',
        condition=is_manual,
    )

    # RViz — use ExecuteProcess directly; Node+IfCondition silently skips
    # inside OnProcessExit handlers (ROS2 Humble launch bug).
    rviz_config = os.path.join(cleaner_mgr_dir, 'config', 'cleaner_manager.rviz')
    display = os.environ.get('DISPLAY', ':1')
    rviz_proc = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config, '--ros-args', '--log-level', 'WARN'],
        additional_env={'DISPLAY': display},
        prefix='nice -n 15 ',
        output='screen',
    )

    # ==================== Readiness gates ====================
    nav_cam_gate = _ready_gate(
        '导航+相机',
        'ros2 topic list 2>/dev/null | grep -q /odom '
        '&& ros2 topic list 2>/dev/null | grep -q /camera/top/color/image_raw',
        timeout=45)

    piper_gate = _ready_gate(
        '机械臂',
        'ros2 node list 2>/dev/null | grep -q piper_grasp_node',
        timeout=20)

    perception_gate = _ready_gate(
        '感知系统',
        'ros2 node list 2>/dev/null | grep -q multi_camera_perception',
        timeout=30)

    manual_gate = _ready_gate(
        '控制节点',
        'ros2 node list 2>/dev/null | grep -qE "manual_control_node|cleaner_manager_node"',
        timeout=15)

    # ==================== Build (event-driven sequential startup) ====================
    ld = LaunchDescription(args)
    ld.add_action(SetEnvironmentVariable('LIBREALSENSE_LOG_LEVEL', 'ERROR'))

    ld.add_action(LogInfo(msg='============================================================'))
    ld.add_action(LogInfo(msg='  手动控制全系统 — 顺序启动'))
    ld.add_action(LogInfo(msg='============================================================'))

    # --- [阶段 0] 导航 + 相机 ---
    ld.add_action(LogInfo(msg=''))
    ld.add_action(LogInfo(msg='[阶段 0] 导航 + 相机...'))
    ld.add_action(nav_launch)
    ld.add_action(camera_launch)
    ld.add_action(nav_cam_gate)

    # --- [阶段 1] 机械臂 (after nav+camera ready) ---
    ld.add_action(RegisterEventHandler(OnProcessExit(
        target_action=nav_cam_gate,
        on_exit=[
            LogInfo(msg=''),
            LogInfo(msg='[阶段 1] 机械臂...'),
            piper_tf,
            piper_node,
            piper_gate,
        ],
    )))

    # --- [阶段 2] 感知 (after piper ready) ---
    ld.add_action(RegisterEventHandler(OnProcessExit(
        target_action=piper_gate,
        on_exit=[
            LogInfo(msg=''),
            LogInfo(msg='[阶段 2] 感知系统...'),
            perception_node,
            perception_grasp_node,
            perception_grasp_rviz_node,
            perception_rviz_node,
            perception_gate,
        ],
    )))

    # --- [阶段 3] 手动控制 + 自主模式 + GUI (after perception ready) ---
    ld.add_action(RegisterEventHandler(OnProcessExit(
        target_action=perception_gate,
        on_exit=[
            LogInfo(msg=''),
            LogInfo(msg='[阶段 3] 控制节点...'),
            manual_node,
            cleaner_manager_node,
            object_selector_node,
            manual_gate,
        ],
    )))

    # --- [阶段 4] RViz — 最后启动，所有 topic 已就绪 ---
    ld.add_action(RegisterEventHandler(OnProcessExit(
        target_action=manual_gate,
        on_exit=[
            LogInfo(msg=''),
            LogInfo(msg='[阶段 4] 启动 RViz...'),
            rviz_proc,
            LogInfo(msg=''),
            LogInfo(msg='============================================================'),
            LogInfo(msg='  [OK] 系统启动完成!'),
            LogInfo(msg='============================================================'),
            LogInfo(msg=''),
            LogInfo(msg='活动组件:'),
            LogInfo(msg='  [x] 导航 (Fast-LIO2 + HDL + Nav2)'),
            LogInfo(msg='  [x] 相机 (Top + Hand + Chassis)'),
            LogInfo(msg='  [x] 机械臂 (Piper)'),
            LogInfo(msg='  [x] 感知 (multi_camera_perception + rviz可视化)'),
            LogInfo(msg='  [x] 控制节点'),
            LogInfo(msg='  [x] RViz'),
            LogInfo(msg=''),
            LogInfo(msg='操作提示:'),
            LogInfo(msg='  - Ctrl+C 停止所有组件'),
            LogInfo(msg='  - RViz 中使用 Nav2 Goal 设置导航目标'),
            LogInfo(msg='  - GUI 面板控制抓取流程'),
            LogInfo(msg='============================================================'),
        ],
    )))

    return ld
