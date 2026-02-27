#!/usr/bin/env python3
"""
Perception Grasp RViz Launch File (ROS2)

启动抓取感知可视化节点 + RViz。
支持性能模式参数控制可视化质量。

使用方法:
    ros2 launch perception perception_grasp_rviz.launch.py
    ros2 launch perception perception_grasp_rviz.launch.py perf_mode:=3
    ros2 launch perception perception_grasp_rviz.launch.py rviz:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _is_package_available(package_name: str) -> bool:
    """检查 ROS2 包是否可用"""
    try:
        get_package_share_directory(package_name)
        return True
    except PackageNotFoundError:
        return False


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception')
    config_dir = os.path.join(bringup_share, 'config')
    rviz_dir = os.path.join(bringup_share, 'rviz')

    # ==================== Launch Arguments ====================

    # 性能模式: 0=禁用, 1=节能模式(1Hz), 3=高性能模式(3Hz)
    perf_mode_arg = DeclareLaunchArgument(
        'perf_mode', default_value='1',
        description='Performance mode: 0=disabled, 1=power_saving(default), 3=high_performance'
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz'
    )

    use_fake_joints_arg = DeclareLaunchArgument(
        'use_fake_joints', default_value='true',
        description='Use fake joint states (true=default zero pose, false=need real driver)'
    )

    # ==================== Nodes ====================

    # Perception Grasp 检测节点 (数据来源)
    perception_grasp_node = Node(
        package='perception',
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

    # Perception Grasp RViz 节点 (可视化)
    perception_grasp_rviz_node = Node(
        package='perception',
        executable='perception_grasp_rviz_node',
        name='perception_grasp_rviz_node',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            os.path.join(config_dir, 'perception_grasp_rviz.yaml') if os.path.exists(
                os.path.join(config_dir, 'perception_grasp_rviz.yaml')
            ) else {},
            {
                # 性能模式自适应参数
                'pointcloud_scale': PythonExpression([
                    '0.1 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                    '(0.25 if ', LaunchConfiguration('perf_mode'), '==1 else 0.4)'
                ]),
                'temporal_filter_size': PythonExpression([
                    '1 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                    '(2 if ', LaunchConfiguration('perf_mode'), '==1 else 3)'
                ]),
                'marker_lifetime': PythonExpression([
                    '10.0 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                    '(5.0 if ', LaunchConfiguration('perf_mode'), '==1 else 3.0)'
                ]),
            }
        ],
    )

    # Joint State Publisher (fake joints) - 仅在包可用且启用时启动
    joint_state_publisher_nodes = []
    if _is_package_available('joint_state_publisher'):
        joint_state_publisher_nodes.append(Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            condition=IfCondition(LaunchConfiguration('use_fake_joints')),
            parameters=[{'rate': 10}],
        ))
    # 如果包不可用，静默跳过（真实机器人环境通常已有 joint_states 发布）

    # RViz
    rviz_config = os.path.join(rviz_dir, 'perception_grasp.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        perf_mode_arg,
        rviz_arg,
        use_fake_joints_arg,

        # Nodes
        *joint_state_publisher_nodes,  # 可选，仅在包可用时添加
        perception_grasp_node,         # 检测节点 (数据来源)
        perception_grasp_rviz_node,    # 可视化节点
        rviz_node,
    ])
