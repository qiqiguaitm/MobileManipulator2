#!/usr/bin/env python3
"""
Multi-Camera 3D RViz Launch File (ROS2)

一键启动双相机感知 + RViz 可视化。
**默认不启动相机驱动**（假设已在运行），如需启动请设置 use_camera_driver:=true

使用方法:
    # 标准用法（相机驱动已启动）
    ros2 launch perception multi_camera_3d_rviz.launch.py
    
    # 同时启动相机驱动
    ros2 launch perception multi_camera_3d_rviz.launch.py use_camera_driver:=true
    
    # 性能模式
    ros2 launch perception multi_camera_3d_rviz.launch.py perf_mode:=3
    
    # 只显示 top 相机
    ros2 launch perception multi_camera_3d_rviz.launch.py enable_chassis:=false
    
    # 不启动 RViz（仅发布可视化数据）
    ros2 launch perception multi_camera_3d_rviz.launch.py rviz:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    bringup_share = get_package_share_directory('perception')
    config_dir = os.path.join(bringup_share, 'config')
    rviz_dir = os.path.join(bringup_share, 'rviz')
    
    # 获取 camera_driver 包路径
    camera_driver_share = get_package_share_directory('camera_driver')

    # ==================== Launch Arguments ====================
    
    # 安全控制：是否启动相机驱动（默认 false，避免冲突）
    use_camera_driver_arg = DeclareLaunchArgument(
        'use_camera_driver', default_value='false',
        description='Launch camera driver. '
                    'WARNING: Set to false if camera driver is already running to avoid USB conflicts. '
                    'Default is false for safety.'
    )
    
    # 相机启用开关
    enable_top_arg = DeclareLaunchArgument(
        'enable_top', default_value='true',
        description='Enable top camera'
    )
    enable_chassis_arg = DeclareLaunchArgument(
        'enable_chassis', default_value='true',
        description='Enable chassis camera'
    )
    
    # 相机序列号（启动驱动时使用）
    top_serial_arg = DeclareLaunchArgument(
        'top_serial_no', default_value="'318122302992'",
        description='Top camera serial number'
    )
    chassis_serial_arg = DeclareLaunchArgument(
        'chassis_serial_no', default_value="'337122071540'",
        description='Chassis camera serial number'
    )
    
    # 性能模式: 0=禁用可视化, 1=节能(1Hz), 2=平衡(2Hz), 3=高性能(3Hz)
    perf_mode_arg = DeclareLaunchArgument(
        'perf_mode', default_value='2',
        description='Performance mode: 0=minimal, 1=power_saving, 2=balanced(default), 3=high_performance'
    )
    
    # 检测器类型
    detector_type_arg = DeclareLaunchArgument(
        'detector_type', default_value='dinox',
        description='Detector type: dinox or sam3'
    )

    # 检测参数
    auto_detect_rate_arg = DeclareLaunchArgument(
        'auto_detect_rate', default_value='6.0',
        description='Auto detection rate in Hz (0=disable)'
    )
    
    # LiDAR 参数
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic', default_value='/rslidar_points',
        description='LiDAR point cloud topic'
    )
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar', default_value='false',
        description='Enable LiDAR fusion'
    )
    
    # RViz 控制
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 GUI'
    )
    
    # 可视化质量
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', default_value='2.0',
        description='Visualization publish rate (Hz). Override by perf_mode.'
    )

    # 外参文件后缀（用于测试新标定）
    extrinsics_suffix_arg = DeclareLaunchArgument(
        'extrinsics_suffix', default_value='',
        description="Extrinsics file suffix for chassis camera (e.g., '_new' for testing)"
    )

    # CDM 深度优化开关
    enable_depth_optimizer_arg = DeclareLaunchArgument(
        'enable_depth_optimizer', default_value='true',
        description='Enable CDM depth optimization (set false to use raw RealSense depth)'
    )

    # ==================== Included Launches ====================
    
    # 条件启动相机驱动
    camera_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(camera_driver_share, 'launch', 'camera_driver.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('use_camera_driver')),
        launch_arguments={
            'top_enable': LaunchConfiguration('enable_top'),
            'chassis_enable': LaunchConfiguration('enable_chassis'),
            'hand_enable': 'false',
            'top_serial_no': LaunchConfiguration('top_serial_no'),
            'chassis_serial_no': LaunchConfiguration('chassis_serial_no'),
        }.items()
    )

    # ==================== Nodes ====================
    
    # 多相机感知节点
    multi_camera_perception_node = Node(
        package='perception',
        executable='multi_camera_perception_node',
        name='multi_camera_perception',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            {'enable_top': LaunchConfiguration('enable_top')},
            {'enable_chassis': LaunchConfiguration('enable_chassis')},
            {'detector_type': LaunchConfiguration('detector_type')},
            {'auto_detect_rate': LaunchConfiguration('auto_detect_rate')},
            {'default_prompt': 'bottle.cup.can.box.barrel.toy.cabinet'},
            # 匈牙利匹配融合参数
            {'fusion_distance_threshold': 0.2},  # 20cm 距离硬门控
            {'fusion_max_cost': 0.7},
            {'fusion_weight_distance': 0.6},
            {'fusion_weight_category': 0.4},
            {'enable_lidar': LaunchConfiguration('enable_lidar')},
            {'lidar_topic': LaunchConfiguration('lidar_topic')},
            {'target_frame': 'base_link'},
            {'extrinsics_dir': config_dir},
            {'extrinsics_suffix': LaunchConfiguration('extrinsics_suffix')},
            {'enable_depth_optimizer': LaunchConfiguration('enable_depth_optimizer')},
            # 数据新鲜度配置
            {'data_max_age': 2.0},
            {'strict_data_freshness': False},
        ],
    )
    
    # 延时启动感知节点
    # 如果启动驱动，延时 10 秒等待相机初始化；否则延时 3 秒确保数据就绪
    delayed_perception_node = TimerAction(
        period=PythonExpression([
            '10.0 if "', LaunchConfiguration('use_camera_driver'), '" == "true" else 3.0'
        ]),
        actions=[multi_camera_perception_node]
    )
    
    # 多相机可视化节点
    multi_camera_rviz_node = Node(
        package='perception',
        executable='multi_camera_rviz_node',
        name='multi_camera_rviz',
        output='screen',
        parameters=[{
            'enable_top': LaunchConfiguration('enable_top'),
            'enable_chassis': LaunchConfiguration('enable_chassis'),
            'target_frame': 'base_link',
            'extrinsics_dir': config_dir,
            'extrinsics_suffix': LaunchConfiguration('extrinsics_suffix'),
            # 性能自适应参数
            'publish_rate': PythonExpression([
                '0.0 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(1.0 if ', LaunchConfiguration('perf_mode'), '==1 else ',
                '(2.0 if ', LaunchConfiguration('perf_mode'), '==2 else 3.0))'
            ]),
            'cloud_skip': PythonExpression([
                '10 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(4 if ', LaunchConfiguration('perf_mode'), '==1 else ',
                '(2 if ', LaunchConfiguration('perf_mode'), '==2 else 1))'
            ]),
            'image_scale': PythonExpression([
                '0.25 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(0.3 if ', LaunchConfiguration('perf_mode'), '==1 else ',
                '(0.5 if ', LaunchConfiguration('perf_mode'), '==2 else 0.75))'
            ]),
            'depth_max': 5.0,
            'enable_outlier_filter': True,
            'outlier_radius': 0.05,
            'outlier_min_neighbors': 5,
            'enable_depth_denoise': True,
            'depth_denoise_ksize': 5,
            'lidar_topic': LaunchConfiguration('lidar_topic'),
            'lidar_min_range': 0.0,
            'lidar_max_range': 10.0,
            'lidar_skip': PythonExpression([
                '10 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(5 if ', LaunchConfiguration('perf_mode'), '==1 else ',
                '(3 if ', LaunchConfiguration('perf_mode'), '==2 else 2))'
            ]),
            'lidar_publish_rate': PythonExpression([
                '0.0 if ', LaunchConfiguration('perf_mode'), '==0 else ',
                '(3.0 if ', LaunchConfiguration('perf_mode'), '==1 else ',
                '(6.0 if ', LaunchConfiguration('perf_mode'), '==2 else 10.0))'
            ]),
        }],
    )
    
    # RViz2 应用程序（自动加载配置文件）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(rviz_dir, 'multi_camera_3d.rviz')],
        output='screen',
    )

    # LiDAR TF (lidar_link -> rslidar)
    lidar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_link_to_rslidar',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'rslidar'],
    )

    return LaunchDescription([
        # Arguments
        use_camera_driver_arg,
        enable_top_arg,
        enable_chassis_arg,
        top_serial_arg,
        chassis_serial_arg,
        detector_type_arg,
        perf_mode_arg,
        auto_detect_rate_arg,
        lidar_topic_arg,
        enable_lidar_arg,
        rviz_arg,
        publish_rate_arg,
        extrinsics_suffix_arg,
        enable_depth_optimizer_arg,

        # TF 基础 (注意: map->base_link 由导航节点发布，禁止感知覆盖)
        lidar_tf_node,
        
        # 条件启动相机驱动（默认不启动）
        camera_driver_launch,
        
        # 延时启动感知节点（仅在启动驱动时延时）
        delayed_perception_node,
        
        # 启动可视化节点
        multi_camera_rviz_node,
        
        # 启动 RViz（默认启动）
        rviz_node,
    ])
