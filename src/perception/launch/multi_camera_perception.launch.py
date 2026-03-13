#!/usr/bin/env python3
"""
Multi-Camera Perception Launch File (ROS2)

启动双相机（Top + Chassis）联合感知节点。

**注意**: 默认不启动相机驱动，假设相机已在运行。
如需启动相机驱动，请设置 use_camera_driver:=true

使用方法:
    ros2 launch perception multi_camera_perception.launch.py
    ros2 launch perception multi_camera_perception.launch.py use_camera_driver:=true
    ros2 launch perception multi_camera_perception.launch.py enable_chassis:=false
    ros2 launch perception multi_camera_perception.launch.py auto_detect_rate:=3.0
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
    
    # 获取 camera_driver 包路径（用于包含相机驱动 launch）
    camera_driver_share = get_package_share_directory('camera_driver')

    # ==================== Launch Arguments ====================

    # 相机驱动控制（安全：默认不启动，避免冲突）
    use_camera_driver_arg = DeclareLaunchArgument(
        'use_camera_driver', default_value='false',
        description='Launch camera driver. WARNING: Set to false if camera driver is already running to avoid conflicts'
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
    
    # 相机序列号（传递给相机驱动）
    top_serial_arg = DeclareLaunchArgument(
        'top_serial_no', default_value="'318122302992'",
        description='Top camera serial number'
    )
    chassis_serial_arg = DeclareLaunchArgument(
        'chassis_serial_no', default_value="'337122071540'",
        description='Chassis camera serial number'
    )

    # 自动检测频率 — 每相机独立，与硬件帧率对齐
    top_detect_rate_arg = DeclareLaunchArgument(
        'top_detect_rate', default_value='5.0',
        description='Top camera (D455) detection rate in Hz'
    )
    chassis_detect_rate_arg = DeclareLaunchArgument(
        'chassis_detect_rate', default_value='6.0',
        description='Chassis camera (D435) detection rate in Hz'
    )

    # 融合距离阈值
    fusion_threshold_arg = DeclareLaunchArgument(
        'fusion_distance_threshold', default_value='0.3',
        description='Fusion distance threshold in meters'
    )

    # LiDAR 配置
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar', default_value='false',
        description='Enable LiDAR fusion'
    )
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic', default_value='/rslidar_points',
        description='LiDAR point cloud topic'
    )

    # 目标坐标系
    target_frame_arg = DeclareLaunchArgument(
        'target_frame', default_value='base_link',
        description='Target coordinate frame'
    )

    # 检测器类型
    detector_type_arg = DeclareLaunchArgument(
        'detector_type', default_value='sam3',
        description='Detector type: sam3 or dinox'
    )

    # ==================== Included Launches ====================
    
    # 包含相机驱动启动（条件：use_camera_driver:=true）
    # 警告：如果相机驱动已在运行，请勿启动，否则会导致设备冲突
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

    multi_camera_node = Node(
        package='perception',
        executable='multi_camera_perception_node',
        name='multi_camera_perception',
        output='screen',
        respawn=True,
        respawn_delay=3.0,
        parameters=[
            # 相机启用
            {'enable_top': LaunchConfiguration('enable_top')},
            {'enable_chassis': LaunchConfiguration('enable_chassis')},
            
            # 相机话题（使用默认值）
            {'top_color_topic': '/camera/top/color/image_raw'},
            {'top_depth_topic': '/camera/top/aligned_depth_to_color/image_raw'},
            {'top_info_topic': '/camera/top/color/camera_info'},
            {'chassis_color_topic': '/camera/chassis/color/image_raw'},
            {'chassis_depth_topic': '/camera/chassis/aligned_depth_to_color/image_raw'},
            {'chassis_info_topic': '/camera/chassis/color/camera_info'},
            
            # 降低分辨率以减少 USB 带宽（如果相机支持）
            # 注意：需要配合相机驱动参数修改
            
            # 检测配置
            {'top_detect_rate': LaunchConfiguration('top_detect_rate')},
            {'chassis_detect_rate': LaunchConfiguration('chassis_detect_rate')},
            {'default_prompt': 'bottle.cup.box.barrel.toy.cabinet'},
            {'min_score': 0.35},
            
            # 数据新鲜度配置（chassis相机可能需要更宽松的设置）
            {'data_max_age': 2.0},  # 2秒数据有效期（默认0.5秒可能太严格）
            {'strict_data_freshness': False},  # 非严格模式：允许使用稍旧的数据
            
            # 融合配置
            {'fusion_distance_threshold': LaunchConfiguration('fusion_distance_threshold')},
            
            # LiDAR
            {'enable_lidar': LaunchConfiguration('enable_lidar')},
            {'lidar_topic': LaunchConfiguration('lidar_topic')},
            
            # 坐标系
            {'target_frame': LaunchConfiguration('target_frame')},

            # 检测器类型
            {'detector_type': LaunchConfiguration('detector_type')},

            # 外参目录
            {'extrinsics_dir': config_dir},
        ],
    )

    # 延迟启动感知节点
    # 如果启动了相机驱动，延时 10 秒等待初始化（足够 RealSense 枚举和启动流）
    delayed_perception_node = TimerAction(
        period=PythonExpression([
            '10.0 if "', LaunchConfiguration('use_camera_driver'), '" == "true" else 0.0'
        ]),
        actions=[multi_camera_node]
    )

    return LaunchDescription([
        # Arguments
        use_camera_driver_arg,  # 控制是否启动相机驱动
        enable_top_arg,
        enable_chassis_arg,
        top_serial_arg,
        chassis_serial_arg,
        top_detect_rate_arg,
        chassis_detect_rate_arg,
        fusion_threshold_arg,
        enable_lidar_arg,
        lidar_topic_arg,
        target_frame_arg,
        detector_type_arg,

        # 1. 条件启动相机驱动（默认不启动，避免冲突）
        camera_driver_launch,
        
        # 2. 延迟启动感知节点（仅在启动驱动时延时）
        delayed_perception_node,
    ])
