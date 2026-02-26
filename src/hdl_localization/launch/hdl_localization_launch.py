#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_dir = get_package_share_directory('hdl_localization')
    hdl_global_localization_pkg = get_package_share_directory('hdl_global_localization')
    
    # Declare arguments
    nodelet_manager_arg = DeclareLaunchArgument(
        'nodelet_manager',
        default_value='velodyne_nodelet_manager',
        description='Nodelet manager name'
    )
    
    points_topic_arg = DeclareLaunchArgument(
        'points_topic',
        default_value='/velodyne_points',
        description='Points topic'
    )
    
    odom_child_frame_id_arg = DeclareLaunchArgument(
        'odom_child_frame_id',
        default_value='velodyne',
        description='Odometry child frame ID'
    )
    
    use_imu_arg = DeclareLaunchArgument(
        'use_imu',
        default_value='false',
        description='Use IMU'
    )
    
    invert_imu_acc_arg = DeclareLaunchArgument(
        'invert_imu_acc',
        default_value='false',
        description='Invert IMU acceleration'
    )
    
    invert_imu_gyro_arg = DeclareLaunchArgument(
        'invert_imu_gyro',
        default_value='false',
        description='Invert IMU gyroscope'
    )
    
    use_global_localization_arg = DeclareLaunchArgument(
        'use_global_localization',
        default_value='true',
        description='Use global localization'
    )
    
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu/data',
        description='IMU topic'
    )
    
    enable_robot_odometry_prediction_arg = DeclareLaunchArgument(
        'enable_robot_odometry_prediction',
        default_value='false',
        description='Enable robot odometry prediction'
    )
    
    robot_odom_frame_id_arg = DeclareLaunchArgument(
        'robot_odom_frame_id',
        default_value='odom',
        description='Robot odometry frame ID'
    )
    
    plot_estimation_errors_arg = DeclareLaunchArgument(
        'plot_estimation_errors',
        default_value='false',
        description='Plot estimation errors'
    )
    
    # Include hdl_global_localization launch if needed
    hdl_global_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hdl_global_localization_pkg, 'launch', 'hdl_global_localization_launch.py')
        ),
        condition=IfCondition(PythonExpression(['"', LaunchConfiguration('use_global_localization'), '" == "true"']))
    )
    
    # HDL localization node
    hdl_localization_node = Node(
        package='hdl_localization',
        executable='hdl_localization_node',
        name='hdl_localization_node',
        remappings=[
            ('points', LaunchConfiguration('points_topic')),
            ('imu', LaunchConfiguration('imu_topic'))
        ],
        parameters=[
            {
                'odom_child_frame_id': LaunchConfiguration('odom_child_frame_id'),
                'use_imu': LaunchConfiguration('use_imu'),
                'invert_acc': LaunchConfiguration('invert_imu_acc'),
                'invert_gyro': LaunchConfiguration('invert_imu_gyro'),
                'cool_time_duration': 2.0,
                'enable_robot_odometry_prediction': LaunchConfiguration('enable_robot_odometry_prediction'),
                'robot_odom_frame_id': LaunchConfiguration('robot_odom_frame_id'),
                'registration_method': 'NDT_OMP',
                'ndt_resolution': 1.0,
                'downsample_resolution': 0.1,
                'use_global_localization': LaunchConfiguration('use_global_localization')
            }
        ],
        output='screen'
    )
    
    # Plot estimation errors script
    plot_estimation_errors_script = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(pkg_dir, 'scripts', 'plot_status.py')
        ],
        condition=IfCondition(PythonExpression(['"', LaunchConfiguration('plot_estimation_errors'), '" == "true"'])),
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        nodelet_manager_arg,
        points_topic_arg,
        odom_child_frame_id_arg,
        use_imu_arg,
        invert_imu_acc_arg,
        invert_imu_gyro_arg,
        use_global_localization_arg,
        imu_topic_arg,
        enable_robot_odometry_prediction_arg,
        robot_odom_frame_id_arg,
        plot_estimation_errors_arg,
        
        # Include global localization launch
        hdl_global_localization_launch,
        
        # Nodes
        hdl_localization_node,
        plot_estimation_errors_script
    ])
