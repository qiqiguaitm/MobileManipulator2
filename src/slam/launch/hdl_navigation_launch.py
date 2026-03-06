"""
HDL Localization + Navigation Launch（严格仿照 ROS1 navigation_3d.launch）

3D Navigation: Fast-LIO 里程计 + HDL 纠正 + Move Base/Nav2。
地图路径与 ROS1 一致：map_dir 下 map.yaml + GlobalMap_hdl.pcd。

Usage:
  ros2 launch slam hdl_navigation_launch.py
  ros2 launch slam hdl_navigation_launch.py map_dir:=/path/to/maps/sc_pgo
  ros2 launch slam hdl_navigation_launch.py use_odom_fusion:=false
  ros2 launch slam hdl_navigation_launch.py launch_chassis:=false
  ros2 launch slam hdl_navigation_launch.py cloud_delay:=0.05
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
import os

# 与 ROS1 navigation_3d.launch 一致：地图目录默认路径
DEFAULT_MAP_DIR = '/home/didi/workspace/MobileManipulator2/maps/sc_pgo'


def generate_launch_description():
    declare_rviz = DeclareLaunchArgument('enable_rviz', default_value='true')
    declare_rviz_delay = DeclareLaunchArgument(
        'rviz_delay',
        default_value='5.0',
        description='Delay seconds before starting RViz to wait for map TF',
    )
    declare_map_dir = DeclareLaunchArgument(
        'map_dir',
        default_value=DEFAULT_MAP_DIR,
        description='Map directory (contains map.yaml and GlobalMap_hdl.pcd), same as ROS1 map_dir',
    )
    declare_map = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([LaunchConfiguration('map_dir'), 'map.yaml']),
        description='Path to map.yaml',
    )
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')
    declare_launch_chassis = DeclareLaunchArgument('launch_chassis', default_value='true',
        description='Launch tracer_base for wheel odom')
    declare_use_odom_fusion = DeclareLaunchArgument('use_odom_fusion', default_value='true',
        description='Fuse wheel odom for accurate twist (requires launch_chassis)')
    declare_use_raw_laserscan = DeclareLaunchArgument('use_raw_laserscan', default_value='true',
        description='Use raw /rslidar_points_delayed for laserscan. true=raw(静态TF); false=aligned_points(NDT校正)')
    declare_cloud_delay = DeclareLaunchArgument('cloud_delay', default_value='0.05',
        description='Point cloud delay (s) for HDL. 0.05=ROS1 一致，确保 Fast-LIO TF 先发布')
    declare_use_odom_fused = DeclareLaunchArgument('use_odom_fused', default_value='true',
        description='hdl_odom_to_tf 从 /odom/fused 取 odom->base。遇崩溃可 false 回退 TF')

    slam_pkg = get_package_share_directory('slam')
    rslidar_pkg = get_package_share_directory('rslidar_sdk')
    hipnuc_pkg = get_package_share_directory('hipnuc_imu')
    robot_desc_pkg = get_package_share_directory('mobile_manipulator2_description')
    try:
        tracer_pkg = get_package_share_directory('tracer_base')
        has_tracer = True
    except PackageNotFoundError:
        has_tracer = False

    # 传感器
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(rslidar_pkg, 'launch', 'start.py')),
        launch_arguments={'rviz': 'false'}.items(),
    )
    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(hipnuc_pkg, 'launch', 'imu_spec_msg.launch.py')),
    )

    # FAST-LIO（输出 /fastlio/odom）
    fastlio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_pkg, 'launch', 'fastlio_odom_launch.py')),
    )

    # 底盘 (wheel_odom) - 可选
    tracer_launch = None
    if has_tracer:
        tracer_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tracer_pkg, 'launch', 'tracer_base.launch.py')),
            launch_arguments={
                'port_name': 'can1',
                'odom_frame': 'wheel_odom',
                'odom_topic_name': '/wheel_odom',
                'publish_tf': 'false',
            }.items(),
            condition=IfCondition(LaunchConfiguration('launch_chassis')),
        )

    # Frame 转换: Fast-LIO camera_init/body -> odom/base_link
    odom_frame_converter_node = Node(
        package='slam',
        executable='odom_frame_converter.py',
        name='odom_frame_converter',
        parameters=[{'input': '/fastlio/odom', 'output': '/odom'}],
    )
    odom_fusion_node = Node(
        package='slam',
        executable='odom_fusion_sync.py',
        name='odom_fusion_sync',
        output='screen',
        parameters=[{
            'fastlio_topic': '/fastlio/odom',
            'wheel_topic': '/wheel_odom',
            'output': '/odom/fused',
            'twist_alpha': 0.3,
            'wz_alpha': 0.7,
            'wz_deadzone': 0.03,
            'yaw_calibration_alpha': 0.9,
        }],
        condition=IfCondition(LaunchConfiguration('use_odom_fusion')),
    )
    odom_relay_node = Node(
        package='slam',
        executable='odom_relay.py',
        name='odom_relay',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('use_odom_fusion'), "' != 'true'"])),
    )

    # 机器人模型
    # 直接订阅 /joint_states (grasp发布)，无数据时用静态 fallback
    urdf_path = os.path.join(robot_desc_pkg, 'urdf', 'mobile_manipulator2_description.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # Robot State Publisher - 直接订阅 /joint_states
    # 当 grasp 未运行时，通过 relay 切换到静态 fallback
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        remappings=[('/joint_states', '/joint_states_combined')],
    )

    # 智能 JointState Relay - 监听 grasp 数据，超时时自动切换到静态 fallback
    joint_state_relay_node = Node(
        package='slam',
        executable='joint_state_relay.py',
        name='joint_state_relay',
        parameters=[{
            'realtime_topic': '/joint_states',
            'static_topic': '/joint_states_static',
            'output_topic': '/joint_states_combined',
            'timeout_sec': 0.3,
        }],
    )

    # 静态 JointState - 机械臂折叠姿态 (fallback)
    joint_state_static = Node(
        package='slam',
        executable='joint_state_static.py',
        name='joint_state_static',
        parameters=[{'topic': '/joint_states_static', 'rate': 50.0}],
    )

    # 延迟点云 -> HDL
    # ⚠️ 延迟目的：确保 Fast-LIO2 的 TF(stamp=T) 在 HDL 处理点云(stamp=T) 前已发布
    # ⚠️ 不改时间戳！保持时序不变量：点云时间戳 = TF时间戳
    delayed_cloud_relay_node = Node(
        package='slam',
        executable='delayed_cloud_relay.py',
        name='delayed_cloud_relay',
        parameters=[{
            'input': '/rslidar_points',
            'output': '/rslidar_points_delayed',
            'delay': LaunchConfiguration('cloud_delay'),
            'timestamp_offset': 0.0,
            'use_system_time': False,  # 保持原始时间戳，维持与Fast-LIO TF的时序一致性
        }],
    )

    # 全局地图（PCD）：与 ROS1 一致，map_dir/GlobalMap_hdl.pcd
    globalmap_pcd_path = PathJoinSubstitution([LaunchConfiguration('map_dir'), 'GlobalMap_hdl.pcd'])
    globalmap_node = Node(
        package='slam',
        executable='globalmap_publisher.py',
        name='globalmap_publisher',
        parameters=[{'globalmap_pcd': globalmap_pcd_path, 'frame_id': 'map', 'use_adaptive': True}],
    )

    # HDL 全局定位
    hdl_global_pkg = get_package_share_directory('hdl_global_localization')
    hdl_global_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(hdl_global_pkg, 'launch', 'hdl_global_localization_launch.py')),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    # HDL 本地定位（严格仿照 ROS1 hdl_localization.launch 参数）
    hdl_local_node = Node(
        package='hdl_localization',
        executable='hdl_localization_node',
        name='hdl_localization_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'use_imu': False,  # 关闭IMU预测，避免ROS2下IMU处理差异导致漂移
            'invert_acc': False,
            'invert_gyro': False,
            'cool_time_duration': 2.0,
            'enable_robot_odometry_prediction': True,   # 必须True: HDL通过TF查odom->base_link计算map->odom; False会直接发map->base_link，与body->base_link静态TF冲突导致TF树断裂
            'robot_odom_frame_id': 'odom',
            'odom_child_frame_id': 'base_link',
            'reg_method': 'NDT_OMP',
            'registration_method': 'NDT_OMP',
            'ndt_neighbor_search_method': 'DIRECT7',
            'ndt_neighbor_search_radius': 2.0,  # ROS2优化
            'ndt_resolution': 0.5,  # ROS2优化，更精细匹配
            'ndt_max_iterations': 50,  # ROS2优化，更稳定收敛
            'downsample_resolution': 0.15,  # 与ROS1一致，15cm下采样
            'status_max_correspondence_dist': 0.5,  # 匹配状态阈值
            'specify_init_pose': False,          # Phase 3: 依靠 SC 自动重定位
            'init_pos_x': 0.0,
            'init_pos_y': 0.0,
            'init_pos_z': 0.0,
            'init_ori_w': 1.0,
            'init_ori_x': 0.0,
            'init_ori_y': 0.0,
            'init_ori_z': 0.0,
            'auto_relocalization': True,         # 启用自动重定位
            'auto_reloc_delay_ms': 5000,         # globalmap 收到后等待5秒 (让scan稳定)
            'auto_reloc_conf_threshold': 0.25,   # 降低阈值以允许通过SC候选点
            'auto_reloc_ndt_candidates': 15,     # NDT 验证候选数 (10→15, 全局采样更多位姿)
            'use_global_localization': True,
        }],
        remappings=[
            ('points', '/rslidar_points_delayed'),
            ('imu', '/imu/data'),
            ('odom', '/odom'),
            ('initialpose', '/initialpose'),
            ('aligned_points', '/aligned_points'),
        ],
    )

    # TF Republisher：接受所有 map->odom，以 now()+stamp_ahead、100Hz 重发供 Nav2
    # stamp_ahead_sec 超前100ms，避免controller_server查询TF时extrapolation into the future错误
    tf_republisher_node = Node(
        package='slam',
        executable='tf_republisher.py',
        name='tf_republisher',
        parameters=[{
            'source_frame': 'map',
            'child_frame': 'odom',
            'rate_hz': 100.0,
            'stamp_ahead_sec': 0.05,  # 超前50ms，FastLIO已改用now()时间戳，只需补偿少量延迟
            'msg_age_threshold': 0.05,
            'smooth_alpha': 0.3,  # EMA平滑：0.3=适度平滑，减小HDL抖动约50%
        }],
    )

    nav2_config = os.path.join(slam_pkg, 'config', 'nav2_minimal_params.yaml')
    map_server_params = os.path.join(slam_pkg, 'config', 'map_server_params.yaml')
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[map_server_params, {'yaml_filename': LaunchConfiguration('map'), 'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    planner_node = Node(package='nav2_planner', executable='planner_server', name='planner_server', parameters=[nav2_config])
    controller_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_config],
        remappings=[('/odom', '/odom/fused')],
    )
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[nav2_config, {'odom_topic': '/odom/fused'}],
    )
    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_config],
        remappings=[('/cmd_vel', '/cmd_vel')],
    )
    lifecycle_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'), 'autostart': True, 'node_names': ['map_server', 'planner_server', 'controller_server', 'behavior_server', 'bt_navigator']}],
            ),
        ],
    )

    # 严格仿照 ROS1 rslidar_driver.launch:24-38
    # angle_min/max: 只保留前方180°，屏蔽后方金属box反射
    # min_height: ROS1 用 -0.08 + costmap PointCloud2 的 min_obstacle_height 二次过滤
    #   ROS2 走 LaserScan (2D,无高度)，costmap 无法二次过滤 → 必须在此过滤地面
    #   lidar 距地面 ~6cm，rslidar 系下地面 ≈ z=-0.06m, 设 0.0 过滤地面回波
    # max_height: 1.36m (检测人、桌椅等)
    # transform_tolerance: 0.1s (系统时间统一后，无需大容错)
    _laserscan_params = {'target_frame': 'rslidar', 'transform_tolerance': 0.1, 'min_height': -0.08, 'max_height': 1.36,
                        'angle_min': -1.5708, 'angle_max': 1.5708, 'angle_increment': 0.00872, 'scan_time': 0.1,
                        'range_min': 0.2, 'range_max': 10.0, 'use_inf': True}
    pointcloud_to_laserscan_aligned = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[('cloud_in', '/aligned_points'), ('scan', '/scan')],
        parameters=[_laserscan_params],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('use_raw_laserscan'), "' != 'true'"])),
    )
    pointcloud_to_laserscan_raw = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[('cloud_in', '/rslidar_points_delayed'), ('scan', '/scan')],
        parameters=[_laserscan_params],
        condition=IfCondition(LaunchConfiguration('use_raw_laserscan')),
    )

    lidar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_link_to_rslidar',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0.0635',
            '--qx', '-0.00218012051', '--qy', '0.000590033588', '--qz', '0.0', '--qw', '0.999997449',
            '--frame-id', 'lidar_link', '--child-frame-id', 'rslidar',
        ],
    )

    rviz_node = TimerAction(
        period=LaunchConfiguration('rviz_delay'),
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                arguments=['-d', os.path.join(slam_pkg, 'rviz', 'hdl_navigation.rviz')],
            )
        ],
    )

    body_to_base_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_base_link',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                   '--frame-id', 'body', '--child-frame-id', 'base_link'],
    )

    launch_entities = [
        declare_rviz, declare_rviz_delay, declare_map_dir, declare_map, declare_use_sim_time, declare_launch_chassis, declare_use_odom_fusion,
        declare_use_raw_laserscan, declare_cloud_delay, declare_use_odom_fused,
        lidar_launch, imu_launch, fastlio_launch,
        odom_frame_converter_node, odom_fusion_node, odom_relay_node,
        robot_state_publisher, joint_state_relay_node, joint_state_static,
        delayed_cloud_relay_node, globalmap_node, hdl_global_node, hdl_local_node,
        tf_republisher_node,
        body_to_base_link_tf,
        lidar_tf_node, pointcloud_to_laserscan_aligned, pointcloud_to_laserscan_raw,
        map_server_node, planner_node, controller_node, behavior_server_node, bt_navigator_node, lifecycle_node,
        rviz_node,
    ]
    if tracer_launch is not None:
        launch_entities.insert(8, tracer_launch)
    return LaunchDescription(launch_entities)
