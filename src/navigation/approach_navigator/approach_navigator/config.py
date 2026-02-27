#!/usr/bin/env python3
"""
Approach Navigator 配置模块

从 ROS1 ThreeStageConfig 迁移，LiDAR 参数替换为 Camera 参数
"""

from dataclasses import dataclass


@dataclass
class ApproachConfig:
    """接近导航器配置

    包含三阶段导航的所有参数配置
    """

    # ========== 阶段1: Nav2 导航到接近点 ==========
    # 接近距离必须 > local_costmap 的 inflation_radius (0.35m)
    approach_distance: float = 0.45     # 接近点到目标的距离 (米)
    nav_timeout: float = 60.0           # Nav2 导航超时时间 (秒)

    # ========== 阶段2: 原地旋转对齐 ==========
    align_tolerance: float = 0.052      # 对齐容差，5度 (弧度)
    align_kp: float = 1.0               # PD控制器 P增益
    align_kd: float = 0.1               # PD控制器 D增益
    align_max_omega: float = 0.5        # 最大角速度 (弧度/秒)
    align_timeout: float = 10.0         # 对齐超时时间 (秒)
    min_align_distance: float = 0.10    # 距离小于此值跳过对齐 (米)

    # ========== 阶段3: 精确接近 ==========
    final_approach_speed: float = 0.12      # 最大前进速度 (米/秒)
    final_approach_distance: float = 0.05   # 目标停止距离：前边缘到目标 (米)
    emergency_stop_distance: float = 0.03   # 紧急刹车距离 (米)
    speed_decel_rate: float = 0.5           # 最大减速率 (米/秒²)
    final_approach_timeout: float = 15.0    # 精确接近超时 (秒)

    # ========== 深度传感器参数 ==========
    depth_topic: str = "/camera/chassis/aligned_depth_to_color/image_raw"  # 深度图话题
    depth_min_valid: float = 0.10           # 最小有效深度，D435约0.1m (米)
    depth_max_valid: float = 2.0            # 最大有效深度 (米)
    depth_data_timeout: float = 0.5         # 深度数据超时判定 (秒)

    # ========== 点云处理参数 ==========
    depth_downsample: int = 4               # 深度图降采样因子 (4=每4像素取1个)
    depth_ground_max_height: float = 0.10   # 地面最高高度 (米)，低于此为地面点
    depth_obstacle_min_height: float = 0.10 # 障碍物最低高度 (米)，需高于地面噪声
    depth_obstacle_max_height: float = 1.5  # 障碍物最高高度 (米)
    depth_detect_width: float = 0.4         # 前方检测宽度 (米)，左右各此值

    # ========== 外参标定文件 ==========
    extrinsics_file: str = "/home/didi/workspace/MobileManipulator2/src/perception/config/extrinsics_chassis_camera_optical_frame_to_base_link.yaml"

    # ========== 机器人几何参数 ==========
    # 来源: 外参标定 extrinsics_chassis_camera_optical_frame_to_base_link.yaml
    # 计算: T_base_to_optical = T_optical_to_base^(-1), 得到 x=0.394m
    # 前边缘 = 相机光学中心 + 4cm余量 (相机外壳 + 安全)
    camera_forward_offset: float = 0.394    # 相机光学中心相对 base_link 前向偏移 (米)
    robot_front_offset: float = 0.40       # base_link 到机器人前边缘距离 (米)

    # ========== 控制参数 ==========
    control_rate: float = 50.0              # 控制循环频率 (Hz)

    # ========== TF 坐标系 ==========
    map_frame: str = "map"                  # 地图坐标系
    base_frame: str = "base_link"           # 机器人基座坐标系


def load_config_from_node(node) -> ApproachConfig:
    """从 ROS2 节点参数加载配置

    Args:
        node: ROS2 节点实例

    Returns:
        ApproachConfig: 加载后的配置对象
    """
    config = ApproachConfig()

    # 参数列表: (参数名, 默认值)
    params = [
        # 阶段1
        ('approach_distance', config.approach_distance),
        ('nav_timeout', config.nav_timeout),
        # 阶段2
        ('align_tolerance', config.align_tolerance),
        ('align_kp', config.align_kp),
        ('align_kd', config.align_kd),
        ('align_max_omega', config.align_max_omega),
        ('align_timeout', config.align_timeout),
        ('min_align_distance', config.min_align_distance),
        # 阶段3
        ('final_approach_speed', config.final_approach_speed),
        ('final_approach_distance', config.final_approach_distance),
        ('emergency_stop_distance', config.emergency_stop_distance),
        ('speed_decel_rate', config.speed_decel_rate),
        ('final_approach_timeout', config.final_approach_timeout),
        # 深度传感器
        ('depth_topic', config.depth_topic),
        ('depth_min_valid', config.depth_min_valid),
        ('depth_max_valid', config.depth_max_valid),
        ('depth_data_timeout', config.depth_data_timeout),
        # 点云处理
        ('depth_downsample', config.depth_downsample),
        ('depth_ground_max_height', config.depth_ground_max_height),
        ('depth_obstacle_min_height', config.depth_obstacle_min_height),
        ('depth_obstacle_max_height', config.depth_obstacle_max_height),
        ('depth_detect_width', config.depth_detect_width),
        # 外参
        ('extrinsics_file', config.extrinsics_file),
        # 机器人几何
        ('camera_forward_offset', config.camera_forward_offset),
        ('robot_front_offset', config.robot_front_offset),
        # 控制
        ('control_rate', config.control_rate),
        # TF
        ('map_frame', config.map_frame),
        ('base_frame', config.base_frame),
    ]

    # 声明所有参数
    for name, default in params:
        node.declare_parameter(name, default)

    # 读取参数值
    config.approach_distance = node.get_parameter('approach_distance').value
    config.nav_timeout = node.get_parameter('nav_timeout').value
    config.align_tolerance = node.get_parameter('align_tolerance').value
    config.align_kp = node.get_parameter('align_kp').value
    config.align_kd = node.get_parameter('align_kd').value
    config.align_max_omega = node.get_parameter('align_max_omega').value
    config.align_timeout = node.get_parameter('align_timeout').value
    config.min_align_distance = node.get_parameter('min_align_distance').value
    config.final_approach_speed = node.get_parameter('final_approach_speed').value
    config.final_approach_distance = node.get_parameter('final_approach_distance').value
    config.emergency_stop_distance = node.get_parameter('emergency_stop_distance').value
    config.speed_decel_rate = node.get_parameter('speed_decel_rate').value
    config.final_approach_timeout = node.get_parameter('final_approach_timeout').value
    config.depth_topic = node.get_parameter('depth_topic').value
    config.depth_min_valid = node.get_parameter('depth_min_valid').value
    config.depth_max_valid = node.get_parameter('depth_max_valid').value
    config.depth_data_timeout = node.get_parameter('depth_data_timeout').value
    config.depth_downsample = node.get_parameter('depth_downsample').value
    config.depth_ground_max_height = node.get_parameter('depth_ground_max_height').value
    config.depth_obstacle_min_height = node.get_parameter('depth_obstacle_min_height').value
    config.depth_obstacle_max_height = node.get_parameter('depth_obstacle_max_height').value
    config.depth_detect_width = node.get_parameter('depth_detect_width').value
    config.extrinsics_file = node.get_parameter('extrinsics_file').value
    config.camera_forward_offset = node.get_parameter('camera_forward_offset').value
    config.robot_front_offset = node.get_parameter('robot_front_offset').value
    config.control_rate = node.get_parameter('control_rate').value
    config.map_frame = node.get_parameter('map_frame').value
    config.base_frame = node.get_parameter('base_frame').value

    return config
