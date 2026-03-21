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
    # 接近距离必须 > global_costmap 的 inflation_radius (0.45m)
    approach_distance: float = 0.40     # 接近点到目标的距离 (米)
    nav_timeout: float = 60.0           # Nav2 导航超时时间 (秒)

    # ========== 阶段2: 原地旋转对齐 ==========
    align_tolerance: float = 0.087      # 对齐容差，5° → 8° (弧度)，宁愿误差大不振荡
    align_kp: float = 0.35              # PD控制器 P增益 (0.6→0.35，降低响应减少超调)
    align_kd: float = 0.4               # PD控制器 D增益 (0.3→0.4，加大阻尼抑制振荡)
    align_max_omega: float = 0.25       # 最大角速度 (0.4→0.25 弧度/秒，限制转速上限)
    align_max_alpha: float = 0.6        # 最大角加速度 (1.0→0.6 弧度/秒²)，更平滑
    align_min_omega: float = 0.02       # 死区角速度 (弧度/秒)，低于此值发0避免抖动
    align_timeout: float = 10.0         # 对齐超时时间 (秒)
    min_align_distance: float = 0.10    # 距离小于此值跳过对齐 (米)
    align_settle_time: float = 0.8      # 粗对齐后停稳等待时间 (秒)
    depth_warmup_frames: int = 5        # 深度传感器预热帧数，丢弃前N帧不稳定数据
    camera_lateral_offset: float = -0.013 # 机器人中心在相机光学系的x偏移 (米，相机光学系x=右，-0.013=中心在相机左侧1.3cm)

    # ========== 阶段3: 精确接近 ==========
    final_approach_speed: float = 0.12      # 最大前进速度 (米/秒)
    final_approach_distance: float = 0.08   # 目标停止距离：前边缘到目标 + 3cm安全余量 (米)
    emergency_stop_distance: float = 0.03   # 紧急刹车距离 (米)
    speed_decel_rate: float = 0.5           # 最大减速率 (米/秒²)
    final_approach_timeout: float = 15.0    # 精确接近超时 (秒)

    # ========== 深度传感器参数 ==========
    depth_topic: str = "/camera/chassis/aligned_depth_to_color/image_raw"  # 深度图话题
    depth_min_valid: float = 0.10           # 最小有效深度，D435约0.1m (米)
    depth_max_valid: float = 2.0            # 最大有效深度 (米)
    depth_data_timeout: float = 0.5         # 深度数据超时判定 (秒)

    # ========== 点云处理参数 ==========
    # 相机坐标系: y 向下为正, 相机离地约 15cm → 地面在 y ≈ +0.15m
    # 空中物体 y < 0 (负), 地面 y ≈ +0.15, 地面以下 y > 0.15
    depth_downsample: int = 4               # 深度图降采样因子 (4=每4像素取1个)
    depth_ground_max_height: float = 0.15   # 地面高度: +15cm (相机离地15cm, y向下为正)
    depth_obstacle_min_height: float = -0.50 # ROI上方截止: -50cm (y向上为负, 排除过高处)
    depth_obstacle_max_height: float = 0.22  # ROI下方截止: +22cm (包含地面y=0.15, 供RANSAC)
    depth_detect_width: float = 0.4         # 前方检测宽度 (米)，左右各此值

    # ========== 去噪参数 ==========
    outlier_radius: float = 0.05            # 半径异常值去除: 搜索半径 (米)
    outlier_min_neighbors: int = 3          # 半径异常值去除: 最少邻居数

    # ========== RANSAC 地面拟合参数 ==========
    ransac_distance_threshold: float = 0.03 # 点到平面距离阈值 (米)
    ransac_max_iterations: int = 50         # 最大迭代次数
    ransac_min_inlier_ratio: float = 0.3    # 最小内点比例
    ground_candidate_margin: float = 0.05   # 地面候选点 y 下限容差 (米)
    ground_inlier_min_dist: float = -0.005  # 地面内点有符号距离下限 (米, 平面上方5mm以内视为地面)
    ground_inlier_max_dist: float = 0.03    # 地面内点有符号距离上限 (米, 平面下方)

    # ========== 聚类参数 ==========
    cluster_tolerance: float = 0.05         # DBSCAN eps / KDTree 邻域半径 (米)
    cluster_min_size: int = 50              # 最小聚类点数
    cluster_max_size: int = 10000           # 最大聚类点数

    # ========== 机器人几何参数 ==========
    # robot_front_offset: 用于导航计算接近点 (total_offset = robot_front_offset + approach_distance)
    # 相机距离: Phase 3 精确接近直接使用相机测量距离，不加 offset
    camera_forward_offset: float = 0.0       # 已废弃，保留仅用于兼容
    robot_front_offset: float = 0.5         # base_link 到机器人前边缘

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
        ('align_max_alpha', config.align_max_alpha),
        ('align_min_omega', config.align_min_omega),
        ('align_timeout', config.align_timeout),
        ('min_align_distance', config.min_align_distance),
        ('align_settle_time', config.align_settle_time),
        ('depth_warmup_frames', config.depth_warmup_frames),
        ('camera_lateral_offset', config.camera_lateral_offset),
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
        # 去噪
        ('outlier_radius', config.outlier_radius),
        ('outlier_min_neighbors', config.outlier_min_neighbors),
        # RANSAC 地面拟合
        ('ransac_distance_threshold', config.ransac_distance_threshold),
        ('ransac_max_iterations', config.ransac_max_iterations),
        ('ransac_min_inlier_ratio', config.ransac_min_inlier_ratio),
        ('ground_candidate_margin', config.ground_candidate_margin),
        ('ground_inlier_min_dist', config.ground_inlier_min_dist),
        ('ground_inlier_max_dist', config.ground_inlier_max_dist),
        # 聚类
        ('cluster_tolerance', config.cluster_tolerance),
        ('cluster_min_size', config.cluster_min_size),
        ('cluster_max_size', config.cluster_max_size),
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
    config.align_max_alpha = node.get_parameter('align_max_alpha').value
    config.align_min_omega = node.get_parameter('align_min_omega').value
    config.align_timeout = node.get_parameter('align_timeout').value
    config.min_align_distance = node.get_parameter('min_align_distance').value
    config.align_settle_time = node.get_parameter('align_settle_time').value
    config.depth_warmup_frames = node.get_parameter('depth_warmup_frames').value
    config.camera_lateral_offset = node.get_parameter('camera_lateral_offset').value
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
    config.outlier_radius = node.get_parameter('outlier_radius').value
    config.outlier_min_neighbors = node.get_parameter('outlier_min_neighbors').value
    config.ransac_distance_threshold = node.get_parameter('ransac_distance_threshold').value
    config.ransac_max_iterations = node.get_parameter('ransac_max_iterations').value
    config.ransac_min_inlier_ratio = node.get_parameter('ransac_min_inlier_ratio').value
    config.ground_candidate_margin = node.get_parameter('ground_candidate_margin').value
    config.ground_inlier_min_dist = node.get_parameter('ground_inlier_min_dist').value
    config.ground_inlier_max_dist = node.get_parameter('ground_inlier_max_dist').value
    config.cluster_tolerance = node.get_parameter('cluster_tolerance').value
    config.cluster_min_size = node.get_parameter('cluster_min_size').value
    config.cluster_max_size = node.get_parameter('cluster_max_size').value
    config.camera_forward_offset = node.get_parameter('camera_forward_offset').value
    config.robot_front_offset = node.get_parameter('robot_front_offset').value
    config.control_rate = node.get_parameter('control_rate').value
    config.map_frame = node.get_parameter('map_frame').value
    config.base_frame = node.get_parameter('base_frame').value

    return config
