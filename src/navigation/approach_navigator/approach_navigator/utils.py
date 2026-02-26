#!/usr/bin/env python3
"""
Approach Navigator 工具函数模块
"""

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
import tf2_ros


def compute_approach_pose(
    target_position: Point,
    robot_position: Point,
    approach_distance: float = 0.45,
    robot_front_offset: float = 0.434,
    frame_id: str = "map"
) -> Optional[PoseStamped]:
    """计算接近位姿

    根据目标位置和机器人当前位置，计算机器人应该导航到的接近点。
    接近点位于目标前方，保持安全距离。

    Args:
        target_position: 目标位置 (map 坐标系)
        robot_position: 机器人当前位置 (map 坐标系)
        approach_distance: 接近点到目标的距离 (米)，默认 0.45m
        robot_front_offset: base_link 到机器人前边缘距离 (米)，默认 0.434m (相机0.394m + 4cm余量)
        frame_id: 输出位姿的坐标系，默认 "map"

    Returns:
        PoseStamped: 接近位姿，机器人应导航到此位置
        None: 如果机器人已经足够接近目标

    几何关系:
        total_offset = robot_front_offset + approach_distance
        接近点 = 目标位置 - 方向向量 * total_offset
    """
    # 计算方向向量 (机器人 -> 目标)
    dx = target_position.x - robot_position.x   # X 方向差
    dy = target_position.y - robot_position.y   # Y 方向差
    dist = math.sqrt(dx**2 + dy**2)             # 距离

    # 检查是否已经足够接近
    total_offset = robot_front_offset + approach_distance
    if dist <= total_offset + 0.1:  # 10cm 容差
        return None

    # 归一化方向向量
    dx_norm = dx / dist     # 单位向量 X
    dy_norm = dy / dist     # 单位向量 Y

    # 构建接近位姿
    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.header.stamp = rclpy.clock.Clock().now().to_msg()

    # 位置: 目标前方 total_offset 处
    goal.pose.position.x = target_position.x - dx_norm * total_offset
    goal.pose.position.y = target_position.y - dy_norm * total_offset
    goal.pose.position.z = 0.0

    # 朝向: 面向目标
    yaw = math.atan2(dy_norm, dx_norm)              # 计算航向角
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = math.sin(yaw / 2)     # 四元数 z
    goal.pose.orientation.w = math.cos(yaw / 2)     # 四元数 w

    return goal


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi] 范围

    Args:
        angle: 输入角度 (弧度)

    Returns:
        float: 归一化后的角度 (弧度)，范围 [-pi, pi]
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_yaw_from_quaternion(q) -> float:
    """从四元数提取航向角 (yaw)

    Args:
        q: 四元数对象，需有 x, y, z, w 属性

    Returns:
        float: 航向角 (弧度)
    """
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def get_robot_pose(
    tf_buffer: tf2_ros.Buffer,
    map_frame: str,
    base_frame: str,
    logger=None
) -> Tuple[Optional[Point], Optional[float]]:
    """获取机器人在 map 坐标系中的位姿

    通过 TF 查询 base_frame 在 map_frame 中的变换

    Args:
        tf_buffer: TF2 缓存对象
        map_frame: 地图坐标系名称
        base_frame: 机器人基座坐标系名称
        logger: 日志记录器，用于输出警告

    Returns:
        (Point, float): 位置和航向角 (弧度)
        (None, None): 查询失败时返回
    """
    try:
        # 查询 TF 变换，超时 1 秒
        transform = tf_buffer.lookup_transform(
            map_frame,          # 目标坐标系
            base_frame,         # 源坐标系
            rclpy.time.Time(),  # 最新时间
            timeout=Duration(seconds=1.0)
        )

        # 提取位置
        pos = Point()
        pos.x = transform.transform.translation.x
        pos.y = transform.transform.translation.y
        pos.z = transform.transform.translation.z

        # 提取航向角
        yaw = get_yaw_from_quaternion(transform.transform.rotation)

        return pos, yaw

    except Exception as e:
        if logger:
            logger.warn(f"获取机器人位姿失败: {e}", throttle_duration_sec=1.0)
        return None, None
