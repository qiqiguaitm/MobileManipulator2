#!/usr/bin/env python3
"""
Approach Navigator - 三阶段精确接近导航器

阶段1 (Navigate): Nav2 导航到接近点 (距目标 45cm)
阶段2 (Align): 原地旋转对准目标方向
阶段3 (Final Approach): 相机深度测距 + 低速精确接近
"""

import math
import time
import threading
from typing import Optional, Callable

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, Twist
import tf2_ros

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from .config import ApproachConfig, load_config_from_node
from .nav_types import NavStage, ApproachResult
from .depth_sensor import DepthSensor
from .utils import normalize_angle, get_yaw_from_quaternion, get_robot_pose, compute_approach_pose


class ApproachNavigator(Node):
    """三阶段接近导航器

    阶段1: Nav2 全局导航到接近点
        - 接近点距离目标 approach_distance (0.45m)
        - 必须大于 local_costmap 的 inflation_radius (0.35m)

    阶段2: PD 控制器原地旋转对齐
        - 使机器人正面朝向目标
        - 对齐容差 5°

    阶段3: 相机深度测距精确接近
        - 使用深度相机替代 LiDAR 测距
        - 目标: 前边缘距目标 8cm
        - 紧急刹车: 3cm
    """

    def __init__(self, config: ApproachConfig = None):
        """初始化接近导航器

        Args:
            config: 配置对象，None 则从 ROS 参数加载
        """
        super().__init__('approach_navigator')

        # 加载配置
        self.config = config or load_config_from_node(self)

        # 导航状态
        self.stage = NavStage.IDLE              # 当前阶段
        self._stage_lock = threading.Lock()      # 状态锁
        self._cancel_requested = False           # 取消标志

        # 目标信息
        self.target_map_position: Optional[Point] = None    # 目标位置 (map 坐标系)
        self.target_yaw: float = 0.0                        # 目标航向

        # TF2 坐标变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 导航器
        self.nav = BasicNavigator()

        # 深度传感器 (替代 LiDAR)
        self.depth_sensor = DepthSensor(self, self.config)

        # 速度发布器
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._log_config()

    def _log_config(self):
        """输出配置信息"""
        self.get_logger().info("接近导航器已初始化")
        self.get_logger().info(f"  接近距离: {self.config.approach_distance}m")
        self.get_logger().info(f"  最终距离: {self.config.final_approach_distance}m")
        self.get_logger().info(f"  深度话题: {self.config.depth_topic}")

    # =========================================================================
    # 主接口
    # =========================================================================

    def approach_to_target(
        self,
        target_position: Point,
        status_callback: Callable[[NavStage, str], None] = None
    ) -> ApproachResult:
        """直接接近目标位置 (简化接口)

        感知系统可直接调用此方法，传入目标位置即可执行三阶段导航。
        内部自动计算接近位姿。

        Args:
            target_position: 目标物体在 map 坐标系中的位置
            status_callback: 阶段状态回调

        Returns:
            ApproachResult: 导航结果
        """
        # 获取当前机器人位置
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos is None:
            self.get_logger().error("无法获取机器人位姿")
            return ApproachResult(False, "TF_ERROR", "无法获取机器人位姿", stage=0)

        # 计算接近位姿
        approach_pose = compute_approach_pose(
            target_position,
            robot_pos,
            approach_distance=self.config.approach_distance,
            robot_front_offset=self.config.robot_front_offset
        )

        if approach_pose is None:
            # 已经在目标附近，直接执行阶段2和3
            self.get_logger().info("已在目标附近，跳过阶段1")
            self.target_map_position = target_position
            self._update_target_yaw()

            # 执行阶段2: 对齐
            success, msg = self._do_alignment()
            if not success:
                return ApproachResult(False, "ALIGN_FAILED", msg, stage=2)

            # 执行阶段3: 精确接近
            success, msg, final_dist = self._do_final_approach()
            if not success:
                return ApproachResult(False, "APPROACH_FAILED", msg, stage=3)

            return ApproachResult(True, final_distance=final_dist)

        # 执行完整三阶段导航
        return self.approach_to_pose(approach_pose, target_position, status_callback)

    def approach_to_pose(
        self,
        approach_pose: PoseStamped,
        target_position: Optional[Point] = None,
        status_callback: Callable[[NavStage, str], None] = None
    ) -> ApproachResult:
        """执行三阶段导航到接近位姿

        Args:
            approach_pose: 预计算的接近位姿 (map 坐标系)
            target_position: 目标物体位置，用于阶段2/3。None 则从 approach_pose 反推
            status_callback: 阶段状态回调，参数为 (NavStage, 消息字符串)

        Returns:
            ApproachResult: 导航结果
        """
        self._cancel_requested = False

        # 计算目标位置 (如果未提供)
        if target_position is None:
            target_position = self._compute_target_from_approach_pose(approach_pose)
        self.target_map_position = target_position

        def notify(stage: NavStage, msg: str):
            """通知阶段变化"""
            with self._stage_lock:
                self.stage = stage
            if status_callback:
                status_callback(stage, msg)
            self.get_logger().info(f"{stage.name}: {msg}")

        # ========== 阶段1: Nav2 导航 ==========
        notify(NavStage.NAVIGATING, "导航到接近点...")
        result1 = self._do_navigation(approach_pose)

        if self._cancel_requested:
            notify(NavStage.FAILED, "用户取消")
            return ApproachResult(False, "NAV_CANCELLED", "用户取消", stage=1)

        if not result1:
            notify(NavStage.FAILED, "导航失败")
            return ApproachResult(False, "NAV_FAILED", "Nav2 导航失败", stage=1)

        # 更新目标航向
        self._update_target_yaw()

        # ========== 阶段2: 对齐 ==========
        notify(NavStage.ALIGNING, "对准目标方向...")
        success, msg = self._do_alignment()

        if self._cancel_requested:
            notify(NavStage.FAILED, "用户取消")
            return ApproachResult(False, "NAV_CANCELLED", "用户取消", stage=2)

        if not success:
            notify(NavStage.FAILED, f"对齐失败: {msg}")
            return ApproachResult(False, "ALIGN_FAILED", msg, stage=2)

        # ========== 阶段3: 精确接近 ==========
        notify(NavStage.FINAL_APPROACH, "精确接近中...")
        success, msg, final_dist = self._do_final_approach()

        if self._cancel_requested:
            notify(NavStage.FAILED, "用户取消")
            return ApproachResult(False, "NAV_CANCELLED", "用户取消", stage=3)

        if not success:
            notify(NavStage.FAILED, f"接近失败: {msg}")
            return ApproachResult(False, "APPROACH_FAILED", msg, stage=3)

        notify(NavStage.ARRIVED, f"已到达，距离: {final_dist:.3f}m")
        return ApproachResult(True, final_distance=final_dist)

    def cancel(self):
        """取消导航"""
        self._cancel_requested = True
        self.nav.cancelTask()
        self._stop_robot()
        self.get_logger().info("导航已取消")

    # =========================================================================
    # 阶段1: Nav2 导航
    # =========================================================================

    def _do_navigation(self, goal: PoseStamped) -> bool:
        """执行 Nav2 导航到目标点

        Args:
            goal: 目标位姿

        Returns:
            bool: True 表示成功到达
        """
        self.get_logger().info(
            f"Nav2 目标: ({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f})"
        )

        # 发送目标
        self.nav.goToPose(goal)

        # 等待完成
        start_time = time.time()
        while not self.nav.isTaskComplete():
            # 检查取消
            if self._cancel_requested:
                self.nav.cancelTask()
                return False

            # 检查超时
            if time.time() - start_time > self.config.nav_timeout:
                self.get_logger().warn("导航超时")
                self.nav.cancelTask()
                return False

            time.sleep(0.1)

        # 检查结果
        result = self.nav.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Nav2 导航成功")
            return True

        self.get_logger().warn(f"Nav2 导航失败: {result}")
        return False

    # =========================================================================
    # 阶段2: 对齐
    # =========================================================================

    def _do_alignment(self) -> tuple:
        """执行 PD 控制器对齐

        原地旋转使机器人正面朝向目标

        Returns:
            (bool, str): (成功标志, 错误消息)
        """
        # 检查距离 - 太近则跳过
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos:
            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < self.config.min_align_distance:
                self.get_logger().info(f"距离过近 ({dist:.3f}m)，跳过对齐")
                return True, ""

        # 控制循环
        rate = self.create_rate(20)     # 20 Hz
        start_time = time.time()
        last_error = 0.0                # 上次角度误差
        last_time = start_time          # 上次时间

        while rclpy.ok() and not self._cancel_requested:
            current_time = time.time()

            # 超时检查
            if current_time - start_time > self.config.align_timeout:
                self._stop_robot()
                return False, "对齐超时"

            # 获取当前位姿
            robot_pos, robot_yaw = self._get_robot_pose()
            if robot_pos is None:
                rate.sleep()
                continue

            # 计算角度误差
            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            target_yaw = math.atan2(dy, dx)                     # 目标航向
            error = normalize_angle(target_yaw - robot_yaw)     # 角度误差

            # 检查是否已对齐
            if abs(error) < self.config.align_tolerance:
                self._stop_robot()
                self.get_logger().info(f"对齐完成，误差: {math.degrees(error):.2f}°")
                return True, ""

            # PD 控制
            actual_dt = current_time - last_time
            if actual_dt > 0.001:
                d_error = (error - last_error) / actual_dt      # 误差变化率
            else:
                d_error = 0.0

            # 计算角速度
            omega = self.config.align_kp * error + self.config.align_kd * d_error
            omega = max(-self.config.align_max_omega, min(self.config.align_max_omega, omega))

            # 发布速度
            cmd = Twist()
            cmd.angular.z = omega
            self.cmd_vel_pub.publish(cmd)

            # 更新状态
            last_error = error
            last_time = current_time
            rate.sleep()

        self._stop_robot()
        return False, "被中断"

    # =========================================================================
    # 阶段3: 精确接近
    # =========================================================================

    def _do_final_approach(self) -> tuple:
        """执行相机深度测距精确接近

        策略：
        1. 有点云数据时：使用点云精确测距
        2. 点云丢失时：基于最后已知距离 + 里程计增量开环前进
        3. 点云恢复时：切回点云模式

        Returns:
            (bool, str, float): (成功标志, 错误消息, 最终距离)
        """
        rate = self.create_rate(self.config.control_rate)
        start_time = time.time()

        self.get_logger().info(f"开始精确接近，目标距离: {self.config.final_approach_distance}m")
        self.get_logger().info(f"  robot_front_offset: {self.config.robot_front_offset}m")
        self.get_logger().info(f"  内参: {'已加载' if self.depth_sensor.has_intrinsics else '未加载'}")
        self.get_logger().info(f"  外参: {'已加载' if self.depth_sensor.has_extrinsics else '未加载'}")

        # 速度平滑参数
        dt = 1.0 / self.config.control_rate
        current_speed = 0.0  # 从0开始，逐渐加速
        max_accel = 0.15     # 最大加速率 m/s²
        max_decel = 0.3      # 最大减速率 m/s²

        # 开环控制：target_distance 是障碍物到机器人初始位置的距离
        # 如果检测到更近的物体，更新target；否则开环
        target_distance: float = None           # 目标距离 (相对初始位置)
        total_traveled: float = 0.0             # 累计行驶距离 (速度积分)

        while rclpy.ok() and not self._cancel_requested:
            # 超时检查
            if time.time() - start_time > self.config.final_approach_timeout:
                self._stop_robot()
                return False, "接近超时", 0.0

            # 获取最近障碍物 (base_link 坐标系)
            obstacle_x = self.depth_sensor.get_target_depth()

            # 累积行驶距离 (速度 × 时间)
            total_traveled += current_speed * dt

            # ========== 距离估算 ==========
            if target_distance is None:
                # 等待首次点云检测
                if obstacle_x is not None:
                    raw_clearance = obstacle_x - self.config.robot_front_offset
                    target_distance = raw_clearance  # 初始目标距离
                    total_traveled = 0.0
                    front_clearance = raw_clearance
                    self.get_logger().info(
                        f"[初始化] target={target_distance:.3f}m"
                    )
                else:
                    self.get_logger().info(
                        f"[等待] 无点云数据...",
                        throttle_duration_sec=1.0
                    )
                    rate.sleep()
                    continue
            else:
                # 检查是否检测到更近的障碍物
                if obstacle_x is not None:
                    current_clearance = obstacle_x - self.config.robot_front_offset
                    # 转换为相对初始位置的距离
                    obstacle_from_start = current_clearance + total_traveled

                    if obstacle_from_start < target_distance:
                        # 检测到更近的障碍物，更新目标
                        target_distance = obstacle_from_start
                        self.get_logger().warn(
                            f"[更新] 检测到更近障碍物! target={target_distance:.3f}m"
                        )

                # 开环计算当前距离
                front_clearance = target_distance - total_traveled
                self.get_logger().info(
                    f"[开环] 距离={front_clearance:.3f}m (target={target_distance:.3f}m, "
                    f"已行驶={total_traveled:.3f}m), 速度={current_speed:.3f}m/s",
                    throttle_duration_sec=0.5
                )

            # ========== 停止条件 ==========
            if front_clearance <= self.config.emergency_stop_distance:
                self._stop_robot()
                self.get_logger().warn(f"紧急刹车! 距离: {front_clearance:.3f}m")
                return True, "", front_clearance

            if front_clearance <= self.config.final_approach_distance:
                self._stop_robot()
                self.get_logger().info(f"到达目标! 距离: {front_clearance:.3f}m")
                return True, "", front_clearance

            # ========== 平滑速度控制 ==========
            remaining = front_clearance - self.config.final_approach_distance

            # 开环模式最大速度
            max_speed = self.config.final_approach_speed

            # 目标速度：基于剩余距离的平滑曲线
            if remaining > 0.3:
                target_speed = max_speed
            else:
                # sqrt 曲线平滑减速
                ratio = remaining / 0.3
                target_speed = 0.02 + (max_speed - 0.02) * math.sqrt(ratio)

            # 速度平滑：限制加速和减速率
            speed_diff = target_speed - current_speed
            if speed_diff > 0:
                max_change = max_accel * dt
                current_speed += min(speed_diff, max_change)
            else:
                max_change = max_decel * dt
                current_speed += max(speed_diff, -max_change)

            # 确保速度在合理范围
            current_speed = max(0.0, min(max_speed, current_speed))

            # 发布速度
            cmd = Twist()
            cmd.linear.x = current_speed
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)

            rate.sleep()

        self._stop_robot()
        return False, "被中断", 0.0

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _get_robot_pose(self):
        """获取机器人在 map 坐标系中的位姿

        Returns:
            (Point, float): 位置和航向角
            (None, None): 查询失败
        """
        return get_robot_pose(
            self.tf_buffer,
            self.config.map_frame,
            self.config.base_frame,
            self.get_logger()
        )

    def _update_target_yaw(self):
        """更新目标航向 (阶段1结束后调用)"""
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos:
            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            self.target_yaw = math.atan2(dy, dx)

    def _compute_target_from_approach_pose(self, approach_pose: PoseStamped) -> Point:
        """从接近位姿反推目标位置

        逆向计算: 目标 = 接近点 + 方向 * offset

        Args:
            approach_pose: 接近位姿

        Returns:
            Point: 目标位置
        """
        yaw = get_yaw_from_quaternion(approach_pose.pose.orientation)
        total_offset = self.config.robot_front_offset + self.config.approach_distance

        target = Point()
        target.x = approach_pose.pose.position.x + total_offset * math.cos(yaw)
        target.y = approach_pose.pose.position.y + total_offset * math.sin(yaw)
        target.z = 0.0

        return target

    def _stop_robot(self):
        """停止机器人"""
        self.cmd_vel_pub.publish(Twist())
