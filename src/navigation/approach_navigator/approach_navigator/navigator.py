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
from nav_msgs.msg import Odometry
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
        self._initial_depth = None              # 用户确认时的初始深度
        self._locked_target_cx = None           # 阶段2→3 锁定目标 cam_x
        self._locked_target_cz = None           # 阶段2→3 锁定目标 cam_z

        # 目标信息
        self.target_map_position: Optional[Point] = None    # 目标位置 (map 坐标系)
        self.target_yaw: float = 0.0                        # 目标航向

        # TF2 坐标变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 导航器
        self.nav = BasicNavigator()

        # 深度传感器 (替代 LiDAR)
        self.depth_sensor = DepthSensor(self, self.config, self.tf_buffer)

        # 速度发布器
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 里程计 (用于阶段3实际行驶距离测量，替代指令速度积分)
        self._odom_lock = threading.Lock()
        self._odom_x: Optional[float] = None
        self._odom_y: Optional[float] = None
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)

        self._log_config()

    def _odom_callback(self, msg: Odometry):
        with self._odom_lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y

    def _get_odom_traveled(self, start_x: float, start_y: float) -> float:
        """返回从起点到当前 odom 位置的直线距离"""
        with self._odom_lock:
            if self._odom_x is None:
                return 0.0
            dx = self._odom_x - start_x
            dy = self._odom_y - start_y
        return math.sqrt(dx * dx + dy * dy)

    def _log_config(self):
        """输出配置信息"""
        self.get_logger().info("接近导航器已初始化")
        self.get_logger().info(f"  接近距离: {self.config.approach_distance}m")
        self.get_logger().info(f"  最终距离: {self.config.final_approach_distance}m")
        self.get_logger().info(f"  深度话题: {self.config.depth_topic}")

    # =========================================================================
    # 公共接口 (供外部调用)
    # =========================================================================

    def get_current_depth(self) -> Optional[float]:
        """获取当前障碍物距离 (到机器人前沿)"""
        return self.depth_sensor.get_target_depth()

    def set_initial_depth(self, depth: float):
        """设置阶段3初始深度（用户确认时的深度）"""
        self._initial_depth = depth

    def get_initial_depth(self) -> Optional[float]:
        """获取阶段3初始深度，如果没有则获取当前深度"""
        if self._initial_depth is not None:
            return self._initial_depth
        return self.get_current_depth()

    def clear_initial_depth(self):
        """清除初始深度"""
        self._initial_depth = None

    def back_up(self, distance_m: float, speed: float = 0.05, timeout: float = 10.0) -> bool:
        """开环后退指定距离，用 odom 测量实际位移。

        后退距离 <= 15cm，不做后方避障 — 无后方传感器，短距风险可控。

        Args:
            distance_m: 后退距离 (m)
            speed: 后退速度 (m/s)，正值，内部取负
            timeout: 超时 (s)

        Returns:
            bool: 到达目标距离返回 True
        """
        with self._odom_lock:
            start_x = self._odom_x
            start_y = self._odom_y
        if start_x is None:
            self.get_logger().error("back_up: 里程计不可用")
            return False

        self.get_logger().info(f"back_up: 后退 {distance_m:.3f}m @ {speed:.3f}m/s")
        start_time = time.time()
        cmd = Twist()
        cmd.linear.x = -abs(speed)

        while rclpy.ok() and not self._cancel_requested:
            if time.time() - start_time > timeout:
                self._stop_robot()
                self.get_logger().warn("back_up: 超时")
                return False

            traveled = self._get_odom_traveled(start_x, start_y)
            if traveled >= distance_m:
                self._stop_robot()
                self.get_logger().info(f"back_up: 完成，实际后退 {traveled:.3f}m")
                return True

            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)

        self._stop_robot()
        return False

    def move_forward(self, distance_m: float, speed: float = 0.08, timeout: float = 10.0) -> bool:
        """开环前进指定距离，用 odom 测量实际位移。

        Args:
            distance_m: 前进距离 (m)，正值
            speed: 前进速度 (m/s)，正值
            timeout: 超时 (s)

        Returns:
            bool: 到达目标距离返回 True
        """
        with self._odom_lock:
            start_x = self._odom_x
            start_y = self._odom_y
        if start_x is None:
            self.get_logger().error("move_forward: 里程计不可用")
            return False

        self.get_logger().info(f"move_forward: 前进 {distance_m:.3f}m @ {speed:.3f}m/s")
        start_time = time.time()
        cmd = Twist()
        cmd.linear.x = abs(speed)

        while rclpy.ok() and not self._cancel_requested:
            if time.time() - start_time > timeout:
                self._stop_robot()
                self.get_logger().warn("move_forward: 超时")
                return False

            traveled = self._get_odom_traveled(start_x, start_y)
            if traveled >= distance_m:
                self._stop_robot()
                self.get_logger().info(f"move_forward: 完成，实际前进 {traveled:.3f}m")
                return True

            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)

        self._stop_robot()
        return False

    # =========================================================================
    # Re-approach: 闭环深度驱动
    # =========================================================================

    def rotate_by_angle(self, angle_rad: float, timeout: float = 8.0) -> tuple:
        """原地旋转指定角度 (P控制, 基于TF反馈)

        Args:
            angle_rad: 旋转角度 (rad), 正=逆时针(左转), 负=顺时针(右转)
            timeout: 超时 (s)

        Returns:
            (bool, str): (成功, 错误消息)
        """
        if abs(angle_rad) < 0.02:  # ~1° 无需旋转
            return True, ""

        robot_pos, start_yaw = self._get_robot_pose()
        if start_yaw is None:
            return False, "TF不可用"

        target_yaw = start_yaw + angle_rad
        cfg = self.config
        start_time = time.time()

        self.get_logger().info(
            f"[rotate] 旋转 {math.degrees(angle_rad):.1f}°, "
            f"当前={math.degrees(start_yaw):.1f}° → 目标={math.degrees(target_yaw):.1f}°")

        while rclpy.ok() and not self._cancel_requested:
            if time.time() - start_time > timeout:
                self._stop_robot()
                return False, "旋转超时"

            _, current_yaw = self._get_robot_pose()
            if current_yaw is None:
                time.sleep(0.05)
                continue

            error = normalize_angle(target_yaw - current_yaw)

            if abs(error) < cfg.align_tolerance:
                self._stop_robot()
                self.get_logger().info(
                    f"[rotate] 完成, 误差={math.degrees(error):.1f}°, "
                    f"耗时={time.time()-start_time:.1f}s")
                return True, ""

            omega = max(-cfg.align_max_omega,
                        min(cfg.align_max_omega, cfg.align_kp * error))
            if 0 < abs(omega) < cfg.align_min_omega:
                omega = math.copysign(cfg.align_min_omega, omega)

            cmd = Twist()
            cmd.angular.z = omega
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)

        self._stop_robot()
        return False, "被中断"

    def depth_reapproach(self, max_travel: float = 0.30,
                        timeout: float = 10.0) -> tuple:
        """闭环深度驱动 re-approach: 精对齐 + 闭环前进

        1. 深度聚类精对齐 (对准目标)
        2. 闭环前进直到 final_approach_distance 或行驶达 max_travel

        Args:
            max_travel: 最大前进距离 (m)，由调用方根据 observe base_x 计算
            timeout: 超时 (s)

        Returns:
            (bool, str, float): (成功, 错误消息, 最终距离)
        """
        self.depth_sensor.activate()
        try:
            # 精对齐
            align_ok, align_msg = self._do_alignment_by_depth()
            if not align_ok:
                return False, f"精对齐失败: {align_msg}", 0.0
            time.sleep(0.15)  # 停稳

            # 闭环前进
            return self._depth_reapproach_impl(max_travel, timeout)
        finally:
            self.depth_sensor.deactivate()

    def _depth_reapproach_impl(self, max_travel: float,
                               timeout: float) -> tuple:
        cfg = self.config
        loop_dt = 1.0 / cfg.control_rate
        start_time = time.time()
        noise_tol = 0.05  # 5cm 噪声容差

        # 速度平滑
        current_speed = 0.0
        max_accel = 0.15   # m/s²
        max_decel = 0.3    # m/s²

        target_distance = None  # 接受的最近距离

        # odom 行驶距离跟踪
        with self._odom_lock:
            odom_start_x = self._odom_x
            odom_start_y = self._odom_y
        use_odom = odom_start_x is not None

        # 预热: 等深度传感器收敛
        warmup_start = time.time()
        warmup_count = 0
        while warmup_count < cfg.depth_warmup_frames and time.time() - warmup_start < 3.0:
            if self._cancel_requested:
                return False, "被中断", 0.0
            if self.depth_sensor.get_all_cluster_centroids():
                warmup_count += 1
            time.sleep(0.1)

        self.get_logger().info(
            f"[reapproach] 开始闭环前进，max_travel={max_travel:.3f}m "
            f"目标距离={cfg.final_approach_distance}m")

        while rclpy.ok() and not self._cancel_requested:
            if time.time() - start_time > timeout:
                self._stop_robot()
                return False, "re-approach 超时", target_distance or 0.0

            # odom 行驶距离
            traveled = 0.0
            if use_odom:
                traveled = self._get_odom_traveled(odom_start_x, odom_start_y)
                if traveled >= max_travel:
                    self._stop_robot()
                    self.get_logger().info(
                        f"[reapproach] 到达最大行驶距离! "
                        f"traveled={traveled:.3f}m >= max={max_travel:.3f}m")
                    return True, "", target_distance or 0.0

            # 安全急停: 任意最近障碍物
            nearest_any = self.depth_sensor.get_nearest_obstacle()
            if nearest_any and nearest_any[2] <= cfg.emergency_stop_distance:
                self._stop_robot()
                self.get_logger().warn(
                    f"[reapproach] 紧急刹车! {nearest_any[2]:.3f}m")
                return True, "", nearest_any[2]

            # 取最近聚类
            clusters = self.depth_sensor.get_all_cluster_centroids()
            if clusters:
                nearest = min(clusters, key=lambda c: c[2])
                new_z = nearest[2]

                if target_distance is None:
                    target_distance = new_z
                    self.get_logger().info(
                        f"[reapproach] 初始距离: {target_distance:.3f}m")
                elif new_z < target_distance:
                    target_distance = new_z
                elif new_z <= target_distance + noise_tol:
                    target_distance = new_z  # 噪声容差内，接受并更新
                else:
                    self.get_logger().info(
                        f"[reapproach] 拒绝跳变: {new_z:.3f}m "
                        f"(当前={target_distance:.3f}m)",
                        throttle_duration_sec=1.0)

            if target_distance is None:
                self.get_logger().info(
                    "[reapproach] 等待深度数据...",
                    throttle_duration_sec=1.0)
                time.sleep(loop_dt)
                continue

            # 停止条件: 深度到位
            if target_distance <= cfg.final_approach_distance:
                self._stop_robot()
                self.get_logger().info(
                    f"[reapproach] 到达! 距离={target_distance:.3f}m "
                    f"traveled={traveled:.3f}m")
                return True, "", target_distance

            # 速度控制: 取深度剩余和里程剩余的较小值
            depth_remaining = target_distance - cfg.final_approach_distance
            odom_remaining = max_travel - traveled if use_odom else float('inf')
            remaining = min(depth_remaining, odom_remaining)

            max_speed = cfg.final_approach_speed
            if remaining > 0.3:
                target_speed = max_speed
            else:
                ratio = remaining / 0.3
                target_speed = 0.02 + (max_speed - 0.02) * math.sqrt(ratio)

            speed_diff = target_speed - current_speed
            if speed_diff > 0:
                current_speed += min(speed_diff, max_accel * loop_dt)
            else:
                current_speed += max(speed_diff, -max_decel * loop_dt)
            current_speed = max(0.0, min(max_speed, current_speed))

            cmd = Twist()
            cmd.linear.x = current_speed
            self.cmd_vel_pub.publish(cmd)

            self.get_logger().info(
                f"[reapproach] dist={target_distance:.3f}m "
                f"v={current_speed:.3f}m/s traveled={traveled:.3f}m/{max_travel:.3f}m",
                throttle_duration_sec=0.5)

            time.sleep(loop_dt)

        self._stop_robot()
        return False, "被中断", 0.0

    # =========================================================================
    # 主接口
    # =========================================================================

    def is_reachable(self, target_position: Point) -> bool:
        """用 Nav2 planner 检查物体位置是否可达（仅规划，不执行）"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = target_position.x
        goal.pose.position.y = target_position.y
        goal.pose.orientation.w = 1.0

        path = self.nav.getPath(
            start=PoseStamped(),
            goal=goal,
            planner_id='GridBased',
            use_start=False
        )
        return path is not None and len(path.poses) > 0

    def approach_to_target(
        self,
        target_position: Point,
        status_callback: Callable[[NavStage, str], None] = None,
        skip_stages: set = None,
        depth_only_align: bool = False
    ) -> ApproachResult:
        """直接接近目标位置 (简化接口)

        感知系统可直接调用此方法，传入目标位置即可执行三阶段导航。
        内部自动计算接近位姿。

        Args:
            target_position: 目标物体在 map 坐标系中的位置
            status_callback: 阶段状态回调
            skip_stages: 要跳过的阶段集合，如 {1}, {1,2}, {2,3} 等
            depth_only_align: True 跳过 map 粗对齐，直接深度精对齐 (re-approach)

        Returns:
            ApproachResult: 导航结果
        """
        # 处理 skip_stages 默认值
        if skip_stages is None:
            skip_stages = set()

        # 获取当前机器人位置
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos is None:
            self.get_logger().error("无法获取机器人位姿")
            return ApproachResult(False, "TF_ERROR", "无法获取机器人位姿", stage=0)

        self.get_logger().info(
            f"approach_to_target: input=({target_position.x:.3f},{target_position.y:.3f}), "
            f"robot=({robot_pos.x:.3f},{robot_pos.y:.3f}) yaw={math.degrees(robot_yaw):.1f}°")

        # 计算接近位姿
        approach_pose = compute_approach_pose(
            target_position,
            robot_pos,
            approach_distance=self.config.approach_distance,
            robot_front_offset=self.config.robot_front_offset
        )

        # 如果跳过阶段1且需要接近位姿，报错
        if 1 in skip_stages and approach_pose is not None:
            self.get_logger().warn("跳过阶段1但需要导航到接近点，将继续导航")
            skip_stages = skip_stages - {1}  # 移除阶段1跳过标记

        if approach_pose is None or 1 in skip_stages:
            # 已经在目标附近或跳过阶段1，直接执行阶段2和3
            if approach_pose is None:
                self.get_logger().info("已在目标附近，跳过阶段1")
            else:
                self.get_logger().info("跳过阶段1")
            self.target_map_position = target_position
            self._update_target_yaw()

            def _notify(stage: NavStage, msg: str):
                with self._stage_lock:
                    self.stage = stage
                if status_callback:
                    status_callback(stage, msg)

            # 执行阶段2: 对齐 (除非跳过)
            if 2 not in skip_stages:
                _notify(NavStage.ALIGNING, "对准目标方向...")
                success, msg = self._do_alignment(depth_only=depth_only_align)
                if not success:
                    self.depth_sensor.deactivate()
                    return ApproachResult(False, "ALIGN_FAILED", msg, stage=2)
                time.sleep(0.2)
            else:
                self.get_logger().info("跳过阶段2")

            # 执行阶段3: 精确接近 (除非跳过或已足够近)
            # _notify 同步调用 status_callback，callback 中的 input() 会阻塞直到用户确认
            skip_approach = (self._locked_target_cz is not None
                             and self._locked_target_cz <= 0.25)
            if skip_approach:
                self.get_logger().info(
                    f"目标距相机仅 {self._locked_target_cz:.3f}m，已足够近，跳过精确接近")
                self.depth_sensor.deactivate()
                final_dist = self._locked_target_cz
            elif 3 not in skip_stages:
                _notify(NavStage.FINAL_APPROACH, "精确接近中...")
                success, msg, final_dist = self._do_final_approach()
                if not success:
                    return ApproachResult(False, "APPROACH_FAILED", msg, stage=3)
            else:
                self.get_logger().info("跳过阶段3")
                self.depth_sensor.deactivate()  # 阶段2可能已激活，清理
                final_dist = 0.0

            return ApproachResult(True, final_distance=final_dist)

        # 执行完整三阶段导航
        return self.approach_to_pose(
            approach_pose, target_position, status_callback, skip_stages,
            depth_only_align=depth_only_align)

    def approach_to_pose(
        self,
        approach_pose: PoseStamped,
        target_position: Optional[Point] = None,
        status_callback: Callable[[NavStage, str], None] = None,
        skip_stages: set = None,
        depth_only_align: bool = False
    ) -> ApproachResult:
        """执行三阶段导航到接近位姿

        Args:
            approach_pose: 预计算的接近位姿 (map 坐标系)
            target_position: 目标物体位置，用于阶段2/3。None 则从 approach_pose 反推
            status_callback: 阶段状态回调，参数为 (NavStage, 消息字符串)
            skip_stages: 要跳过的阶段集合，如 {1}, {1,2}, {2,3} 等
            depth_only_align: True 跳过 map 粗对齐，直接深度精对齐

        Returns:
            ApproachResult: 导航结果
        """
        # 处理 skip_stages 默认值
        if skip_stages is None:
            skip_stages = set()

        self._cancel_requested = False

        # 计算目标位置 (如果未提供)
        if target_position is None:
            target_position = self._compute_target_from_approach_pose(approach_pose)
        self.target_map_position = target_position
        self.get_logger().info(
            f"目标设定: target_map=({target_position.x:.3f},{target_position.y:.3f}), "
            f"approach=({approach_pose.pose.position.x:.3f},{approach_pose.pose.position.y:.3f})")

        def notify(stage: NavStage, msg: str):
            """通知阶段变化"""
            with self._stage_lock:
                self.stage = stage
            if status_callback:
                status_callback(stage, msg)
            self.get_logger().info(f"{stage.name}: {msg}")

        # ========== 阶段1: Nav2 导航 ==========
        if 1 in skip_stages:
            self.get_logger().info("跳过阶段1 (Nav2 导航)")
            notify(NavStage.NAVIGATING, "跳过阶段1")
            result1 = True
        else:
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

            # 阶段1完成后等待车辆停稳
            self.get_logger().info("Nav2 到达，等待车辆停稳...")
            time.sleep(0.15)

        # ========== 阶段2: 对齐 ==========
        if 2 in skip_stages:
            self.get_logger().info("跳过阶段2 (对齐)")
            notify(NavStage.ALIGNING, "跳过阶段2")
            success = True
            msg = "skipped"
        else:
            notify(NavStage.ALIGNING, "对准目标方向...")
            success, msg = self._do_alignment(depth_only=depth_only_align)

            if self._cancel_requested:
                self.depth_sensor.deactivate()
                notify(NavStage.FAILED, "用户取消")
                return ApproachResult(False, "NAV_CANCELLED", "用户取消", stage=2)

            if not success:
                self.depth_sensor.deactivate()
                notify(NavStage.FAILED, f"对齐失败: {msg}")
                return ApproachResult(False, "ALIGN_FAILED", msg, stage=2)

            # 阶段2完成后等待车辆停稳
            self.get_logger().info("对齐完成，等待车辆停稳...")
            time.sleep(0.15)

        # ========== 阶段3: 精确接近 ==========
        skip_approach = (self._locked_target_cz is not None
                         and self._locked_target_cz <= 0.25)
        if skip_approach:
            self.get_logger().info(
                f"目标距相机仅 {self._locked_target_cz:.3f}m，已足够近，跳过精确接近")
            self.depth_sensor.deactivate()
            notify(NavStage.FINAL_APPROACH, "已足够近，跳过精确接近")
            final_dist = self._locked_target_cz
            success = True
            msg = "close_enough"
        elif 3 in skip_stages:
            self.get_logger().info("跳过阶段3 (精确接近)")
            self.depth_sensor.deactivate()  # 阶段2可能已激活，清理
            notify(NavStage.FINAL_APPROACH, "跳过阶段3")
            final_dist = 0.0
            success = True
            msg = "skipped"
        else:
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
        """取消导航 — 仅设置标志 + 停车，不直接调用 cancelTask()。
        cancelTask() 由 worker 线程的 poll 循环负责调用，
        避免与 BasicNavigator 内部 executor generator 的并发竞争。
        """
        self._cancel_requested = True
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

    def _do_alignment(self, depth_only=False) -> tuple:
        """对齐: 可选 map 粗对齐 + 深度聚类精对齐

        Args:
            depth_only: True 则跳过 map 粗对齐，直接深度精对齐

        Returns:
            (bool, str): (成功标志, 错误消息)
        """
        # 粗对齐: 角度偏差大时先用 map 坐标旋转到大致朝向
        # 阶段1跳过时机器人可能面向任意方向，深度相机 FOV 有限
        if not depth_only:
            robot_pos, robot_yaw = self._get_robot_pose()
            if robot_pos and self.target_map_position:
                dx = self.target_map_position.x - robot_pos.x
                dy = self.target_map_position.y - robot_pos.y
                target_heading = math.atan2(dy, dx)
                error = abs(normalize_angle(target_heading - robot_yaw))
                coarse_threshold = self.config.align_tolerance * 2  # ~10°
                if error > coarse_threshold:
                    self.get_logger().info(
                        f"角度偏差 {math.degrees(error):.1f}° > "
                        f"{math.degrees(coarse_threshold):.1f}°，先执行粗对齐")
                    success, msg = self._do_alignment_by_map()
                    if not success:
                        return success, msg
                    time.sleep(self.config.align_settle_time)

        # 深度聚类精对齐
        self.get_logger().info("深度聚类精对齐")
        return self._do_alignment_by_depth()

    def _map_to_camera(self, target_map, robot_pos, robot_yaw) -> tuple:
        """map 坐标 → 相机光学系期望位置 (cam_x, cam_z)

        camera optical: x=右, z=前
        base_link: x=前, y=左
        相机在 base_link 前方 robot_front_offset 处
        逆变换: base → camera: cam_z=bx-offset, cam_x=-by
        """
        dx = target_map.x - robot_pos.x
        dy = target_map.y - robot_pos.y
        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        bx = dx * cos_y + dy * sin_y
        by = -dx * sin_y + dy * cos_y
        cam_z = bx - self.config.robot_front_offset
        cam_x = -by
        return cam_x, cam_z

    def _select_cluster(self, clusters, robot_pos=None, robot_yaw=None,
                        prev_cx=None, prev_cz=None):
        """聚类选择: 直接取最近的 (front_z 最小)

        Returns:
            (cx, cy, front_z, n_pts) or None
        """
        if not clusters:
            return None
        # front_z = c[2]，取最近的聚类
        nearest = min(clusters, key=lambda c: c[2])
        return nearest

    def _match_locked_target(self, clusters, prev_cx, prev_cz, tolerance=0.15):
        """在聚类中匹配锁定目标（纯时序连续性）

        Args:
            clusters: get_all_cluster_centroids() 返回值
            prev_cx, prev_cz: 上帧目标位置
            tolerance: 最大匹配距离 (米)
        Returns:
            (cx, cy, cz) or None
        """
        if not clusters or prev_cx is None:
            return None
        best = min(clusters, key=lambda c: (c[0]-prev_cx)**2 + (c[2]-prev_cz)**2)
        dist = math.sqrt((best[0]-prev_cx)**2 + (best[2]-prev_cz)**2)
        return best if dist < tolerance else None

    def _do_alignment_by_depth(self) -> tuple:
        """基于深度点云聚类的精对齐

        利用 map 期望位置 + 时序连续性从聚类中锁定正确目标，
        PD 控制使其居中于相机正前方。

        Returns:
            (bool, str): (成功标志, 错误消息)
        """
        # TF 可选: 有则用 map 匹配，无则靠最大簇兜底
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos is None:
            self.get_logger().warn("TF不可用，将使用最大簇兜底选择目标")

        # 激活深度传感器
        self.depth_sensor.activate()

        # 预热: 等待前N帧稳定 (RealSense 自动曝光需要几帧收敛)
        warmup = self.config.depth_warmup_frames
        warmup_count = 0
        warmup_start = time.time()
        while warmup_count < warmup and time.time() - warmup_start < 3.0:
            if self._cancel_requested:
                return False, "被中断"
            if self.depth_sensor.get_all_cluster_centroids() is not None:
                warmup_count += 1
            time.sleep(0.1)

        # 等待首次有效聚类结果 (最多5秒)
        wait_start = time.time()
        clusters = []
        while time.time() - wait_start < 5.0:
            if self._cancel_requested:
                return False, "被中断"
            clusters = self.depth_sensor.get_all_cluster_centroids()
            if clusters:
                break
            time.sleep(0.1)

        if not clusters:
            self.get_logger().warn("深度聚类无结果，跳过精对齐")
            return True, ""

        # 选择: map期望 > 时序连续 > 最大簇
        best = self._select_cluster(clusters, robot_pos, robot_yaw)

        if best is None:
            # 诊断: 打印每个聚类位置
            cls_info = ", ".join(
                f"({c[0]:+.2f},{c[2]:.2f},n={c[3] if len(c)>3 else '?'})"
                for c in clusters)
            self.get_logger().warn(
                f"无匹配聚类 共{len(clusters)}个: [{cls_info}]，"
                f"跳过精对齐，保留map粗对齐")
            return True, ""

        cx, cy, cz = best[0], best[1], best[2]
        n_pts = best[3] if len(best) > 3 else 0
        self.get_logger().info(
            f"选定目标聚类: cam_x={cx:.3f}m cam_z={cz:.3f}m n={n_pts} "
            f"共{len(clusters)}个聚类"
        )

        # 相机横向偏移补偿 (向左为正)
        camera_offset = self.config.camera_lateral_offset
        if abs(camera_offset) > 0.001:
            self.get_logger().info(f"相机横向偏移补偿: {camera_offset * 100:.1f}cm")

        # 纯 P 控制循环 — 对齐到目标聚类中心
        # TF 可选: 有则用 map 匹配，无则靠时序连续 + 最大簇兜底
        cfg = self.config
        start_time = time.time()
        first_error_logged = False
        prev_cx, prev_cz = cx, cz

        while rclpy.ok() and not self._cancel_requested:
            if time.time() - start_time > cfg.align_timeout:
                self._stop_robot()
                return False, "精对齐超时"

            clusters = self.depth_sensor.get_all_cluster_centroids()
            if not clusters:
                time.sleep(0.05)
                continue

            # TF 可选: 有则传入，无则 None (走时序/最大簇)
            robot_pos, robot_yaw = self._get_robot_pose()
            best = self._select_cluster(
                clusters, robot_pos, robot_yaw, prev_cx, prev_cz)
            if best is None:
                time.sleep(0.05)
                continue
            cx, cy, cz = best[0], best[1], best[2]
            prev_cx, prev_cz = cx, cz

            # 角度误差: cx>0 → 偏右 → 需右转 (omega<0)
            error = -math.atan2(cx - camera_offset, cz)

            if not first_error_logged:
                self.get_logger().info(
                    f"精对齐开始: err={math.degrees(error):.1f}° "
                    f"cx={cx:.3f}m cz={cz:.3f}m")
                first_error_logged = True

            if abs(error) < cfg.align_tolerance:
                self._stop_robot()
                self._locked_target_cx = cx
                self._locked_target_cz = cz
                self.get_logger().info(
                    f"精对齐完成: err={math.degrees(error):.1f}° "
                    f"cx={cx:.3f}m cz={cz:.3f}m "
                    f"耗时={time.time()-start_time:.1f}s")
                return True, ""

            # 纯 P + 限幅 + 死区
            omega = max(-cfg.align_max_omega,
                        min(cfg.align_max_omega, cfg.align_kp * error))
            if 0 < abs(omega) < cfg.align_min_omega:
                omega = math.copysign(cfg.align_min_omega, omega)

            cmd = Twist()
            cmd.angular.z = omega
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)

        self._stop_robot()
        return False, "被中断"

    def _do_alignment_by_map(self) -> tuple:
        """基于 map 坐标原地旋转对齐 — 纯 P 控制

        差速底盘直接接受角速度指令，电机驱动器内部已有速度闭环。
        外层只需 P 控制: 误差大→转快，误差小→转慢，到位→停。
        """
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos:
            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < self.config.min_align_distance:
                self.get_logger().info(f"距离过近 ({dist:.3f}m)，跳过对齐")
                return True, ""
            target_heading = math.atan2(dy, dx)
            init_err = math.degrees(normalize_angle(target_heading - robot_yaw))
            self.get_logger().info(
                f"粗对齐开始: robot=({robot_pos.x:.3f},{robot_pos.y:.3f}) "
                f"yaw={math.degrees(robot_yaw):.1f}°, "
                f"target=({self.target_map_position.x:.3f},"
                f"{self.target_map_position.y:.3f}), "
                f"heading={math.degrees(target_heading):.1f}°, "
                f"误差={init_err:.1f}°, dist={dist:.2f}m, "
                f"kp={self.config.align_kp} max_ω={self.config.align_max_omega}")

        cfg = self.config
        start_time = time.time()
        last_log_time = 0.0

        while rclpy.ok() and not self._cancel_requested:
            elapsed = time.time() - start_time
            if elapsed > cfg.align_timeout:
                self._stop_robot()
                robot_pos, robot_yaw = self._get_robot_pose()
                if robot_pos:
                    dx = self.target_map_position.x - robot_pos.x
                    dy = self.target_map_position.y - robot_pos.y
                    err = normalize_angle(math.atan2(dy, dx) - robot_yaw)
                    self.get_logger().warn(
                        f"对齐超时! 最终误差={math.degrees(err):.1f}° "
                        f"yaw={math.degrees(robot_yaw):.1f}°")
                return False, "对齐超时"

            robot_pos, robot_yaw = self._get_robot_pose()
            if robot_pos is None:
                time.sleep(0.05)
                continue

            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            target_heading = math.atan2(dy, dx)
            error = normalize_angle(target_heading - robot_yaw)

            coarse_tol = cfg.align_tolerance * 2  # 粗对齐放宽 (~10°)，精对齐兜底
            if abs(error) < coarse_tol:
                self._stop_robot()
                self.get_logger().info(
                    f"粗对齐完成: 误差={math.degrees(error):.1f}°, "
                    f"耗时={elapsed:.1f}s, "
                    f"yaw={math.degrees(robot_yaw):.1f}°, "
                    f"heading={math.degrees(target_heading):.1f}°")
                return True, ""

            # 纯 P + 限幅 + 死区
            omega = max(-cfg.align_max_omega,
                        min(cfg.align_max_omega, cfg.align_kp * error))
            if 0 < abs(omega) < cfg.align_min_omega:
                omega = math.copysign(cfg.align_min_omega, omega)

            # 周期性诊断日志 (每0.5秒)
            now = time.time()
            if now - last_log_time >= 0.5:
                self.get_logger().info(
                    f"对齐中: err={math.degrees(error):.1f}° "
                    f"ω={omega:.3f} yaw={math.degrees(robot_yaw):.1f}° "
                    f"heading={math.degrees(target_heading):.1f}°")
                last_log_time = now

            cmd = Twist()
            cmd.angular.z = omega
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)

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
        self.depth_sensor.activate()
        try:
            return self._do_final_approach_impl()
        finally:
            self.depth_sensor.deactivate()

    def _do_final_approach_impl(self) -> tuple:
        loop_dt = 1.0 / self.config.control_rate   # time.sleep 替代 create_rate
        start_time = time.time()

        self.get_logger().info(f"开始精确接近，目标距离: {self.config.final_approach_distance}m")
        self.get_logger().info(f"  robot_front_offset: {self.config.robot_front_offset}m")
        self.get_logger().info(f"  内参: {'已加载' if self.depth_sensor.has_intrinsics else '未加载'}")

        # 速度平滑参数
        dt = 1.0 / self.config.control_rate
        current_speed = 0.0  # 从0开始，逐渐加速
        max_accel = 0.15     # 最大加速率 m/s²
        max_decel = 0.3      # 最大减速率 m/s²

        # 横向修正参数
        lateral_kp = 0.3     # 横向修正 P 增益 (保守，避免蛇行)
        camera_offset = self.config.camera_lateral_offset

        # 记录 odom 起始位置，用于实际行驶距离测量
        with self._odom_lock:
            odom_start_x = self._odom_x
            odom_start_y = self._odom_y
        use_odom = odom_start_x is not None
        if use_odom:
            self.get_logger().info("阶段3: 使用里程计测量实际行驶距离")
        else:
            self.get_logger().warn("阶段3: 里程计不可用，回退到指令速度积分")

        # 开环控制：target_distance 是障碍物到机器人初始位置的距离
        # 如果检测到更近的物体，更新target；否则开环
        target_distance: float = None           # 目标距离 (相对初始位置)
        total_traveled: float = 0.0             # 累计行驶距离

        # P1b: 中位数滤波窗口 (取最近 5 帧的中位数，抗单帧噪声)
        from collections import deque
        depth_window: deque = deque(maxlen=5)

        # 最近目标跟踪 (单调递减: 只有更近才更新)
        tracked_cx = None
        tracked_cz = None

        while rclpy.ok() and not self._cancel_requested:
            # 超时检查
            if time.time() - start_time > self.config.final_approach_timeout:
                self._stop_robot()
                return False, "接近超时", 0.0

            # (A) 安全急停：获取任意最近障碍物距离
            nearest_any = self.depth_sensor.get_nearest_obstacle()
            nearest_any_z = nearest_any[2] if nearest_any else None

            # (B) 选择最近聚类，带跳变保护
            clusters = self.depth_sensor.get_all_cluster_centroids()
            obstacle_x = None
            lateral_cx = None

            if clusters:
                nearest = min(clusters, key=lambda c: c[2])
                if tracked_cz is not None:
                    # 跳变保护: 深度骤降超过 max(10cm, 30%) → 假目标，忽略
                    max_drop = max(0.10, tracked_cz * 0.3)
                    if nearest[2] < tracked_cz - max_drop:
                        self.get_logger().warn(
                            f"[跳变过滤] z={nearest[2]:.3f}m → 忽略 "
                            f"(跟踪={tracked_cz:.3f}m, 阈值={max_drop:.2f}m)",
                            throttle_duration_sec=1.0
                        )
                    elif nearest[2] <= tracked_cz + 0.03:
                        # 正常更新: 更近或噪声容差内
                        tracked_cx, tracked_cz = nearest[0], nearest[2]
                        obstacle_x = tracked_cz
                        lateral_cx = tracked_cx
                else:
                    # 首次读取
                    tracked_cx, tracked_cz = nearest[0], nearest[2]
                    obstacle_x = tracked_cz
                    lateral_cx = tracked_cx

            # 累计行驶距离: 优先用 odom 位置差，不可用时回退到指令速度积分
            if use_odom:
                total_traveled = self._get_odom_traveled(odom_start_x, odom_start_y)
            else:
                total_traveled += current_speed * dt

            # ========== 距离估算 ==========
            if target_distance is None:
                # 使用用户确认时的初始深度
                initial_depth = self.get_initial_depth()
                if initial_depth is not None:
                    target_distance = initial_depth
                    total_traveled = 0.0
                    front_clearance = initial_depth
                    self.get_logger().info(
                        f"[初始化] target={target_distance:.3f}m (用户确认深度)"
                    )
                elif obstacle_x is not None:
                    target_distance = obstacle_x
                    total_traveled = 0.0
                    front_clearance = obstacle_x
                    self.get_logger().info(
                        f"[初始化] target={target_distance:.3f}m"
                    )
                else:
                    self.get_logger().info(
                        f"[等待] 无点云数据...",
                        throttle_duration_sec=1.0
                    )
                    time.sleep(loop_dt)
                    continue
            else:
                # P1b: 中位数滤波更新 target_distance
                if obstacle_x is not None:
                    obstacle_from_start = obstacle_x + total_traveled
                    depth_window.append(obstacle_from_start)

                    if len(depth_window) >= 3:
                        median_dist = float(sorted(depth_window)[len(depth_window) // 2])
                        if median_dist < target_distance:
                            old = target_distance
                            target_distance = median_dist
                            self.get_logger().warn(
                                f"[更新] 更近障碍物 (中位数): "
                                f"{old:.3f}→{target_distance:.3f}m"
                            )

                # 开环计算当前距离
                front_clearance = target_distance - total_traveled
                self.get_logger().info(
                    f"[开环] 距离={front_clearance:.3f}m (target={target_distance:.3f}m, "
                    f"已行驶={total_traveled:.3f}m), 速度={current_speed:.3f}m/s",
                    throttle_duration_sec=0.5
                )

            # ========== 停止条件 ==========
            # 急停：任意障碍物 <= 3cm（无论是否为目标）
            if nearest_any_z is not None and nearest_any_z <= self.config.emergency_stop_distance:
                self._stop_robot()
                self.get_logger().warn(
                    f"紧急刹车! 最近障碍物: {nearest_any_z:.3f}m (深度)")
                return True, "", min(front_clearance, nearest_any_z)

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

            max_speed = self.config.final_approach_speed

            if remaining > 0.3:
                target_speed = max_speed
            else:
                ratio = remaining / 0.3
                target_speed = 0.02 + (max_speed - 0.02) * math.sqrt(ratio)

            speed_diff = target_speed - current_speed
            if speed_diff > 0:
                max_change = max_accel * dt
                current_speed += min(speed_diff, max_change)
            else:
                max_change = max_decel * dt
                current_speed += max(speed_diff, -max_change)

            current_speed = max(0.0, min(max_speed, current_speed))

            # ========== P1a: 横向修正 ==========
            angular_z = 0.0
            if lateral_cx is not None:
                lateral_error = -math.atan2(lateral_cx - camera_offset, obstacle_x)
                angular_z = lateral_kp * lateral_error
                # 限幅：前进中微调，不要大幅转向
                angular_z = max(-0.15, min(0.15, angular_z))

            cmd = Twist()
            cmd.linear.x = current_speed
            cmd.angular.z = angular_z
            self.cmd_vel_pub.publish(cmd)

            time.sleep(loop_dt)

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
