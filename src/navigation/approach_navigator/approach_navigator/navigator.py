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
        skip_stages: set = None
    ) -> ApproachResult:
        """直接接近目标位置 (简化接口)

        感知系统可直接调用此方法，传入目标位置即可执行三阶段导航。
        内部自动计算接近位姿。

        Args:
            target_position: 目标物体在 map 坐标系中的位置
            status_callback: 阶段状态回调
            skip_stages: 要跳过的阶段集合，如 {1}, {1,2}, {2,3} 等

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
                success, msg = self._do_alignment()
                if not success:
                    self.depth_sensor.deactivate()
                    return ApproachResult(False, "ALIGN_FAILED", msg, stage=2)
                time.sleep(1.0)
            else:
                self.get_logger().info("跳过阶段2")

            # 执行阶段3: 精确接近 (除非跳过)
            # _notify 同步调用 status_callback，callback 中的 input() 会阻塞直到用户确认
            if 3 not in skip_stages:
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
        return self.approach_to_pose(approach_pose, target_position, status_callback, skip_stages)

    def approach_to_pose(
        self,
        approach_pose: PoseStamped,
        target_position: Optional[Point] = None,
        status_callback: Callable[[NavStage, str], None] = None,
        skip_stages: set = None
    ) -> ApproachResult:
        """执行三阶段导航到接近位姿

        Args:
            approach_pose: 预计算的接近位姿 (map 坐标系)
            target_position: 目标物体位置，用于阶段2/3。None 则从 approach_pose 反推
            status_callback: 阶段状态回调，参数为 (NavStage, 消息字符串)
            skip_stages: 要跳过的阶段集合，如 {1}, {1,2}, {2,3} 等

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
            time.sleep(0.3)  # 等待0.3秒

        # ========== 阶段2: 对齐 ==========
        if 2 in skip_stages:
            self.get_logger().info("跳过阶段2 (对齐)")
            notify(NavStage.ALIGNING, "跳过阶段2")
            success = True
            msg = "skipped"
        else:
            notify(NavStage.ALIGNING, "对准目标方向...")
            success, msg = self._do_alignment()

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
            time.sleep(0.3)  # 等待0.3秒让车子停稳

        # ========== 阶段3: 精确接近 ==========
        if 3 in skip_stages:
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

    def _do_alignment(self) -> tuple:
        """两步对齐: 先 map 粗对齐 (把目标带入相机视野)，再深度聚类精对齐

        步骤1: 基于 map 坐标原地旋转，使机器人大致朝向目标
        步骤2: 激活深度传感器，用点云聚类精确对齐到目标中心

        深度传感器在对齐完成后保持激活状态，供阶段3复用。

        Returns:
            (bool, str): (成功标志, 错误消息)
        """
        # ===== 步骤1: map 粗对齐 =====
        self.get_logger().info("对齐步骤1: map 坐标粗对齐")
        success, msg = self._do_alignment_by_map()
        if not success:
            return False, msg

        # 粗对齐后停稳，等惯性消散 + IMU收敛
        settle = self.config.align_settle_time
        self.get_logger().info(f"粗对齐完成，停稳等待 {settle:.1f}s")
        self._stop_robot()
        time.sleep(settle)

        # ===== 步骤2: 深度聚类精对齐 =====
        self.get_logger().info("对齐步骤2: 深度聚类精对齐")
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

    def _select_cluster(self, clusters, robot_pos, robot_yaw,
                        prev_cx=None, prev_cz=None):
        """两级聚类选择: map期望 > 时序连续，无兜底

        1. 距 map 期望位置最近的聚类 (容差 0.5m)
        2. 距上帧聚类最近的 (容差 0.3m，时序连续性)
        3. 都不匹配 → 返回 None (跳过深度对齐)

        Returns:
            (cx, cy, cz) or None
        """
        # map 期望位置
        exp_cx, exp_cz = self._map_to_camera(
            self.target_map_position, robot_pos, robot_yaw)
        by_map = min(clusters, key=lambda c:
                     (c[0] - exp_cx) ** 2 + (c[2] - exp_cz) ** 2)
        map_dist = math.sqrt(
            (by_map[0] - exp_cx) ** 2 + (by_map[2] - exp_cz) ** 2)
        if map_dist < 0.8:
            return by_map

        # 时序连续性
        if prev_cx is not None:
            by_prev = min(clusters, key=lambda c:
                         (c[0] - prev_cx) ** 2 + (c[2] - prev_cz) ** 2)
            prev_dist = math.sqrt(
                (by_prev[0] - prev_cx) ** 2 + (by_prev[2] - prev_cz) ** 2)
            if prev_dist < 0.3:
                return by_prev

        # 无匹配 — 目标可能被遮挡或太小不可见
        return None

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
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos is None:
            return False, "无法获取机器人位姿"

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

        # 选择: map期望 > 时序连续，无匹配则跳过
        exp_cx, exp_cz = self._map_to_camera(
            self.target_map_position, robot_pos, robot_yaw)
        best = self._select_cluster(clusters, robot_pos, robot_yaw)

        if best is None:
            self.get_logger().warn(
                f"无匹配聚类 (期望cam=({exp_cx:.3f},{exp_cz:.3f}), "
                f"共{len(clusters)}个聚类)，跳过精对齐，保留map粗对齐")
            return True, ""

        cx, cy, cz = best
        self.get_logger().info(
            f"选定目标聚类: cam_x={cx:.3f}m cam_z={cz:.3f}m "
            f"期望cam=({exp_cx:.3f},{exp_cz:.3f}) "
            f"共{len(clusters)}个聚类"
        )

        # 相机横向偏移补偿 (向左为正)
        camera_offset = self.config.camera_lateral_offset
        if abs(camera_offset) > 0.001:
            self.get_logger().info(f"相机横向偏移补偿: {camera_offset * 100:.1f}cm")

        # PD 控制循环 — 对齐到目标聚类中心
        loop_dt = 0.05                  # 20 Hz
        start_time = time.time()
        last_error = 0.0
        last_time = start_time
        current_omega = 0.0             # 平滑用
        prev_cx, prev_cz = cx, cz      # 时序连续性

        while rclpy.ok() and not self._cancel_requested:
            current_time = time.time()

            if current_time - start_time > self.config.align_timeout:
                self._stop_robot()
                return False, "精对齐超时"

            robot_pos, robot_yaw = self._get_robot_pose()
            if robot_pos is None:
                time.sleep(loop_dt)
                continue

            clusters = self.depth_sensor.get_all_cluster_centroids()
            if not clusters:
                time.sleep(loop_dt)
                continue

            best = self._select_cluster(
                clusters, robot_pos, robot_yaw, prev_cx, prev_cz)
            if best is None:
                time.sleep(loop_dt)
                continue
            cx, cy, cz = best
            prev_cx, prev_cz = cx, cz

            # 角度误差: cx>0 → 偏右 → 需右转 (omega<0)
            # camera_offset 补偿相机与机器人中心线的横向偏移
            error = -math.atan2(cx - camera_offset, cz)

            if abs(error) < self.config.align_tolerance:
                self._stop_robot()
                # 保存锁定种子，供阶段3继承
                self._locked_target_cx = cx
                self._locked_target_cz = cz
                self.get_logger().info(
                    f"精对齐完成，误差: {math.degrees(error):.2f}°, "
                    f"目标距离: {cz:.3f}m"
                )
                return True, ""

            # PD 控制
            actual_dt = current_time - last_time
            if actual_dt > 0.001:
                d_error = (error - last_error) / actual_dt
            else:
                d_error = 0.0

            target_omega = self.config.align_kp * error + self.config.align_kd * d_error
            target_omega = max(-self.config.align_max_omega,
                               min(self.config.align_max_omega, target_omega))

            # 角加速度限幅 — 平滑omega变化，防止突变导致晃动
            max_delta = self.config.align_max_alpha * loop_dt
            omega_diff = target_omega - current_omega
            if abs(omega_diff) > max_delta:
                current_omega += max_delta if omega_diff > 0 else -max_delta
            else:
                current_omega = target_omega

            # 死区 — 极小omega不足以克服摩擦力，发0避免电机抖动
            cmd = Twist()
            if abs(current_omega) >= self.config.align_min_omega:
                cmd.angular.z = current_omega
            self.cmd_vel_pub.publish(cmd)

            last_error = error
            last_time = current_time
            time.sleep(loop_dt)

        self._stop_robot()
        return False, "被中断"

    def _do_alignment_by_map(self) -> tuple:
        """回退方案: 基于地图坐标的对齐

        使用 self.target_map_position 计算目标航向，PD 控制原地旋转对齐。

        Returns:
            (bool, str): (成功标志, 错误消息)
        """
        robot_pos, robot_yaw = self._get_robot_pose()
        if robot_pos:
            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < self.config.min_align_distance:
                self.get_logger().info(f"距离过近 ({dist:.3f}m)，跳过对齐")
                return True, ""

        loop_dt = 0.05                  # 20 Hz
        start_time = time.time()
        last_error = 0.0
        last_time = start_time
        current_omega = 0.0             # 平滑用

        while rclpy.ok() and not self._cancel_requested:
            current_time = time.time()

            if current_time - start_time > self.config.align_timeout:
                self._stop_robot()
                return False, "对齐超时"

            robot_pos, robot_yaw = self._get_robot_pose()
            if robot_pos is None:
                time.sleep(loop_dt)
                continue

            dx = self.target_map_position.x - robot_pos.x
            dy = self.target_map_position.y - robot_pos.y
            target_yaw = math.atan2(dy, dx)
            error = normalize_angle(target_yaw - robot_yaw)

            if abs(error) < self.config.align_tolerance:
                self._stop_robot()
                self.get_logger().info(f"对齐完成 (地图回退)，误差: {math.degrees(error):.2f}°")
                return True, ""

            actual_dt = current_time - last_time
            if actual_dt > 0.001:
                d_error = (error - last_error) / actual_dt
            else:
                d_error = 0.0

            target_omega = self.config.align_kp * error + self.config.align_kd * d_error
            target_omega = max(-self.config.align_max_omega,
                               min(self.config.align_max_omega, target_omega))

            # 角加速度限幅
            max_delta = self.config.align_max_alpha * loop_dt
            omega_diff = target_omega - current_omega
            if abs(omega_diff) > max_delta:
                current_omega += max_delta if omega_diff > 0 else -max_delta
            else:
                current_omega = target_omega

            # 死区
            cmd = Twist()
            if abs(current_omega) >= self.config.align_min_omega:
                cmd.angular.z = current_omega
            self.cmd_vel_pub.publish(cmd)

            last_error = error
            last_time = current_time
            time.sleep(loop_dt)

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

            # (B) 选择最近聚类，仅比当前更近时才更新
            clusters = self.depth_sensor.get_all_cluster_centroids()
            obstacle_x = None
            lateral_cx = None

            if clusters:
                nearest = min(clusters, key=lambda c: c[2])
                # 容差 3cm: 允许深度噪声抖动，防止间歇丢失横向修正
                if tracked_cz is None or nearest[2] <= tracked_cz + 0.03:
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
