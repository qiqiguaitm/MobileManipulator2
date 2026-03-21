#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF Republisher（严格仿照 ROS1 _MobileManipulator）

将 HDL 的 map->odom 用当前时间戳重发，避免 DWA/Nav2 查 TF 时 extrapolation 错误。
订阅 /tf，只接受「时间戳比当前旧 >50ms」的 map->odom（视为来自 HDL），
以 now() + stamp_ahead_sec（默认 20ms）高频重发。ROS1：rate 默认 50，launch 可改为 100。

新增：EMA 平滑滤波，减小 HDL 定位抖动（smooth_alpha 参数）
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Transform
from tf2_msgs.msg import TFMessage


def quaternion_to_yaw(q):
    """从四元数提取 yaw 角"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw):
    """从 yaw 角创建四元数 (仅绕 Z 轴)"""
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


def normalize_angle(angle):
    """归一化角度到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class TFRepublisher(Node):
    def __init__(self):
        super().__init__('tf_republisher')
        self.declare_parameter('source_frame', 'map')
        self.declare_parameter('child_frame', 'odom')
        self.declare_parameter('rate_hz', 50.0)   # 与 ROS1 脚本默认一致，launch 可覆盖为 100
        self.declare_parameter('stamp_ahead_sec', 0.02)  # 与 ROS1 一致：超前 20ms
        self.declare_parameter('msg_age_threshold', 0.05)  # 与 ROS1 一致：只接受 >50ms 的（HDL 的）
        self.declare_parameter('smooth_alpha', 0.0)  # EMA 平滑系数，0=关闭，0.3=适度平滑，0.5=强平滑

        pv = lambda name: self.get_parameter(name).get_parameter_value()
        self.source_frame = pv('source_frame').string_value
        self.child_frame = pv('child_frame').string_value
        self.rate_hz = pv('rate_hz').double_value
        self.stamp_ahead_sec = pv('stamp_ahead_sec').double_value
        self.msg_age_threshold = pv('msg_age_threshold').double_value
        self.smooth_alpha = pv('smooth_alpha').double_value

        self.latest_transform = None
        self.smoothed_transform = None  # EMA 平滑后的变换
        self._logged_once = False
        self._waiting_logged = False

        # callback group 隔离：timer 不被高流量 /tf subscription 阻塞
        self._sub_group = ReentrantCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(TFMessage, '/tf', self.tf_callback, 5,
                                 callback_group=self._sub_group)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / self.rate_hz, self.timer_callback,
                          callback_group=self._timer_group)

        smooth_info = f', smooth_alpha={self.smooth_alpha}' if self.smooth_alpha > 0 else ' (no smoothing)'
        self.get_logger().info(
            f'TF Republisher: {self.source_frame} -> {self.child_frame} @ {self.rate_hz} Hz, '
            f'stamp_ahead={self.stamp_ahead_sec}s{smooth_info}'
        )

    def tf_callback(self, msg):
        for transform in msg.transforms:
            if (transform.header.frame_id == self.source_frame and
                    transform.child_frame_id == self.child_frame):
                # ROS2 HDL 使用未来时间戳（now + offset），不是 ROS1 的传感器时间戳（now - delay）
                # 由于 map->odom 只有 HDL 发布，直接接受所有 map->odom TF，无需 msg_age 检查
                self.latest_transform = transform

                # EMA 平滑处理
                if self.smooth_alpha > 0:
                    self._apply_ema_smoothing(transform.transform)

                if not self._logged_once:
                    now = self.get_clock().now()
                    stamp_sec = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                    now_sec = now.nanoseconds * 1e-9
                    msg_age = now_sec - stamp_sec
                    self.get_logger().info(
                        f'Received HDL map->odom TF (age={msg_age:.3f}s, accepting all map->odom from HDL)'
                    )
                    self._logged_once = True

    def _apply_ema_smoothing(self, raw_tf: Transform):
        """
        EMA（指数移动平均）平滑

        smoothed = alpha * raw + (1 - alpha) * prev_smoothed
        alpha 越小越平滑（响应越慢）
        """
        alpha = self.smooth_alpha

        if self.smoothed_transform is None:
            # 首次，直接使用原始值
            self.smoothed_transform = Transform()
            self.smoothed_transform.translation.x = raw_tf.translation.x
            self.smoothed_transform.translation.y = raw_tf.translation.y
            self.smoothed_transform.translation.z = raw_tf.translation.z
            self.smoothed_transform.rotation = raw_tf.rotation
            return

        prev = self.smoothed_transform

        # 位置 EMA
        prev.translation.x = alpha * raw_tf.translation.x + (1 - alpha) * prev.translation.x
        prev.translation.y = alpha * raw_tf.translation.y + (1 - alpha) * prev.translation.y
        prev.translation.z = alpha * raw_tf.translation.z + (1 - alpha) * prev.translation.z

        # 姿态 EMA（只处理 yaw，因为 map->odom 主要是 2D 平移+旋转）
        raw_yaw = quaternion_to_yaw(raw_tf.rotation)
        prev_yaw = quaternion_to_yaw(prev.rotation)

        # 处理角度跨越 ±pi 的情况
        yaw_diff = normalize_angle(raw_yaw - prev_yaw)
        smoothed_yaw = normalize_angle(prev_yaw + alpha * yaw_diff)

        prev.rotation = yaw_to_quaternion(smoothed_yaw)

    def timer_callback(self):
        # HDL 未初始化时：发布 identity TF 让 RViz 能显示地图，用户可手动设 initialpose
        if self.latest_transform is None:
            if not self._waiting_logged:
                self.get_logger().warn('Waiting for HDL map->odom, publishing identity TF for RViz')
                self._waiting_logged = True
            now = self.get_clock().now()
            from rclpy.duration import Duration
            ahead_ns = int(self.stamp_ahead_sec * 1e9)
            out = TransformStamped()
            out.header.stamp = (now + Duration(nanoseconds=ahead_ns)).to_msg()
            out.header.frame_id = self.source_frame
            out.child_frame_id = self.child_frame
            # identity transform: 位置(0,0,0) + 旋转(0,0,0,1)
            out.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(out)
            return

        if not self._logged_once:
            self.get_logger().info('TF Republisher: Active, republishing map->odom from HDL')
            self._logged_once = True

        # 有 HDL TF 了，用当前时间戳重发（超前 20ms 避免 extrapolation）
        now = self.get_clock().now()
        ahead_ns = int(self.stamp_ahead_sec * 1e9)
        from rclpy.duration import Duration

        out = TransformStamped()
        out.header.stamp = (now + Duration(nanoseconds=ahead_ns)).to_msg()
        out.header.frame_id = self.source_frame
        out.child_frame_id = self.child_frame

        # 使用平滑后的变换（如果启用）或原始变换
        if self.smooth_alpha > 0 and self.smoothed_transform is not None:
            out.transform = self.smoothed_transform
        else:
            out.transform = self.latest_transform.transform

        self.tf_broadcaster.sendTransform(out)


def main(args=None):
    import signal
    def _sigterm(s, f):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init(args=args)
    node = TFRepublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
