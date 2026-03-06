#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odom Fusion Sync (ROS2)
FastLIO 校准 + wheel_odom 高频插值，pose 与 twist 时间同步

算法:
  wheel_odom (50Hz): 增量累积 pose，计算 twist，发布 /odom/fused
  FastLIO (10Hz): 校准 current_pose，消除 wheel 漂移

Usage:
  ros2 run slam odom_fusion_sync.py
"""

import rclpy
from rclpy.node import Node
import numpy as np
import threading
from array import array
from nav_msgs.msg import Odometry


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw):
    import math
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return [0.0, 0.0, sy, cy]  # x, y, z, w


class OdomFusionSync(Node):
    def __init__(self):
        super().__init__('odom_fusion_sync')

        self.declare_parameter('fastlio_topic', '/fastlio/odom')
        self.declare_parameter('wheel_topic', '/wheel_odom')
        self.declare_parameter('output', '/odom/fused')
        self.declare_parameter('output_frame', 'odom')
        self.declare_parameter('output_child_frame', 'base_link')
        self.declare_parameter('twist_alpha', 0.3)
        self.declare_parameter('wz_alpha', 0.7)
        self.declare_parameter('wz_deadzone', 0.03)
        self.declare_parameter('yaw_calibration_alpha', 0.9)

        self.fastlio_topic = self.get_parameter('fastlio_topic').value
        self.wheel_topic = self.get_parameter('wheel_topic').value
        self.output_topic = self.get_parameter('output').value
        self.output_frame = self.get_parameter('output_frame').value
        self.output_child_frame = self.get_parameter('output_child_frame').value
        self.twist_alpha = self.get_parameter('twist_alpha').value
        self.wz_alpha = self.get_parameter('wz_alpha').value
        self.wz_deadzone = self.get_parameter('wz_deadzone').value
        self.yaw_calibration_alpha = self.get_parameter('yaw_calibration_alpha').value

        self.lock = threading.Lock()
        self.fastlio_pose = None
        self.fastlio_updated = False
        self.current_pose = None
        self.wheel_last_pose = None
        self.wheel_last_time = None
        self.last_twist = np.array([0.0, 0.0, 0.0])

        self.odom_pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.create_subscription(Odometry, self.fastlio_topic, self.fastlio_callback, 10)
        self.create_subscription(Odometry, self.wheel_topic, self.wheel_callback, 10)

        self.get_logger().info(f'Odom Fusion Sync: {self.fastlio_topic} + {self.wheel_topic} -> {self.output_topic}')

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def fastlio_callback(self, msg):
        with self.lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            o = msg.pose.pose.orientation
            yaw = quat_to_yaw(o.x, o.y, o.z, o.w)

            self.fastlio_pose = np.array([x, y, yaw])
            self.fastlio_updated = True

            if self.current_pose is None:
                self.current_pose = self.fastlio_pose.copy()
                self.get_logger().info(f'Initialized from FastLIO: ({x:.3f}, {y:.3f}, {np.degrees(yaw):.1f}°)')
            else:
                self.current_pose[0] = self.fastlio_pose[0]
                self.current_pose[1] = self.fastlio_pose[1]
                yaw_diff = self._normalize_angle(self.fastlio_pose[2] - self.current_pose[2])
                self.current_pose[2] = self._normalize_angle(
                    self.current_pose[2] + (1 - self.yaw_calibration_alpha) * yaw_diff
                )

    def wheel_callback(self, msg):
        with self.lock:
            if self.current_pose is None:
                return

            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            o = msg.pose.pose.orientation
            yaw = quat_to_yaw(o.x, o.y, o.z, o.w)
            wheel_pose = np.array([x, y, yaw])

            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.wheel_last_pose is None:
                self.wheel_last_pose = wheel_pose.copy()
                self.wheel_last_time = t
                return

            dt = t - self.wheel_last_time
            if dt <= 0 or dt > 0.5:
                self.wheel_last_pose = wheel_pose.copy()
                self.wheel_last_time = t
                return

            if self.fastlio_updated:
                self.fastlio_updated = False
                self.wheel_last_pose = wheel_pose.copy()
                self.wheel_last_time = t
                self._publish_odom(msg.header.stamp, self.last_twist)
                return

            dx = wheel_pose[0] - self.wheel_last_pose[0]
            dy = wheel_pose[1] - self.wheel_last_pose[1]
            dyaw = self._normalize_angle(wheel_pose[2] - self.wheel_last_pose[2])

            # 物理合理性检查: 50Hz@0.3m/s → 单步最大 ~6mm，0.1m 已极其宽松
            MAX_DELTA_XY = 0.1   # 单步最大位移 (m)
            MAX_DELTA_YAW = 0.5  # 单步最大转角 (rad)
            if abs(dx) > MAX_DELTA_XY or abs(dy) > MAX_DELTA_XY or abs(dyaw) > MAX_DELTA_YAW:
                self.get_logger().warn(
                    f'wheel_odom outlier rejected: dx={dx:.3f}, dy={dy:.3f}, dyaw={dyaw:.3f}',
                    throttle_duration_sec=5.0)
                self.wheel_last_pose = wheel_pose.copy()
                self.wheel_last_time = t
                return

            self.current_pose[0] += dx
            self.current_pose[1] += dy
            self.current_pose[2] = self._normalize_angle(self.current_pose[2] + dyaw)

            vx_w = dx / dt
            vy_w = dy / dt
            vyaw = dyaw / dt
            yaw = self.current_pose[2]
            c, s = np.cos(yaw), np.sin(yaw)
            vx_r = c * vx_w + s * vy_w
            vy_r = -s * vx_w + c * vy_w
            twist = np.array([vx_r, vy_r, vyaw])

            twist[0] = self.twist_alpha * self.last_twist[0] + (1 - self.twist_alpha) * twist[0]
            twist[1] = self.twist_alpha * self.last_twist[1] + (1 - self.twist_alpha) * twist[1]
            twist[2] = self.wz_alpha * self.last_twist[2] + (1 - self.wz_alpha) * twist[2]
            if abs(twist[2]) < self.wz_deadzone:
                twist[2] = 0.0

            # 速度硬上限保护 (防止 EMA 残留异常值)
            MAX_VEL = 1.0  # m/s (实际最大 0.3)
            MAX_WZ = 2.0   # rad/s (实际最大 0.8)
            twist[0] = np.clip(twist[0], -MAX_VEL, MAX_VEL)
            twist[1] = np.clip(twist[1], -MAX_VEL, MAX_VEL)
            twist[2] = np.clip(twist[2], -MAX_WZ, MAX_WZ)
            self.last_twist = twist.copy()

            self.wheel_last_pose = wheel_pose.copy()
            self.wheel_last_time = t
            self._publish_odom(msg.header.stamp, twist)

    def _publish_odom(self, stamp, twist):
        out = Odometry()
        # 使用系统时间戳，避免与其他节点时间不同步
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.output_frame
        out.child_frame_id = self.output_child_frame
        out.pose.pose.position.x = self.current_pose[0]
        out.pose.pose.position.y = self.current_pose[1]
        out.pose.pose.position.z = 0.0
        q = yaw_to_quat(self.current_pose[2])
        out.pose.pose.orientation.x = q[0]
        out.pose.pose.orientation.y = q[1]
        out.pose.pose.orientation.z = q[2]
        out.pose.pose.orientation.w = q[3]
        out.twist.twist.linear.x = twist[0]
        out.twist.twist.linear.y = twist[1]
        out.twist.twist.linear.z = 0.0
        out.twist.twist.angular.z = twist[2]
        # ROS2 要求 covariance 是 float 类型的 array
        cov = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.02]
        out.pose.covariance = array('d', cov)
        out.twist.covariance = array('d', cov)
        self.odom_pub.publish(out)


def main(args=None):
    import signal
    def _sigterm(s, f):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init(args=args)
    node = OdomFusionSync()
    try:
        rclpy.spin(node)
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
