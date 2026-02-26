#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wait for Localization (ROS2) - 等待 HDL/定位成功

用于阶段2启动时，确保在继承地图上正确定位后再开始移动。
检查 map->base_link TF，位置稳定 min_stable_time 秒后返回成功。

Usage:
  ros2 run slam wait_for_localization.py
  ros2 run slam wait_for_localization.py --ros-args -p timeout:=30.0 -p min_stable_time:=3.0
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener, TransformException
import sys


class LocalizationWaiter(Node):
    def __init__(self):
        super().__init__('wait_for_localization')

        self.declare_parameter('timeout', 30.0)
        self.declare_parameter('min_stable_time', 3.0)
        self.declare_parameter('position_threshold', 0.5)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')

        self.timeout = self.get_parameter('timeout').value
        self.min_stable_time = self.get_parameter('min_stable_time').value
        self.position_threshold = self.get_parameter('position_threshold').value
        self.map_frame = self.get_parameter('map_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value

        self.last_pose = None
        self.stable_start_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.status_pub = self.create_publisher(Bool, '~/localized', 1)

        self.get_logger().info('Waiting for localization...')
        self.get_logger().info(f'  Timeout: {self.timeout}s, Stable: {self.min_stable_time}s')

    def check_localization(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, rclpy.time.Time(),
                Duration(seconds=1.0)
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            current_pose = (x, y)

            if self.last_pose is None:
                self.last_pose = current_pose
                self.stable_start_time = self.get_clock().now()
                self.get_logger().info(f'Initial pose: ({x:.2f}, {y:.2f})')
                return False

            dx = current_pose[0] - self.last_pose[0]
            dy = current_pose[1] - self.last_pose[1]
            dist = (dx**2 + dy**2)**0.5

            if dist > self.position_threshold:
                self.last_pose = current_pose
                self.stable_start_time = self.get_clock().now()
                self.get_logger().info(f'Position changed: ({x:.2f}, {y:.2f}), restarting timer')
                return False

            elapsed = (self.get_clock().now() - self.stable_start_time).nanoseconds / 1e9
            if elapsed >= self.min_stable_time:
                self.get_logger().info(f'Localization stable at ({x:.2f}, {y:.2f}) for {elapsed:.1f}s')
                return True

            self.get_logger().info_throttle(1.0, f'Stabilizing... {elapsed:.1f}/{self.min_stable_time}s')
            return False

        except TransformException as e:
            self.get_logger().warn_throttle(2.0, f'TF not available: {e}')
            return False

    def wait(self):
        start_time = self.get_clock().now()
        rate = self.create_rate(2, self.get_clock())

        while rclpy.ok():
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > self.timeout:
                self.get_logger().error(f'Localization timeout after {self.timeout}s')
                self.status_pub.publish(Bool(data=False))
                return False

            if self.check_localization():
                self.get_logger().info('Localization successful!')
                self.status_pub.publish(Bool(data=True))
                return True

            rate.sleep()
        return False


def main(args=None):
    rclpy.init(args=args)
    waiter = LocalizationWaiter()
    success = waiter.wait()
    waiter.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
