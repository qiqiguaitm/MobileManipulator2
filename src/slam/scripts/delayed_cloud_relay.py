#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟点云转发节点（严格仿照 ROS1 _MobileManipulator 时间同步逻辑）

让 HDL Localization 收到的点云比 Fast-LIO 晚一点，确保 Fast-LIO 已发布对应时间戳的 TF。
可选 timestamp_offset：回退点云时间戳，使 TF 查找能命中已存在的变换。

ROS1 对照：delay 默认 0.15，timestamp_offset 默认 0.0；发布时间戳 = 原 stamp（或原 stamp - offset）。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from collections import deque
from copy import deepcopy


class DelayedCloudRelay(Node):
    def __init__(self):
        super().__init__('delayed_cloud_relay')
        # 与 ROS1 一致：脚本默认 delay=0.15，launch 可覆盖为 0.05
        self.declare_parameter('delay', 0.15)
        self.declare_parameter('timestamp_offset', 0.0)
        self.declare_parameter('input', '/rslidar_points')
        self.declare_parameter('output', '/rslidar_points_delayed')
        self.declare_parameter('max_rate', 0.0)  # 0 = 不限制
        self.declare_parameter('use_system_time', False)  # ROS1 无此选项，保持 false

        def to_float(p):
            v = p.get_parameter_value()
            try:
                return float(v.double_value)
            except (TypeError, AttributeError):
                return float(v.string_value)
        self.delay = to_float(self.get_parameter('delay'))
        self.timestamp_offset = to_float(self.get_parameter('timestamp_offset'))
        self.max_rate = to_float(self.get_parameter('max_rate'))
        self.use_system_time = self.get_parameter('use_system_time').get_parameter_value().bool_value
        input_topic = self.get_parameter('input').get_parameter_value().string_value
        output_topic = self.get_parameter('output').get_parameter_value().string_value

        self.buffer = deque()
        self.last_pub_time = None
        self.min_interval = 1.0 / self.max_rate if self.max_rate > 0 else 0.0

        self.pub = self.create_publisher(PointCloud2, output_topic, 2)
        self.sub = self.create_subscription(PointCloud2, input_topic, self.callback, 2)
        self.timer = self.create_timer(0.01, self.publish_delayed)  # 100Hz 与 ROS1 一致

        rate_str = f'{self.max_rate}Hz' if self.max_rate > 0 else 'unlimited'
        time_str = 'system_time' if self.use_system_time else f'sensor_time(offset={self.timestamp_offset}s)'
        self.get_logger().info(
            f'DelayedCloudRelay: {input_topic} -> {output_topic}, '
            f'delay={self.delay}s, stamp={time_str}, max_rate={rate_str}'
        )

    def callback(self, msg):
        recv_time = self.get_clock().now()
        self.buffer.append((recv_time, msg))

    def publish_delayed(self):
        now = self.get_clock().now()
        while self.buffer:
            recv_time, msg = self.buffer[0]
            if (now - recv_time).nanoseconds / 1e9 < self.delay:
                break
            if self.min_interval > 0 and self.last_pub_time is not None:
                if (now - self.last_pub_time).nanoseconds / 1e9 < self.min_interval:
                    break
            self.buffer.popleft()
            # 与 ROS1 一致：仅当 timestamp_offset > 0 时回退时间戳后发布，否则原样发布
            if self.timestamp_offset > 0:
                out = deepcopy(msg)
                offset_ns = int(self.timestamp_offset * 1e9)
                orig_ns = out.header.stamp.sec * 10**9 + out.header.stamp.nanosec
                if orig_ns > offset_ns:
                    new_ns = orig_ns - offset_ns
                    out.header.stamp.sec = new_ns // 10**9
                    out.header.stamp.nanosec = int(new_ns % 10**9)
                self.pub.publish(out)
            elif self.use_system_time:
                out = deepcopy(msg)
                out.header.stamp = now.to_msg()
                self.pub.publish(out)
            else:
                self.pub.publish(msg)
            self.last_pub_time = now


def main(args=None):
    import signal
    def _sigterm(s, f):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init(args=args)
    node = DelayedCloudRelay()
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
