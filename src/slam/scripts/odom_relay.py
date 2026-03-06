#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odom Relay - 将 /odom 转发到 /odom/fused
用于 use_odom_fusion:=false 时，使 Nav2 始终订阅 /odom/fused
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomRelay(Node):
    def __init__(self):
        super().__init__('odom_relay')
        self.pub = self.create_publisher(Odometry, '/odom/fused', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.callback, 10)
        self.get_logger().info('OdomRelay: /odom -> /odom/fused')

    def callback(self, msg):
        # 使用系统时间戳，避免与其他节点时间不同步
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
