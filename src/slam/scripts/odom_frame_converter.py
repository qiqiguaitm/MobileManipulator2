#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odom Frame Converter (ROS2)
将 Fast-LIO 的 odom 消息转换为标准 ROS 导航格式

Fast-LIO 输出: frame_id=camera_init, child_frame_id=body
DWA/Nav2 期望: frame_id=odom, child_frame_id=base_link

Usage:
  ros2 run slam odom_frame_converter.py
  ros2 run slam odom_frame_converter.py --ros-args -p input:=/fastlio/odom -p output:=/odom
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomFrameConverter(Node):
    def __init__(self):
        super().__init__('odom_frame_converter')

        self.declare_parameter('input', '/fastlio/odom')
        self.declare_parameter('output', '/odom')
        self.declare_parameter('target_frame', 'odom')
        self.declare_parameter('target_child_frame', 'base_link')

        input_topic = self.get_parameter('input').value
        output_topic = self.get_parameter('output').value
        self.target_frame = self.get_parameter('target_frame').value
        self.target_child_frame = self.get_parameter('target_child_frame').value

        self.pub = self.create_publisher(Odometry, output_topic, 10)
        self.sub = self.create_subscription(Odometry, input_topic, self.callback, 10)

        self.get_logger().info(f'OdomFrameConverter: {input_topic} -> {output_topic}')
        self.get_logger().info(f'  Frames: * -> {self.target_frame}, * -> {self.target_child_frame}')

    def callback(self, msg):
        out_msg = Odometry()
        out_msg.header = msg.header
        out_msg.header.frame_id = self.target_frame
        out_msg.child_frame_id = self.target_child_frame
        out_msg.pose = msg.pose
        out_msg.twist = msg.twist
        self.pub.publish(out_msg)


def main(args=None):
    import signal
    def _sigterm(s, f):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init(args=args)
    node = OdomFrameConverter()
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
