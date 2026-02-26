#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态 JointState 发布 - 替代 joint_state_publisher 避免 "invalid JointState" 错误

robot_state_publisher 仅在 name.size() != position.size() 时报 "invalid JointState"。
本节点保证 name/position 严格匹配，并发布到独立话题，由 launch 将 RSP remap 到该话题，
从而隔离可能的其他（格式错误的）/joint_states 发布者。

Usage:
  ros2 run slam joint_state_static.py
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# URDF 中 joint1-8 的默认姿态（与 display.launch zeros 一致）
# 使用 list 保证顺序稳定，避免 dict 迭代与 position 错位
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
JOINT_POSITIONS = [0.0, 0.7854, -1.5708, 0.0, 0.0, 0.0, 0.0, 0.0]


class JointStateStatic(Node):
    def __init__(self):
        super().__init__('joint_state_static')
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('topic', '/joint_states_static')
        rate = self.get_parameter('rate').value
        topic = self.get_parameter('topic').value

        self.pub = self.create_publisher(JointState, topic, 10)
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.get_logger().info(f'Publishing {len(JOINT_NAMES)} joints to {topic} at {rate} Hz')

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = list(JOINT_NAMES)
        msg.position = list(JOINT_POSITIONS)
        # RSP 仅校验 name.size()==position.size()；velocity 可省，空数组更安全
        msg.velocity = []
        msg.effort = []
        assert len(msg.name) == len(msg.position), 'name/position length mismatch'
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateStatic()
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
