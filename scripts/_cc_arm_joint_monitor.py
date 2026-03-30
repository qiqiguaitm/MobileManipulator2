#!/usr/bin/env python3
"""
机械臂关节角度监控脚本
订阅 /joint_states，实时打印并记录关节角度变化

Usage:
    python3 scripts/_cc_arm_joint_monitor.py [output_file]
"""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ArmJointMonitor(Node):
    def __init__(self, output_file=None):
        super().__init__('arm_joint_monitor')

        self.declare_parameter('topic', '/joint_states')
        topic = self.get_parameter('topic').value

        self.sub = self.create_subscription(
            JointState, topic, self.callback, 10)

        self.prev_positions = None
        self.first_msg = True

        # 文件记录
        if output_file is None:
            output_file = '/tmp/arm_joint_log.csv'
        self.output_file = output_file
        self.log_file = open(output_file, 'w')
        self.log_file.write("timestamp,joint,position,delta\n")

        self.get_logger().info(f'监听关节角度: {topic}')
        self.get_logger().info(f'记录到: {output_file}')

    def callback(self, msg: JointState):
        if not msg.name or not msg.position:
            return

        # 首次：打印标题和所有位置
        if self.first_msg:
            print(f"\n=== 关节: {msg.name} ===")
            print(f"位置: {[f'{p:.4f}' for p in msg.position]}")
            self.first_msg = False
            self.prev_positions = list(msg.position)
            return

        # 后续：只打印有变化的位置
        if len(msg.position) != len(self.prev_positions):
            self.prev_positions = list(msg.position)
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        changes = [(i, msg.name[i], msg.position[i] - self.prev_positions[i])
                   for i in range(len(msg.position)) if abs(msg.position[i] - self.prev_positions[i]) > 0.001]

        if changes:
            print(f"[{timestamp:.3f}] ", end="")
            for i, name, delta in changes:
                print(f"{name}: {msg.position[i]:.4f} ({delta:+.4f})  ", end="")
                # 写入文件
                self.log_file.write(f"{timestamp:.6f},{name},{msg.position[i]:.6f},{delta:.6f}\n")
                self.log_file.flush()
            print()

        self.prev_positions = list(msg.position)


def main(args=None):
    output_file = sys.argv[1] if len(sys.argv) > 1 else None

    rclpy.init(args=args)
    node = ArmJointMonitor(output_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log_file.close()
        print(f"\n日志已保存到: {node.output_file}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
