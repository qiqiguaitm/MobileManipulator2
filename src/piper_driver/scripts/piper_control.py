#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper Control Utility Class
ROS2 version

Simple utility class for publishing joint/pose commands to Piper arm.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from piper_msgs.msg import PosCmd


class PiperControl:
    """
    Utility class for controlling Piper arm via ROS2 topics.

    Usage:
        node = rclpy.create_node('my_control_node')
        piper = PiperControl(node)
        piper.joint_control_piper(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05)
    """

    def __init__(self, node: Node):
        """
        Initialize Piper control publishers.

        Args:
            node: ROS2 node to attach publishers to
        """
        self.node = node

        # Publishers for piper arm control
        self.pub_descartes = node.create_publisher(PosCmd, 'pos_cmd', 10)
        self.pub_joint = node.create_publisher(JointState, '/joint_states', 10)
        self.left_pub_joint = node.create_publisher(JointState, '/left_joint_states', 100)
        self.right_pub_joint = node.create_publisher(JointState, '/right_joint_states', 100)

        self.descartes_msgs = PosCmd()

    def init_pose(self):
        """Send zero pose to /joint_states"""
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.node.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_joint.publish(joint_states_msgs)
        self.node.get_logger().info('Sent init pose command')

    def left_init_pose(self):
        """Send zero pose to left arm"""
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.node.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_pub_joint.publish(joint_states_msgs)
        self.node.get_logger().info('Sent left arm init pose command')

    def right_init_pose(self):
        """Send zero pose to right arm"""
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.node.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_pub_joint.publish(joint_states_msgs)
        self.node.get_logger().info('Sent right arm init pose command')

    def descartes_control_piper(self, x, y, z, roll, pitch, yaw, gripper):
        """
        Send Cartesian pose command.

        Args:
            x, y, z: Position in meters
            roll, pitch, yaw: Orientation in radians
            gripper: Gripper position in meters (0 ~ 0.08)
        """
        self.descartes_msgs.x = float(x)
        self.descartes_msgs.y = float(y)
        self.descartes_msgs.z = float(z)
        self.descartes_msgs.roll = float(roll)
        self.descartes_msgs.pitch = float(pitch)
        self.descartes_msgs.yaw = float(yaw)
        self.descartes_msgs.gripper = float(gripper)
        self.pub_descartes.publish(self.descartes_msgs)

    def joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        """
        Send joint command to default arm.

        Args:
            j1-j6: Joint angles in radians
            gripper: Gripper position
        """
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.node.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [
            float(j1), float(j2), float(j3),
            float(j4), float(j5), float(j6), float(gripper)
        ]
        self.pub_joint.publish(joint_states_msgs)
        self.node.get_logger().info('Sent joint control command')

    def left_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        """
        Send joint command to left arm.

        Args:
            j1-j6: Joint angles in radians
            gripper: Gripper position
        """
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.node.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [
            float(j1), float(j2), float(j3),
            float(j4), float(j5), float(j6), float(gripper)
        ]
        self.left_pub_joint.publish(joint_states_msgs)
        self.node.get_logger().info('Sent left arm joint control command')

    def right_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        """
        Send joint command to right arm.

        Args:
            j1-j6: Joint angles in radians
            gripper: Gripper position
        """
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.node.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(7)]
        joint_states_msgs.position = [
            float(j1), float(j2), float(j3),
            float(j4), float(j5), float(j6), float(gripper)
        ]
        self.right_pub_joint.publish(joint_states_msgs)
        self.node.get_logger().info('Sent right arm joint control command')


# Standalone node for testing
def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('control_piper_node')

    piper = PiperControl(node)
    node.get_logger().info('PiperControl node ready')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
