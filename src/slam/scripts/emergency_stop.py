#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emergency Stop (ROS2) - 急停节点

持续发送零速度命令，确保机器人完全停止。

Usage:
  ros2 run slam emergency_stop.py          # 立即执行急停后退出
  ros2 run slam emergency_stop.py --daemon # 常驻节点，通过服务触发
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
import subprocess


class EmergencyStop(Node):
    def __init__(self):
        super().__init__('emergency_stop')

        self.declare_parameter('stop_duration', 5.0)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.stop_duration = self.get_parameter('stop_duration').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.stopped = False
        self.stop_timer = None
        self.stop_end_time = None

        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_service(Trigger, '~/stop', self.stop_callback)
        self.create_service(Trigger, '~/release', self.release_callback)

        self.get_logger().info('Emergency Stop node ready')
        self.get_logger().info('  Call /emergency_stop/stop to trigger')
        self.get_logger().info('  Call /emergency_stop/release to release')

    def _kill_nodes(self):
        for name in ['/trajectory_player', '/explore']:
            try:
                subprocess.run(
                    ['ros2', 'lifecycle', 'set', name, 'shutdown'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1
                )
            except Exception:
                pass
            try:
                subprocess.run(
                    ['ros2', 'node', 'kill', name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1
                )
            except Exception:
                pass

    def stop_callback(self, req, response):
        self.get_logger().warn('!!! EMERGENCY STOP TRIGGERED !!!')
        self.stopped = True
        self.stop_end_time = self.get_clock().now() + Duration(seconds=self.stop_duration)
        self._send_stop()
        self._kill_nodes()

        if self.stop_timer is None:
            self.stop_timer = self.create_timer(0.05, self._stop_timer_callback)

        response.success = True
        response.message = 'Emergency stop activated!'
        return response

    def release_callback(self, req, response):
        self.stopped = False
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None
        self.get_logger().info('Emergency stop released')
        response.success = True
        response.message = 'Emergency stop released'
        return response

    def _stop_timer_callback(self):
        if self.stopped:
            self._send_stop()
            if self.get_clock().now() > self.stop_end_time:
                self.stopped = False
                self.get_logger().info('Stop duration ended')

    def _send_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


def immediate_stop():
    """立即执行急停（不作为常驻节点）"""
    rclpy.init()
    node = Node('emergency_stop_immediate')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)

    node.get_logger().warn('[EStop] !!! EMERGENCY STOP !!!')
    import time
    time.sleep(0.2)

    cmd = Twist()
    for _ in range(60):
        pub.publish(cmd)
        time.sleep(0.05)
    node.get_logger().info('[EStop] Emergency stop complete')
    node.destroy_node()
    rclpy.shutdown()


def main(args=None):
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        rclpy.init(args=args)
        node = EmergencyStop()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        immediate_stop()


if __name__ == '__main__':
    main()
