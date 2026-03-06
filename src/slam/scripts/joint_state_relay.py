#!/usr/bin/env python3
"""
JointState Relay Node - 智能切换实时/静态关节状态

功能：
- 订阅 /joint_states (grasp 发布的实时数据)
- 订阅 /joint_states_static (导航发布的静态数据)
- 优先使用实时数据，如果实时数据超时则使用静态数据

Usage:
    ros2 run slam joint_state_relay.py
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from copy import deepcopy
from sensor_msgs.msg import JointState


class JointStateRelay(Node):
    def __init__(self):
        super().__init__('joint_state_relay')

        # 参数
        self.declare_parameter('realtime_topic', '/joint_states')
        self.declare_parameter('static_topic', '/joint_states_static')
        self.declare_parameter('output_topic', '/joint_states')
        self.declare_parameter('timeout_sec', 0.5)  # 实时数据超时时间

        realtime_topic = self.get_parameter('realtime_topic').value
        static_topic = self.get_parameter('static_topic').value
        output_topic = self.get_parameter('output_topic').value
        timeout_sec = self.get_parameter('timeout_sec').value

        # 状态
        self._last_realtime_msg = None
        self._last_static_msg = None
        self._last_realtime_time = None

        # 订阅
        self.realtime_sub = self.create_subscription(
            JointState, realtime_topic, self._realtime_callback, 10)
        self.static_sub = self.create_subscription(
            JointState, static_topic, self._static_callback, 10)

        # 发布
        self.pub = self.create_publisher(JointState, output_topic, 10)

        # 定时检查
        self.timer = self.create_timer(0.01, self._timer_callback)  # 100Hz

        self.get_logger().info(f'JointState Relay initialized')
        self.get_logger().info(f'  realtime: {realtime_topic}')
        self.get_logger().info(f'  static: {static_topic}')
        self.get_logger().info(f'  output: {output_topic}')
        self.get_logger().info(f'  timeout: {timeout_sec}s')

    def _realtime_callback(self, msg: JointState):
        """收到实时数据"""
        self._last_realtime_msg = msg
        self._last_realtime_time = self.get_clock().now()

    def _static_callback(self, msg: JointState):
        """收到静态数据"""
        self._last_static_msg = msg

    def _timer_callback(self):
        """发布数据"""
        now = self.get_clock().now()
        output_msg = None

        # 检查实时数据是否有效
        if self._last_realtime_msg is not None and self._last_realtime_time is not None:
            elapsed = now - self._last_realtime_time
            if elapsed < Duration(seconds=self.get_parameter('timeout_sec').value):
                # 实时数据有效，深拷贝后使用（避免修改原始消息）
                output_msg = deepcopy(self._last_realtime_msg)
            else:
                # 实时数据超时，使用静态数据作为 fallback
                if self._last_static_msg is not None:
                    output_msg = deepcopy(self._last_static_msg)
        else:
            # 没有实时数据，使用静态数据
            if self._last_static_msg is not None:
                output_msg = deepcopy(self._last_static_msg)

        if output_msg is not None:
            # 使用当前系统时间戳，避免TF查找时时间不同步导致机器人漂移
            output_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateRelay()
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
