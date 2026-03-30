#!/usr/bin/env python3
"""
底盘速度后台监控 — 实时显示 cmd_vel 指令 + tracer_status 实际反馈

用法:
    python3 scripts/_cc_chassis_vel_monitor.py           # 前台运行
    python3 scripts/_cc_chassis_vel_monitor.py &          # 后台运行
    python3 scripts/_cc_chassis_vel_monitor.py --hz 5     # 5Hz 刷新
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist

try:
    from tracer_msgs.msg import TracerStatus
    HAS_TRACER = True
except ImportError:
    HAS_TRACER = False


class ChassisVelMonitor(Node):

    def __init__(self, hz: float):
        super().__init__('chassis_vel_monitor')

        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.cmd_stamp = 0.0

        self.actual_linear = 0.0
        self.actual_angular = 0.0
        self.actual_stamp = 0.0
        self.battery_v = 0.0
        self.error_code = 0

        qos_be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        if HAS_TRACER:
            self.create_subscription(TracerStatus, '/tracer_status',
                                     self._status_cb, qos_be)

        self.create_timer(1.0 / hz, self._print_tick)
        self.get_logger().info(
            f'底盘速度监控启动 ({hz}Hz), tracer_status={"✓" if HAS_TRACER else "✗"}')

    def _cmd_cb(self, msg: Twist):
        self.cmd_linear = msg.linear.x
        self.cmd_angular = msg.angular.z
        self.cmd_stamp = time.time()

    def _status_cb(self, msg):
        self.actual_linear = msg.linear_velocity
        self.actual_angular = msg.angular_velocity
        self.battery_v = msg.battery_voltage
        self.error_code = msg.error_code
        self.actual_stamp = time.time()

    def _print_tick(self):
        now = time.time()
        cmd_age = now - self.cmd_stamp if self.cmd_stamp > 0 else -1
        act_age = now - self.actual_stamp if self.actual_stamp > 0 else -1

        parts = [
            f'cmd=({self.cmd_linear:+.3f}, {self.cmd_angular:+.3f})',
            f'age={cmd_age:.1f}s' if cmd_age >= 0 else 'cmd=N/A',
        ]

        if HAS_TRACER and self.actual_stamp > 0:
            parts.append(
                f'actual=({self.actual_linear:+.3f}, {self.actual_angular:+.3f})')
            parts.append(f'batt={self.battery_v:.1f}V')
            if self.error_code != 0:
                parts.append(f'ERR={self.error_code}')
        elif HAS_TRACER:
            parts.append('actual=N/A')

        self.get_logger().info(' | '.join(parts))


def main():
    parser = argparse.ArgumentParser(description='底盘速度后台监控')
    parser.add_argument('--hz', type=float, default=2.0, help='刷新频率 (default: 2)')
    args = parser.parse_args()

    rclpy.init()
    node = ChassisVelMonitor(args.hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
