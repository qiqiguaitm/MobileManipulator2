#!/usr/bin/env python3
"""
Tracer2 状态监控脚本
实时显示机器人状态信息
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import os

try:
    from tracer_msgs.msg import TracerStatus, TracerRCState
    TRACER_MSGS_AVAILABLE = True
except ImportError:
    TRACER_MSGS_AVAILABLE = False


class StatusMonitor(Node):
    def __init__(self):
        super().__init__('status_monitor')

        self.status = None
        self.odom = None
        self.rc = None

        if TRACER_MSGS_AVAILABLE:
            self.create_subscription(TracerStatus, '/tracer_status', self.status_cb, 10)
            self.create_subscription(TracerRCState, '/tracer_rc_status', self.rc_cb, 10)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # 定时刷新显示
        self.create_timer(0.2, self.display)

    def status_cb(self, msg):
        self.status = msg

    def odom_cb(self, msg):
        self.odom = msg

    def rc_cb(self, msg):
        self.rc = msg

    def display(self):
        # 清屏
        os.system('clear')

        print("=" * 50)
        print("  Tracer2 状态监控")
        print("=" * 50)

        if self.status:
            print(f"\n[机器人状态]")
            print(f"  电池电压: {self.status.battery_voltage:.2f} V")
            print(f"  线速度:   {self.status.linear_velocity:.3f} m/s")
            print(f"  角速度:   {self.status.angular_velocity:.3f} rad/s")
            print(f"  控制模式: {self.status.control_mode}")
            print(f"  错误码:   {self.status.error_code}")

            if len(self.status.actuator_states) >= 2:
                print(f"\n[电机状态]")
                for i, act in enumerate(self.status.actuator_states):
                    print(f"  电机{i}: RPM={act.rpm:5d}, 电流={act.current:.2f}A")
        else:
            print("\n[机器人状态] 等待数据...")

        if self.odom:
            pos = self.odom.pose.pose.position
            ori = self.odom.pose.pose.orientation
            # 四元数转欧拉角 (仅取 yaw)
            siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
            cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            vel = self.odom.twist.twist
            print(f"\n[里程计]")
            print(f"  位置: x={pos.x:.3f}, y={pos.y:.3f}")
            print(f"  航向: {math.degrees(yaw):.1f} deg")
            print(f"  速度: vx={vel.linear.x:.3f}, wz={vel.angular.z:.3f}")
        else:
            print("\n[里程计] 等待数据...")

        if self.rc:
            print(f"\n[遥控器]")
            print(f"  开关: SWA={self.rc.swa} SWB={self.rc.swb} SWC={self.rc.swc} SWD={self.rc.swd}")
            print(f"  右摇杆: V={self.rc.stick_right_v:4d} H={self.rc.stick_right_h:4d}")
            print(f"  左摇杆: V={self.rc.stick_left_v:4d} H={self.rc.stick_left_h:4d}")

        print("\n" + "-" * 50)
        print("按 Ctrl+C 退出")


def main():
    rclpy.init()
    node = StatusMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
