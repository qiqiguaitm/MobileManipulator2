#!/usr/bin/env python3
"""
Tracer2 键盘遥控脚本
使用 WASD 控制机器人移动

按键说明:
  W - 前进
  S - 后退
  A - 左转
  D - 右转
  Q - 左前
  E - 右前
  空格 - 停止
  ESC/Ctrl+C - 退出
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 速度参数
        self.linear_speed = 0.3   # m/s
        self.angular_speed = 0.5  # rad/s

        self.get_logger().info('键盘遥控已启动')
        self.get_logger().info(f'线速度: {self.linear_speed} m/s, 角速度: {self.angular_speed} rad/s')

    def send_velocity(self, linear=0.0, angular=0.0):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)


def get_key(timeout=0.1):
    """获取键盘输入"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
            return key
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    print("=" * 40)
    print("  Tracer2 键盘遥控")
    print("=" * 40)
    print("按键说明:")
    print("  W - 前进    S - 后退")
    print("  A - 左转    D - 右转")
    print("  Q - 左前    E - 右前")
    print("  空格 - 停止")
    print("  +/- 调整速度")
    print("  ESC - 退出")
    print("=" * 40)

    rclpy.init()
    node = TeleopKeyboard()

    linear = 0.0
    angular = 0.0

    try:
        while True:
            key = get_key()

            if key is None:
                # 无输入时保持当前速度
                pass
            elif key == '\x1b':  # ESC
                break
            elif key == '\x03':  # Ctrl+C
                break
            elif key.lower() == 'w':
                linear = node.linear_speed
                angular = 0.0
            elif key.lower() == 's':
                linear = -node.linear_speed
                angular = 0.0
            elif key.lower() == 'a':
                linear = 0.0
                angular = node.angular_speed
            elif key.lower() == 'd':
                linear = 0.0
                angular = -node.angular_speed
            elif key.lower() == 'q':
                linear = node.linear_speed
                angular = node.angular_speed * 0.5
            elif key.lower() == 'e':
                linear = node.linear_speed
                angular = -node.angular_speed * 0.5
            elif key == ' ':
                linear = 0.0
                angular = 0.0
            elif key == '+' or key == '=':
                node.linear_speed = min(node.linear_speed + 0.1, 1.0)
                node.angular_speed = min(node.angular_speed + 0.1, 1.5)
                print(f"\r速度: linear={node.linear_speed:.1f}, angular={node.angular_speed:.1f}  ", end='')
            elif key == '-':
                node.linear_speed = max(node.linear_speed - 0.1, 0.1)
                node.angular_speed = max(node.angular_speed - 0.1, 0.1)
                print(f"\r速度: linear={node.linear_speed:.1f}, angular={node.angular_speed:.1f}  ", end='')

            node.send_velocity(linear, angular)

    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        # 停止机器人
        node.send_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        print("\n已退出")


if __name__ == '__main__':
    main()
