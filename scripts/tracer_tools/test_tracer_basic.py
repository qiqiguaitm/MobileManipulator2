#!/usr/bin/env python3
"""
Tracer2 基本功能验证脚本
默认使用 can1 接口
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import subprocess
import time
import sys

# 尝试导入 tracer_msgs
try:
    from tracer_msgs.msg import TracerStatus, TracerLightCmd, TracerRCState
    TRACER_MSGS_AVAILABLE = True
except ImportError:
    TRACER_MSGS_AVAILABLE = False
    print("[WARN] tracer_msgs 未安装，部分测试将跳过")


class TracerTester(Node):
    def __init__(self):
        super().__init__('tracer_tester')

        self.status_received = False
        self.odom_received = False
        self.rc_received = False
        self.last_status = None
        self.last_odom = None

        # 发布者
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        if TRACER_MSGS_AVAILABLE:
            self.light_pub = self.create_publisher(TracerLightCmd, '/light_control', 10)

            # 订阅者
            self.status_sub = self.create_subscription(
                TracerStatus, '/tracer_status', self.status_callback, 10)
            self.rc_sub = self.create_subscription(
                TracerRCState, '/tracer_rc_status', self.rc_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

    def status_callback(self, msg):
        self.status_received = True
        self.last_status = msg

    def odom_callback(self, msg):
        self.odom_received = True
        self.last_odom = msg

    def rc_callback(self, msg):
        self.rc_received = True

    def send_velocity(self, linear_x=0.0, angular_z=0.0):
        """发送速度命令"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f'发送速度命令: linear={linear_x}, angular={angular_z}')

    def send_light_cmd(self, front_mode=0, rear_mode=0):
        """发送灯光命令"""
        if not TRACER_MSGS_AVAILABLE:
            self.get_logger().warn('tracer_msgs 不可用，跳过灯光测试')
            return
        msg = TracerLightCmd()
        msg.cmd_ctrl_allowed = True
        msg.front_mode = front_mode
        msg.rear_mode = rear_mode
        self.light_pub.publish(msg)
        self.get_logger().info(f'发送灯光命令: front={front_mode}, rear={rear_mode}')

    def stop(self):
        """停止机器人"""
        self.send_velocity(0.0, 0.0)


def check_can_interface(interface='can1'):
    """检查 CAN 接口状态"""
    print(f"\n[1] 检查 CAN 接口 ({interface})...")
    try:
        result = subprocess.run(['ip', 'link', 'show', interface],
                              capture_output=True, text=True)
        if result.returncode == 0:
            if 'UP' in result.stdout:
                print(f"  [OK] {interface} 已启动")
                return True
            else:
                print(f"  [WARN] {interface} 存在但未启动")
                print(f"  提示: 运行 'sudo ip link set {interface} up type can bitrate 500000'")
                return False
        else:
            print(f"  [FAIL] {interface} 不存在")
            return False
    except Exception as e:
        print(f"  [ERROR] 检查失败: {e}")
        return False


def check_ros_topics():
    """检查 ROS2 话题"""
    print("\n[2] 检查 ROS2 话题...")
    try:
        result = subprocess.run(['ros2', 'topic', 'list'],
                              capture_output=True, text=True, timeout=5)
        topics = result.stdout.strip().split('\n')

        required_topics = ['/cmd_vel', '/odom', '/tracer_status']
        for topic in required_topics:
            if topic in topics:
                print(f"  [OK] {topic}")
            else:
                print(f"  [MISS] {topic}")
        return True
    except subprocess.TimeoutExpired:
        print("  [ERROR] ros2 命令超时")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def check_ros_nodes():
    """检查 ROS2 节点"""
    print("\n[3] 检查 ROS2 节点...")
    try:
        result = subprocess.run(['ros2', 'node', 'list'],
                              capture_output=True, text=True, timeout=5)
        nodes = result.stdout.strip().split('\n')

        if '/tracer_base_node' in nodes:
            print("  [OK] tracer_base_node 正在运行")
            return True
        else:
            print("  [MISS] tracer_base_node 未运行")
            print("  提示: 运行 'ros2 launch tracer_base tracer_base.launch.py port_name:=can1'")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def run_motion_test(tester, duration=2.0):
    """运动测试"""
    print("\n[4] 运动控制测试...")

    tests = [
        ("前进", 0.15, 0.0),
        ("后退", -0.15, 0.0),
        ("左转", 0.0, 0.3),
        ("右转", 0.0, -0.3),
        ("停止", 0.0, 0.0),
    ]

    for name, linear, angular in tests:
        print(f"  测试: {name}...")
        tester.send_velocity(linear, angular)
        time.sleep(duration)

    tester.stop()
    print("  [OK] 运动测试完成")


def run_light_test(tester):
    """灯光测试"""
    print("\n[5] 灯光控制测试...")

    if not TRACER_MSGS_AVAILABLE:
        print("  [SKIP] tracer_msgs 不可用")
        return

    tests = [
        ("前灯常亮", 1, 0),
        ("后灯常亮", 0, 1),
        ("呼吸灯", 2, 2),
        ("关闭", 0, 0),
    ]

    for name, front, rear in tests:
        print(f"  测试: {name}...")
        tester.send_light_cmd(front, rear)
        time.sleep(1.5)

    print("  [OK] 灯光测试完成")


def run_status_check(tester):
    """状态检查"""
    print("\n[6] 状态数据检查...")

    # 等待数据
    timeout = 3.0
    start = time.time()
    while time.time() - start < timeout:
        rclpy.spin_once(tester, timeout_sec=0.1)
        if tester.odom_received:
            break

    if tester.odom_received:
        print("  [OK] 里程计数据正常")
        odom = tester.last_odom
        print(f"       位置: x={odom.pose.pose.position.x:.3f}, y={odom.pose.pose.position.y:.3f}")
    else:
        print("  [FAIL] 未收到里程计数据")

    if TRACER_MSGS_AVAILABLE and tester.status_received:
        print("  [OK] 状态数据正常")
        status = tester.last_status
        print(f"       电池电压: {status.battery_voltage:.2f}V")
        print(f"       线速度: {status.linear_velocity:.3f} m/s")
        print(f"       角速度: {status.angular_velocity:.3f} rad/s")
    elif TRACER_MSGS_AVAILABLE:
        print("  [FAIL] 未收到状态数据")


def main():
    print("=" * 50)
    print("  Tracer2 基本功能验证")
    print("  默认 CAN 接口: can1")
    print("=" * 50)

    # 检查 CAN 接口
    can_ok = check_can_interface('can1')

    # 初始化 ROS2
    rclpy.init()

    try:
        # 检查节点和话题
        check_ros_nodes()
        check_ros_topics()

        # 创建测试节点
        tester = TracerTester()

        # 状态检查
        run_status_check(tester)

        # 询问是否进行运动测试
        print("\n" + "-" * 50)
        response = input("是否进行运动测试? (y/n): ").strip().lower()
        if response == 'y':
            run_motion_test(tester)

        # 询问是否进行灯光测试
        response = input("是否进行灯光测试? (y/n): ").strip().lower()
        if response == 'y':
            run_light_test(tester)

        print("\n" + "=" * 50)
        print("  验证完成")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
