#!/usr/bin/env python3
"""
MCU 速度验证测试脚本
发送已知的 cmd_vel 指令，采集 MCU 反馈的 odom/status 速度，对比是否一致。

安全措施：
- 使用小速度 (最大 0.3 m/s)
- 每个测试阶段短时间 (2秒)
- 测试结束自动发送零速停车
- Ctrl+C 立即停车

使用方法：
  先启动 tracer_base 驱动:
    ros2 launch tracer_base tracer_base.launch.py publish_tf:=false
  然后运行:
    python3 scripts/_cc_mcu_vel_test.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tracer_msgs.msg import TracerStatus
import time
import signal
import sys
import threading


class MCUVelTester(Node):
    def __init__(self):
        super().__init__('mcu_vel_tester')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.status_sub = self.create_subscription(
            TracerStatus, '/tracer_status', self.status_cb, 10)

        # State
        self.odom_linear = 0.0
        self.odom_angular = 0.0
        self.status_linear = 0.0
        self.status_angular = 0.0
        self.odom_count = 0
        self.status_count = 0
        self.battery_voltage = 0.0

        # For collecting samples during a test phase
        self.collecting = False
        self.samples_odom = []
        self.samples_status = []

        self.get_logger().info('MCU 速度测试节点已启动')

    def odom_cb(self, msg):
        self.odom_linear = msg.twist.twist.linear.x
        self.odom_angular = msg.twist.twist.angular.z
        self.odom_count += 1
        if self.collecting:
            self.samples_odom.append((self.odom_linear, self.odom_angular))

    def status_cb(self, msg):
        self.status_linear = msg.linear_velocity
        self.status_angular = msg.angular_velocity
        self.battery_voltage = msg.battery_voltage
        self.status_count += 1
        if self.collecting:
            self.samples_status.append((self.status_linear, self.status_angular))

    def send_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        for _ in range(10):
            self.send_cmd(0.0, 0.0)
            time.sleep(0.02)

    def run_test_phase(self, cmd_linear, cmd_angular, duration=2.0, settle=0.5):
        """运行一个测试阶段：发送指定速度，采集反馈"""
        # 清空采样
        self.samples_odom = []
        self.samples_status = []

        # 先发送命令，等待稳定
        t_start = time.time()
        while time.time() - t_start < settle:
            self.send_cmd(cmd_linear, cmd_angular)
            rclpy.spin_once(self, timeout_sec=0.02)

        # 开始采集
        self.collecting = True
        t_start = time.time()
        while time.time() - t_start < duration:
            self.send_cmd(cmd_linear, cmd_angular)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.collecting = False

        # 计算平均值
        if self.samples_odom:
            avg_odom_lin = sum(s[0] for s in self.samples_odom) / len(self.samples_odom)
            avg_odom_ang = sum(s[1] for s in self.samples_odom) / len(self.samples_odom)
        else:
            avg_odom_lin = avg_odom_ang = float('nan')

        if self.samples_status:
            avg_status_lin = sum(s[0] for s in self.samples_status) / len(self.samples_status)
            avg_status_ang = sum(s[1] for s in self.samples_status) / len(self.samples_status)
        else:
            avg_status_lin = avg_status_ang = float('nan')

        return {
            'cmd_linear': cmd_linear,
            'cmd_angular': cmd_angular,
            'odom_linear': avg_odom_lin,
            'odom_angular': avg_odom_ang,
            'status_linear': avg_status_lin,
            'status_angular': avg_status_ang,
            'odom_samples': len(self.samples_odom),
            'status_samples': len(self.samples_status),
        }


def main():
    rclpy.init()
    tester = MCUVelTester()

    # Ctrl+C 安全停车
    def signal_handler(sig, frame):
        print('\n[!] 紧急停车...')
        tester.stop()
        tester.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # 等待话题连接
    print('等待 /odom 和 /tracer_status 话题...')
    timeout = time.time() + 10.0
    while tester.odom_count == 0 or tester.status_count == 0:
        rclpy.spin_once(tester, timeout_sec=0.1)
        if time.time() > timeout:
            print('[ERROR] 超时！未收到 odom/status 消息。请确认 tracer_base 已启动。')
            tester.destroy_node()
            rclpy.shutdown()
            return
    print(f'话题已连接! 电池电压: {tester.battery_voltage:.1f}V')

    # 测试用例: (linear_vel, angular_vel, 描述)
    test_cases = [
        # 静止基准
        (0.0, 0.0, '静止'),
        # 纯线速度
        (0.1, 0.0, '前进 0.1 m/s'),
        (0.2, 0.0, '前进 0.2 m/s'),
        (0.3, 0.0, '前进 0.3 m/s'),
        (-0.1, 0.0, '后退 0.1 m/s'),
        (-0.2, 0.0, '后退 0.2 m/s'),
        # 纯角速度
        (0.0, 0.3, '左转 0.3 rad/s'),
        (0.0, -0.3, '右转 0.3 rad/s'),
        (0.0, 0.5, '左转 0.5 rad/s'),
        # 组合速度
        (0.2, 0.3, '前进+左转'),
        (0.2, -0.3, '前进+右转'),
        # 回到静止
        (0.0, 0.0, '停车'),
    ]

    results = []
    print('\n' + '='*90)
    print('MCU 速度验证测试')
    print('='*90)
    print(f'{"测试":　<12} | {"指令 lin":>8} {"指令 ang":>8} | '
          f'{"odom lin":>8} {"odom ang":>8} | '
          f'{"status lin":>10} {"status ang":>10} | {"误差 lin":>8} {"误差 ang":>8}')
    print('-'*90)

    for cmd_lin, cmd_ang, desc in test_cases:
        result = tester.run_test_phase(cmd_lin, cmd_ang, duration=2.0, settle=0.8)
        results.append((desc, result))

        err_lin = result['odom_linear'] - cmd_lin
        err_ang = result['odom_angular'] - cmd_ang

        print(f'{desc:<12} | {cmd_lin:>8.3f} {cmd_ang:>8.3f} | '
              f'{result["odom_linear"]:>8.3f} {result["odom_angular"]:>8.3f} | '
              f'{result["status_linear"]:>10.3f} {result["status_angular"]:>10.3f} | '
              f'{err_lin:>+8.3f} {err_ang:>+8.3f}')

    # 最终停车
    tester.stop()

    # 总结
    print('\n' + '='*90)
    print('测试总结')
    print('='*90)

    max_lin_err = 0
    max_ang_err = 0
    for desc, r in results:
        lin_err = abs(r['odom_linear'] - r['cmd_linear'])
        ang_err = abs(r['odom_angular'] - r['cmd_angular'])
        max_lin_err = max(max_lin_err, lin_err)
        max_ang_err = max(max_ang_err, ang_err)

    print(f'最大线速度误差: {max_lin_err:.4f} m/s')
    print(f'最大角速度误差: {max_ang_err:.4f} rad/s')

    if max_lin_err < 0.05 and max_ang_err < 0.05:
        print('\n[PASS] MCU 速度反馈基本准确 (误差 < 0.05)')
    elif max_lin_err < 0.1 and max_ang_err < 0.1:
        print('\n[WARN] MCU 速度有一定偏差 (误差 0.05~0.1)，可能需要校准')
    else:
        print('\n[FAIL] MCU 速度偏差较大 (误差 > 0.1)，需要排查！')

    print(f'\n电池电压: {tester.battery_voltage:.1f}V')
    print(f'odom 消息总数: {tester.odom_count}, status 消息总数: {tester.status_count}')

    tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
