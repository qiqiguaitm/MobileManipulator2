#!/usr/bin/env python3
"""
Tracer2 灯光控制测试脚本
持续发送灯光命令
"""

import rclpy
from rclpy.node import Node
import time

try:
    from tracer_msgs.msg import TracerLightCmd
    TRACER_MSGS_AVAILABLE = True
except ImportError:
    TRACER_MSGS_AVAILABLE = False
    print("错误: tracer_msgs 不可用")
    print("请先运行: source <your_ws>/install/setup.bash")
    print("例如: source /data/workspace/MobileManipulator2/install/setup.bash")
    exit(1)


class LightTester(Node):
    def __init__(self):
        super().__init__('light_tester')
        self.publisher = self.create_publisher(TracerLightCmd, '/light_control', 10)
        self.get_logger().info('灯光测试节点已启动')

    def send_light(self, front_mode, rear_mode=0, custom_value=0):
        msg = TracerLightCmd()
        msg.cmd_ctrl_allowed = True
        msg.front_mode = front_mode
        msg.front_custom_value = custom_value
        msg.rear_mode = rear_mode
        msg.rear_custom_value = 0
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = LightTester()

    print("\n灯光控制测试")
    print("=" * 40)
    print("模式: 0=关闭, 1=常亮, 2=呼吸, 3=自定义")
    print("=" * 40)

    tests = [
        ("常亮 (持续发送 3 秒)", 1, 30),
        ("呼吸灯 (持续发送 3 秒)", 2, 30),
        ("自定义亮度 50% (持续发送 3 秒)", 3, 30),
        ("关闭", 0, 10),
    ]

    for name, mode, count in tests:
        print(f"\n测试: {name}")
        for i in range(count):
            node.send_light(mode, mode, 128 if mode == 3 else 0)
            time.sleep(0.1)
        print(f"  已发送 {count} 次命令")

    print("\n测试完成")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
