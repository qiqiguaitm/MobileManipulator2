#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast-LIO Odom to TF - 将 /fastlio/odom 转为 camera_init -> body TF

Fast-LIO 仅在成功处理点云时发布 TF，无点时 ("No point, skip this scan!") 不发布，
导致 base_link -> map 链断裂、RViz 全黑。本节点订阅 odom，无数据时发布 identity，
确保 TF 链始终完整。
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class FastlioOdomToTf(Node):
    def __init__(self):
        super().__init__('fastlio_odom_to_tf')
        self.declare_parameter('odom_topic', '/fastlio/odom')
        self.declare_parameter('rate_hz', 50.0)

        topic = self.get_parameter('odom_topic').value
        rate_hz = self.get_parameter('rate_hz').value

        self.latest_transform = None
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, topic, self.odom_callback, 10)
        self.create_timer(1.0 / rate_hz, self.timer_callback)

        self.get_logger().info(f'FastlioOdomToTf: {topic} -> camera_init->body TF @ {rate_hz} Hz')

    def odom_callback(self, msg):
        if msg.header.frame_id != 'camera_init' or msg.child_frame_id != 'body':
            return
        t = TransformStamped()
        # 使用当前系统时间，避免传感器时间与系统时间不同步
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_init'
        t.child_frame_id = 'body'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.latest_transform = t
        # 立即发布 TF，减少延迟
        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'camera_init'
        t.child_frame_id = 'body'
        if self.latest_transform is not None:
            t.transform = self.latest_transform.transform
        else:
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    import signal
    def _sigterm(s, f):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init(args=args)
    node = FastlioOdomToTf()
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
