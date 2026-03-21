#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast-LIO Odom to TF - 将 /fastlio/odom 转为 odom -> base_link TF

Fast-LIO 仅在成功处理点云时发布 camera_init->body TF (~10Hz)，CPU 繁忙时
可能出现 >1s 的间隙。本节点将 FastLIO odom 以 50Hz 重发为 odom->base_link TF，
避免与 Fast-LIO 的 camera_init->body 冲突（两个发布者抢同一 TF 会导致 TF_OLD_DATA）。

由于 odom≡camera_init 和 body≡base_link 都是 identity 关系，
FastLIO odom 中的 pose 直接就是 odom->base_link 变换。
"""
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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

        # callback group 隔离：timer 不被 odom subscription 阻塞
        self._timer_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(Odometry, topic, self.odom_callback, 10)
        self.create_timer(1.0 / rate_hz, self.timer_callback,
                          callback_group=self._timer_group)

        self.get_logger().info(f'FastlioOdomToTf: {topic} -> odom->base_link TF @ {rate_hz} Hz')

    def odom_callback(self, msg):
        # 接受 camera_init/body（FastLIO 原始帧名）
        if msg.header.frame_id != 'camera_init' or msg.child_frame_id != 'body':
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        # 直接发 odom->base_link（odom≡camera_init, body≡base_link 都是 identity）
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.latest_transform = t
        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
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
