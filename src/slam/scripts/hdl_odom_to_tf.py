#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDL Odom to TF - 将 HDL 的 Odometry (map->base_link) 转为 map->odom TF

ROS2 hdl_localization_node 只发布 Odometry，不发布 TF。本节点订阅 odom_out，
取 odom->base_link 用于计算 map->odom（优先 /odom/fused，回退 TF）。

公式: map->odom = map->base_link * inv(odom->base_link)

use_odom_fused=true 时订阅 /odom/fused，与 HDL 预测同源，map->base 链更一致。
若遇 rclpy 转换崩溃，可设 use_odom_fused:=false 回退到 TF 查询。
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import TransformException
import numpy as np


def _quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def _rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


class HdlOdomToTf(Node):
    def __init__(self):
        super().__init__('hdl_odom_to_tf')
        self.declare_parameter('hdl_odom_topic', '/hdl_localization/odom_out')
        self.declare_parameter('odom_fused_topic', '/odom/fused')
        self.declare_parameter('use_odom_fused', True)  # True=同源一致；遇崩溃可 false 回退 TF
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        topic = self.get_parameter('hdl_odom_topic').value
        odom_fused_topic = self.get_parameter('odom_fused_topic').value
        uof = self.get_parameter('use_odom_fused').value
        self.use_odom_fused = uof if isinstance(uof, bool) else str(uof).lower() in ('true', '1', 'yes')
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.latest_odom = None
        self.latest_odom_fused = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Odometry, topic, self.odom_callback, 10)
        if self.use_odom_fused:
            self.create_subscription(Odometry, odom_fused_topic, self.odom_fused_callback, 10)
        self.create_timer(0.02, self.timer_callback)  # 50Hz

        src = odom_fused_topic if self.use_odom_fused else 'TF'
        self.get_logger().info(f'HdlOdomToTf: {topic} -> map->odom, odom->base from {src}')

    def odom_callback(self, msg):
        if msg.header.frame_id == self.map_frame and msg.child_frame_id == self.base_frame:
            self.latest_odom = msg

    def odom_fused_callback(self, msg):
        if msg.header.frame_id == self.odom_frame and msg.child_frame_id == self.base_frame:
            self.latest_odom_fused = msg

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        # 未收到 HDL odom 时，发布 identity map->odom，确保 map 帧存在（RViz 不黑屏）
        if self.latest_odom is None:
            out = TransformStamped()
            out.header.stamp = now
            out.header.frame_id = self.map_frame
            out.child_frame_id = self.odom_frame
            out.transform.translation.x = 0.0
            out.transform.translation.y = 0.0
            out.transform.translation.z = 0.0
            out.transform.rotation.w = 1.0
            out.transform.rotation.x = 0.0
            out.transform.rotation.y = 0.0
            out.transform.rotation.z = 0.0
            self.tf_broadcaster.sendTransform(out)
            return

        # odom->base_link: 优先 /odom/fused（与 HDL 同源），无则 TF 查
        if self.use_odom_fused and self.latest_odom_fused is not None:
            otb = self.latest_odom_fused.pose.pose
            odom_to_base_m = self._pose_to_matrix(otb.position, otb.orientation)
        else:
            try:
                odom_to_base = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame,
                    rclpy.time.Time(), Duration(seconds=0.5)
                )
                odom_to_base_m = self._transform_to_matrix(odom_to_base.transform)
            except TransformException as e:
                self.get_logger().warn_throttle(2.0, f'No {self.odom_frame}->{self.base_frame}: {e}')
                out = TransformStamped()
                out.header.stamp = now
                out.header.frame_id = self.map_frame
                out.child_frame_id = self.odom_frame
                out.transform.translation.x = 0.0
                out.transform.translation.y = 0.0
                out.transform.translation.z = 0.0
                out.transform.rotation.w = 1.0
                out.transform.rotation.x = 0.0
                out.transform.rotation.y = 0.0
                out.transform.rotation.z = 0.0
                self.tf_broadcaster.sendTransform(out)
                return

        # map->base_link from HDL odom (pose of base in map)
        p = self.latest_odom.pose.pose.position
        o = self.latest_odom.pose.pose.orientation
        map_to_base = self._pose_to_matrix(p, o)

        # map->odom = map->base * inv(odom->base)
        base_to_odom = np.linalg.inv(odom_to_base_m)
        map_to_odom = map_to_base @ base_to_odom

        # ROS2: 使用系统时间（避免传感器时间不同步导致 TF_OLD_DATA）
        # tf_republisher 会用当前时间高频重发，这里也用当前时间保持一致
        out = TransformStamped()
        out.header.stamp = now
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.odom_frame
        out.transform = self._matrix_to_transform(map_to_odom)
        self.tf_broadcaster.sendTransform(out)

    def _pose_to_matrix(self, p, o):
        R = _quat_to_rot(o.x, o.y, o.z, o.w)
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = [p.x, p.y, p.z]
        return m

    def _transform_to_matrix(self, t):
        R = _quat_to_rot(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        return m

    def _matrix_to_transform(self, m):
        from geometry_msgs.msg import Transform
        q = _rot_to_quat(m[:3, :3])
        t = Transform()
        t.translation.x = float(m[0, 3])
        t.translation.y = float(m[1, 3])
        t.translation.z = float(m[2, 3])
        t.rotation.x = q[0]
        t.rotation.y = q[1]
        t.rotation.z = q[2]
        t.rotation.w = q[3]
        return t


def main(args=None):
    import signal
    def _sigterm(s, f):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)
    rclpy.init(args=args)
    node = HdlOdomToTf()
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
