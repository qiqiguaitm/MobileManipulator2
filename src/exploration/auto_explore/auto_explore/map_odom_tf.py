"""Publish the mapper's last committed correction without sharing its CPU/GIL.

Repeating a correction does not extrapolate the robot pose: FAST-LIO owns that.
Source heartbeat expiry stops TF publication; stale mapper state is not hidden.
"""
import copy
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


class MapOdomTF(Node):
    def __init__(self):
        super().__init__('exploration_map_odom_tf')
        self.declare_parameter('source_timeout',1.5)
        self.timeout=float(self.get_parameter('source_timeout').value)
        if not math.isfinite(self.timeout) or self.timeout<=0:raise ValueError('invalid source_timeout')
        self.correction=None;self.received=0.;self.source_ns=0
        self.last_publish=None;self.max_gap=0.;self.published=0;self.expired=False
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf=TransformBroadcaster(self)
        self.status_pub=self.create_publisher(String,'/explore/tf_status',qos)
        self.create_subscription(TransformStamped,'/explore/map_correction',self.on_correction,qos)
        self.create_timer(.05,self.broadcast)
        self.create_timer(.5,self.status)

    def on_correction(self,msg):
        q=msg.transform.rotation;p=msg.transform.translation
        values=[p.x,p.y,p.z,q.x,q.y,q.z,q.w]
        source_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        age=(self.get_clock().now().nanoseconds-source_ns)*1e-9
        if (msg.header.frame_id!='map' or msg.child_frame_id!='odom'
                or not all(math.isfinite(v) for v in values)
                or abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.)>1e-3
                or age < -.1 or age > self.timeout):
            self.get_logger().warning('Ignoring invalid or stale map correction')
            return
        self.correction=copy.deepcopy(msg);self.source_ns=source_ns
        self.received=time.monotonic()

    def fresh(self):
        if self.correction is None:return False
        age=(self.get_clock().now().nanoseconds-self.source_ns)*1e-9
        return time.monotonic()-self.received<=self.timeout and -.1<=age<=self.timeout

    def broadcast(self):
        if not self.fresh():
            if self.correction is not None and not self.expired:
                self.get_logger().warning('Mapper heartbeat expired; stopping map->odom TF')
                self.expired=True
            self.last_publish=None
            return
        self.expired=False
        # The correction changes only after the mapper commits a map generation.
        msg=copy.deepcopy(self.correction)
        msg.header.stamp=self.get_clock().now().to_msg()
        self.tf.sendTransform(msg)
        now=time.monotonic()
        if self.last_publish is not None:self.max_gap=max(self.max_gap,now-self.last_publish)
        self.last_publish=now;self.published+=1

    def status(self):
        age=(self.get_clock().now().nanoseconds-self.source_ns)*1e-9 if self.correction else None
        self.status_pub.publish(String(data=json.dumps(dict(
            fresh=self.fresh(),source_age=age,published=self.published,
            max_publish_gap=self.max_gap,source_timeout=self.timeout))))


def main():
    rclpy.init();node=MapOdomTF()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
