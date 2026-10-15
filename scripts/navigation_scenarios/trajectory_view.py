"""Persistent RViz paths; refresh the fixed reference across clock resets."""
import argparse
import copy
from pathlib import Path as FilePath
import sqlite3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


def read_reference(bag):
    if not bag:
        return None
    # Read only the small reference Path from a finalized playback bag.
    for db in sorted(FilePath(bag).glob('*.db3')):
        with sqlite3.connect('file:'+str(db)+'?mode=ro', uri=True) as connection:
            row=connection.execute("select data from messages where topic_id=(select id from topics where name='/reference/path' and type='nav_msgs/msg/Path') order by timestamp limit 1").fetchone()
        if row:
            return deserialize_message(row[0], Path)
    return None


class TrajectoryView(Node):
    def __init__(self, reference):
        super().__init__('recorded_trajectory_view')
        self.path=Path()
        self.last=-1
        self.reference=reference
        qos=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub=self.create_publisher(Path,'/replay/path',qos)
        self.reference_pub=self.create_publisher(Path,'/replay/reference_path',qos)
        self.box_pub=self.create_publisher(Marker,'/replay/robot_box',qos)
        self.create_subscription(PoseStamped,'/reference/trajectory',self.receive,qos_profile_sensor_data)
        self.create_subscription(Path,'/replay/reference_path_input',self.receive_reference,10)
        self.create_timer(0.5,self.refresh,clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))

    def receive(self,msg):
        stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        if stamp<self.last:
            self.path=Path()
        self.last=stamp
        self.path.header=msg.header
        self.path.poses.append(msg)
        self.pub.publish(self.path)

    def receive_reference(self,msg):
        if msg.poses:
            self.reference=msg
            self.refresh()

    def refresh(self):
        box=Marker()
        box.header.frame_id='base_link'
        box.ns='navigation_footprint_box'
        box.id=0
        box.type=Marker.CUBE
        box.action=Marker.ADD
        box.frame_locked=True
        box.pose.orientation.w=1.0
        # Navigation footprint x=[-0.375,0.434], y=[-0.31,0.31].
        # Height is only a display extrusion, not the arm's collision geometry.
        box.pose.position.x=(0.434-0.375)/2
        box.pose.position.z=0.125
        box.scale.x=0.809
        box.scale.y=0.62
        box.scale.z=0.25
        box.color.r=1.0
        box.color.g=0.55
        box.color.b=0.12
        box.color.a=0.4
        self.box_pub.publish(box)
        # Geometry stays in map. Refresh display stamps only, never source data.
        if self.reference:
            msg=copy.deepcopy(self.reference)
            msg.header.stamp=self.get_clock().now().to_msg()
            for pose in msg.poses:
                pose.header.stamp=msg.header.stamp
            self.reference_pub.publish(msg)
        if self.path.poses:
            self.pub.publish(self.path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bag')
    args,ros_args=parser.parse_known_args()
    reference=read_reference(args.bag)
    rclpy.init(args=ros_args)
    node=TrajectoryView(reference)
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__=='__main__':main()
