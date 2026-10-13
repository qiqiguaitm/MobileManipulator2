"""TF heartbeat must keep a committed correction, and stop on source expiry."""
import copy
import time
from types import SimpleNamespace
import rclpy
from geometry_msgs.msg import TransformStamped
from auto_explore.map_odom_tf import MapOdomTF


def test_repeats_same_correction_without_restamping_source():
    rclpy.init();n=MapOdomTF();sent=[];n.tf=SimpleNamespace(sendTransform=sent.append)
    try:
        msg=TransformStamped();msg.header.frame_id='map';msg.child_frame_id='odom'
        msg.header.stamp=n.get_clock().now().to_msg();msg.transform.rotation.w=1.;msg.transform.translation.x=2.
        n.on_correction(msg);original=copy.deepcopy(n.correction);n.broadcast();n.broadcast()
        assert len(sent)==2 and all(m.transform.translation.x==2. for m in sent)
        assert n.correction==original
        n.received=time.monotonic()-n.timeout-.1;n.broadcast()
        assert len(sent)==2 and n.expired
    finally:n.destroy_node();rclpy.try_shutdown()


def test_latched_old_message_cannot_renew_freshness():
    rclpy.init();n=MapOdomTF()
    try:
        msg=TransformStamped();msg.header.frame_id='map';msg.child_frame_id='odom';msg.transform.rotation.w=1.
        msg.header.stamp.sec=int(n.get_clock().now().nanoseconds/1e9)-5
        n.on_correction(msg)
        assert n.correction is None and not n.fresh()
        msg.header.stamp=n.get_clock().now().to_msg();msg.transform.rotation.w=0.
        n.on_correction(msg);assert n.correction is None
    finally:n.destroy_node();rclpy.try_shutdown()
