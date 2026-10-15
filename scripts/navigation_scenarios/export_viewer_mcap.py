#!/usr/bin/env python3
"""Export a finalized ROS 2 simulation bag plus seekable display topics to MCAP.
Run with ROS Humble sourced and mcap/rosbags installed. Source data is untouched.
"""
import argparse
import copy
from bag_io import messages, reader
from pathlib import Path

from mcap.writer import Writer
from rosbags.typesys import Stores, get_typestore
from rclpy.serialization import deserialize_message, serialize_message
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from visualization_msgs.msg import Marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    store = get_typestore(Stores.ROS2_HUMBLE)
    trajectory = RosPath()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        writer = Writer(stream)
        writer.start(profile="ros2", library="navigation_scenarios")
        schemas, channels = {}, {}

        def channel(topic, typename):
            if typename not in schemas:
                definition, _ = store.generate_msgdef(typename, ros_version=2)
                schemas[typename] = writer.register_schema(
                    typename, "ros2msg", definition.encode()
                )
            if topic not in channels:
                channels[topic] = writer.register_channel(
                    topic, "cdr", schemas[typename]
                )
            return channels[topic]

        topics = {t.name: t.type for t in reader(args.bag).get_all_topics_and_types()}
        # Resolve schemas before copying messages so missing types fail early.
        for topic, typename in topics.items():
            channel(topic, typename)
        count = 0
        for topic, raw, stamp in messages(args.bag):
            typename = topics[topic]
            writer.add_message(channel(topic, typename), stamp, raw, stamp)
            count += 1
            if topic in ("/current_pose", "/trajectory", "/simulation/trajectory"):
                pose = deserialize_message(raw, PoseStamped)
                trajectory.header = copy.deepcopy(pose.header)
                trajectory.poses.append(pose)
                if "/path" not in topics:
                    writer.add_message(
                        channel("/path", "nav_msgs/msg/Path"),
                        stamp,
                        serialize_message(trajectory),
                        stamp,
                    )
                box = Marker()
                box.header = copy.deepcopy(pose.header)
                box.header.frame_id = "base_link"
                box.ns = "navigation_footprint_box"
                box.id = 0
                box.type = Marker.CUBE
                box.action = Marker.ADD
                box.frame_locked = True
                box.pose.orientation.w = 1.0
                box.pose.position.x = (0.434 - 0.375) / 2
                box.pose.position.z = 0.125
                box.scale.x, box.scale.y, box.scale.z = 0.809, 0.62, 0.25
                box.color.r, box.color.g, box.color.b, box.color.a = (
                    1.0,
                    0.55,
                    0.12,
                    0.4,
                )
                if "/robot_box" not in topics:
                    writer.add_message(
                        channel("/robot_box", "visualization_msgs/msg/Marker"),
                        stamp,
                        serialize_message(box),
                        stamp,
                    )
        writer.finish()
    print(
        f"Copied {count} source messages; exported {len(channels)} channels to {args.output}"
    )


if __name__ == "__main__":
    main()
