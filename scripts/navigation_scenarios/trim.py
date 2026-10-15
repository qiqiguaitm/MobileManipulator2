#!/usr/bin/env python3
"""Trim a finalized scene at operator confirmation, retaining TF/map context."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from bag_io import messages, reader, writer as mcap_writer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    a = p.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from geometry_msgs.msg import PoseStamped
    from tf2_msgs.msg import TFMessage
    from display_messages import RecordedDisplay

    src = a.scene
    dst = src.with_name(src.name + "_trimmed")
    if dst.exists():
        raise SystemExit("Output already exists: " + str(dst))
    control = json.loads((src / "operator_confirmation.json").read_text())
    start = control["confirmation_received_ros_ns"]
    end = control["stop_confirmation_received_ros_ns"]
    bag = src / "bag"
    topics = reader(bag).get_all_topics_and_types()
    manifest = json.loads((src / "scenario.json").read_text())
    display = RecordedDisplay(
        manifest.get("base_frame", "base_link"), path_topic="/reference/path"
    )
    seed_names = {
        "/reference/map",
        "/scenario/task",
        "/reference/camera/top/color/camera_info",
        "/reference/camera/top/aligned_depth_to_color/camera_info",
        "/reference/camera/top/extrinsics/depth_to_color",
    }
    latest, static, dynamic = {}, {}, []
    for topic, raw, stamp in messages(
        bag, seed_names | {"/reference/tf_static", "/reference/tf"}, end=start - 1
    ):
        if topic in seed_names:
            latest[topic] = raw
        elif topic == "/reference/tf_static":
            for tf in deserialize_message(raw, TFMessage).transforms:
                static[tf.child_frame_id] = tf
        elif stamp >= start - 2_000_000_000:
            dynamic.append((topic, raw))
    seeds = list(latest.items())
    if static:
        seeds.append(
            (
                "/reference/tf_static",
                serialize_message(TFMessage(transforms=list(static.values()))),
            )
        )
    seeds.extend(dynamic)
    dst.mkdir()
    writer = mcap_writer(dst / "bag")
    for metadata in topics:
        if metadata.name not in ("/replay/path", "/replay/robot_box"):
            writer.create_topic(metadata)
    names = {t.name for t in topics}
    for name, kind in [
        ("/reference/path", "nav_msgs/msg/Path"),
        ("/robot_box", "visualization_msgs/msg/Marker"),
    ]:
        if name not in names:
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=name, type=kind, serialization_format="cdr"
                )
            )
    counts = Counter()

    def write(topic, raw, stamp):
        writer.write(topic, raw, stamp)
        counts[topic] += 1

    for topic, raw in seeds:
        write(topic, raw, start)
    poses = []
    for topic, raw, stamp in messages(bag, start=start, end=end):
        if topic in (
            "/scenario/start",
            "/reference/path",
            "/robot_box",
            "/replay/path",
            "/replay/robot_box",
        ):
            continue
        write(topic, raw, stamp)
        if topic == "/reference/trajectory":
            msg = deserialize_message(raw, PoseStamped)
            if not poses:
                write("/scenario/start", raw, stamp)
            poses.append(msg)
            for name, message in display.sample(msg):
                write(name, serialize_message(message), stamp)
    writer = None
    if not poses:
        raise RuntimeError("No trajectory in selected interval")

    def pose(msg):
        q = msg.pose.orientation
        return dict(
            stamp_ns=msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            yaw=math.atan2(
                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
            ),
        )

    manifest = json.loads((src / "scenario.json").read_text())
    manifest.update(
        start=pose(poses[0]),
        recorded_endpoint=pose(poses[-1]),
        started_ros_ns=start,
        started_utc=datetime.fromtimestamp(start / 1e9, timezone.utc).isoformat(),
        elapsed_wall_seconds=(end - start) / 1e9,
        message_counts=dict(counts),
        estimated_travel_m=sum(
            math.hypot(
                b.pose.position.x - a.pose.position.x,
                b.pose.position.y - a.pose.position.y,
            )
            for a, b in zip(poses, poses[1:])
        ),
    )
    manifest["trim"] = dict(
        source=str(src),
        start_ns=start,
        end_ns=end,
        note="Boundary is assistant receipt of operator confirmation; exact UI click time unavailable. Original header stamps preserved. Map/static TF/calibration and 2-second TF pre-roll seeded at bag start; no pre-start images or trajectory retained.",
    )
    (dst / "scenario.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (dst / "operator_confirmation.json").write_text(
        json.dumps(control, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(output=str(dst), duration=(end - start) / 1e9, counts=dict(counts)),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
