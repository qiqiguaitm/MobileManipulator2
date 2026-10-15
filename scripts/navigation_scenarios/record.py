#!/usr/bin/env python3
"""Read-only ROS 2 scene recorder. Writes reference topics; publishes no commands."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="Scene name, e.g. straight_stop")
    parser.add_argument("--check", action="store_true", help="Check live inputs without recording")
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 means until Ctrl-C")
    parser.add_argument("--output", type=Path, default=Path("data/navigation_scenarios"))
    parser.add_argument("--odom-topic", default="/odom/fused")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "YAW"))
    parser.add_argument("--description", default="")
    parser.add_argument("--extra-topic", action="append", default=[], help="Additional discovered topic to record under /reference")
    parser.add_argument("--task-description", default="", help="Record task intent (does not execute a task)")
    parser.add_argument("--expected-map", type=Path)
    parser.add_argument("--status-topic", default="/status")
    parser.add_argument("--min-inlier", type=float, default=0.8)
    parser.add_argument("--ready-timeout", type=float, default=120)
    parser.add_argument("--ready-seconds", type=float, default=3)
    args = parser.parse_args()
    if not args.check and (not args.name or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name)):
        parser.error("A scene name containing only letters, digits, '_' or '-' is required")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative")
    if args.goal and not all(math.isfinite(v) for v in args.goal):
        parser.error("--goal must contain finite numbers")
    if not 0 < args.min_inlier <= 1:
        parser.error("--min-inlier must be in (0, 1]")
    if any(not math.isfinite(v) or v <= 0 for v in (args.ready_timeout, args.ready_seconds)):
        parser.error("Readiness timeouts must be positive finite numbers")
    return args


def main():
    args = arguments()
    # Keep --help usable on machines without ROS.
    import rclpy
    import rosbag2_py
    import yaml
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage
    from tf2_ros import Buffer, TransformException
    from hdl_localization.msg import ScanMatchingStatus
    from rclpy.executors import SingleThreadedExecutor
    from readiness import Readiness
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    class Recorder(Node):
        def __init__(self):
            super().__init__("navigation_scene_recorder")
            self.counts = Counter()
            self.tf = Buffer(cache_time=Duration(seconds=60))
            self.writer = None
            self.folder = None
            self.first_pose = None
            self.last_pose = None
            self.last_pose_stamp = None
            self.distance = 0.0
            self.ready = False
            self.scan = None
            self.scan_received = 0.0
            self.scan_tf_ok = False
            self.last_tf_error = None
            self.started = time.monotonic()
            self.started_ros_ns = self.get_clock().now().nanoseconds
            self.started_utc = datetime.now(timezone.utc).isoformat()
            live = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
            latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            # Map/static TF must remain available to a late-joining recorder.
            self.specs = [
                ("/map", OccupancyGrid, latched),
                ("/scan", LaserScan, live),
                ("/tf", TFMessage, live),
                ("/tf_static", TFMessage, latched),
                (args.odom_topic, Odometry, live),
                ("/cmd_vel", Twist, live),
                ("/goal_pose", PoseStamped, live),
                ("/initialpose", PoseWithCovarianceStamped, live),
                (args.status_topic, ScanMatchingStatus, live),
            ]
            if args.extra_topic:
                discovery_deadline = time.monotonic() + 5
                types = {}
                while time.monotonic() < discovery_deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    types = dict(self.get_topic_names_and_types())
                    if all(t in types for t in args.extra_topic):
                        break
                known = {t for t, _, _ in self.specs}
                for topic in args.extra_topic:
                    if topic in known:
                        continue
                    if topic not in types or len(types[topic]) != 1:
                        raise RuntimeError('Missing or ambiguous extra topic: '+topic)
                    endpoints = self.get_publishers_info_by_topic(topic)
                    qos = latched if endpoints and all(e.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for e in endpoints) else live
                    self.specs.append((topic, get_message(types[topic][0]), qos))
                    known.add(topic)
            if not args.check:
                suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                self.folder = args.output / f"{args.name}_{suffix}"
                self.folder.mkdir(parents=True, exist_ok=False)
                self.writer = rosbag2_py.SequentialWriter()
                self.writer.open(
                    rosbag2_py.StorageOptions(uri=str(self.folder / "bag"), storage_id="sqlite3"),
                    rosbag2_py.ConverterOptions("cdr", "cdr"))
                for topic, msg_type, qos in self.specs:
                    self.add_topic("/reference" + topic, msg_type, qos)
                for topic in ("/reference/trajectory", "/scenario/start", "/scenario/goal"):
                    self.add_topic(topic, PoseStamped, live)
                if args.task_description:
                    self.add_topic('/scenario/task', String, latched)
                    self.write('/scenario/task', String(data=args.task_description))
                if args.goal:
                    goal = PoseStamped()
                    goal.header.frame_id = args.map_frame
                    goal.header.stamp = self.get_clock().now().to_msg()
                    goal.pose.position.x, goal.pose.position.y = args.goal[:2]
                    goal.pose.orientation.z = math.sin(args.goal[2] / 2)
                    goal.pose.orientation.w = math.cos(args.goal[2] / 2)
                    self.write("/scenario/goal", goal)
                self.save_manifest("recording")
                print(f"Recording into {self.folder.resolve()}", flush=True)
            self.subscriptions_kept = []
            for topic, msg_type, qos in self.specs:
                self.subscriptions_kept.append(self.create_subscription(
                    msg_type, topic, lambda msg, topic=topic: self.receive(topic, msg), qos))
            self.create_timer(0.05, self.sample_pose)

        def add_topic(self, topic, msg_type, qos):
            package, namespace = msg_type.__module__.split(".")[:2]
            typename = package + "/" + namespace + "/" + msg_type.__name__
            profile = [{"history": 1, "depth": qos.depth,
                        "reliability": int(qos.reliability), "durability": int(qos.durability),
                        "deadline": {"sec": 9223372036, "nsec": 854775807},
                        "lifespan": {"sec": 9223372036, "nsec": 854775807},
                        "liveliness": 1,
                        "liveliness_lease_duration": {"sec": 9223372036, "nsec": 854775807},
                        "avoid_ros_namespace_conventions": False}]
            self.writer.create_topic(rosbag2_py.TopicMetadata(
                name=topic, type=typename, serialization_format="cdr",
                offered_qos_profiles=yaml.safe_dump(profile)))

        def write(self, topic, msg):
            if self.writer:
                self.writer.write(topic, serialize_message(msg), self.get_clock().now().nanoseconds)
            self.counts[topic] += 1

        def receive(self, topic, msg):
            self.write("/reference" + topic, msg)
            if topic in ("/tf", "/tf_static"):
                for transform in msg.transforms:
                    if topic == "/tf_static":
                        self.tf.set_transform_static(transform, "scene_input")
                    else:
                        self.tf.set_transform(transform, "scene_input")
            if topic == "/scan":
                self.scan = msg
                self.scan_received = time.monotonic()

        def sample_pose(self):
            if self.scan is not None and time.monotonic() - self.scan_received < 1.0:
                try:
                    self.tf.lookup_transform(args.map_frame, self.scan.header.frame_id,
                                             Time.from_msg(self.scan.header.stamp))
                    self.scan_tf_ok = True
                except TransformException as exc:
                    self.last_tf_error = str(exc)
            try:
                transform = self.tf.lookup_transform(args.map_frame, args.base_frame, Time())
            except TransformException as exc:
                self.last_tf_error = str(exc)
                return
            stamp = Time.from_msg(transform.header.stamp).nanoseconds
            age = (self.get_clock().now().nanoseconds - stamp) / 1e9
            if stamp == self.last_pose_stamp or age > 1.0 or age < -1.0:
                return
            pose = PoseStamped()
            pose.header = transform.header
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            self.last_pose_stamp = stamp
            self.write("/reference/trajectory", pose)
            q = pose.pose.orientation
            current = {"stamp_ns": stamp, "frame_id": args.map_frame,
                       "x": pose.pose.position.x, "y": pose.pose.position.y,
                       "yaw": math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))}
            if self.first_pose is None:
                self.first_pose = current
                self.write("/scenario/start", pose)
            if self.last_pose:
                self.distance += math.hypot(current["x"] - self.last_pose["x"],
                                            current["y"] - self.last_pose["y"])
            self.last_pose = current
            if not self.ready and not self.missing():
                self.ready = True
                print("READY: map, scan, odometry, TF and trajectory received. You may teleoperate.", flush=True)

        def missing(self):
            required = ["/map", "/scan", "/tf", "/tf_static", args.odom_topic, "/trajectory"]
            missing = ["/reference" + t for t in required if not self.counts["/reference" + t]]
            missing += ["/reference" + t for t in args.extra_topic
                        if not self.counts["/reference" + t]]
            if not self.scan_tf_ok:
                missing.append("map <- scan.frame at scan timestamp")
            return missing

        def save_manifest(self, status):
            if not self.folder:
                return
            try:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None
            manifest = {
                "schema_version": 1, "name": args.name, "description": args.description,
                "task_description": args.task_description,
                "status": status, "started_utc": self.started_utc,
                "failure_reason": getattr(self, 'failure_reason', None),
                "started_ros_ns": self.started_ros_ns,
                "elapsed_wall_seconds": time.monotonic() - self.started,
                "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"), "git_revision": revision,
                "map_frame": args.map_frame, "base_frame": args.base_frame,
                "localization_gate": {"topic": args.status_topic, "min_inlier": args.min_inlier,
                                      "stable_seconds": args.ready_seconds},
                "expected_map": str(args.expected_map) if args.expected_map else None,
                "topic_mapping": {t: "/reference" + t for t, _, _ in self.specs},
                "trajectory_source": "TF map->base_link, latest common timestamp; not cmd_vel integration",
                "start": self.first_pose, "recorded_endpoint": self.last_pose,
                "goal": {"frame_id": args.map_frame, "x": args.goal[0],
                         "y": args.goal[1], "yaw": args.goal[2]} if args.goal else None,
                "goal_note": "Explicit --goal is the benchmark target; endpoint is NOT an implicit goal.",
                "estimated_travel_m": self.distance,
                "message_counts": dict(self.counts), "missing": self.missing(),
                "scan_tf_resolved_at_least_once": self.scan_tf_ok,
                "last_tf_error": self.last_tf_error,
                "clock_policy": "Original message stamps preserved; bag uses ROS receive time. No clock remapping.",
            }
            (self.folder / "scenario.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    gate = Readiness(args)
    ready_since = None
    deadline = time.monotonic() + args.ready_timeout
    next_notice = 0
    gate_passed = False
    try:
        while rclpy.ok() and not stopping:
            rclpy.spin_once(gate, timeout_sec=0.05)
            problems = gate.problems()
            now = time.monotonic()
            ready_since = (ready_since or now) if not problems else None
            if ready_since and now-ready_since >= args.ready_seconds:
                print('LOCALIZATION READY: stable localization, expected map, fresh scan/odom and TF.', flush=True)
                gate_passed = True
                break
            if now >= next_notice:
                print('WAITING: '+('; '.join(problems) or 'confirming stable readiness'), flush=True)
                next_notice = now+5
            if now >= deadline:
                print('Readiness timeout. No bag created.', flush=True)
                return 2
        if not rclpy.ok() or stopping:
            return 2
        if args.check:
            return 0
    except KeyboardInterrupt:
        return 2
    finally:
        if args.check or not gate_passed or stopping:
            gate.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    node = Recorder()
    # Preserve pre-roll transforms so the first delayed scan can be reconstructed.
    # Header stamps remain original; these seed messages are written at bag start.
    node.receive('/map', gate.data['map'])
    node.receive('/tf_static', TFMessage(transforms=list(gate.static_transforms.values())))
    for _, message in gate.tf_history:
        node.receive('/tf', message)
    executor = SingleThreadedExecutor()
    executor.add_node(gate)
    executor.add_node(node)
    duration = args.duration or (8.0 if args.check else float("inf"))
    next_status = time.monotonic() + 5
    failed = False
    unhealthy_since = None
    try:
        while rclpy.ok() and not stopping and time.monotonic() - node.started < duration:
            executor.spin_once(timeout_sec=0.05)
            problems = gate.problems()
            now = time.monotonic()
            unhealthy_since = (unhealthy_since or now) if problems else None
            if unhealthy_since and now-unhealthy_since > 1.0:
                failed = True
                node.failure_reason = '; '.join(problems)
                print('STOP: readiness lost: '+ '; '.join(problems), flush=True)
                break
            if time.monotonic() >= next_status:
                print(f"scans={node.counts['/reference/scan']} poses={node.counts['/reference/trajectory']} "
                      f"rgb={node.counts['/reference/camera/top/color/image_raw']} "
                      f"depth={node.counts['/reference/camera/top/aligned_depth_to_color/image_raw']} "
                      f"travel≈{node.distance:.2f}m missing={node.missing()}", flush=True)
                next_status += 5
    except KeyboardInterrupt:
        pass
    except Exception:
        failed = True
        raise
    finally:
        missing = node.missing()
        status = "incomplete" if missing or failed else "captured_needs_review"
        node.save_manifest(status)
        # Humble writer finalizes metadata on destruction.
        node.writer = None
        print(json.dumps({"status": status, "missing": missing,
                          "counts": dict(node.counts), "travel_m": node.distance}, ensure_ascii=False), flush=True)
        node.destroy_node()
        executor.shutdown()
        gate.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 2 if missing or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
