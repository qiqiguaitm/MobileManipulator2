#!/usr/bin/env python3
"""Approximate closed-loop Nav2 test around a recorded static-scene trajectory.
No hardware nodes. Recorded endpoint never supplies the navigation goal.
"""
import argparse, copy, json, math, os, signal, subprocess, time
from pathlib import Path as FilePath
from bag_io import messages, writer as mcap_writer
from scene_topics import SceneInput, REFERENCE_TOPICS
from rosidl_runtime_py.utilities import get_message
from display_messages import RecordedDisplay, detected_target_messages
import numpy as np, yaml
import rclpy, rosbag2_py
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformBroadcaster, StaticTransformBroadcaster
from nav2_msgs.action import ComputePathToPose, FollowPath
from robot_tasks.geometry import rotation, wrap


def xytheta(p):
    q = p.orientation
    return [
        p.position.x,
        p.position.y,
        math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
    ]


def load(scene):
    buf = Buffer(cache_time=Duration(seconds=180))
    source = SceneInput(scene / "bag")

    def rows(topic):
        return ((raw,) for _, raw, _ in source.messages([topic]))

    for topic in ["/tf_static", "/tf"]:
        for (raw,) in rows(topic):
            for tf in deserialize_message(raw, TFMessage).transforms:
                (
                    buf.set_transform_static
                    if topic.endswith("static")
                    else buf.set_transform
                )(tf, "source")
    grid = deserialize_message(next(iter(rows("/map")))[0], OccupancyGrid)
    refs = [deserialize_message(raw, PoseStamped) for raw, in rows("/current_pose")]
    scans = []
    skipped = 0
    for (raw,) in rows("/scan"):
        scan = deserialize_message(raw, LaserScan)
        try:
            tf = buf.lookup_transform(
                "map", scan.header.frame_id, Time.from_msg(scan.header.stamp)
            )
            base = buf.lookup_transform(
                "map", "base_link", Time.from_msg(scan.header.stamp)
            )
            extrinsic = buf.lookup_transform(
                "base_link", scan.header.frame_id, Time.from_msg(scan.header.stamp)
            )
        except Exception:
            skipped += 1
            continue
        q = tf.transform.rotation
        p = tf.transform.translation
        r = rotation([q.x, q.y, q.z, q.w])
        ranges = np.array(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        good = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges <= scan.range_max)
        )
        points = np.column_stack(
            [
                ranges[good] * np.cos(angles[good]),
                ranges[good] * np.sin(angles[good]),
                np.zeros(good.sum()),
            ]
        ) @ r.T + np.array([p.x, p.y, p.z])
        b = base.transform.translation
        q = base.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        scans.append((np.array([b.x, b.y, yaw]), points[:, :2], scan, base, extrinsic))
    return grid, refs, scans, skipped


class Sim(Node):
    def __init__(self, grid, refs, scans, start, out, target_position):
        super().__init__("recorded_scene_simulator")
        self.display = RecordedDisplay()
        self.target_position = target_position
        self.grid, self.refs, self.scans = grid, refs, scans
        self.refposes = np.array([xytheta(p.pose) for p in refs])
        self.scanposes = np.array([s[0] for s in scans])
        self.pose = np.array(start, dtype=float)
        self.command = Twist()
        self.cmd_time = -1.0
        self.last = time.monotonic()
        self.elapsed = 0.0
        self.active = False
        self.halt = None
        self.samples = []
        self.path = Path()
        self.max_deviation = 0.0
        self.max_yaw = 0.0
        self.min_clearance = float("inf")
        self.writer = mcap_writer(out / "result_bag")
        self.created = set()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.odom_pub = self.create_publisher(Odometry, "/odom/fused", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.ref_pub = self.create_publisher(Path, "/reference/path", latched)
        self.tf = TransformBroadcaster(self)
        self.st = StaticTransformBroadcaster(self)
        static = []
        for parent, child, z in [("map", "odom", 0.0)]:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.rotation.w = 1.0
            t.transform.translation.z = z
            static.append(t)
        self.st.sendTransform(static)
        self.write("/tf_static", TFMessage(transforms=static))
        self.grid.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(grid)
        self.write("/map", grid)
        reference = Path()
        reference.header.frame_id = "map"
        reference.header.stamp = self.get_clock().now().to_msg()
        reference.poses = refs
        self.ref_pub.publish(reference)
        self.write("/reference/path", reference)
        self.create_subscription(Twist, "/cmd_vel", self.receive_command, 10)
        self.create_timer(0.02, self.tick)
        self.create_timer(0.1, self.observation)
        cells = np.array(grid.data).reshape(grid.info.height, grid.info.width)
        yy, xx = np.where(cells >= 65)
        res = grid.info.resolution
        q = grid.info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        r = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        self.map_points = np.column_stack(
            [(xx + 0.5) * res, (yy + 0.5) * res]
        ) @ r.T + np.array([grid.info.origin.position.x, grid.info.origin.position.y])

    def write(self, topic, msg):
        if topic not in self.created:
            pkg, ns = type(msg).__module__.split(".")[:2]
            kind = pkg + "/" + ns + "/" + type(msg).__name__
            self.writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=topic, type=kind, serialization_format="cdr"
                )
            )
            self.created.add(topic)
        self.writer.write(
            topic, serialize_message(msg), self.get_clock().now().nanoseconds
        )

    def load_reference(self, scene):
        source = SceneInput(scene / "bag")
        self.reference_events = []
        self.reference_index = 0
        for topic, raw, stamp in source.messages(REFERENCE_TOPICS):
            metadata = source.metadata[source.topics[topic]]
            if metadata.type != REFERENCE_TOPICS[topic]:
                raise ValueError("Unexpected reference type: " + topic)
            self.reference_events.append(
                (
                    stamp,
                    "/reference" + topic,
                    deserialize_message(raw, get_message(metadata.type)),
                )
            )
        self.reference_t0 = next(messages(scene / "bag"))[2]
        self.reference_started = time.monotonic()
        self.reference_pubs = {}
        for _, topic, msg in self.reference_events:
            if topic not in self.reference_pubs:
                self.reference_pubs[topic] = self.create_publisher(type(msg), topic, 10)
        self.create_timer(0.02, self.replay_reference)

    def replay_reference(self):
        elapsed = int((time.monotonic() - self.reference_started) * 1e9)
        while self.reference_index < len(self.reference_events):
            stamp, topic, msg = self.reference_events[self.reference_index]
            if stamp - self.reference_t0 > elapsed:
                break
            msg = copy.deepcopy(msg)
            if hasattr(msg, "header"):
                msg.header.stamp = self.get_clock().now().to_msg()
            self.reference_pubs[topic].publish(msg)
            self.write(topic, msg)
            self.reference_index += 1

    def receive_command(self, msg):
        self.command = msg
        self.cmd_time = time.monotonic()
        self.write("/cmd_vel", msg)

    def pose_msg(self, p=None):
        p = self.pose if p is None else p
        m = PoseStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = float(p[0])
        m.pose.position.y = float(p[1])
        m.pose.orientation.z = math.sin(p[2] / 2)
        m.pose.orientation.w = math.cos(p[2] / 2)
        return m

    def tick(self):
        now = time.monotonic()
        dt = min(now - self.last, 0.1)
        self.last = now
        v = w = 0.0
        if self.active and not self.halt and now - self.cmd_time < 0.4:
            v = self.command.linear.x
            w = self.command.angular.z
            if not np.isfinite([v, w]).all() or abs(v) > 0.6 or abs(w) > 2.0:
                self.halt = "INVALID_COMMAND"
                v = w = 0.0
            theta = self.pose[2]
            if abs(w) > 1e-8:
                self.pose[:2] += np.array(
                    [
                        v / w * (math.sin(theta + w * dt) - math.sin(theta)),
                        v / w * (-math.cos(theta + w * dt) + math.cos(theta)),
                    ]
                )
            else:
                self.pose[:2] += v * dt * np.array([math.cos(theta), math.sin(theta)])
            self.pose[2] = wrap(theta + w * dt)
            self.elapsed += dt
        m = self.pose_msg()
        t = TransformStamped()
        t.header.stamp = m.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = m.pose.position.x
        t.transform.translation.y = m.pose.position.y
        t.transform.rotation = m.pose.orientation
        self.tf.sendTransform(t)
        self.write("/tf", TFMessage(transforms=[t]))
        odom = Odometry()
        odom.header = t.header
        odom.child_frame_id = "base_link"
        odom.pose.pose = m.pose
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)
        self.write("/odom/fused", odom)
        self.path.header = m.header
        self.path.poses.append(m)
        self.path_pub.publish(self.path)
        self.write("/current_pose", m)
        for topic, message in self.display.sample(m):
            self.write(topic, message)
        for topic, message in detected_target_messages(
            self.target_position, m.header.stamp
        ):
            self.write(topic, message)
        if self.active:
            self.samples.append([self.elapsed, *self.pose, v, w])

    def observation(self):
        delta = self.scanposes[:, :2] - self.pose[:2]
        dist = np.linalg.norm(delta, axis=1)
        angles = np.abs(
            np.arctan2(
                np.sin(self.scanposes[:, 2] - self.pose[2]),
                np.cos(self.scanposes[:, 2] - self.pose[2]),
            )
        )
        idx = int(np.argmin(dist + 0.15 * angles))
        self.max_deviation = max(self.max_deviation, float(dist[idx]))
        self.max_yaw = max(self.max_yaw, float(angles[idx]))
        if self.active and (dist[idx] > 0.5 or angles[idx] > math.radians(45)):
            self.halt = "OBSERVATION_APPROXIMATION_LIMIT"
        points = self.scans[idx][1]
        theta = self.pose[2]
        r = np.array(
            [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
        )
        # Reuse the recorded beam measurements at their recorded world pose.
        # Only the playback timestamp and frame names change; no range rebinning.
        original_scan, original_base, original_extrinsic = self.scans[idx][2:]
        stamp = self.get_clock().now().to_msg()
        base = copy.deepcopy(original_base)
        base.header.stamp = stamp
        base.header.frame_id = "map"
        base.child_frame_id = "reference/base_link"
        laser = copy.deepcopy(original_extrinsic)
        laser.header.stamp = stamp
        laser.header.frame_id = "reference/base_link"
        laser.child_frame_id = "reference/scan"
        transforms = TFMessage(transforms=[base, laser])
        self.tf.sendTransform(transforms.transforms)
        self.write("/tf", transforms)
        scan = copy.deepcopy(original_scan)
        scan.header.stamp = stamp
        scan.header.frame_id = "reference/scan"
        self.scan_pub.publish(scan)
        self.write("/scan", scan)
        near = self.map_points[
            np.linalg.norm(self.map_points - self.pose[:2], axis=1) < 1.5
        ]
        allpoints = np.concatenate([points, near])
        local = (allpoints - self.pose[:2]) @ r.T
        dx = np.maximum(np.maximum(-0.375 - local[:, 0], local[:, 0] - 0.434), 0)
        dy = np.maximum(np.abs(local[:, 1]) - 0.31, 0)
        clearance = float(np.min(np.hypot(dx, dy)))
        if self.active:
            self.min_clearance = min(self.min_clearance, clearance)
            if clearance <= 0:
                self.halt = "APPROXIMATE_FOOTPRINT_COLLISION"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=FilePath)
    p.add_argument("output", type=FilePath)
    p.add_argument("--workspace", type=FilePath, default=FilePath.cwd())
    a = p.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "149":
        raise SystemExit("Test requires isolated ROS_DOMAIN_ID=149")
    out = a.output
    perception = json.loads((out / "perception.json").read_text())
    assert perception["status"] == "prepared"
    rclpy.init()
    children = []
    node = None
    summary = {
        "status": "failed",
        "scope": "Static-scene approximate closed loop; unchanged recorded scan beams at recorded poses and extrinsics, nearest-reference selection, marking only without ray clearing. Ideal differential-drive dynamics; no hardware or localization test.",
        "observation_mode": "recorded_scan_at_reference_pose",
    }
    try:
        grid, refs, scans, skipped = load(a.scene)
        node = Sim(
            grid,
            refs,
            scans,
            perception["start"],
            out,
            perception["selected"]["xyz_map"],
        )
        summary["untransformable_reference_scans"] = skipped
        node.load_reference(a.scene)
        config = yaml.safe_load(
            (a.workspace / "src/slam/config/nav2_minimal_params.yaml").read_text()
        )
        config["controller_server"] = yaml.safe_load(
            (out / "controller_server.yaml").read_text()
        )["/controller_server"]
        # Snapshot the running costmaps as well as the controller where available.
        for name in ["local_costmap", "global_costmap"]:
            files = list(out.glob("*" + name + "*.yaml"))
            if files:
                d = yaml.safe_load(files[0].read_text())
                params = next(iter(d.values()))["ros__parameters"]
                config[name] = {name: {"ros__parameters": params}}
        config["controller_server"]["ros__parameters"]["odom_topic"] = "/odom/fused"
        # Recorded rays are historical observations, not free-space evidence from
        # the simulated robot. Apply only to this run's generated configuration.
        for name in ["local_costmap", "global_costmap"]:
            costmap = config[name][name]["ros__parameters"]
            layer = costmap.setdefault("obstacle_layer", {})
            source = layer.setdefault("scan", {})
            source.update(
                topic="/scan",
                sensor_frame="reference/scan",
                marking=True,
                clearing=False,
                inf_is_valid=False,
            )
            # Parameter snapshots may also contain dotted ROS parameter names.
            for key, value in source.items():
                dotted = "obstacle_layer.scan." + key
                if dotted in costmap:
                    costmap[dotted] = value

        # ROS parameter dumps encode empty arrays without element types; the Humble
        # YAML loader cannot redeclare these. Let the plugins supply their defaults.
        def clean(value):
            if isinstance(value, dict):
                return {
                    k: clean(v) for k, v in value.items() if v is not None and v != []
                }
            return value

        config = clean(config)
        params = out / "test_params.yaml"
        params.write_text(yaml.safe_dump(config))

        def spawn(name, args):
            f = (out / (name + ".log")).open("w")
            child = subprocess.Popen(
                args,
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            f.close()
            children.append(child)

        for pkg, exe in [
            ("nav2_planner", "planner_server"),
            ("nav2_controller", "controller_server"),
        ]:
            spawn(
                exe,
                ["ros2", "run", pkg, exe, "--ros-args", "--params-file", str(params)],
            )
        spawn(
            "lifecycle",
            [
                "ros2",
                "run",
                "nav2_lifecycle_manager",
                "lifecycle_manager",
                "--ros-args",
                "-r",
                "__node:=test_lifecycle",
                "-p",
                "autostart:=true",
                "-p",
                "node_names:=[planner_server,controller_server]",
            ],
        )

        def wait(f, seconds):
            end = time.monotonic() + seconds
            while not f.done() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.02)
                if any(c.poll() is not None for c in children):
                    raise RuntimeError("Nav2 process exited")
            if not f.done():
                raise RuntimeError("Action timeout")
            return f.result()

        planner = ActionClient(node, ComputePathToPose, "/compute_path_to_pose")
        controller = ActionClient(node, FollowPath, "/follow_path")
        deadline = time.monotonic() + 35
        while not (planner.server_is_ready() and controller.server_is_ready()):
            rclpy.spin_once(node, timeout_sec=0.05)
            if any(c.poll() is not None for c in children):
                raise RuntimeError("Nav2 process exited during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("Nav2 servers did not activate")
        for _ in range(100):
            rclpy.spin_once(node, timeout_sec=0.02)
        choices = []
        summary["candidates"] = []
        for candidate in perception["candidates"]:
            req = ComputePathToPose.Goal()
            req.goal = node.pose_msg(candidate)
            req.planner_id = "GridBased"
            req.use_start = False
            h = wait(planner.send_goal_async(req), 5)
            if not h.accepted:
                summary["candidates"].append({"pose": candidate, "status": "rejected"})
                continue
            result = wait(h.get_result_async(), 12)
            path = result.result.path
            summary["candidates"].append(
                {"pose": candidate, "status": result.status, "poses": len(path.poses)}
            )
            if (
                result.status == 4
                and path.poses
                and math.hypot(
                    path.poses[-1].pose.position.x - candidate[0],
                    path.poses[-1].pose.position.y - candidate[1],
                )
                <= 0.1
            ):
                length = sum(
                    math.hypot(
                        b.pose.position.x - c.pose.position.x,
                        b.pose.position.y - c.pose.position.y,
                    )
                    for c, b in zip(path.poses, path.poses[1:])
                )
                choices.append((length, candidate, path))
        if not choices:
            raise RuntimeError("No reachable stopping pose")
        length, goal, path = min(choices, key=lambda x: x[0])
        summary.update(goal=goal, planned_length_m=length)
        node.write("/plan", path)
        node.write("/goal_pose", node.pose_msg(goal))
        req = FollowPath.Goal()
        req.path = path
        req.controller_id = "FollowPath"
        req.goal_checker_id = "goal_checker"
        node.active = True
        handle = wait(controller.send_goal_async(req), 5)
        if not handle.accepted:
            raise RuntimeError("Controller rejected path")
        future = handle.get_result_async()
        deadline = time.monotonic() + 90
        last_report = 0
        while not future.done() and time.monotonic() < deadline and not node.halt:
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic() - last_report > 5:
                print(
                    json.dumps(
                        dict(
                            t=node.elapsed,
                            pose=node.pose.tolist(),
                            deviation=node.max_deviation,
                            halt=node.halt,
                        )
                    ),
                    flush=True,
                )
                last_report = time.monotonic()
        if not future.done():
            wait(handle.cancel_goal_async(), 4)
            wait(future, 5)
            summary["status"] = node.halt or "TIMEOUT"
        else:
            summary["status"] = (
                "navigation_succeeded"
                if future.result().status == 4
                else "controller_failed"
            )
            summary["action_status"] = future.result().status
            summary["controller_result"] = str(future.result().result)
        node.active = False
        summary.update(
            final_pose=node.pose.tolist(),
            goal_position_error_m=math.hypot(
                node.pose[0] - goal[0], node.pose[1] - goal[1]
            ),
            goal_yaw_error_rad=abs(wrap(node.pose[2] - goal[2])),
        )
    except Exception as e:
        summary["error"] = str(e)
    finally:
        if node:
            node.active = False
            samples = np.array(node.samples)
            if len(samples):
                np.savetxt(
                    out / "trajectory.csv",
                    samples,
                    delimiter=",",
                    header="time,x,y,yaw,v,w",
                    comments="",
                )
                ref = node.refposes
                errors = np.min(
                    np.linalg.norm(samples[:, None, 1:3] - ref[None, :, :2], axis=2),
                    axis=1,
                )
                summary.update(
                    duration_seconds=float(samples[-1, 0]),
                    reference_nearest_rmse_m=float(np.sqrt(np.mean(errors**2))),
                    reference_max_distance_m=float(errors.max()),
                    reference_endpoint_error_m=float(
                        np.linalg.norm(samples[-1, 1:3] - ref[-1, :2])
                    ),
                    distance_m=float(
                        np.linalg.norm(np.diff(samples[:, 1:3], axis=0), axis=1).sum()
                    ),
                )
            summary.update(
                max_scan_reference_distance_m=node.max_deviation,
                max_scan_reference_yaw_rad=node.max_yaw,
                minimum_approximate_obstacle_clearance_m=(
                    node.min_clearance if math.isfinite(node.min_clearance) else None
                ),
            )
            node.writer = None
            node.destroy_node()
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        (out / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        rclpy.shutdown()
    return 0 if summary["status"] == "navigation_succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
