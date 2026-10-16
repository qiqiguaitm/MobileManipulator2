#!/usr/bin/env python3
"""Run real scene perception on a recorded RGB-D snapshot, without ROS publication."""
import argparse, json, time, math
from bag_io import messages
from scene_topics import SceneInput
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--target", default="垃圾桶", help="Exact object description; no fallback to other classes")
    p.add_argument("--snapshot-offset", type=float, default=0., help="Recorded RGB-D offset in seconds")
    a = p.parse_args(argv)
    a.target = a.target.strip()
    if not a.target or not math.isfinite(a.snapshot_offset) or a.snapshot_offset < 0:
        p.error("Target must not be empty; snapshot offset must be finite and nonnegative")
    return a


def main():
    a = parse_args()
    import numpy as np
    import cv2, yaml
    import rclpy
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.serialization import deserialize_message
    from tf2_ros import Buffer
    from tf2_msgs.msg import TFMessage
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    from robot_tasks.geometry import rotation, depth_position, inspection_goals
    from robot_tasks.vlm import Detector

    a.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "stage": "load_snapshot", "scene": str(a.scene),
              "target_description": a.target, "snapshot_offset_seconds": a.snapshot_offset}
    rclpy.init()
    try:
        bag = a.scene / "bag"
        source = SceneInput(bag)
        t0 = next(messages(bag))[2] + round(a.snapshot_offset * 1e9)
        buffer = Buffer(cache_time=Duration(seconds=180))
        for topic, raw, _ in source.messages(["/tf", "/tf_static"]):
            for t in deserialize_message(raw, TFMessage).transforms:
                (
                    buffer.set_transform_static
                    if topic.endswith("tf_static")
                    else buffer.set_transform
                )(t, "recorded")

        def read(topic, kind, margin=0):
            return [
                (stamp, deserialize_message(raw, kind))
                for _, raw, stamp in source.messages(
                    [topic], start=t0-margin, end=t0 + 2_000_000_000
                )
            ]

        def stamp(m):
            return m.header.stamp.sec * 10**9 + m.header.stamp.nanosec

        rgb = read("/camera/top/color/image_raw", Image)
        dep = read("/camera/top/aligned_depth_to_color/image_raw", Image, 1_000_000_000)
        infos = read("/camera/top/color/camera_info", CameraInfo, 1_000_000_000)
        if not rgb or not dep or not infos:
            raise ValueError("No synchronized RGB-D/calibration in selected window")
        pairs = [(abs(stamp(im) - stamp(dm)), im, dm) for _, im in rgb for _, dm in dep]
        delta, im, dm = min(pairs, key=lambda x: x[0])
        info = min(infos, key=lambda item: abs(stamp(item[1])-stamp(im)))[1]
        if delta >= 50_000_000:
            raise ValueError("RGB/depth timestamp mismatch")
        if abs(stamp(info)-stamp(im)) > 100_000_000 or not (im.header.frame_id == dm.header.frame_id == info.header.frame_id):
            raise ValueError("RGB-D/calibration frame or timestamp mismatch")
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(im, "bgr8")
        depth = bridge.imgmsg_to_cv2(dm, "passthrough").astype(float)
        if dm.encoding == "16UC1":
            depth *= 0.001
        elif dm.encoding != "32FC1":
            raise ValueError("Unsupported depth encoding")
        cv2.imwrite(str(a.output / "rgb.jpg"), image)
        np.save(a.output / "depth_m.npy", depth)
        profile = yaml.safe_load(
            (
                a.workspace / "src/task_control/robot_tasks/config/robot02.yaml"
            ).read_text()
        )["scene_perception"]["ros__parameters"]
        calib_path = a.workspace / profile["camera_extrinsics_file"]
        saved_calib = a.scene / "capture_assets/extrinsics_lidar_to_top_camera_optical_frame.yaml"
        if saved_calib.is_file():
            calib_path = saved_calib
        calib = yaml.safe_load(calib_path.read_text())
        (a.output / "camera_calibration.yaml").write_text(yaml.safe_dump(calib))
        tf = buffer.lookup_transform("map", "rslidar", Time.from_msg(im.header.stamp))
        q = tf.transform.rotation
        tr = tf.transform.translation
        r = rotation([q.x, q.y, q.z, q.w])
        cal = calib["transform"]
        q = cal["rotation"]
        cr = rotation([q[k] for k in ("x", "y", "z", "w")])
        cp = np.array([cal["translation"][k] for k in ("x", "y", "z")])
        trans = r @ cp + np.array([tr.x, tr.y, tr.z])
        rot = r @ cr
        result.update(
            stage="detection",
            snapshot_stamp_ns=stamp(im),
            rgb_depth_delta_seconds=delta / 1e9,
            camera_frame=im.header.frame_id,
            calibration_source=str(calib_path),
        )
        started = time.monotonic()
        detections, meta = Detector(timeout=25).detect(image, a.target)
        if any(d.get("name") != a.target for d in detections):
            raise ValueError("Detector returned a different target class")
        result.update(
            detections=detections,
            vlm=meta,
            detection_seconds=time.monotonic() - started,
        )
        (a.output / "detections.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        result["stage"] = "depth_localization"
        objects = []
        for d in detections:
            xyz, quality = depth_position(
                depth,
                d["bbox"],
                info.k,
                max_spread=profile["depth_max_spread_m"],
                distortion=info.d,
            )
            xyz = rot @ xyz + trans
            objects.append(dict(**d, xyz_map=xyz.tolist(), quality=quality))
            b = list(map(int, d["bbox"]))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cv2.imwrite(str(a.output / "detections.jpg"), image)
        if not objects:
            raise ValueError("No matching object detected")
        manifest = json.loads((a.scene / "scenario.json").read_text())
        robot = manifest["start"]
        pose = [robot["x"], robot["y"], robot["yaw"]]
        result["start_source"] = "scenario_start"
        if a.snapshot_offset > 0:
            base = buffer.lookup_transform("map", "base_link", Time.from_msg(im.header.stamp))
            b, q = base.transform.translation, base.transform.rotation
            pose = [b.x, b.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
            result["start_source"] = "recorded_base_tf_at_snapshot"
        selected = min(
            objects,
            key=lambda o: math.hypot(
                o["xyz_map"][0] - pose[0], o["xyz_map"][1] - pose[1]
            ),
        )
        candidates = inspection_goals(
            selected["xyz_map"],
            pose,
            0.934,
            [math.radians(v) for v in [0, 30, -30, 60, -60]],
        )
        result.update(
            status="prepared",
            stage="perception_complete",
            objects=objects,
            selected=selected,
            start=pose,
            candidates=candidates,
            depth_spread_limit_m=profile["depth_max_spread_m"],
        )
    except Exception as e:
        result["error"] = str(e)
    finally:
        (a.output / "perception.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False))
        rclpy.shutdown()
    return 0 if result["status"] == "prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
