#!/usr/bin/env python3
"""Run real scene perception on a recorded RGB-D snapshot, without ROS publication."""
import argparse, json, time, math
from bag_io import messages
from scene_topics import SceneInput
from pathlib import Path
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "stage": "load_snapshot", "scene": str(a.scene)}
    rclpy.init()
    try:
        bag = a.scene / "bag"
        source = SceneInput(bag)
        t0 = next(messages(bag))[2]
        buffer = Buffer(cache_time=Duration(seconds=180))
        for topic, raw, _ in source.messages(["/tf", "/tf_static"]):
            for t in deserialize_message(raw, TFMessage).transforms:
                (
                    buffer.set_transform_static
                    if topic.endswith("tf_static")
                    else buffer.set_transform
                )(t, "recorded")

        def read(topic, kind):
            return [
                (stamp, deserialize_message(raw, kind))
                for _, raw, stamp in source.messages(
                    [topic], start=t0, end=t0 + 2_000_000_000
                )
            ]

        def stamp(m):
            return m.header.stamp.sec * 10**9 + m.header.stamp.nanosec

        rgb = read("/camera/top/color/image_raw", Image)
        dep = read("/camera/top/aligned_depth_to_color/image_raw", Image)
        info = read("/camera/top/color/camera_info", CameraInfo)[0][1]
        pairs = [(abs(stamp(im) - stamp(dm)), im, dm) for _, im in rgb for _, dm in dep]
        delta, im, dm = min(pairs, key=lambda x: x[0])
        assert delta < 50_000_000, "RGB/depth timestamp mismatch"
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
        detections, meta = Detector(timeout=20).detect(image, "垃圾桶")
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
