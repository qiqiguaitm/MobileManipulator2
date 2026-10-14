"""GC perception service: capture one RGB-D/TF snapshot before the VLM request."""

import json, threading, time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2, numpy as np, yaml
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from robot_task_interfaces.srv import DetectScene
from robot_task_interfaces.msg import ObjectTarget
from .ros_support import LocatedNode, run
from .geometry import depth_position, rotation
from .vlm import Detector


def stamp(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


class ScenePerception(LocatedNode):
    def __init__(self):
        super().__init__("scene_perception")
        self.savedir = Path(
            self.param("evidence_dir", str(Path.home() / ".ros/robot_tasks/perception"))
        ).expanduser()
        self.http_timeout = self.param("vlm_timeout", 20.0)
        self.param("depth_max_spread_m", 1.0)
        self.calibration = None
        calibration_file = self.param("camera_extrinsics_file", "")
        workspace = Path(self.param("workspace", str(Path.cwd()))).expanduser()
        if calibration_file:
            calibration_file = Path(calibration_file).expanduser()
            if not calibration_file.is_absolute():
                calibration_file = workspace / calibration_file
            calibration_file = str(calibration_file)
        self.sensor_frame = self.param("extrinsics_sensor_frame", "rslidar")
        self.optical_frame = self.param(
            "extrinsics_camera_frame", "top_color_optical_frame"
        )
        if calibration_file:
            data = yaml.safe_load(Path(calibration_file).expanduser().read_text())[
                "transform"
            ]
            q = data["rotation"]
            p = data["translation"]
            self.calibration = (
                rotation([q[k] for k in ("x", "y", "z", "w")]),
                np.array([p[k] for k in ("x", "y", "z")], float),
            )
            if not np.isfinite(self.calibration[1]).all():
                raise ValueError("Invalid camera calibration translation")
        self.frames = {
            "rgb": deque(maxlen=10),
            "depth": deque(maxlen=10),
            "info": deque(maxlen=10),
        }
        self.frame_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.bridge = CvBridge()
        self.last_used = 0
        for key, kind, topic in [
            ("rgb", Image, "/camera/top/color/image_raw"),
            ("depth", Image, "/camera/top/aligned_depth_to_color/image_raw"),
            ("info", CameraInfo, "/camera/top/color/camera_info"),
        ]:
            self.create_subscription(
                kind,
                self.param(key + "_topic", topic),
                lambda m, k=key: self.receive(k, m),
                QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT),
                callback_group=self.group,
            )
        self.create_service(
            DetectScene, "/gc/detect_scene", self.detect, callback_group=self.group
        )

    def receive(self, key, msg):
        with self.frame_lock:
            self.frames[key].append(msg)

    def camera_transform(self, header):
        if self.calibration is None:
            return self.tf.lookup_transform(
                self.map_frame,
                header.frame_id,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.1),
            )
        if header.frame_id != self.optical_frame:
            raise ValueError("Image frame does not match calibrated optical frame")
        sensor = self.tf.lookup_transform(
            self.map_frame,
            self.sensor_frame,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=0.1),
        )
        q = sensor.transform.rotation
        p = sensor.transform.translation
        r = rotation([q.x, q.y, q.z, q.w])
        cr, cp = self.calibration
        xyz = r @ cp + np.array([p.x, p.y, p.z])
        quat = Rotation.from_matrix(r @ cr).as_quat()
        result = TransformStamped()
        result.header.frame_id = self.map_frame
        result.header.stamp = header.stamp
        result.child_frame_id = header.frame_id
        (
            result.transform.translation.x,
            result.transform.translation.y,
            result.transform.translation.z,
        ) = map(float, xyz)
        (
            result.transform.rotation.x,
            result.transform.rotation.y,
            result.transform.rotation.z,
            result.transform.rotation.w,
        ) = map(float, quat)
        return result

    def capture(self):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            with self.frame_lock:
                frames = {k: list(v) for k, v in self.frames.items()}
            now = self.get_clock().now().nanoseconds
            for rgb in reversed(frames["rgb"]):
                t = stamp(rgb)
                if t <= self.last_used or not 0 <= (now - t) / 1e9 < 0.7:
                    continue
                ds = sorted(frames["depth"], key=lambda m: abs(stamp(m) - t))
                infos = sorted(frames["info"], key=lambda m: abs(stamp(m) - t))
                if (
                    not ds
                    or not infos
                    or abs(stamp(ds[0]) - t) > 30_000_000
                    or abs(stamp(infos[0]) - t) > 200_000_000
                ):
                    continue
                depth, info = ds[0], infos[0]
                if (
                    not rgb.header.frame_id
                    == depth.header.frame_id
                    == info.header.frame_id
                ):
                    raise ValueError(
                        "RGB/depth/calibration frame mismatch; aligned depth required"
                    )
                if (rgb.width, rgb.height) != (depth.width, depth.height) or (
                    rgb.width,
                    rgb.height,
                ) != (info.width, info.height):
                    raise ValueError("RGB/depth/calibration size mismatch")
                try:
                    tf = self.camera_transform(rgb.header)
                except Exception:
                    continue
                image = self.bridge.imgmsg_to_cv2(rgb, "bgr8").copy()
                values = self.bridge.imgmsg_to_cv2(depth, "passthrough").copy()
                if depth.encoding == "16UC1":
                    values = values.astype(float) * 0.001
                elif depth.encoding == "32FC1":
                    values = values.astype(float)
                else:
                    raise ValueError("Unsupported depth encoding")
                if info.distortion_model not in (
                    "",
                    "plumb_bob",
                    "rational_polynomial",
                ):
                    raise ValueError("Unsupported camera distortion model")
                self.last_used = t
                return rgb, image, values, info, tf
            time.sleep(0.03)
        raise ValueError(
            "No fresh synchronized RGB/aligned-depth/calibration/capture-time TF"
        )

    def detect(self, request, response):
        if not self.request_lock.acquire(blocking=False):
            response.error = "Perception request already running"
            return response
        directory = self.savedir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            description = request.description.strip()
            if not 0 < len(description) <= 500:
                raise ValueError(
                    "Inspection requires an object description (1–500 characters)"
                )
            directory.mkdir(parents=True, exist_ok=False)
            response.evidence_dir = str(directory)
            rgb, image, depth, info, tf = self.capture()
            cv2.imwrite(str(directory / "rgb.jpg"), image)
            np.save(directory / "depth_m.npy", depth)
            q = tf.transform.rotation
            r = rotation([q.x, q.y, q.z, q.w])
            p = tf.transform.translation
            translation = np.array([p.x, p.y, p.z])
            (directory / "snapshot.json").write_text(
                json.dumps(
                    dict(
                        timestamp_ns=stamp(rgb),
                        camera_frame=rgb.header.frame_id,
                        camera_matrix=list(info.k),
                        distortion=list(info.d),
                        rotation=r.tolist(),
                        translation=translation.tolist(),
                    ),
                    indent=2,
                )
            )
            detections, meta = Detector(self.http_timeout).detect(image, description)
            (directory / "detections.json").write_text(
                json.dumps(
                    dict(description=description, detections=detections, vlm=meta),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            q = tf.transform.rotation
            r = rotation([q.x, q.y, q.z, q.w])
            p = tf.transform.translation
            translation = np.array([p.x, p.y, p.z])
            annotated = image.copy()
            records = []
            for i, item in enumerate(detections):
                x1, y1, x2, y2 = map(int, item["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imwrite(str(directory / "detections.jpg"), annotated)
            max_spread = float(self.get_parameter("depth_max_spread_m").value)
            if not np.isfinite(max_spread) or max_spread <= 0:
                raise ValueError("depth_max_spread_m must be finite and positive")
            for i, item in enumerate(detections):
                xyz, quality = depth_position(
                    depth,
                    item["bbox"],
                    info.k,
                    max_spread=max_spread,
                    distortion=info.d,
                )
                quality["max_spread_m"] = max_spread
                xyz = r @ xyz + translation
                obj = ObjectTarget()
                obj.name = item["name"]
                obj.instance_id = f"{stamp(rgb)}_{i}"
                obj.position.header.frame_id = self.map_frame
                obj.position.header.stamp = rgb.header.stamp
                obj.position.point.x, obj.position.point.y, obj.position.point.z = map(
                    float, xyz
                )
                response.objects.append(obj)
                records.append(dict(**item, xyz_map=xyz.tolist(), quality=quality))
            (directory / "observation.json").write_text(
                json.dumps(
                    dict(
                        timestamp_ns=stamp(rgb),
                        camera_frame=rgb.header.frame_id,
                        camera_matrix=list(info.k),
                        distortion=list(info.d),
                        rotation=r.tolist(),
                        translation=translation.tolist(),
                        detections=detections,
                        objects=records,
                        vlm=meta,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            response.success = True
        except Exception as e:
            response.success = False
            response.objects = []
            response.error = str(e)
            try:
                (directory / "error.json").write_text(
                    json.dumps(dict(error=str(e)), ensure_ascii=False)
                )
            except OSError:
                pass
        finally:
            self.request_lock.release()
        return response


def main():
    run(ScenePerception)
