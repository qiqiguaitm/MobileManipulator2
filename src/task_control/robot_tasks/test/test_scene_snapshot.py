import os, time, threading

os.environ["ROS_DOMAIN_ID"] = "166"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
import numpy as np
import pytest
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from robot_task_interfaces.srv import DetectScene
import robot_tasks.scene as module


@pytest.mark.parametrize(
    "depth_mm,expected_success,calibrated",
    [(2000, True, False), (0, False, False), (2000, True, True)],
)
def test_capture_before_vlm_keeps_original_depth_and_tf(
    tmp_path, monkeypatch, depth_mm, expected_success, calibrated
):
    rclpy.init()
    scene = module.ScenePerception()
    scene.savedir = tmp_path
    if calibrated:
        scene.calibration = (np.eye(3), np.array([0.2, 0.3, 0.4]))
        scene.optical_frame = "camera_optical"
    source = Node("snapshot_fixture")
    bridge = CvBridge()
    broadcaster = TransformBroadcaster(source)
    rgbpub = source.create_publisher(Image, "/camera/top/color/image_raw", 10)
    depthpub = source.create_publisher(
        Image, "/camera/top/aligned_depth_to_color/image_raw", 10
    )
    infopub = source.create_publisher(CameraInfo, "/camera/top/color/camera_info", 10)
    current = {"mm": depth_mm, "x": 1.0}

    def publish():
        stamp = source.get_clock().now().to_msg()
        rgb = bridge.cv2_to_imgmsg(np.zeros((100, 100, 3), np.uint8), "bgr8")
        depth = bridge.cv2_to_imgmsg(
            np.full((100, 100), current["mm"], np.uint16), "16UC1"
        )
        info = CameraInfo()
        info.width = 100
        info.height = 100
        info.k = [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        info.distortion_model = "plumb_bob"
        for m in (rgb, depth, info):
            m.header.stamp = stamp
            m.header.frame_id = "camera_optical"
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = scene.sensor_frame if calibrated else "camera_optical"
        t.transform.rotation.w = 1.0
        t.transform.translation.x = current["x"]
        broadcaster.sendTransform(t)
        rgbpub.publish(rgb)
        depthpub.publish(depth)
        infopub.publish(info)

    timer = source.create_timer(0.05, publish)

    class FakeDetector:
        def __init__(self, *_):
            pass

        def detect(self, image, names):
            current.update(mm=3500, x=20.0)
            time.sleep(0.3)  # New camera frames/TF arrive during model latency.
            return [dict(name="杯子", bbox=[10, 10, 90, 90])], dict(provider="mock")

    monkeypatch.setattr(module, "Detector", FakeDetector)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(scene)
    ex.add_node(source)
    thread = threading.Thread(target=ex.spin, daemon=True)
    thread.start()
    try:
        time.sleep(0.5)
        req = DetectScene.Request()
        req.description = "杯子"
        response = scene.detect(req, DetectScene.Response())
        assert response.success == expected_success, response.error
        assert (tmp_path / response.evidence_dir / "snapshot.json").is_file()
        if expected_success:
            p = response.objects[0].position.point
            assert p.x == pytest.approx(1.19 if calibrated else 0.99, abs=0.02)
            assert p.z == pytest.approx(2.4 if calibrated else 2.0)
        else:
            assert response.objects == [] and "Insufficient" in response.error
    finally:
        timer.cancel()
        ex.shutdown(timeout_sec=3)
        thread.join(timeout=3)
        scene.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("absolute", [False, True])
def test_camera_calibration_path_uses_explicit_workspace(tmp_path, absolute):
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(
        """transform:
  rotation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  translation: {x: 0.2, y: 0.3, z: 0.4}
"""
    )
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "workspace:=" + str(tmp_path),
            "-p",
            "camera_extrinsics_file:="
            + (str(calibration) if absolute else calibration.name),
        ]
    )
    scene = None
    try:
        scene = module.ScenePerception()
        assert scene.calibration[1] == pytest.approx([0.2, 0.3, 0.4])
    finally:
        if scene is not None:
            scene.destroy_node()
        rclpy.shutdown()
