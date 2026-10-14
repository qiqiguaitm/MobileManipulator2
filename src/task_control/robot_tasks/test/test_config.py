"""Verify layered robot02 configuration and its ROS parameter-file loading."""

from pathlib import Path

import pytest
import rclpy
import yaml
from rclpy.node import Node

CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_generic_configuration_does_not_enable_measured_mock():
    generic = yaml.safe_load((CONFIG / "controllers.yaml").read_text())
    assert not generic["navigation_controller"]["ros__parameters"][
        "allow_mock_front_execution"
    ]
    assert generic["scene_perception"]["ros__parameters"]["depth_max_spread_m"] == 0.12


def test_robot02_ros_overrides_preserve_verified_standoff_and_depth():
    # Loading through rclpy validates actual ROS merge order, not just YAML syntax.
    rclpy.init(
        args=[
            "--ros-args",
            "--params-file",
            str(CONFIG / "controllers.yaml"),
            "--params-file",
            str(CONFIG / "robot02.yaml"),
        ]
    )
    nodes = []
    try:
        nav = Node(
            "navigation_controller",
            automatically_declare_parameters_from_overrides=True,
        )
        scene = Node(
            "scene_perception", automatically_declare_parameters_from_overrides=True
        )
        nodes.extend([nav, scene])
        assert nav.get_parameter("allow_execution").value is False
        assert nav.get_parameter("allow_mock_front_execution").value is True
        assert nav.get_parameter("front_extent").value + nav.get_parameter(
            "grasp_front_clearance"
        ).value == pytest.approx(0.934)
        assert scene.get_parameter("depth_max_spread_m").value == 1.0
        assert scene.get_parameter("camera_extrinsics_file").value.startswith("src/")
        evidence = yaml.safe_load((CONFIG / "washer_pose_evidence.json").read_text())
        assert nav.get_parameter("washer_front_pose").value == pytest.approx(
            evidence["pose"]
        )
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()
