import json, math
from pathlib import Path
import pytest
from robot_tasks.semantic_mock import SemanticMapMock
from robot_tasks.geometry import front_goal


def test_measured_lookup_and_navigation_goal():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "config/washer_pose_evidence.json").read_text())
    provider = SemanticMapMock(data["pose"])
    frame, pose = provider.lookup_front_pose("洗衣机")
    assert frame == "map"
    assert data["sample_count"] == 12
    assert front_goal(pose[:3], pose[3], 0.5, 0.434) == pytest.approx(data["goal_map"])
    assert provider.lookup_front_pose("washer") == (frame, pose)
    with pytest.raises(KeyError):
        provider.lookup_front_pose("冰箱")


def test_invalid_record_is_rejected():
    with pytest.raises(ValueError):
        SemanticMapMock([0, 0, 0, float("nan")]).lookup_front_pose("洗衣机")
