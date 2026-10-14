import math
import numpy as np
import pytest
from robot_tasks.geometry import (
    front_goal,
    inspection_goals,
    depth_position,
    pose_errors,
    target_in_standoff_bounds,
    rotation,
)


def test_front_is_face_not_center():
    goal = front_goal((2, 0, 0.5), math.pi, 0.5, 0.434)
    assert goal == pytest.approx((1.066, 0, 0))
    assert pose_errors(goal, goal) == (0, 0)


def test_grasp_candidates_face_object():
    target = (2, 1, 0.3)
    for goal in inspection_goals(
        target, (0, 0, 0), 0.5, [0, math.pi / 6, -math.pi / 6]
    ):
        assert math.hypot(goal[0] - 2, goal[1] - 1) == pytest.approx(0.5)
        assert target_in_standoff_bounds(target, goal, [0.4, 0.6], 0.01, [0, 1])


def test_depth_rejects_missing_and_mixed():
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    with pytest.raises(ValueError, match="Insufficient"):
        depth_position(np.zeros((100, 100)), [10, 10, 90, 90], k)
    d = np.ones((100, 100))
    d[:, 50:] = 2
    with pytest.raises(ValueError, match="Mixed"):
        depth_position(d, [10, 10, 90, 90], k)


def test_depth_position_and_small_outliers():
    d = np.ones((100, 100)) * 2
    d[40:42, 40:42] = 3
    p, quality = depth_position(d, [10, 10, 90, 90], [100, 0, 50, 0, 100, 50, 0, 0, 1])
    assert p == pytest.approx([-0.01, -0.01, 2], abs=0.02)
    assert quality["valid_pixels"] > 1500


@pytest.mark.parametrize(
    "b", [[20, 10, 10, 20], [-1, 0, 10, 20], [0, 0, 200, 20], [0, 0, float("nan"), 20]]
)
def test_bad_bbox(b):
    with pytest.raises(ValueError):
        depth_position(np.ones((100, 100)), b, [100, 0, 50, 0, 100, 50, 0, 0, 1])


def test_quaternion_validation():
    with pytest.raises(ValueError):
        rotation([0, 0, 0, 0])
    assert np.allclose(rotation([0, 0, 0, 1]), np.eye(3))


def test_inspection_clearance_is_measured_from_physical_front():
    from robot_tasks.geometry import inspection_base_distance

    extent = 0.434
    clearance = 0.5
    distance = inspection_base_distance(extent, clearance)
    assert distance == pytest.approx(0.934)
    target = (2.0, 1.0, 0.3)
    for x, y, yaw in inspection_goals(
        target, (0.0, 0.0, 0.0), distance, [0.0, math.pi / 6, -math.pi / 6]
    ):
        front = (x + extent * math.cos(yaw), y + extent * math.sin(yaw))
        assert math.hypot(target[0] - front[0], target[1] - front[1]) == pytest.approx(
            clearance
        )
    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            inspection_base_distance(extent, value)
