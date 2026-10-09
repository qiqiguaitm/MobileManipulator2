import math
import numpy as np
import pytest
from board_approach.segmented import SegmentedController, validate_straight_path


def test_forward_only_complete_sequence():
    controller = SegmentedController([0, 0], [.7, -.15], 1.57, .05)
    xy, yaw = np.zeros(2), 2.2
    phases = set()
    for _ in range(4500):
        phases.add(controller.phase)
        v, w = controller.command(xy, yaw)
        assert 0 <= v <= .05
        assert abs(w) <= .1
        if controller.phase.startswith('turn'):
            assert v == 0
        xy += .02*v*np.array([math.cos(yaw), math.sin(yaw)])
        yaw += .02*w
        if controller.phase == 'done':
            break
    assert controller.phase == 'done'
    assert {'turn_to_path', 'forward', 'turn_to_board'} <= phases
    assert np.linalg.norm(xy-[.7, -.15]) <= .04
    assert abs(yaw-1.57) < .04


def test_point_turn_drift_stops():
    c = SegmentedController([0, 0], [1, 0], 0, .05)
    with pytest.raises(RuntimeError, match='translated'):
        c.command([.1, 0], 1)


def test_cross_track_stops():
    c = SegmentedController([0, 0], [1, 0], 0, .05)
    c.command([0, 0], 0)
    with pytest.raises(RuntimeError, match='corridor'):
        c.command([.3, .1], 0)


def test_overshoot_never_reverses():
    c = SegmentedController([0, 0], [1, 0], 0, .05)
    c.command([0, 0], 0)
    with pytest.raises(RuntimeError, match='overshot'):
        c.command([1.1, 0], 0)


def test_planner_detour_rejected():
    with pytest.raises(RuntimeError, match='detour'):
        validate_straight_path([[0, 0], [.5, .3], [1, 0]], [0, 0], [1, 0])
    validate_straight_path([[0, 0], [.5, .02], [1, 0]], [0, 0], [1, 0])


def test_long_coarse_segment_keeps_speed_limit_and_finishes():
    controller = SegmentedController([0, 0], [1.5, 0], 0, .05)
    xy = np.zeros(2)
    for _ in range(2400):
        v, w = controller.command(xy, 0.)
        assert 0 <= v <= .05
        xy[0] += .02*v
        if controller.phase == 'done':
            break
    assert controller.phase == 'done'
    assert 1.46 <= xy[0] <= 1.5
