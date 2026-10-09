import numpy as np
from board_approach.route import short_waypoint


def test_route_stops_before_a_corner_instead_of_cutting_it():
    goal = dict(goal_map=[.5, .5, 0], goal_yaw=0)
    path = np.r_[np.linspace([0, 0], [.5, 0], 20), np.linspace([.5, 0], [.5, .5], 20)]
    step = short_waypoint(path, [0, 0], goal, [2, 2, 0])
    assert step['goal_map'][0] <= .51
    assert step['goal_map'][1] < .10


def test_coarse_planner_two_point_path_is_subdivided():
    goal = dict(goal_map=[2, 0, 0], goal_yaw=0)
    step = short_waypoint([[0, 0], [2, 0]], [0, 0], goal, [3, 0, 0])
    assert .65 < step['goal_map'][0] <= .70


def test_tiny_adjustment_preserves_final_goal():
    goal = dict(goal_map=[.05, 0, 0], goal_yaw=.7)
    assert short_waypoint([[0, 0], [.05, 0]], [0, 0], goal, [2, 0, 0]) == goal


def test_coarse_approach_no_longer_stops_every_forty_centimeters():
    # Geometry reconstructed from the user's failed run, stage 0.
    start = np.array([1.6845, -.251])
    board = np.array([.5296, 2.4420])
    yaw = 1.733
    end = board-1.5*np.array([np.cos(yaw), np.sin(yaw)])
    goal = dict(goal_map=[*end, 0.], goal_yaw=yaw)
    path = np.linspace(start, end, 80)
    step = short_waypoint(path, start, goal, [*board, 0.], max_step=1.5)
    assert np.linalg.norm(np.array(step['goal_map'][:2])-start) > 1.4
    assert np.linalg.norm(np.array(step['goal_map'][:2])-end) < .05

