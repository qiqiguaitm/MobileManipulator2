import math
import pytest
from robot_tasks.planning_debug import grid_cell, cell_cost, cell_world, cost_name


def test_rotated_map_origin_and_half_open_boundaries():
    grid = dict(
        width=3,
        height=2,
        resolution=0.05,
        origin=[-2.0, 4.0, math.pi / 2],
        data=[0, 1, 253, 254, 255, 12],
    )
    for y in range(2):
        for x in range(3):
            world = cell_world(grid, x, y)
            assert grid_cell(grid, *world) == (x, y)
            assert cell_cost(grid, *world) == grid["data"][y * 3 + x]
    assert grid_cell(grid, -1.0, 4.0) is None
    assert cell_cost(None, 0.0, 0.0) is None
    grid["origin"] = [0.0, 0.0, 0.0]
    grid["resolution"] = 1.0
    assert grid_cell(grid, 3.0, 0.0) is None
    assert grid_cell(grid, -0.001, 0.0) is None


def test_raw_costs_are_not_occupancy_percentages():
    assert cost_name(253) == "INSCRIBED"
    assert cost_name(254) == "OBSTACLE"
    assert cost_name(255) == "UNKNOWN"
    assert cost_name(100) == "INFLATED"
    assert cost_name(0) == "FREE"
