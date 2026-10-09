"""Short collision-checked waypoint geometry. No stored object locations."""
import math
import numpy as np
from .segmented import validate_straight_path


def short_waypoint(points, start, goal, board, max_step=.70):
    """Take only a straight prefix of the collision-checked planner path."""
    start = np.asarray(start, float)
    points = np.asarray(points, float)
    if np.linalg.norm(np.asarray(goal['goal_map'][:2])-start) <= .04:
        return dict(goal)
    selected = None
    prefix = [start]
    for point in points:
        delta = point-prefix[-1]
        length = np.linalg.norm(delta)
        # Resample to avoid skipping a short path with few planner points.
        for weight in np.linspace(0, 1, max(2, int(length/.025)+1))[1:]:
            candidate = prefix[-1]+weight*delta
            trial = np.array(prefix+[candidate])
            if np.linalg.norm(candidate-start) > max_step:
                break
            try:
                validate_straight_path(trial, start, candidate)
            except RuntimeError:
                break
            if np.linalg.norm(candidate-start) >= .10:
                selected = candidate.copy()
        else:
            prefix.append(point)
            continue
        break
    if selected is None:
        # A fine adjustment shorter than 10 cm is allowed if the entire path is straight.
        end = np.asarray(goal['goal_map'][:2])
        if np.linalg.norm(end-start) <= .10:
            validate_straight_path(points, start, end)
            return dict(goal)
        raise RuntimeError('No safe straight path prefix')
    final = np.linalg.norm(selected-np.asarray(goal['goal_map'][:2])) < .04
    result = dict(goal)
    result['goal_map'] = [float(selected[0]), float(selected[1]), 0.]
    if not final:
        result['goal_yaw'] = math.atan2(board[1]-selected[1], board[0]-selected[0])
    return result
