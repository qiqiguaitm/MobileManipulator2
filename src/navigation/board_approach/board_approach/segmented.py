"""Forward-only point turns and a short, validated straight segment."""
import math
import numpy as np


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class SegmentedController:
    def __init__(self, start, goal, final_yaw, speed):
        self.start, self.goal = np.array(start, float), np.array(goal, float)
        delta = self.goal-self.start
        self.length = float(np.linalg.norm(delta))
        if self.length > 1.6:
            raise RuntimeError('Segment exceeds 1.6 m')
        self.direction = delta/max(self.length, 1e-9)
        self.bearing = math.atan2(delta[1], delta[0])
        self.final_yaw, self.speed = final_yaw, min(speed, .05)
        self.phase = 'turn_to_path' if self.length > .04 else 'turn_to_board'
        self.turn_anchor = self.start.copy()

    def command(self, xy, yaw):
        xy = np.asarray(xy)
        if self.phase == 'done':
            return 0., 0.
        if self.phase.startswith('turn'):
            if np.linalg.norm(xy-self.turn_anchor) > .08:
                raise RuntimeError('Robot translated during point turn')
            desired = self.bearing if self.phase == 'turn_to_path' else self.final_yaw
            error = angle(desired-yaw)
            if abs(error) <= .035:
                self.phase = 'forward' if self.phase == 'turn_to_path' else 'done'
                return 0., 0.
            return 0., float(np.clip(.8*error, -.10, .10))
        cross = abs(np.cross(self.direction, xy-self.start))
        if cross > .08:
            raise RuntimeError('Departed validated straight corridor')
        remaining = float((self.goal-xy) @ self.direction)
        distance = float(np.linalg.norm(self.goal-xy))
        if distance <= .04:
            self.phase, self.turn_anchor = 'turn_to_board', xy.copy()
            return 0., 0.
        if remaining <= 0:
            raise RuntimeError('Goal overshot; reversing is disabled')
        error = angle(math.atan2(*(self.goal-xy)[::-1])-yaw)
        return (min(self.speed, .5*remaining) if abs(error) < .12 else 0.,
                float(np.clip(.8*error, -.10, .10)))


def validate_straight_path(points, start, goal):
    start, goal, points = np.asarray(start), np.asarray(goal), np.asarray(points)
    delta = goal-start
    length = np.linalg.norm(delta)
    if length <= .04:
        return
    direction = delta/length
    along = (points-start) @ direction
    cross = np.abs(np.cross(direction, points-start))
    if (cross.max() > .06 or along.min() < -.08 or along.max() > length+.08
            or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > length+.15):
        raise RuntimeError('Planner requires a detour; straight execution refused')
