"""Bounded terminal alignment policy; no motor or ROS access."""
import math
import numpy as np
from .continuous import validate_feedback, continuous_goal, proximity_weight

MAX_ENDPOINT_CORRECTIONS = 2
ENDPOINT_BUDGET_S = 25.
PENDING_WAIT_S = 4.


def latency_speed_limit(target, robot_xy, feedback, requested=.10):
    base=min(requested,.10-.06*proximity_weight(target,robot_xy))
    if feedback is None:return base
    latency=float(feedback.get('timings',{}).get('processing_seconds',0.))
    if not math.isfinite(latency) or latency<0:return base
    latency=min(latency,6.)
    center=np.asarray(feedback['object_map'],float)[:2]
    normal=np.asarray(feedback['surface_normal_map'],float)[:2]
    normal/=np.linalg.norm(normal)
    remaining=float((np.asarray(robot_xy)-center)@normal)-target['standoff']
    # Keep at least one inference duration of travel before the endpoint.
    return min(base,max(.015,max(0.,remaining-.35)/(latency+1.)))


def endpoint_correction(feedback,target,base_in_map,now):
    center=validate_feedback(feedback,target,now)
    normal=np.asarray(feedback['surface_normal_map'],float).copy()
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal[:2])<.9:
        raise RuntimeError('Invalid endpoint normal')
    normal[2]=0.;normal/=np.linalg.norm(normal)
    if normal@np.asarray(target['surface_normal_map'])<math.cos(math.radians(25)):
        raise RuntimeError('Endpoint normal changed beyond limit')
    stand=target.get('standoff',float(np.linalg.norm(np.asarray(target['goal_map'])[:2]-np.asarray(target['board_map'])[:2])))
    goal=continuous_goal(dict(target,board_map=center.tolist(),surface_normal_map=normal.tolist()),stand)
    point=np.array(goal['goal_map']); transform=np.asarray(base_in_map)
    delta=transform[:3,:3].T@(point-transform[:3,3])
    yaw=math.atan2(transform[1,0],transform[0,0])
    turn=math.atan2(math.sin(goal['goal_yaw']-yaw),math.cos(goal['goal_yaw']-yaw))
    if (np.linalg.norm(point[:2]-np.asarray(target['goal_map'])[:2])>.35
            or np.linalg.norm(delta[:2])>.20 or delta[0]<-.015
            or abs(delta[1])>.05 or abs(turn)>math.radians(15)):
        raise RuntimeError('Endpoint correction needs more maneuvering space; stopping')
    return goal


class ApproachObservationGate:
    """Reserve forward maneuvering room before committing the last approach."""
    def __init__(self):
        self.started = None
        self.reserve = None
        self.complete = False
        self.holding_since = None

    def update(self, target, robot_xy, feedback, now, speed=.10):
        if self.complete:
            return 'continue'
        center = np.asarray(target['board_map'],float)[:2]
        normal = np.asarray(target['surface_normal_map'],float)[:2]
        remaining = float((np.asarray(robot_xy)-center)@normal)-target['standoff']
        latency = float((feedback or {}).get('timings',{}).get('processing_seconds',6.))
        if not math.isfinite(latency) or latency < 0: latency=6.
        reserve = .35 + min(max(speed,0.),.10)*(min(latency,6.)+1.)
        if self.started is None and remaining <= reserve+.30:
            self.started,self.reserve=now,reserve
        if self.started is None:return 'continue'
        if feedback is not None and feedback['captured_monotonic'] >= self.started:
            validate_feedback(feedback,target,now)
            self.complete=True
            return 'refresh'
        if remaining <= self.reserve:
            if self.holding_since is None:self.holding_since=now
            if now-self.holding_since > 20.:
                raise RuntimeError('No fresh approach observation within reserved maneuvering space')
            return 'hold'
        return 'slow'


def approach_refresh_goal(feedback,target,base_in_map,now):
    center=validate_feedback(feedback,target,now)
    normal=np.asarray(feedback['surface_normal_map'],float).copy()
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal[:2])<.9:
        raise RuntimeError('Invalid approach normal')
    normal[2]=0.;normal/=np.linalg.norm(normal)
    if normal@np.asarray(target['surface_normal_map'])<math.cos(math.radians(25)):
        raise RuntimeError('Approach normal changed beyond limit')
    goal=continuous_goal(dict(target,board_map=center.tolist(),surface_normal_map=normal.tolist()),target['standoff'])
    point=np.asarray(goal['goal_map']);transform=np.asarray(base_in_map)
    delta=transform[:3,:3].T@(point-transform[:3,3])
    if np.linalg.norm(point[:2]-np.asarray(target['goal_map'])[:2])>.35 or delta[0]<.10 or abs(delta[1])>.30:
        raise RuntimeError('Fresh approach goal has insufficient forward maneuvering room')
    goal['approach_observation_refreshed']=True
    return goal
