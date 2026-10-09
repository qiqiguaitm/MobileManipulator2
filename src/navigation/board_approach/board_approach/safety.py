"""Conservative guard predicates, independently testable without a robot."""
import math
import numpy as np


class InputFreshnessError(RuntimeError):
    """No motion is allowed until new, sufficiently recent sensor data arrives."""


class FreshDataGate:
    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self.since = self.good_since = self.last_stamp = None
        self.fresh_frames = 0

    def update(self, now, healthy, scan_stamp):
        if self.since is not None and now-self.since >= self.timeout:
            raise RuntimeError('Sensor freshness did not recover within 2 seconds')
        if not healthy:
            if self.since is None:
                self.since = now
            self.good_since = self.last_stamp = None
            self.fresh_frames = 0
            return False
        if self.since is None:
            return True
        if self.good_since is None:
            self.good_since = now
        if self.last_stamp is None or scan_stamp > self.last_stamp:
            self.fresh_frames += 1
            self.last_stamp = scan_stamp
        if self.fresh_frames >= 3 and now-self.good_since >= .20:
            self.since = self.good_since = self.last_stamp = None
            self.fresh_frames = 0
            return True
        return False


def freshness_error(now, received, required, timeout=.65):
    for key in required:
        if key not in received or not 0 <= now-received[key] <= timeout:
            return key + ' stale or missing'
    return None


def can_error(state, age, require_can=True):
    raw = 'none' if state is None else ' '.join(f'{v:02x}' for v in state)
    detail = f'raw=[{raw}], age={age:.3f}s'
    if state is None or len(state) < 6 or not 0 <= age <= .4:
        return 'CAN feedback stale or incomplete; '+detail
    detail += f', vehicle=0x{state[0]:02x}, mode=0x{state[1]:02x}, fault_bytes=0x{state[4]:02x} 0x{state[5]:02x}'
    if state[0] != 0 or state[4] or state[5]:
        return 'Chassis emergency/error state abnormal; '+detail
    # Before driver initialization, accept standby/RC without granting motion.
    # UART and unknown modes may belong to another controller: do not take over.
    if state[1] not in ((1,) if require_can else (0, 1, 3)):
        return 'Chassis control mode not ready; '+detail
    return None


FORWARD_MIN_POINTS = 10
FORWARD_NEIGHBOR_RADIUS_M = .12
OBSTACLE_LINK_DISTANCE_M = .04

def obstacle_cluster_evidence(points, region):
    """Count local support only within a spatially connected component."""
    points=np.asarray(points)
    ids=np.flatnonzero(region)
    trigger=np.zeros(len(points),dtype=bool)
    max_neighbors=0
    sizes=[]
    if len(ids):
        xy=points[ids,:2]
        distances=np.sum((xy[:,None,:]-xy[None,:,:])**2,axis=2)
        links=distances<=OBSTACLE_LINK_DISTANCE_M**2
        unseen=np.ones(len(ids),bool)
        while unseen.any():
            seed=int(np.flatnonzero(unseen)[0])
            stack=[seed];unseen[seed]=False;component=[]
            while stack:
                current=stack.pop();component.append(current)
                adjacent=np.flatnonzero(links[current]&unseen)
                unseen[adjacent]=False;stack.extend(adjacent.tolist())
            sizes.append(len(component))
            component=np.asarray(component,int)
            for current in component:
                neighbors=component[distances[current,component]<=FORWARD_NEIGHBOR_RADIUS_M**2]
                max_neighbors=max(max_neighbors,len(neighbors))
                if len(neighbors)>=FORWARD_MIN_POINTS:
                    trigger[ids[neighbors]]=True
    return region,trigger,dict(min_points=FORWARD_MIN_POINTS,neighbor_radius_m=FORWARD_NEIGHBOR_RADIUS_M,
                              link_distance_m=OBSTACLE_LINK_DISTANCE_M,component_sizes=sorted(sizes,reverse=True),
                              region_points=len(ids),max_neighbor_points=max_neighbors)


def forward_bounds(target=None, robot_xy=None):
    bounds=dict(x_min=.15,x_max=.9,abs_y_max=.48)
    if not target or target.get('approach_phase') not in ('continuous_approach','precise_docking') or robot_xy is None:
        return bounds
    try:
        center=np.asarray(target['board_map'],float)
        normal=np.asarray(target['surface_normal_map'],float)
        xy=np.asarray(robot_xy,float)
        if center.shape!=(3,) or normal.shape!=(3,) or xy.shape!=(2,) or not np.isfinite(np.r_[center,normal,xy]).all():
            return bounds
        length=np.linalg.norm(normal[:2])
        if length<.9:return bounds
        distance=float((xy-center[:2])@(normal[:2]/length))
        if distance<=0:return bounds
        fraction=float(np.clip((1.5-distance)/.4,0.,1.))
        bounds.update(x_max=.9-.2*fraction,abs_y_max=.48-.10*fraction)
    except (ValueError,KeyError,TypeError):
        pass
    return bounds


def forward_obstacle_evidence(points, bounds=None):
    points=np.asarray(points)
    bounds=forward_bounds() if bounds is None else bounds
    region=(points[:,0]>bounds["x_min"])&(points[:,0]<bounds["x_max"])&(np.abs(points[:,1])<bounds["abs_y_max"])
    return obstacle_cluster_evidence(points,region)


def turning_obstacle_evidence(points):
    points=np.asarray(points)
    return obstacle_cluster_evidence(points,np.linalg.norm(points[:,:2],axis=1)<.65)


APPROACH_MAX_WZ = .08
APPROACH_MAX_AZ = .15


def smooth_steering(v, w, previous, dt):
    """Limit steering reversals; explicit zero commands always stop immediately."""
    if not all(math.isfinite(x) for x in (v,w,previous,dt)):
        raise RuntimeError('Non-finite steering state')
    if abs(v)<1e-9 and abs(w)<1e-9:
        return 0.
    desired=float(np.clip(w,-APPROACH_MAX_WZ,APPROACH_MAX_WZ))
    step=APPROACH_MAX_AZ*max(0.,min(dt,.1))
    return float(np.clip(desired,previous-step,previous+step))


def guarded_velocity(v, vy, w, points, speed, bounds=None):
    if not all(math.isfinite(x) for x in (v, vy, w)):
        raise RuntimeError('Non-finite velocity command')
    if v < -.005 or abs(vy) > .001:
        raise RuntimeError(f'Reverse/lateral motion is disabled: vx={v:.6f}, vy={vy:.6f}, wz={w:.6f}')
    if points is None or len(points) < 30:
        raise RuntimeError('Insufficient lidar returns')
    if v > 0 and forward_obstacle_evidence(points,bounds)[1].any():
        raise RuntimeError('Obstacle in forward stopping envelope')
    if abs(w) > .01 and turning_obstacle_evidence(points)[1].any():
        raise RuntimeError('Obstacle in turning clearance')
    scale = min(1., speed/max(abs(v), 1e-6), .15/max(abs(w), 1e-6))
    return max(0., v)*scale, w*scale
