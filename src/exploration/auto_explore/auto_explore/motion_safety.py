"""Live scan collision checks and current-body unknown-cell clearing."""
import math
import numpy as np
from .geometry import HALF_LENGTH, HALF_WIDTH, PLANNING_PADDING


def collision_risk(points, vx, wz):
    """Short swept footprint. This holds motion; it does not abort exploration."""
    points=np.asarray(points)
    if not points.size:return False
    for t in np.linspace(0,1.,11):
        theta=wz*t
        dx=vx*t if abs(wz)<1e-6 else vx/wz*math.sin(theta)
        dy=0 if abs(wz)<1e-6 else vx/wz*(1-math.cos(theta))
        x=(points[:,0]-dx)*math.cos(theta)+(points[:,1]-dy)*math.sin(theta)
        y=-(points[:,0]-dx)*math.sin(theta)+(points[:,1]-dy)*math.cos(theta)
        if np.any((np.abs(x)<HALF_LENGTH+PLANNING_PADDING)&(np.abs(y)<HALF_WIDTH+PLANNING_PADDING)):return True
    return False


def footprint_cells(shape, resolution, origin, pose, padding=0.):
    """Conservative cell intersection with the physical base rectangle."""
    yy,xx=np.indices(shape)
    dx=origin[0]+(xx+.5)*resolution-pose[0]
    dy=origin[1]+(yy+.5)*resolution-pose[1]
    c,s=math.cos(pose[2]),math.sin(pose[2]);pad=resolution*math.sqrt(.5)+padding
    x=dx*c+dy*s;y=-dx*s+dy*c
    return (np.abs(x)<=HALF_LENGTH+pad)&(np.abs(y)<=HALF_WIDTH+pad)


def clear_body_unknown(grid, resolution, origin, pose):
    """Only current body cells; never erase measured obstacles or clear ahead."""
    result=np.array(grid,copy=True)
    # Only nearby cells can overlap the body. Keep the same intersection test
    # but avoid constructing full-map coordinate arrays on every callback.
    pad=resolution*math.sqrt(.5)
    radius=math.hypot(HALF_LENGTH+pad,HALF_WIDTH+pad)
    x0=max(0,math.floor((pose[0]-radius-origin[0])/resolution))
    y0=max(0,math.floor((pose[1]-radius-origin[1])/resolution))
    x1=min(result.shape[1],math.ceil((pose[0]+radius-origin[0])/resolution)+1)
    y1=min(result.shape[0],math.ceil((pose[1]+radius-origin[1])/resolution)+1)
    if x0>=x1 or y0>=y1:return result
    nearby=result[y0:y1,x0:x1]
    local_origin=(origin[0]+x0*resolution,origin[1]+y0*resolution)
    nearby[(nearby<0)&footprint_cells(nearby.shape,resolution,local_origin,pose)]=0
    return result


