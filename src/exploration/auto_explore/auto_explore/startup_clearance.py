"""A temporary, explicit rear free-space assumption for mapping startup."""
import math
import numpy as np
from .geometry import HALF_LENGTH, HALF_WIDTH, PLANNING_PADDING


class StartupRearClearance:
    def __init__(self, extension=.20, duration=90., departure=.60):
        if not math.isfinite(extension) or not 0 <= extension <= .25:
            raise ValueError('startup rear clearance must be between 0 and 0.25 m')
        self.extension=extension;self.duration=duration;self.departure=departure
        self.anchor=None;self.started=None;self.start_travel=None;self.finished=False
        self.cells=[]

    def apply(self, grid, resolution, origin, pose, travel, now):
        self.cells=[]
        if self.finished or self.extension==0:return grid
        if self.anchor is None:
            self.anchor=tuple(pose);self.started=now;self.start_travel=travel
        if (now-self.started>=self.duration or travel-self.start_travel>=self.departure
                or math.hypot(pose[0]-self.anchor[0],pose[1]-self.anchor[1])>=self.departure):
            self.finished=True
            return grid
        # A rear semicircle covering the initial padded footprint's turning
        # sweep. The distance beyond that rectangle is capped by extension.
        # This stays at the starting pose, never at the moving robot's tail.
        length=HALF_LENGTH+PLANNING_PADDING;width=HALF_WIDTH+PLANNING_PADDING
        cell_pad=resolution*math.sqrt(.5)
        radius=math.hypot(length+cell_pad,width+cell_pad)
        px,py,yaw=self.anchor
        x0=max(0,math.floor((px-radius-origin[0])/resolution))
        y0=max(0,math.floor((py-radius-origin[1])/resolution))
        x1=min(grid.shape[1],math.ceil((px+radius-origin[0])/resolution))
        y1=min(grid.shape[0],math.ceil((py+radius-origin[1])/resolution))
        if x0>=x1 or y0>=y1:return grid
        yy,xx=np.indices((y1-y0,x1-x0))
        dx=origin[0]+(xx+x0+.5)*resolution-px
        dy=origin[1]+(yy+y0+.5)*resolution-py
        c,s=math.cos(yaw),math.sin(yaw);x=c*dx+s*dy;y=-s*dx+c*dy
        beyond=np.hypot(np.maximum(np.abs(x)-length,0),np.maximum(np.abs(y)-width,0))
        mask=(x<0)&(x*x+y*y<=radius*radius)&(beyond<=self.extension)&(grid[y0:y1,x0:x1]<0)
        if not np.any(mask):return grid
        result=grid.copy();result[y0:y1,x0:x1][mask]=0
        self.cells=np.column_stack((origin[0]+(xx[mask]+x0+.5)*resolution,
                                    origin[1]+(yy[mask]+y0+.5)*resolution)).tolist()
        return result
