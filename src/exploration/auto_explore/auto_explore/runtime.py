"""Shared sensor state, exclusive velocity gate and diagnostic map persistence.

Frontier selection and navigation goals are owned solely by explore_lite.
"""
from array import array
import copy
import json, math, socket, struct, time
from pathlib import Path
import numpy as np
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from rcl_interfaces.srv import GetParameters
from lifecycle_msgs.srv import GetState
from tf2_ros import Buffer, TransformListener
from board_approach.safety import can_error
from .startup_clearance import StartupRearClearance
from .motion_safety import collision_risk, clear_body_unknown


def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9


class ExplorationRuntime(Node):
    def __init__(self,args):
        super().__init__('auto_explorer');self.args=args;self.out=Path(args.output);self.out.mkdir(parents=True,exist_ok=True)
        self.last_gate_report=0.;self.tf_wait=None
        self.input_wait_since=None;self.input_recovery_timeout=10.
        self.started=time.monotonic();self.last_progress=self.started;self.known=0
        self.map=None;self.raw_map=None;self.scan=None;self.cloud=None;self.odom=None;self.received={};self.travel=0.;self.last_xy=None
        self.startup_clearance=StartupRearClearance(getattr(args,'startup_rear_clearance',.20))
        self.command=Twist();self.command_time=0.;self.enabled=False;self.done=False;self.error=None
        self.can_state=None;self.can_time=0.;self.can=None;self.last_tick=time.monotonic();self.watchdog_ok=not args.execute
        self.status='starting';self.home=None
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.map_output=self.create_publisher(OccupancyGrid,'/map',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.startup_clearance_output=self.create_publisher(Marker,'/explore/startup_clearance',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(OccupancyGrid,'/explore/full_map',self.on_map,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(LaserScan,'/explore/scan',self.on_scan,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odom',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'/aft_pgo_map',self.on_cloud,qos_profile_sensor_data)
        self.create_subscription(Twist,'/explore/nav_cmd_vel',self.on_command,10)
        self.output=self.create_publisher(Twist,'/explore/safe_cmd_vel',10)
        self.nav_states={name:dict(client=self.create_client(GetState,'/'+name+'/get_state'),future=None,active=False,stamp=0.) for name in ('planner_server','controller_server','bt_navigator')}
        self.watchdog=self.create_client(GetParameters,'/tracer/get_parameters');self.watchdog_future=None
        if args.execute:
            self.can=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW);self.can.bind(('can2',));self.can.setblocking(False)
        self.create_timer(.05,self.gate);self.create_timer(1.,self.coordinate)
    def event(self,kind,**values):
        record=dict(event=kind,elapsed=round(time.monotonic()-self.started,2),**values)
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        self.get_logger().info(json.dumps(record))
    def on_map(self,m):
        self.raw_map=m
        if abs(m.info.origin.orientation.z)>1e-6 or abs(m.info.origin.orientation.w-1)>1e-6:
            self.error='rotated_grid_not_supported';return
        limited=copy.deepcopy(m)
        grid=np.asarray(m.data,dtype=np.int16).reshape(m.info.height,m.info.width).copy()
        x=m.info.origin.position.x+(np.arange(m.info.width)+.5)*m.info.resolution
        y=m.info.origin.position.y+(np.arange(m.info.height)+.5)*m.info.resolution
        self.startup_clearance.cells=[]
        try:
            pose=self.pose3();origin=(m.info.origin.position.x,m.info.origin.position.y)
            grid=clear_body_unknown(grid,m.info.resolution,origin,pose)
            grid=self.startup_clearance.apply(grid,m.info.resolution,origin,pose,self.travel,time.monotonic())
        except Exception:pass  # No fresh pose: preserve every unknown cell.
        b=self.args.bounds;grid[:,(x<b[0])|(x>b[2])]=100;grid[(y<b[1])|(y>b[3]),:]=100
        limited.data=array('b',grid.astype(np.int8).tobytes());self.map=limited;self.received['map']=time.monotonic();self.map_output.publish(limited)
        marker=Marker();marker.header=m.header;marker.ns='startup_assumed_rear_free';marker.id=0
        marker.type=Marker.CUBE_LIST;marker.pose.orientation.w=1.
        marker.scale.x=marker.scale.y=m.info.resolution;marker.scale.z=.015
        marker.color.r=1.;marker.color.g=.55;marker.color.a=.65
        marker.points=[Point(x=float(px),y=float(py),z=.045) for px,py in self.startup_clearance.cells if b[0]<=px<=b[2] and b[1]<=py<=b[3]]
        marker.action=Marker.ADD if marker.points else Marker.DELETE
        self.startup_clearance_output.publish(marker)
    def on_scan(self,m):self.scan=m;self.received['scan']=time.monotonic()
    def on_cloud(self,m):self.cloud=m
    def on_odom(self,m):
        p=m.pose.pose.position;xy=np.array([p.x,p.y])
        if self.last_xy is not None:
            delta=float(np.linalg.norm(xy-self.last_xy))
            if delta>.5:self.error='odometry_jump'
            elif delta>=.01:self.travel+=delta;self.last_xy=xy
        else:self.last_xy=xy
        self.odom=m;self.received['odom']=time.monotonic()
    def on_command(self,m):self.command=m;self.command_time=time.monotonic()
    def navigation_ready(self):
        now=time.monotonic()
        for state in self.nav_states.values():
            future=state['future']
            if future is not None and future.done():
                try:state['active']=future.result().current_state.id==3
                except Exception:state['active']=False
                state['stamp']=now;state['future']=None
            if state['future'] is None and state['client'].service_is_ready():
                state['future']=state['client'].call_async(GetState.Request())
        return all(s['active'] and now-s['stamp']<3. for s in self.nav_states.values())
    def pose(self):return self.pose3()[:2]
    def pose3(self):
        t=self.buffer.lookup_transform('map','base_link',Time())
        if self.get_clock().now().nanoseconds/1e9-(t.header.stamp.sec+t.header.stamp.nanosec*1e-9)>1.:
            raise RuntimeError('stale_map_transform')
        q=t.transform.rotation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        return (t.transform.translation.x,t.transform.translation.y,yaw)
    def input_problem(self):
        now=time.monotonic();rosnow=self.get_clock().now().nanoseconds/1e9
        for key,limit in (('scan',.7),('odom',.7),('map',8.)):
            age=now-self.received.get(key,0)
            if age>limit:return dict(source=key,check='receive_age',age=age,limit=limit)
        ranges=np.asarray(self.scan.ranges)
        if self.scan.header.frame_id != 'base_link':
            return dict(source='scan',check='frame',value=self.scan.header.frame_id)
        valid=int(np.count_nonzero(np.isfinite(ranges)&(ranges>=self.scan.range_min)&(ranges<=self.scan.range_max)))
        if valid<20:return dict(source='scan',check='valid_points',value=valid,minimum=20)
        for key,msg in (('scan',self.scan),('odom',self.odom)):
            age=rosnow-stamp(msg)
            if not -.2<=age<=.7:return dict(source=key,check='stamp_age',age=age,limit=.7)
        return None
    def fresh(self):return self.input_problem() is None
    def inputs_ready_for_motion(self,now):
        problem=self.input_problem()
        if problem is not None:
            if self.input_wait_since is None:
                self.input_wait_since=now
                self.event('input_hold',timeout=self.input_recovery_timeout,**problem)
            if now-self.input_wait_since>=self.input_recovery_timeout:
                if self.error is None:
                    self.event('input_timeout',waited=now-self.input_wait_since,**problem)
                    self.error='sensor_or_map_stale'
            # No old command may survive a sensor hold, including a short one.
            self.command_time=0.;self.tf_wait=None
            return False
        if self.input_wait_since is not None:
            self.event('input_recovered',waited=now-self.input_wait_since)
            self.input_wait_since=None;self.command_time=0.
        return True
    def gate(self):
        now=time.monotonic();delayed=now-self.last_tick>.25;self.last_tick=now;out=Twist()
        if self.can:
            for _ in range(2048):
                try:raw=self.can.recv(16)
                except BlockingIOError:break
                ident,n,data=struct.unpack('=IB3x8s',raw)
                if ident&0x7ff==0x211:self.can_state=list(data[:n]);self.can_time=now
        if self.args.execute and self.enabled and not self.done:
            if self.args.execute:
                reason=can_error(self.can_state,now-self.can_time)
                if reason:self.error=reason
            if not self.inputs_ready_for_motion(now):
                self.output.publish(out)
                return
            tf_ok=True
            try:
                xy=self.pose()
                b=self.args.bounds
                if not(b[0]<=xy[0]<=b[2] and b[1]<=xy[1]<=b[3]):self.error='outside_exploration_bounds'
                self.tf_wait=None
            except Exception as e:
                tf_ok=False
                if self.tf_wait is None:self.tf_wait=now;self.event('waiting_for_transform',detail=str(e))
                if now-self.tf_wait>2.:self.error='map_transform_unavailable'
            if not self.error and tf_ok and not delayed and now-self.command_time<.4:
                vx=float(self.command.linear.x);wz=float(self.command.angular.z)
                if not all(map(math.isfinite,(vx,wz))):self.error='invalid_velocity'
                else:
                    vx=max(0.,min(.10,vx));wz=max(-.18,min(.18,wz))
                    r=np.asarray(self.scan.ranges);a=self.scan.angle_min+np.arange(len(r))*self.scan.angle_increment
                    valid=np.isfinite(r)&(r>=self.scan.range_min)&(r<=self.scan.range_max)
                    points=np.c_[r[valid]*np.cos(a[valid]),r[valid]*np.sin(a[valid])]
                    # scan is explicitly generated in base_link, self points removed upstream.
                    scan_risk=collision_risk(points,vx,wz)
                    # Nav2 owns the costmap footprint checks; this gate adds a
                    # live scan sweep without repeating the retired map planner.
                    if now-self.last_gate_report>=2.:
                        self.event('velocity_check',requested=[vx,wz],scan_blocked=bool(scan_risk),travel=self.travel)
                        self.last_gate_report=now
                    if not scan_risk:
                        out.linear.x=vx;out.angular.z=wz
        if self.error:out=Twist()
        self.output.publish(out)
    def stop(self,status,reason):
        self.enabled=False;self.output.publish(Twist());self.status=status;self.done=True
        self.cancel_goal_async()
        self.event('finished',status=status,reason=reason,travel=self.travel)
        self.final_reason=reason
    def coordinate(self):
        raise NotImplementedError

    def cancel_goal_async(self):
        raise NotImplementedError

    def save(self):
        preview=self.out/'exploration_preview';preview.mkdir(exist_ok=True)
        report=dict(status=self.status,reason=getattr(self,'final_reason',self.error or 'interrupted'),travel=self.travel,
                    bounds=self.args.bounds,execute=self.args.execute,map_saved=False,cloud_saved=False,
                    max_seconds=self.args.max_seconds,max_distance=self.args.max_distance,home=self.home,
                    map_frame='map (SC-PGO optimized anchors + aligned LIO tail)',
                    cloud_frame='camera_init (SC-PGO; same global coordinates as map)')
        try:
            if self.map is not None:
                m=self.raw_map if self.raw_map is not None else self.map;a=np.asarray(m.data).reshape(m.info.height,m.info.width)
                pixels=np.full(a.shape,205,dtype=np.uint8);pixels[(a>=0)&(a<=25)]=254;pixels[a>=65]=0
                (preview/'map.pgm').write_bytes(f'P5\n{m.info.width} {m.info.height}\n255\n'.encode()+np.flipud(pixels).tobytes())
                (preview/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',mode='trinary',resolution=m.info.resolution,
                    origin=[m.info.origin.position.x,m.info.origin.position.y,0.],negate=0,occupied_thresh=.65,free_thresh=.25)))
                report['map_saved']=True
            if self.cloud is not None:
                from sensor_msgs_py import point_cloud2
                pts=np.array(list(point_cloud2.read_points(self.cloud,field_names=('x','y','z'),skip_nans=True)))
                if pts.dtype.names:pts=np.column_stack([pts[k] for k in ('x','y','z')])
                pts=np.asarray(pts,dtype='<f4').reshape(-1,3)
                header=f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA binary\n'
                (preview/'GlobalMap_sc_pgo.pcd').write_bytes(header.encode()+pts.tobytes());report['cloud_saved']=True
        except Exception as e:report['save_error']=type(e).__name__+': '+str(e)
        (self.out/'result.json').write_text(json.dumps(report,indent=2));return report

