"""Online occupancy: optimized historical anchors plus aligned LIO tail.

Archive observations separately from occupancy; rebuild on changed corrections,
publish map and map->odom from the same completed generation.
"""
import json,time,copy,hashlib
from array import array
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry,Path as PosePath,OccupancyGrid
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from .cloud_filter import outside_body
from .archive import ObservationArchive
from .occupancy import VoxelMap,transform,correction_at,corrections_changed


def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

def pose_matrix(p):return transform([p.position.x,p.position.y,p.position.z],[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])


class ExplorationMapper(Node):
    def __init__(self):
        super().__init__('exploration_mapper')
        for key,value in [('output','/tmp/exploration_occupancy'),('bounds',[-25.,-25.,25.,25.]),('resolution',.05),('sensor_origin',[.27254,.00035,.095])]:self.declare_parameter(key,value)
        self.out=Path(self.get_parameter('output').value);self.out.mkdir(parents=True,exist_ok=True)
        self.bounds=self.get_parameter('bounds').value;self.res=self.get_parameter('resolution').value
        self.sensor=np.asarray(self.get_parameter('sensor_origin').value)
        self.records=[];self.odoms=deque(maxlen=200);self.clouds=deque(maxlen=20);self.anchors=[]
        self.anchor_digest='';self.epoch=0;self.active_epoch=-1;self.integrated=0;self.rebuilds=0
        self.last_observation=0.;self.last_receive=0.;self.last_snapshot=0.;self.voxels=None;self.correction=np.eye(4)
        self.status_writer=ThreadPoolExecutor(max_workers=1);self.status_write=None
        self.archive=ObservationArchive()
        self.preprocessing=ThreadPoolExecutor(max_workers=1);self.prepared=None;self.last_selected=0.
        self.max_prepare_seconds=0.;self.max_update_gap=0.;self.last_update=None
        self.pool=ThreadPoolExecutor(max_workers=1);self.work=None;self.msg=None;self.error=None;self.published_correction=None
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub=self.create_publisher(OccupancyGrid,'/explore/full_map',qos)
        self.status_pub=self.create_publisher(String,'/explore/map_status',qos)
        self.correction_pub=self.create_publisher(TransformStamped,'/explore/map_correction',qos)
        self.create_subscription(Odometry,'/fastlio/odom_measurement',self.odom,50)
        self.create_subscription(PointCloud2,'/fastlio/cloud_body',self.cloud,qos_profile_sensor_data)
        self.create_subscription(PosePath,'/sc_pgo/pose_snapshot',self.snapshot,qos)
        self.create_timer(.05,self.match);self.create_timer(.5,self.update)
    def odom(self,msg):
        if msg.header.frame_id!='camera_init' or msg.child_frame_id!='body':self.error='unexpected_lio_frames';return
        self.odoms.append((stamp(msg),pose_matrix(msg.pose.pose)))
    def cloud(self,msg):
        if msg.header.frame_id!='body':self.error='unexpected_cloud_frame';return
        self.clouds.append(msg)
    def snapshot(self,msg):
        if msg.header.frame_id!='camera_init' or len(msg.poses)%2:self.error='invalid_pgo_snapshot';return
        self.last_snapshot=time.monotonic();anchors=[]
        for raw,opt in zip(msg.poses[::2],msg.poses[1::2]):
            t=stamp(raw)
            if abs(t-stamp(opt))>1e-6 or (anchors and t<=anchors[-1][0]):self.error='invalid_pgo_timestamps';return
            anchors.append((t,pose_matrix(opt.pose)@np.linalg.inv(pose_matrix(raw.pose))))
        if not anchors:return
        packed=np.array([[t,*c.ravel()] for t,c in anchors]);digest=hashlib.sha256(np.round(packed,5).tobytes()).hexdigest()
        if digest!=self.anchor_digest:
            changed=corrections_changed(self.anchors,anchors)
            self.anchors=anchors;self.anchor_digest=digest
            if changed:self.epoch+=1
            if not self.archive.submit(self.out/'anchors.npz',times=packed[:,0],corrections=np.array([c for _,c in anchors])):
                self.error=self.archive.error
    def prepare(self,msg,t,raw):
        started=time.monotonic()
        pts=point_cloud2.read_points(msg,field_names=('x','y','z'),skip_nans=True)
        if isinstance(pts,np.ndarray) and pts.dtype.names:pts=np.column_stack([pts[k] for k in ('x','y','z')])
        else:pts=np.asarray(list(pts))
        pts=np.asarray(pts,dtype=np.float32).reshape(-1,3)
        pts=outside_body(pts)
        _,idx=np.unique(np.floor(pts/self.res).astype(int),axis=0,return_index=True)
        return (t,raw,pts[idx]),time.monotonic()-started
    def match(self):
        if self.prepared is not None:
            if not self.prepared.done():return
            try:
                record,duration=self.prepared.result()
                self.max_prepare_seconds=max(self.max_prepare_seconds,duration)
                t,raw,pts=record
                # Reserve bounded storage before adding a record to the map.
                if self.archive.submit(self.out/f'observation_{len(self.records):06d}.npz',stamp=t,pose=raw,points=pts,sensor_origin=self.sensor):
                    self.records.append(record);self.last_observation=t
                else:self.error=self.archive.error
            except Exception as e:self.error='cloud_prepare:'+str(e)
            self.prepared=None
        if self.error:return
        if not self.odoms:return
        candidate=None
        # Coalesce a receive burst to its newest matched cloud. Never decode a
        # backlog inside this timer: one bounded preprocessing job is in flight.
        while self.clouds:
            msg=self.clouds[0];t=stamp(msg)
            if t>self.odoms[-1][0]+.002:break
            self.clouds.popleft();nearest=min(self.odoms,key=lambda a:abs(a[0]-t))
            if abs(nearest[0]-t)>.005:continue
            self.last_receive=time.monotonic()
            if t-self.last_selected<.24:continue
            candidate=(msg,t,nearest[1].copy())
        if candidate is not None:
            self.last_selected=candidate[1]
            self.prepared=self.preprocessing.submit(self.prepare,*candidate)
    def build(self,records,anchors,epoch,existing,begin):
        vox=existing or VoxelMap(self.bounds,self.res)
        for t,raw,points in records[begin:]:
            pose=correction_at(t,anchors)@raw
            world=points@pose[:3,:3].T+pose[:3,3];origin=pose[:3,:3]@self.sensor+pose[:3,3]
            vox.integrate(world,origin)
        return vox,vox.grid(),epoch,len(records),anchors[-1][1].copy()
    def update(self):
        now=time.monotonic()
        if self.last_update is not None:self.max_update_gap=max(self.max_update_gap,now-self.last_update)
        self.last_update=now
        archive_error=self.archive.poll()
        if archive_error:self.error=archive_error
        if self.work and self.work.done():
            try:
                vox,grid,epoch,count,c=self.work.result()
                if epoch==self.epoch:
                    msg=OccupancyGrid();msg.header.frame_id='map';msg.info.resolution=float(self.res)
                    msg.info.height,msg.info.width=grid.shape;msg.info.origin.position.x=float(self.bounds[0]);msg.info.origin.position.y=float(self.bounds[1]);msg.info.origin.orientation.w=1.
                    # Humble's list setter validates every cell in Python twice.
                    # int8 storage preserves unknown=-1 without that long callback.
                    msg.data=array('b',np.asarray(grid,dtype=np.int8).tobytes())
                    self.voxels=vox;self.active_epoch=epoch;self.integrated=count;self.correction=c;self.msg=msg
            except Exception as e:self.error='map_build:'+str(e)
            self.work=None
        if not self.work and self.anchors and self.records and (self.epoch!=self.active_epoch or len(self.records)>self.integrated):
            rebuild=self.epoch!=self.active_epoch
            if rebuild:self.rebuilds+=1
            self.work=self.pool.submit(self.build,list(self.records),list(self.anchors),self.epoch,None if rebuild else self.voxels,0 if rebuild else self.integrated)
        integration_lag=(self.records[-1][0]-self.records[self.integrated-1][0]) if self.integrated else float('inf')
        ready=bool(integration_lag<2. and self.msg is not None and self.active_epoch==self.epoch and not self.error and time.monotonic()-self.last_receive<1. and time.monotonic()-self.last_snapshot<5.)
        if ready:
            self.msg.header.stamp=self.get_clock().now().to_msg();self.map_pub.publish(self.msg)
            self.published_correction=self.correction.copy()
        # Only the correction of the committed map is renewed, including while
        # rebuilding its replacement. A separate process handles periodic TF.
        self.publish_correction()
        status=dict(ready=ready,error=self.error,epoch=self.epoch,active_epoch=self.active_epoch,rebuilding=self.active_epoch!=self.epoch,
                    observations=len(self.records),integrated=self.integrated,integration_lag=integration_lag if np.isfinite(integration_lag) else None,rebuilds=self.rebuilds,anchors=len(self.anchors),
                    archive_pending=len(self.archive.pending),archive_written=self.archive.written,archive_error=self.archive.error,
                    max_archive_write_seconds=self.archive.max_write_seconds,max_prepare_seconds=self.max_prepare_seconds,max_update_gap=self.max_update_gap)
        self.status_pub.publish(String(data=json.dumps(status)))
        self.archive_status(status)
    def archive_status(self,status):
        # Diagnostic storage must never stall map/TF callbacks. Keep at most one
        # write in flight; the next update supplies a fresh snapshot, no backlog.
        if self.status_write is not None:
            if not self.status_write.done():return
            try:self.status_write.result()
            except Exception as e:self.get_logger().warning(f'Status archive failed: {e}')
        self.status_write=self.status_writer.submit(self.write_status,json.dumps(status,indent=2))
    def write_status(self,text):
        temp=self.out/'status.json.tmp';temp.write_text(text);temp.replace(self.out/'status.json')
    def destroy_node(self):
        self.preprocessing.shutdown(wait=True)
        self.archive.close()
        if self.archive.error:self.get_logger().error(self.archive.error)
        self.status_writer.shutdown(wait=True)
        return super().destroy_node()
    def publish_correction(self):
        if self.published_correction is None:return
        t=TransformStamped();t.header.stamp=self.get_clock().now().to_msg();t.header.frame_id='map';t.child_frame_id='odom'
        c=self.published_correction;t.transform.translation.x=float(c[0,3]);t.transform.translation.y=float(c[1,3]);t.transform.translation.z=float(c[2,3])
        q=Rotation.from_matrix(c[:3,:3]).as_quat();t.transform.rotation.x=float(q[0]);t.transform.rotation.y=float(q[1]);t.transform.rotation.z=float(q[2]);t.transform.rotation.w=float(q[3]);self.correction_pub.publish(t)


def main():
    rclpy.init();n=ExplorationMapper()
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    finally:n.pool.shutdown(wait=True);n.destroy_node();rclpy.try_shutdown()
