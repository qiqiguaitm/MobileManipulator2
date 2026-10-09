from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from board_approach.lidar_snapshot import decode_cloud, deskew
from board_approach.perception import DetectionError


def odom(t,x=0,yaw=0):
    q=Rotation.from_euler('z',yaw).as_quat()
    return NS(header=NS(stamp=NS(sec=int(t),nanosec=int(round((t-int(t))*1e9))),frame_id='odom'),
              child_frame_id='base_link',pose=NS(pose=NS(position=NS(x=x,y=0,z=0),
              orientation=NS(x=q[0],y=q[1],z=q[2],w=q[3]))))


def test_per_point_translation_and_rotation_deskew_recovers_stationary_object():
    times=np.linspace(10,10.2,21);image_time=10.1
    odometry=[odom(t,.05*(t-10),.15*(t-10)) for t in times]
    point_times=np.linspace(10.01,10.19,15)
    rot=Rotation.from_euler('z',.15*(point_times-10))
    stationary=np.tile([3.,.4,.8],(len(point_times),1))
    positions=np.c_[.05*(point_times-10),np.zeros((len(point_times),2))]
    raw=rot.inv().apply(stationary-positions)
    result,report=deskew(raw,point_times,image_time,odometry,np.eye(4))
    expected=Rotation.from_euler('z',.015).inv().apply(stationary-[.005,0,0])
    assert np.max(np.abs(result-expected))<1e-7
    assert report['deskew']=='per_point_odometry_SE3'


def test_missing_history_and_odometry_jumps_fail_closed():
    points=np.array([[3.,0.,.5]])
    with pytest.raises(DetectionError,match='bracket'):
        deskew(points,np.array([9.]),10.,[odom(10),odom(10.05)],np.eye(4))
    with pytest.raises(DetectionError,match='gap'):
        deskew(points,np.array([10.1]),10.2,[odom(10),odom(11)],np.eye(4))
    with pytest.raises(DetectionError,match='jump'):
        deskew(points,np.array([10.01]),10.04,[odom(10),odom(10.05,x=1.)],np.eye(4))


def test_rewritten_callback_timestamps_can_fake_a_speed_jump():
    points=np.array([[3.,0.,.5]])
    original=[odom(10,0.),odom(10.1,.005),odom(10.2,.010)]
    aligned,_=deskew(points,np.array([10.05]),10.1,original,np.eye(4))
    assert aligned[0,0]==pytest.approx(2.9975)
    # Same poses arriving in a burst: receipt timestamps imply a false .5m/s.
    retimed=[odom(10,0.),odom(10.19,.005),odom(10.2,.010)]
    with pytest.raises(DetectionError,match='excessive speed'):
        deskew(points,np.array([10.05]),10.1,retimed,np.eye(4))


def test_raw_body_pose_is_transformed_to_base_not_just_relabelled():
    ts=np.linspace(10,10.2,21)
    messages=[odom(t,.05*(t-10),.15*(t-10)) for t in ts]
    for m in messages:m.child_frame_id='body';m.header.frame_id='camera_init'
    b=np.eye(4);b[0,3]=.2
    pts_times=np.array([10.05,10.15]);world=np.tile([3.,.4,.8],(2,1))
    rotations=Rotation.from_euler('z',.15*(pts_times-10))
    origins=rotations.apply(np.tile([.2,0,0],(2,1)))+np.c_[.05*(pts_times-10),np.zeros((2,2))]
    measured=rotations.inv().apply(world-origins)
    out,_=deskew(measured,pts_times,10.1,messages,np.eye(4),b)
    r=Rotation.from_euler('z',.015);origin=r.apply([.2,0,0])+[.005,0,0]
    assert np.max(np.abs(out-r.inv().apply(world-origin)))<1e-7
    with pytest.raises(DetectionError,match='Missing'):
        deskew(measured,pts_times,10.1,messages,np.eye(4))


def test_cloud_padding_and_publisher_time_convention():
    dtype=np.dtype(dict(names=['x','y','z','time','ring'],formats=['<f4']*4+['<u2'],offsets=[0,4,8,16,20],itemsize=22))
    buf=bytearray(48*2)
    data=np.ndarray((2,2),dtype=dtype,buffer=buf,strides=(48,22))
    for k,v in [('x',3),('y',.2),('z',.5),('time',.12),('ring',2)]:data[k]=v
    fields=[NS(name=k,offset=off,datatype=typ,count=1) for k,off,typ in [('x',0,7),('y',4,7),('z',8,7),('time',16,7),('ring',20,4)]]
    msg=NS(fields=fields,point_step=22,row_step=48,width=2,height=2,data=buf,is_bigendian=False,header=NS(stamp=NS(sec=10,nanosec=0)))
    xyz,times,rings=decode_cloud(msg)
    assert xyz.shape==(4,3) and times==pytest.approx(np.full(4,10.02))
    assert (rings==2).all()
    data['time']=1.
    with pytest.raises(DetectionError,match='point times'):decode_cloud(msg)
