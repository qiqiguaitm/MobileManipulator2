"""Moving RGB-D snapshots use capture-time poses; no ROS node or motor is started."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('rclpy')
from board_approach.ros_io import RobotIO
import board_approach.ros_io as ros_io


def fixture_node():
    header=NS(frame_id='camera',stamp=NS(sec=2,nanosec=0))
    rgb=NS(header=header,encoding='bgr8',height=2,width=2,step=6,data=bytes(12))
    depth=NS(header=header,encoding='16UC1',height=2,width=2,step=4,is_bigendian=False,
             data=np.full((2,2),2000,dtype=np.uint16).tobytes())
    pose=NS(header=header,pose=NS(pose=NS(position=NS(x=1.,y=0.,z=0.),
                                         orientation=NS(x=0.,y=0.,z=0.,w=1.))))
    current=np.eye(4);current[0,3]=1.05
    return NS(pump=lambda *a:None,health=lambda **k:None,
              data={'odom':NS(twist=NS(twist=NS(linear=NS(x=.05),angular=NS(z=0.)))),
                    'info':NS(header=header,k=np.eye(3).ravel().tolist())},
              images={'rgb':{2000000000:rgb},'depth':{2000000000:depth}},poses=[pose],
              last_snapshot_stamp=0,get_clock=lambda:NS(now=lambda:NS(nanoseconds=2000000000)),
              transform=lambda *a,**k:current)


def test_async_capture_uses_historical_pose_while_robot_moves():
    node=fixture_node()
    image,depth,k,pose,timestamp=RobotIO.snapshot(node,moving=True)
    assert pose[0,3]==1.  # Not the current TF position of 1.05.
    assert timestamp==2000000000 and np.all(depth==2.)
    with pytest.raises(RuntimeError,match='stationary'):
        RobotIO.snapshot(fixture_node())


def test_async_capture_refuses_poor_pose_synchronization():
    node=fixture_node()
    node.poses[0]=NS(header=NS(stamp=NS(sec=1,nanosec=850000000)))
    with pytest.raises(RuntimeError,match='No fresh RGB-D pair'):
        RobotIO.snapshot(node,moving=True)


def test_latest_image_waiting_for_pose_does_not_hide_older_matched_pair():
    node=fixture_node()
    # Latest image is 200ms ahead of the delayed localization stream.
    node.poses[0].header=NS(stamp=NS(sec=1,nanosec=800000000))
    node.images['rgb'][1800000000]=node.images['rgb'][2000000000]
    node.images['depth'][1800000000]=node.images['depth'][2000000000]
    _,_,_,pose,timestamp=RobotIO.snapshot(node,moving=True)
    assert timestamp==1800000000
    assert pose[0,3]==1.
    # Reusing the same image is forbidden; the newer one still has no matching pose.
    with pytest.raises(RuntimeError,match='No fresh RGB-D pair'):
        RobotIO.snapshot(node,moving=True)


def test_lidar_capture_does_not_depend_on_camera_depth_and_requires_scan(monkeypatch):
    node=fixture_node()
    node.require_lidar=True
    node.images['depth'].clear()
    node.data['info'].width=node.data['info'].height=2
    node.clouds=[NS(header=NS(stamp=NS(sec=2,nanosec=0)))]
    node.odometry=[NS(header=NS(stamp=NS(sec=1,nanosec=700000000))),
                   NS(header=NS(stamp=NS(sec=2,nanosec=200000000)))]
    saved=dict(metadata={'deskew':'test'},points_base=np.array([[3.,0.,.5]]))
    monkeypatch.setattr(ros_io,'capture_lidar',lambda n,t:saved)
    image,depth,k,pose,ts=RobotIO.snapshot(node,moving=True)
    assert depth is None and node.last_lidar_frame is saved
    node.last_snapshot_stamp=0
    node.clouds=[]
    with pytest.raises(RuntimeError,match='No fresh'):
        RobotIO.snapshot(node,moving=True)


def test_deskew_history_only_accepts_original_fastlio_subscription():
    from collections import deque
    node=NS(data={},received={},images={},odometry=deque())
    converted=object();original=object()
    RobotIO.receive(node,'odom',converted)
    assert not node.odometry
    RobotIO.receive(node,'deskew_odom',original)
    assert list(node.odometry)==[original]


@pytest.mark.parametrize('moving', [True,False])
def test_tf_comparison_uses_pose_timestamp_not_latest_or_image(moving):
    node=fixture_node()
    node.data['odom'].twist.twist.linear.x=0.
    node.poses[0].header=NS(stamp=NS(sec=1,nanosec=950000000))
    calls=[]
    def transform(parent,child,timestamp_ns=None):
        calls.append(timestamp_ns)
        tf=np.eye(4)
        if timestamp_ns is None:
            # A newer pose differs enough during motion to trigger the old bug.
            angle=.3;tf[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
            tf[0,3]=1.2
        else: tf[0,3]=1.
        return tf
    node.transform=transform
    _,_,_,pose,_=RobotIO.snapshot(node,moving=moving)
    assert calls==[1950000000] and pose[0,3]==1.


def test_missing_historical_tf_never_falls_back_to_latest():
    from tf2_ros import ExtrapolationException
    from board_approach.safety import InputFreshnessError
    node=fixture_node();calls=[]
    def transform(parent,child,timestamp_ns=None):
        calls.append(timestamp_ns)
        raise ExtrapolationException('history not available')
    node.transform=transform
    with pytest.raises(InputFreshnessError,match='TF unavailable at localization timestamp'):
        RobotIO.snapshot(node,moving=True)
    assert calls==[2000000000] and node.last_snapshot_stamp==0


@pytest.mark.parametrize('translation,angle',[(.11,0.),(0.,.13)])
def test_true_same_time_pose_disagreement_still_rejected(translation,angle):
    node=fixture_node();tf=np.eye(4);tf[0,3]=1.+translation
    tf[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    node.transform=lambda *a,**k:tf
    with pytest.raises(RuntimeError,match='disagree at matched timestamp'):
        RobotIO.snapshot(node,moving=True)
    assert node.last_snapshot_stamp==0


def test_transform_passes_explicit_nanoseconds_to_tf_buffer():
    from rclpy.clock import ClockType
    calls=[]
    def lookup(parent,child,query):
        calls.append((parent,child,query.nanoseconds))
        return NS(transform=NS(translation=NS(x=1.,y=0.,z=0.),rotation=NS(x=0.,y=0.,z=0.,w=1.)))
    node=NS(tf=NS(lookup_transform=lookup),get_clock=lambda:NS(clock_type=ClockType.ROS_TIME))
    matrix=RobotIO.transform(node,'map','base_link',timestamp_ns=1950000000)
    assert calls==[('map','base_link',1950000000)] and matrix[0,3]==1.
