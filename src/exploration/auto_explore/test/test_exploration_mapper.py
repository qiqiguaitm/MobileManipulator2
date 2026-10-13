"""Backend correction must replace occupancy and TF as one generation."""
import time
from concurrent.futures import Future
import numpy as np
import rclpy
from auto_explore.exploration_mapper import ExplorationMapper


def test_backend_rebuild_replaces_map_and_transform(tmp_path):
    rclpy.init(args=['--ros-args','-p',f'output:={tmp_path}'])
    n=ExplorationMapper()
    try:
        assert n.res==.05
        n.bounds=[-1.,-1.,4.,3.];n.sensor=np.array([0.,0.,.5])
        n.records=[(1.,np.eye(4),np.array([[2.,0.,.5]]))]
        c=np.eye(4);n.anchors=[(1.,c)];n.epoch=1
        def finish(epoch,correction):
            f=Future();f.set_result(n.build(n.records,[(1.,correction)],epoch,None,0));n.work=f
            n.last_receive=n.last_snapshot=time.monotonic();n.update()
        finish(1,c)
        old=np.asarray(n.msg.data).reshape(n.msg.info.height,n.msg.info.width)
        assert old[20,60]==100
        shifted=c.copy();shifted[1,3]=1.;n.anchors=[(1.,shifted)];n.epoch=2
        finish(2,shifted)
        new=np.asarray(n.msg.data).reshape(n.msg.info.height,n.msg.info.width)
        assert new[20,60]==-1 and new[40,60]==100
        assert n.active_epoch==2 and n.correction[1,3]==1.
        # An obsolete worker result cannot roll TF/map back.
        finish(1,c)
        assert n.active_epoch==2 and n.correction[1,3]==1.
        # A ready=false replacement must not change the public TF ahead of the
        # occupancy map that consumers have actually received.
        replacement=np.eye(4);replacement[0,3]=9.
        n.epoch=3;n.anchors=[(1.,replacement)]
        n.work=Future();n.work.set_result(n.build(n.records,n.anchors,3,None,0))
        n.last_receive=0.;n.update()
        assert n.active_epoch==3 and n.correction[0,3]==9.
        assert n.published_correction[1,3]==1. and n.published_correction[0,3]==0.
    finally:
        n.pool.shutdown(wait=True);n.destroy_node();rclpy.try_shutdown()


def test_slow_status_disk_write_does_not_block_updates(tmp_path):
    from threading import Event
    rclpy.init(args=['--ros-args','-p',f'output:={tmp_path}'])
    n=ExplorationMapper();entered=Event();release=Event()
    def slow_write(text):entered.set();release.wait(3.)
    n.write_status=slow_write
    try:
        n.archive_status(dict(sequence=1));assert entered.wait(1.)
        pending=n.status_write
        start=time.monotonic();n.update();n.update()
        assert time.monotonic()-start<.3
        assert n.status_write is pending and not pending.done()
    finally:
        release.set();n.pool.shutdown(wait=True);n.destroy_node();rclpy.try_shutdown()


def test_slow_cloud_preparation_keeps_update_alive_and_coalesces_bursts(tmp_path):
    from threading import Event
    from types import SimpleNamespace
    rclpy.init(args=['--ros-args','-p',f'output:={tmp_path}'])
    n=ExplorationMapper();entered=Event();release=Event();selected=[]
    def cloud(t):
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=t,nanosec=0)))
    def slow_prepare(msg,t,raw):
        selected.append(t);entered.set();assert release.wait(3.)
        return (t,raw,np.array([[1.,0.,.5]],dtype=np.float32)),.1
    n.prepare=slow_prepare
    try:
        for t in (1,2,3):n.odoms.append((float(t),np.eye(4)));n.clouds.append(cloud(t))
        n.match();assert entered.wait(1.)
        assert selected==[3.]
        for t in (4,5,6):n.odoms.append((float(t),np.eye(4)));n.clouds.append(cloud(t))
        pending=n.prepared
        start=time.monotonic();n.match();n.update();n.update()
        assert time.monotonic()-start<.3
        assert n.prepared is pending and len(n.records)==0
        release.set();pending.result(timeout=1.)
        n.match();n.prepared.result(timeout=1.);n.match()
        assert selected==[3.,6.]
        assert [record[0] for record in n.records]==[3.,6.]
        assert n.error is None
    finally:
        release.set();n.pool.shutdown(wait=True);n.destroy_node();rclpy.try_shutdown()
    assert len(list(tmp_path.glob('observation_*.npz')))==2


def test_archive_failure_makes_map_not_ready(tmp_path):
    import json
    from types import SimpleNamespace
    rclpy.init(args=['--ros-args','-p',f'output:={tmp_path}'])
    n=ExplorationMapper();statuses=[]
    n.status_pub=SimpleNamespace(publish=lambda m:statuses.append(json.loads(m.data)))
    try:
        n.archive.error='archive_write:disk full';n.update()
        assert statuses[-1]['ready'] is False
        assert statuses[-1]['error']=='archive_write:disk full'
    finally:
        n.pool.shutdown(wait=True);n.destroy_node();rclpy.try_shutdown()
