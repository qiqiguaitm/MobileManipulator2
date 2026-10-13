import json,time
from types import SimpleNamespace as NS
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from auto_explore.explorer import Explorer

@pytest.fixture
def node(tmp_path):
    rclpy.init();n=Explorer(NS(output=str(tmp_path),execute=False,bounds=[-5,-5,5,5],max_seconds=300,max_distance=20))
    n.navigation_ready=lambda: True
    n.mapper_ready=lambda: True
    yield n
    n.destroy_node();rclpy.shutdown()

def test_preview_never_forwards_nonzero_command(node):
    outputs=[];node.output=NS(publish=outputs.append);node.enabled=True;node.command=Twist();node.command.linear.x=.1
    node.command_time=time.monotonic()
    node.gate()
    assert len(outputs)==1 and outputs[0].linear.x==0 and outputs[0].angular.z==0

def test_save_map_preserves_unknown_and_y_axis(node,tmp_path):
    m=OccupancyGrid();m.info.width=2;m.info.height=2;m.info.resolution=.1;m.info.origin.orientation.w=1.
    m.data=[0,100,-1,50];node.map=m
    node.cloud=point_cloud2.create_cloud_xyz32(Header(frame_id='camera_init'),[(1.,2.,3.),(4.,5.,6.)])
    node.status='partial';report=node.save()
    assert report['map_saved'] and report['cloud_saved']
    assert (tmp_path/'exploration_preview/map.pgm').read_bytes().endswith(bytes([205,205,254,0]))
    assert b'POINTS 2\nDATA binary\n' in (tmp_path/'exploration_preview/GlobalMap_sc_pgo.pcd').read_bytes()
    assert json.loads((tmp_path/'result.json').read_text())['status']=='partial'

def test_missing_map_not_saved_as_complete(node):
    node.status='fault';assert not node.save()['map_saved']


def test_startup_rear_patch_only_changes_published_navigation_grid(node):
    node.pose3=lambda:(0.,0.,0.)
    m=OccupancyGrid();m.info.width=80;m.info.height=80;m.info.resolution=.05
    m.info.origin.position.x=m.info.origin.position.y=-2.;m.info.origin.orientation.w=1.
    grid=np.full((80,80),-1,dtype=int);grid[:,40:]=0;m.data=grid.ravel().tolist()
    rear=40*80+30
    node.on_map(m)
    assert node.map.data[rear]==0 and node.raw_map.data[rear]==-1 and m.data[rear]==-1
    m.data[rear]=100;node.on_map(m);assert node.map.data[rear]==100
    m.data[rear]=-1;node.startup_clearance.started=time.monotonic()-91
    node.on_map(m)
    assert node.map.data[rear]==-1 and node.startup_clearance.finished


def test_bounds_mask_for_navigation_does_not_change_saved_map(node,tmp_path):
    m=OccupancyGrid();m.info.width=4;m.info.height=4;m.info.resolution=1.;m.info.origin.orientation.w=1.
    m.data=[0]*16;node.args.bounds=[0,0,2,2];node.on_map(m)
    assert node.map.data[3]==100 and node.raw_map.data[3]==0
    node.save();assert (tmp_path/'exploration_preview/map.pgm').read_bytes().endswith(bytes([254]*16))

def test_self_filter_preserves_obstacles_outside_body():
    from auto_explore.cloud_filter import outside_body
    points=np.array([[0,0,.2],[.35,.2,.3],[.40,0,.2],[0,.34,.2],[-.40,0,.2]])
    assert np.array_equal(outside_body(points),points[2:])


@pytest.mark.parametrize('seconds,distance,travel',[(1,0,0),(0,1,2)])
def test_explicit_test_limits_still_stop(node,seconds,distance,travel):
    node.args.max_seconds=seconds;node.args.max_distance=distance
    node.started=time.monotonic()-2;node.travel=travel
    node.coordinate()
    assert node.done and node.status=='limited'


def test_stale_input_holds_zero_then_times_out(node,monkeypatch):
    import auto_explore.runtime as module
    monkeypatch.setattr(module,'can_error',lambda *_:None)
    outputs=[];node.output=NS(publish=outputs.append)
    node.args.execute=True;node.enabled=True
    node.command.linear.x=.1;node.command_time=time.monotonic()
    node.input_problem=lambda:dict(source='scan',check='stamp_age',age=.8,limit=.7)
    node.gate()
    assert node.error is None and node.input_wait_since is not None
    assert outputs[-1].linear.x==0 and outputs[-1].angular.z==0
    assert node.command_time==0
    node.input_wait_since=time.monotonic()-10.1
    node.gate()
    assert node.error=='sensor_or_map_stale' and outputs[-1].linear.x==0


def test_input_recovery_discards_old_command(node):
    node.input_wait_since=time.monotonic()-1.
    node.command_time=time.monotonic();node.input_problem=lambda:None
    assert node.inputs_ready_for_motion(time.monotonic())
    assert node.input_wait_since is None and node.error is None
    assert node.command_time==0


def test_input_hold_pauses_and_resumes_without_ending(tmp_path):
    rclpy.init();n=Explorer(NS(output=str(tmp_path),execute=False,bounds=[-5,-5,5,5],max_seconds=0,max_distance=0))
    try:
        n.enabled=True;n.navigation_ready=lambda:True;n.mapper_ready=lambda:True
        n.fresh=lambda:False;n.input_wait_since=time.monotonic();sent=[]
        n.resume=NS(publish=lambda msg:sent.append(msg.data))
        n.coordinate()
        assert n.paused and not n.done and sent==[False]
        n.fresh=lambda:True;n.input_wait_since=None;n.coordinate()
        assert not n.paused and not n.done and sent==[False,True]
    finally:
        n.destroy_node();rclpy.shutdown()


@pytest.mark.parametrize('unknown',[False,True])
def test_upstream_return_completes_even_with_remaining_boundaries(node,unknown):
    grid=np.full((20,20),100,dtype=int);grid[2:18,2:18]=0
    if unknown:grid[:,18:]=-1
    node.map=OccupancyGrid();node.map.info.width=node.map.info.height=20
    node.map.info.resolution=.1;node.map.data=grid.ravel().tolist()
    node.explore_status(NS(status='exploration_complete'))
    assert not node.done  # Keep the existing automatic return before export.
    node.explore_status(NS(status='returning_to_origin'))
    assert node.returning and not node.done
    node.explore_status(NS(status='returned_to_origin'))
    assert node.done and node.status=='complete' and not node.enabled
    report=node.save()
    assert report['reason']=='exploration_finished_and_returned'
    assert (report['remaining_boundary_cells']>0)==unknown


def test_late_return_status_cannot_overwrite_fault(node):
    node.stop('fault','sensor_or_map_stale')
    node.explore_status(NS(status='returned_to_origin'))
    assert node.status=='fault' and node.final_reason=='sensor_or_map_stale'


def test_disabled_limits_do_not_stop_long_running_upstream(node):
    node.args.max_seconds=node.args.max_distance=0
    node.args.execute=True;node.enabled=True;node.travel=150.
    node.started=time.monotonic()-10000;node.last_progress=time.monotonic()
    node.fresh=lambda:True
    node.coordinate()
    assert not node.done and node.enabled


def test_motion_gate_still_clamps_checks_collision_and_holds_for_mapper(node,monkeypatch):
    import auto_explore.runtime as runtime
    monkeypatch.setattr(runtime,'can_error',lambda *_:None)
    node.args.execute=True;node.enabled=True;node.pose3=lambda:(0.,0.,0.)
    node.input_problem=lambda:None
    node.scan=NS(ranges=[3.]*30,angle_min=-1.5,angle_increment=.1,range_min=.2,range_max=8.)
    outputs=[];node.output=NS(publish=outputs.append)
    node.command.linear.x=.8;node.command.angular.z=.8
    node.command_time=time.monotonic();node.gate()
    assert outputs[-1].linear.x==.10 and outputs[-1].angular.z==.18
    node.scan=NS(ranges=[.39],angle_min=0.,angle_increment=.1,range_min=.2,range_max=8.)
    node.gate()
    assert outputs[-1].linear.x==0 and outputs[-1].angular.z==0
    node.scan.ranges=[3.];node.mapper_ready=lambda:False
    node.gate()
    assert outputs[-1].linear.x==0 and outputs[-1].angular.z==0
