import json
import time
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest
from board_approach.continuous import stage_goal, validate_feedback, corrected_navigation_goal, VisualMonitor


def target():
    return dict(board_map=[3.,1.,0.],surface_normal_map=[-1.,0.,0.],goal_map=[2.,1.,0.],
                goal_yaw=0.,distance=3.,coarse_orientation=False)


def test_coarse_goal_keeps_space_on_normal_axis_and_fine_uses_precise_pose():
    rough=stage_goal(target(),1.,False)
    assert rough['goal_map']==[1.,1.,0.]
    assert rough['approach_phase']=='continuous_coarse'
    fine=stage_goal(target(),1.,True)
    assert fine['goal_map']==[2.,1.,0.]
    with pytest.raises(RuntimeError,match='not stable'):
        stage_goal(dict(target(),coarse_orientation=True),1.,True)
    with pytest.raises(RuntimeError,match='clearance'):
        stage_goal(dict(target(),distance=1.2),1.,False)


def test_old_or_changed_target_feedback_cannot_drive():
    proof=dict(captured_monotonic=10.,object_map=[3.,1.,0.])
    assert validate_feedback(proof,target(),11.) == pytest.approx([3.,1.,0.])
    for bad,now in ((proof,17.),(proof,9.),(dict(proof,object_map=[4.,1.,0.]),11.),
                    (dict(proof,object_map=[float('nan'),0,0]),11.)):
        with pytest.raises(RuntimeError): validate_feedback(bad,target(),now)


def test_updates_are_anchored_for_coarse_and_precise_goals():
    coarse=stage_goal(target(),1.,False)
    a=np.deg2rad(20)
    proof=dict(object_map=[3.,1.1,0.],surface_normal_map=[-np.cos(a),np.sin(a),0.])
    update=corrected_navigation_goal(proof,coarse,coarse)
    assert update is not None
    assert np.linalg.norm(np.array(update['goal_map'])-coarse['goal_map']) <= .35
    fine=stage_goal(target(),1.,True)
    a=np.deg2rad(10)
    proof=dict(proof,surface_normal_map=[-np.cos(a),np.sin(a),0.])
    precise_update=corrected_navigation_goal(proof,fine,fine)
    assert precise_update is not None
    assert np.linalg.norm(np.array(precise_update['goal_map'])-fine['goal_map']) <= .35
    assert corrected_navigation_goal(proof,fine,precise_update) is None
    assert corrected_navigation_goal(dict(proof,surface_normal_map=[0,1,0]),fine,fine) is None
    assert corrected_navigation_goal(dict(proof,surface_normal_map=[0,1,0]),coarse,coarse) is None


def test_lidar_visible_vertical_crop_can_change_but_horizontal_identity_cannot():
    t=dict(target(),geometry_source='lidar')
    proof=dict(captured_monotonic=10.,object_map=[3.,1.,.7])
    validate_feedback(proof,t,11.)
    with pytest.raises(RuntimeError,match='identity'):
        validate_feedback(dict(proof,object_map=[3.,1.3,.7]),t,11.)


def test_detector_runs_off_thread_without_queuing_frames(tmp_path):
    entered,release=threading.Event(),threading.Event()
    monitor=VisualMonitor.__new__(VisualMonitor)
    calls=[]
    class Node:
        def snapshot(self,moving=False):
            assert moving
            calls.append(1)
            return np.zeros((8,8,3)),np.ones((8,8)),np.eye(3),np.eye(4),1000000000
        def get_clock(self):
            return NS(now=lambda:NS(nanoseconds=1000000000))
    VisualMonitor.__init__(monitor,Node(),None,False,np.eye(4),target(),tmp_path)
    def infer(frame):
        entered.set()
        release.wait(2)
        return dict(captured_monotonic=time.monotonic(),object_map=[3.,1.,0.])
    monitor.infer=infer
    try:
        monitor.poll()
        assert entered.wait(1)
        monitor.next_capture=0
        for _ in range(10): monitor.poll()
        assert len(calls)==1 and not monitor.path.exists()
        release.set()
        monitor.future.result(timeout=2)
        monitor.next_capture=time.monotonic()+100
        monitor.poll()
        assert json.loads(monitor.path.read_text())['object_map']==[3.,1.,0.]
    finally:
        release.set()
        monitor.close()


def test_delayed_result_is_saved_but_not_published_or_fatal(tmp_path):
    from concurrent.futures import Future
    monitor=VisualMonitor(NS(),None,False,np.eye(4),target(),tmp_path)
    future=Future()
    future.set_result(dict(captured_monotonic=time.monotonic()-10,object_map=[3.,1.,0.]))
    monitor.future=future;monitor.next_capture=time.monotonic()+100
    try:
        monitor.poll()
        assert monitor.future is None and monitor.valid_updates==0 and not monitor.path.exists()
        saved=json.loads((monitor.directory/"startup_feedback.json").read_text())
        assert saved["captured_monotonic"]==future.result()["captured_monotonic"]
    finally:
        monitor.close()


@pytest.mark.parametrize('distance',[4.23,5.5,6.0,6.1])
def test_complete_async_lidar_pipeline_uses_stationary_surface_geometry(tmp_path,monkeypatch,distance):
    from test_lidar_geometry import scene
    from board_approach.surface import surface_target_geometry
    from board_approach.lidar_geometry import measure_lidar
    region,frame,k,transform=scene(angle=0,noise=0)
    origin=transform[:3,3]
    frame['points_base']=origin+(frame['points_base']-origin)*(distance/3.)
    def legacy_path(*args):
        raise AssertionError('LiDAR must not pass through centre-only depth-camera geometry')
    monkeypatch.setattr('board_approach.continuous.center_target_geometry',legacy_path)
    detector=NS(detect_region=lambda image:region,last_metadata=None)
    monitor=VisualMonitor(NS(),detector,False,transform,target(),tmp_path,geometry_source='lidar')
    try:
        report=monitor.infer((np.zeros((480,640,3),np.uint8),None,k,np.eye(4),1,time.monotonic(),0,frame))
        if distance<=6.:
            assert 'error' not in report,report.get('error')
            stationary=surface_target_geometry(measure_lidar(region,frame,k,transform),transform,np.eye(4),1.)
            assert report['object_map']==pytest.approx(stationary['object_map'])
            assert report['surface_normal_map']==pytest.approx(stationary['surface_normal_map'])
        else:
            assert 'depth=6.100' in report['error']
    finally:
        monitor.close()


def test_preflight_error_reports_underlying_failure_and_preserves_evidence(tmp_path):
    from concurrent.futures import Future
    monitor=VisualMonitor(NS(),None,False,np.eye(4),target(),tmp_path)
    monitor.future=Future();monitor.future.set_result({'error':'LiDAR has too few points: 8'})
    monitor.next_capture=time.monotonic()+100
    try:
        monitor.poll()
        assert 'too few points: 8' in monitor.failure_summary()
        status=json.loads((monitor.directory/'status.json').read_text())
        assert status['error_counts']=={'LiDAR has too few points: 8':1}
        assert not monitor.path.exists()
    finally:
        monitor.close()


def test_stopping_drains_pending_result_without_capturing_another_frame(tmp_path):
    from concurrent.futures import Future
    def forbidden(**kwargs):raise AssertionError('New inference started during motor cleanup')
    monitor=VisualMonitor(NS(snapshot=forbidden),None,False,np.eye(4),target(),tmp_path)
    future=Future();future.set_result(dict(captured_monotonic=time.monotonic(),object_map=[3.,1.,0.]))
    monitor.future=future;monitor.next_capture=0
    try:
        monitor.poll(capture=False)
        assert monitor.future is None and monitor.valid_updates==1
        assert monitor.path.exists()
        monitor.poll(capture=False)
    finally:
        monitor.close()


def test_fine_update_corrects_small_bias_without_relaxing_anchor():
    fine=stage_goal(target(),1.,True)
    proof=dict(object_map=[3.,1.02,0.],surface_normal_map=fine['surface_normal_map'])
    updated=corrected_navigation_goal(proof,fine,fine)
    assert updated is not None
    assert updated['goal_map'][1]-fine['goal_map'][1] == pytest.approx(.016)
    assert corrected_navigation_goal(proof,fine,updated) is None
    assert corrected_navigation_goal(dict(proof,object_map=[3.,2.,0.]),fine,fine) is None


@pytest.mark.parametrize('phase',['precise_docking','continuous_approach'])
def test_live_arrival_requires_fresh_geometry_and_current_pose(phase):
    from board_approach.continuous import fine_alignment_ready
    t=dict(approach_phase=phase,geometry_source='lidar',
           board_map=[1.,0.,0.],surface_normal_map=[-1.,0.,0.],goal_map=[0.,0.,0.])
    f=dict(captured_monotonic=10.,object_map=[1.,0.,0.],surface_normal_map=[-1.,0.,0.])
    assert fine_alignment_ready(f,t,np.eye(4),11.)
    assert fine_alignment_ready(f,t,np.eye(4),14.)
    assert not fine_alignment_ready(f,t,np.eye(4),17.)
    assert not fine_alignment_ready(None,t,np.eye(4),11.)
    assert not fine_alignment_ready(f,dict(t,approach_phase='continuous_coarse'),np.eye(4),11.)
    shifted=np.eye(4);shifted[1,3]=.085
    assert not fine_alignment_ready(f,t,shifted,11.)
    turned=np.eye(4); a=np.deg2rad(1.45)
    turned[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    assert not fine_alignment_ready(f,t,turned,11.)
    shifted=np.eye(4);shifted[0,3]=.15
    assert not fine_alignment_ready(f,t,shifted,11.)
    assert not fine_alignment_ready(dict(f,surface_normal_map=[0.,1.,0.]),t,np.eye(4),11.)
    with pytest.raises(RuntimeError,match='identity'):
        fine_alignment_ready(dict(f,object_map=[2.,0.,0.]),t,np.eye(4),11.)


@pytest.mark.parametrize('distance',[1.,1.5,3.,6.])
def test_continuous_goal_always_points_to_final_standoff(distance):
    from board_approach.continuous import continuous_goal, proximity_weight
    t=dict(target(),distance=distance)
    g=continuous_goal(t,.934)
    assert g['approach_phase']=='continuous_approach'
    assert np.linalg.norm(np.array(g['goal_map'])[:2]-np.array(t['board_map'])[:2])==pytest.approx(.934)
    n=np.array(t['surface_normal_map'])[:2]
    xy=np.array(t['board_map'])[:2]+distance*n
    expected=np.clip((2.5-distance)/1.3,0,1)
    assert proximity_weight(g,xy)==pytest.approx(expected)


def test_continuous_updates_remain_enabled_near_endpoint():
    from board_approach.continuous import continuous_goal, corrected_navigation_goal
    g=continuous_goal(target(),1.)
    f=dict(object_map=[3.,1.02,0.],surface_normal_map=g['surface_normal_map'])
    assert corrected_navigation_goal(f,g,g,[2.,1.]) is not None
    assert corrected_navigation_goal(f,g,g,[0.,1.]) is None
