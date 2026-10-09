"""Exercise the real guard with fake sensors and publisher; never open CAN."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

pytest.importorskip('rclpy', reason='Source ROS Humble to run the guard integration tests')
from geometry_msgs.msg import Twist
from board_approach.motion import Motion, wait_for_start_feedback
from board_approach.safety import FreshDataGate, InputFreshnessError


class EmptyCAN:
    def recv(self, size):
        raise BlockingIOError


def fixture_node(monkeypatch):
    clock = {'now': 0., 'error': None}
    monkeypatch.setattr('board_approach.motion.time.monotonic', lambda: clock['now'])
    output = []
    scan = NS(header=NS(frame_id='laser', stamp=NS(sec=0, nanosec=0)),
              ranges=[3.]*100, range_min=.05, range_max=10.,
              angle_min=-np.pi, angle_increment=2*np.pi/100)
    cmd = Twist(); cmd.linear.x = .05
    def health():
        if clock['error']:
            raise clock['error']
    node = NS(can=EmptyCAN(), enabled=True, failure=None, segmented=True, active_goals=set(),
              health=health, can_state=[0, 1, 0, 0, 0, 0], can_time=0., paused=False,
              fresh_gate=FreshDataGate(), record=lambda event: None,
              data={'scan':scan, 'pose':NS(pose=NS(pose=NS(position=NS(x=0., y=0.))))},
              last_xy=np.zeros(2), start_xy=np.zeros(2), travel=0.,
              target={'scan_frame':'laser', 'base_from_scan':np.eye(4).tolist()},
              started=0., speed=.05, command=cmd, command_time=0.,
              output=NS(publish=lambda msg: output.append((msg.linear.x, msg.angular.z))))
    def tick(now, error=None, sequence=1):
        clock.update(now=now, error=error)
        node.can_time = now
        scan.header.stamp.nanosec = sequence
        Motion.guard(node)
    return node, output, tick


def test_real_guard_zeros_stale_commands_and_requires_a_new_command(monkeypatch):
    node, output, tick = fixture_node(monkeypatch)
    tick(0.)
    assert output[-1] == (.05, 0.)
    tick(.1, InputFreshnessError('scan stale'))
    assert node.paused and node.failure is None and output[-1] == (0., 0.)
    tick(.2, sequence=2)
    tick(.3, sequence=3)
    assert output[-1] == (0., 0.)
    tick(.41, sequence=4)
    assert not node.paused and output[-1] == (0., 0.)
    node.command = Twist(); node.command.linear.x = .05; node.command_time = .42
    tick(.42, sequence=5)
    assert output[-1] == (.05, 0.)


def test_real_guard_latches_persistent_loss(monkeypatch):
    node, output, tick = fixture_node(monkeypatch)
    tick(0., InputFreshnessError('scan stale'))
    tick(2.01, InputFreshnessError('scan stale'))
    assert 'recover' in node.failure
    tick(2.1, sequence=8)
    assert output[-1] == (0., 0.)


def test_localization_failure_does_not_auto_resume(monkeypatch):
    node, output, tick = fixture_node(monkeypatch)
    tick(0., RuntimeError('Localization unreliable'))
    assert node.failure == 'Localization unreliable'
    tick(.1)
    assert output[-1] == (0., 0.)


def test_guard_stops_even_if_parent_stalls_and_camera_feedback_expires(monkeypatch,tmp_path):
    import json
    node,output,tick=fixture_node(monkeypatch)
    node.target['board_map']=[3.,0.,0.]
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=0.,object_map=[3.,0.,0.])))
    tick(0.)
    assert output[-1]==(.05,0.)
    tick(6.1)
    assert 'expired' in node.failure
    assert output[-1]==(0.,0.)


def test_startup_wait_keeps_zero_until_a_new_feedback_arrives(monkeypatch,tmp_path):
    import json,time
    node,output,tick=fixture_node(monkeypatch)
    node.enabled=False;node.target['board_map']=[3.,0.,0.]
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=-30.,object_map=[3.,0.,0.])))
    tick(10.)
    def pump(seconds):
        now=time.monotonic()+seconds;tick(now)
        if now>10.25:
            node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=now,object_map=[3.,0.,0.])))
    node.pump=pump
    wait_for_start_feedback(node,timeout=1.)
    assert 10.25<time.monotonic()<11.
    assert not node.enabled and all(v==(0.,0.) for v in output)


def test_startup_refuses_to_enable_navigation_with_expired_feedback(monkeypatch,tmp_path):
    import json,time
    node,output,tick=fixture_node(monkeypatch)
    node.enabled=False;node.target['board_map']=[3.,0.,0.]
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=-40.,object_map=[3.,0.,0.])))
    node.pump=lambda seconds:tick(time.monotonic()+seconds)
    with pytest.raises(RuntimeError,match='after chassis initialization'):
        wait_for_start_feedback(node,timeout=.5)
    assert not node.enabled and all(v==(0.,0.) for v in output)


def test_controller_policy_and_guard_never_output_reverse(monkeypatch):
    from board_approach.motion import approach_controller_parameters
    policy = approach_controller_parameters(.05)
    assert 0 < policy['FollowPath.penalty_epsilon'] < policy['FollowPath.max_vel_x_backwards'] < .005
    assert policy['FollowPath.max_vel_y'] == 0.
    node, output, tick = fixture_node(monkeypatch)
    node.command.linear.x = -policy['FollowPath.max_vel_x_backwards']
    node.command.angular.z = .1
    tick(0.)
    assert node.failure is None and output[-1][0] == 0. and 0.<output[-1][1]<=.00301
    node.command.linear.x = -.1
    tick(.1)
    assert 'vx=-0.100000' in node.failure and output[-1] == (0., 0.)


def test_parameter_snapshot_survives_partial_rejection():
    from board_approach.motion import configure_approach_controller
    original = {'FollowPath.max_vel_x': .3, 'FollowPath.max_vel_x_backwards': .15}
    current = dict(original)
    def parameters(values=None, names=None):
        if values is None:
            return {k: current[k] for k in names}
        current['FollowPath.max_vel_x'] = values['FollowPath.max_vel_x']
        raise RuntimeError('Controller rejected parameters')
    saved = {}
    with pytest.raises(RuntimeError, match='rejected'):
        configure_approach_controller(NS(parameters=parameters),
            {k:.001 for k in original}, saved)
    assert saved == original
    assert current != original


def test_parameter_readback_detects_ignored_updates():
    from board_approach.motion import configure_approach_controller
    def parameters(values=None, names=None):
        return {'FollowPath.max_vel_x': .3} if values is None else None
    saved = {}
    with pytest.raises(RuntimeError, match='did not take effect'):
        configure_approach_controller(NS(parameters=parameters), {'FollowPath.max_vel_x':.05}, saved)
    assert saved == {'FollowPath.max_vel_x':.3}


def test_precise_loss_keeps_goal_through_endpoint(monkeypatch, tmp_path):
    import json
    node, output, tick = fixture_node(monkeypatch)
    node.target.update(board_map=[3.,0.,0.], goal_map=[2.,0.,0.], approach_phase='precise_docking')
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=-30.,object_map=[3.,0.,0.])))
    node.command_time=10.
    tick(10.)
    assert node.failure is None and output[-1]==(.05,0.)
    node.data['pose'].pose.pose.position.x=1.61
    tick(10.1)
    assert node.failure is None and output[-1]==(.05,0.)


def test_fresh_reacquisition_then_expiry_keeps_precise_goal(monkeypatch, tmp_path):
    import json
    node, output, tick = fixture_node(monkeypatch)
    node.target.update(board_map=[3.,0.,0.], goal_map=[2.,0.,0.], approach_phase='precise_docking')
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=10.,object_map=[3.,0.,0.])))
    node.command_time=10.;node.data['pose'].pose.pose.position.x=1.65
    tick(10.)
    assert node.failure is None and output[-1]==(.05,0.)
    node.current_goal=dict(node.target,goal_map=[3.,0.,0.])
    node.command_time=17.
    tick(17.)
    assert node.failure is None and output[-1]==(.05,0.)


def test_identity_change_stops_midleg_but_coarse_completion_allows_missing_feedback(monkeypatch,tmp_path):
    import json
    from board_approach.motion import motion_feedback
    node, output, tick=fixture_node(monkeypatch)
    node.target.update(board_map=[3.,0.,0.],goal_map=[2.,0.,0.],approach_phase='continuous_coarse')
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=0.,object_map=[4.,0.,0.])))
    tick(0.)
    assert 'identity changed' in node.failure and output[-1]==(0.,0.)
    node.visual_feedback.unlink()
    assert motion_feedback(node,force=True) is None


def test_can_transition_waits_without_enabling_motion(monkeypatch):
    from board_approach.motion import wait_for_can_mode
    clock=[0.];states=iter([3,3,1]);events=[]
    monkeypatch.setattr('board_approach.motion.time.monotonic',lambda:clock[0])
    node=NS(can_state=[0,3,0,0,0,0],can_time=0.,enabled=False,record=events.append)
    def pump(dt):
        clock[0]+=dt;node.can_time=clock[0];node.can_state[1]=next(states)
    node.pump=pump
    wait_for_can_mode(node,NS(poll=lambda:None))
    assert not node.enabled and events[-1]['raw_can_state'][1]==1


@pytest.mark.parametrize('fault,stale,exited', [(True,False,False),(False,True,False),
                                             (False,False,True),(False,False,False)])
def test_can_transition_fails_closed(monkeypatch,fault,stale,exited):
    from board_approach.motion import wait_for_can_mode
    clock=[0.]
    monkeypatch.setattr('board_approach.motion.time.monotonic',lambda:clock[0])
    node=NS(can_state=[0,3,0,0,0x80 if fault else 0,0],can_time=0.,enabled=False,record=lambda e:None)
    def pump(dt):
        clock[0]+=dt
        if not stale:node.can_time=clock[0]
    node.pump=pump
    with pytest.raises(RuntimeError):
        wait_for_can_mode(node,NS(poll=lambda:1 if exited else None),timeout=.6)
    assert not node.enabled


@pytest.mark.parametrize('case', ['delayed', 'timeout', 'duplicate', 'exit'])
def test_driver_discovery_is_bounded_and_records_cause(monkeypatch,case):
    from board_approach.motion import wait_for_driver_subscription
    clock=[0.];events=[]
    monkeypatch.setattr('board_approach.motion.time.monotonic',lambda:clock[0])
    def count():return 2 if case=='duplicate' else (1 if case=='delayed' and clock[0]>.2 else 0)
    node=NS(can_state=[0,1,0,0,0,0],can_time=0.,enabled=False,record=events.append,
            output=NS(get_subscription_count=count),
            get_subscriptions_info_by_topic=lambda topic:[NS(node_name='tracer',node_namespace='/')]*count())
    def pump(dt):clock[0]+=dt;node.can_time=clock[0]
    node.pump=pump
    driver=NS(poll=lambda:2 if case=='exit' else None)
    if case=='delayed':wait_for_driver_subscription(node,driver,timeout=.5)
    else:
        with pytest.raises(RuntimeError,match={'timeout':'alive','duplicate':'Multiple','exit':'returncode=2'}[case]):
            wait_for_driver_subscription(node,driver,timeout=.5)
    assert events and not node.enabled


def test_driver_discovery_waits_for_real_node_identity(monkeypatch):
    from board_approach.motion import wait_for_driver_subscription
    clock=[0.];events=[]
    monkeypatch.setattr('board_approach.motion.time.monotonic',lambda:clock[0])
    node=NS(can_state=[0,1,0,0,0,0],can_time=0.,enabled=False,record=events.append,
        output=NS(get_subscription_count=lambda:1),
        get_subscriptions_info_by_topic=lambda t:[NS(
            node_name='_NODE_NAME_UNKNOWN_' if clock[0]<.3 else 'tracer',
            node_namespace='_NODE_NAMESPACE_UNKNOWN_' if clock[0]<.3 else '/')])
    def pump(dt):clock[0]+=dt;node.can_time=clock[0]
    node.pump=pump
    wait_for_driver_subscription(node,NS(poll=lambda:None),timeout=.5)
    assert clock[0]>=.3 and not node.enabled


@pytest.mark.parametrize('phase', ['continuous_coarse', 'precise_docking'])
@pytest.mark.parametrize('position', [0., 1.7, 2.])
def test_expired_feedback_continues_through_endpoint(monkeypatch,tmp_path,position,phase):
    import json
    from board_approach.motion import motion_feedback
    node,output,tick=fixture_node(monkeypatch)
    node.target.update(board_map=[3.,0.,0.],goal_map=[2.,0.,0.],approach_phase=phase)
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=-30.,object_map=[3.,0.,0.])))
    node.data['pose'].pose.pose.position.x=position
    node.last_xy=np.array([position,0.]);node.command_time=10.
    tick(10.)
    assert node.failure is None and output[-1]==(.05,0.)
    assert motion_feedback(node,force=True) is None
    node.visual_feedback.unlink()
    assert motion_feedback(node,force=True) is None
    assert not getattr(node,'near_goal_feedback',False)


@pytest.mark.parametrize('distance,travel,stopped', [(3.,3.,False),(4.51,4.51,True),(3.,5.51,True)])
def test_extended_approach_keeps_displacement_and_travel_guards(monkeypatch,distance,travel,stopped):
    node,output,tick=fixture_node(monkeypatch)
    node.data['pose'].pose.pose.position.x=distance
    node.last_xy=np.array([distance,0.])
    node.travel=travel
    tick(0.)
    if stopped:
        assert node.failure=='Motion distance budget exceeded'
        assert output[-1]==(0.,0.)
    else:
        assert node.failure is None and output[-1]==(.05,0.)


@pytest.mark.parametrize('age', [3.3, 6.1, 10.65, 29., 30.1])
def test_start_feedback_accepts_delayed_results_but_rejects_over_thirty_seconds(monkeypatch,tmp_path,age):
    import json,time
    node,output,tick=fixture_node(monkeypatch)
    node.enabled=False
    node.target['board_map']=[3.,0.,0.]
    node.visual_feedback=tmp_path/'feedback.json'
    node.visual_feedback.write_text(json.dumps(dict(captured_monotonic=10.-age,object_map=[3.,0.,0.])))
    tick(10.)
    node.pump=lambda seconds:tick(time.monotonic()+seconds)
    if age<30:
        wait_for_start_feedback(node,timeout=.4)
    else:
        with pytest.raises(RuntimeError,match='No fresh visual'):
            wait_for_start_feedback(node,timeout=.4)
    assert not node.enabled and all(v==(0.,0.) for v in output)


def test_obstacle_guard_zeros_before_retaining_exact_trigger_scan(monkeypatch):
    node,output,tick=fixture_node(monkeypatch)
    node.data['scan'].ranges[45:55]=[.3]*10
    tick(0.)
    assert node.failure=='Obstacle in forward stopping envelope'
    assert output[-1]==(0.,0.)
    evidence=node.obstacle_evidence
    assert evidence['metadata']['scan_timestamp_ns']==1
    assert np.min(np.linalg.norm(evidence['points_base'][:,:2],axis=1))==pytest.approx(.3)
    node.data['scan'].ranges[50]=3.
    assert np.min(np.linalg.norm(evidence['points_base'][:,:2],axis=1))==pytest.approx(.3)


def test_smoothed_turn_does_not_delay_emergency_zero(monkeypatch):
    node,out,tick=fixture_node(monkeypatch)
    node.command.angular.z=.15
    tick(0.)
    assert 0.<out[-1][1]<=.00301
    tick(.02)
    assert out[-1][1]<=.00601
    tick(.04,InputFreshnessError('scan stale'))
    assert out[-1]==(0.,0.) and node.steering_w==0.


def test_residual_smoothed_rotation_still_checks_turning_clearance(monkeypatch):
    node,out,tick=fixture_node(monkeypatch)
    node.steering_w=.08
    node.data['scan'].angle_min=1.
    node.data['scan'].angle_increment=.01
    node.data['scan'].ranges[45:55]=[.5]*10  # supported side obstacle
    node.command.angular.z=0.
    tick(0.)
    assert node.failure=='Obstacle in turning clearance'
    assert out[-1]==(0.,0.)


def test_double_speed_is_bounded_and_stop_remains_immediate(monkeypatch):
    node,out,tick=fixture_node(monkeypatch)
    node.speed=.10;node.command.linear.x=.3
    tick(0.)
    assert out[-1]==(.10,0.)
    node.command=Twist();node.command_time=.02
    tick(.02)
    assert out[-1]==(0.,0.)


@pytest.mark.parametrize('cloud_stamp,pose_stamp,expected', [(1,1,1.2),(2,1,.8),(1,200000001,.8)])
def test_door_filter_requires_matching_cloud_and_pose(monkeypatch,cloud_stamp,pose_stamp,expected):
    from test_door_filter import model
    import board_approach.door_filter as filtering
    node,output,tick=fixture_node(monkeypatch)
    scan=node.data['scan'];scan.header.stamp.nanosec=1
    scan.angle_min=0.;scan.angle_increment=.1;scan.ranges=[.8]
    node.target.update(approach_phase='precise_docking')
    node.door_clouds=[NS(header=NS(frame_id='laser',stamp=NS(sec=0,nanosec=cloud_stamp)))]
    node.poses=[NS(header=NS(stamp=NS(sec=0,nanosec=pose_stamp)),pose=NS(pose=NS(
        position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))))]
    node.door_filter_cache=None
    monkeypatch.setattr(filtering,'cloud_xyz',lambda cloud:np.array([[.8,0,.5],[1.2,0,.5]]))
    # Avoid exact 40cm boundary: the face is at 1.1m.
    feedback=dict(captured_monotonic=0.,door_model=model(x=1.1,captured=0.))
    result=Motion.near_door_ranges(node,scan,feedback)
    assert result[0]==pytest.approx(expected)


def test_fine_controller_and_teb_stop_at_same_tighter_tolerance():
    from board_approach.motion import approach_controller_parameters
    from board_approach.limits import FINAL_LATERAL_TOLERANCE_M, FINAL_HEADING_TOLERANCE_DEG
    import math
    p=approach_controller_parameters(.1, fine=True)
    assert p['goal_checker.xy_goal_tolerance'] < FINAL_LATERAL_TOLERANCE_M
    assert 'FollowPath.xy_goal_tolerance' not in p
    assert p['goal_checker.yaw_goal_tolerance'] < math.radians(FINAL_HEADING_TOLERANCE_DEG)
    assert 'FollowPath.yaw_goal_tolerance' not in p


def test_navigation_retry_waits_at_zero_and_is_bounded(monkeypatch):
    from board_approach.motion import hold_for_navigation_retry
    import board_approach.motion as module
    now=[0.];output=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(module,'can_error',lambda *a:None)
    node=NS(enabled=True,failure=None,can_state=[],can_time=0.,record=lambda e:None,
            output=NS(publish=lambda v:output.append((v.linear.x,v.angular.z))),health=lambda:None)
    def pump(dt):
        assert not node.enabled
        now[0]+=dt
    node.pump=pump
    driver=NS(poll=lambda:None)
    hold_for_navigation_retry(node,driver,1,6)
    assert now[0]>=1.5 and output and all(x==(0.,0.) for x in output)
    assert not node.enabled
    with pytest.raises(RuntimeError,match='Navigation failed'):
        hold_for_navigation_retry(node,driver,3,6)
    with pytest.raises(RuntimeError,match='Navigation failed'):
        hold_for_navigation_retry(node,driver,1,5) # Cancellation must not restart.
    node.failure='Obstacle in forward region'
    with pytest.raises(RuntimeError,match='Obstacle'):
        hold_for_navigation_retry(node,driver,2,6)


@pytest.mark.parametrize('completes',[True,False])
def test_endpoint_wait_is_zero_bounded_and_drains_pending(monkeypatch,tmp_path,completes):
    import board_approach.motion as m
    now=[0.];output=[];checks=[]
    monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(m,'can_error',lambda *a:None)
    monkeypatch.setattr(m,'motion_feedback',lambda n:{'latest':True})
    pending=tmp_path/'pending.json';pending.write_text('{}')
    pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
    n=NS(enabled=True,failure=None,visual_feedback=tmp_path/'feedback.json',can_state=[],can_time=0.,
         record=lambda e:None,health=lambda:checks.append(1),data={'pose':NS(pose=NS(pose=pose))},
         output=NS(publish=lambda v:output.append((v.linear.x,v.angular.z))))
    def pump(dt):
        assert not n.enabled and (tmp_path/'endpoint_wait.json').exists()
        now[0]+=dt
        if completes and now[0]>=.5:pending.unlink(missing_ok=True)
    n.pump=pump
    f,transform=m.endpoint_observation(n,NS(poll=lambda:None),10.)
    assert f=={'latest':True} and np.allclose(transform,np.eye(4))
    assert output and all(x==(0.,0.) for x in output) and checks
    assert now[0] <=4.1 and not n.enabled
    if completes:assert now[0]<1.
