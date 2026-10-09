"""No hardware: validate zero-output holds and readiness gates with a fake clock."""
from types import SimpleNamespace as NS
import json
import pytest
pytest.importorskip('rclpy')
from geometry_msgs.msg import Twist
from board_approach.motion import (hold_for_next_leg, wait_for_inputs,
                                   configure_approach_controller, Motion)
from test_motion_guard_ros import fixture_node


def held_node(monkeypatch):
    node, output, tick = fixture_node(monkeypatch)
    node.enabled = False
    node.data['odom'] = NS(twist=NS(twist=Twist()))
    clock = [0.]
    def pump(seconds):
        clock[0] += seconds
        tick(clock[0], sequence=int(clock[0]*1000))
    node.pump = pump
    driver = NS(poll=lambda: None)
    return node, output, clock, driver


def test_transition_wait_publishes_zero_even_with_pending_nonzero_cmd(monkeypatch,tmp_path):
    node, output, clock, driver = held_node(monkeypatch)
    request = tmp_path/'request.json'
    pump = node.pump
    def delayed(seconds):
        pump(seconds)
        node.command.linear.x = .05
        if clock[0] >= 1.:
            request.write_text(json.dumps({'action':'precise'}))
    node.pump = delayed
    assert hold_for_next_leg(node, driver, request)['action'] == 'precise'
    assert len(output) >= 20 and all(v == (0.,0.) for v in output)
    assert not node.enabled


@pytest.mark.parametrize('fault', ['driver','can','pose','localization'])
def test_transition_faults_do_not_enable_motion(monkeypatch,tmp_path,fault):
    node, output, clock, driver = held_node(monkeypatch)
    if fault == 'driver': driver.poll = lambda: 1
    if fault == 'can': node.can_state = [1,1,0,0,0,0]
    if fault == 'pose':
        pump = node.pump
        def drift(dt):
            pump(dt)
            node.data['pose'].pose.pose.position.x = .03
        node.pump = drift
    if fault == 'localization':
        def bad(): raise RuntimeError('Localization unreliable')
        node.health = bad
    with pytest.raises(RuntimeError): hold_for_next_leg(node,driver,tmp_path/'missing',timeout=.3)
    assert not node.enabled and all(v == (0.,0.) for v in output)


def test_hold_timeout_and_finish(monkeypatch,tmp_path):
    node, output, clock, driver = held_node(monkeypatch)
    request = tmp_path/'request.json'
    with pytest.raises(RuntimeError,match='timed out'):
        hold_for_next_leg(node,driver,request,timeout=.2)
    request.write_text('{"action":"finish"}')
    assert hold_for_next_leg(node,driver,request)['action'] == 'finish'
    assert not node.enabled


def test_ready_inputs_do_not_wait_four_seconds(monkeypatch):
    node, output, clock, driver = held_node(monkeypatch)
    wait_for_inputs(node)
    assert clock[0] < .2 and not node.enabled


def test_controller_restore_keeps_original_across_two_legs():
    values={'speed':.3,'tolerance':.2}
    def parameters(update=None,names=None):
        if update is not None: values.update(update)
        else: return {k:values[k] for k in names}
    node = NS(parameters=parameters)
    original = {}
    configure_approach_controller(node,{'speed':.05},original)
    configure_approach_controller(node,{'speed':.05,'tolerance':.06},original)
    assert original == {'speed':.3,'tolerance':.2}


@pytest.mark.parametrize('fail_transition',[False,True])
def test_two_leg_session_owns_driver_once_and_cleans_up(monkeypatch,tmp_path,fail_transition):
    import board_approach.motion as m
    from rclpy.task import Future
    coarse=dict(goal_map=[1.,0.,0.],goal_yaw=0.,observed_robot_xy=[0.,0.],approach_phase='continuous_coarse')
    precise=dict(coarse,goal_map=[2.,0.,0.],approach_phase='precise_docking')
    (tmp_path/'coarse.json').write_text(json.dumps(coarse))
    (tmp_path/'precise.json').write_text(json.dumps(precise))
    monkeypatch.setattr('sys.argv',['motion','--goal',str(tmp_path/'coarse.json'),'--result',str(tmp_path/'coarse_result.json'),
                                   '--speed','.05','--next-request',str(tmp_path/'next.json')])
    calls=[];goals=[];values={'FollowPath.max_vel_x':.3,'goal_checker.xy_goal_tolerance':.2,'goal_checker.yaw_goal_tolerance':.2}
    class FakeNode:
        def __init__(self,*a):
            self.data={'pose':NS(pose=NS(pose=NS(position=NS(x=0.,y=0.))))}
            self.enabled=False;self.failure=None;self.can_state=[0,1,0,0,0,0];self.can_time=0.
            self.active_goals=set();self.events=[];self.travel=0.;self.visual_feedback=None
            self.output=NS(publish=lambda msg: None)
            self.can=NS(close=lambda:calls.append('socket_closed'))
            self.guard_timer=NS(cancel=lambda:calls.append('guard_cancelled'))
        def pump(self,*a): pass
        def wait_healthy(self): pass
        def record(self,event): self.events.append(event)
        def parameters(self,update=None,names=None):
            if update is not None: values.update(update)
            else: return {k:values.get(k,1.) for k in names}
        def get_publishers_info_by_topic(self,*a): return [1]
        def get_subscriptions_info_by_topic(self,*a):return [NS(node_namespace='/',node_name='tracer')]
        def create_client(self,*a):return NS(wait_for_service=lambda **k:True,call_async=lambda req:NS(values=[NS(double_value=.4)]))
        def await_result(self,future,*a):return future
        def get_clock(self):return NS(now=lambda:NS(to_msg=lambda:None))
        def destroy_node(self):calls.append('node_destroyed')
    # ROS message stamp assignment needs a real Time message.
    from builtin_interfaces.msg import Time
    FakeNode.get_clock=lambda self:NS(now=lambda:NS(to_msg=lambda:Time()))
    class FakeAction:
        def __init__(self,*a):pass
        def wait_for_server(self,**k):return True
        def send_goal_async(self,goal,**k):
            goals.append(goal.pose.pose.position.x)
            return NS(accepted=True,get_result_async=lambda:NS(done=lambda:True,result=lambda:NS(status=4)))
    driver=NS(poll=lambda:None,send_signal=lambda sig:calls.append('driver_stop'),wait=lambda **k:0)
    def popen(*a,**k):calls.append('driver_start');return driver
    def hold(node,*a):
        assert not node.enabled and node.command.linear.x==0.
        if fail_transition:raise RuntimeError('transition fault')
        return dict(action='precise',goal=str(tmp_path/'precise.json'),result=str(tmp_path/'fine_result.json'),visual_feedback=str(tmp_path/'feedback.json'))
    monkeypatch.setattr(m,'Motion',FakeNode);monkeypatch.setattr(m,'ActionClient',FakeAction)
    monkeypatch.setattr(m,'hold_for_next_leg',hold)
    for name in ('wait_for_inputs','wait_for_can_mode','wait_for_driver_subscription','wait_for_start_feedback'):
        monkeypatch.setattr(m,name,lambda *a:None)
    monkeypatch.setattr(m,'motion_feedback',lambda *a,**k:None)
    monkeypatch.setattr(m,'can_error',lambda *a,**k:None)
    monkeypatch.setattr(m,'can_command',lambda *a:calls.append(a))
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:NS(returncode=1))
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:b'[{"flags":[]}]')
    monkeypatch.setattr(m.subprocess,'Popen',popen)
    monkeypatch.setattr(m,'get_package_prefix',lambda *a:str(tmp_path))
    monkeypatch.setattr('ament_index_python.packages.get_package_share_directory',lambda *a:str(tmp_path))
    monkeypatch.setattr(m.rclpy,'init',lambda **k:None);monkeypatch.setattr(m.rclpy,'shutdown',lambda:None)
    monkeypatch.setattr(m.signal,'signal',lambda *a:None)
    assert m.main()==(2 if fail_transition else 0)
    assert calls.count('driver_start')==1 and calls.count('driver_stop')==1 and calls.count(('down',))==1
    assert goals==([1.] if fail_transition else [1.,2.])
    assert values['FollowPath.max_vel_x']==.3 and values['goal_checker.xy_goal_tolerance']==.2
    report=json.loads((tmp_path/('coarse_result.json' if fail_transition else 'fine_result.json')).read_text())
    assert report['success'] is (not fail_transition)


@pytest.mark.parametrize('completes,consistent',[(True,True),(True,False),(False,False)])
def test_transition_uses_pending_moving_frame_without_new_capture(monkeypatch,completes,consistent):
    import board_approach.runner as r
    clock=[0.]; polls=[]
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    monitor=NS(recent_reports=[1],target={},future=object())
    def poll(capture=True):
        polls.append(capture)
        if completes and clock[0]>=.2:
            monitor.recent_reports=[1,2];monitor.future=None
    monitor.poll=poll
    pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
    def pump(dt):clock[0]+=dt
    node=NS(health=lambda:None,pump=pump,data={'pose':NS(pose=NS(pose=pose))})
    monkeypatch.setattr(r,'moving_transition_target',lambda reports,*a: {'ready':True} if consistent and len(reports)==2 else None)
    result=r.reuse_moving_transition(node,monitor,1.,pending_timeout=.5)
    assert all(capture is False for capture in polls)
    if completes and consistent:
        assert result['ready'] and .2<=result['transition_pending_wait_seconds']<.3
    else:
        assert result is None
    assert clock[0]<=.55


@pytest.mark.parametrize('available',[True,False])
def test_endpoint_reuse_never_starts_or_waits_for_a_new_detection(monkeypatch,available):
    import board_approach.runner as r
    polls=[]
    monitor=NS(recent_reports=[1],target={},future=object(),poll=lambda capture:polls.append(capture))
    pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
    def forbidden(*a):raise AssertionError('Endpoint waited for another detection')
    node=NS(health=lambda:None,pump=forbidden,data={'pose':NS(pose=NS(pose=pose))})
    monkeypatch.setattr(r,'moving_transition_target',lambda *a:{'ready':True} if available else None)
    result=r.reuse_moving_transition(node,monitor,1.,pending_timeout=0.)
    assert (result is not None)==available
    assert polls==[False]


@pytest.mark.parametrize('passes',[True,False])
def test_terminal_completion_requires_geometry_and_bounds_corrections(monkeypatch,tmp_path,passes):
    import numpy as np
    import board_approach.motion as m
    from rclpy.task import Future
    coarse=dict(goal_map=[1.,0.,0.],goal_yaw=0.,observed_robot_xy=[0.,0.],approach_phase='continuous_approach')
    precise=dict(coarse,goal_map=[2.,0.,0.],approach_phase='precise_docking')
    (tmp_path/'coarse.json').write_text(json.dumps(coarse))
    (tmp_path/'precise.json').write_text(json.dumps(precise))
    monkeypatch.setattr('sys.argv',['motion','--goal',str(tmp_path/'coarse.json'),'--result',str(tmp_path/'coarse_result.json'),
                                   '--speed','.05','--visual-feedback',str(tmp_path/'feedback.json')])
    calls=[];goals=[];values={'FollowPath.max_vel_x':.3,'goal_checker.xy_goal_tolerance':.2,'goal_checker.yaw_goal_tolerance':.2}
    class FakeNode:
        def __init__(self,*a):
            self.data={'pose':NS(pose=NS(pose=NS(position=NS(x=0.,y=0.))))}
            self.enabled=False;self.failure=None;self.can_state=[0,1,0,0,0,0];self.can_time=0.
            self.active_goals=set();self.events=[];self.travel=0.;self.visual_feedback=None
            self.output=NS(publish=lambda msg: None)
            self.can=NS(close=lambda:calls.append('socket_closed'))
            self.guard_timer=NS(cancel=lambda:calls.append('guard_cancelled'))
        def pump(self,*a): pass
        def wait_healthy(self): pass
        def record(self,event): self.events.append(event)
        def parameters(self,update=None,names=None):
            if update is not None: values.update(update)
            else: return {k:values.get(k,1.) for k in names}
        def get_publishers_info_by_topic(self,*a): return [1]
        def get_subscriptions_info_by_topic(self,*a):return [NS(node_namespace='/',node_name='tracer')]
        def create_client(self,*a):return NS(wait_for_service=lambda **k:True,call_async=lambda req:NS(values=[NS(double_value=.4)]))
        def await_result(self,future,*a):return future
        def get_clock(self):return NS(now=lambda:NS(to_msg=lambda:None))
        def destroy_node(self):calls.append('node_destroyed')
    # ROS message stamp assignment needs a real Time message.
    from builtin_interfaces.msg import Time
    FakeNode.get_clock=lambda self:NS(now=lambda:NS(to_msg=lambda:Time()))
    class FakeAction:
        def __init__(self,*a):pass
        def wait_for_server(self,**k):return True
        def send_goal_async(self,goal,**k):
            goals.append(goal.pose.pose.position.x)
            return NS(accepted=True,get_result_async=lambda:NS(done=lambda:True,result=lambda:NS(status=4)))
    driver=NS(poll=lambda:None,send_signal=lambda sig:calls.append('driver_stop'),wait=lambda **k:0)
    def popen(*a,**k):calls.append('driver_start');return driver
    def hold(node,*a):
        assert not node.enabled and node.command.linear.x==0.
        if fail_transition:raise RuntimeError('transition fault')
        return dict(action='precise',goal=str(tmp_path/'precise.json'),result=str(tmp_path/'fine_result.json'),visual_feedback=str(tmp_path/'feedback.json'))
    monkeypatch.setattr(m,'Motion',FakeNode);monkeypatch.setattr(m,'ActionClient',FakeAction)
    monkeypatch.setattr(m,'hold_for_next_leg',hold)
    for name in ('wait_for_inputs','wait_for_can_mode','wait_for_driver_subscription','wait_for_start_feedback'):
        monkeypatch.setattr(m,name,lambda *a:None)
    monkeypatch.setattr(m,'motion_feedback',lambda *a,**k:None)
    monkeypatch.setattr(m,'can_error',lambda *a,**k:None)
    monkeypatch.setattr(m,'can_command',lambda *a:calls.append(a))
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:NS(returncode=1))
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:b'[{"flags":[]}]')
    monkeypatch.setattr(m.subprocess,'Popen',popen)
    monkeypatch.setattr(m,'get_package_prefix',lambda *a:str(tmp_path))
    monkeypatch.setattr('ament_index_python.packages.get_package_share_directory',lambda *a:str(tmp_path))
    monkeypatch.setattr(m.rclpy,'init',lambda **k:None);monkeypatch.setattr(m.rclpy,'shutdown',lambda:None)
    monkeypatch.setattr(m.signal,'signal',lambda *a:None)
    def endpoint(node,*args):
        node.enabled=False
        return {'timestamp_ns':1},np.eye(4)
    monkeypatch.setattr(m,'endpoint_observation',endpoint)
    monkeypatch.setattr(m,'fine_alignment_ready',lambda *a:passes and len(goals)>1)
    monkeypatch.setattr(m,'endpoint_correction',lambda *a:dict(coarse,goal_map=[1.05,0.,0.]))
    assert m.main()==(0 if passes else 2)
    assert calls.count('driver_start')==1 and calls.count('driver_stop')==1 and calls.count(('down',))==1
    assert goals==([1.,1.05] if passes else [1.,1.05,1.05])
    assert values['FollowPath.max_vel_x']==.3 and values['goal_checker.xy_goal_tolerance']==.2
    report=json.loads((tmp_path/'coarse_result.json').read_text())
    assert report['success'] is passes
    if not passes:assert 'budget exhausted' in report['error']
