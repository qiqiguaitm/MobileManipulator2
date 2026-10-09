import numpy as np
import pytest
from board_approach.endpoint import latency_speed_limit, endpoint_correction
from board_approach.continuous import FeedbackExpired


def scene():
    t=dict(approach_phase='continuous_approach',geometry_source='lidar',standoff=.934,
           board_map=[1.,0.,0.],surface_normal_map=[-1.,0.,0.],goal_map=[.066,0.,0.],goal_yaw=0.)
    f=dict(object_map=[1.,.02,0.],surface_normal_map=[-1.,0.,0.],captured_monotonic=10.,
           timings=dict(processing_seconds=4.))
    return t,f


def test_delayed_detection_reduces_speed_before_endpoint():
    t,f=scene()
    assert latency_speed_limit(t,[-3.,0.],f)==pytest.approx(.1)
    assert latency_speed_limit(t,[0.,0.],f)==pytest.approx(.015)
    assert latency_speed_limit(t,[-.15,0.],f)<=latency_speed_limit(t,[-.15,0.],dict(f,timings=dict(processing_seconds=.1)))
    assert 0<latency_speed_limit(t,[0.,0.],f)<=.04


def test_small_forward_correction_uses_latest_plane():
    t,f=scene();g=endpoint_correction(f,t,np.eye(4),11.)
    assert g['goal_map']==pytest.approx([.066,.02,0.])
    assert g['goal_yaw']==pytest.approx(0.)


@pytest.mark.parametrize('fault',['behind','lateral','large','normal','stale','identity'])
def test_correction_rejects_unsafe_or_invalid_adjustments(fault):
    t,f=scene();pose=np.eye(4);now=11.
    if fault=='behind':pose[0,3]=.15
    if fault=='lateral':f['object_map'][1]=.06
    if fault=='large':pose[0,3]=-.2
    if fault=='normal':f['surface_normal_map']=[0.,1.,0.]
    if fault=='stale':now=17.
    if fault=='identity':f['object_map'][1]=.4
    with pytest.raises(RuntimeError):endpoint_correction(f,t,pose,now)


def test_approach_gate_keeps_room_and_requires_new_capture():
    from board_approach.endpoint import ApproachObservationGate, approach_refresh_goal
    t,f=scene();g=ApproachObservationGate()
    assert g.update(t,[-2,0],f,11.)=='continue'
    assert g.update(t,[-1,0],f,11.)=='slow'
    assert g.update(t,[-.7,0],f,12.)=='hold'
    assert g.update(t,[-.7,0],dict(f,captured_monotonic=11.5),13.)=='refresh'
    pose=np.eye(4);pose[0,3]=-.7
    assert approach_refresh_goal(dict(f,captured_monotonic=11.5),t,pose,13.)['goal_map']==pytest.approx([.066,.02,0.])
    assert g.update(t,[0,0],None,14.)=='continue'


def test_approach_gate_timeout_and_late_goal_rejected():
    from board_approach.endpoint import ApproachObservationGate, approach_refresh_goal
    t,f=scene();g=ApproachObservationGate()
    assert g.update(t,[-.7,0],f,11.)=='hold'
    with pytest.raises(RuntimeError,match='reserved maneuvering'):
        g.update(t,[-.7,0],None,32.)
    with pytest.raises(RuntimeError,match='insufficient forward'):
        approach_refresh_goal(f,t,np.eye(4),11.)


def test_refreshed_goal_does_not_blend_back_to_far_observation():
    from board_approach.endpoint import approach_refresh_goal
    from board_approach.continuous import corrected_navigation_goal
    t,f=scene();pose=np.eye(4);pose[0,3]=-.7
    g=approach_refresh_goal(f,t,pose,11.)
    assert corrected_navigation_goal(f,t,g,[-.7,0.]) is None
