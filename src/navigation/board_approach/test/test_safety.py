import numpy as np
import pytest
from board_approach.safety import can_error, freshness_error, guarded_velocity
from board_approach.safety import FreshDataGate


def cluster(x, y, n=10):
    return np.column_stack([np.full(n,x), y+np.arange(n)*.005, np.zeros(n)])


def clear_scan():
    angle = np.linspace(-np.pi, np.pi, 100)
    return np.column_stack([3*np.cos(angle), 3*np.sin(angle), np.zeros(100)])


def test_missing_stale_and_future_inputs_fail_closed():
    assert freshness_error(10, {}, ['scan'])
    assert freshness_error(10, {'scan': 9}, ['scan'])
    assert freshness_error(10, {'scan': 11}, ['scan'])
    assert freshness_error(10, {'scan': 9.8}, ['scan']) is None


@pytest.mark.parametrize('state,age', [
    (None, 0), ([0], 0), ([0, 1, 0, 0, 0, 0], .5),
    ([1, 1, 0, 0, 0, 0], .1), ([0, 0, 0, 0, 0, 0], .1),
    ([0, 1, 0, 0, 1, 0], .1)])
def test_can_emergency_and_errors_stop(state, age):
    assert can_error(state, age)


def test_valid_can_state():
    assert can_error([0, 1, 10, 80, 0, 0], .1) is None


def test_speed_scaling_preserves_curvature():
    v, w = guarded_velocity(.4, 0, .4, clear_scan(), .08)
    assert v == pytest.approx(.08)
    assert w == pytest.approx(.08)


@pytest.mark.parametrize('v,vy,w', [(-.1, 0, 0), (0, .1, 0), (np.nan, 0, 0), (.1, 0, np.inf)])
def test_reverse_lateral_nan_and_inf_are_rejected(v, vy, w):
    with pytest.raises(RuntimeError):
        guarded_velocity(v, vy, w, clear_scan(), .08)


def test_forward_obstacle_stops():
    with pytest.raises(RuntimeError, match='forward'):
        guarded_velocity(.08, 0, 0, np.vstack([clear_scan(), cluster(.8,.1)]), .08)


def test_side_obstacle_stops_turning():
    with pytest.raises(RuntimeError, match='turning'):
        guarded_velocity(0, 0, .1, np.vstack([clear_scan(), cluster(0,.5)]), .08)


def test_empty_scan_stops():
    with pytest.raises(RuntimeError):
        guarded_velocity(.08, 0, 0, np.empty((0, 3)), .08)


def test_point_turn_uses_radial_clearance_but_forward_still_stops():
    points = np.vstack([clear_scan(), cluster(.88,-.476)])
    assert guarded_velocity(0, 0, .1, points, .05) == (0., .1)
    with pytest.raises(RuntimeError, match='forward'):
        guarded_velocity(.01, 0, .1, points, .05)
    with pytest.raises(RuntimeError, match='turning'):
        guarded_velocity(0, 0, .1, np.vstack([points, cluster(.6,0)]), .05)


def test_stale_scan_stops_until_three_new_fresh_frames():
    gate = FreshDataGate()
    assert gate.update(0., True, 1)
    assert not gate.update(.1, False, 1)
    assert not gate.update(.3, True, 2)
    assert not gate.update(.4, True, 3)
    assert gate.update(.51, True, 4)


def test_repeated_same_scan_cannot_resume():
    gate = FreshDataGate()
    assert not gate.update(0., False, 1)
    for now in (.1, .3, .8, 1.5):
        assert not gate.update(now, True, 2)
    with pytest.raises(RuntimeError, match='recover'):
        gate.update(2., True, 2)


def test_intermittent_delays_do_not_reset_total_pause_deadline():
    gate = FreshDataGate()
    assert not gate.update(0., False, 1)
    assert not gate.update(.2, True, 2)
    assert not gate.update(.8, False, 2)
    assert not gate.update(1.8, True, 3)
    with pytest.raises(RuntimeError, match='recover'):
        gate.update(2.1, True, 4)


def test_out_of_order_scans_do_not_count_as_new_frames():
    gate = FreshDataGate()
    assert not gate.update(0., False, 10)
    assert not gate.update(.1, True, 12)
    assert not gate.update(.3, True, 11)
    assert not gate.update(.5, True, 12)
    assert not gate.update(.6, True, 13)
    assert gate.update(.7, True, 14)


@pytest.mark.parametrize('mode', [0, 3])
def test_initialization_accepts_idle_rc_but_motion_does_not(mode):
    state=[0,mode,0,0,0,0]
    assert can_error(state,.1,require_can=False) is None
    assert 'mode' in can_error(state,.1)


@pytest.mark.parametrize('state', [[1,3,0,0,0,0], [0,3,0,0,0x80,0],
    [0,0,0,0,0,0x80], [0,2,0,0,0,0], [0,255,0,0,0,0]])
def test_initialization_keeps_emergency_fault_and_foreign_mode_protection(state):
    error=can_error(state,.1,require_can=False)
    assert error and 'raw=[' in error and 'fault_bytes=' in error


def test_initialization_rejects_stale_feedback():
    assert 'stale' in can_error([0,3,0,0,0,0],.5,require_can=False)


@pytest.mark.parametrize('points,stop',[
    ([[.88,0,0]],False),
    ([[.8,0,0],[.81,.02,0]],False),
    ([[.8,0,0],[.81,.02,0],[.82,.04,0]],False),
    ([[.8,-.4,0],[.8,0,0],[.8,.4,0]],False),
    ([[.59,0,0]],False),
    ([[.61,0,0]],False),
])
def test_forward_noise_requires_supported_cluster_without_single_point_exception(points,stop):
    scan=np.vstack([clear_scan(),points])
    if stop:
        with pytest.raises(RuntimeError,match='forward'):guarded_velocity(.05,0,0,scan,.05)
    else:
        assert guarded_velocity(.05,0,0,scan,.05)==(.05,0.)


def test_steering_reversal_is_slew_limited_and_bounded():
    from board_approach.safety import smooth_steering
    w=0.
    for _ in range(100):w=smooth_steering(.05,.15,w,.02)
    assert w==pytest.approx(.08)
    new=smooth_steering(.05,-.15,w,.02)
    assert new==pytest.approx(.077)
    for _ in range(100):new=smooth_steering(.05,-.15,new,.02)
    assert new==pytest.approx(-.08)
    assert smooth_steering(0,0,.08,.02)==0.
    assert smooth_steering(.05,-.15,.08,5.)==pytest.approx(.065)


def test_near_isolated_point_does_not_stop_forward_or_turn():
    points=np.vstack([clear_scan(),[.516435,.065337,0]])
    assert guarded_velocity(.10,0,-.04466,points,.10)==(.10,-.04466)
    assert guarded_velocity(0,0,.08,points,.10)==(0.,.08)


def test_three_scattered_points_do_not_count_as_turning_obstacle():
    points=np.vstack([clear_scan(),[0,.5,0],[0,-.5,0],[-.5,0,0]])
    assert guarded_velocity(0,0,.08,points,.10)==(0.,.08)


@pytest.mark.parametrize('n,stop', [(9,False),(10,True),(11,True)])
@pytest.mark.parametrize('v,w', [(.1,0),(0,.08)])
def test_ten_point_boundary_for_forward_and_turning(n,stop,v,w):
    points=np.vstack([clear_scan(),cluster(.5,0,n)])
    if stop:
        with pytest.raises(RuntimeError,match='Obstacle'):
            guarded_velocity(v,0,w,points,.1)
    else:
        assert guarded_velocity(v,0,w,points,.1)==(v,w)


def test_ten_scattered_points_are_not_a_supported_cluster():
    points=np.vstack([clear_scan(),[[x,y,0] for x in (.3,.7) for y in (-.4,-.2,0,.2,.4)]])
    assert guarded_velocity(.1,0,.08,points,.1)==(.1,.08)



def test_separated_groups_inside_large_neighborhood_do_not_combine():
    from board_approach.safety import forward_obstacle_evidence
    points=np.array([[.80+i*.002,y,0] for y in (-.06,0,.06) for i in range(4)])
    _,mask,meta=forward_obstacle_evidence(points)
    assert not mask.any() and meta['component_sizes']==[4,4,4]


def test_saved_scattered_door_trigger_splits_into_small_components():
    from board_approach.safety import forward_obstacle_evidence
    xy=[[.898440,.155483],[.875276,.155333],[.889471,.188003],[.887811,.193376],
        [.891466,.206445],[.891981,.224780],[.804420,.247378],[.829305,.264862],
        [.826977,.269707],[.811192,.267868],[.896196,.337692],[.897554,.345505],
        [.888067,.347305],[.876456,.347734]]
    _,mask,meta=forward_obstacle_evidence(np.column_stack([xy,np.zeros(len(xy))]))
    assert not mask.any() and meta['component_sizes']==[6,4,4]


def test_continuous_wall_still_triggers_with_ten_local_points():
    from board_approach.safety import forward_obstacle_evidence
    points=np.column_stack([np.full(15,.85),np.linspace(-.09,.09,15),np.zeros(15)])
    _,mask,meta=forward_obstacle_evidence(points)
    assert mask.all() and meta['component_sizes']==[15]


@pytest.mark.parametrize('distance,x,y',[(2.,.9,.48),(1.5,.9,.48),(1.3,.8,.43),(1.1,.7,.38),(.934,.7,.38)])
def test_precise_envelope_shrinks_continuously(distance,x,y):
    from board_approach.safety import forward_bounds
    target=dict(approach_phase='precise_docking',board_map=[distance,0,0],surface_normal_map=[-1,0,0])
    bounds=forward_bounds(target,[0,0])
    assert bounds['x_max']==pytest.approx(x)
    assert bounds['abs_y_max']==pytest.approx(y)
    assert bounds['x_max']>.434 and bounds['abs_y_max']>.31
    target['approach_phase']='continuous_coarse'
    assert forward_bounds(target,[0,0])['x_max']==.9


def test_near_envelope_keeps_close_obstacles_and_turning_guard():
    from board_approach.safety import forward_bounds
    bounds=forward_bounds(dict(approach_phase='precise_docking',board_map=[1,0,0],surface_normal_map=[-1,0,0]),[0,0])
    assert guarded_velocity(.1,0,0,np.vstack([clear_scan(),cluster(.85,0)]),.1,bounds)==(.1,0.)
    with pytest.raises(RuntimeError,match='forward'):
        guarded_velocity(.1,0,0,np.vstack([clear_scan(),cluster(.6,0)]),.1,bounds)
    with pytest.raises(RuntimeError,match='turning'):
        guarded_velocity(0,0,.08,np.vstack([clear_scan(),cluster(0,.5)]),.1,bounds)


def test_invalid_target_does_not_shrink_envelope():
    from board_approach.safety import forward_bounds
    for target in ({},dict(approach_phase='precise_docking'),dict(approach_phase='precise_docking',board_map=[1,0,0],surface_normal_map=[float('nan'),0,0])):
        assert forward_bounds(target,[0,0])==forward_bounds()
