import math
import numpy as np
import pytest
from board_approach.transition import moving_transition_target


def scene():
    reference=dict(geometry_source='lidar',use_aruco=False,board_map=[3.,0.,0.],surface_normal_map=[-1.,0.,0.])
    pose=np.eye(4);pose[0,3]=1.
    reports=[]
    for i in range(3):
        prior=pose.copy();prior[0,3]-=(2-i)*.15
        reports.append(dict(geometry_source='lidar',timestamp_ns=i+1,captured_monotonic=93.+i*2.5,
                            map_from_base=prior.tolist(),object_map=[3.,0.,0.],surface_normal_map=[-1.,0.,0.]))
    return reference,pose,reports


def test_one_moving_frame_uses_current_pose_without_stationary_frame():
    ref,pose,reports=scene()
    report=reports[-1];report['map_from_base'][0][3]=.8
    target=moving_transition_target([report],ref,pose,100.,1.)
    assert target['prefetched_observations']==1 and target['stationary_observations']==0
    assert target['distance']==2. and target['goal_map']==[2.,0.,0.]
    assert target['observed_robot_xy']==[1.,0.]
    assert target['heading_error']==0. and target['coarse_orientation'] is False
    assert target['multi_frame_consistent'] is None


def test_old_inconsistent_frames_do_not_veto_or_shift_latest_fit():
    ref,pose,reports=scene()
    reports[0]['object_map'][1]=.5
    reports[0]['surface_normal_map']=[-math.cos(.3),math.sin(.3),0.]
    target=moving_transition_target(reports,ref,pose,100.,1.)
    assert target['board_map']==reports[-1]['object_map']
    assert target['surface_normal_map']==reports[-1]['surface_normal_map']
    assert target['diagnostic_observations']==3
    assert target['multi_frame_consistent'] is False


def test_gap_over_four_seconds_is_not_a_transition_gate():
    ref,pose,reports=scene()
    target=moving_transition_target([reports[0],reports[-1]],ref,pose,100.,1.)
    assert target is not None and target['prefetched_observations']==1


@pytest.mark.parametrize('fault',['stale','future','missing','center','normal','far','turn','identity','error','depth','aruco','nan'])
def test_single_frame_retains_freshness_geometry_and_identity_guards(fault):
    ref,pose,reports=scene();reports=reports[-1:];r=reports[0]
    if fault=='stale':r['captured_monotonic']=93.9
    if fault=='future':r['captured_monotonic']=101.
    if fault=='missing':reports=[]
    if fault=='center':r['object_map'][1]=.3
    if fault=='normal':r['surface_normal_map']=[-math.cos(.6),math.sin(.6),0.]
    if fault=='far':r['map_from_base'][0][3]=0.
    if fault=='turn':r['map_from_base'][0][0]=math.cos(1.);r['map_from_base'][1][0]=math.sin(1.)
    if fault=='identity':ref['board_map']=[3.,.4,0.]
    if fault=='error':r['error']='plane failed'
    if fault=='depth':ref['geometry_source']='depth'
    if fault=='aruco':ref['use_aruco']=True
    if fault=='nan':r['object_map'][0]=float('nan')
    assert moving_transition_target(reports,ref,pose,100.,1.) is None


def test_endpoint_uses_current_pose_and_keeps_strict_arrival_limits():
    from board_approach.perception import arrived
    ref,pose,reports=scene()
    pose[0,3]=2.;pose[1,3]=.02
    reports[-1]['map_from_base']=pose.tolist()
    target=moving_transition_target(reports[-1:],ref,pose,100.,1.)
    target['detection_source']='object_lidar'
    assert arrived(target,1.,require_aruco=False)
    pose[1,3]=.06
    target=moving_transition_target(reports[-1:],ref,pose,100.,1.)
    target['detection_source']='object_lidar'
    assert not arrived(target,1.,require_aruco=False)
