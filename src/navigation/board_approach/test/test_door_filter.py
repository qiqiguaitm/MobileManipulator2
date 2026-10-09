from types import SimpleNamespace as NS
import numpy as np
import pytest
from board_approach.door_filter import reflection_mask, reconstruct_scan, door_model


def model(x=1.2, captured=10.):
    t=np.linspace(0,2*np.pi,48,endpoint=False)
    return dict(polygon_map=np.column_stack([np.full(48,x),.3*np.cos(t),.5+.3*np.sin(t)]).tolist(),
                normal_map=[-1.,0.,0.],captured_monotonic=captured)


def test_only_inner_door_front_prism_is_candidate():
    points=np.array([[1.,0,.5],[1.2,0,.5],[1.,.4,.5],[1.,0,.05],[.7,0,.5],[1.4,0,.5],[.6,0,.5]])
    assert reflection_mask(points,model(),np.eye(4),'precise_docking',11.).tolist()==[True,False,False,False,False,False,False]


@pytest.mark.parametrize('phase,x,now', [('continuous_coarse',1.2,11),('precise_docking',1.6,11),
    ('precise_docking',.7,11),('precise_docking',1.2,16.01),('precise_docking',1.2,9)])
def test_far_coarse_stale_and_future_do_not_filter(phase,x,now):
    assert not reflection_mask([[x-.2,0,.5]],model(x),np.eye(4),phase,now).any()


def test_invalid_model_keeps_all_points():
    for value in (None,{},dict(model(),normal_map=[float('nan'),0,0]),dict(model(),polygon_map=[[0,0,0]])):
        assert not reflection_mask([[1.,0,.5]],value,np.eye(4),'precise_docking',11.).any()


def test_rebin_keeps_next_obstacle_and_unmatched_scan():
    scan=NS(ranges=[.8,.6],angle_min=0.,angle_increment=.1,range_min=.2,range_max=10.)
    points=np.array([[.8,0,.5],[1.,0,.5],[.7*np.cos(.15),.7*np.sin(.15),.5]])
    values,changed=reconstruct_scan(points,scan,np.array([True,False,True]))
    assert values.tolist()==[1.,.6]
    assert changed.tolist()==[True,False]


def test_same_bin_real_return_is_never_erased_with_candidate():
    scan=NS(ranges=[.8],angle_min=0.,angle_increment=.1,range_min=.2,range_max=10.)
    values,changed=reconstruct_scan([[.8,0,.5],[.8,0,.1]],scan,[True,False])
    assert values.tolist()==[.8] and not changed.any()


def test_model_uses_distortion_corrected_rays_and_real_capture_time():
    t=np.linspace(0,2*np.pi,48,endpoint=False)
    region=NS(exclusion_polygons=[np.column_stack([100+25*np.cos(t),100+25*np.sin(t)])])
    obs=NS(normal=np.array([0,0,-1.]),center=np.array([0,0,1.2]))
    k=np.array([[100.,0,100],[0,100.,100],[0,0,1.]])
    value=door_model(region,obs,k,np.zeros(5),np.eye(4),np.eye(4),10.)
    assert value['captured_monotonic']==10.
    assert np.allclose(np.asarray(value['polygon_map'])[:,2],1.2)
