from types import SimpleNamespace as NS
import numpy as np
import pytest
from board_approach.object_center import object_observation, center_target_geometry, ObjectDetector
from board_approach.perception import DetectionError, BoardUnavailable, arrived


def inputs():
    return np.zeros((480,640,3),np.uint8), np.full((480,640),2.), np.array([[640.,0,320.],[0,640.,240.],[0,0,1.]])


def test_generic_box_needs_no_board_pattern_or_plane_normal():
    _,depth,k=inputs()
    obs=object_observation(depth,k,[350,170,470,310],.8)
    assert obs.center == pytest.approx([.28125,0,2.])
    assert obs.ids == []
    assert obs.source == 'object_depth'
    base_camera=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,.6],[0,0,0,1.]])
    target=center_target_geometry(obs,base_camera,np.eye(4),1.)
    assert target['distance'] == pytest.approx(np.hypot(2,.28125))
    # Stop on the ray from robot to target; no front-plane orientation is used.
    assert np.linalg.norm(np.array(target['goal_map'][:2])-np.array(target['object_map'][:2])) == pytest.approx(1.)
    assert target['goal_yaw'] == pytest.approx(np.arctan2(-.28125,2))
    assert not target['identity_verified']


def test_uncertain_or_cropped_depth_is_rejected():
    _,depth,k=inputs()
    with pytest.raises(DetectionError,match='cropped'):
        object_observation(depth,k,[0,100,150,300],.9)
    with pytest.raises(DetectionError,match='valid depth'):
        object_observation(np.zeros_like(depth),k,[200,100,400,300],.9)
    depth[:,300:]=3.
    with pytest.raises(DetectionError,match='ambiguous'):
        object_observation(depth,k,[200,100,400,300],.9)


def test_object_arrival_requires_range_and_bearing():
    target=dict(distance=1.,lateral=0.,heading_error=0.,detection_source='object_depth',identity_verified=False)
    assert arrived(target,1.,require_aruco=False)
    assert not arrived(target,1.)
    assert not arrived(dict(target,heading_error=.2),1.,require_aruco=False)
    assert not arrived(dict(target,detection_source='appearance_depth'),1.,require_aruco=False)


def test_detector_never_chooses_arbitrarily_between_two_objects():
    image,depth,k=inputs()
    detector=ObjectDetector('http://127.0.0.1:1','washing machine')
    objects=[dict(score=.8,bbox=[200,100,400,300])]
    def post(*args,**kwargs):
        assert kwargs['data']['text_prompt']=='washing machine'
        assert kwargs['timeout']==5.
        return NS(raise_for_status=lambda:None,json=lambda:dict(results=[dict(objects=list(objects))]))
    detector.session=NS(post=post)
    assert detector.detect(image,depth,k).source=='object_depth'
    objects.append(dict(score=.7,bbox=[420,100,600,300]))
    with pytest.raises(DetectionError,match='Multiple objects'):
        detector.detect(image,depth,k)
    objects.clear()
    with pytest.raises(BoardUnavailable):
        detector.detect(image,depth,k)


def test_wide_oblique_surface_is_not_mistaken_for_mixed_background():
    _,_,k=inputs()
    y,x=np.indices((480,640))
    depth=2.8/(1.-.8*(x-320)/640.)
    obs=object_observation(depth,k,[100,80,540,400],.9)
    assert obs.center==pytest.approx([0.,0.,2.8],abs=.01)
    assert obs.source=='object_depth'
    assert obs.plane_fit['inlier_fraction']>.95
