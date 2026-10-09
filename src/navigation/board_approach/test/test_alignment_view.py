import numpy as np
import pytest
import cv2
from board_approach.alignment_view import alignment_metrics,save_alignment


def test_aligned_geometry(tmp_path):
    p=tmp_path/'alignment.png'
    result=save_alignment(p,[1.,0.,.5],[-1.,0.,0.])
    assert result==dict(distance=1.,lateral=0.,heading_deg=0.)
    assert cv2.imread(str(p)).shape==(720,760,3)


def test_signs_match_controller_geometry():
    from board_approach.perception import target_geometry,Observation
    n=np.array([-np.cos(.2),-np.sin(.2),0.])
    c=np.array([1.,.15,.5])
    observation=Observation(c,n,[],np.zeros((4,2)),.01,1.)
    target=target_geometry(observation,np.eye(4),np.eye(4),.934)
    result=alignment_metrics(c,n)
    assert result['heading_deg']==pytest.approx(np.degrees(target['heading_error']))
    assert result['distance']==pytest.approx(target['distance'])
    assert result['lateral']==pytest.approx(target['lateral'])
