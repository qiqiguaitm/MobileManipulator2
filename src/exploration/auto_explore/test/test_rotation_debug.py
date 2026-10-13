from types import SimpleNamespace as P
import numpy as np
from auto_explore.rotation_debug import first_blockage,outline

BODY=[(-.425,-.36),(-.425,.36),(.484,.36),(.484,-.36)]

def test_rotating_into_unknown_reports_only_intersecting_cells():
    g=np.zeros((40,40),dtype=np.uint8)
    # Unknown strip just beyond the rear of the initial footprint.
    g[:,10]=255
    poses=[P(x=0.,y=0.,theta=t) for t in (0.,.03,.07,.15)]
    p,u,o,reason=first_blockage(poses,BODY,g,.05,(-1.,-1.))
    assert p.theta>0 and u and not o and reason=='cells'
    assert all(x==10 for x,y in u)

def test_lethal_separate_from_unknown_and_inflation_not_lethal():
    g=np.full((40,40),253,dtype=np.uint8);pose=P(x=0.,y=0.,theta=0.)
    assert first_blockage([pose],BODY,g,.05,(-1.,-1.))[0] is None
    g[:,11]=254
    p,u,o,reason=first_blockage([pose],BODY,g,.05,(-1.,-1.))
    assert p is not None and not u and o

def test_out_of_grid_is_not_reported_as_obstacle():
    p,u,o,reason=first_blockage([P(x=5.,y=0.,theta=0.)],BODY,np.zeros((40,40),dtype=np.uint8),.05,(-1.,-1.))
    assert reason=='outside_costmap' and not u and not o


def test_goal_stamp_refresh_does_not_rewrite_map(tmp_path):
    from visualization_msgs.msg import Marker,MarkerArray
    from auto_explore.rotation_debug import RotationDebug
    n=P(goal_signature=None,goal_serialized=None,goal_markers=None,global_map=None,out=tmp_path,
        get_clock=lambda:P(now=lambda:P(nanoseconds=123)))
    m=Marker();m.header.frame_id='map';m.header.stamp.sec=1;m.text='ACTIVE'
    msg=MarkerArray(markers=[m]);RotationDebug.on_goal(n,msg)
    original=(tmp_path/'navigation_goal.json').read_text()
    m.header.stamp.sec=2;RotationDebug.on_goal(n,msg)
    assert (tmp_path/'navigation_goal.json').read_text()==original
    assert len((tmp_path/'navigation_goals.jsonl').read_text().splitlines())==1
    m.text='FAILED';RotationDebug.on_goal(n,msg)
    assert len((tmp_path/'navigation_goals.jsonl').read_text().splitlines())==2
