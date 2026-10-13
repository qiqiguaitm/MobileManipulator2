import numpy as np
from auto_explore.occupancy import VoxelMap,correction_at,transform


def test_tail_alignment_and_historical_interpolation():
    c0=np.eye(4);c1=np.eye(4);c1[0,3]=2.
    anchors=[(1.,c0),(3.,c1)]
    assert correction_at(2.,anchors)[0,3]==1.
    assert correction_at(10.,anchors)[0,3]==2.
    raw=transform([5,0,0],[0,0,0,1]);opt=transform([7,1,0],[0,0,0,1])
    tail=transform([6,0,0],[0,0,0,1]);result=(opt@np.linalg.inv(raw))@tail
    assert np.allclose(result[:3,3],[8,1,0])


def test_rays_mark_free_and_obstacle_but_not_behind_hit():
    v=VoxelMap([-1,-1,4,2]);v.integrate(np.array([[2,0,.5]]),[0,0,.5]);g=v.grid()
    assert g[20,40]==0 and g[20,60]==100 and g[20,70]==-1


def test_rebuild_moves_obstacle_without_old_ghost():
    old=VoxelMap([-1,-1,4,2]);old.integrate(np.array([[2,0,.5]]),[0,0,.5])
    new=VoxelMap([-1,-1,4,2]);new.integrate(np.array([[2,1,.5]]),[0,1,.5])
    assert old.grid()[20,60]==100 and new.grid()[20,60]==-1 and new.grid()[40,60]==100


def test_ground_endpoints_are_not_obstacles():
    v=VoxelMap([-1,-1,4,2]);v.integrate(np.array([[2,0,-.2]]),[0,0,.5])
    assert not np.any(v.grid()==100)


def test_append_keyframe_does_not_rebuild_unchanged_correction():
    from auto_explore.occupancy import corrections_changed
    c=np.eye(4)
    assert not corrections_changed([(1.,c)],[(1.,c),(2.,c.copy())])
    c2=c.copy();c2[0,3]=.02
    assert corrections_changed([(1.,c)],[(1.,c),(2.,c2)])
    # Same latest correction is insufficient: an old pose may have moved.
    assert corrections_changed([(1.,c),(2.,c)],[(1.,c2),(2.,c)])
