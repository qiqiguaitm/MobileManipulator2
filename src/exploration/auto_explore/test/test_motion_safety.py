import numpy as np
from auto_explore.motion_safety import collision_risk


def test_swept_footprint_not_blanket_forward_stop():
    assert not collision_risk([[1,0]],.1,0)  # Planner can route around distant obstacle.
    assert collision_risk([[.48,0]],.1,0)
    assert not collision_risk([[.6,.8]],.1,0)
    assert collision_risk([[.43,-.25]],0,.18)


def test_body_clear_does_not_erase_obstacles_or_clear_unknown_ahead():
    from auto_explore.motion_safety import clear_body_unknown
    g=np.full((80,80),-1);g[40,40]=100
    c=clear_body_unknown(g,.05,(-2,-2),(0,0,0))
    assert c[40,40]==100 and c[40,60]==-1 and c[40,35]==0
    assert g[40,35]==-1


def test_official_body_margin_does_not_keep_old_front_overhang():
    # Old .484 m padded front rejected this point even while stationary.
    assert not collision_risk([[.45,0]],0,0)
    assert collision_risk([[.39,0]],0,0)
    assert collision_risk([[0,.35]],0,0)
    assert not collision_risk([[0,.37]],0,0)


def test_cropped_body_clearing_matches_full_map_at_rotations_and_edges():
    from auto_explore.motion_safety import clear_body_unknown,footprint_cells
    rng=np.random.default_rng(42);g=rng.choice([-1,0,100],size=(120,140))
    for pose in [(0,0,0),(1.3,-.2,1.1),(-2.9,-2.8,-.8),(3.95,2.95,2.4),(20,20,0)]:
        expected=g.copy();expected[(g<0)&footprint_cells(g.shape,.05,(-3,-3),pose)]=0
        assert np.array_equal(clear_body_unknown(g,.05,(-3,-3),pose),expected)
