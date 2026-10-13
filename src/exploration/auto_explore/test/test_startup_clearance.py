import math
import numpy as np
import pytest
from auto_explore.startup_clearance import StartupRearClearance
from auto_explore.motion_safety import clear_body_unknown,footprint_cells
from auto_explore.geometry import PLANNING_PADDING


def turn_is_free(grid, angle):
    """Test oracle: sample the padded body across a stationary turn."""
    return all(np.all(grid[footprint_cells(grid.shape,.05,(-2.,-2.),(0.,0.,yaw),
                                          padding=PLANNING_PADDING)]==0)
               for yaw in np.linspace(0.,angle,21))


def test_rear_unknown_patch_enables_small_turns_without_erasing_obstacles():
    g=np.full((80,80),-1,dtype=np.int16);g[:,40:]=0
    g[40,29]=100  # A measured rear obstacle must survive.
    before=clear_body_unknown(g,.05,(-2.,-2.),(0.,0.,0.))
    assert not turn_is_free(before,.18)
    patch=StartupRearClearance()
    after=patch.apply(before,.05,(-2.,-2.),(0.,0.,0.),0.,10.)
    assert after[40,29]==100 and np.array_equal(g[:,40:],after[:,40:])
    assert before[40,30]==-1 and after[40,30]==0
    assert after[40,25]==-1 and after[55,35]==-1
    # Cover .18 rad/s over the 1.7 s DWB rollout in both directions.
    for wz in (-.306,.306):
        assert turn_is_free(after,wz)
    obstacle=before.copy();obstacle[47,32]=100
    blocked=StartupRearClearance().apply(obstacle,.05,(-2.,-2.),(0.,0.,0.),0.,10.)
    assert blocked[47,32]==100
    assert not turn_is_free(blocked,-.18)


def test_patch_is_fixed_at_start_and_never_rearms_after_departure():
    g=np.full((100,100),-1);p=StartupRearClearance()
    first=p.apply(g,.05,(-2.5,-2.5),(0,0,0),0.,0.)
    moved=p.apply(g,.05,(-2.5,-2.5),(.2,.1,.6),.3,10.)
    assert np.array_equal(first,moved)
    assert np.array_equal(p.apply(g,.05,(-2.5,-2.5),(.61,0,0),.61,11.),g)
    assert np.array_equal(p.apply(g,.05,(-2.5,-2.5),(0,0,0),.7,12.),g)
    assert p.finished and p.cells==[]


def test_expiry_restores_unknown_and_disable_does_not_change_map():
    g=np.full((40,40),-1);p=StartupRearClearance()
    assert np.any(p.apply(g,.05,(-1,-1),(0,0,0),0.,10.)==0)
    assert p.apply(g,.05,(-1,-1),(0,0,0),0.,100.) is g
    assert p.finished
    assert StartupRearClearance(0).apply(g,.05,(-1,-1),(0,0,0),0.,0.) is g


def test_patch_uses_starting_heading_and_keeps_raw_map_unchanged():
    g=np.full((80,80),-1);p=StartupRearClearance()
    out=p.apply(g,.05,(-2,-2),(0,0,math.pi/2),0.,0.)
    assert out[30,40]==0 and out[49,40]==-1
    assert np.all(g==-1)


@pytest.mark.parametrize('extension',[-.1,float('nan'),float('inf'),.3])
def test_invalid_extension_is_rejected(extension):
    with pytest.raises(ValueError):StartupRearClearance(extension)
