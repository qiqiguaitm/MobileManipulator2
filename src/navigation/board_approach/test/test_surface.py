import numpy as np
import pytest
from board_approach.object_center import object_observation
from board_approach.surface import surface_observation, surface_target_geometry, approach_goal
from board_approach.perception import arrived, DetectionError


def test_oblique_depth_plane_yields_normal_offset_goal_not_camera_ray():
    k = np.array([[500.,0,160],[0,500.,120],[0,0,1.]])
    y,x = np.indices((240,320))
    normal = np.array([.3,0.,-1.]); normal /= np.linalg.norm(normal)
    depth = -2*normal[2]/(-(normal[0]*(x-160)/500+normal[2]))
    obs = surface_observation(object_observation(depth,k,[80,40,240,200],.9),depth,k)
    assert obs.normal == pytest.approx(normal, abs=.001)
    base = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,.6],[0,0,0,1.]])
    target = surface_target_geometry(obs,base,np.eye(4),1.)
    offset = np.array(target['goal_map'])[:2]-np.array(target['object_map'])[:2]
    assert offset == pytest.approx(np.array(target['surface_normal_map'])[:2])
    assert abs(target['goal_yaw']) > .2
    assert abs(target['goal_map'][1]) > .2
    assert not arrived(target,1.,require_aruco=False)


def test_axis_staging_does_not_cut_diagonally_toward_near_target():
    target = dict(board_map=[3.,1.,0.],surface_normal_map=[-1.,0.,0.],
                  goal_map=[2.,1.,0.],goal_yaw=0.,distance=3.,lateral=1.)
    goal = approach_goal(target,1.)
    assert goal['goal_map'] == [1.5,1.,0.]
    assert goal['goal_yaw'] == 0.
    assert goal['approach_phase'] == 'reach_surface_normal_axis'
    with pytest.raises(DetectionError,match='Too close'):
        approach_goal(dict(target,distance=1.14),1.)
    assert approach_goal(dict(target,distance=1.5,lateral=.03),1.)['goal_map'] == [2.,1.,0.]


def test_facing_center_is_not_sufficient_for_perpendicular_arrival():
    target = dict(distance=1.,lateral=.25,heading_error=0.,identity_verified=False,detection_source='object_depth')
    assert not arrived(target,1.,require_aruco=False)
    assert not arrived(dict(target,lateral=0.,heading_error=.25),1.,require_aruco=False)
    assert arrived(dict(target,lateral=0.),1.,require_aruco=False)


def test_missing_normal_cannot_fall_back_to_center_alignment():
    k=np.array([[500.,0,160],[0,500.,120],[0,0,1.]])
    obs=object_observation(np.full((240,320),2.),k,[80,40,240,200],.9)
    with pytest.raises(DetectionError,match='normal'):
        surface_target_geometry(obs,np.eye(4),np.eye(4),1.)


def test_distance_thresholds_are_bounded_and_keep_near_precision():
    from board_approach.perception import plane_thresholds
    assert plane_thresholds(1.) == pytest.approx((.02,.012))
    assert plane_thresholds(1.5) == pytest.approx((.02,.012))
    assert .02 < plane_thresholds(3.4)[0] < .05
    assert plane_thresholds(10.) == pytest.approx((.05,.03))
    with pytest.raises(DetectionError):
        plane_thresholds(float('nan'))


def test_far_noisy_plane_passes_but_nonplanar_depth_still_fails():
    from board_approach.perception import fit_board_plane
    k=np.array([[500.,0,160],[0,500.,120],[0,0,1.]])
    polygon=np.float32([[60,40],[260,40],[260,200],[60,200]])
    rng=np.random.default_rng(4)
    depth=3.4+rng.normal(0,.015,(240,320))
    obs=fit_board_plane(depth,k,polygon,[160,120],[])
    assert obs.normal @ np.array([0,0,-1]) > .99
    assert obs.plane_fit['inlier_tolerance_m'] > .035
    depth=rng.uniform(2.8,3.9,(240,320))
    with pytest.raises(DetectionError,match='consistent plane'):
        fit_board_plane(depth,k,polygon,[160,120],[])


def test_normal_limit_relaxes_only_at_distance_and_is_capped():
    from board_approach.surface import normal_angle_limit, average_surface_normals
    assert normal_angle_limit(1.) == 8.
    assert normal_angle_limit(1.5) == 8.
    assert 16 < normal_angle_limit(3.4) < 17
    assert normal_angle_limit(8.) == 20.
    def normal(angle):
        angle=np.deg2rad(angle)
        return np.array([np.sin(angle),0.,-np.cos(angle)])
    with pytest.raises(DetectionError,match='ambiguous'):
        average_surface_normals(normal(10),normal(13),normal(0),1.)
    mean, report=average_surface_normals(normal(10),normal(13),normal(0),3.4)
    assert mean == pytest.approx(normal(11.5))
    assert np.linalg.norm(mean) == pytest.approx(1.)
    assert report['normal_method'] == 'normalized_left_right_mean'
    with pytest.raises(DetectionError):
        average_surface_normals(normal(30),normal(0),normal(0),4.)
    with pytest.raises(DetectionError):
        average_surface_normals(normal(0),normal(180),normal(0),4.)
    with pytest.raises(DetectionError):
        average_surface_normals([float('nan'),0,1],normal(0),normal(0),3.)


def test_rough_heading_is_only_allowed_far_from_docking_zone():
    from board_approach.surface import observation_heading_limit
    far=[dict(plane_fit={'depth_m':3.4},distance=2.4)]*3
    assert np.deg2rad(10) < observation_heading_limit(far,1.) < np.deg2rad(20)
    near=[dict(plane_fit={'depth_m':1.7},distance=1.4)]*3
    assert observation_heading_limit(near,1.) == .12
    grazing=[dict(plane_fit={'depth_m':3.4},distance=1.5)]*3
    assert observation_heading_limit(grazing,1.) == .12
    assert observation_heading_limit(far,2.) == .12


@pytest.mark.parametrize('distance',[.41,.5,.59])
def test_half_meter_standoff_has_valid_near_geometry(distance):
    from types import SimpleNamespace as NS
    obs=NS(center=np.array([distance,0.,0.]),normal=np.array([-1.,0.,0.]),
           ids=[],rmse=.005,plane_fit={},source='object_lidar',score=None)
    target=surface_target_geometry(obs,np.eye(4),np.eye(4),.5)
    assert target['goal_map'][0]==pytest.approx(distance-.5)
    assert arrived(target,.5,require_aruco=False)


def test_precise_arrival_rejects_previous_loose_success():
    t=dict(distance=.934,lateral=.02,heading_error=np.deg2rad(.8),detection_source='object_lidar')
    assert arrived(t,.934,require_aruco=False)
    assert not arrived(dict(t,lateral=.085),.934,require_aruco=False)
    assert not arrived(dict(t,heading_error=np.deg2rad(1.45)),.934,require_aruco=False)
