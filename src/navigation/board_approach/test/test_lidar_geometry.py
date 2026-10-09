import numpy as np
import pytest
from board_approach.perception import DetectionError, arrived, target_geometry
from board_approach.object_center import object_region
from board_approach.lidar_geometry import plane_fit, measure_lidar


def scene(angle=.25, noise=.002):
    # Camera: x right, y down, z forward. Base: x forward, y left, z up.
    b_c=np.eye(4);b_c[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];b_c[2,3]=.7
    k=np.array([[400.,0,320.],[0,400.,240.],[0,0,1.]])
    yy,zz=np.meshgrid(np.linspace(-.7,.7,80),np.linspace(.25,1.15,12))
    xx=3+np.tan(angle)*yy
    points=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    points+=np.random.default_rng(7).normal(0,noise,points.shape)
    region=object_region((480,640,3),[245,180,395,300])
    frame=dict(points_base=points,rings=np.repeat(np.arange(12),80),distortion=np.zeros(5),metadata={})
    return region,frame,k,b_c


def test_plane_depth_and_normal_from_lidar_without_camera_depth(tmp_path):
    region,frame,k,transform=scene()
    obs=measure_lidar(region,frame,k,transform,np.zeros((480,640,3),np.uint8),tmp_path/'frame')
    expected=np.array([-np.cos(.25),np.sin(.25),0])
    assert transform[:3,:3]@obs.normal == pytest.approx(expected,abs=.01)
    assert obs.center[2] == pytest.approx(3.,abs=.02)
    assert obs.source=='object_lidar' and obs.plane_fit['normal_method']=='normalized_left_right_mean'
    assert obs.plane_fit['geometry_source']=='lidar'
    assert (tmp_path/'frame_lidar.png').exists() and (tmp_path/'frame_lidar.npz').exists()


def test_no_depth_fallback_when_lidar_missing_or_too_sparse(tmp_path):
    region,frame,k,transform=scene()
    frame['points_base']=frame['points_base'][:4];frame['rings']=frame['rings'][:4]
    with pytest.raises(DetectionError,match='too few'):
        measure_lidar(region,frame,k,transform,np.zeros((480,640,3),np.uint8),tmp_path/'bad')
    assert (tmp_path/'bad_lidar.png').exists() and (tmp_path/'bad_lidar.json').exists()


def test_single_scan_line_cannot_establish_a_plane():
    points=np.c_[np.full(100,3.),np.linspace(-.4,.4,100),np.full(100,.5)]
    with pytest.raises(DetectionError):plane_fit(points)


def test_mixed_faces_rejected_even_if_one_plane_can_be_fit():
    rng=np.random.default_rng(2)
    a=np.c_[np.full(100,2.),rng.uniform(-.5,.5,100),rng.uniform(.2,1.2,100)]
    b=a.copy();b[:,0]=3.
    with pytest.raises(DetectionError,match='mixed'):
        plane_fit(np.vstack([a,b]))


def test_ground_cannot_be_target_plane():
    x,y=np.meshgrid(np.linspace(1,3,20),np.linspace(-1,1,20))
    with pytest.raises(DetectionError):plane_fit(np.c_[x.ravel(),y.ravel(),np.zeros(x.size)])


def test_independent_halves_reject_two_different_surface_angles():
    region,frame,k,transform=scene(angle=0)
    points=frame['points_base'];points[:,0]=3+np.tan(.6)*np.abs(points[:,1])
    with pytest.raises(DetectionError):measure_lidar(region,frame,k,transform)


def test_lidar_target_arrival_keeps_aruco_requirement_optional():
    target=dict(detection_source='object_lidar',identity_verified=False,distance=1.,lateral=0.,heading_error=0.)
    assert arrived(target,1.,require_aruco=False)
    assert not arrived(target,1.,require_aruco=True)


@pytest.mark.parametrize('distance', [4.23,5.5,6.0,6.1])
def test_extended_lidar_depth_limit(distance):
    region,frame,k,transform=scene(angle=0,noise=0)
    # Scale about the camera to preserve image coverage and scan-line support.
    origin=transform[:3,3]
    frame['points_base']=origin+(frame['points_base']-origin)*(distance/3.)
    if distance<=6.:
        observation=measure_lidar(region,frame,k,transform)
        assert observation.center[2]==pytest.approx(distance,abs=1e-8)
        target=target_geometry(observation,transform,np.eye(4),1.)
        assert target['distance']==pytest.approx(distance,abs=1e-8)
    else:
        with pytest.raises(DetectionError,match=r'depth=6.100.*6.0'):
            measure_lidar(region,frame,k,transform)


def near_tall_scene():
    region,frame,k,transform=scene(angle=0,noise=0)
    transform[2,3]=.6
    y,z=np.meshgrid(np.linspace(-.3,.3,80),np.linspace(.2,.36,8))
    frame['points_base']=np.c_[np.ones(y.size),y.ravel(),z.ravel()]
    frame['rings']=np.repeat(np.arange(8),80)
    sensor=np.eye(4);sensor[2,3]=.1
    frame['metadata']=dict(base_from_lidar=sensor.tolist(),beam_elevations_deg=np.linspace(-15,15,16).tolist())
    region=object_region((480,640,3),[180,2,460,478])
    return region,frame,k,transform


def test_near_tall_face_uses_independent_sensor_envelope(monkeypatch):
    # Isolate sensor-envelope behavior using the original calibrated ROI fixture.
    monkeypatch.setattr("board_approach.lidar_geometry.TARGET_REGION_SCALE", .7)
    region,frame,k,transform=near_tall_scene()
    obs=measure_lidar(region,frame,k,transform)
    assert obs.plane_fit['full_face_coverage'][1]<.35
    assert obs.plane_fit['coverage'][1]>.35
    assert obs.center[2]==pytest.approx(1.)
    assert obs.plane_fit['ring_count']>=3


def test_without_complete_beam_calibration_original_coverage_remains(monkeypatch):
    # Isolate sensor-envelope behavior using the original calibrated ROI fixture.
    monkeypatch.setattr("board_approach.lidar_geometry.TARGET_REGION_SCALE", .7)
    region,frame,k,transform=near_tall_scene()
    frame['metadata']['beam_elevations_deg']=[-2,0,2]
    with pytest.raises(DetectionError,match='coverage too small'):
        measure_lidar(region,frame,k,transform)


def test_narrow_or_single_ring_support_cannot_pass_with_sensor_envelope(monkeypatch):
    # Isolate sensor-envelope behavior using the original calibrated ROI fixture.
    monkeypatch.setattr("board_approach.lidar_geometry.TARGET_REGION_SCALE", .7)
    region,frame,k,transform=near_tall_scene()
    frame['points_base'][:,2]=.3
    with pytest.raises(DetectionError):measure_lidar(region,frame,k,transform)
    region,frame,k,transform=near_tall_scene()
    frame['rings'][:]=0
    with pytest.raises(DetectionError,match='at least 2 scan rings'):
        measure_lidar(region,frame,k,transform)


def test_small_fragment_inside_larger_visible_face_is_still_rejected(monkeypatch):
    # Isolate sensor-envelope behavior using the original calibrated ROI fixture.
    monkeypatch.setattr("board_approach.lidar_geometry.TARGET_REGION_SCALE", .7)
    region,frame,k,transform=scene(angle=0,noise=0)
    sensor=np.eye(4);sensor[2,3]=.1
    frame['metadata']=dict(base_from_lidar=sensor.tolist(),beam_elevations_deg=np.linspace(-15,15,16).tolist())
    frame['points_base'][:,2]=.6+(frame['points_base'][:,2]-.7)*.2
    with pytest.raises(DetectionError,match='coverage too small'):
        measure_lidar(region,frame,k,transform)


def test_central_half_rejects_near_tall_face_when_crop_removes_support():
    region,frame,k,transform=near_tall_scene()
    with pytest.raises(DetectionError,match='two-dimensional support'):
        measure_lidar(region,frame,k,transform)


def test_sparse_plane_keeps_spatial_support_and_point_floor():
    y,z=np.meshgrid(np.linspace(-.2,.2,11),[.3,.5,.7])
    points=np.c_[np.full(y.size,3.),y.ravel(),z.ravel()]
    _,normal,inliers,rmse=plane_fit(points)
    assert inliers.sum()==33 and rmse<1e-8
    assert abs(normal[0])==pytest.approx(1.)
    with pytest.raises(DetectionError,match='too few points: 24 < 25'):
        plane_fit(points[:24])
    points[:,2]=.5
    with pytest.raises(DetectionError):
        plane_fit(points)


def test_two_spatially_separated_scan_lines_can_establish_target_plane():
    region,frame,k,transform=scene(angle=0,noise=0)
    keep=np.isin(frame['rings'],[4,7])
    frame['points_base']=frame['points_base'][keep]
    frame['rings']=frame['rings'][keep]
    obs=measure_lidar(region,frame,k,transform)
    assert obs.plane_fit['ring_count']==2
    assert obs.center[2]==pytest.approx(3.,abs=1e-8)
    assert transform[:3,:3]@obs.normal==pytest.approx([-1.,0.,0.],abs=1e-8)


def test_off_axis_ray_cannot_fold_into_target_via_distortion():
    from board_approach.lidar_geometry import camera_ray_mask,project
    k=np.array([[644.1759,0,640.3016],[0,643.2646,364.9872],[0,0,1.]])
    distortion=np.array([-.05550872,.06422378,-.000435956,.00026293,-.02034985])
    points=np.array([[2.619561,.072234,1.198671],[0.,0.,4.]])
    pixel=project(points,k,distortion)[0]
    assert 0<pixel[0]<1280 and 0<pixel[1]<720  # Polynomial folds this bad ray into image.
    assert camera_ray_mask(points,k,distortion,(720,1280,3)).tolist()==[False,True]


def test_excluded_door_points_never_enter_plane_fit(tmp_path):
    region,frame,k,transform=scene(angle=0,noise=0)
    region.region_scale=.9
    region.exclusion_polygons=(np.float32([[300,215],[340,215],[340,265],[300,265]]),)
    obs=measure_lidar(region,frame,k,transform,np.zeros((480,640,3),np.uint8),tmp_path/'door')
    evidence=np.load(tmp_path/'door_lidar.npz')
    assert evidence['excluded'].any()
    assert not (evidence['selected'] & evidence['excluded']).any()
    assert obs.center[2]==pytest.approx(3.,abs=1e-8)


def test_refined_face_keeps_low_rows_when_base_link_is_above_ground(tmp_path):
    region,frame,k,transform=scene(angle=0,noise=0)
    frame['points_base'][:,2]-=.7
    transform[2,3]-=.7  # Same image, only base_link vertical origin changes.
    region.min_base_height=None
    obs=measure_lidar(region,frame,k,transform,np.zeros((480,640,3),np.uint8),tmp_path/'low')
    f=np.load(tmp_path/'low_lidar.npz')
    assert (f['projected_points_base'][f['selected'],2]<0).any()
    assert obs.center[2]==pytest.approx(3.,abs=1e-8)


def test_door_support_needs_exterior_plane_and_rejects_reflections():
    from board_approach.lidar_geometry import recover_door_support
    y,z=np.meshgrid(np.linspace(-.2,.2,5),np.linspace(.2,.6,3))
    shell=np.c_[np.full(15,2.),y.ravel(),z.ravel()]
    door=shell.copy();door[:,1]+=.01
    ghost=door.copy();ghost[:,0]-=.3
    points=np.vstack([shell,door,ghost]);rings=np.tile(np.repeat([1,2,3],5),3)
    eligible=np.ones(45,bool);excluded=np.arange(45)>=15
    selected,info=recover_door_support(points,rings,eligible,excluded)
    assert info['door_support_added']==15 and selected.sum()==30
    assert not selected[30:].any()
    eligible[:5]=False
    selected,info=recover_door_support(points,rings,eligible,excluded)
    assert info['door_support_added']==0 and selected.sum()==10
