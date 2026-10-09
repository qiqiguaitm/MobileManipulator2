"""Observation retry tests; fake sensors only, never start a motor process."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
pytest.importorskip('rclpy')
import board_approach.runner as runner
from board_approach.perception import BoardUnavailable, DetectionError


@pytest.fixture(autouse=True)
def mocked_surface_fit(monkeypatch):
    # These tests exercise observation retries; actual plane fitting has separate tests.
    monkeypatch.setattr(runner, "surface_observation", lambda observation, *args: observation)


def test_missing_appearance_frame_does_not_abort_retry(monkeypatch, tmp_path):
    frames = []
    def snapshot():
        frames.append(len(frames))
        return np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)), np.eye(3), np.eye(4), len(frames)
    def unavailable(*args):
        raise BoardUnavailable('not visible in this frame')
    def appearance(*args):
        if len(frames) == 4:
            raise BoardUnavailable('one missed appearance frame')
        return NS(source='appearance_depth', polygon=np.float32([[1,1],[6,1],[6,6],[1,6]]), ids=[])
    monkeypatch.setattr(runner, 'detect_board', unavailable)
    monkeypatch.setattr(runner, 'AppearanceDetector', lambda **kwargs: NS(detect=appearance))
    monkeypatch.setattr(runner, 'camera_transform', lambda node: np.eye(4))
    monkeypatch.setattr(runner, 'surface_target_geometry', lambda *args: dict(
        surface_normal_map=[0.,-1.,0.], board_map=[0.,2.,0.], goal_map=[0.,1.,0.], goal_yaw=0.,
        distance=2., lateral=0., heading_error=0., identity_verified=False, detection_source=args[0].source))
    node = NS(verify_stationary_since=lambda pose:None, snapshot=snapshot, data={'scan':NS(header=NS(frame_id='laser'))},
              transform=lambda *args: np.eye(4))
    result = runner.observe(node, tmp_path, 1.)
    assert len(frames) == 7
    assert result['identity_verified'] is False


def test_ambiguous_appearance_is_fatal(monkeypatch, tmp_path):
    def missing(*args):
        raise BoardUnavailable('no IDs')
    def ambiguous(*args):
        raise DetectionError('Multiple appearance/depth candidates; target ambiguous')
    monkeypatch.setattr(runner, 'detect_board', missing)
    monkeypatch.setattr(runner, 'AppearanceDetector', lambda **kwargs: NS(detect=ambiguous))
    node = NS(snapshot=lambda: (np.zeros((8,8,3),np.uint8), np.ones((8,8)), np.eye(3), np.eye(4), 0))
    with pytest.raises(DetectionError, match='ambiguous'):
        runner.observe(node, tmp_path, 1.)


@pytest.mark.parametrize('required,frames_expected',[(1,2),(2,3),(3,4)])
@pytest.mark.parametrize('error_type',[BoardUnavailable, runner.DetectorFormatError])
def test_no_aruco_observation_count(monkeypatch, tmp_path, required, frames_expected, error_type):
    frames = []
    def snapshot():
        frames.append(len(frames))
        return np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)), np.eye(3), np.eye(4), len(frames)
    def unavailable(*args):
        raise AssertionError('ArUco decoder called in disabled mode')
    def appearance(*args):
        if len(frames) == 1:
            raise error_type('one missed appearance frame')
        return NS(source='object_depth', polygon=np.float32([[1,1],[6,1],[6,6],[1,6]]), ids=[])
    monkeypatch.setattr(runner, 'detect_board', unavailable)
    monkeypatch.setattr(runner, 'AppearanceDetector', lambda **kwargs: NS(detect=unavailable))
    monkeypatch.setattr(runner, 'camera_transform', lambda node: np.eye(4))
    monkeypatch.setattr(runner, 'surface_target_geometry', lambda *args: dict(
        surface_normal_map=[0.,-1.,0.], board_map=[0.,2.,0.], goal_map=[0.,1.,0.], goal_yaw=0.,
        distance=2., lateral=0., heading_error=0., identity_verified=False, detection_source=args[0].source))
    node = NS(verify_stationary_since=lambda pose:None, snapshot=snapshot, data={'scan':NS(header=NS(frame_id='laser'))},
              transform=lambda *args: np.eye(4))
    result = runner.observe(node, tmp_path, 1., use_aruco=False, object_detector=NS(detect=appearance), required_observations=required)
    assert len(frames) == frames_expected
    assert result["stationary_observations"] == required
    assert result['identity_verified'] is False
    assert result['use_aruco'] is False



def test_aruco_mode_never_calls_object_model(monkeypatch, tmp_path):
    frames=[]
    def snapshot():
        frames.append(1)
        return np.zeros((8,8,3),np.uint8),np.ones((8,8)),np.eye(3),np.eye(4),len(frames)
    obs=NS(source='markers',polygon=np.float32([[1,1],[6,1],[6,6],[1,6]]),ids=[1,8,12,15])
    monkeypatch.setattr(runner,'detect_board',lambda *args:obs)
    monkeypatch.setattr(runner,'camera_transform',lambda node:np.eye(4))
    monkeypatch.setattr(runner,'surface_target_geometry',lambda *args:dict(
        surface_normal_map=[0.,-1.,0.], board_map=[0.,2.,0.],goal_map=[0.,1.,0.],goal_yaw=0.,distance=2.,lateral=0.,
        heading_error=0.,identity_verified=True,detection_source='markers'))
    def forbidden(*args):
        raise AssertionError('Model called in ArUco mode')
    node=NS(snapshot=snapshot,verify_stationary_since=lambda pose:None,
            data={'scan':NS(header=NS(frame_id='laser'))},transform=lambda *args:np.eye(4))
    t=runner.observe(node,tmp_path,1.,use_aruco=True,object_detector=NS(detect=forbidden))
    assert t['use_aruco'] and t['identity_verified']
    assert len(frames)==3


def test_coarse_allows_far_heading_spread_but_transition_demands_precision(monkeypatch,tmp_path):
    angles=iter([0.,.17,0.])
    observation=NS(source='object_depth',polygon=np.float32([[1,1],[6,1],[6,6],[1,6]]),ids=[])
    def geometry(*args):
        yaw=next(angles)
        return dict(board_map=[3.4,0.,0.],goal_map=[2.4,0.,0.],goal_yaw=yaw,
                    surface_normal_map=[-np.cos(yaw),-np.sin(yaw),0.],distance=3.4,
                    lateral=0.,heading_error=yaw,identity_verified=False,
                    detection_source='object_depth',plane_fit={'depth_m':3.4})
    monkeypatch.setattr(runner,'surface_target_geometry',geometry)
    monkeypatch.setattr(runner,'camera_transform',lambda node:np.eye(4))
    node=NS(snapshot=lambda:(np.zeros((8,8,3),np.uint8),np.ones((8,8)),np.eye(3),np.eye(4),1),
            verify_stationary_since=lambda pose:None,data={'scan':NS(header=NS(frame_id='laser'))},
            transform=lambda *args:np.eye(4))
    detector=NS(detect=lambda *args:observation)
    result=runner.observe(node,tmp_path,1.,False,detector,precise=False)
    assert result['coarse_orientation'] and result['heading_span_rad']==pytest.approx(.17)
    angles=iter([0.,.17,0.])
    with pytest.raises(RuntimeError,match='unstable'):
        runner.observe(node,tmp_path,1.,False,detector,precise=True)


def test_one_bad_surface_frame_is_saved_and_retried(monkeypatch,tmp_path):
    count={'n':0}
    observation=NS(source='object_depth',polygon=np.float32([[1,1],[6,1],[6,6],[1,6]]),ids=[])
    def snapshot():
        count['n']+=1
        return np.zeros((8,8,3),np.uint8),np.ones((8,8)),np.eye(3),np.eye(4),count['n']
    def surface(obs,*args):
        if count['n']==3:
            raise DetectionError('normal ambiguous in this frame')
        return obs
    monkeypatch.setattr(runner,'surface_observation',surface)
    monkeypatch.setattr(runner,'camera_transform',lambda node:np.eye(4))
    monkeypatch.setattr(runner,'surface_target_geometry',lambda *args:dict(
        board_map=[3.,0.,0.],goal_map=[2.,0.,0.],goal_yaw=0.,surface_normal_map=[-1.,0.,0.],
        distance=3.,lateral=0.,heading_error=0.,identity_verified=False,detection_source='object_depth',
        plane_fit={'depth_m':3.4}))
    node=NS(snapshot=snapshot,verify_stationary_since=lambda pose:None,
            data={'scan':NS(header=NS(frame_id='laser'))},transform=lambda *args:np.eye(4))
    result=runner.observe(node,tmp_path,1.,False,NS(detect=lambda *args:observation))
    assert count['n']==4 and result['distance']==3.
    import json
    assert json.loads((tmp_path/'frame_2_error.json').read_text())['retryable']
    assert (tmp_path/'rgb_2.png').exists() and (tmp_path/'detection_2.png').exists()
    # Persistent bad geometry still fails after the bounded sampling budget.
    count['n']=0
    def always_bad(*args):raise DetectionError('normal persistently ambiguous')
    monkeypatch.setattr(runner,'surface_observation',always_bad)
    with pytest.raises(DetectionError,match='Fewer than 3'):
        runner.observe(node,tmp_path,1.,False,NS(detect=lambda *args:observation))
    assert count['n']==10


def test_lidar_static_check_ignores_vertical_crop_but_not_horizontal_target_shift(monkeypatch,tmp_path):
    # Reproduce live pillar detections: height changes 16 cm, horizontal centre < 3 cm.
    centers=iter([[.455,2.482,.371],[.493,2.490,.528],[.464,2.488,.495]])
    obs=NS(source='object_lidar',polygon=np.float32([[1,1],[6,1],[6,6],[1,6]]),ids=[])
    monkeypatch.setattr(runner,'measure_lidar',lambda *args:obs)
    monkeypatch.setattr(runner,'camera_transform',lambda node:np.eye(4))
    def geometry(*args):
        c=next(centers)
        return dict(board_map=c,goal_map=[c[0],c[1]-1,0],surface_normal_map=[0.,-1.,0.],
                    goal_yaw=0.,distance=2.66,lateral=0.,heading_error=0.,identity_verified=False,
                    detection_source='object_lidar',plane_fit={'depth_m':3.4})
    monkeypatch.setattr(runner,'surface_target_geometry',geometry)
    def forbidden(*args):raise AssertionError('Camera depth was used in LiDAR mode')
    node=NS(snapshot=lambda:(np.zeros((8,8,3),np.uint8),None,np.eye(3),np.eye(4),1),
            last_lidar_frame=dict(points_base=np.zeros((3,3)),rings=np.zeros(3),distortion=np.zeros(5),metadata={}),
            verify_stationary_since=lambda pose:None,data={'scan':NS(header=NS(frame_id='rslidar'))},
            transform=lambda *args:np.eye(4))
    detector=NS(detect=forbidden,detect_region=lambda image:obs)
    result=runner.observe(node,tmp_path,1.,False,detector,precise=True,geometry_source='lidar')
    assert result['center_check_axes']=='map_xy' and result['center_spread_m']<.03
    centers=iter([[.455,2.482,.371],[.693,2.490,.528],[.464,2.488,.495]])
    with pytest.raises(RuntimeError,match='unstable'):
        runner.observe(node,tmp_path,1.,False,detector,precise=True,geometry_source='lidar')


def test_stationary_snapshot_recovers_without_relaxing_validation(monkeypatch,tmp_path):
    clock=[0.];attempts=[0]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    def snapshot():
        attempts[0]+=1;clock[0]+=.5
        if attempts[0]<3:raise RuntimeError('No fresh RGB/LiDAR pair with a matching localization pose')
        return 'valid'
    node=NS(snapshot=snapshot,pump=lambda dt:None)
    assert runner.stationary_snapshot(node,tmp_path)=='valid'
    assert len((tmp_path/'snapshot_retries.jsonl').read_text().splitlines())==2


def test_stationary_snapshot_stops_at_budget(monkeypatch,tmp_path):
    clock=[0.]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    def snapshot():
        clock[0]+=.5
        raise RuntimeError('No fresh RGB/LiDAR pair with a matching localization pose')
    with pytest.raises(runner.InputFreshnessError,match='budget exhausted'):
        runner.stationary_snapshot(NS(snapshot=snapshot,pump=lambda dt:None),tmp_path)
    assert clock[0]==3.


@pytest.mark.parametrize('message', ['Localization unreliable; set initial pose in RViz first',
    'Robot must be stationary while observing board',
    'No synchronized RGB/LiDAR/odometry observation: Odometry jump or excessive speed during LiDAR capture'])
def test_stationary_snapshot_does_not_retry_unsafe_state(tmp_path,message):
    def snapshot():raise RuntimeError(message)
    with pytest.raises(RuntimeError,match=message):
        runner.stationary_snapshot(NS(snapshot=snapshot),tmp_path)
    assert not (tmp_path/'snapshot_retries.jsonl').exists()
