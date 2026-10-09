import json
from types import SimpleNamespace as NS
import numpy as np
import cv2
from board_approach.obstacle_evidence import save_trigger,save_after_stop
from board_approach.diagnostics import save_failure


def test_exact_trigger_mask_and_cloud_are_saved(tmp_path):
    points=np.array([[.3,0,0],[.95,0,0],[.5,.49,0],[.15,0,0],[.5,.48,0],[.31,.02,0],[.32,.04,0]])
    points=np.vstack([points,[[.33+i*.004,.01,0] for i in range(7)]])
    save_trigger(tmp_path,dict(points_base=points,metadata={'error':'Obstacle in forward stopping envelope','scan_timestamp_ns':123}))
    data=np.load(tmp_path/'obstacle_trigger.npz')
    assert data['trigger_mask'].tolist()==[True,False,False,False,False,True,True]+[True]*7
    assert np.array_equal(data['points_base'],points)
    assert cv2.imread(str(tmp_path/'obstacle_trigger.png')).shape==(800,800,3)
    assert 'FIELDS x y z marked' in (tmp_path/'obstacle_trigger.pcd').read_text()
    assert json.loads((tmp_path/'obstacle_trigger.json').read_text())['scan_timestamp_ns']==123


def test_after_stop_saves_synchronized_rgb_and_full_cloud(tmp_path):
    points=np.array([[.5,0,1.],[1.,0,1.],[2.,0,1.]])
    image=np.zeros((100,100,3),np.uint8)
    k=np.array([[40.,0,50],[0,40.,50],[0,0,1.]])
    node=NS(pump=lambda t:None,snapshot=lambda **kw:(image,None,k,np.eye(4),456),
            last_lidar_frame=dict(points_base=points,rings=np.arange(3),distortion=np.zeros(5),metadata={}))
    meta=save_after_stop(tmp_path,node,np.eye(4))
    assert meta['timestamp_ns']==456 and 'not_trigger_instant' in meta['source']
    assert np.array_equal(np.load(tmp_path/'after_stop_lidar.npz')['points_base'],points)
    assert (tmp_path/'after_stop_lidar.pcd').exists()
    assert cv2.imread(str(tmp_path/'after_stop_lidar_overlay.png')).shape==image.shape


def test_capture_failure_keeps_exact_scan_and_reports_missing_sync(tmp_path):
    stage=tmp_path/'stage';stage.mkdir()
    save_trigger(stage,dict(points_base=np.array([[.5,0,0]]),metadata={'error':'Obstacle in forward stopping envelope'}))
    def fail(*args):raise RuntimeError('camera unavailable')
    save_failure(tmp_path,stage,'Obstacle in forward stopping envelope',NS(data={}),camera_transform=fail)
    report=json.loads((tmp_path/'failure/failure.json').read_text())
    assert any('camera unavailable' in s for s in report['save_errors'])
    assert (tmp_path/'failure/obstacle_trigger.pcd').exists()
