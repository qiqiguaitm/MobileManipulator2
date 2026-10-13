import json,os
from pathlib import Path
import numpy as np
import pytest
import cv2,yaml
from pypcd4 import PointCloud
from auto_explore.export_map import export_map,validate_source,finalize_run

ROOT=Path(os.environ.get('MM2_PROJECT_ROOT','/home/didi/junbin.zhang/MobileManipulator2'))

def pcd(path,points):
    a=np.asarray(points,dtype='<f4').reshape(-1,3)
    path.write_bytes((f'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(a)}\nHEIGHT 1\nPOINTS {len(a)}\nDATA binary\n').encode()+a.tobytes())

@pytest.fixture
def source(tmp_path):
    s=tmp_path/'recording';(s/'Scans').mkdir(parents=True)
    points=[[x,y,0.] for x in np.arange(1,3,.1) for y in np.arange(-1,1,.1)]
    points += [[3.,y,z] for y in np.arange(-1,1,.1) for z in (.2,.5,1.)]
    for i in range(2):pcd(s/'Scans'/f'{i:06d}.pcd',points)
    poses=np.tile(np.eye(4)[:3].reshape(1,12),(2,1));poses[:,3]=10.;poses[:,7]=20.
    np.savetxt(s/'optimized_poses.txt',poses)
    return s

def test_paired_export_uses_same_optimized_coordinates(source,tmp_path):
    destination=tmp_path/'navigation';result=export_map(source,destination,ROOT)
    assert result['ready'] and result['scans']==2
    pts=PointCloud.from_path(str(destination/'GlobalMap_hdl.pcd')).numpy()[:,:3]
    assert pts[:,0].min()>=11 and pts[:,1].min()>=19
    meta=yaml.safe_load((destination/'map.yaml').read_text());img=cv2.imread(str(destination/'map.pgm'),0)
    assert 205 in np.unique(img)
    x,y,z=13.,20.,.5
    u=int((x-meta['origin'][0])/meta['resolution']);v=img.shape[0]-1-int((y-meta['origin'][1])/meta['resolution'])
    assert np.min(img[v-1:v+2,u-1:u+2])==0
    assert not (destination/'Scans').exists()
    assert json.loads((destination/'manifest.json').read_text())['unknown_policy'].startswith('preserve')
    with pytest.raises(FileExistsError):export_map(source,destination,ROOT)

def test_mismatch_is_rejected_not_silently_truncated(source,tmp_path):
    (source/'Scans/000001.pcd').unlink()
    with pytest.raises(ValueError,match='mismatch'):export_map(source,tmp_path/'navigation',ROOT)
    assert not (tmp_path/'navigation').exists()

def test_corrupt_scan_is_not_skipped(source,tmp_path):
    pcd(source/'Scans/000001.pcd',[[float('nan'),0,0]])
    with pytest.raises(ValueError,match='Invalid point cloud'):validate_source(source)

def test_forced_shutdown_cannot_claim_compatible_map(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps(dict(status='complete')))
    report=finalize_run(tmp_path,ROOT,graceful=False)
    assert not report['navigation_export']['ready'] and not (tmp_path/'navigation').exists()
