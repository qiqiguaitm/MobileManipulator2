"""Export paired HDL/Nav2 maps exclusively from finalized SC-PGO keyframes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import yaml


def validate_source(source):
    from pypcd4 import PointCloud
    source=Path(source).resolve()
    poses=np.loadtxt(source/'optimized_poses.txt',ndmin=2)
    scans=sorted((source/'Scans').glob('*.pcd'))
    if poses.shape[1]!=12 or len(poses)<2 or not np.isfinite(poses).all():
        raise ValueError('Need at least two finite optimized 3x4 poses')
    if len(scans)!=len(poses):raise ValueError(f'Scan/pose count mismatch: {len(scans)} vs {len(poses)}; refusing silent truncation')
    digest=hashlib.sha256((source/'optimized_poses.txt').read_bytes())
    for i,(scan,pose) in enumerate(zip(scans,poses)):
        if scan.stem!=f'{i:06d}':raise ValueError('Scan indices must start at 000000 with no gaps')
        matrix=pose.reshape(3,4)[:,:3]
        if not np.allclose(matrix.T@matrix,np.eye(3),atol=.002) or abs(np.linalg.det(matrix)-1)>.002:
            raise ValueError('Invalid optimized rotation')
        pc=PointCloud.from_path(str(scan));points=pc.numpy()[:,:3]
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f'Invalid point cloud: {scan.name}')
        digest.update(scan.name.encode());digest.update(hashlib.sha256(scan.read_bytes()).digest())
    return dict(scans=len(scans),source_sha256=digest.hexdigest())


def export_map(source,destination,project_root):
    import cv2
    from pypcd4 import PointCloud
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(f'Will not overwrite {destination}')
    evidence=validate_source(source)
    tools=Path(project_root)/'src/slam/scripts/map_tools'
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.navigation-export-',dir=destination.parent))
    try:
        (stage/'Scans').symlink_to(source/'Scans',target_is_directory=True)
        shutil.copy2(source/'optimized_poses.txt',stage/'optimized_poses.txt')
        commands=[['merge_scans.py',str(stage),'GlobalMap','0.02'],
                  ['downsample_pcd.py',str(stage/'GlobalMap.pcd'),str(stage/'GlobalMap_hdl.pcd'),'0.1'],
                  ['pcd_to_2dmap.py',str(stage/'GlobalMap.pcd'),str(stage),'0.05']]
        with (stage/'export.log').open('w') as log:
            for command in commands:
                subprocess.run([sys.executable,str(tools/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        # Verify source remained immutable during processing.
        if validate_source(source)!=evidence:raise ValueError('SC-PGO source changed during export; stop mapping first')
        for name in ('GlobalMap.pcd','GlobalMap_hdl.pcd'):
            points=PointCloud.from_path(str(stage/name)).numpy()[:,:3]
            if len(points)<10 or not np.isfinite(points).all():raise ValueError('Exported cloud is empty or invalid')
        meta=yaml.safe_load((stage/'map.yaml').read_text())
        if meta.get('image')!='map.pgm' or meta.get('resolution')!=.05 or len(meta.get('origin',[]))!=3:
            raise ValueError('Unexpected map metadata')
        grid=cv2.imread(str(stage/'map.pgm'),cv2.IMREAD_GRAYSCALE)
        if grid is None or not np.any(grid==0) or not np.any(grid==254):raise ValueError('Map lacks obstacles or observed free space')
        if not set(np.unique(grid)).issubset({0,205,254}):raise ValueError('Unexpected occupancy values')
        cv2.imwrite(str(stage/'map_preview.png'),grid)
        manifest=dict(schema=1,compatible_with='slam/hdl_navigation_launch.py map_dir',
          source=str(source),source_frame='SC-PGO camera_init',navigation_frame='map',
          frame_mapping='identity: both 2D and 3D derived from the same optimized cloud',
          unknown_policy='preserve 205; no unknown-to-free fill',resolution=.05,
          height_band_m=[.05,1.5],height_reference='SC-PGO origin, unchanged from existing conversion',
          **evidence)
        manifest['files']={name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in ('GlobalMap.pcd','GlobalMap_hdl.pcd','map.yaml','map.pgm')}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (stage/'Scans').unlink();(stage/'optimized_poses.txt').unlink()
        os.rename(stage,destination)
        return dict(ready=True,directory=str(destination),**evidence)
    finally:
        if stage.exists():shutil.rmtree(stage)


def finalize_run(run,project_root,graceful=True):
    run=Path(run);result=run/'result.json'
    report=json.loads(result.read_text()) if result.exists() else dict(status='fault',reason='explorer_report_missing')
    try:
        if not graceful:raise RuntimeError('Mapping shutdown required force; final optimized poses unverified')
        sources=[p for p in (run/'sc_pgo').iterdir() if p.is_dir() and not p.is_symlink() and (p/'Scans').is_dir()]
        if len(sources)!=1:raise RuntimeError('Expected exactly one SC-PGO recording in this run')
        report['navigation_export']=export_map(sources[0],run/'navigation',project_root)
    except Exception as error:
        report['navigation_export']=dict(ready=False,error=f'{type(error).__name__}: {error}')
    temp=result.with_suffix('.json.tmp');temp.write_text(json.dumps(report,indent=2));os.replace(temp,result)
    return report


def main():
    p=argparse.ArgumentParser(description='Export a stopped SC-PGO recording as paired HDL/Nav2 maps')
    p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--project-root',default=os.environ.get('MM2_PROJECT_ROOT'))
    a=p.parse_args()
    if not a.project_root:p.error('--project-root is required')
    print(json.dumps(export_map(a.source,a.destination,a.project_root)))

if __name__=='__main__':main()
