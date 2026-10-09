"""Render/export obstacle evidence after motor cleanup, never in the control loop."""
import json
from pathlib import Path
import cv2
import numpy as np


def masks(points, bounds=None):
    from .safety import forward_bounds
    bounds=forward_bounds() if bounds is None else bounds
    p=np.asarray(points)
    return ((p[:,0]>bounds["x_min"])&(p[:,0]<bounds["x_max"])&(np.abs(p[:,1])<bounds["abs_y_max"]),
            np.linalg.norm(p[:,:2],axis=1)<.65)


def pcd(path, points, marked):
    points=np.asarray(points)
    header=('VERSION .7\nFIELDS x y z marked\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n'
            f'WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii')
    np.savetxt(str(path),np.column_stack((points,np.asarray(marked,int))),
               fmt=['%.6f','%.6f','%.6f','%d'],header=header,comments='')


def topdown(path, points, marked, title, bounds=None):
    from .safety import forward_bounds
    bounds=forward_bounds() if bounds is None else bounds
    canvas=np.full((800,800,3),25,np.uint8)
    def pixel(p): return tuple(np.rint([400-160*p[1],640-160*p[0]]).astype(int))
    for value in range(-1,4):
        cv2.line(canvas,pixel([value,-2]),pixel([value,2]),(65,65,65),1)
    cv2.rectangle(canvas,pixel([bounds["x_max"],bounds["abs_y_max"]]),pixel([bounds["x_min"],-bounds["abs_y_max"]]),(0,220,255),2)
    cv2.circle(canvas,pixel([0,0]),round(.65*160),(200,80,200),1)
    for point,hit in zip(points,marked):
        uv=pixel(point)
        if 0<=uv[0]<800 and 0<=uv[1]<800:cv2.circle(canvas,uv,3 if hit else 1,(0,0,255) if hit else (170,170,170),-1)
    cv2.arrowedLine(canvas,pixel([0,0]),pixel([.3,0]),(0,255,0),3)
    cv2.putText(canvas,title,(15,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.putText(canvas,'Front up; red: marked points; yellow: forward envelope',(15,54),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    if not cv2.imwrite(str(path),canvas):raise OSError('Could not save obstacle plot')


def save_trigger(directory, evidence):
    directory=Path(directory);points=evidence['points_base']
    from .safety import forward_bounds
    bounds=evidence["metadata"].get("forward_bounds_m",forward_bounds())
    front,turn=masks(points,bounds)
    from .safety import forward_obstacle_evidence,turning_obstacle_evidence
    _,forward_trigger,thresholds=forward_obstacle_evidence(points,bounds)
    _,turn_trigger,turn_thresholds=turning_obstacle_evidence(points)
    marked=forward_trigger if 'forward' in evidence['metadata']['error'] else turn_trigger
    np.savez_compressed(directory/'obstacle_trigger.npz',points_base=points,
                        forward_mask=front,turning_mask=turn,trigger_mask=marked)
    pcd(directory/'obstacle_trigger.pcd',points,marked)
    topdown(directory/'obstacle_trigger.png',points,marked,'EXACT guard scan: 2D LiDAR projection, not full 3D cloud',bounds)
    metadata=dict(evidence['metadata'],marked_points=int(marked.sum()),
                  forward_thresholds=thresholds,turning_thresholds=turn_thresholds,forward_bounds_m=bounds,turning_radius_m=.65)
    (directory/'obstacle_trigger.json').write_text(json.dumps(metadata,indent=2))


def save_after_stop(directory, node, base_from_camera):
    """Fresh synchronized RGB/3D LiDAR AFTER stop, labelled separately from trigger."""
    from .lidar_geometry import project,camera_ray_mask
    directory=Path(directory)
    node.pump(.3)
    image,depth,k,pose,timestamp=node.snapshot(moving=True)
    frame=node.last_lidar_frame
    points=np.asarray(frame['points_base']);finite=np.isfinite(points).all(1)
    points=points[finite];rings=np.asarray(frame['rings'])[finite]
    from .safety import forward_bounds
    bounds=forward_bounds()
    trigger=directory/"obstacle_trigger.json"
    if trigger.exists():bounds=json.loads(trigger.read_text()).get("forward_bounds_m",bounds)
    front,_=masks(points,bounds)
    np.savez_compressed(directory/'after_stop_lidar.npz',points_base=points,rings=rings,
                        camera_matrix=k,distortion=frame['distortion'],map_from_base=pose,
                        base_from_camera=base_from_camera,forward_mask=front)
    pcd(directory/'after_stop_lidar.pcd',points,front)
    topdown(directory/'after_stop_lidar_top.png',points,front,'AFTER STOP: full 3D cloud; red: forward XY region (all heights)',bounds)
    if not cv2.imwrite(str(directory/'after_stop_camera.png'),image):raise OSError('Could not save after-stop RGB')
    transform=np.linalg.inv(base_from_camera)
    camera=points@transform[:3,:3].T+transform[:3,3]
    visible=(camera[:,2]>.1)&camera_ray_mask(camera,k,frame['distortion'],image.shape)
    overlay=image.copy()
    for pixel,marked in zip(project(camera[visible],k,frame['distortion']) if visible.any() else [],front[visible]):
        if not np.isfinite(pixel).all():continue
        x,y=np.rint(pixel).astype(int)
        if 0<=x<image.shape[1] and 0<=y<image.shape[0]:cv2.circle(overlay,(x,y),2,(0,0,255) if marked else (0,220,0),-1)
    cv2.putText(overlay,'AFTER STOP synchronized RGB + LiDAR; not trigger instant',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
    if not cv2.imwrite(str(directory/'after_stop_lidar_overlay.png'),overlay):raise OSError('Could not save LiDAR overlay')
    metadata=dict(source='after_stop_synchronized_capture_not_trigger_instant',timestamp_ns=timestamp,
                  frame='base_link',point_count=len(points),lidar=frame['metadata'])
    (directory/'after_stop_capture.json').write_text(json.dumps(metadata,indent=2))
    return metadata
