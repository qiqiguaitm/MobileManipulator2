"""Metric target planes from time-aligned LiDAR, never from predicted depth."""
import json
from pathlib import Path
import cv2
import numpy as np
from .perception import DetectionError, Observation
from .surface import average_surface_normals
from .limits import LIDAR_TARGET_DEPTH_M, LIDAR_POINT_DEPTH_MARGIN_M


INLIER_DISTANCE_M = .025
MIN_INLIER_RATIO = .70
TARGET_REGION_SCALE = .50
# Central 50% ROI: retain a support floor plus independent spatial checks.
MIN_PLANE_POINTS = 25
MIN_HALF_PLANE_POINTS = 10
MIN_SCAN_RINGS = 2

def project(points, k, distortion):
    pixels, _ = cv2.projectPoints(np.asarray(points, np.float64).reshape(-1,1,3),
                                 np.zeros(3), np.zeros(3), k, distortion)
    return pixels.reshape(-1,2)


def camera_ray_mask(camera, k, distortion, image_shape=None):
    """Reject rays outside calibrated image FOV before distortion can fold them in."""
    h,w=image_shape[:2] if image_shape is not None else (2*k[1,2],2*k[0,2])
    x=np.linspace(0,w-1,64);y=np.linspace(0,h-1,64)
    edge=np.vstack([np.c_[x,np.zeros(64)],np.c_[x,np.full(64,h-1)],
                    np.c_[np.zeros(64),y],np.c_[np.full(64,w-1),y]])
    rays=cv2.undistortPoints(edge.reshape(-1,1,2),k,distortion).reshape(-1,2)
    if not np.isfinite(rays).all():raise DetectionError('Invalid camera FOV calibration')
    lo=rays.min(0);hi=rays.max(0);margin=(hi-lo)*.01
    normalized=np.full((len(camera),2),np.inf)
    front=np.isfinite(camera).all(1)&(camera[:,2]>.35)
    normalized[front]=camera[front,:2]/camera[front,2,None]
    return front & (normalized>=lo-margin).all(1) & (normalized<=hi+margin).all(1)


def recover_door_support(points, rings, eligible, excluded):
    """Supplement a sparse exterior plane; reflective door points cannot set it."""
    shell=eligible & ~excluded
    info=dict(door_support_added=0,exterior_points=int(shell.sum()))
    if shell.sum()>=MIN_PLANE_POINTS or shell.sum()<13:return shell,info
    try:
        center,normal,inliers,_=plane_fit(points[shell],min_points=13)
        if len(np.unique(rings[shell][inliers]))<MIN_SCAN_RINGS:return shell,info
    except DetectionError:return shell,info
    candidates=np.flatnonzero(eligible & excluded)
    residual=np.abs((points[candidates]-center)@normal)
    candidates=candidates[residual<INLIER_DISTANCE_M]
    # Door contributes no more than the exterior consensus and cannot dominate.
    candidates=candidates[:int(inliers.sum())]
    result=shell.copy();result[candidates]=True
    info['door_support_added']=len(candidates)
    return result,info


def plane_fit(points, min_points=MIN_PLANE_POINTS):
    """Robust vertical plane, with independent support/coverage validation."""
    if len(points) < min_points:
        raise DetectionError(f'LiDAR plane has too few points: {len(points)} < {min_points}')
    rng = np.random.default_rng(42)
    best = np.zeros(len(points), bool)
    for _ in range(240):
        a,b,c = points[rng.choice(len(points),3,replace=False)]
        normal = np.cross(b-a,c-a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if abs(normal[2]) > .25:
            continue
        mask = np.abs((points-a) @ normal) < INLIER_DISTANCE_M
        if mask.sum() > best.sum():
            best = mask
    if best.mean() < MIN_INLIER_RATIO or best.sum() < min_points:
        raise DetectionError(f'LiDAR surface mixed or not vertical: inliers={best.sum()}/{len(points)}')
    for _ in range(3):
        center = points[best].mean(0)
        _, _, v = np.linalg.svd(points[best]-center, full_matrices=False)
        best = np.abs((points-center)@v[-1]) < INLIER_DISTANCE_M
        if best.mean() < MIN_INLIER_RATIO or best.sum() < min_points:
            raise DetectionError('LiDAR plane lost consistent support during refinement')
    center = points[best].mean(0)
    _, s, v = np.linalg.svd(points[best]-center, full_matrices=False)
    normal = v[-1]
    if normal @ center > 0:
        normal = -normal
    residual = (points-center)@normal
    rmse = float(np.sqrt(np.mean(residual[best]**2)))
    spread = np.ptp((points[best]-center) @ v[:2].T, axis=0)
    # Reject one scan line or a tiny patch even if its residual is near zero.
    if abs(normal[2]) > .25 or min(spread) < .08 or s[1]/np.sqrt(best.sum()) < .015:
        raise DetectionError('LiDAR plane lacks two-dimensional support or is not vertical')
    if rmse > .015:
        raise DetectionError(f'LiDAR plane residual too large: {rmse:.4f} m')
    return center, normal, best, rmse


def visible_height(inner, centroid, normal, frame, k, base_from_camera, min_base_height=.15):
    """Project the full sensor beam envelope onto the fitted face, not its inliers.

    Retain the original full-ROI requirement if scan calibration is unavailable.
    This avoids treating missing target returns as a smaller sensor field of view.
    """
    lo,hi=inner.min(0),inner.max(0)
    height=float(hi[1]-lo[1])
    meta=frame['metadata']
    angles=np.asarray(meta.get('beam_elevations_deg',[]),float)
    transform=np.asarray(meta.get('base_from_lidar',[]),float)
    if (angles.shape!=(16,) or transform.shape!=(4,4)
            or not np.isfinite(angles).all() or not np.isfinite(transform).all()
            or np.ptp(angles)<20 or np.max(np.abs(angles))>40):
        return height
    origin=transform[:3,3]
    # Convert the four image corners to face points to obtain the azimuth range.
    uv=cv2.undistortPoints(np.asarray(inner,float).reshape(-1,1,2),k,frame['distortion']).reshape(-1,2)
    rays=np.c_[uv,np.ones(len(uv))]@base_from_camera[:3,:3].T
    denom=rays@normal
    if np.any(np.abs(denom)<.1): return height
    scales=((centroid-base_from_camera[:3,3])@normal)/denom
    corners=base_from_camera[:3,3]+rays*scales[:,None]
    lidar=(corners-origin)@transform[:3,:3]
    azimuth=np.unwrap(np.arctan2(lidar[:,1],lidar[:,0]))
    if np.ptp(azimuth)>np.pi/2: return height
    projected=[]
    camera_from_base=np.linalg.inv(base_from_camera)
    for az in np.linspace(azimuth.min(),azimuth.max(),17):
        elev=np.radians([angles.min(),angles.max()])
        directions=np.c_[np.cos(elev)*np.cos(az),np.cos(elev)*np.sin(az),np.sin(elev)]@transform[:3,:3].T
        denominator=directions@normal
        if np.any(np.abs(denominator)<.1):continue
        distance=((centroid-origin)@normal)/denominator
        if np.any(distance<=0):continue
        ends=origin+directions*distance[:,None]
        ends=ends[np.argsort(ends[:,2])]
        if min_base_height is not None:
            if ends[1,2]<=min_base_height:continue
            if ends[0,2]<min_base_height:
                ends[0]+=((min_base_height-ends[0,2])/(ends[1,2]-ends[0,2]))*(ends[1]-ends[0])
        cam=ends@camera_from_base[:3,:3].T+camera_from_base[:3,3]
        if np.any(cam[:,2]<=.35):continue
        projected.extend(project(cam,k,frame['distortion'])[:,1].tolist())
    if not projected:return height
    overlap=max(0.,min(hi[1],max(projected))-max(lo[1],min(projected)))
    return min(height,overlap) if overlap>1. else height


def plane_coverage(pixels, inliers, inner, centroid, normal, frame, k, transform, min_base_height=.15):
    full_span=np.ptp(inner,axis=0)
    span=np.ptp(pixels[inliers],axis=0)
    observable_height=visible_height(inner,centroid,normal,frame,k,transform,min_base_height)
    return span/np.array([full_span[0],observable_height]), dict(
        full_face_coverage=(span/full_span).tolist(),
        observable_height_px=observable_height,full_roi_height_px=float(full_span[1]))


def measure_lidar(region, frame, k, base_from_camera, image=None, prefix=None):
    """Region centre ray intersects the measured plane; average left/right normals.

    frame points are already deskewed into base_link at the image timestamp.
    All inputs are immutable snapshots; this function makes no ROS calls.
    """
    points = np.asarray(frame['points_base'], float)
    rings = np.asarray(frame['rings'])
    camera_from_base = np.linalg.inv(base_from_camera)
    camera = points @ camera_from_base[:3,:3].T + camera_from_base[:3,3]
    valid = np.isfinite(camera).all(1) & (camera[:,2] > .35) & (camera[:,2] < LIDAR_TARGET_DEPTH_M + LIDAR_POINT_DEPTH_MARGIN_M)
    ray_ok=camera_ray_mask(camera,k,frame['distortion'],image.shape if image is not None else None)
    fov_rejected=int((valid & ~ray_ok).sum())
    valid &= ray_ok
    points, camera, rings = points[valid], camera[valid], rings[valid]
    pixels = project(camera,k,frame['distortion']) if len(camera) else np.empty((0,2))
    polygon = np.asarray(region.polygon, np.float32)
    center_uv = np.asarray(region.uv, float)
    if (polygon.shape!=(4,2) or not np.isfinite(polygon).all() or
            not cv2.isContourConvex(polygon) or center_uv.shape!=(2,) or not np.isfinite(center_uv).all()):
        raise DetectionError('Invalid target polygon for LiDAR projection')
    if image is not None and (polygon.min()<1 or (polygon[:,0]>=image.shape[1]-1).any()
                              or ((polygon[:,1]>image.shape[0]-1).any() if getattr(region,'allow_bottom_crop',False) else (polygon[:,1]>=image.shape[0]-1).any())):
        raise DetectionError('LiDAR target region cropped by camera view')
    scale=getattr(region,'region_scale',None)
    scale=TARGET_REGION_SCALE if scale is None else scale
    if not np.isfinite(scale) or not 0<scale<=1:raise DetectionError('Invalid target region scale')
    inner = (polygon.mean(0) + scale*(polygon-polygon.mean(0))).astype(np.float32)
    selected = np.array([cv2.pointPolygonTest(inner,tuple(map(float,p)),False)>=0 for p in pixels], bool)
    min_base_height=getattr(region,'min_base_height',.15)
    if min_base_height is not None:
        if not np.isfinite(min_base_height):raise DetectionError('Invalid base height cutoff')
        selected &= points[:,2] > min_base_height
    # A refined washer face with a door mask uses semantic boundaries, not base_link z as ground.
    exclusions=getattr(region,'exclusion_polygons',())
    core_scale=getattr(region,'exclusion_core_scale',1.)
    if not np.isfinite(core_scale) or not 0<core_scale<=1:raise DetectionError('Invalid door core scale')
    exclusions=tuple((np.asarray(p).mean(0)+core_scale*(np.asarray(p)-np.asarray(p).mean(0))).astype(np.float32) for p in exclusions)
    excluded=np.zeros(len(points),bool)
    for exclusion in exclusions:
        exclusion=np.asarray(exclusion,np.float32)
        if exclusion.ndim!=2 or exclusion.shape[1]!=2 or len(exclusion)<3 or not np.isfinite(exclusion).all() or not cv2.isContourConvex(exclusion):
            raise DetectionError('Invalid exclusion polygon')
        excluded |= np.array([cv2.pointPolygonTest(exclusion,tuple(map(float,p)),False)>=0 for p in pixels],bool)
    door_excluded=int((selected & excluded).sum())
    eligible=selected.copy()
    selected &= ~excluded
    support_info=dict(door_support_added=0,exterior_points=int(selected.sum()))
    if getattr(region,'allow_door_support',False):
        selected,support_info=recover_door_support(points,rings,eligible,excluded)
    chosen = points[selected]
    evidence = dict(frame=frame['metadata'], geometry_source='lidar',
                    polygon=polygon.tolist(), inner_polygon=inner.tolist(), candidate_points=len(chosen),
                    region_scale=scale,min_base_height=min_base_height,excluded_points=door_excluded,fov_rejected_points=fov_rejected,
                    exclusion_polygons=[np.asarray(p).tolist() for p in exclusions],
                    min_plane_points=MIN_PLANE_POINTS, min_half_plane_points=MIN_HALF_PLANE_POINTS, min_scan_rings=MIN_SCAN_RINGS)
    evidence.update(door_core_scale=core_scale,**support_info)
    inliers = np.zeros(len(chosen), bool)
    observation = None
    try:
        centroid, full, inliers, rmse = plane_fit(chosen)
        chosen_pixels, chosen_rings = pixels[selected], rings[selected]
        if len(np.unique(chosen_rings[inliers])) < MIN_SCAN_RINGS:
            raise DetectionError(f'LiDAR plane needs at least {MIN_SCAN_RINGS} scan rings')
        lo,hi = inner.min(0),inner.max(0)
        coverage, coverage_info = plane_coverage(chosen_pixels,inliers,inner,centroid,full,frame,k,base_from_camera,min_base_height)
        evidence.update(coverage=coverage.tolist(),**coverage_info)
        if coverage[0]<.5 or coverage[1]<.35:
            raise DetectionError(f'LiDAR observable face coverage too small: horizontal={coverage[0]:.3f}, vertical={coverage[1]:.3f}; required 0.50/0.35')
        halves, half_reports = [], []
        # Fit each half independently, NOT filtered by the full-plane inlier mask.
        for half_mask in (chosen_pixels[:,0]<center_uv[0], chosen_pixels[:,0]>=center_uv[0]):
            hc,hn,hm,he = plane_fit(chosen[half_mask],min_points=MIN_HALF_PLANE_POINTS)
            if len(np.unique(chosen_rings[half_mask][hm])) < MIN_SCAN_RINGS:
                raise DetectionError('LiDAR half-plane needs multiple scan rings')
            halves.append(hn)
            half_reports.append(dict(points=int(half_mask.sum()),inliers=int(hm.sum()),rmse_m=he))
        distance = float(np.median(camera[selected][inliers,2]))
        normal, diagnostics = average_surface_normals(halves[0],halves[1],full,distance)
        # Recheck residual for the actually used averaged normal.
        residual = (chosen-centroid)@normal
        inliers = np.abs(residual)<INLIER_DISTANCE_M
        averaged_rmse = float(np.sqrt(np.mean(residual[inliers]**2))) if inliers.any() else float('inf')
        if inliers.sum()<MIN_PLANE_POINTS or inliers.mean()<MIN_INLIER_RATIO or averaged_rmse>.015 or abs(normal[2])>.25:
            raise DetectionError(f'Averaged LiDAR normal does not explain the measured plane: inliers={inliers.sum()}/{len(inliers)} (min {MIN_PLANE_POINTS}, ratio {inliers.mean():.3f}/{MIN_INLIER_RATIO:.3f}), rmse={averaged_rmse:.4f}/0.0150m, abs_normal_z={abs(normal[2]):.3f}/0.250')
        if support_info['door_support_added']:
            exterior=~excluded[selected]
            supported=exterior & inliers
            if supported.sum()<13 or supported.sum()<.8*exterior.sum() or len(np.unique(chosen_rings[supported]))<MIN_SCAN_RINGS:
                raise DetectionError('Supplemented plane lost exterior support')
        coverage, coverage_info = plane_coverage(chosen_pixels,inliers,inner,centroid,normal,frame,k,base_from_camera,min_base_height)
        evidence.update(coverage=coverage.tolist(),**coverage_info)
        if len(np.unique(chosen_rings[inliers]))<MIN_SCAN_RINGS or coverage[0]<.5 or coverage[1]<.35:
            raise DetectionError('Averaged LiDAR plane has insufficient scan coverage')
        ray = np.r_[cv2.undistortPoints(center_uv.reshape(1,1,2),k,frame['distortion'])[0,0],1.]
        normal_camera = base_from_camera[:3,:3].T@normal
        centroid_camera = (camera_from_base@np.r_[centroid,1])[:3]
        if abs(normal_camera@ray)<.3:
            raise DetectionError('LiDAR target viewed too obliquely')
        center = ray*((normal_camera@centroid_camera)/(normal_camera@ray))
        if not np.isfinite(center).all() or not .35<center[2]<=LIDAR_TARGET_DEPTH_M:
            raise DetectionError(f'LiDAR target centre outside valid approach range: depth={center[2]:.3f} m; allowed (0.35, {LIDAR_TARGET_DEPTH_M:.1f}] m')
        diagnostics.update(depth_m=float(center[2]),inlier_tolerance_m=INLIER_DISTANCE_M, min_inlier_ratio=MIN_INLIER_RATIO,rmse_limit_m=.015,
                           inlier_fraction=float(inliers.mean()),rmse_m=averaged_rmse,
                           points=len(chosen),inliers=int(inliers.sum()),coverage=coverage.tolist(),
                           ring_count=len(np.unique(chosen_rings[inliers])),half_plane_fits=half_reports,
                           geometry_source='lidar',sensor_alignment=frame['metadata'],**coverage_info)
        observation = Observation(center,normal_camera,region.ids,polygon,averaged_rmse,
                                  float(inliers.mean()),source='markers' if region.source=='markers' else 'object_lidar',
                                  score=region.score,plane_fit=diagnostics)
        evidence.update(plane_fit=diagnostics,center_camera=center.tolist(),normal_base=normal.tolist())
        return observation
    except Exception as error:
        evidence['error'] = str(error)
        raise
    finally:
        if prefix is not None:
            prefix = str(prefix)
            np.savez_compressed(prefix+'_lidar.npz',points_base=frame['points_base'],rings=frame['rings'],
                                distortion=frame['distortion'],projected_pixels=pixels,selected=selected,inliers=inliers,
                                projected_points_base=points,projected_rings=rings,excluded=excluded)
            Path(prefix+'_lidar.json').write_text(json.dumps(evidence,indent=2))
            if image is not None:
                canvas = image.copy()
                for uv in pixels:
                    if 0<=uv[0]<canvas.shape[1] and 0<=uv[1]<canvas.shape[0]:
                        cv2.circle(canvas,tuple(np.rint(uv).astype(int)),1,(150,150,150),-1)
                for uv in pixels[excluded]:
                    if 0<=uv[0]<canvas.shape[1] and 0<=uv[1]<canvas.shape[0]:
                        cv2.circle(canvas,tuple(np.rint(uv).astype(int)),2,(220,80,220),-1)
                for exclusion in exclusions:
                    cv2.polylines(canvas,[np.asarray(exclusion,np.int32)],True,(220,80,220),2)
                for uv,good in zip(pixels[selected],inliers):
                    cv2.circle(canvas,tuple(np.rint(uv).astype(int)),3,(30,230,30) if good else (30,30,240),-1)
                cv2.polylines(canvas,[polygon.astype(np.int32)],True,(0,220,255),2)
                cv2.polylines(canvas,[inner.astype(np.int32)],True,(255,255,255),1)
                if observation is not None:
                    from .alignment_view import save_alignment
                    center_base=(base_from_camera@np.r_[observation.center,1])[:3]
                    normal_base=base_from_camera[:3,:3]@observation.normal
                    alignment=save_alignment(prefix+'_alignment.png',center_base,normal_base,observation.plane_fit,
                                             points=frame['points_base'],fit_points=points[selected],
                                             inliers=inliers,excluded_points=points[excluded])
                    evidence['alignment']=alignment
                    Path(prefix+'_lidar.json').write_text(json.dumps(evidence,indent=2))
                    arrow=project(np.array([observation.center,observation.center+.4*observation.normal]),k,frame['distortion'])
                    a,b=np.rint(arrow).astype(int)
                    cv2.arrowedLine(canvas,tuple(a),tuple(b),(255,255,0),3,tipLength=.2)
                label = 'LiDAR validated plane' if observation is not None else 'LiDAR rejected - inspect JSON'
                cv2.rectangle(canvas,(0,0),(canvas.shape[1],44),(20,20,20),-1)
                cv2.putText(canvas,label+' | green=inlier red=rejected cyan=normal',(12,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                if observation is not None:
                    cv2.putText(canvas,f'Heading {alignment["heading_deg"]:+.1f} deg | lateral {alignment["lateral"]:+.3f} m | distance {alignment["distance"]:.3f} m',
                                (12,canvas.shape[0]-16),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                cv2.imwrite(prefix+'_lidar.png',canvas)
