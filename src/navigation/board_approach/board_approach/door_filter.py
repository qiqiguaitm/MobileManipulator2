"""Bounded near-door reflection candidates; retain raw scan on missing evidence."""
import cv2
import numpy as np

NEAR_DISTANCE = 1.5
MAX_MODEL_AGE = 6.0


def door_model(region, observation, k, distortion, base_from_camera, map_from_base, captured):
    exclusions = getattr(region, 'exclusion_polygons', ())
    if len(exclusions) != 1:
        return None
    pixels = np.asarray(exclusions[0], float)
    rays = np.column_stack((cv2.undistortPoints(pixels.reshape(-1,1,2), k, distortion).reshape(-1,2), np.ones(len(pixels))))
    normal = np.asarray(observation.normal, float)
    denom = rays @ normal
    if np.any(np.abs(denom) < .3):
        return None
    camera = rays * ((normal @ observation.center)/denom)[:,None]
    transform = map_from_base @ base_from_camera
    polygon = camera @ transform[:3,:3].T + transform[:3,3]
    normal = transform[:3,:3] @ normal
    normal /= np.linalg.norm(normal)
    if normal @ (map_from_base[:3,3]-polygon.mean(0)) < 0:
        normal = -normal
    return dict(polygon_map=polygon.tolist(), normal_map=normal.tolist(), captured_monotonic=captured)


def reflection_mask(points_base, model, map_from_base, phase, now):
    """Only the inner door prism, 8–40 cm in front of the measured face."""
    points = np.asarray(points_base, float)
    empty = np.zeros(len(points), bool)
    if phase not in ('continuous_approach','precise_docking') or not model:
        return empty
    try:
        age = now-float(model['captured_monotonic'])
        polygon = np.asarray(model['polygon_map'],float)
        normal = np.asarray(model['normal_map'],float)
        pose = np.asarray(map_from_base,float)
        if not 0 <= age <= MAX_MODEL_AGE or polygon.ndim!=2 or polygon.shape[1]!=3 or len(polygon)<8:
            return empty
        if not np.isfinite(polygon).all() or normal.shape!=(3,) or not np.isfinite(normal).all() or pose.shape!=(4,4) or not np.isfinite(pose).all():
            return empty
        if abs(np.linalg.norm(normal)-1)>.01:
            return empty
        center = polygon.mean(0)
        distance = (pose[:3,3]-center) @ normal
        if not .8 <= distance <= NEAR_DISTANCE:
            return empty
        horizontal = np.cross([0.,0.,1.],normal)
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal,horizontal)
        axes = np.column_stack((horizontal,vertical))
        outline = ((polygon-center) @ axes * .85).astype(np.float32)
        extent = np.ptp(outline,axis=0)
        if not (.2<extent[0]<.9 and .2<extent[1]<.9) or not cv2.isContourConvex(outline):
            return empty
        world = points @ pose[:3,:3].T + pose[:3,3]
        offset = world-center
        depth = offset @ normal
        candidates = np.isfinite(points).all(1) & (depth>.08) & (depth<.40) & (points[:,0]>.65)
        ids = np.flatnonzero(candidates)
        projected = offset[ids] @ axes
        empty[ids] = [cv2.pointPolygonTest(outline,tuple(map(float,p)),False)>=0 for p in projected]
        return empty
    except (ValueError,KeyError,TypeError,cv2.error):
        return empty


def reconstruct_scan(xyz, scan, removed):
    """Re-bin same-frame 3D returns; retain the next real obstacle in each bin."""
    xyz=np.asarray(xyz,float)
    ranges=np.asarray(scan.ranges,float)
    radial=np.linalg.norm(xyz[:,:2],axis=1)
    angle=np.arctan2(xyz[:,1],xyz[:,0])
    valid=np.isfinite(xyz).all(1)&(xyz[:,2]>=0)&(xyz[:,2]<=1)&(radial>=scan.range_min)&(radial<=scan.range_max)
    valid &= (angle>=scan.angle_min)&(angle<=scan.angle_min+len(ranges)*scan.angle_increment)
    ids=np.flatnonzero(valid)
    bins=((angle[ids]-scan.angle_min)/scan.angle_increment).astype(int)
    inside=(bins>=0)&(bins<len(ranges));ids,bins=ids[inside],bins[inside]
    original=np.full(len(ranges),np.inf);filtered=original.copy()
    np.minimum.at(original,bins,radial[ids])
    keep=~np.asarray(removed,bool)[ids]
    np.minimum.at(filtered,bins[keep],radial[ids[keep]])
    # Only change bins whose raw minimum agrees with the actual /scan source.
    matched=np.isfinite(ranges)&np.isfinite(original)
    difference=np.full(len(ranges),np.inf)
    difference[matched]=np.abs(ranges[matched]-original[matched])
    changed=(difference<.005)&(filtered>original+.005)
    output=ranges.copy();output[changed]=filtered[changed]
    return output,changed


def cloud_xyz(message):
    fields={f.name:f for f in message.fields}
    if any(k not in fields or fields[k].datatype!=7 or fields[k].count!=1 for k in ('x','y','z')):
        raise ValueError('Unsupported cloud layout')
    endian='>' if message.is_bigendian else '<'
    dtype=np.dtype(dict(names=['x','y','z'],formats=[endian+'f4']*3,
                       offsets=[fields[k].offset for k in ('x','y','z')],itemsize=message.point_step))
    array=np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,strides=(message.row_step,message.point_step))
    return np.column_stack([array[k].ravel() for k in ('x','y','z')]).astype(float)
