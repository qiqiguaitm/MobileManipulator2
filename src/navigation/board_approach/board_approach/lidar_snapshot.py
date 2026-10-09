"""Read-only scan decoding and odometry deskew for robot02's RS16 publisher."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .perception import DetectionError


def stamp_seconds(message):
    return message.header.stamp.sec + message.header.stamp.nanosec*1e-9


def decode_cloud(message):
    # source_pointcloud_ros.hpp exports time = point.timestamp - header + 0.1.
    # Recover absolute point time using header + time - 0.1, not header + time.
    fields = {f.name:f for f in message.fields}
    names = ('x','y','z','time','ring')
    if any(k not in fields for k in names):
        raise DetectionError('LiDAR must provide x/y/z/time/ring for deskew')
    if any(fields[k].datatype!=7 or fields[k].count!=1 for k in names[:4]) or fields['ring'].datatype!=4:
        raise DetectionError('Unexpected robot02 LiDAR point-field layout')
    endian = '>' if message.is_bigendian else '<'
    dtype = np.dtype(dict(names=names,formats=[endian+'f4']*4+[endian+'u2'],
                         offsets=[fields[k].offset for k in names],itemsize=message.point_step))
    try:
        array = np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,
                           strides=(message.row_step,message.point_step))
        xyz = np.stack([array[k].ravel() for k in names[:3]],axis=1).astype(float)
        offsets = array['time'].ravel().astype(float)-.1
        rings = array['ring'].ravel().copy()
    except (ValueError,TypeError) as error:
        raise DetectionError('Malformed LiDAR message') from error
    keep = np.isfinite(xyz).all(1)&np.isfinite(offsets)&(np.linalg.norm(xyz,axis=1)>.2)
    xyz, offsets, rings = xyz[keep],offsets[keep],rings[keep]
    if not len(xyz) or offsets.min()<-.12 or offsets.max()>.12 or np.ptp(offsets)>.12:
        raise DetectionError('LiDAR point times do not match the calibrated 10 Hz RS16 time convention')
    return xyz, stamp_seconds(message)+offsets, rings


def deskew(points_lidar, point_times, image_time, odometry, base_from_lidar, child_from_base=None):
    """Transform each return into base_link at image exposure time, using SE(3) odometry."""
    odometry = sorted(odometry,key=stamp_seconds)
    times = np.array([stamp_seconds(m) for m in odometry])
    unique = np.r_[True,np.diff(times)>1e-6] if len(times) else []
    odometry = [m for m,keep in zip(odometry,unique) if keep]
    times = np.array([stamp_seconds(m) for m in odometry])
    query = np.r_[point_times,image_time]
    if len(times)<2 or query.min()<times[0] or query.max()>times[-1]:
        raise DetectionError('Odometry history does not bracket LiDAR scan and camera exposure')
    indices = np.clip(np.searchsorted(times,query),1,len(times)-1)
    # Dedicated FAST-LIO measurement timestamps (~10 Hz), not callback receipt times.
    if np.max(times[indices]-times[indices-1])>.20:
        raise DetectionError('Odometry gap too large for LiDAR deskew')
    frames = {(m.header.frame_id,m.child_frame_id) for m in odometry}
    if len(frames)!=1:
        raise DetectionError('LiDAR deskew requires a consistent odometry frame pair')
    child=next(iter(frames))[1]
    if child_from_base is None:
        if child!='base_link':
            raise DetectionError('Missing odometry-child to base_link transform for LiDAR deskew')
        child_from_base=np.eye(4)
    child_from_base=np.asarray(child_from_base,float)
    if child_from_base.shape!=(4,4) or not np.isfinite(child_from_base).all():
        raise DetectionError('Invalid odometry-child to base_link transform')
    positions,quats = [],[]
    for m in odometry:
        p=m.pose.pose
        positions.append([p.position.x,p.position.y,p.position.z])
        quats.append([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
    positions,quats=np.asarray(positions),np.asarray(quats)
    if not np.isfinite(positions).all() or not np.isfinite(quats).all() or np.any(np.abs(np.linalg.norm(quats,axis=1)-1)>.01):
        raise DetectionError('Invalid odometry for LiDAR deskew')
    body_rotations=Rotation.from_quat(quats)
    positions=positions+body_rotations.apply(np.broadcast_to(child_from_base[:3,3],positions.shape))
    quats=(body_rotations*Rotation.from_matrix(child_from_base[:3,:3])).as_quat()
    # Bounds catch resets/discontinuities; controlled motion is <= .05 m/s and .15 rad/s.
    used=(times>=query.min()-.20)&(times<=query.max()+.20)
    if used.sum()>1:
        dt=np.diff(times[used])
        shifts=np.linalg.norm(np.diff(positions[used],axis=0),axis=1)
        speeds=shifts/dt
        index=int(np.argmax(speeds))
        if speeds[index]>.15:
            raise DetectionError(f'Odometry jump or excessive speed during LiDAR capture: '
                                 f'{speeds[index]:.3f} m/s, delta={shifts[index]:.4f}m, dt={dt[index]:.4f}s')
        rot=Rotation.from_quat(quats[used])
        angular=(rot[:-1].inv()*rot[1:]).magnitude()/dt
        if np.max(angular)>.3:
            raise DetectionError(f'Odometry rotation jump during LiDAR capture: {np.max(angular):.3f} rad/s')
    # Subtract epoch first to improve interpolation conditioning.
    rotations = Slerp(times-times[0],Rotation.from_quat(quats))(query-times[0])
    translations = np.column_stack([np.interp(query,times,positions[:,axis]) for axis in range(3)])
    points_base = points_lidar@base_from_lidar[:3,:3].T+base_from_lidar[:3,3]
    world = rotations[:-1].apply(points_base)+translations[:-1]
    aligned = rotations[-1].inv().apply(world-translations[-1])
    return aligned, dict(deskew='per_point_odometry_SE3',scan_span_ms=float(np.ptp(point_times)*1000),
                         point_to_image_max_ms=float(np.max(np.abs(point_times-image_time))*1000))


def capture_lidar(node, image_stamp):
    cloud = min(node.clouds,key=lambda m:abs(stamp_seconds(m)-image_stamp/1e9))
    if cloud.header.frame_id!='rslidar':
        raise DetectionError('Unexpected LiDAR frame; refusing to reuse another sensor calibration')
    gap=abs(stamp_seconds(cloud)-image_stamp/1e9)
    if gap>.06:
        raise DetectionError('No LiDAR scan within 60 ms of camera exposure')
    points,times,rings=decode_cloud(cloud)
    base_from_lidar=node.transform('base_link','rslidar')
    # Derive the scan's calibrated beam elevations before deskew, independently
    # of which points later hit the target. Require all 16 RS16 rings.
    beam_angles = []
    if np.array_equal(np.unique(rings), np.arange(16)):
        elevation = np.degrees(np.arctan2(points[:,2], np.linalg.norm(points[:,:2],axis=1)))
        beam_angles = [float(np.median(elevation[rings==r])) for r in range(16)]
    child=node.odometry[-1].child_frame_id
    if child not in ('body','base_link'):
        raise DetectionError('Unexpected FAST-LIO child frame for LiDAR deskew')
    child_from_base=node.transform(child,'base_link') if child!='base_link' else np.eye(4)
    points,metadata=deskew(points,times,image_stamp/1e9,list(node.odometry),base_from_lidar,child_from_base)
    info=node.data['info']
    if info.header.frame_id!='top_color_optical_frame' or info.distortion_model not in ('plumb_bob',''):
        raise DetectionError('Camera frame/distortion differs from saved LiDAR calibration')
    distortion=np.asarray(info.d,float)
    if distortion.size not in (0,4,5,8) or not np.isfinite(distortion).all():
        raise DetectionError('Invalid RGB distortion coefficients')
    metadata.update(lidar_stamp=stamp_seconds(cloud),image_stamp=image_stamp/1e9,
                    pair_gap_ms=gap*1000,lidar_frame=cloud.header.frame_id,camera_frame=info.header.frame_id,
                    time_convention='RS16 point_time = header + time_field - 0.1 seconds',
                    base_from_lidar=base_from_lidar.tolist(),
                    beam_elevations_deg=beam_angles,
                    odometry_source='/fastlio/odom_measurement',odometry_child=child,
                    odometry_child_from_base=child_from_base.tolist())
    return dict(points_base=points,rings=rings,distortion=distortion,metadata=metadata)
