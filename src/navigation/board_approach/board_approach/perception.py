"""Pure geometry and full-image target detection. No ROS or motor access."""
from dataclasses import dataclass
import cv2
import numpy as np
from .limits import target_range_limit, FINAL_LATERAL_TOLERANCE_M, FINAL_HEADING_TOLERANCE_DEG
from scipy.spatial.transform import Rotation


class DetectionError(RuntimeError):
    pass


class BoardUnavailable(DetectionError):
    """No usable board identity in this frame; another fresh frame may help."""


@dataclass
class Observation:
    center: np.ndarray
    normal: np.ndarray
    ids: list
    polygon: np.ndarray
    rmse: float
    valid_fraction: float
    source: str = "markers"
    score: float = 1.0
    plane_fit: dict = None


@dataclass
class Region:
    """Image-only target evidence, deliberately containing no guessed depth."""
    polygon: np.ndarray
    uv: np.ndarray
    ids: list
    source: str = 'object'
    score: float = None
    exclusion_polygons: tuple = ()
    region_scale: float = None
    min_base_height: float = .15


def board_layout():
    # This board has 7 columns x 5 rows; top-left square is black.
    # Only unitless marker centres are used: physical square size is NOT assumed.
    return np.array([(x + .5, y + .5) for y in range(5)
                     for x in range(7) if (x + y) % 2 == 1], np.float32)


def sufficient_layout(ids):
    if len(ids) >= 4 and {8, 15} <= set(ids):
        return True
    # A glare-affected centre marker is not essential when many known IDs span
    # the board: their verified layout determines its centre geometrically.
    if len(ids) >= 8:
        span = np.ptp(board_layout()[list(ids)], axis=0)
        return bool(span[0] >= 4 and span[1] >= 3)
    return False


def detect_board(image, depth_m, k):
    """Detect the configured board across the FULL frame, then fit its depth plane."""
    if image.shape[:2] != depth_m.shape or not np.isfinite(k).all():
        raise DetectionError('RGB/depth dimensions or intrinsics invalid')
    region = detect_board_region(image)
    return fit_board_plane(depth_m, k, region.polygon, region.uv, region.ids)


def detect_board_region(image):
    """Decode the board without consulting camera depth."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMax = 81
    params.adaptiveThreshWinSizeStep = 4
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000), params)
    found = {}
    # Keep the normal path cheap. Small, aliased markers need alternative
    # contour extraction and interpolation, without relaxing decoded-ID checks.
    variants = [(scale, cv2.INTER_CUBIC, False, False) for scale in (1, 2, 3)]
    variants += [(scale, cv2.INTER_CUBIC, False, True) for scale in (2, 3, 4)]
    variants += [(scale, cv2.INTER_LANCZOS4, True, False) for scale in (2, 3, 4)]
    for scale, interpolation, sharpen, april in variants:
        params.cornerRefinementMethod = (cv2.aruco.CORNER_REFINE_APRILTAG if april
                                         else cv2.aruco.CORNER_REFINE_NONE)
        detector.setDetectorParameters(params)
        pixels = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=interpolation)
        if sharpen:
            pixels = cv2.addWeighted(pixels, 2, cv2.GaussianBlur(pixels, (0, 0), 1), -1, 0)
        corners, ids, _ = detector.detectMarkers(pixels)
        if ids is None:
            continue
        for marker, corner in zip(ids.ravel(), corners):
            if not 0 <= marker < 17:
                continue
            center = corner.reshape(4, 2).mean(axis=0) / scale
            if int(marker) in found and np.linalg.norm(found[int(marker)]-center) > 8:
                raise DetectionError('Multiple boards / duplicate marker IDs: ambiguous target')
            found[int(marker)] = center
        if len(found) >= 8 and {8, 15} <= found.keys():
            break
    ids = sorted(found)
    if not sufficient_layout(ids):
        raise BoardUnavailable('Need at least 4 markers including 8/15, or 8 well-distributed board IDs')
    layout = board_layout()
    homography, mask = cv2.findHomography(layout[ids], np.float32([found[i] for i in ids]),
                                        cv2.RANSAC, 2.5)
    if homography is None or mask is None:
        raise DetectionError('Marker layout is degenerate')
    ids = [i for i, good in zip(ids, mask.ravel()) if good]
    if not sufficient_layout(ids):
        raise DetectionError('Marker layout does not match the 7x5 target board')
    def project(points):
        return cv2.perspectiveTransform(np.float32(points)[None], homography)[0]
    polygon = project([[.25, .25], [6.75, .25], [6.75, 4.75], [.25, 4.75]])
    return Region(polygon, project([[3.5, 2.5]])[0], ids, source='markers', score=1.)


def plane_thresholds(distance):
    """Bounded empirical depth-noise allowance; near-field precision is unchanged."""
    if not np.isfinite(distance) or distance <= 0:
        raise DetectionError('Invalid depth for plane thresholds')
    growth = np.clip((distance**2 - 1.5**2)/(4.0**2 - 1.5**2), 0., 1.)
    inlier_distance = float(.020 + .030*growth)
    return inlier_distance, .6*inlier_distance


def fit_board_plane(depth_m, k, polygon, uv, ids):
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
        raise DetectionError('Invalid board outline')
    h, w = depth_m.shape
    if (polygon[:, 0] < 0).any() or (polygon[:, 0] >= w).any() or (polygon[:, 1] < 0).any() or (polygon[:, 1] >= h).any():
        raise BoardUnavailable('Board is partially outside the camera view')
    mask = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 1)
    yy, xx = np.where(mask)
    z = depth_m[yy, xx]
    valid = np.isfinite(z) & (z > .35) & (z < 4.0)
    fraction = float(valid.mean()) if len(valid) else 0
    if valid.sum() < 100 or fraction < .6:
        raise DetectionError('Insufficient valid depth on board')
    xx, yy, z = xx[valid], yy[valid], z[valid]
    distance = float(np.median(z))
    inlier_distance, rmse_limit = plane_thresholds(distance)
    points = np.column_stack(((xx-k[0, 2])*z/k[0, 0], (yy-k[1, 2])*z/k[1, 1], z))
    points = points[::max(1, len(points)//4000)]
    rng = np.random.default_rng(42)
    best = np.zeros(len(points), bool)
    for _ in range(80):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        hits = np.abs((points-a) @ (normal/length)) < inlier_distance
        if hits.sum() > best.sum():
            best = hits
    if best.mean() < .8:
        raise DetectionError(f'Board depth is not a consistent plane: inliers={best.mean():.3f} < 0.800, depth={distance:.3f}m, tolerance={inlier_distance:.4f}m')
    points = points[best]
    centroid = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points-centroid, full_matrices=False)
    normal = vectors[-1]
    if normal @ centroid > 0:
        normal = -normal
    error = float(np.sqrt(np.mean(((points-centroid) @ normal)**2)))
    if error > rmse_limit:
        raise DetectionError(f'Depth plane residual too large: rmse={error:.4f}m > {rmse_limit:.4f}m at depth={distance:.3f}m')
    ray = np.array([(uv[0]-k[0, 2])/k[0, 0], (uv[1]-k[1, 2])/k[1, 1], 1.])
    if abs(normal @ ray) < .3:
        raise DetectionError('Board viewed too obliquely')
    center = ray * ((normal @ centroid)/(normal @ ray))
    return Observation(center, normal, ids, polygon, error, fraction, plane_fit=dict(
        depth_m=distance, inlier_tolerance_m=inlier_distance, rmse_limit_m=rmse_limit,
        inlier_fraction=float(best.mean()), rmse_m=error))


def pose_matrix(xyz, quat):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def target_geometry(observation, base_from_camera, map_from_base, standoff):
    center = (base_from_camera @ np.r_[observation.center, 1])[:3]
    normal = base_from_camera[:3, :3] @ observation.normal
    if abs(normal[2]) > .4:
        raise DetectionError('Target plane is not approximately vertical')
    normal2 = normal[:2]/np.linalg.norm(normal[:2])
    distance = float(-center[:2] @ normal2)
    limit = target_range_limit(observation)
    if not .35 < distance <= limit:
        raise DetectionError(f'Board distance outside approach range: distance={distance:.3f} m; limit={limit:.1f} m')
    mapped = (map_from_base @ np.r_[center, 1])[:3]
    world_normal = map_from_base[:3, :3] @ normal
    world_normal[2] = 0
    world_normal /= np.linalg.norm(world_normal)
    goal = mapped + standoff*world_normal
    return dict(board_map=mapped.tolist(), goal_map=[float(goal[0]), float(goal[1]), 0.],
                goal_yaw=float(np.arctan2(-world_normal[1], -world_normal[0])),
                distance=distance, lateral=float(center[:2] @ np.array([-normal2[1], normal2[0]])),
                heading_error=float(np.arctan2(-normal2[1], -normal2[0])),
                marker_ids=observation.ids, plane_rmse=observation.rmse, plane_fit=observation.plane_fit,
                identity_verified=observation.source == 'markers',
                detection_source=observation.source, detection_score=observation.score)


def arrived(target, standoff, require_aruco=True):
    verified = target.get('identity_verified', False)
    appearance = target.get('detection_source') in ('object_depth', 'object_lidar')
    eligible = verified or (not require_aruco and appearance)
    return (eligible and abs(target['distance']-standoff) <= .10
            and abs(target['lateral']) <= FINAL_LATERAL_TOLERANCE_M
            and abs(target['heading_error']) <= np.deg2rad(FINAL_HEADING_TOLERANCE_DEG))
