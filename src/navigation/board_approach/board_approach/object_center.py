"""Generic visible-object centre from a detector box and aligned RGB-D.

No marker decoding, checkerboard layout, plane normal or saved target position.
The centre is the detected image centre at robust visible-surface depth, not the
unobservable volumetric centre of an object.
"""
import math
import cv2
import numpy as np
from .limits import target_range_limit
import requests
from .perception import BoardUnavailable, DetectionError, Observation, Region, fit_board_plane


def object_region(image_shape, bbox, score=None, allow_bottom_crop=False):
    h, w = image_shape[:2]
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise DetectionError('Invalid object bounding box')
    x1,y1,x2,y2 = box
    bottom_ok = y2 <= h if allow_bottom_crop else y2 < h-1
    if not (1 <= x1 < x2 < w-1 and 1 <= y1 < y2 and bottom_ok) or min(x2-x1,y2-y1)<12:
        raise BoardUnavailable('Object cropped or bounding box too small; wait for a usable full region')
    polygon = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    return Region(polygon, np.array([(x1+x2)/2,(y1+y2)/2]), [], score=score)


def object_observation(depth, k, bbox, score):
    box = np.asarray(bbox, dtype=float)
    h, w = depth.shape
    if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(k).all():
        raise DetectionError('Invalid object box or camera intrinsics')
    x1, y1, x2, y2 = box
    if not (1 <= x1 < x2 < w-1 and 1 <= y1 < y2 < h-1):
        raise DetectionError('Object is cropped by the camera view or box is invalid')
    if min(x2-x1, y2-y1) < 12 or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise DetectionError('Object is too small or camera intrinsics invalid')
    # Use the central half of the box, avoiding background along box edges.
    cx, cy = (x1+x2)/2, (y1+y2)/2
    dx, dy = (x2-x1)/4, (y2-y1)/4
    patch = depth[int(cy-dy):int(cy+dy)+1, int(cx-dx):int(cx+dx)+1]
    valid = np.isfinite(patch) & (patch > .35) & (patch < 4.)
    if valid.sum() < 25 or valid.mean() < .65:
        raise DetectionError('Insufficient valid depth at object centre')
    values = patch[valid]
    lo, hi = np.percentile(values, [10, 90])
    if hi-lo > .25:
        # A broad oblique face legitimately spans depths. Require a coherent
        # plane instead of accepting arbitrary mixed foreground/background.
        inner = np.float32([[cx-dx,cy-dy],[cx+dx,cy-dy],
                            [cx+dx,cy+dy],[cx-dx,cy+dy]])
        try:
            plane = fit_board_plane(depth, k, inner, [cx,cy], [])
        except DetectionError as error:
            raise DetectionError('Object centre depth is ambiguous / mixed with background: '+str(error)) from error
        plane.polygon = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        plane.source, plane.score = 'object_depth', float(score)
        return plane
    z = float(np.median(values))
    center = np.array([(cx-k[0, 2])*z/k[0, 0], (cy-k[1, 2])*z/k[1, 1], z])
    polygon = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    spread = float(np.median(np.abs(values-z)))
    return Observation(center, np.zeros(3), [], polygon, spread, float(valid.mean()),
                       source='object_depth', score=float(score))


class ObjectDetector:
    """The project's DINO-X HTTP API; one unambiguous prompt target only."""
    def __init__(self, url, prompt, min_score=.4, timeout=5.):
        if not prompt.strip():
            raise ValueError('Object prompt cannot be empty')
        self.url = url.rstrip('/')+'/api/predict'
        self.prompt, self.min_score, self.timeout = prompt, min_score, timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def detect(self, image, depth, k):
        if image.shape[:2] != depth.shape:
            raise DetectionError('RGB/depth dimensions differ')
        region = self.detect_region(image)
        return object_observation(depth, k, self.last_metadata['bbox'], region.score)

    def detect_region(self, image):
        self.last_metadata = None
        ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise DetectionError('Cannot encode camera frame')
        try:
            response = self.session.post(self.url,
                files={'images':('frame.jpg', jpeg.tobytes(), 'image/jpeg')},
                data={'text_prompt':self.prompt, 'min_score':self.min_score,
                      'iou_threshold':.5, 'chosen_policy':'det'}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise DetectionError('Object detector unavailable: '+str(error)) from error
        results = payload.get('results') if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != 1:
            raise DetectionError('Unexpected detector response: expected one image result')
        objects = results[0].get('objects', [])
        accepted = []
        for item in objects:
            score = item.get('score')
            if not isinstance(score, (int,float)) or not math.isfinite(score) or score < self.min_score:
                continue
            # Do not silently pick the closest object when two match the prompt.
            accepted.append(item)
        if not accepted:
            raise BoardUnavailable('No confident object matching prompt: '+self.prompt)
        if len(accepted) != 1:
            raise DetectionError('Multiple objects match the prompt; specify a unique target')
        item = accepted[0]
        region = object_region(image.shape, item.get('bbox'), item['score'])
        self.last_metadata = dict(provider='dinox', bbox=list(item['bbox']), score=item['score'])
        return region


def center_target_geometry(observation, base_from_camera, map_from_base, standoff):
    center = (base_from_camera @ np.r_[observation.center, 1])[:3]
    distance = float(np.linalg.norm(center[:2]))
    limit = target_range_limit(observation)
    if not np.isfinite(center).all() or not .5 < distance <= limit:
        raise DetectionError(f'Object centre outside approach range: distance={distance:.3f} m; limit={limit:.1f} m')
    direction = center[:2]/distance
    goal_base = np.r_[(distance-standoff)*direction, 0., 1.]
    mapped = (map_from_base @ np.r_[center, 1])[:3]
    goal = (map_from_base @ goal_base)[:3]
    world_direction = map_from_base[:3, :3] @ np.r_[direction, 0.]
    return dict(board_map=mapped.tolist(), object_map=mapped.tolist(),
                goal_map=[float(goal[0]),float(goal[1]),0.],
                goal_yaw=float(np.arctan2(world_direction[1],world_direction[0])),
                distance=distance, lateral=float(center[1]),
                heading_error=float(np.arctan2(center[1],center[0])),
                marker_ids=observation.ids, identity_verified=observation.source=='markers',
                detection_source=observation.source, detection_score=observation.score,
                depth_spread=observation.rmse, alignment='face_object_center')
