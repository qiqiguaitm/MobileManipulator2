"""Full-frame board appearance search followed by measured RGB-D geometry.

Templates encode only the printed 7x5 board pattern, never its scene location.
Appearance hypotheses are only a coarse fallback for the optional ArUco mode.
Generic object mode never calls this detector.
"""
from functools import lru_cache
import cv2
import numpy as np
from .perception import BoardUnavailable, DetectionError, fit_board_plane


@lru_cache(maxsize=1)
def templates():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard((7, 5), 1., .737, dictionary).generateImage((350, 250))
    result = []
    for width in np.geomspace(45, 180, 18).astype(int):
        for aspect in (.85, 1., 1.15):
            height = int(width*5/7*aspect)
            pattern = cv2.resize(board, (width, height), interpolation=cv2.INTER_AREA)
            matrices = []
            for angle in (-16, -8, 0, 8, 16):
                m = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                w = int(abs(m[0, 0])*width+abs(m[0, 1])*height)+2
                h = int(abs(m[0, 0])*height+abs(m[0, 1])*width)+2
                m[:, 2] += np.array([w/2-width/2, h/2-height/2])
                matrices.append((m, w, h))
            for shear in (-.3, -.15, .15, .3):
                matrices.append((np.array([[1., 0., 1.], [shear, 1., 1.+max(0., -shear*width)]]),
                                 int(width+2), int(height+abs(shear)*width)+2))
            corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            for m, w, h in matrices:
                texture = cv2.warpAffine(pattern, m, (w, h), borderValue=255)
                polygon = (corners @ m[:, :2].T+m[:, 2]).astype(np.float32)
                result.append((texture, polygon))
    return result


def _fit(depth, k, polygon, score):
    center = polygon.mean(axis=0)
    # Avoid the printed border and nearby floor when fitting the actual board plane.
    inner = (center+.85*(polygon-center)).astype(np.float32)
    observation = fit_board_plane(depth, k, inner, center, [])
    width_pixels = (np.linalg.norm(polygon[1]-polygon[0])+np.linalg.norm(polygon[2]-polygon[3]))/2
    apparent_width = width_pixels*observation.center[2]/k[0, 0]
    if not .14 <= apparent_width <= .42:
        raise DetectionError('Candidate physical size is not plausible for this board')
    observation.source, observation.score = 'appearance_depth', float(score)
    return observation


class AppearanceDetector:
    def __init__(self):
        self.texture = self.polygon = None

    def detect(self, image, depth, k):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Within one stationary observation, reuse the automatically found shape,
        # while still searching the full frame and rejecting a second similar target.
        bank = [(self.texture, self.polygon)] if self.texture is not None else templates()
        candidates = []
        for texture, polygon in bank:
            h, w = texture.shape
            if h >= gray.shape[0] or w >= gray.shape[1]:
                continue
            scores = cv2.matchTemplate(gray, texture, cv2.TM_CCOEFF_NORMED)
            z = depth[h//2:h//2+scores.shape[0], w//2:w//2+scores.shape[1]]
            # Invalid/far depth must not make windows or distant repeating structures targets.
            scores[(z < .4) | (z > 4.) | ~np.isfinite(z)] = -1
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if score < .62:
                    break
                shifted = polygon+np.float32([x, y])
                candidates.append((score, shifted, texture, polygon))
                scores[max(0, y-h//2):y+h//2+1, max(0, x-w//2):x+w//2+1] = -1
        accepted = []
        for score, polygon, texture, local_polygon in sorted(candidates, key=lambda c: c[0], reverse=True):
            center = polygon.mean(axis=0)
            if any(np.linalg.norm(center-p[0]) < .6*texture.shape[1] for p in accepted):
                continue
            try:
                observation = _fit(depth, k, polygon, score)
            except DetectionError:
                continue
            accepted.append((center, observation, texture, local_polygon))
            if len(accepted) == 2:
                raise DetectionError('Multiple appearance/depth candidates; target ambiguous')
        if not accepted:
            raise BoardUnavailable('No reliable full-frame board appearance and depth match')
        _, observation, self.texture, self.polygon = accepted[0]
        return observation
