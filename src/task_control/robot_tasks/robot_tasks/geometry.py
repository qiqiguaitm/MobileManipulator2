"""Pure approach geometry. Object positions never imply measured surface normals."""

import math
import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError("Invalid unit quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def front_goal(front, outward_yaw, clearance, front_extent):
    """front is the front-face center, not the appliance center."""
    if (
        not np.isfinite([*front, outward_yaw, clearance, front_extent]).all()
        or min(clearance, front_extent) <= 0
    ):
        raise ValueError("Invalid front alignment geometry")
    d = clearance + front_extent
    return (
        front[0] + d * math.cos(outward_yaw),
        front[1] + d * math.sin(outward_yaw),
        wrap(outward_yaw + math.pi),
    )


def inspection_base_distance(front_extent, front_clearance):
    """Convert physical front clearance to the base_link-to-target goal distance."""
    if (
        not np.isfinite([front_extent, front_clearance]).all()
        or min(front_extent, front_clearance) <= 0
    ):
        raise ValueError("Front extent and front clearance must be finite and positive")
    return front_extent + front_clearance


def inspection_goals(target, robot, distance, angles):
    if (
        len(target) != 3
        or not np.isfinite([*target, *robot, distance, *angles]).all()
        or distance <= 0
    ):
        raise ValueError("Invalid inspection stand-off geometry")
    a = math.atan2(robot[1] - target[1], robot[0] - target[0])
    return [
        (
            target[0] + distance * math.cos(a + d),
            target[1] + distance * math.sin(a + d),
            wrap(a + d + math.pi),
        )
        for d in angles
    ]


def pose_errors(robot, goal):
    return math.hypot(robot[0] - goal[0], robot[1] - goal[1]), abs(
        wrap(robot[2] - goal[2])
    )


def target_in_standoff_bounds(target, robot, x_range, y_limit, z_range):
    """Validate base-relative navigation stand-off, not manipulator reachability."""
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    c, s = math.cos(robot[2]), math.sin(robot[2])
    x, y = c * dx + s * dy, -s * dx + c * dy
    return (
        x_range[0] <= x <= x_range[1]
        and abs(y) <= y_limit
        and z_range[0] <= target[2] <= z_range[1]
    )


def depth_position(
    depth_m,
    bbox,
    k,
    roi_scale=0.5,
    min_pixels=30,
    min_fraction=0.35,
    max_spread=0.12,
    distortion=(),
):
    """Robust median of central aligned-depth ROI; reject mixed/weak depth.

    Estimates visible surface position, not object center-of-mass or grasp pose.
    """
    d = np.asarray(depth_m, float)
    k = np.asarray(k, float).reshape(3, 3)
    b = np.asarray(bbox, float)
    if (
        d.ndim != 2
        or b.shape != (4,)
        or not np.isfinite(b).all()
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
    ):
        raise ValueError("Invalid depth image, bbox or intrinsics")
    h, w = d.shape
    x1, y1, x2, y2 = b
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("BBox outside image")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) * roi_scale / 2, (y2 - y1) * roi_scale / 2
    xa, xb = max(0, int(cx - rx)), min(w, int(math.ceil(cx + rx)))
    ya, yb = max(0, int(cy - ry)), min(h, int(math.ceil(cy + ry)))
    roi = d[ya:yb, xa:xb]
    v, u = np.mgrid[ya:yb, xa:xb]
    valid = np.isfinite(roi) & (roi >= 0.2) & (roi <= 10.0)
    if valid.sum() < min_pixels or valid.mean() < min_fraction:
        raise ValueError("Insufficient valid stereo depth")
    z = roi[valid]
    median = float(np.median(z))
    mad = float(np.median(abs(z - median)))
    spread = float(np.quantile(z, 0.9) - np.quantile(z, 0.1))
    if spread > max_spread:
        raise ValueError(f"Mixed stereo depth: spread={spread:.3f}m")
    keep = valid & (abs(roi - median) <= max(0.015, 3 * 1.4826 * mad))
    if keep.sum() < min_pixels:
        raise ValueError("Insufficient depth after outlier rejection")
    z = roi[keep]
    if len(distortion) and any(distortion):
        import cv2

        pixels = np.column_stack((u[keep], v[keep])).astype(float).reshape(-1, 1, 2)
        uv = cv2.undistortPoints(pixels, k, np.asarray(distortion, float)).reshape(
            -1, 2
        )
        x, y = uv[:, 0] * z, uv[:, 1] * z
    else:
        x, y = (u[keep] - k[0, 2]) * z / k[0, 0], (v[keep] - k[1, 2]) * z / k[1, 1]
    return np.median(np.column_stack((x, y, z)), axis=0), dict(
        valid_pixels=int(keep.sum()), depth_m=median, spread_m=spread
    )
