"""
TargetPool — thread-safe tracker-to-map target management.

All internal coordinates are SI (meters). TF transform from base_link to map
is applied on each update so targets accumulate in a global frame.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional

import numpy as np
from geometry_msgs.msg import Point, TransformStamped


class TargetStatus(IntEnum):
    ACTIVE = 0
    PICKED = 1
    FAILED = 2


@dataclass
class TargetRecord:
    track_id: int
    category: str
    position_map: Point          # in map frame, meters
    track_score: float
    position_confidence: float
    status: TargetStatus = TargetStatus.ACTIVE
    attempts: int = 0
    last_seen: float = field(default_factory=time.time)
    fail_reason: str = ""


def _point_distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def _transform_point(point: Point, tf: TransformStamped) -> Point:
    """Apply TF transform to a point using manual quaternion rotation."""
    t = tf.transform.translation
    q = tf.transform.rotation

    # Quaternion rotation: p' = q * p * q_inv
    px, py, pz = point.x, point.y, point.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w

    # Rotation matrix from quaternion (more efficient than double cross product)
    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz

    out = Point()
    out.x = (1 - 2*(yy+zz)) * px + 2*(xy-wz) * py + 2*(xz+wy) * pz + t.x
    out.y = 2*(xy+wz) * px + (1-2*(xx+zz)) * py + 2*(yz-wx) * pz + t.y
    out.z = 2*(xz-wy) * px + 2*(yz+wx) * py + (1-2*(xx+yy)) * pz + t.z
    return out


class TargetPool:
    """Thread-safe pool of tracked objects projected into map frame."""

    MATCH_THRESHOLD = 0.08  # 8 cm

    def __init__(self, tf_buffer, logger,
                 min_track_score: float = 0.3,
                 min_position_confidence: float = 0.3,
                 min_distance: float = 0.3,
                 max_distance: float = 3.0,
                 max_attempts: int = 3):
        self._tf_buffer = tf_buffer
        self._logger = logger
        self._min_track_score = min_track_score
        self._min_position_confidence = min_position_confidence
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._max_attempts = max_attempts

        self._lock = threading.Lock()
        self._targets: List[TargetRecord] = []
        self._update_enabled = True

        # Stats
        self._total_added = 0
        self._total_picked = 0
        self._total_failed = 0

    def pause(self):
        with self._lock:
            self._update_enabled = False

    def resume(self):
        with self._lock:
            self._update_enabled = True

    def update_from_tracker(self, msg) -> None:
        """Called from subscription callback with TrackedObject3DArray."""
        with self._lock:
            if not self._update_enabled:
                return

        # Lookup TF outside lock (may block briefly)
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link',
                msg.header.stamp,
                timeout=None
            )
        except Exception as e:
            self._logger.debug(f'TF lookup failed: {e}')
            return

        with self._lock:
            if not self._update_enabled:
                return
            self._update_locked(msg, tf)

    def _update_locked(self, msg, tf: TransformStamped) -> None:
        """Must be called with lock held."""
        now = time.time()
        for obj in msg.objects:
            if obj.track_score < self._min_track_score:
                continue
            if obj.position_confidence < self._min_position_confidence:
                continue
            if obj.distance < self._min_distance or obj.distance > self._max_distance:
                continue

            pos_map = _transform_point(obj.position, tf)

            # Match against existing targets
            matched = False
            for t in self._targets:
                if t.category != obj.category:
                    continue
                if _point_distance(t.position_map, pos_map) < self.MATCH_THRESHOLD:
                    # Update existing — running average for position stability
                    t.position_map.x = 0.7 * t.position_map.x + 0.3 * pos_map.x
                    t.position_map.y = 0.7 * t.position_map.y + 0.3 * pos_map.y
                    t.position_map.z = 0.7 * t.position_map.z + 0.3 * pos_map.z
                    t.track_score = max(t.track_score, obj.track_score)
                    t.position_confidence = max(t.position_confidence, obj.position_confidence)
                    t.last_seen = now
                    matched = True
                    break

            if not matched:
                self._targets.append(TargetRecord(
                    track_id=obj.track_id,
                    category=obj.category,
                    position_map=pos_map,
                    track_score=obj.track_score,
                    position_confidence=obj.position_confidence,
                    last_seen=now,
                ))
                self._total_added += 1

    def get_nav_target(self, robot_pos_map: Point) -> Optional[TargetRecord]:
        """Return nearest active target that hasn't exceeded max attempts."""
        with self._lock:
            candidates = [
                t for t in self._targets
                if t.status == TargetStatus.ACTIVE and t.attempts < self._max_attempts
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda t: _point_distance(t.position_map, robot_pos_map))
            return candidates[0]

    def get_workspace_targets(
        self,
        in_working_area_fn: Callable[[Point], bool],
        robot_pos_map: Point,
    ) -> List[TargetRecord]:
        """Return active targets within working area, sorted by distance."""
        with self._lock:
            results = []
            for t in self._targets:
                if t.status != TargetStatus.ACTIVE:
                    continue
                if t.attempts >= self._max_attempts:
                    continue
                if in_working_area_fn(t.position_map):
                    results.append(t)
            results.sort(key=lambda t: _point_distance(t.position_map, robot_pos_map))
            return results

    def mark_picked(self, pos: Point) -> None:
        with self._lock:
            t = self._find_nearest_locked(pos)
            if t:
                t.status = TargetStatus.PICKED
                self._total_picked += 1

    def mark_failed(self, pos: Point, reason: str = "") -> None:
        with self._lock:
            t = self._find_nearest_locked(pos)
            if t:
                t.attempts += 1
                if t.attempts >= self._max_attempts:
                    t.status = TargetStatus.FAILED
                    self._total_failed += 1
                t.fail_reason = reason

    def _find_nearest_locked(self, pos: Point) -> Optional[TargetRecord]:
        """Find nearest active target to pos. Must be called with lock held."""
        best = None
        best_dist = float('inf')
        for t in self._targets:
            if t.status != TargetStatus.ACTIVE:
                continue
            d = _point_distance(t.position_map, pos)
            if d < best_dist:
                best_dist = d
                best = t
        return best

    def get_active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._targets if t.status == TargetStatus.ACTIVE)

    @property
    def stats(self):
        with self._lock:
            active = sum(1 for t in self._targets if t.status == TargetStatus.ACTIVE)
            return {
                'total': len(self._targets),
                'active': active,
                'picked': self._total_picked,
                'failed': self._total_failed,
                'remaining': active,
            }

    def reset(self) -> None:
        with self._lock:
            self._targets.clear()
            self._total_added = 0
            self._total_picked = 0
            self._total_failed = 0
            self._update_enabled = True
