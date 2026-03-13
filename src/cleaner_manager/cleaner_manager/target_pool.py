"""
TargetPool — thread-safe tracker-to-map target management.

All internal coordinates are SI (meters). Perception publishes positions
already in map frame, so no TF transform is needed here.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional

import numpy as np
from geometry_msgs.msg import Point


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
    observation_count: int = 1   # Fix 3: incremented each matched frame


def _point_distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


class TargetPool:
    """Thread-safe pool of tracked objects projected into map frame."""

    def __init__(self, logger,

                 min_distance: float = 0.3,
                 max_distance: float = 3.0,
                 max_attempts: int = 3,
                 target_ttl: float = 60.0,
                 match_threshold: float = 0.20,
                 min_observations: int = 2):
        self._logger = logger
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._max_attempts = max_attempts
        self._target_ttl = target_ttl           # Fix 1: 0 = disabled
        self._match_threshold = match_threshold  # Fix 2: was hardcoded 0.08
        self._min_observations = min_observations  # Fix 3

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
        """Called from subscription callback with Object3DArray.

        Positions in msg.objects are already in map frame (transformed by
        perception), so no TF lookup is needed here.
        """
        with self._lock:
            if not self._update_enabled:
                return
            self._update_locked(msg)

    @staticmethod
    def _parse_track_id(object_id: str) -> int:
        """Parse int track_id from Object3D.object_id (e.g. 'bottle_1' → 1)."""
        try:
            return int(object_id.rsplit('_', 1)[-1])
        except (ValueError, IndexError):
            return hash(object_id) & 0x7FFFFFFF

    def _update_locked(self, msg) -> None:
        """Must be called with lock held."""
        now = time.time()
        for obj in msg.objects:
            if obj.distance < self._min_distance or obj.distance > self._max_distance:
                self._logger.debug(
                    f"Pool skip {obj.object_id}: dist={obj.distance:.2f}m "
                    f"(range [{self._min_distance}, {self._max_distance}])",
                    throttle_duration_sec=5.0)
                continue

            pos_map = obj.position  # already in map frame

            # Match against existing targets
            matched = False
            for t in self._targets:
                if t.category != obj.category:
                    continue
                if _point_distance(t.position_map, pos_map) < self._match_threshold:
                    # Update existing — running average for position stability
                    t.position_map.x = 0.7 * t.position_map.x + 0.3 * pos_map.x
                    t.position_map.y = 0.7 * t.position_map.y + 0.3 * pos_map.y
                    t.position_map.z = 0.7 * t.position_map.z + 0.3 * pos_map.z
                    t.track_score = max(t.track_score, obj.score)
                    t.position_confidence = max(t.position_confidence, obj.confidence)
                    t.last_seen = now
                    t.observation_count += 1  # Fix 3
                    matched = True
                    break

            if not matched:
                self._targets.append(TargetRecord(
                    track_id=self._parse_track_id(obj.object_id),
                    category=obj.category,
                    position_map=pos_map,
                    track_score=obj.score,
                    position_confidence=obj.confidence,
                    last_seen=now,
                    observation_count=1,
                ))
                self._total_added += 1

        # Post-hoc merge: collapse ACTIVE same-category targets within match_threshold
        self._merge_nearby_locked()

    def _merge_nearby_locked(self) -> None:
        """Merge ACTIVE same-category targets within match_threshold. Must hold lock."""
        merged = True
        while merged:
            merged = False
            for i in range(len(self._targets)):
                ti = self._targets[i]
                if ti.status != TargetStatus.ACTIVE:
                    continue
                for j in range(i + 1, len(self._targets)):
                    tj = self._targets[j]
                    if tj.status != TargetStatus.ACTIVE:
                        continue
                    if ti.category != tj.category:
                        continue
                    if _point_distance(ti.position_map, tj.position_map) < self._match_threshold:
                        # Merge j into i, weighted by observation count
                        total = ti.observation_count + tj.observation_count
                        w_i = ti.observation_count / total
                        w_j = tj.observation_count / total
                        ti.position_map.x = w_i * ti.position_map.x + w_j * tj.position_map.x
                        ti.position_map.y = w_i * ti.position_map.y + w_j * tj.position_map.y
                        ti.position_map.z = w_i * ti.position_map.z + w_j * tj.position_map.z
                        ti.track_score = max(ti.track_score, tj.track_score)
                        ti.observation_count += tj.observation_count
                        ti.last_seen = max(ti.last_seen, tj.last_seen)
                        ti.attempts = max(ti.attempts, tj.attempts)
                        del self._targets[j]
                        merged = True
                        break
                if merged:
                    break

    def is_viable(self, t: 'TargetRecord') -> bool:
        """Check if a target passes all navigation filters (no lock required, reads immutable params)."""
        now = time.time()
        return (
            t.status == TargetStatus.ACTIVE
            and t.attempts < self._max_attempts
            and t.observation_count >= self._min_observations
            and (self._target_ttl <= 0 or (now - t.last_seen) < self._target_ttl)
        )

    def get_nav_target(self, robot_pos_map: Point) -> Optional[TargetRecord]:
        """Return nearest active target that hasn't exceeded max attempts."""
        with self._lock:
            now = time.time()
            candidates = [
                t for t in self._targets
                if t.status == TargetStatus.ACTIVE
                and t.attempts < self._max_attempts
                and t.observation_count >= self._min_observations  # Fix 3
                and (self._target_ttl <= 0 or (now - t.last_seen) < self._target_ttl)  # Fix 1
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
            now = time.time()
            results = []
            for t in self._targets:
                if t.status != TargetStatus.ACTIVE:
                    continue
                if t.attempts >= self._max_attempts:
                    continue
                if t.observation_count < self._min_observations:  # Fix 3
                    continue
                if self._target_ttl > 0 and (now - t.last_seen) >= self._target_ttl:  # Fix 1
                    continue
                if in_working_area_fn(t.position_map):
                    results.append(t)
            results.sort(key=lambda t: _point_distance(t.position_map, robot_pos_map))
            return results

    def get_all_active(self, robot_pos_map: Point) -> List[TargetRecord]:
        """Return all ACTIVE targets passing TTL and min_observations filters, sorted by distance.
        Does NOT call in_working_area RPC — used for geometric pre-filter in SCANNING."""
        with self._lock:
            now = time.time()
            results = [
                t for t in self._targets
                if t.status == TargetStatus.ACTIVE
                and t.attempts < self._max_attempts
                and t.observation_count >= self._min_observations
                and (self._target_ttl <= 0 or (now - t.last_seen) < self._target_ttl)
            ]
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
            now = time.time()
            active = sum(1 for t in self._targets if t.status == TargetStatus.ACTIVE)
            viable = sum(
                1 for t in self._targets
                if t.status == TargetStatus.ACTIVE
                and t.attempts < self._max_attempts
                and t.observation_count >= self._min_observations
                and (self._target_ttl <= 0 or (now - t.last_seen) < self._target_ttl)
            )
            return {
                'total': len(self._targets),
                'active': active,
                'picked': self._total_picked,
                'failed': self._total_failed,
                'remaining': viable,
            }

    def get_filter_diagnostics(self, robot_pos_map: Optional[Point] = None) -> str:
        """返回每层过滤器的命中数，用于调试 get_nav_target 为空的原因。"""
        with self._lock:
            now = time.time()
            total = len(self._targets)
            active = sum(1 for t in self._targets if t.status == TargetStatus.ACTIVE)
            attempts_ok = sum(
                1 for t in self._targets
                if t.status == TargetStatus.ACTIVE and t.attempts < self._max_attempts
            )
            obs_ok = sum(
                1 for t in self._targets
                if t.status == TargetStatus.ACTIVE and t.attempts < self._max_attempts
                and t.observation_count >= self._min_observations
            )
            ttl_ok = sum(
                1 for t in self._targets
                if t.status == TargetStatus.ACTIVE and t.attempts < self._max_attempts
                and t.observation_count >= self._min_observations
                and (self._target_ttl <= 0 or (now - t.last_seen) < self._target_ttl)
            )
            # 显示第一个被各层卡住的目标信息
            sample = ""
            for t in self._targets[:3]:
                sample += (
                    f"[{t.category} obs={t.observation_count} "
                    f"age={now - t.last_seen:.1f}s "
                    f"status={t.status.name} att={t.attempts}] "
                )
            return (
                f"active={active}/{total} attempts_ok={attempts_ok} "
                f"obs>={self._min_observations}={obs_ok} ttl_ok={ttl_ok} "
                f"samples: {sample.strip()}"
            )

    def get_snapshot(self) -> List[TargetRecord]:
        """Thread-safe shallow copy of all target records (for publishing)."""
        with self._lock:
            return list(self._targets)

    def reset(self) -> None:
        with self._lock:
            self._targets.clear()
            self._total_added = 0
            self._total_picked = 0
            self._total_failed = 0
            self._update_enabled = True
