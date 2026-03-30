"""
target_filter.py — Grasp-suitability filter for Object3D detections.

Owns all grasping constraints (height, physical size, distance, costmap).
Config is updated at runtime via PerceptionConfig messages
(update_grasp_filter=True).

Deliberately separate from TargetPool: pool manages tracking lifetime,
this filter decides what enters the pool in the first place.
"""

import math


class TargetFilter:
    """Filter Object3D detections for grasp suitability.

    Uses a fixed D435 focal length approximation for physical size
    estimation; accurate enough for go/no-go decisions.

    Costmap check: rejects targets whose position falls on occupied /
    inscribed cells.  Supports TF transform (map→costmap_frame) and
    nearby-radius search so objects near walls are not rejected when the
    robot can still reach them with its arm.
    """

    # Fallback focal length (RealSense D435 @ 1280x720)
    # 优先使用 obj.physical_size（感知节点用真实内参预算）
    _FX = 920.0
    _FY = 920.0

    def __init__(self,
                 z_min: float = -0.20,   # map frame; ground ≈ -0.12m, 此值允许地面以下 8cm
                 z_max: float = 0.35,    # map frame; 地面以上 ~47cm, 低于桌面高度
                 dist_min: float = 0.15,
                 dist_max: float = 6.0,
                 size_min: float = 0.02,
                 size_max: float = 0.20,
                 costmap_max_cost: int = 99,
                 nearby_radius: float = 0.5,
                 tf_buffer=None,
                 logger=None):
        self.z_min    = z_min
        self.z_max    = z_max
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.size_min = size_min
        self.size_max = size_max
        self._costmap_max_cost = costmap_max_cost
        self._nearby_radius = nearby_radius
        self._costmap = None      # nav_msgs/OccupancyGrid, set by node
        self._tf_buffer = tf_buffer
        self._log     = logger

    # ------------------------------------------------------------------
    # Costmap
    # ------------------------------------------------------------------

    def set_costmap(self, costmap) -> None:
        """Update global costmap reference (nav_msgs/OccupancyGrid)."""
        self._costmap = costmap

    # ------------------------------------------------------------------
    # TF: map → costmap frame
    # ------------------------------------------------------------------

    def _lookup_map_to_costmap_tf(self):
        """Return 2D affine (tx, ty, r00, r01, r10, r11) for map→costmap_frame.

        Returns None on failure (caller should fallback).
        """
        cm = self._costmap
        if cm is None or self._tf_buffer is None:
            return None

        frame = cm.header.frame_id
        if frame == 'map':
            return (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)  # identity

        try:
            from rclpy.time import Time
            import rclpy.duration
            tf = self._tf_buffer.lookup_transform(
                frame, 'map', Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            t = tf.transform.translation
            q = tf.transform.rotation
            # yaw from quaternion (2D rotation only)
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            c, s = math.cos(yaw), math.sin(yaw)
            return (t.x, t.y, c, -s, s, c)
        except Exception as e:
            if self._log:
                self._log.debug(f'[TargetFilter] TF map→{frame} failed: {e}')
            return None

    @staticmethod
    def _apply_tf2d(tf, x, y):
        """Apply 2D affine transform: (tx, ty, r00, r01, r10, r11)."""
        tx, ty, r00, r01, r10, r11 = tf
        return (r00 * x + r01 * y + tx,
                r10 * x + r11 * y + ty)

    # ------------------------------------------------------------------
    # Costmap passable check (approach-ring sampling)
    # ------------------------------------------------------------------

    # 8 directions on the approach ring (every 45°)
    _N_APPROACH_SAMPLES = 8
    _APPROACH_ANGLES = [2.0 * math.pi * i / 8 for i in range(8)]

    def _costmap_passable(self, x: float, y: float) -> bool:
        """Check if an approach point around (x, y) is navigable.

        The robot navigates to a point *approach_radius* away from the
        object, not to the object itself.  So we sample 8 points on
        the approach ring and accept the target if at least one
        approach direction is passable.

        When nearby_radius == 0, degrades to exact single-cell check
        at the object position (legacy behavior).

        Returns True when no costmap is available (graceful degradation).
        """
        cm = self._costmap
        if cm is None:
            return True

        info = cm.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        w = info.width
        h = info.height
        data = cm.data
        max_cost = self._costmap_max_cost
        approach_r = self._nearby_radius

        if approach_r <= 0 or res <= 0:
            # exact single-cell check at object position
            col = int((x - ox) / res) if res > 0 else 0
            row = int((y - oy) / res) if res > 0 else 0
            if col < 0 or col >= w or row < 0 or row >= h:
                return False
            cost = data[row * w + col]
            return 0 <= cost < max_cost

        # sample 8 points on the approach ring
        for angle in self._APPROACH_ANGLES:
            px = x + approach_r * math.cos(angle)
            py = y + approach_r * math.sin(angle)
            col = int((px - ox) / res)
            row = int((py - oy) / res)
            if col < 0 or col >= w or row < 0 or row >= h:
                continue
            cost = data[row * w + col]
            if 0 <= cost < max_cost:
                return True

        return False

    # ------------------------------------------------------------------
    # Runtime config update
    # ------------------------------------------------------------------

    def update_from_config(self, msg) -> None:
        """Apply grasp filter params from PerceptionConfig.
        No-op if msg.update_grasp_filter is False."""
        if not msg.update_grasp_filter:
            return
        self.z_min    = msg.grasp_z_min
        self.z_max    = msg.grasp_z_max
        self.dist_max = msg.grasp_distance_max
        self.size_min = msg.grasp_physical_min_size
        self.size_max = msg.grasp_physical_max_size
        if self._log:
            self._log.debug(
                f'[TargetFilter] updated: z=[{self.z_min},{self.z_max}]m '
                f'size=[{self.size_min},{self.size_max}]m '
                f'dist=[{self.dist_min},{self.dist_max}]m'
            )

    def is_graspable(self, obj, _costmap_xy=None) -> bool:
        """Return True if obj passes all grasp-suitability checks.

        Args:
            obj: Object3D detection.
            _costmap_xy: Pre-transformed (x, y) in costmap frame. If None,
                         uses obj.position.x/y directly (legacy behavior).
        """
        cat  = obj.category
        z    = obj.position.z
        dist = obj.distance

        # 1. Height (map frame z)
        if not (self.z_min <= z <= self.z_max):
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} z={z:.2f}m 超出map高度范围'
                    f'[{self.z_min},{self.z_max}]')
            return False

        # 2. Distance
        if dist < self.dist_min or dist > self.dist_max:
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} dist={dist:.2f}m '
                    f'超出范围[{self.dist_min},{self.dist_max}]m')
            return False

        # 3. Physical size — 优先用感知节点预算值，fallback 硬编码 fx
        phys_max = getattr(obj, 'physical_size', 0.0) or 0.0
        bbox  = obj.bbox
        depth = obj.depth
        if phys_max > 0:
            # 感知节点已用真实内参计算 physical_size (max dimension)
            # 估算 min dimension: bbox 短边/长边 × phys_max
            if bbox is not None and len(bbox) >= 4:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if max(bw, bh) > 0:
                    phys_min = min(bw, bh) / max(bw, bh) * phys_max
                else:
                    phys_min = phys_max
            else:
                phys_min = phys_max
        elif depth > 0.1 and bbox is not None and len(bbox) >= 4:
            # fallback: 硬编码焦距
            phys_w = (bbox[2] - bbox[0]) * depth / self._FX
            phys_h = (bbox[3] - bbox[1]) * depth / self._FY
            phys_max = max(phys_w, phys_h)
            phys_min = min(phys_w, phys_h)
        else:
            phys_max = phys_min = 0.0

        if phys_max > 0:
            if phys_max > self.size_max:
                if self._log:
                    self._log.debug(
                        f'[FILTER✗] {cat} 过大 '
                        f'phys_max={phys_max:.3f}m > {self.size_max}m')
                return False
            if phys_min < self.size_min:
                if self._log:
                    self._log.debug(
                        f'[FILTER✗] {cat} 过小 '
                        f'phys_min={phys_min:.3f}m < {self.size_min}m')
                return False

        # 4. Costmap navigability
        cx, cy = (_costmap_xy if _costmap_xy is not None
                  else (obj.position.x, obj.position.y))
        if not self._costmap_passable(cx, cy):
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} costmap不可达 '
                    f'map=({obj.position.x:.2f},{obj.position.y:.2f}) '
                    f'cm=({cx:.2f},{cy:.2f}) r={self._nearby_radius}m')
            return False

        return True

    def filter(self, objects: list) -> list:
        """Return subset of objects passing all grasp-suitability checks.

        Performs a single TF lookup (map→costmap_frame) and batch-
        transforms all object positions for costmap checking.
        """
        if not objects:
            return []

        tf = self._lookup_map_to_costmap_tf()

        result = []
        for o in objects:
            if tf is not None:
                cm_xy = self._apply_tf2d(tf, o.position.x, o.position.y)
            else:
                cm_xy = None  # fallback: use map coords directly
            if self.is_graspable(o, _costmap_xy=cm_xy):
                result.append(o)
        return result
