"""
target_filter.py — Grasp-suitability filter for Object3D detections.

Owns all grasping constraints (height, physical size, distance, costmap).
Config is updated at runtime via PerceptionConfig messages
(update_grasp_filter=True).

Deliberately separate from TargetPool: pool manages tracking lifetime,
this filter decides what enters the pool in the first place.
"""


class TargetFilter:
    """Filter Object3D detections for grasp suitability.

    Uses a fixed D435 focal length approximation for physical size
    estimation; accurate enough for go/no-go decisions.

    Costmap check: rejects targets whose map-frame position falls on
    occupied / inscribed cells in the global costmap.
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
                 logger=None):
        self.z_min    = z_min
        self.z_max    = z_max
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.size_min = size_min
        self.size_max = size_max
        self._costmap_max_cost = costmap_max_cost
        self._costmap = None      # nav_msgs/OccupancyGrid, set by node
        self._log     = logger

    # ------------------------------------------------------------------
    # Costmap
    # ------------------------------------------------------------------

    def set_costmap(self, costmap) -> None:
        """Update global costmap reference (nav_msgs/OccupancyGrid)."""
        self._costmap = costmap

    def _costmap_passable(self, x: float, y: float) -> bool:
        """Check if (x, y) in map frame is navigable in the global costmap.

        OccupancyGrid values: -1=unknown, 0=free, 1-98=inflated, 99=inscribed, 100=lethal.
        Returns True when no costmap is available (graceful degradation).
        """
        cm = self._costmap
        if cm is None:
            return True

        info = cm.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)

        if col < 0 or col >= info.width or row < 0 or row >= info.height:
            return False  # outside costmap bounds

        cost = cm.data[row * info.width + col]
        if cost < 0:
            return False  # unknown space
        return cost < self._costmap_max_cost

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

    def is_graspable(self, obj) -> bool:
        """Return True if obj passes all grasp-suitability checks."""
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

        # 4. Costmap navigability (map frame)
        if not self._costmap_passable(obj.position.x, obj.position.y):
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} costmap不可达 '
                    f'({obj.position.x:.2f}, {obj.position.y:.2f})')
            return False

        return True

    def filter(self, objects: list) -> list:
        """Return subset of objects passing all grasp-suitability checks."""
        return [o for o in objects if self.is_graspable(o)]
