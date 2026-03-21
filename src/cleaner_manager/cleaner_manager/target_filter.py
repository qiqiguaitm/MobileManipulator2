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

    # RealSense D435 @ 1280x720 typical focal length
    _FX = 920.0
    _FY = 920.0

    def __init__(self,
                 z_min: float = -0.20,   # base_link frame; floor at -0.15m (chassis 15cm)
                 z_max: float = 0.35,    # 35cm above base_link = 50cm above floor
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

        # 1. Height (base_link z-axis)
        if not (self.z_min <= z <= self.z_max):
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} z={z:.2f}m 超出高度范围'
                    f'[{self.z_min},{self.z_max}]')
            return False

        # 2. Distance
        if dist < self.dist_min or dist > self.dist_max:
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} dist={dist:.2f}m '
                    f'超出范围[{self.dist_min},{self.dist_max}]m')
            return False

        # 3. Physical size (bbox + depth back-projection)
        bbox  = obj.bbox
        depth = obj.depth
        if depth > 0.1 and bbox is not None and len(bbox) >= 4:
            phys_w = (bbox[2] - bbox[0]) * depth / self._FX
            phys_h = (bbox[3] - bbox[1]) * depth / self._FY
            phys_max = max(phys_w, phys_h)
            phys_min = min(phys_w, phys_h)
            if phys_max > self.size_max:
                if self._log:
                    self._log.debug(
                        f'[FILTER✗] {cat} 过大 '
                        f'{phys_w:.2f}x{phys_h:.2f}m > {self.size_max}m')
                return False
            if phys_min < self.size_min:
                if self._log:
                    self._log.debug(
                        f'[FILTER✗] {cat} 过小 '
                        f'{phys_w:.2f}x{phys_h:.2f}m < {self.size_min}m')
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
