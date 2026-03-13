"""
target_filter.py — Grasp-suitability filter for Object3D detections.

Owns all grasping constraints (height, physical size, distance).
Config is updated at runtime via PerceptionConfig messages
(update_grasp_filter=True).

Deliberately separate from TargetPool: pool manages tracking lifetime,
this filter decides what enters the pool in the first place.
"""


class TargetFilter:
    """Filter Object3D detections for grasp suitability.

    Uses a fixed D435 focal length approximation for physical size
    estimation; accurate enough for go/no-go decisions.
    """

    # RealSense D435 @ 1280x720 typical focal length
    _FX = 920.0
    _FY = 920.0

    def __init__(self,
                 z_min: float = -0.20,   # base_link frame; floor at -0.15m (chassis 15cm)
                 z_max: float = 0.35,    # 35cm above base_link = 50cm above floor

                 dist_max: float = 6.0,
                 size_min: float = 0.02,
                 size_max: float = 0.20,
                 logger=None):
        self.z_min    = z_min
        self.z_max    = z_max
        self.dist_max = dist_max
        self.size_min = size_min
        self.size_max = size_max
        self._log     = logger

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
                f'dist_max={self.dist_max}m'
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
        if dist > self.dist_max:
            if self._log:
                self._log.debug(
                    f'[FILTER✗] {cat} dist={dist:.2f}m '
                    f'超出最远距离{self.dist_max}m')
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

        return True

    def filter(self, objects: list) -> list:
        """Return subset of objects passing all grasp-suitability checks."""
        return [o for o in objects if self.is_graspable(o)]
