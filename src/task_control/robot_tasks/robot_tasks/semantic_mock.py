"""Mock semantic lookup by name, using a configured measured map-frame pose.

Only the lookup backend is mocked. Runtime must not estimate a surface normal.
Replace this provider with the Semantic Mapper client once its API is ready.
"""

import math


class SemanticMapMock:
    def __init__(self, washer_front_pose, frame="map"):
        self.washer = tuple(washer_front_pose)
        self.frame = frame

    def lookup_front_pose(self, name):
        if name not in ("洗衣机", "washer", "washing_machine"):
            raise KeyError("No semantic entry for " + name)
        if len(self.washer) != 4 or not all(math.isfinite(v) for v in self.washer):
            raise ValueError("Washer front pose must be finite [x, y, z, outward_yaw]")
        return self.frame, self.washer
