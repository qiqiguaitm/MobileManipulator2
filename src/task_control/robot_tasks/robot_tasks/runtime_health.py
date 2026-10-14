"""Task runtime localization gate; legacy board_approach defaults stay unchanged."""

import math
import time
from board_approach.ros_io import stamp
from board_approach.safety import freshness_error, InputFreshnessError


def check_localization(status, minimum):
    if not math.isfinite(minimum) or not 0 < minimum <= 1:
        raise RuntimeError("localization_min_inlier_fraction must be in (0, 1]")
    if (
        not status.has_converged
        or not math.isfinite(status.inlier_fraction)
        or status.inlier_fraction < minimum
    ):
        raise RuntimeError(
            f"Localization unreliable: converged={status.has_converged}, "
            f"inlier_fraction={status.inlier_fraction:.3f}, required>={minimum:.3f}"
        )


class RuntimeHealth:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_parameter("localization_min_inlier_fraction", 0.80)

    def health(self, camera=False):
        keys = ["pose", "odom", "scan", "status"]
        if camera:
            keys += ["rgb", "info"] + (
                ["cloud", "deskew_odom"] if self.require_lidar else ["depth"]
            )
        error = freshness_error(time.monotonic(), self.received, keys)
        if error:
            raise InputFreshnessError(error)
        now = self.get_clock().now().nanoseconds
        for key in ("pose", "odom", "scan"):
            age = (now - stamp(self.data[key])) / 1e9
            if not -0.1 <= age <= 0.8:
                raise InputFreshnessError(
                    f"{key} message timestamp is stale: age={age:.3f}s, allowed=-0.100..0.800s"
                )
        check_localization(
            self.data["status"],
            self.get_parameter("localization_min_inlier_fraction").value,
        )
