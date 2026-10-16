"""Default storage location for new captures, separate from accepted scenes."""

import os
from pathlib import Path


def capture_output():
    override = os.environ.get("NAVIGATION_SCENARIOS_ROOT")
    if override:
        root = Path(override).expanduser()
    else:
        robot_root = Path("/data/junbin.zhang/navigation_scenarios")
        root = (
            robot_root
            if robot_root.is_dir()
            else Path(__file__).resolve().parents[2] / "data/navigation_scenarios"
        )
    return root / "pending"
