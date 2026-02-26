"""
Perception Core - 感知核心算法库

这是一个纯 Python 库，不依赖 ROS。
包含在线检测服务客户端、坐标变换、3D 测量算法等。
"""

from .coordinate_transformer import CoordinateTransformer
from .scene_perception_core import ScenePerceptionCore, MeasurementResult, SimpleConfig
from .percept import (
    DinoXDetectorCloud,
    DinoXDetectorOnline,
    GraspAnythingOnline,
    SAM2TrackerOnline,
    DepthOptimizerOnline,
    LocalTaskResult,
)
from .dual_camera_matcher import (
    DualCameraMatcher,
    DetectionWithDepth,
    MatchPair,
    MatchQuality,
    CategoryCompatibility,
)
from .intelligent_fusion import fuse_dual_camera_positions

__version__ = "1.0.0"
__all__ = [
    "CoordinateTransformer",
    "ScenePerceptionCore",
    "MeasurementResult",
    "SimpleConfig",
    "DinoXDetectorCloud",
    "DinoXDetectorOnline",
    "GraspAnythingOnline",
    "SAM2TrackerOnline",
    "DepthOptimizerOnline",
    "LocalTaskResult",
    "DualCameraMatcher",
    "DetectionWithDepth",
    "MatchPair",
    "MatchQuality",
    "CategoryCompatibility",
    "fuse_dual_camera_positions",
]
