#!/usr/bin/env python3
"""
Approach Navigator - 三阶段精确接近导航器

公共接口:
    ApproachNavigator     - 主导航器类
    ApproachConfig        - 配置数据类
    ApproachResult        - 导航结果
    NavStage              - 导航阶段枚举
    compute_approach_pose - 计算接近位姿的工具函数

使用示例:
    from approach_navigator import ApproachNavigator, ApproachResult
    from geometry_msgs.msg import Point

    navigator = ApproachNavigator()
    target = Point(x=1.0, y=2.0, z=0.0)
    result = navigator.approach_to_target(target)

    if result.success:
        print(f"到达目标，最终距离: {result.final_distance}m")
"""

from .nav_types import NavStage, ApproachResult
from .config import ApproachConfig
from .navigator import ApproachNavigator
from .utils import compute_approach_pose

__all__ = [
    'ApproachNavigator',
    'ApproachConfig',
    'ApproachResult',
    'NavStage',
    'compute_approach_pose',
]
