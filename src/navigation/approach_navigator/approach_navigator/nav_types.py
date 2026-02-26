#!/usr/bin/env python3
"""
Approach Navigator 类型定义
"""

from dataclasses import dataclass
from enum import IntEnum


class NavStage(IntEnum):
    """导航阶段枚举"""
    IDLE = 0            # 空闲状态
    NAVIGATING = 1      # 阶段1: Nav2 导航到接近点
    ALIGNING = 2        # 阶段2: 原地旋转对准目标
    FINAL_APPROACH = 3  # 阶段3: 相机深度测距精确接近
    ARRIVED = 4         # 已到达目标
    FAILED = 5          # 导航失败


@dataclass
class ApproachResult:
    """接近导航结果

    Attributes:
        success: 是否成功到达目标
        error_code: 错误代码 (NAV_FAILED, ALIGN_FAILED, APPROACH_FAILED, NAV_CANCELLED)
        error_message: 错误详细信息
        final_distance: 最终停止时与目标的距离 (米)
        stage: 失败发生在哪个阶段 (0=成功, 1/2/3=对应阶段失败)
    """
    success: bool                   # 是否成功
    error_code: str = ""            # 错误代码
    error_message: str = ""         # 错误信息
    final_distance: float = 0.0     # 最终距离 (米)
    stage: int = 0                  # 失败阶段 (0=成功)
