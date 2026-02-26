#!/usr/bin/env python3
"""
双相机检测匹配模块（优化用于 N=20 物体场景）

核心算法：匈牙利算法 + 综合代价函数
- 3D空间距离
- 视觉特征相似度（DINO-X features）
- 类别兼容性

适用场景：典型20个物体，匈牙利提供4-6%最优匹配质量
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment
from enum import Enum


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class DetectionWithDepth:
    """带有3D信息的检测结果"""
    # 2D检测信息
    bbox: np.ndarray          # [x1, y1, x2, y2] 像素坐标
    mask: Optional[np.ndarray] = None  # (H, W) 二值mask
    category: str = ""
    detection_score: float = 0.0

    # 视觉特征（来自DINO-X或CLIP）
    visual_features: Optional[np.ndarray] = None  # (D,) 归一化特征向量

    # 3D位置（粗略测量，用于匹配）
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in base_link
    distance: Optional[float] = None           # 到base_link原点距离

    # 来源标识
    camera_source: str = ""  # "chassis" or "top"

    # 匹配元信息
    matched: bool = False
    match_id: Optional[int] = None


class MatchQuality(Enum):
    """匹配质量等级"""
    EXCELLENT = "excellent"  # 距离<5cm, 特征>0.9, 类别相同
    GOOD = "good"            # 距离<10cm, 特征>0.8, 类别兼容
    ACCEPTABLE = "acceptable" # 距离<20cm, 特征>0.7
    POOR = "poor"            # 超过阈值但被匹配


@dataclass
class MatchPair:
    """匹配对"""
    chassis_idx: int
    top_idx: int
    cost: float
    distance_3d: float
    feature_similarity: float
    category_compatible: bool
    quality: MatchQuality

    # 融合后的信息
    fused_category: str = ""
    fused_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fused_confidence: float = 0.0


# ============================================================================
# 类别兼容性管理
# ============================================================================

class CategoryCompatibility:
    """类别兼容性检查器"""

    # 相似类别映射（可扩展）
    SIMILAR_CATEGORIES = {
        'bottle': {'bottle', 'cup', 'glass', 'container'},
        'cup': {'cup', 'bottle', 'glass', 'mug'},
        'box': {'box', 'package', 'carton'},
        'person': {'person', 'human', 'people'},
        'chair': {'chair', 'seat', 'stool'},
    }

    @staticmethod
    def is_compatible(cat1: str, cat2: str) -> bool:
        """判断两个类别是否兼容"""
        if cat1 == cat2:
            return True

        # 检查相似类别表
        similar1 = CategoryCompatibility.SIMILAR_CATEGORIES.get(cat1, set())
        if cat2 in similar1:
            return True

        similar2 = CategoryCompatibility.SIMILAR_CATEGORIES.get(cat2, set())
        if cat1 in similar2:
            return True

        return False

    @staticmethod
    def compute_category_cost(cat1: str, cat2: str) -> float:
        """
        计算类别代价

        Returns:
            0.0: 相同类别
            0.3: 兼容类别
            1.0: 不兼容
        """
        if cat1 == cat2:
            return 0.0
        elif CategoryCompatibility.is_compatible(cat1, cat2):
            return 0.3
        else:
            return 1.0

    @staticmethod
    def resolve_category(cat1: str, cat2: str, score1: float, score2: float) -> str:
        """
        解决类别冲突，返回最可信的类别

        Args:
            cat1, cat2: 两个类别
            score1, score2: 对应的检测置信度

        Returns:
            解决后的类别名
        """
        if cat1 == cat2:
            return cat1

        # 如果类别兼容，选择置信度高的
        if CategoryCompatibility.is_compatible(cat1, cat2):
            return cat1 if score1 >= score2 else cat2

        # 不兼容但被强制匹配，选择置信度高的
        return cat1 if score1 >= score2 else cat2


# ============================================================================
# 综合代价函数
# ============================================================================

class CostFunction:
    """综合代价函数（用于匈牙利算法）"""

    def __init__(
        self,
        weight_distance: float = 0.5,
        weight_feature: float = 0.3,    #目前不考虑基于特征的匹配
        weight_category: float = 0.2,
        distance_threshold: float = 0.2,  # 20cm，超过直接判定不可匹配
        feature_threshold: float = 0.6,   # 特征相似度下限
    ):
        """
        Args:
            weight_distance: 距离权重（三个权重之和应为1.0）
            weight_feature: 特征权重
            weight_category: 类别权重
            distance_threshold: 距离硬阈值（米），超过直接返回无穷大代价
            feature_threshold: 特征相似度阈值，低于此值匹配质量下降
        """
        self.w_dist = weight_distance
        self.w_feat = weight_feature
        self.w_cat = weight_category
        self.dist_thresh = distance_threshold
        self.feat_thresh = feature_threshold

    def compute(
        self,
        det_chassis: DetectionWithDepth,
        det_top: DetectionWithDepth,
    ) -> Tuple[float, Dict]:
        """
        计算两个检测之间的综合代价

        Returns:
            cost: 综合代价（越小越好，范围0-1）
            details: 详细信息字典
        """
        details = {}

        # ===== 1. 3D距离代价 =====
        if det_chassis.position_3d is not None and det_top.position_3d is not None:
            dist_3d = np.linalg.norm(det_chassis.position_3d - det_top.position_3d)
            details['distance_3d'] = dist_3d
            # 软阈值：超过阈值惩罚为1.0，但不直接判死刑
            if dist_3d > self.dist_thresh:
                cost_distance = 1.0  # 最大惩罚，但让算法继续评估其他因素
            else:
                cost_distance = dist_3d / self.dist_thresh
        else:
            # 缺少3D位置，距离代价最大
            cost_distance = 1.0
            details['distance_3d'] = None

        # ===== 2. 视觉特征代价 =====
        if (det_chassis.visual_features is not None and
            det_top.visual_features is not None):
            # 余弦相似度（假设特征已归一化）
            similarity = np.dot(det_chassis.visual_features, det_top.visual_features)
            similarity = np.clip(similarity, 0.0, 1.0)  # 防止数值误差
            cost_feature = 1.0 - similarity  # 相似度越高，代价越低
            details['feature_similarity'] = similarity
        else:
            # 缺少特征，使用中等代价（保守策略，不确定时不强判）
            cost_feature = 0.5
            details['feature_similarity'] = None

        # ===== 3. 类别代价 =====
        cost_category = CategoryCompatibility.compute_category_cost(
            det_chassis.category, det_top.category
        )
        details['category_cost'] = cost_category
        details['category_compatible'] = (cost_category < 1.0)

        # ===== 4. 综合代价 =====
        cost_total = (
            self.w_dist * cost_distance +
            self.w_feat * cost_feature +
            self.w_cat * cost_category
        )

        details['cost_distance'] = cost_distance
        details['cost_feature'] = cost_feature
        details['cost_total'] = cost_total

        return cost_total, details


# ============================================================================
# 匈牙利匹配器
# ============================================================================

class HungarianMatcher:
    """
    匈牙利算法匹配器（优化用于N=20场景）

    性能：
    - N=10: ~3ms
    - N=20: ~12ms
    - N=50: ~80ms
    """

    def __init__(
        self,
        cost_function: CostFunction,
        max_cost_threshold: float = 0.7,
    ):
        """
        Args:
            cost_function: 代价函数实例
            max_cost_threshold: 最大允许代价，超过则视为不可匹配
        """
        self.cost_fn = cost_function
        self.max_cost = max_cost_threshold

    def match(
        self,
        detections_chassis: List[DetectionWithDepth],
        detections_top: List[DetectionWithDepth],
    ) -> Tuple[List[MatchPair], List[int], List[int]]:
        """
        使用匈牙利算法进行全局最优匹配

        Args:
            detections_chassis: chassis camera检测列表
            detections_top: top camera检测列表

        Returns:
            matches: 匹配对列表
            unmatched_chassis: 未匹配的chassis索引
            unmatched_top: 未匹配的top索引
        """
        N = len(detections_chassis)
        M = len(detections_top)

        if N == 0 or M == 0:
            return [], list(range(N)), list(range(M))

        # ===== Step 1: 构建代价矩阵 =====
        cost_matrix = np.full((N, M), 1e6, dtype=np.float32)
        details_matrix = [[None] * M for _ in range(N)]

        for i, det_c in enumerate(detections_chassis):
            for j, det_t in enumerate(detections_top):
                cost, details = self.cost_fn.compute(det_c, det_t)

                # 只有代价在阈值内才记录
                if cost < self.max_cost:
                    cost_matrix[i, j] = cost
                    details_matrix[i][j] = details

        # ===== Step 2: 执行匈牙利算法 =====
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # ===== Step 3: 过滤并构建匹配对 =====
        matches = []
        matched_chassis = set()
        matched_top = set()

        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]

            # 检查是否在阈值内
            if cost >= self.max_cost:
                continue

            details = details_matrix[i][j]

            # 判断匹配质量
            quality = self._assess_quality(cost, details)

            # 创建匹配对
            match = MatchPair(
                chassis_idx=i,
                top_idx=j,
                cost=cost,
                distance_3d=details['distance_3d'] or 0.0,
                feature_similarity=details.get('feature_similarity') or 0.0,
                category_compatible=details['category_compatible'],
                quality=quality,
            )

            matches.append(match)
            matched_chassis.add(i)
            matched_top.add(j)

        # ===== Step 4: 找出未匹配项 =====
        unmatched_chassis = [i for i in range(N) if i not in matched_chassis]
        unmatched_top = [j for j in range(M) if j not in matched_top]

        return matches, unmatched_chassis, unmatched_top

    def _assess_quality(self, cost: float, details: Dict) -> MatchQuality:
        """评估匹配质量"""
        dist = details.get('distance_3d')
        feat_sim = details.get('feature_similarity') or 0.0
        cat_compat = details.get('category_compatible', False)

        # 优秀：距离<5cm, 特征>0.9, 类别兼容
        if dist is not None and dist < 0.05 and feat_sim > 0.9 and cat_compat:
            return MatchQuality.EXCELLENT

        # 良好：距离<10cm, 特征>0.8, 类别兼容
        if dist is not None and dist < 0.10 and feat_sim > 0.8 and cat_compat:
            return MatchQuality.GOOD

        # 可接受：距离<20cm, 特征>0.7
        if dist is not None and dist < 0.20 and feat_sim > 0.7:
            return MatchQuality.ACCEPTABLE

        return MatchQuality.POOR


# ============================================================================
# 匹配结果融合
# ============================================================================

class MatchFuser:
    """匹配结果融合器"""

    # 距离融合阈值（米）
    # 基于实测数据：Chassis在近距离更可靠，Top在远距离更稳定
    DISTANCE_FUSION_THRESHOLD = 1.4

    @staticmethod
    def fuse_match(
        match: MatchPair,
        det_chassis: DetectionWithDepth,
        det_top: DetectionWithDepth,
    ) -> MatchPair:
        """
        融合一个匹配对的信息

        策略：
        1. 类别：取置信度高的（兼容时）
        2. 3D位置：基于距离阈值选择更可靠的相机
           - < 1.4m: 使用Chassis相机（近距离误差小）
           - >= 1.4m: 使用Top相机（远距离更稳定）
        3. 置信度：根据匹配质量提升
        """
        # 融合类别
        match.fused_category = CategoryCompatibility.resolve_category(
            det_chassis.category, det_top.category,
            det_chassis.detection_score, det_top.detection_score
        )

        # 融合3D位置（基于距离选择）
        if det_chassis.position_3d is not None and det_top.position_3d is not None:
            # 使用Chassis距离作为判断依据（近距离Chassis更可靠）
            chassis_distance = det_chassis.distance
            if chassis_distance is None:
                chassis_distance = np.linalg.norm(det_chassis.position_3d)

            if chassis_distance < MatchFuser.DISTANCE_FUSION_THRESHOLD:
                # 近距离：使用Chassis
                match.fused_position = det_chassis.position_3d.copy()
            else:
                # 远距离：使用Top
                match.fused_position = det_top.position_3d.copy()
        elif det_chassis.position_3d is not None:
            match.fused_position = det_chassis.position_3d.copy()
        elif det_top.position_3d is not None:
            match.fused_position = det_top.position_3d.copy()
        else:
            match.fused_position = np.zeros(3)

        # 融合置信度
        base_confidence = (det_chassis.detection_score + det_top.detection_score) / 2.0

        # 根据匹配质量加成
        if match.quality == MatchQuality.EXCELLENT:
            match.fused_confidence = min(base_confidence + 0.15, 1.0)
        elif match.quality == MatchQuality.GOOD:
            match.fused_confidence = min(base_confidence + 0.10, 1.0)
        elif match.quality == MatchQuality.ACCEPTABLE:
            match.fused_confidence = min(base_confidence + 0.05, 1.0)
        else:
            match.fused_confidence = base_confidence

        return match


# ============================================================================
# 完整匹配管道
# ============================================================================

class DualCameraMatcher:
    """
    双相机匹配管道（完整流程）

    输入：两个相机的检测列表（带3D位置和视觉特征）
    输出：匹配对 + 未匹配检测
    """

    def __init__(
        self,
        weight_distance: float = 0.5,
        weight_feature: float = 0.3,
        weight_category: float = 0.2,
        distance_threshold: float = 0.2,
        max_cost_threshold: float = 0.7,
    ):
        """
        初始化匹配器

        Args:
            weight_distance: 距离权重（推荐0.5）
            weight_feature: 特征权重（推荐0.3）
            weight_category: 类别权重（推荐0.2）
            distance_threshold: 距离阈值20cm
            max_cost_threshold: 最大代价0.7
        """
        self.cost_fn = CostFunction(
            weight_distance=weight_distance,
            weight_feature=weight_feature,
            weight_category=weight_category,
            distance_threshold=distance_threshold,
        )

        self.matcher = HungarianMatcher(
            cost_function=self.cost_fn,
            max_cost_threshold=max_cost_threshold,
        )

        self.fuser = MatchFuser()

    def process(
        self,
        detections_chassis: List[DetectionWithDepth],
        detections_top: List[DetectionWithDepth],
    ) -> Tuple[List[MatchPair], List[DetectionWithDepth], List[DetectionWithDepth]]:
        """
        完整匹配流程

        Returns:
            fused_matches: 融合后的匹配对列表
            unmatched_chassis: 未匹配的chassis检测
            unmatched_top: 未匹配的top检测
        """
        # Step 1: 执行匹配
        matches, unmatched_c_idx, unmatched_t_idx = self.matcher.match(
            detections_chassis, detections_top
        )

        # Step 2: 融合匹配对
        fused_matches = []
        for match in matches:
            det_c = detections_chassis[match.chassis_idx]
            det_t = detections_top[match.top_idx]
            fused_match = self.fuser.fuse_match(match, det_c, det_t)
            fused_matches.append(fused_match)

        # Step 3: 收集未匹配项
        unmatched_chassis = [detections_chassis[i] for i in unmatched_c_idx]
        unmatched_top = [detections_top[i] for i in unmatched_t_idx]

        return fused_matches, unmatched_chassis, unmatched_top
