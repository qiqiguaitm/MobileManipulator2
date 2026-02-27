#!/usr/bin/env python3
"""
测试 dual_camera_matcher 模块
"""

import pytest
import numpy as np

from perception.dual_camera_matcher import (
    DualCameraMatcher,
    DetectionWithDepth,
    MatchPair,
    MatchQuality,
    CategoryCompatibility,
    CostFunction,
    HungarianMatcher,
)


class TestCategoryCompatibility:
    """类别兼容性测试"""

    def test_same_category(self):
        """相同类别应该兼容"""
        assert CategoryCompatibility.is_compatible('bottle', 'bottle')
        assert CategoryCompatibility.compute_category_cost('bottle', 'bottle') == 0.0

    def test_similar_categories(self):
        """相似类别应该兼容"""
        assert CategoryCompatibility.is_compatible('bottle', 'cup')
        assert CategoryCompatibility.compute_category_cost('bottle', 'cup') == 0.3

    def test_incompatible_categories(self):
        """不同类别应该不兼容"""
        assert not CategoryCompatibility.is_compatible('bottle', 'chair')
        assert CategoryCompatibility.compute_category_cost('bottle', 'chair') == 1.0

    def test_resolve_category_same(self):
        """相同类别解析"""
        result = CategoryCompatibility.resolve_category('bottle', 'bottle', 0.8, 0.9)
        assert result == 'bottle'

    def test_resolve_category_compatible(self):
        """兼容类别解析（选择置信度高的）"""
        result = CategoryCompatibility.resolve_category('bottle', 'cup', 0.8, 0.9)
        assert result == 'cup'


class TestCostFunction:
    """代价函数测试"""

    def test_identical_detections(self):
        """相同检测的代价应该很低"""
        cost_fn = CostFunction()

        position = np.array([1.0, 0.0, 0.5])
        features = np.random.randn(512)
        features = features / np.linalg.norm(features)

        det1 = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=position,
            visual_features=features,
        )
        det2 = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=position,
            visual_features=features,
        )

        cost, details = cost_fn.compute(det1, det2)
        assert cost < 0.1
        assert details['distance_3d'] == 0.0
        assert details['feature_similarity'] == pytest.approx(1.0, abs=0.01)
        assert details['category_compatible']

    def test_different_positions(self):
        """不同位置的代价应该较高"""
        cost_fn = CostFunction(distance_threshold=0.2)

        det1 = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=np.array([0.0, 0.0, 0.5]),
        )
        det2 = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=np.array([0.3, 0.0, 0.5]),  # 30cm 偏移
        )

        cost, details = cost_fn.compute(det1, det2)
        assert details['distance_3d'] == pytest.approx(0.3, abs=0.01)
        # 距离超过阈值，代价应该较高
        assert cost > 0.3


class TestHungarianMatcher:
    """匈牙利匹配器测试"""

    def test_empty_input(self):
        """空输入测试"""
        cost_fn = CostFunction()
        matcher = HungarianMatcher(cost_fn)

        matches, unmatched_c, unmatched_t = matcher.match([], [])
        assert len(matches) == 0
        assert len(unmatched_c) == 0
        assert len(unmatched_t) == 0

    def test_one_to_one_match(self):
        """一对一匹配测试"""
        cost_fn = CostFunction()
        matcher = HungarianMatcher(cost_fn)

        det_c = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=np.array([1.0, 0.0, 0.5]),
        )
        det_t = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=np.array([1.02, 0.0, 0.5]),  # 2cm 偏移
        )

        matches, unmatched_c, unmatched_t = matcher.match([det_c], [det_t])
        assert len(matches) == 1
        assert len(unmatched_c) == 0
        assert len(unmatched_t) == 0
        assert matches[0].chassis_idx == 0
        assert matches[0].top_idx == 0


class TestDualCameraMatcher:
    """完整匹配管道测试"""

    def test_basic_matching(self):
        """基本匹配测试"""
        matcher = DualCameraMatcher()

        # 创建两个相似的检测
        det_c = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=np.array([1.0, 0.0, 0.5]),
        )
        det_t = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.85,
            position_3d=np.array([1.03, 0.01, 0.52]),  # 小偏移
        )

        matches, unmatched_c, unmatched_t = matcher.process([det_c], [det_t])

        assert len(matches) == 1
        assert len(unmatched_c) == 0
        assert len(unmatched_t) == 0

        # 检查融合结果
        match = matches[0]
        assert match.fused_category == 'bottle'
        assert match.fused_confidence > 0.8

    def test_unmatched_detections(self):
        """未匹配检测测试"""
        matcher = DualCameraMatcher(distance_threshold=0.1)

        det_c = DetectionWithDepth(
            bbox=np.array([100, 100, 200, 200]),
            category='bottle',
            detection_score=0.9,
            position_3d=np.array([1.0, 0.0, 0.5]),
        )
        det_t = DetectionWithDepth(
            bbox=np.array([300, 300, 400, 400]),
            category='cup',
            detection_score=0.85,
            position_3d=np.array([2.0, 1.0, 1.0]),  # 完全不同的位置
        )

        matches, unmatched_c, unmatched_t = matcher.process([det_c], [det_t])

        # 由于位置差距太大，应该无法匹配
        assert len(matches) == 0
        assert len(unmatched_c) == 1
        assert len(unmatched_t) == 1


class TestMatchQuality:
    """匹配质量测试"""

    def test_quality_levels(self):
        """测试质量等级"""
        assert MatchQuality.EXCELLENT.value == "excellent"
        assert MatchQuality.GOOD.value == "good"
        assert MatchQuality.ACCEPTABLE.value == "acceptable"
        assert MatchQuality.POOR.value == "poor"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
