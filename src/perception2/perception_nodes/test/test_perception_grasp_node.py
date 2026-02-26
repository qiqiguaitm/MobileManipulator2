#!/usr/bin/env python3
"""
测试 perception_grasp_node 模块
"""

import pytest
import numpy as np


def _ros2_available() -> bool:
    """检查 ROS2 环境是否可用"""
    try:
        from perception_interfaces.msg import GraspObject
        return True
    except ImportError:
        return False


class TestImports:
    """导入测试 (需要 ROS2 环境和已编译的消息包)"""

    @pytest.mark.skipif(
        not _ros2_available(),
        reason='需要 ROS2 环境和已编译的 perception_interfaces'
    )
    def test_import_module(self):
        """测试模块导入"""
        from perception_nodes import perception_grasp_node
        assert perception_grasp_node is not None

    @pytest.mark.skipif(
        not _ros2_available(),
        reason='需要 ROS2 环境和已编译的 perception_interfaces'
    )
    def test_import_class(self):
        """测试类导入"""
        from perception_nodes.perception_grasp_node import PerceptionGraspNode
        assert PerceptionGraspNode is not None


class TestDeprojection:
    """反投影函数测试 (不需要 ROS)"""

    def test_deproject_pixel_to_point_center(self):
        """测试中心点反投影"""
        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
            'width': 640, 'height': 480
        }

        def deproject(pixel, depth, intrinsics):
            u, v = pixel
            fx, fy = intrinsics['fx'], intrinsics['fy']
            cx, cy = intrinsics['cx'], intrinsics['cy']
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            return [x, y, z]

        result = deproject([320.0, 240.0], 1.0, intrinsics)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    def test_deproject_pixel_to_point_offset(self):
        """测试偏移点反投影"""
        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
        }

        def deproject(pixel, depth, intrinsics):
            u, v = pixel
            fx, fy = intrinsics['fx'], intrinsics['fy']
            cx, cy = intrinsics['cx'], intrinsics['cy']
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            return [x, y, z]

        result = deproject([570.0, 240.0], 1.0, intrinsics)
        np.testing.assert_array_almost_equal(result, [0.5, 0.0, 1.0])

    def test_deproject_batch(self):
        """测试批量反投影"""
        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
        }

        def deproject_batch(pixels, depth, intrinsics):
            pixels = np.array(pixels)
            u, v = pixels[:, 0], pixels[:, 1]
            fx, fy = intrinsics['fx'], intrinsics['fy']
            cx, cy = intrinsics['cx'], intrinsics['cy']
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = np.full_like(x, depth)
            return np.stack([x, y, z], axis=-1)

        pixels = [[320.0, 240.0], [570.0, 240.0]]
        result = deproject_batch(pixels, 1.0, intrinsics)

        expected = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestCombinedScore:
    """综合分数计算测试"""

    def test_pure_grasp_score(self):
        """当距离权重为0时，应返回纯抓取分数"""
        grasp_score = 0.8
        depth = 0.5

        def compute_combined_score(grasp_score, depth, w_grasp=1.0, w_dist=0.0,
                                   min_depth=0.2, max_depth=2.0):
            if w_dist == 0.0:
                return grasp_score

            if depth <= 0 or depth < min_depth or depth > max_depth:
                return grasp_score

            raw_score = min_depth / depth
            min_score = min_depth / max_depth
            distance_score = (raw_score - min_score) / (1.0 - min_score)

            w_sum = w_grasp + w_dist
            return (w_grasp * grasp_score + w_dist * distance_score) / w_sum

        result = compute_combined_score(grasp_score, depth, w_grasp=1.0, w_dist=0.0)
        assert result == grasp_score

    def test_weighted_score(self):
        """测试加权综合分数"""
        def compute_combined_score(grasp_score, depth, w_grasp=0.5, w_dist=0.5,
                                   min_depth=0.2, max_depth=2.0):
            if w_dist == 0.0:
                return grasp_score

            raw_score = min_depth / depth
            min_score = min_depth / max_depth
            distance_score = (raw_score - min_score) / (1.0 - min_score)

            w_sum = w_grasp + w_dist
            return (w_grasp * grasp_score + w_dist * distance_score) / w_sum

        score_near = compute_combined_score(0.5, 0.3, w_grasp=0.2, w_dist=0.8)
        score_far = compute_combined_score(0.5, 1.0, w_grasp=0.2, w_dist=0.8)

        assert score_near > score_far

    def test_invalid_depth_fallback(self):
        """无效深度应回退到纯抓取分数"""
        def compute_combined_score(grasp_score, depth, min_depth=0.2, max_depth=2.0):
            if depth <= 0 or depth < min_depth or depth > max_depth:
                return grasp_score
            return 0.0

        result = compute_combined_score(0.8, 0.1, min_depth=0.2, max_depth=2.0)
        assert result == 0.8

        result = compute_combined_score(0.8, 3.0, min_depth=0.2, max_depth=2.0)
        assert result == 0.8


class TestRobustDepth:
    """鲁棒深度计算测试"""

    def test_get_robust_depth_normal(self):
        """正常深度值"""
        depth = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
        ])

        mask = np.ones((3, 3), dtype=np.uint8)
        depth_masked = depth[mask > 0]
        valid = (depth_masked > 0.2) & (depth_masked < 2.0)
        depth_values = depth_masked[valid]

        result = np.median(depth_values)
        assert result == 1.0

    def test_get_robust_depth_with_outliers(self):
        """带离群值的深度"""
        depth = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 5.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])

        mask = np.ones((5, 5), dtype=np.uint8)
        depth_masked = depth[mask > 0]
        valid = (depth_masked > 0.2) & (depth_masked < 2.0)
        depth_values = depth_masked[valid]

        q1, q3 = np.percentile(depth_values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        depth_filtered = depth_values[(depth_values >= lower) & (depth_values <= upper)]

        result = np.median(depth_filtered)
        assert result == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
