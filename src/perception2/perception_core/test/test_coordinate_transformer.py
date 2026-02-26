#!/usr/bin/env python3
"""
测试坐标变换器
"""

import pytest
import numpy as np
from perception_core import CoordinateTransformer


class TestCoordinateTransformer:
    """CoordinateTransformer 单元测试"""

    def test_init(self):
        """测试初始化"""
        transformer = CoordinateTransformer()
        assert transformer.config_dir is None
        assert len(transformer._transforms) == 0

    def test_add_transform(self):
        """测试手动添加变换"""
        transformer = CoordinateTransformer()
        T = np.eye(4)
        transformer.add_transform('test', T)

        assert 'test' in transformer._transforms
        np.testing.assert_array_equal(transformer.get_transform('test'), T)

    def test_get_transform_not_found(self):
        """测试获取不存在的变换"""
        transformer = CoordinateTransformer()

        with pytest.raises(KeyError):
            transformer.get_transform('nonexistent')

    def test_transform_points_identity(self):
        """测试单位变换"""
        transformer = CoordinateTransformer()
        T = np.eye(4)

        points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        transformed = transformer.transform_points(points, T)
        np.testing.assert_array_almost_equal(transformed, points)

    def test_transform_points_translation(self):
        """测试平移变换"""
        transformer = CoordinateTransformer()

        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]  # 平移

        points = np.array([[0.0, 0.0, 0.0]])
        transformed = transformer.transform_points(points, T)

        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_transform_points_rotation_z(self):
        """测试绕 Z 轴旋转 90 度"""
        transformer = CoordinateTransformer()

        # 绕 Z 轴旋转 90 度
        T = np.eye(4)
        T[:3, :3] = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        points = np.array([[1.0, 0.0, 0.0]])
        transformed = transformer.transform_points(points, T)

        expected = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_array_almost_equal(transformed, expected, decimal=5)

    def test_transform_points_with_intensity(self):
        """测试带强度值的点云变换"""
        transformer = CoordinateTransformer()
        T = np.eye(4)
        T[:3, 3] = [1.0, 0.0, 0.0]

        # (x, y, z, intensity)
        points = np.array([[0.0, 0.0, 0.0, 100.0]])
        transformed = transformer.transform_points(points, T)

        # 强度值应该保持不变
        assert transformed.shape == (1, 4)
        np.testing.assert_almost_equal(transformed[0, 3], 100.0)
        np.testing.assert_array_almost_equal(transformed[0, :3], [1.0, 0.0, 0.0])

    def test_transform_points_empty(self):
        """测试空点云"""
        transformer = CoordinateTransformer()
        T = np.eye(4)

        points = np.empty((0, 3))
        transformed = transformer.transform_points(points, T)

        assert len(transformed) == 0

    def test_project_to_image(self):
        """测试投影到图像平面"""
        transformer = CoordinateTransformer()

        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
        }

        # 光轴上的点
        points = np.array([[0.0, 0.0, 1.0]])
        uv = transformer.project_to_image(points, intrinsics)

        expected = np.array([[320.0, 240.0]])
        np.testing.assert_array_almost_equal(uv, expected)

    def test_project_to_image_offset(self):
        """测试偏移点的投影"""
        transformer = CoordinateTransformer()

        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
        }

        # 向右偏移的点
        points = np.array([[0.5, 0.0, 1.0]])  # X=0.5, Z=1.0
        uv = transformer.project_to_image(points, intrinsics)

        # u = fx * X / Z + cx = 500 * 0.5 / 1.0 + 320 = 570
        expected = np.array([[570.0, 240.0]])
        np.testing.assert_array_almost_equal(uv, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
