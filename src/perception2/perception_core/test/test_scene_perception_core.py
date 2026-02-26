#!/usr/bin/env python3
"""
测试场景感知核心算法
"""

import pytest
import numpy as np
from perception_core import CoordinateTransformer, ScenePerceptionCore, MeasurementResult


class TestMeasurementResult:
    """MeasurementResult 数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        result = MeasurementResult()
        assert result.valid is False
        assert result.confidence == 0.0
        assert result.centroid is None
        assert result.distance == 0.0
        assert result.error_msg is None

    def test_custom_values(self):
        """测试自定义值"""
        result = MeasurementResult(
            valid=True,
            confidence=0.95,
            centroid=np.array([1.0, 2.0, 3.0]),
            distance=3.74
        )
        assert result.valid is True
        assert result.confidence == 0.95
        np.testing.assert_array_equal(result.centroid, [1.0, 2.0, 3.0])


class TestScenePerceptionCore:
    """ScenePerceptionCore 单元测试"""

    @pytest.fixture
    def setup_core(self):
        """设置测试环境"""
        transformer = CoordinateTransformer()
        # 添加单位变换用于测试
        transformer.add_transform('optical_to_base', np.eye(4))
        transformer.add_transform('optical_to_arm', np.eye(4))

        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
            'width': 640, 'height': 480,
        }

        core = ScenePerceptionCore(
            transformer=transformer,
            intrinsics=intrinsics,
            target_frame='base_link',
        )
        return core

    def test_init(self, setup_core):
        """测试初始化"""
        core = setup_core
        assert core.target_frame == 'base_link'
        assert core.intrinsics['fx'] == 500.0

    def test_init_invalid_frame(self):
        """测试无效目标坐标系"""
        transformer = CoordinateTransformer()
        intrinsics = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}

        with pytest.raises(ValueError):
            ScenePerceptionCore(
                transformer=transformer,
                intrinsics=intrinsics,
                target_frame='invalid_frame',
            )

    def test_camera_3d_percept_valid(self, setup_core):
        """测试有效的相机 3D 感知"""
        core = setup_core

        # 创建测试深度图和 mask
        depth = np.ones((480, 640), dtype=np.float32) * 1.5  # 1.5m
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:280, 280:360] = 1  # 中心区域

        result = core.camera_3d_percept(depth, mask)

        assert result.valid is True
        assert result.confidence > 0
        assert result.centroid is not None
        assert len(result.centroid) == 3
        assert result.distance > 0

    def test_camera_3d_percept_small_mask(self, setup_core):
        """测试太小的 mask"""
        core = setup_core

        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[240, 320] = 1  # 只有一个像素

        result = core.camera_3d_percept(depth, mask)

        assert result.valid is False
        assert result.error_msg is not None

    def test_camera_3d_percept_invalid_depth(self, setup_core):
        """测试无效深度值"""
        core = setup_core

        depth = np.zeros((480, 640), dtype=np.float32)  # 全零深度
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:280, 280:360] = 1

        result = core.camera_3d_percept(depth, mask)

        assert result.valid is False

    def test_camera_3d_percept_skip_transform(self, setup_core):
        """测试跳过坐标变换"""
        core = setup_core

        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:280, 280:360] = 1

        result = core.camera_3d_percept(depth, mask, skip_transform=True)

        assert result.valid is True
        # 结果应该在 optical 坐标系
        assert result.centroid_optical is not None

    def test_compute_confidence(self, setup_core):
        """测试置信度计算"""
        core = setup_core

        # 高质量测量：多点、低标准差
        conf_high = core._compute_confidence(num_points=200, depth_std=0.05)

        # 低质量测量：少点、高标准差
        conf_low = core._compute_confidence(num_points=20, depth_std=0.25)

        assert conf_high > conf_low
        assert 0 <= conf_high <= 1
        assert 0 <= conf_low <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
