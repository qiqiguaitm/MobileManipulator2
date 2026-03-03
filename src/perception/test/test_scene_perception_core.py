#!/usr/bin/env python3
"""
测试场景感知核心算法
"""

import pytest
import numpy as np
from perception.coordinate_transformer import CoordinateTransformer
from perception.scene_perception_core import ScenePerceptionCore, MeasurementResult


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


class TestNearRangeDepth:
    """近距离深度处理测试"""

    @pytest.fixture
    def setup_core(self):
        transformer = CoordinateTransformer()
        transformer.add_transform('optical_to_base', np.eye(4))
        intrinsics = {
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0,
            'width': 640, 'height': 480,
        }
        return ScenePerceptionCore(
            transformer=transformer,
            intrinsics=intrinsics,
            target_frame='base_link',
        )

    def _make_depth_mask(self, depth_val, zero_ratio=0.0, bg_ratio=0.0, bg_depth=2.0):
        """构造测试深度图和 mask

        Args:
            depth_val: 物体深度 (m)
            zero_ratio: 零值像素占比 (0~1)
            bg_ratio: 背景穿透像素占比 (0~1)
            bg_depth: 背景穿透深度 (m)

        Returns:
            (depth, mask) — 480x640
        """
        depth = np.zeros((480, 640), dtype=np.float32)
        mask = np.zeros((480, 640), dtype=np.uint8)

        # mask 区域: 200:280, 280:360 = 80x80 = 6400 像素
        # 腐蚀后 (5x5 kernel) 大约 70x70 = 4900 有效像素
        mask[200:280, 280:360] = 1

        # 填充深度
        roi = depth[200:280, 280:360]
        h, w = roi.shape
        total = h * w
        n_zero = int(total * zero_ratio)
        n_bg = int(total * bg_ratio)
        n_valid = total - n_zero - n_bg

        # 按行优先排列: 前 n_valid 个有效, 然后 n_bg 个背景, 最后 n_zero 个零
        flat = roi.ravel()
        flat[:n_valid] = depth_val + np.random.normal(0, 0.01, n_valid)
        flat[n_valid:n_valid + n_bg] = bg_depth
        # flat[n_valid+n_bg:] 保持 0
        np.random.shuffle(flat)
        depth[200:280, 280:360] = flat.reshape(h, w)

        return depth, mask

    def test_too_close_all_zero(self, setup_core):
        """全零深度 → 丢弃"""
        depth, mask = self._make_depth_mask(0.0, zero_ratio=1.0)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is False
        assert "zero" in result.error_msg.lower() or "No valid" in result.error_msg

    def test_too_close_mostly_zero(self, setup_core):
        """90% 零值 (~0.15m) → 丢弃"""
        depth, mask = self._make_depth_mask(0.15, zero_ratio=0.90, bg_ratio=0.02)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is False

    def test_too_close_35pct_zero(self, setup_core):
        """35% 零值 → 超过 30% 阈值，丢弃"""
        depth, mask = self._make_depth_mask(0.22, zero_ratio=0.35, bg_ratio=0.05)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is False

    def test_near_range_valid(self, setup_core):
        """0.25m 目标, 20% 零值 → 近距离路径，测量成功"""
        depth, mask = self._make_depth_mask(0.25, zero_ratio=0.20, bg_ratio=0.05)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is True
        assert 0.20 < result.stats['depth_median'] < 0.30

    def test_near_range_with_background(self, setup_core):
        """0.25m 目标, 20% 零值 + 10% 背景穿透 → 背景被 0.5m 上限切掉"""
        depth, mask = self._make_depth_mask(0.25, zero_ratio=0.20, bg_ratio=0.10, bg_depth=1.5)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is True
        assert 0.20 < result.stats['depth_median'] < 0.30

    def test_normal_range_unchanged(self, setup_core):
        """1.5m 目标 → 正常路径，不受近距离逻辑影响"""
        depth, mask = self._make_depth_mask(1.5, zero_ratio=0.05)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is True
        assert 1.3 < result.stats['depth_median'] < 1.7

    def test_batch_near_range_discard(self, setup_core):
        """batch 路径: 90% 零值 → 丢弃"""
        depth, mask = self._make_depth_mask(0.15, zero_ratio=0.90)
        detections = [{'mask': mask, 'bbox': [280, 200, 360, 280]}]
        results = setup_core.batch_camera_3d_percept(depth, detections)
        assert len(results) == 1
        assert results[0].valid is False

    def test_batch_near_range_valid(self, setup_core):
        """batch 路径: 0.25m, 20% 零值 → 测量成功"""
        depth, mask = self._make_depth_mask(0.25, zero_ratio=0.20, bg_ratio=0.05)
        detections = [{'mask': mask, 'bbox': [280, 200, 360, 280]}]
        results = setup_core.batch_camera_3d_percept(depth, detections)
        assert len(results) == 1
        assert results[0].valid is True
        assert 0.20 < results[0].stats['depth_median'] < 0.30

    def test_027m_no_jump(self, setup_core):
        """0.27m + 30% 背景穿透 → p25 触发近距离 + 深度剔除砍背景，无跳变"""
        depth, mask = self._make_depth_mask(0.27, zero_ratio=0.15, bg_ratio=0.30, bg_depth=1.5)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is True
        # 关键：中值应在 0.27m 附近，不能跳到 1.0m+
        assert 0.20 < result.stats['depth_median'] < 0.35

    def test_depth_outlier_filter_normal(self, setup_core):
        """0.8m + 15% 背景穿透 → 深度剔除砍背景，IQR 正常"""
        depth, mask = self._make_depth_mask(0.8, zero_ratio=0.03, bg_ratio=0.15, bg_depth=3.0)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is True
        assert 0.7 < result.stats['depth_median'] < 0.9

    def test_depth_outlier_filter_no_trigger(self, setup_core):
        """1.5m 单峰无背景 → 深度剔除不触发"""
        depth, mask = self._make_depth_mask(1.5, zero_ratio=0.02)
        result = setup_core.camera_3d_percept(depth, mask)
        assert result.valid is True
        assert 1.3 < result.stats['depth_median'] < 1.7

    def test_batch_027m_no_jump(self, setup_core):
        """batch 路径: 0.27m + 背景穿透 → 无跳变"""
        depth, mask = self._make_depth_mask(0.27, zero_ratio=0.15, bg_ratio=0.30, bg_depth=1.5)
        detections = [{'mask': mask, 'bbox': [280, 200, 360, 280]}]
        results = setup_core.batch_camera_3d_percept(depth, detections)
        assert len(results) == 1
        assert results[0].valid is True
        assert 0.20 < results[0].stats['depth_median'] < 0.35

    def test_batch_normal_range_unchanged(self, setup_core):
        """batch 路径: 1.5m → 正常路径不受影响"""
        depth, mask = self._make_depth_mask(1.5, zero_ratio=0.05)
        detections = [{'mask': mask, 'bbox': [280, 200, 360, 280]}]
        results = setup_core.batch_camera_3d_percept(depth, detections)
        assert len(results) == 1
        assert results[0].valid is True
        assert 1.3 < results[0].stats['depth_median'] < 1.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
