#!/usr/bin/env python3
"""
相机精度置信度模型 (基于实验数据)

基于实际测量数据的相机精度边界：
- Chassis相机：大物体近距离准，远距离崩溃
- Top相机：大物体全程准，小物体近距离不准

用于双相机融合时的智能加权。
"""

import numpy as np
from typing import Dict, Tuple


class CameraConfidenceModel:
    """
    相机测量置信度模型

    基于实验数据（/docs/experiments/results/）提炼的规律：

    Chassis相机:
    - 大物体: ≤1.10m准确, ≥1.20m退化
    - 小物体: ≤2.00m准确, ≥2.50m退化

    Top相机:
    - 大物体: 全程准确
    - 小物体: ≤1.28m退化(10cm误差), ≥1.50m准确
    """

    # 精度边界常量（米）
    CHASSIS_LARGE_ACCURATE_MAX = 1.10
    CHASSIS_LARGE_DEGRADED_MIN = 1.20
    CHASSIS_SMALL_ACCURATE_MAX = 2.00
    CHASSIS_SMALL_DEGRADED_MIN = 2.50

    TOP_SMALL_DEGRADED_MAX = 1.28
    TOP_SMALL_ACCURATE_MIN = 1.50

    # 物体尺寸判断阈值（米）
    SIZE_THRESHOLD_HEIGHT = 0.20  # 高度>20cm视为大物体
    SIZE_THRESHOLD_WIDTH = 0.15   # 宽度>15cm视为大物体（用于俯视）

    @staticmethod
    def estimate_object_size(bbox: np.ndarray, distance: float,
                            intrinsics: Dict, camera: str) -> str:
        """
        从bbox反推物体真实尺寸

        Args:
            bbox: [x1, y1, x2, y2] 像素坐标
            distance: 物体距离（米）
            intrinsics: 相机内参 {fx, fy, cx, cy}
            camera: 'chassis' or 'top'

        Returns:
            'large' or 'small'
        """
        x1, y1, x2, y2 = bbox
        bbox_height_px = y2 - y1
        bbox_width_px = x2 - x1

        # 反投影到3D空间估计真实尺寸
        fy = intrinsics.get('fy', 400)
        fx = intrinsics.get('fx', 400)

        estimated_height = bbox_height_px * distance / fy
        estimated_width = bbox_width_px * distance / fx

        # 根据相机视角选择判断依据
        if camera == 'chassis':
            # Chassis前视：高度更可靠
            return 'large' if estimated_height > CameraConfidenceModel.SIZE_THRESHOLD_HEIGHT else 'small'
        else:  # top
            # Top俯视：高度不可见，用宽度或最大维度
            size_metric = max(estimated_width, estimated_height)
            return 'large' if size_metric > CameraConfidenceModel.SIZE_THRESHOLD_WIDTH else 'small'

    @staticmethod
    def compute_confidence(camera: str, object_size: str, distance: float) -> float:
        """
        计算相机测量的置信度 (0.0 - 1.0)

        Args:
            camera: 'chassis' or 'top'
            object_size: 'large' or 'small'
            distance: 物体距离（米）

        Returns:
            置信度 [0.0, 1.0], 1.0表示完全可信（误差<5cm）
        """
        if camera == 'chassis':
            if object_size == 'large':
                # 大物体（茶罐）：1.1m前完美，1.2m后快速退化
                if distance <= CameraConfidenceModel.CHASSIS_LARGE_ACCURATE_MAX:
                    return 1.0
                elif distance <= CameraConfidenceModel.CHASSIS_LARGE_DEGRADED_MIN:
                    # 线性过渡 1.0 -> 0.3 (10cm内)
                    t = (distance - CameraConfidenceModel.CHASSIS_LARGE_ACCURATE_MAX) / 0.10
                    return 1.0 - 0.7 * t
                else:
                    # 继续衰减，2m时接近0.1
                    decay = 0.3 - 0.2 * (distance - CameraConfidenceModel.CHASSIS_LARGE_DEGRADED_MIN) / 0.80
                    return max(0.1, decay)
            else:  # small
                # 小物体（红牛罐/饮料盒）：2.0m前完美
                if distance <= CameraConfidenceModel.CHASSIS_SMALL_ACCURATE_MAX:
                    return 1.0
                elif distance <= CameraConfidenceModel.CHASSIS_SMALL_DEGRADED_MIN:
                    # 线性过渡 1.0 -> 0.5 (50cm内)
                    t = (distance - CameraConfidenceModel.CHASSIS_SMALL_ACCURATE_MAX) / 0.50
                    return 1.0 - 0.5 * t
                else:
                    # 继续衰减
                    decay = 0.5 - 0.3 * (distance - CameraConfidenceModel.CHASSIS_SMALL_DEGRADED_MIN) / 1.0
                    return max(0.2, decay)

        elif camera == 'top':
            if object_size == 'large':
                # 大物体（茶罐）：全程完美（实验验证1.2m-3.0m）
                return 1.0
            else:  # small
                # 小物体（饮料盒/红牛罐）：近距离不可信，1.5m后完美
                if distance < CameraConfidenceModel.TOP_SMALL_DEGRADED_MAX:
                    # 非常不可信（实验显示10cm误差）
                    return 0.2
                elif distance <= CameraConfidenceModel.TOP_SMALL_ACCURATE_MIN:
                    # 线性恢复 0.2 -> 1.0 (22cm内)
                    t = (distance - CameraConfidenceModel.TOP_SMALL_DEGRADED_MAX) / 0.22
                    return 0.2 + 0.8 * t
                else:
                    return 1.0

        # 未知相机或参数异常，返回中等置信度
        return 0.5

    @staticmethod
    def fuse_positions(
        pos_chassis: np.ndarray,
        pos_top: np.ndarray,
        bbox_chassis: np.ndarray,
        bbox_top: np.ndarray,
        intrinsics_chassis: Dict,
        intrinsics_top: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        智能融合双相机的3D位置测量

        Args:
            pos_chassis: Chassis相机测量的3D位置 [x, y, z]
            pos_top: Top相机测量的3D位置 [x, y, z]
            bbox_chassis: Chassis bbox [x1, y1, x2, y2]
            bbox_top: Top bbox [x1, y1, x2, y2]
            intrinsics_chassis: Chassis相机内参
            intrinsics_top: Top相机内参

        Returns:
            fused_position: 融合后的3D位置
            meta: 融合元信息（用于调试）
        """
        # 计算距离
        dist_chassis = np.linalg.norm(pos_chassis)
        dist_top = np.linalg.norm(pos_top)

        # 估计物体尺寸（两个相机独立估计）
        size_chassis = CameraConfidenceModel.estimate_object_size(
            bbox_chassis, dist_chassis, intrinsics_chassis, 'chassis'
        )
        size_top = CameraConfidenceModel.estimate_object_size(
            bbox_top, dist_top, intrinsics_top, 'top'
        )

        # 如果尺寸判断不一致，保守取large（避免错误惩罚chassis大物体）
        object_size = 'large' if (size_chassis == 'large' or size_top == 'large') else 'small'

        # 计算置信度
        conf_chassis = CameraConfidenceModel.compute_confidence('chassis', object_size, dist_chassis)
        conf_top = CameraConfidenceModel.compute_confidence('top', object_size, dist_top)

        # 归一化权重
        total_weight = conf_chassis + conf_top

        if total_weight > 0.1:
            w_chassis = conf_chassis / total_weight
            w_top = conf_top / total_weight

            # 加权融合
            fused_position = w_chassis * pos_chassis + w_top * pos_top
        else:
            # 两个都不可信（理论上不应该发生），取距离更近的
            fused_position = pos_chassis if dist_chassis < dist_top else pos_top
            w_chassis = 1.0 if dist_chassis < dist_top else 0.0
            w_top = 1.0 - w_chassis

        # 融合元信息
        meta = {
            'object_size': object_size,
            'size_chassis': size_chassis,
            'size_top': size_top,
            'distance_chassis': dist_chassis,
            'distance_top': dist_top,
            'confidence_chassis': conf_chassis,
            'confidence_top': conf_top,
            'weight_chassis': w_chassis,
            'weight_top': w_top,
        }

        return fused_position, meta


# ============================================================================
# 测试和验证
# ============================================================================

def test_confidence_model():
    """测试置信度模型与实验数据的一致性"""
    print("=" * 70)
    print("相机置信度模型测试（基于实验数据验证）")
    print("=" * 70)

    model = CameraConfidenceModel

    # 定义测试用例（从实验数据中提取）
    test_cases = [
        # (camera, object_size, distance, expected_accuracy_cm, description)
        ('chassis', 'large', 0.80, 1, "茶罐@0.8m"),
        ('chassis', 'large', 1.10, 4, "茶罐@1.1m"),
        ('chassis', 'large', 1.20, 6, "茶罐@1.2m [退化开始]"),
        ('chassis', 'large', 2.03, 20, "茶罐@2.03m [严重退化]"),

        ('chassis', 'small', 1.48, 1, "红牛罐@1.48m"),
        ('chassis', 'small', 2.00, 3, "红牛罐@2.0m"),
        ('chassis', 'small', 2.50, 6, "红牛罐@2.5m [退化开始]"),

        ('top', 'large', 1.20, 1, "茶罐@1.2m"),
        ('top', 'large', 2.00, 0, "茶罐@2.0m"),
        ('top', 'large', 3.00, 5, "茶罐@3.0m"),

        ('top', 'small', 1.28, 10, "饮料盒@1.28m [近距离退化]"),
        ('top', 'small', 1.50, 1, "饮料盒@1.5m"),
        ('top', 'small', 2.00, 1, "饮料盒@2.0m"),
    ]

    print("\n【置信度计算结果】")
    print(f"{'场景':<25} {'距离(m)':<8} {'误差(cm)':<8} {'置信度':<8} {'判断'}")
    print("-" * 70)

    for camera, size, dist, error_cm, desc in test_cases:
        conf = model.compute_confidence(camera, size, dist)

        # 根据误差判断期望的置信度范围
        if error_cm <= 5:
            expected = "高"
            status = "✅" if conf >= 0.7 else "❌"
        elif error_cm <= 10:
            expected = "中"
            status = "✅" if 0.3 <= conf < 0.7 else "❌"
        else:
            expected = "低"
            status = "✅" if conf < 0.4 else "❌"

        print(f"{desc:<25} {dist:<8.2f} {error_cm:<8} {conf:<8.2f} {expected} {status}")

    print("\n" + "=" * 70)
    print("【融合场景测试】")
    print("=" * 70)

    # 测试融合场景
    fusion_cases = [
        {
            'desc': '茶罐@1.2m (Chassis:6cm, Top:1cm)',
            'pos_chassis': np.array([0.0, 0.0, 1.14]),  # 距离1.14m（6cm误差）
            'pos_top': np.array([0.0, 0.0, 1.19]),      # 距离1.19m（1cm误差）
            'bbox_chassis': np.array([200, 150, 350, 400]),  # 大物体bbox
            'bbox_top': np.array([180, 170, 320, 380]),
            'expected_weight': 'Top主导',
        },
        {
            'desc': '红牛罐@1.5m (Chassis:1cm, Top:3cm)',
            'pos_chassis': np.array([0.0, 0.0, 1.49]),
            'pos_top': np.array([0.0, 0.0, 1.53]),
            'bbox_chassis': np.array([250, 200, 320, 320]),  # 小物体bbox
            'bbox_top': np.array([240, 210, 310, 330]),
            'expected_weight': 'Chassis为主',
        },
        {
            'desc': '饮料盒@1.3m (Chassis:2cm, Top:10cm)',
            'pos_chassis': np.array([0.0, 0.0, 1.28]),
            'pos_top': np.array([0.0, 0.0, 1.38]),
            'bbox_chassis': np.array([260, 210, 330, 350]),  # 小物体
            'bbox_top': np.array([250, 200, 320, 340]),
            'expected_weight': 'Chassis主导',
        },
    ]

    intrinsics = {'fx': 386, 'fy': 386, 'cx': 320, 'cy': 240}

    for case in fusion_cases:
        fused, meta = model.fuse_positions(
            case['pos_chassis'], case['pos_top'],
            case['bbox_chassis'], case['bbox_top'],
            intrinsics, intrinsics
        )

        print(f"\n{case['desc']}")
        print(f"  物体尺寸: {meta['object_size']}")
        print(f"  Chassis置信度: {meta['confidence_chassis']:.3f} (权重: {meta['weight_chassis']:.3f})")
        print(f"  Top置信度: {meta['confidence_top']:.3f} (权重: {meta['weight_top']:.3f})")
        print(f"  期望: {case['expected_weight']}")

        # 判断是否符合期望
        if 'Top主导' in case['expected_weight']:
            status = "✅" if meta['weight_top'] > 0.7 else "❌"
        elif 'Chassis主导' in case['expected_weight']:
            status = "✅" if meta['weight_chassis'] > 0.7 else "❌"
        else:
            status = "✅" if 0.3 < meta['weight_chassis'] < 0.7 else "❌"

        print(f"  结果: {status}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_confidence_model()
