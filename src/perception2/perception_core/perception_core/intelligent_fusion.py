#!/usr/bin/env python3
"""
智能双相机融合模块

基于实验数据的保守鲁棒融合策略：
1. 只用可靠的信息（distance, diff, relative_std）
2. 基于实验数据的硬规律（距离分段）
3. 保守决策（不确定时选更安全的）

设计原则：
- 不依赖未验证的假设（如IQR估计）
- 阈值基于实际实验数据
- 逻辑简单清晰，易于调试
"""

from typing import Dict, Tuple
import numpy as np


def fuse_dual_camera_positions(
    pos_chassis: np.ndarray,
    pos_top: np.ndarray,
    stats_chassis: Dict,
    stats_top: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    双相机3D位置智能融合

    Args:
        pos_chassis: Chassis相机测量的3D位置 [x, y, z] in base_link
        pos_top: Top相机测量的3D位置 [x, y, z] in base_link
        stats_chassis: Chassis测量统计量 {depth_median, depth_std, num_points, ...}
        stats_top: Top测量统计量

    Returns:
        fused_position: 融合后的3D位置
        meta: 融合元信息（策略、权重、置信度等）

    融合策略：
    1. 差异检查：如果两个测量差异大，选择质量更好的
    2. 距离分段：基于实验数据的硬规律分配权重
    3. 质量微调：用relative_std微调权重
    """
    dist_chassis = np.linalg.norm(pos_chassis)
    dist_top = np.linalg.norm(pos_top)
    avg_dist = (dist_chassis + dist_top) / 2.0
    position_diff = np.linalg.norm(pos_chassis - pos_top)

    # ========== 第一层：差异检查（优先级最高） ==========
    # 如果测量差异大，至少一个有问题，需要选择而不是融合

    # 阈值设计（基于实验数据调整）
    RELATIVE_DIFF_THRESHOLD = 0.12  # 12%
    ABSOLUTE_DIFF_THRESHOLD = 0.10  # 10cm（降低以便更早触发选择逻辑）

    is_inconsistent = (
        position_diff > ABSOLUTE_DIFF_THRESHOLD or
        position_diff > RELATIVE_DIFF_THRESHOLD * avg_dist
    )

    if is_inconsistent:
        # 差异大，需要选择
        # 新策略：主要依靠质量（relative_std），辅以简单的距离规律

        # 1. 质量分（基于relative_std）- 最重要的指标
        rel_std_chassis = stats_chassis['depth_std'] / stats_chassis['depth_median']
        rel_std_top = stats_top['depth_std'] / stats_top['depth_median']

        # std越小越好，>10%视为差
        # 使用更陡峭的曲线，放大质量差异
        quality_chassis = max(0.0, 1.0 - rel_std_chassis / 0.08)  # 8%以上快速衰减
        quality_top = max(0.0, 1.0 - rel_std_top / 0.08)

        # 2. 区间因子（基于实验观察的简化规律）
        # 关键规律：
        # - Chassis: 小物体准，大物体不准（depth_std会大）
        # - Top: 大物体准（depth_std会小），远距离更稳定
        # - 远距离时（>1.5m），Top明显更可靠

        if avg_dist < 1.5:
            # 近中距离：主要看质量（因为两者都可能准或不准）
            range_factor_chassis = 1.0
            range_factor_top = 1.0
        else:
            # 远距离：明显偏向Top（Chassis对大物体退化明显）
            range_factor_chassis = 0.6
            range_factor_top = 1.4

        # 综合得分（移除距离因子，避免测量错误的循环依赖）
        score_chassis = quality_chassis * range_factor_chassis
        score_top = quality_top * range_factor_top

        # 判断是否有明显更好的（>1.3倍）
        SCORE_RATIO_THRESHOLD = 1.3

        # 特殊情况：如果两个得分都很低，标记低置信度
        if max(score_chassis, score_top) < 0.2:
            # 两个都很差，保守选择距离近的
            selected = pos_chassis if dist_chassis < dist_top else pos_top
            return selected, {
                'strategy': 'fallback_both_poor',
                'confidence_level': 'low',
                'reason': 'both_measurements_unreliable',
                'score_chassis': score_chassis,
                'score_top': score_top,
                'position_diff': position_diff,
                'quality_chassis': quality_chassis,
                'quality_top': quality_top
            }

        # 选择得分显著更高的
        if score_chassis > score_top * SCORE_RATIO_THRESHOLD:
            return pos_chassis, {
                'strategy': 'select_chassis',
                'confidence_level': 'medium',
                'reason': 'inconsistent_chassis_significantly_better',
                'score_chassis': score_chassis,
                'score_top': score_top,
                'position_diff': position_diff,
                'quality_chassis': quality_chassis,
                'quality_top': quality_top
            }
        elif score_top > score_chassis * SCORE_RATIO_THRESHOLD:
            return pos_top, {
                'strategy': 'select_top',
                'confidence_level': 'medium',
                'reason': 'inconsistent_top_significantly_better',
                'score_chassis': score_chassis,
                'score_top': score_top,
                'position_diff': position_diff,
                'quality_chassis': quality_chassis,
                'quality_top': quality_top
            }
        else:
            # 得分接近，选距离近的（保守策略）
            selected = pos_chassis if dist_chassis < dist_top else pos_top
            return selected, {
                'strategy': 'select_closer',
                'confidence_level': 'medium',
                'reason': 'inconsistent_scores_similar',
                'score_chassis': score_chassis,
                'score_top': score_top,
                'position_diff': position_diff,
                'selected': 'chassis' if dist_chassis < dist_top else 'top'
            }

    # ========== 第二层：差异小，加权融合 ==========
    # 测量一致，根据距离区间和质量分配权重

    # 1. 基础权重（基于距离区间）- 简化为两个区间
    if avg_dist < 1.5:
        # 近中距离：平衡起点，主要让质量决定
        w_chassis, w_top = 0.5, 0.5
        strategy_base = 'balanced_near_mid_range'
    else:
        # 远距离：偏向Top（Chassis对大物体退化）
        w_chassis, w_top = 0.35, 0.65
        strategy_base = 'favor_top_far_range'

    # 2. 质量调整（根据relative_std）- 更激进的调整
    rel_std_chassis = stats_chassis['depth_std'] / stats_chassis['depth_median']
    rel_std_top = stats_top['depth_std'] / stats_top['depth_median']

    # 使用与选择逻辑一致的质量计算
    quality_chassis = max(0.0, 1.0 - rel_std_chassis / 0.08)
    quality_top = max(0.0, 1.0 - rel_std_top / 0.08)

    # 如果质量差距明显，调整权重（更大的调整幅度）
    quality_diff = abs(quality_chassis - quality_top)
    if quality_diff > 0.15:  # 降低阈值，更容易触发
        if quality_chassis > quality_top:
            # Chassis质量更好，增加其权重（最多25%）
            adjustment = min(0.25, quality_diff * 0.5)
            w_chassis = min(0.80, w_chassis + adjustment)
            w_top = 1.0 - w_chassis
            quality_adjusted = 'favor_chassis'
        else:
            # Top质量更好，增加其权重
            adjustment = min(0.25, quality_diff * 0.5)
            w_top = min(0.80, w_top + adjustment)
            w_chassis = 1.0 - w_top
            quality_adjusted = 'favor_top'
    else:
        quality_adjusted = 'none'

    # 3. 加权融合
    fused_position = w_chassis * pos_chassis + w_top * pos_top

    # 4. 评估融合置信度
    # 如果两个测量都质量好且一致，置信度高
    if quality_chassis > 0.6 and quality_top > 0.6 and position_diff < 0.10:
        confidence_level = 'high'
    elif max(quality_chassis, quality_top) > 0.5:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'

    return fused_position, {
        'strategy': f'weighted_fusion_{strategy_base}',
        'confidence_level': confidence_level,
        'weight_chassis': w_chassis,
        'weight_top': w_top,
        'quality_chassis': quality_chassis,
        'quality_top': quality_top,
        'quality_adjusted': quality_adjusted,
        'position_diff': position_diff,
        'avg_distance': avg_dist,
        'relative_std_chassis': rel_std_chassis,
        'relative_std_top': rel_std_top
    }


# ============================================================================
# 测试代码
# ============================================================================

def test_fusion():
    """测试融合函数"""
    print("=" * 70)
    print("双相机智能融合测试")
    print("=" * 70)

    # 测试场景
    test_cases = [
        {
            'name': '茶罐@1.2m (Chassis不准6cm, Top准1cm)',
            'pos_chassis': np.array([0.0, 0.0, 1.14]),
            'pos_top': np.array([0.0, 0.0, 1.19]),
            'stats_chassis': {'depth_median': 1.14, 'depth_std': 0.05, 'num_points': 500},
            'stats_top': {'depth_median': 1.19, 'depth_std': 0.02, 'num_points': 600},
            'expected': 'Top主导或选择Top'
        },
        {
            'name': '红牛罐@1.5m (两个都准)',
            'pos_chassis': np.array([0.0, 0.0, 1.49]),
            'pos_top': np.array([0.0, 0.0, 1.53]),
            'stats_chassis': {'depth_median': 1.49, 'depth_std': 0.03, 'num_points': 300},
            'stats_top': {'depth_median': 1.53, 'depth_std': 0.03, 'num_points': 400},
            'expected': '均衡融合'
        },
        {
            'name': '牛奶盒@1.3m (Chassis准2cm, Top不准10cm)',
            'pos_chassis': np.array([0.0, 0.0, 1.28]),
            'pos_top': np.array([0.0, 0.0, 1.38]),
            'stats_chassis': {'depth_median': 1.28, 'depth_std': 0.025, 'num_points': 350},
            'stats_top': {'depth_median': 1.38, 'depth_std': 0.10, 'num_points': 200},
            'expected': 'Chassis主导'
        },
        {
            'name': '茶罐@2m (Chassis很差20cm, Top准0cm)',
            'pos_chassis': np.array([0.0, 0.0, 1.80]),
            'pos_top': np.array([0.0, 0.0, 2.00]),
            'stats_chassis': {'depth_median': 1.80, 'depth_std': 0.18, 'num_points': 200},
            'stats_top': {'depth_median': 2.00, 'depth_std': 0.02, 'num_points': 500},
            'expected': '选择Top'
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n【测试{i}】{case['name']}")
        print(f"期望结果: {case['expected']}")

        fused_pos, meta = fuse_dual_camera_positions(
            case['pos_chassis'],
            case['pos_top'],
            case['stats_chassis'],
            case['stats_top']
        )

        print(f"\n融合结果:")
        print(f"  策略: {meta['strategy']}")
        print(f"  置信度: {meta['confidence_level']}")

        if 'weight_chassis' in meta:
            print(f"  权重: Chassis={meta['weight_chassis']:.2f}, Top={meta['weight_top']:.2f}")

        if 'score_chassis' in meta:
            print(f"  得分: Chassis={meta['score_chassis']:.3f}, Top={meta['score_top']:.3f}")

        print(f"  质量: Chassis={meta['quality_chassis']:.2f}, Top={meta['quality_top']:.2f}")
        print(f"  测量差异: {meta['position_diff']*100:.1f}cm")
        print(f"  融合位置: {fused_pos}")

        # 简单验证
        if '主导' in case['expected'] or '选择' in case['expected']:
            if 'Top' in case['expected']:
                if 'weight_top' in meta:
                    status = "✅" if meta['weight_top'] > 0.6 else "⚠️"
                else:
                    status = "✅" if 'select_top' in meta['strategy'] else "⚠️"
            else:  # Chassis
                if 'weight_chassis' in meta:
                    status = "✅" if meta['weight_chassis'] > 0.6 else "⚠️"
                else:
                    status = "✅" if 'select_chassis' in meta['strategy'] else "⚠️"
        else:
            status = "✅"  # 均衡融合，不好自动判断

        print(f"  验证: {status}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == '__main__':
    test_fusion()
