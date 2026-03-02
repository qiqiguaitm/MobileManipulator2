# -*- coding: utf-8 -*-
"""
Quality evaluation utilities for camera-LiDAR calibration.
"""

import numpy as np


def evaluate_quality(reprojection_rms, tilt_angle, inlier_ratio,
                     distance_diff, normal_diff, num_lidar_points):
    """Evaluate calibration frame quality.

    Args:
        reprojection_rms: reprojection RMS error (pixels)
        tilt_angle: board tilt angle (degrees)
        inlier_ratio: LiDAR plane fitting inlier ratio
        distance_diff: camera-LiDAR distance difference (meters)
        normal_diff: normal alignment (cosine, closer to 1 is better)
        num_lidar_points: number of LiDAR points

    Returns:
        grade: overall grade (A+/A/B/C/D)
        details: per-metric grades dict
        score: average score
    """
    details = {}
    scores = []

    # 1. Reprojection error
    if reprojection_rms < 1.0:
        details['reprojection'] = ('A+', '优秀')
        scores.append(100)
    elif reprojection_rms < 1.5:
        details['reprojection'] = ('A', '良好')
        scores.append(90)
    elif reprojection_rms < 2.5:
        details['reprojection'] = ('B', '中等')
        scores.append(75)
    elif reprojection_rms < 4.0:
        details['reprojection'] = ('C', '较差')
        scores.append(60)
    else:
        details['reprojection'] = ('D', '差')
        scores.append(40)

    # 2. Tilt angle
    if tilt_angle < 15:
        details['tilt'] = ('A+', '优秀')
        scores.append(100)
    elif tilt_angle < 25:
        details['tilt'] = ('A', '良好')
        scores.append(90)
    elif tilt_angle < 35:
        details['tilt'] = ('B', '中等')
        scores.append(75)
    elif tilt_angle < 50:
        details['tilt'] = ('C', '较差')
        scores.append(60)
    else:
        details['tilt'] = ('D', '差')
        scores.append(40)

    # 3. LiDAR fitting quality
    if inlier_ratio > 0.95:
        details['lidar_fit'] = ('A+', '优秀')
        scores.append(100)
    elif inlier_ratio > 0.90:
        details['lidar_fit'] = ('A', '良好')
        scores.append(90)
    elif inlier_ratio > 0.80:
        details['lidar_fit'] = ('B', '中等')
        scores.append(75)
    elif inlier_ratio > 0.65:
        details['lidar_fit'] = ('C', '较差')
        scores.append(60)
    else:
        details['lidar_fit'] = ('D', '差')
        scores.append(40)

    # 4. Distance consistency
    if distance_diff < 0.05:
        details['distance'] = ('A+', '优秀')
        scores.append(100)
    elif distance_diff < 0.10:
        details['distance'] = ('A', '良好')
        scores.append(90)
    elif distance_diff < 0.20:
        details['distance'] = ('B', '中等')
        scores.append(75)
    elif distance_diff < 0.40:
        details['distance'] = ('C', '较差')
        scores.append(60)
    else:
        details['distance'] = ('D', '差')
        scores.append(40)

    # 5. Normal alignment
    angle_deg = np.degrees(np.arccos(np.clip(normal_diff, -1, 1)))
    if angle_deg < 5:
        details['normal'] = ('A+', '优秀')
        scores.append(100)
    elif angle_deg < 10:
        details['normal'] = ('A', '良好')
        scores.append(90)
    elif angle_deg < 20:
        details['normal'] = ('B', '中等')
        scores.append(75)
    elif angle_deg < 30:
        details['normal'] = ('C', '较差')
        scores.append(60)
    else:
        details['normal'] = ('D', '差')
        scores.append(40)

    # 6. Point count
    if num_lidar_points >= 100:
        details['points'] = ('A+', '优秀')
        scores.append(100)
    elif num_lidar_points >= 80:
        details['points'] = ('A', '良好')
        scores.append(90)
    elif num_lidar_points >= 60:
        details['points'] = ('B', '中等')
        scores.append(75)
    elif num_lidar_points >= 40:
        details['points'] = ('C', '较差')
        scores.append(60)
    else:
        details['points'] = ('D', '差')
        scores.append(40)

    # Overall score
    avg_score = np.mean(scores)

    if avg_score >= 95:
        grade = 'A+'
    elif avg_score >= 85:
        grade = 'A'
    elif avg_score >= 70:
        grade = 'B'
    elif avg_score >= 55:
        grade = 'C'
    else:
        grade = 'D'

    return grade, details, avg_score


def print_quality_summary_table(frame_qualities):
    """Print quality comparison table for all frames."""
    if not frame_qualities:
        return

    print(f"\n{'='*125}")
    print("质量评估汇总表")
    print(f"{'='*125}")

    print(f"{'Frame':<8}{'重投影RMS':>12}  {'对齐角':>10}  {'LiDAR拟合率':>12}  {'3D位置差':>11}  {'法向量夹角':>12}  {'LiDAR':>8}  {'总评':<6}  {'得分':>6}")
    print(f"{'ID':<8}{'(像素)':>12}  {'(度)':>10}  {'(%)':>12}  {'(米)':>11}  {'(度)':>12}  {'点数':>8}  {'等级':<6}  {'(分)':>6}")
    print(f"{'-'*125}")

    for fq in frame_qualities:
        print(f"{fq['frame_id']:<8}"
              f"{fq['reproj_rms']:>12.2f}  "
              f"{fq['tilt_angle']:>10.1f}  "
              f"{fq['inlier_ratio']*100:>12.1f}  "
              f"{fq['distance_diff']:>11.3f}  "
              f"{fq['normal_angle']:>12.1f}  "
              f"{fq['num_points']:>8}  "
              f"{fq['grade']:<6}  "
              f"{fq['score']:>6.1f}")

    print(f"{'='*125}")

    avg_score = sum(fq['score'] for fq in frame_qualities) / len(frame_qualities)
    grades = [fq['grade'] for fq in frame_qualities]
    grade_counts = {g: grades.count(g) for g in ['A+', 'A', 'B', 'C', 'D']}

    print(f"总帧数: {len(frame_qualities)} | 平均得分: {avg_score:.1f} | ", end='')
    print(f"等级分布: ", end='')
    for g in ['A+', 'A', 'B', 'C', 'D']:
        if grade_counts[g] > 0:
            print(f"{g}:{grade_counts[g]} ", end='')
    print()
    print(f"{'='*125}\n")


def print_frame_quality(frame_id, reprojection_rms, reprojection_mean, reprojection_max,
                       tilt_angle, inlier_ratio, num_points, num_inliers,
                       distance_diff, cam_center_in_lidar, lid_center, normal_diff,
                       grade, details, score):
    """Print detailed frame quality report."""
    print(f"\n  {'='*60}")
    print(f"  Frame {frame_id} 质量报告 - 总评: {grade} ({score:.1f}/100)")
    print(f"  {'='*60}")

    def get_color(level):
        colors = {'A+': '\033[92m', 'A': '\033[92m', 'B': '\033[93m',
                  'C': '\033[91m', 'D': '\033[91m'}
        return colors.get(level, '')

    reset = '\033[0m'

    print(f"  1. 角点重投影误差:")
    grade_repr, desc = details['reprojection']
    color = get_color(grade_repr)
    print(f"     RMS={reprojection_rms:.2f}px, Mean={reprojection_mean:.2f}px, Max={reprojection_max:.2f}px")
    print(f"     等级: {color}{grade_repr}{reset} ({desc})")

    print(f"  2. 标定板对齐角度:")
    grade_tilt, desc = details['tilt']
    color = get_color(grade_tilt)
    print(f"     对齐角度: {tilt_angle:.1f}°")
    print(f"     等级: {color}{grade_tilt}{reset} ({desc})")

    print(f"  3. LiDAR平面拟合:")
    grade_fit, desc = details['lidar_fit']
    color = get_color(grade_fit)
    print(f"     内点比例: {inlier_ratio:.1%} ({num_inliers}/{num_points})")
    print(f"     等级: {color}{grade_fit}{reset} ({desc})")

    print(f"  4. 距离一致性 (LiDAR坐标系下3D位置差异):")
    grade_dist, desc = details['distance']
    color = get_color(grade_dist)
    print(f"     相机投影位置: [{cam_center_in_lidar[0]:.3f}, {cam_center_in_lidar[1]:.3f}, {cam_center_in_lidar[2]:.3f}]m")
    print(f"     LiDAR拟合位置: [{lid_center[0]:.3f}, {lid_center[1]:.3f}, {lid_center[2]:.3f}]m")
    print(f"     3D位置差异: {distance_diff:.3f}m")
    print(f"     等级: {color}{grade_dist}{reset} ({desc})")

    print(f"  5. 法向量对齐:")
    grade_norm, desc = details['normal']
    color = get_color(grade_norm)
    angle_deg = np.degrees(np.arccos(np.clip(normal_diff, -1, 1)))
    print(f"     对齐误差: {angle_deg:.2f}°")
    print(f"     等级: {color}{grade_norm}{reset} ({desc})")

    print(f"  6. 点云密度:")
    grade_pts, desc = details['points']
    color = get_color(grade_pts)
    print(f"     LiDAR点数: {num_points}")
    print(f"     等级: {color}{grade_pts}{reset} ({desc})")

    print(f"\n  总评: {get_color(grade)}{grade}{reset} - ", end='')
    if grade in ['A+', 'A']:
        print("质量优秀，可用于标定 ✓")
    elif grade == 'B':
        print("质量中等，建议检查是否可以改进")
    else:
        print("质量较差，建议跳过或重新采集 ✗")

    print(f"  {'='*60}")


def evaluate_calibration_result(center_errors, frame_count, residual=None):
    """Evaluate overall calibration result quality.

    Args:
        center_errors: list of center errors for each frame
        frame_count: number of frames used
        residual: optimization residual

    Returns:
        total_score: overall score
        grade: letter grade
        recommendation: usage recommendation
    """
    center_errors = np.array(center_errors)
    mean_error = np.mean(center_errors) * 100  # cm
    max_error = np.max(center_errors) * 100  # cm

    # Residual score
    if residual is not None:
        if residual < 0.5:
            residual_score = 100
        elif residual < 1.0:
            residual_score = 90
        elif residual < 2.0:
            residual_score = 75
        elif residual < 3.0:
            residual_score = 60
        else:
            residual_score = 40
    else:
        residual_score = 70

    # Consistency score
    if max_error < 3.0:
        consistency_score = 100
    elif max_error < 5.0:
        consistency_score = 90
    elif max_error < 8.0:
        consistency_score = 75
    elif max_error < 12.0:
        consistency_score = 60
    else:
        consistency_score = 40

    # Frame count score
    if frame_count >= 6:
        frame_score = 100
    elif frame_count >= 5:
        frame_score = 90
    elif frame_count >= 4:
        frame_score = 75
    elif frame_count >= 3:
        frame_score = 60
    else:
        frame_score = 40

    total_score = residual_score * 0.5 + consistency_score * 0.3 + frame_score * 0.2

    if total_score >= 95:
        grade = "A+"
        recommendation = "标定质量极佳，可直接使用"
    elif total_score >= 85:
        grade = "A"
        recommendation = "标定质量良好，推荐使用"
    elif total_score >= 75:
        grade = "B"
        recommendation = "标定质量合格，建议验证后使用"
    elif total_score >= 60:
        grade = "C"
        recommendation = "标定质量一般，建议重新标定"
    else:
        grade = "D"
        recommendation = "标定质量差，需要重新标定"

    return total_score, grade, recommendation
