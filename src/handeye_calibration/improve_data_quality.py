#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
improve_data_quality.py - 标定数据质量管理模块

提供:
- 数据质量过滤器 (运动不足、工作空间边界、极端姿态等)
- RANSAC异常值过滤
- 重投影误差过滤

Design: Linus "Good Taste" 原则
- 每个过滤器是独立函数,可单独使用或组合
- 统一的输入/输出格式
- 失败时返回原数据,不破坏流程
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from calibration_common import normalize_angle_deg, angle_difference_deg


# ============================================================================
# 数据质量过滤器
# ============================================================================

class DataQualityFilter:
    """数据质量过滤器集合

    每个过滤器返回 (filtered_data, removed_frames)
    """

    @staticmethod
    def filter_insufficient_motion(collected_data, min_motion_mm=5.0,
                                   min_rotation_deg=2.0, require_both=True):
        """过滤运动不足的帧

        Args:
            collected_data: 采集数据列表
            min_motion_mm: 最小位移(mm)
            min_rotation_deg: 最小旋转(度)
            require_both: 是否要求位移和旋转都满足

        Returns:
            tuple: (filtered_data, removed_frames)
        """
        if len(collected_data) < 2:
            return collected_data, []

        filtered = []
        removed = []

        for i, data in enumerate(collected_data):
            if i == 0:
                filtered.append(data)
                continue

            # 计算与前一帧的运动
            prev_pose = np.array(collected_data[i-1]['pose'])
            curr_pose = np.array(data['pose'])

            # 位移 (mm)
            motion_mm = np.linalg.norm((curr_pose[:3] - prev_pose[:3]) * 1000)

            # 旋转 (度) - 注意：pose中的角度是弧度，需要先转换为度数
            roll_diff = angle_difference_deg(np.degrees(curr_pose[3]), np.degrees(prev_pose[3]))
            pitch_diff = angle_difference_deg(np.degrees(curr_pose[4]), np.degrees(prev_pose[4]))
            yaw_diff = angle_difference_deg(np.degrees(curr_pose[5]), np.degrees(prev_pose[5]))
            rotation_deg = np.sqrt(roll_diff**2 + pitch_diff**2 + yaw_diff**2)

            # 判断条件
            motion_ok = motion_mm >= min_motion_mm
            rotation_ok = rotation_deg >= min_rotation_deg

            if require_both:
                passed = motion_ok and rotation_ok
            else:
                passed = motion_ok or rotation_ok

            if passed:
                filtered.append(data)
            else:
                removed.append({
                    'frame_id': data['frame_id'],
                    'motion_mm': motion_mm,
                    'rotation_deg': rotation_deg,
                    'reason': '运动不足'
                })

        return filtered, removed

    @staticmethod
    def filter_workspace_boundary(collected_data, boundary_margin_ratio=0.10,
                                  min_boundary_axes=1):
        """过滤工作空间边界的帧

        Args:
            collected_data: 采集数据列表
            boundary_margin_ratio: 边界裕度比例(0-1)
            min_boundary_axes: 至少几个轴在边界才过滤

        Returns:
            tuple: (filtered_data, removed_frames)
        """
        if not collected_data:
            return collected_data, []

        # 计算工作空间范围
        positions = np.array([data['pose'][:3] for data in collected_data])
        min_vals = positions.min(axis=0)
        max_vals = positions.max(axis=0)
        ranges = max_vals - min_vals

        filtered = []
        removed = []

        for data in collected_data:
            pos = np.array(data['pose'][:3])

            # 计算每个轴的边界距离
            boundary_count = 0
            for axis in range(3):
                margin = ranges[axis] * boundary_margin_ratio
                if (pos[axis] - min_vals[axis]) < margin or \
                   (max_vals[axis] - pos[axis]) < margin:
                    boundary_count += 1

            if boundary_count >= min_boundary_axes:
                removed.append({
                    'frame_id': data['frame_id'],
                    'position': pos * 1000,  # mm
                    'reason': f'{boundary_count}个轴在边界{boundary_margin_ratio*100:.0f}%内'
                })
            else:
                filtered.append(data)

        return filtered, removed

    @staticmethod
    def filter_extreme_poses(collected_data, max_pitch_deg=70.0, max_roll_deg=None,
                            max_yaw_deg=None, verbose=True):
        """过滤极端姿态角的帧（相对于初始位姿的变化量）

        Args:
            collected_data: 采集数据列表
            max_pitch_deg: 最大允许pitch变化量(度)，None表示不限制
            max_roll_deg: 最大允许roll变化量(度)，None表示不限制
            max_yaw_deg: 最大允许yaw变化量(度)，None表示不限制
            verbose: 是否显示详细信息

        Returns:
            tuple: (filtered_data, removed_frames)
        """
        if not collected_data:
            return collected_data, []

        # 参考位姿（第一帧）
        ref_pose = collected_data[0]['pose']
        ref_roll_deg = np.degrees(ref_pose[3])
        ref_pitch_deg = np.degrees(ref_pose[4])
        ref_yaw_deg = np.degrees(ref_pose[5])

        if verbose:
            print(f"    参考位姿 (帧 {collected_data[0]['frame_id']}): Roll={ref_roll_deg:.1f}°, Pitch={ref_pitch_deg:.1f}°, Yaw={ref_yaw_deg:.1f}°")
            print(f"    阈值: ΔRoll≤{max_roll_deg}°, ΔPitch≤{max_pitch_deg}°, ΔYaw≤{max_yaw_deg}°")

        filtered = []
        removed = []
        max_diffs = {'roll': 0, 'pitch': 0, 'yaw': 0}

        for data in collected_data:
            roll, pitch, yaw = data['pose'][3:6]
            roll_deg = np.degrees(roll)
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)

            # 计算相对于初始位姿的变化量
            from calibration_common import angle_difference_deg
            pitch_diff = abs(angle_difference_deg(pitch_deg, ref_pitch_deg))
            roll_diff = abs(angle_difference_deg(roll_deg, ref_roll_deg))
            yaw_diff = abs(angle_difference_deg(yaw_deg, ref_yaw_deg))

            # 跟踪最大变化量
            max_diffs['roll'] = max(max_diffs['roll'], roll_diff)
            max_diffs['pitch'] = max(max_diffs['pitch'], pitch_diff)
            max_diffs['yaw'] = max(max_diffs['yaw'], yaw_diff)

            reasons = []
            if max_pitch_deg is not None and pitch_diff > max_pitch_deg:
                reasons.append(f'ΔPitch={pitch_diff:.1f}°(>{max_pitch_deg}°)')
            if max_roll_deg is not None and roll_diff > max_roll_deg:
                reasons.append(f'ΔRoll={roll_diff:.1f}°(>{max_roll_deg}°)')
            if max_yaw_deg is not None and yaw_diff > max_yaw_deg:
                reasons.append(f'ΔYaw={yaw_diff:.1f}°(>{max_yaw_deg}°)')

            if reasons:
                removed.append({
                    'frame_id': data['frame_id'],
                    'roll_deg': roll_deg,
                    'pitch_deg': pitch_deg,
                    'yaw_deg': yaw_deg,
                    'roll_diff': roll_diff,
                    'pitch_diff': pitch_diff,
                    'yaw_diff': yaw_diff,
                    'reason': ' & '.join(reasons)
                })
            else:
                filtered.append(data)

        if verbose:
            print(f"    实际最大变化: ΔRoll={max_diffs['roll']:.1f}°, ΔPitch={max_diffs['pitch']:.1f}°, ΔYaw={max_diffs['yaw']:.1f}°")

        return filtered, removed

    @staticmethod
    def filter_reprojection_error(R_gripper2base, t_gripper2base, R_target2cam,
                                   t_target2cam, frame_ids, reprojection_errors,
                                   max_error_px=2.0):
        """过滤重投影误差过大的帧

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            frame_ids: 帧ID列表
            reprojection_errors: 重投影误差列表
            max_error_px: 最大允许重投影误差(像素)

        Returns:
            tuple: (filtered R_g, filtered t_g, filtered R_t, filtered t_t,
                    filtered frame_ids, filtered errors, removed_frames)
        """
        keep_indices = [i for i, err in enumerate(reprojection_errors) if err <= max_error_px]
        removed_indices = [i for i, err in enumerate(reprojection_errors) if err > max_error_px]

        removed_frames = [
            {
                'frame_id': frame_ids[idx],
                'reprojection_error_px': reprojection_errors[idx],
                'reason': f'Reprojection error {reprojection_errors[idx]:.3f}px > {max_error_px}px'
            }
            for idx in removed_indices
        ]

        # Filter all arrays
        R_g_filtered = [R_gripper2base[i] for i in keep_indices]
        t_g_filtered = [t_gripper2base[i] for i in keep_indices]
        R_t_filtered = [R_target2cam[i] for i in keep_indices]
        t_t_filtered = [t_target2cam[i] for i in keep_indices]
        frame_ids_filtered = [frame_ids[i] for i in keep_indices]
        errors_filtered = [reprojection_errors[i] for i in keep_indices]

        return (R_g_filtered, t_g_filtered, R_t_filtered, t_t_filtered,
                frame_ids_filtered, errors_filtered, removed_frames)

    @staticmethod
    def apply_all_filters(collected_data, config=None, verbose=True):
        """应用所有质量过滤器

        Args:
            collected_data: 原始采集数据
            config: 配置字典（quality_filter 部分），None 则使用默认值
            verbose: 是否显示详细信息

        Returns:
            tuple: (filtered_data, filter_report)
        """
        # 默认配置
        if config is None:
            config = {}

        min_motion_mm = config.get('min_motion_mm', 5.0)
        min_rotation_deg = config.get('min_rotation_deg', 2.0)
        boundary_margin_ratio = config.get('boundary_margin_ratio', 0.10)
        max_pitch_deg = config.get('max_pitch_deg', 70.0)
        max_roll_deg = config.get('max_roll_deg', None)
        max_yaw_deg = config.get('max_yaw_deg', None)

        if verbose:
            print("\n" + "="*60)
            print("🔍 数据质量过滤")
            print("="*60)
            print(f"原始数据: {len(collected_data)} 帧")

        original_count = len(collected_data)
        all_removed = []

        # 第1层：过滤运动不足的帧
        if verbose:
            print("\n步骤1: 过滤运动不足的帧")

        filtered_data, removed = DataQualityFilter.filter_insufficient_motion(
            collected_data,
            min_motion_mm=min_motion_mm,
            min_rotation_deg=min_rotation_deg,
            require_both=True
        )

        if verbose and removed:
            print(f"  移除 {len(removed)} 帧 (位移<{min_motion_mm}mm 或 旋转<{min_rotation_deg}°)")
        all_removed.extend(removed)

        # 第2层：过滤工作空间边界的帧（如果启用）
        if boundary_margin_ratio is not None and boundary_margin_ratio > 0:
            if verbose:
                print("\n步骤2: 过滤工作空间边界帧")
            filtered_data, removed = DataQualityFilter.filter_workspace_boundary(
                filtered_data,
                boundary_margin_ratio=boundary_margin_ratio,
                min_boundary_axes=1
            )

            if verbose and removed:
                print(f"  移除 {len(removed)} 帧 (在边界{boundary_margin_ratio*100:.0f}%内)")
            all_removed.extend(removed)
        else:
            if verbose:
                print("\n步骤2: 跳过工作空间边界过滤 (已禁用)")

        # 第3层：过滤极端姿态角的帧
        if verbose:
            print("\n步骤3: 过滤极端姿态角帧")

        filtered_data, removed = DataQualityFilter.filter_extreme_poses(
            filtered_data,
            max_pitch_deg=max_pitch_deg,
            max_roll_deg=max_roll_deg,
            max_yaw_deg=max_yaw_deg,
            verbose=True  # 开启详细输出
        )

        if verbose:
            if removed:
                limits = []
                if max_pitch_deg is not None:
                    limits.append(f"ΔPitch>{max_pitch_deg}°")
                if max_roll_deg is not None:
                    limits.append(f"ΔRoll>{max_roll_deg}°")
                if max_yaw_deg is not None:
                    limits.append(f"ΔYaw>{max_yaw_deg}°")
                print(f"  移除 {len(removed)} 帧 ({' 或 '.join(limits) if limits else '无限制'})")
                # 显示被移除的帧详情
                for r in removed[:5]:  # 只显示前5个
                    print(f"    ❌ 帧 {r['frame_id']}: {r['reason']}")
                if len(removed) > 5:
                    print(f"    ... 以及其他 {len(removed)-5} 帧")
            else:
                print(f"  无帧被移除 (所有姿态变化在阈值内)")
        all_removed.extend(removed)

        # 统计
        final_count = len(filtered_data)
        removed_count = original_count - final_count

        if verbose:
            print("\n" + "-"*60)
            print("📊 过滤统计:")
            print(f"  原始: {original_count} 帧")
            print(f"  保留: {final_count} 帧 ({final_count/original_count*100:.1f}%)")
            print(f"  移除: {removed_count} 帧 ({removed_count/original_count*100:.1f}%)")

        filter_report = {
            'original_count': original_count,
            'final_count': final_count,
            'removed_count': removed_count,
            'removed_frames': all_removed,
            'removal_rate': removed_count / original_count if original_count > 0 else 0
        }

        return filtered_data, filter_report


# ============================================================================
# RANSAC 异常值过滤
# ============================================================================

class RANSACFilter:
    """RANSAC异常值过滤器

    使用AX=XB一致性误差进行鲁棒的异常值检测
    """

    @staticmethod
    def ransac_filter_handeye(R_gripper2base, t_gripper2base, R_target2cam,
                              t_target2cam, frame_ids, threshold=6.0,
                              iterations=None, min_inlier_ratio=0.3):
        """使用RANSAC过滤手眼标定异常值

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            frame_ids: 帧ID列表
            threshold: 内点阈值(mm)
            iterations: RANSAC迭代次数
            min_inlier_ratio: 最小内点比例

        Returns:
            list: 内点索引列表
        """
        n_samples = len(R_gripper2base)

        if n_samples < 5:
            return list(range(n_samples))

        if iterations is None:
            iterations = min(200, n_samples * 20)

        best_inliers = []
        best_median_error = float('inf')

        for iteration in range(iterations):
            # 随机采样8个点
            sample_size = min(8, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)

            try:
                # 使用样本计算标定
                R_sample, t_sample = cv2.calibrateHandEye(
                    [R_gripper2base[i] for i in sample_indices],
                    [t_gripper2base[i] for i in sample_indices],
                    [R_target2cam[i] for i in sample_indices],
                    [t_target2cam[i] for i in sample_indices],
                    method=cv2.CALIB_HAND_EYE_TSAI
                )

                # 计算所有点的AX=XB误差
                errors = []
                for i in range(n_samples):
                    # AX
                    AX_R = R_gripper2base[i] @ R_sample
                    AX_t = R_gripper2base[i] @ t_sample + t_gripper2base[i]

                    # XB
                    XB_R = R_sample @ R_target2cam[i]
                    XB_t = R_sample @ t_target2cam[i] + t_sample

                    # 旋转误差 (转换为mm当量)
                    R_diff = AX_R @ XB_R.T
                    angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

                    # 平移误差 (mm)
                    t_error = np.linalg.norm(AX_t - XB_t) * 1000

                    # 综合误差
                    total_error = t_error + np.degrees(angle_error) * 1.0
                    errors.append(total_error)

                errors_array = np.array(errors)

                # 动态阈值
                median_error = np.median(errors_array)
                mad = np.median(np.abs(errors_array - median_error))
                dynamic_threshold = min(threshold, median_error + 2.5 * mad)

                # 找出内点
                inliers = [i for i, e in enumerate(errors) if e < dynamic_threshold]

                min_inliers = max(4, int(n_samples * min_inlier_ratio))

                if len(inliers) >= min_inliers:
                    # 使用内点精化
                    R_refined, t_refined = cv2.calibrateHandEye(
                        [R_gripper2base[i] for i in inliers],
                        [t_gripper2base[i] for i in inliers],
                        [R_target2cam[i] for i in inliers],
                        [t_target2cam[i] for i in inliers],
                        method=cv2.CALIB_HAND_EYE_TSAI
                    )

                    # 计算精化后的误差
                    refined_errors = []
                    for i in inliers:
                        AX_t = R_gripper2base[i] @ t_refined + t_gripper2base[i]
                        XB_t = R_refined @ t_target2cam[i] + t_refined
                        t_error = np.linalg.norm(AX_t - XB_t) * 1000
                        refined_errors.append(t_error)

                    median_error = np.median(refined_errors)

                    # 选择最优结果
                    if len(inliers) > len(best_inliers) or \
                       (len(inliers) == len(best_inliers) and median_error < best_median_error):
                        best_inliers = inliers
                        best_median_error = median_error

            except:
                continue

        # 如果RANSAC失败,使用保守的误差过滤
        if len(best_inliers) < max(4, int(n_samples * min_inlier_ratio)):
            print(f"   ⚠️ RANSAC内点不足，使用保守的误差阈值过滤")

            try:
                # 使用所有数据计算
                R_all, t_all = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )

                errors = []
                for i in range(n_samples):
                    AX_t = R_gripper2base[i] @ t_all + t_gripper2base[i]
                    XB_t = R_all @ t_target2cam[i] + t_all
                    t_error = np.linalg.norm(AX_t - XB_t) * 1000
                    errors.append(t_error)

                # 使用中位数 + 3倍MAD
                median = np.median(errors)
                mad = np.median(np.abs(np.array(errors) - median))
                conservative_threshold = median + 3.0 * mad

                # 绝对上限15mm
                final_threshold = min(conservative_threshold, 15.0)

                best_inliers = [i for i, e in enumerate(errors) if e < final_threshold]

                # 如果过滤太多,保留所有
                if len(best_inliers) < int(n_samples * 0.8):
                    print(f"   ℹ️  保守过滤仍会移除过多数据，保留所有帧")
                    best_inliers = list(range(n_samples))

            except:
                best_inliers = list(range(n_samples))

        return best_inliers

    @staticmethod
    def filter_reprojection_errors(collected_data, camera_matrix, dist_coeffs,
                                   board_size, chessboard_size_mm,
                                   max_error_px=2.0):
        """根据重投影误差过滤帧

        Args:
            collected_data: 采集数据列表
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            board_size: 棋盘格尺寸 (cols, rows)
            chessboard_size_mm: 方格大小(mm)
            max_error_px: 最大重投影误差(像素)

        Returns:
            tuple: (filtered_data, removed_frames)
        """
        # 准备棋盘格世界坐标
        square_size = chessboard_size_mm / 1000.0
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size

        filtered = []
        removed = []

        for data in collected_data:
            ret, rvec, tvec = cv2.solvePnP(objp, data['corners'],
                                           camera_matrix, dist_coeffs)

            if ret:
                # 计算重投影误差
                projected, _ = cv2.projectPoints(objp, rvec, tvec,
                                                camera_matrix, dist_coeffs)
                projected = projected.reshape(-1, 2)
                detected = data['corners'].reshape(-1, 2)
                error = np.sqrt(np.mean(np.sum((projected - detected)**2, axis=1)))

                if error < max_error_px:
                    filtered.append(data)
                else:
                    removed.append({
                        'frame_id': data['frame_id'],
                        'reprojection_error': error,
                        'reason': f'重投影误差{error:.2f}px>阈值{max_error_px}px'
                    })
            else:
                removed.append({
                    'frame_id': data['frame_id'],
                    'reason': 'solvePnP失败'
                })

        return filtered, removed
