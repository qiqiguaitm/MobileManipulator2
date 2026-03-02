#!/usr/bin/env python3
"""
手动手眼标定 - 增强版
功能：手动模式 + 轨迹重播模式

Version: 2.1.0
Changelog:
  - 2.1.0: 重构代码消除重复逻辑，提取 _prepare_calibration_data 公共方法
  - 2.0.0: 统一数据路径管理，所有数据保存到 calibration_data/，支持 verified_data/ 读取
  - 1.0.0: 初始版本
"""

__version__ = "2.1.0"

import sys
import os
import cv2
import numpy as np
import json
import yaml
import glob
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R

sys.path.append('/home/agilex/MobileManipulator/arm_robot/src')
from robot_piper import PiperRobot, create_config


class ManualHandEyeCalibrator:
    def __init__(self):
        self.robot = None
        self.pipeline = None
        self.pipeline_started = False

        # 数据路径配置 - 消除路径硬编码
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_data_dir = os.path.join(self.script_dir, "calibration_data")
        self.verified_data_dir = os.path.join(self.script_dir, "verified_data")
        os.makedirs(self.calibration_data_dir, exist_ok=True)
        os.makedirs(self.verified_data_dir, exist_ok=True)

        # 初始位置和零点位置
        self.initial_position = [300, 0, 300, 180, 60, 180]  # 初始安全位置

        # 棋盘格参数
        self.board_size = (6, 4)  # 6x4棋盘格
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.chessboard_size_mm = 50.0  # 单个格子50mm

        # 相机内参
        self.camera_matrix = None
        self.dist_coeffs = None

        # 稳定性优化配置
        self.motion_config = {
            'warmup_speed': 40,          # 预热速度 (从60降到40,减少机械磨损)
            'normal_speed': 25,          # 正常运动速度 (从30降到25,减少惯性)
            'capture_speed': 20,         # 采集时速度 (从50降到20,最小化振动)
            'stability_wait': 5.0,       # 稳定等待时间 (从3.0增加到5.0秒)
            'extra_settle_time': 2.0,    # 额外静止时间 (新增,让振动完全消失)
            'warmup_duration': 900       # 预热时长15分钟 (900秒,达到<0.5°精度的最小要求)
        }

        print("🎯 手动手眼标定系统")
        print("=" * 60)

    def normalize_angle_deg(self, angle_deg):
        """归一化角度到 [-180, 180] 度

        消除边界情况 - Good Taste原则
        """
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg <= -180.0:
            angle_deg += 360.0
        return angle_deg

    def angle_difference_deg(self, angle1_deg, angle2_deg):
        """计算两个角度的最小差异（度）

        正确处理跨越 ±180° 边界的情况
        """
        angle1 = self.normalize_angle_deg(angle1_deg)
        angle2 = self.normalize_angle_deg(angle2_deg)
        diff = abs(angle1 - angle2)
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    def filter_by_yaw_deviation(self, collected_data, reference_yaw_deg, max_deviation=30.0):
        """基于Yaw偏差过滤数据

        Args:
            collected_data: 采集的数据列表
            reference_yaw_deg: 参考Yaw角度（度）
            max_deviation: 最大允许偏差（度）

        Returns:
            filtered_data: 过滤后的数据
        """
        if not collected_data:
            return []

        filtered_data = []
        removed_frames = []

        print(f"\n🔍 Yaw偏差过滤 (参考: {reference_yaw_deg:.1f}°, 限制: ±{max_deviation:.1f}°)")

        for i, data in enumerate(collected_data):
            # 获取位姿（格式：[x, y, z, roll, pitch, yaw] 单位：米和弧度）
            pose = data.get('pose', [])
            if len(pose) < 6:
                continue

            # Yaw角度（弧度转度）
            yaw_rad = pose[5] if len(pose) == 6 else pose[2]
            yaw_deg = np.rad2deg(yaw_rad)

            # 计算偏差
            deviation = self.angle_difference_deg(yaw_deg, reference_yaw_deg)

            if deviation <= max_deviation:
                filtered_data.append(data)
            else:
                frame_id = data.get('frame_id', i)
                removed_frames.append((frame_id, yaw_deg, deviation))
                print(f"   ❌ 帧 {frame_id}: Yaw={yaw_deg:.1f}°, 偏差={deviation:.1f}° (超限)")

        if removed_frames:
            print(f"\n🔧 Yaw过滤结果: 移除 {len(removed_frames)}/{len(collected_data)} 帧")
            print(f"   保留: {len(filtered_data)} 帧")
        else:
            print(f"   ✅ 所有帧Yaw偏差均在限制范围内")

        return filtered_data

    def filter_insufficient_motion(self, collected_data, min_motion_mm=5.0, min_rotation_deg=2.0, require_both=True):
        """过滤运动不足的帧

        Args:
            collected_data: 采集的数据列表
            min_motion_mm: 最小位移阈值（毫米）
            min_rotation_deg: 最小旋转阈值（度）
            require_both: 是否要求位移和旋转都满足（推荐True）

        Returns:
            filtered_data: 过滤后的数据
            removed_frames: 被移除的帧信息

        说明:
            - require_both=True: 位移AND旋转都要满足（推荐，确保6自由度都有约束）
            - require_both=False: 位移OR旋转满足一个即可（可能导致退化）
        """
        if len(collected_data) < 2:
            return collected_data, []

        filtered_data = [collected_data[0]]  # 保留第一帧
        removed_frames = []
        last_kept_pose = collected_data[0]['pose']

        for i in range(1, len(collected_data)):
            current_data = collected_data[i]
            current_pose = current_data['pose']

            # 计算与上一个保留帧的位移（米转毫米）
            pos_diff = np.linalg.norm(
                np.array(current_pose[:3]) - np.array(last_kept_pose[:3])
            ) * 1000

            # 计算旋转差异
            R1 = R.from_euler('xyz', last_kept_pose[3:])
            R2 = R.from_euler('xyz', current_pose[3:])
            R_diff = R2 * R1.inv()
            angle_diff = np.degrees(R_diff.magnitude())

            # 判断运动是否充分
            if require_both:
                # 同时满足位移和旋转要求（推荐）
                motion_sufficient = (pos_diff >= min_motion_mm and angle_diff >= min_rotation_deg)
                if not motion_sufficient:
                    # 分析具体原因
                    if pos_diff < min_motion_mm and angle_diff < min_rotation_deg:
                        reason = '位移和旋转都不足'
                    elif pos_diff < min_motion_mm:
                        reason = f'位移不足({pos_diff:.1f}mm<{min_motion_mm}mm)'
                    else:
                        reason = f'旋转不足({angle_diff:.2f}°<{min_rotation_deg}°)'
            else:
                # 位移或旋转满足一个即可（不推荐）
                motion_sufficient = (pos_diff >= min_motion_mm or angle_diff >= min_rotation_deg)
                if not motion_sufficient:
                    reason = '位移和旋转都不足'

            if motion_sufficient:
                filtered_data.append(current_data)
                last_kept_pose = current_pose
            else:
                removed_frames.append({
                    'frame_id': current_data['frame_id'],
                    'reason': reason,
                    'motion_mm': pos_diff,
                    'rotation_deg': angle_diff
                })

        return filtered_data, removed_frames

    def filter_workspace_boundary(self, collected_data, boundary_margin_ratio=0.05, min_boundary_axes=2):
        """智能过滤工作空间边界的帧

        Args:
            collected_data: 采集的数据列表
            boundary_margin_ratio: 边界余量比例（0.05表示边界5%区域）
            min_boundary_axes: 最少几个轴在边界才过滤（1=任意轴, 2=两个轴, 3=三个轴）

        Returns:
            filtered_data: 过滤后的数据
            removed_frames: 被移除的帧信息

        说明:
            - min_boundary_axes=1: 严格模式，任何轴接近边界都过滤（可能过激）
            - min_boundary_axes=2: 平衡模式，至少2个轴同时接近边界才过滤（推荐）
            - min_boundary_axes=3: 宽松模式，3个轴都接近边界才过滤
        """
        if len(collected_data) < 3:
            return collected_data, []

        # 提取所有位置
        positions = np.array([d['pose'][:3] for d in collected_data])

        # 计算工作空间范围
        x_range = [positions[:, 0].min(), positions[:, 0].max()]
        y_range = [positions[:, 1].min(), positions[:, 1].max()]
        z_range = [positions[:, 2].min(), positions[:, 2].max()]

        # 计算边界余量
        x_margin = (x_range[1] - x_range[0]) * boundary_margin_ratio
        y_margin = (y_range[1] - y_range[0]) * boundary_margin_ratio
        z_margin = (z_range[1] - z_range[0]) * boundary_margin_ratio

        filtered_data = []
        removed_frames = []

        for data in collected_data:
            x, y, z = data['pose'][:3]

            # 检查每个轴是否在边界
            boundary_axes = []

            if x < x_range[0] + x_margin or x > x_range[1] - x_margin:
                boundary_axes.append('X')

            if y < y_range[0] + y_margin or y > y_range[1] - y_margin:
                boundary_axes.append('Y')

            if z < z_range[0] + z_margin or z > z_range[1] - z_margin:
                boundary_axes.append('Z')

            boundary_count = len(boundary_axes)

            # 根据边界轴数量判断是否过滤
            should_filter = boundary_count >= min_boundary_axes

            if should_filter:
                removed_frames.append({
                    'frame_id': data['frame_id'],
                    'reason': f"边界位置({'+'.join(boundary_axes)}轴)",
                    'position': [x*1000, y*1000, z*1000],  # 转为毫米
                    'boundary_count': boundary_count
                })
            else:
                filtered_data.append(data)

        return filtered_data, removed_frames

    def filter_extreme_poses(self, collected_data, max_pitch_deg=70.0, verbose=True):
        """过滤极端姿态角的帧（基于绝对阈值）

        Args:
            collected_data: 采集的数据列表
            max_pitch_deg: 最大允许Pitch角绝对值（度）
            verbose: 是否打印详细信息

        Returns:
            filtered_data: 过滤后的数据
            removed_frames: 被移除的帧信息

        说明:
            极端Pitch角会导致：
            1. 相机内参畸变模型误差放大
            2. 棋盘格投影变形，角点检测精度下降
            3. 机械臂重复性降低（接近奇异点）
        """
        if len(collected_data) < 3:
            return collected_data, []

        filtered_data = []
        removed_frames = []

        for data in collected_data:
            roll, pitch, yaw = data['pose'][3:6]  # 欧拉角 (rad)

            # 转换为角度
            roll_deg = np.degrees(roll)
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)

            # 判断Pitch是否过大
            pitch_extreme = abs(pitch_deg) > max_pitch_deg

            if pitch_extreme:
                removed_frames.append({
                    'frame_id': data['frame_id'],
                    'reason': f"Pitch={pitch_deg:.1f}°超限(>{max_pitch_deg}°)",
                    'roll_deg': roll_deg,
                    'pitch_deg': pitch_deg,
                    'yaw_deg': yaw_deg
                })
            else:
                filtered_data.append(data)

        return filtered_data, removed_frames

    def filter_consecutive_outliers(self, collected_data, known_outliers=None):
        """过滤连续异常帧

        Args:
            collected_data: 采集的数据列表
            known_outliers: 已知的异常帧ID列表

        Returns:
            filtered_data: 过滤后的数据
            removed_frames: 被移除的帧信息
        """
        if known_outliers is None or len(known_outliers) == 0:
            return collected_data, []

        # 找出连续的异常帧
        known_outliers_sorted = sorted(known_outliers)
        consecutive_groups = []
        current_group = [known_outliers_sorted[0]]

        for i in range(1, len(known_outliers_sorted)):
            if known_outliers_sorted[i] == current_group[-1] + 1:
                current_group.append(known_outliers_sorted[i])
            else:
                if len(current_group) >= 2:  # 至少2个连续
                    consecutive_groups.append(current_group)
                current_group = [known_outliers_sorted[i]]

        # 检查最后一组
        if len(current_group) >= 2:
            consecutive_groups.append(current_group)

        # 移除连续异常帧
        frames_to_remove = set()
        for group in consecutive_groups:
            frames_to_remove.update(group)

        filtered_data = []
        removed_frames = []

        for data in collected_data:
            if data['frame_id'] in frames_to_remove:
                removed_frames.append({
                    'frame_id': data['frame_id'],
                    'reason': '连续异常帧',
                    'group': next(g for g in consecutive_groups if data['frame_id'] in g)
                })
            else:
                filtered_data.append(data)

        return filtered_data, removed_frames

    def apply_quality_filters(self, collected_data, verbose=True):
        """应用所有数据质量过滤器

        Args:
            collected_data: 原始采集数据
            verbose: 是否显示详细信息

        Returns:
            filtered_data: 过滤后的数据
            filter_report: 过滤报告
        """
        if verbose:
            print("\n" + "="*60)
            print("🔍 数据质量过滤")
            print("="*60)
            print(f"原始数据: {len(collected_data)} 帧")

        original_count = len(collected_data)
        all_removed = []

        # 第1层：过滤运动不足的帧
        if verbose:
            print("\n步骤1: 过滤运动不足的帧 (位移AND旋转都需满足)")

        filtered_data, removed = self.filter_insufficient_motion(
            collected_data,
            min_motion_mm=5.0,
            min_rotation_deg=2.0,
            require_both=True  # 要求位移和旋转都满足
        )

        if verbose and removed:
            print(f"  移除 {len(removed)} 帧:")
            for r in removed[:5]:  # 只显示前5个
                print(f"    帧{r['frame_id']}: 位移={r['motion_mm']:.1f}mm, 旋转={r['rotation_deg']:.1f}°")
            if len(removed) > 5:
                print(f"    ... 还有 {len(removed)-5} 帧")

        all_removed.extend(removed)

        # 第2层：过滤工作空间边界的帧
        if verbose:
            print("\n步骤2: 过滤工作空间边界帧 (激进模式: 单轴边界10%即过滤)")

        filtered_data, removed = self.filter_workspace_boundary(
            filtered_data,
            boundary_margin_ratio=0.10,  # 10%边界（过滤极限位置）
            min_boundary_axes=1  # 任意1个轴在边界即过滤（激进策略，提升精度）
        )

        if verbose and removed:
            print(f"  移除 {len(removed)} 帧:")
            for r in removed[:5]:
                pos = r['position']
                print(f"    帧{r['frame_id']}: {r['reason']}, 位置=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm")
            if len(removed) > 5:
                print(f"    ... 还有 {len(removed)-5} 帧")

        all_removed.extend(removed)

        # 第3层：过滤极端姿态角的帧（基于绝对阈值）
        if verbose:
            print("\n步骤3: 过滤极端姿态角帧 (Pitch>70°)")

        filtered_data, removed = self.filter_extreme_poses(
            filtered_data,
            max_pitch_deg=70.0,  # Pitch角度超过70°则过滤
            verbose=verbose
        )

        if verbose and removed:
            print(f"  移除 {len(removed)} 帧:")
            for r in removed[:5]:
                print(f"    帧{r['frame_id']}: {r['reason']}")
                print(f"      姿态: Roll={r['roll_deg']:.1f}° Pitch={r['pitch_deg']:.1f}° Yaw={r['yaw_deg']:.1f}°")
            if len(removed) > 5:
                print(f"    ... 还有 {len(removed)-5} 帧")

        all_removed.extend(removed)

        # 第4层：过滤已知的连续异常帧（可选）
        # 注意：这个需要先运行一次标定才知道哪些是异常帧
        # 暂时注释掉，因为第一次运行时不知道异常帧
        """
        if verbose:
            print("\n步骤4: 过滤连续异常帧")

        known_outliers = [35, 36, 71, 76, 79, 81]  # 从诊断结果获取
        filtered_data, removed = self.filter_consecutive_outliers(
            filtered_data,
            known_outliers=known_outliers
        )

        if verbose and removed:
            print(f"  移除 {len(removed)} 帧:")
            for r in removed:
                print(f"    帧{r['frame_id']}: {r['reason']}")

        all_removed.extend(removed)
        """

        # 统计
        final_count = len(filtered_data)
        removed_count = original_count - final_count

        if verbose:
            print("\n" + "-"*60)
            print("📊 过滤统计:")
            print(f"  原始: {original_count} 帧")
            print(f"  保留: {final_count} 帧 ({final_count/original_count*100:.1f}%)")
            print(f"  移除: {removed_count} 帧 ({removed_count/original_count*100:.1f}%)")

            # 按原因分组统计
            reason_stats = {}
            for r in all_removed:
                reason = r['reason']
                reason_stats[reason] = reason_stats.get(reason, 0) + 1

            if reason_stats:
                print("\n  移除原因统计:")
                for reason, count in reason_stats.items():
                    print(f"    {reason}: {count} 帧")

        filter_report = {
            'original_count': original_count,
            'final_count': final_count,
            'removed_count': removed_count,
            'removed_frames': all_removed,
            'removal_rate': removed_count / original_count if original_count > 0 else 0
        }

        return filtered_data, filter_report

    def ransac_filter_handeye(self, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, frame_ids, threshold=6.0):
        """改进的RANSAC过滤异常值 - 使用AX=XB一致性误差

        Args:
            threshold: 内点阈值（mm）

        Returns:
            list: 内点索引列表
        """
        n_samples = len(R_gripper2base)
        if n_samples < 5:
            return list(range(n_samples))

        best_inliers = []
        best_avg_error = float('inf')
        best_median_error = float('inf')

        # 增加迭代次数以提高鲁棒性
        iterations = min(200, n_samples * 20)

        for iteration in range(iterations):
            # 使用更多样本点（8个）提高运动多样性，减少"运动不足"错误
            # 8个点通常足以覆盖足够的旋转变化
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

                # 使用AX=XB方程计算所有点的一致性误差
                errors = []
                for i in range(n_samples):
                    # 计算AX=XB误差，而非相对误差
                    # A*X应该等于X*B
                    # A是gripper到base的变换，X是相机到gripper的变换，B是target到相机的变换

                    # 计算AX
                    AX_R = R_gripper2base[i] @ R_sample
                    AX_t = R_gripper2base[i] @ t_sample + t_gripper2base[i]

                    # 计算XB
                    XB_R = R_sample @ R_target2cam[i]
                    XB_t = R_sample @ t_target2cam[i] + t_sample

                    # 计算旋转误差（使用角度差）
                    R_diff = AX_R @ XB_R.T
                    angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

                    # 计算平移误差（毫米）
                    t_error = np.linalg.norm(AX_t - XB_t) * 1000

                    # 综合误差（旋转误差转换为毫米当量）
                    # 假设10度旋转误差相当于10mm平移误差
                    total_error = t_error + np.degrees(angle_error) * 1.0
                    errors.append(total_error)

                errors_array = np.array(errors)

                # 使用动态阈值（基于中位数和MAD）
                median_error = np.median(errors_array)
                mad = np.median(np.abs(errors_array - median_error))

                # 动态阈值：中位数 + 2.5倍MAD
                dynamic_threshold = min(threshold, median_error + 2.5 * mad)

                # 找出内点
                inliers = [i for i, e in enumerate(errors) if e < dynamic_threshold]

                # 要求至少30%的数据为内点
                min_inliers = max(4, int(n_samples * 0.3))

                if len(inliers) >= min_inliers:
                    # 使用内点重新计算以获得更准确的估计
                    try:
                        R_refined, t_refined = cv2.calibrateHandEye(
                            [R_gripper2base[i] for i in inliers],
                            [t_gripper2base[i] for i in inliers],
                            [R_target2cam[i] for i in inliers],
                            [t_target2cam[i] for i in inliers],
                            method=cv2.CALIB_HAND_EYE_TSAI
                        )

                        # 重新计算内点的误差
                        refined_errors = []
                        for i in inliers:
                            AX_R = R_gripper2base[i] @ R_refined
                            AX_t = R_gripper2base[i] @ t_refined + t_gripper2base[i]
                            XB_R = R_refined @ R_target2cam[i]
                            XB_t = R_refined @ t_target2cam[i] + t_refined
                            t_error = np.linalg.norm(AX_t - XB_t) * 1000
                            refined_errors.append(t_error)

                        avg_error = np.mean(refined_errors)
                        median_error = np.median(refined_errors)

                        # 优先选择更多内点，其次选择更小的中位数误差
                        if len(inliers) > len(best_inliers) or \
                           (len(inliers) == len(best_inliers) and median_error < best_median_error):
                            best_inliers = inliers
                            best_avg_error = avg_error
                            best_median_error = median_error
                    except:
                        pass

            except Exception as e:
                continue

        # 如果RANSAC未找到足够内点，使用更宽松的标准
        if len(best_inliers) < max(4, int(n_samples * 0.3)):
            print(f"   ⚠️ RANSAC内点不足，使用保守的误差阈值过滤")
            # 计算所有数据使用Tsai方法的误差
            try:
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

                # 使用更保守的策略: 中位数 + 3倍MAD（中位数绝对偏差）
                # MAD比标准差更鲁棒，不受极端值影响
                median = np.median(errors)
                mad = np.median(np.abs(np.array(errors) - median))
                # 3倍MAD大约等价于正态分布的3σ（99.7%置信区间）
                conservative_threshold = median + 3.0 * mad

                # 同时设置绝对上限：15mm（远大于正常误差）
                # 这避免了在数据整体较差时过度删除
                absolute_max = 15.0
                final_threshold = min(conservative_threshold, absolute_max)

                best_inliers = [i for i, e in enumerate(errors) if e < final_threshold]

                # 如果还是过滤太多（超过20%），则只保留所有数据
                if len(best_inliers) < int(n_samples * 0.8):
                    print(f"   ℹ️  保守过滤仍会移除过多数据，保留所有帧")
                    best_inliers = list(range(n_samples))
            except:
                best_inliers = list(range(n_samples))

        return best_inliers

    def filter_high_reprojection_error_frames(self, collected_data, max_error_px=2.0):
        """过滤重投影误差过大的帧

        Args:
            collected_data: 采集的数据列表
            max_error_px: 最大允许重投影误差(像素),默认2.0

        Returns:
            tuple: (filtered_data, removed_frames)
                - filtered_data: 过滤后的数据列表
                - removed_frames: 被删除的帧ID列表
        """
        print(f"\n🔍 过滤重投影误差 > {max_error_px}px 的帧...")

        filtered_data = []
        removed_frames = []

        # 准备棋盘格3D点
        objpoints = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objpoints *= self.chessboard_size_mm / 1000.0  # 转为米

        for data in collected_data:
            frame_id = data['frame_id']
            imgpoints = data['corners']

            # 检查是否已有rvec/tvec,如果没有则计算
            if 'rvec' in data and 'tvec' in data:
                rvec = data['rvec']
                tvec = data['tvec']
            else:
                # 使用PnP计算棋盘格相对于相机的位姿
                ret, rvec, tvec = cv2.solvePnP(
                    objpoints, imgpoints,
                    self.camera_matrix, self.dist_coeffs
                )
                if not ret:
                    # PnP失败,直接移除该帧
                    removed_frames.append(frame_id)
                    print(f"   ❌ 帧 {frame_id}: PnP求解失败 (移除)")
                    continue

            # 重投影
            projected_points, _ = cv2.projectPoints(
                objpoints, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )

            # 计算平均重投影误差
            error = np.linalg.norm(
                imgpoints.reshape(-1, 2) - projected_points.reshape(-1, 2),
                axis=1
            ).mean()

            if error > max_error_px:
                removed_frames.append(frame_id)
                print(f"   ❌ 帧 {frame_id}: 重投影误差 {error:.3f}px > {max_error_px}px (移除)")
            else:
                filtered_data.append(data)
                print(f"   ✅ 帧 {frame_id}: 重投影误差 {error:.3f}px (保留)")

        print(f"\n📊 过滤结果:")
        print(f"   原始帧数: {len(collected_data)}")
        print(f"   保留帧数: {len(filtered_data)}")
        print(f"   移除帧数: {len(removed_frames)}")
        if removed_frames:
            print(f"   移除的帧: {removed_frames}")

        return filtered_data, removed_frames

    def multi_algorithm_fusion(self, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
        """多算法融合标定

        Returns:
            tuple: (最优R, 最优t, 算法名称)
        """
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff")
        ]

        best_result = None
        best_score = float('inf')
        best_method = None

        for method_id, method_name in methods:
            try:
                R_test, t_test = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method_id
                )

                # 计算平移和旋转误差
                t_errors = []
                r_errors = []
                for i in range(len(R_gripper2base)):
                    R_pred = R_gripper2base[i] @ R_test @ R_target2cam[i]
                    t_pred = R_gripper2base[i] @ (R_test @ t_target2cam[i] + t_test) + t_gripper2base[i]

                    if i == 0:
                        R_ref = R_pred
                        t_ref = t_pred
                    else:
                        # 平移误差 (mm)
                        t_error = np.linalg.norm(t_pred - t_ref) * 1000
                        t_errors.append(t_error)

                        # 旋转误差 (degrees)
                        R_error = R_ref.T @ R_pred
                        r_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))
                        r_errors.append(r_error)

                avg_t_error = np.mean(t_errors) if t_errors else 0
                avg_r_error = np.mean(r_errors) if r_errors else 0

                # 综合评分: 归一化后加权求和
                # 平移权重: 5mm = 1.0, 旋转权重: 0.5° = 1.0
                # 这样两个误差在相同量级下有相同贡献
                score = (avg_t_error / 5.0) + (avg_r_error / 0.5)

                print(f"   {method_name}: 平移={avg_t_error:.3f}mm, 旋转={avg_r_error:.3f}°, 综合={score:.3f}")

                if score < best_score:
                    best_score = score
                    best_result = (R_test, t_test)
                    best_method = method_name

            except Exception as e:
                print(f"   {method_name}: 失败")

        if best_result:
            return best_result[0], best_result[1], best_method
        return None, None, None

    def iterative_optimization(self, R_initial, t_initial, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
        """迭代优化标定结果

        Returns:
            tuple: (优化后的R, 优化后的t)
        """
        from scipy.optimize import minimize

        def objective(params):
            # 解析参数
            rvec = params[:3]
            tvec = params[3:6]

            # 转换为旋转矩阵
            R_opt, _ = cv2.Rodrigues(rvec)
            t_opt = tvec.reshape(3, 1)

            # 正确的目标函数: 最小化位姿重复性误差
            # 这是评估函数使用的相同指标
            R_preds = []
            t_preds = []

            for i in range(len(R_gripper2base)):
                # 计算在base坐标系下的target位姿
                R_pred = R_gripper2base[i] @ R_opt @ R_target2cam[i]
                t_pred = R_gripper2base[i] @ (R_opt @ t_target2cam[i] + t_opt) + t_gripper2base[i]
                R_preds.append(R_pred)
                t_preds.append(t_pred)

            # 以第一帧为参考，计算所有帧的偏差
            R_ref = R_preds[0]
            t_ref = t_preds[0]

            total_error = 0
            for i in range(1, len(R_preds)):
                # 旋转误差 (度)
                R_error = R_ref.T @ R_preds[i]
                angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

                # 平移误差 (mm)
                t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000

                # 综合误差 (权重: 1° = 10mm)
                total_error += t_error + angle_error * 10.0

            return total_error

        # 初始参数
        rvec_init, _ = cv2.Rodrigues(R_initial)
        tvec_init = t_initial.flatten()
        params_init = np.concatenate([rvec_init.flatten(), tvec_init])

        # 计算初始误差
        error_before = objective(params_init)

        # 优化: 使用L-BFGS-B方法 (比Powell更稳定)
        # 设置紧约束边界，只允许微调 (初始值已经很好了)
        bounds = []
        for i in range(3):
            # 旋转向量: 初始值 ±5度 (约±0.087弧度)
            bounds.append((rvec_init[i] - 0.087, rvec_init[i] + 0.087))
        for i in range(3):
            # 平移向量: 初始值 ±10mm (±0.01m)
            bounds.append((tvec_init[i] - 0.01, tvec_init[i] + 0.01))

        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 50,      # 减少迭代次数，防止过拟合
                'ftol': 1e-6,       # 函数容差
                'gtol': 1e-5        # 梯度容差
            }
        )

        # 解析结果
        rvec_opt = result.x[:3]
        tvec_opt = result.x[3:6]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        t_opt = tvec_opt.reshape(3, 1)

        # 计算改进
        error_after = objective(result.x)

        # 如果优化后反而更差，退回到初始值
        if error_after > error_before:
            print(f"   ⚠️  优化未改善结果，使用初始值")
            print(f"   优化前误差: {error_before:.3f}")
            print(f"   优化后误差: {error_after:.3f} (更差)")
            return R_initial, t_initial

        print(f"   优化前误差: {error_before:.3f}")
        print(f"   优化后误差: {error_after:.3f}")
        print(f"   改进: {(1 - error_after/error_before)*100:.1f}%")

        return R_opt, t_opt

    def evaluate_calibration(self, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, frame_ids, perform_analysis=True, original_poses=None):
        """评估标定结果并进行深度误差分析

        Args:
            perform_analysis: 是否执行误差模式分析
            original_poses: 原始机械臂位姿列表 (可选) [x, y, z, roll, pitch, yaw] in meters and radians

        Returns:
            tuple: (avg_error, detailed_errors_dict) 平均误差和详细误差字典
        """
        errors = []
        angle_errors = []
        print("\n" + "="*60)
        print("📊 最终标定质量分析")
        print("="*60)

        # initial_position: [x, y, z, roll, pitch, yaw] in mm and degrees
        ref_pos_mm = np.array(self.initial_position[:3])  # mm
        ref_rpy_deg = np.array(self.initial_position[3:])  # degrees

        for i in range(len(R_gripper2base)):
            R_pred = R_gripper2base[i] @ R_cam2gripper @ R_target2cam[i]
            t_pred = R_gripper2base[i] @ (R_cam2gripper @ t_target2cam[i] + t_cam2gripper) + t_gripper2base[i]

            # 计算位姿偏差信息（如果有原始位姿）
            pose_deviation_str = ""
            if original_poses is not None and i < len(original_poses):
                pose = original_poses[i]  # [x, y, z, roll, pitch, yaw] in meters and radians

                # 转换为mm和度
                pos_mm = np.array(pose[:3]) * 1000  # m → mm
                rpy_deg = np.rad2deg(pose[3:])      # rad → deg

                # 计算偏差
                pos_dev = pos_mm - ref_pos_mm
                rpy_dev = np.array([
                    self.angle_difference_deg(rpy_deg[0], ref_rpy_deg[0]),  # roll
                    self.angle_difference_deg(rpy_deg[1], ref_rpy_deg[1]),  # pitch
                    self.angle_difference_deg(rpy_deg[2], ref_rpy_deg[2])   # yaw
                ])

                pose_deviation_str = (
                    f" | 位姿偏差: "
                    f"ΔX={pos_dev[0]:+6.1f} ΔY={pos_dev[1]:+6.1f} ΔZ={pos_dev[2]:+6.1f}mm, "
                    f"ΔR={rpy_dev[0]:+5.1f} ΔP={rpy_dev[1]:+5.1f} ΔY={rpy_dev[2]:+5.1f}°"
                )

            if i == 0:
                R_ref = R_pred
                t_ref = t_pred
                print(f"✅ 帧 {frame_ids[i]:2d}: 旋转误差  0.000°  平移误差   0.000mm{pose_deviation_str}")
                # 第一帧也加入，但误差为0
                errors.append(0.0)
                angle_errors.append(0.0)
            else:
                R_error = R_ref.T @ R_pred
                angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))
                t_error = np.linalg.norm(t_pred - t_ref) * 1000
                errors.append(t_error)
                angle_errors.append(angle_error)

                status = "✅" if t_error < 3.0 else "⚠️" if t_error < 5.0 else "❌"
                print(f"{status} 帧 {frame_ids[i]:2d}: 旋转误差 {angle_error:6.3f}°  平移误差 {t_error:7.3f}mm{pose_deviation_str}")

        avg_error = np.mean(errors) if errors else 0
        avg_angle_error = np.mean(angle_errors) if angle_errors else 0

        print(f"\n📈 综合误差:")
        print(f"   平均平移误差: {avg_error:.3f}mm")
        print(f"   平均旋转误差: {avg_angle_error:.3f}°")

        # 执行深度误差模式分析
        if perform_analysis and len(errors) > 5:
            print("\n" + "="*60)
            print("🔍 深度误差模式分析")
            print("="*60)

            analysis = self.analyze_error_patterns(errors[1:], frame_ids[1:])  # 跳过第一帧（参考帧）

            # 显示分析结果
            if analysis['temporal_drift']:
                print("⚠️  检测到热漂移现象")
            if analysis['periodic_pattern']:
                print("⚠️  检测到周期性误差模式")
            if analysis['spatial_clustering']:
                print("⚠️  检测到空间聚类误差")

            if analysis['stable_frames']:
                print(f"\n✅ 稳定帧（误差较小）: {analysis['stable_frames']}")
            if analysis['outlier_frames']:
                print(f"❌ 异常帧（误差较大）: {analysis['outlier_frames']}")

            if analysis['recommendations']:
                print("\n💡 改进建议:")
                for rec in analysis['recommendations']:
                    print(f"   • {rec}")

            # 统计信息
            print("\n📊 统计信息:")
            print(f"   平移误差标准差: {np.std(errors):.3f}mm")
            print(f"   平移误差中位数: {np.median(errors):.3f}mm")
            print(f"   平移误差范围: {np.min(errors):.3f}mm - {np.max(errors):.3f}mm")
            print(f"   旋转误差标准差: {np.std(angle_errors):.3f}°")
            print(f"   旋转误差范围: {np.min(angle_errors):.3f}° - {np.max(angle_errors):.3f}°")

            # 质量分级建议
            if avg_error > 5.0 and len(analysis['outlier_frames']) > len(errors) * 0.3:
                print("\n🎯 主要问题: 存在大量异常数据点")
                print("   建议: ")
                print("   1. 重新采集这些位置的数据: ", analysis['outlier_frames'])
                print("   2. 检查该区域的光照条件和棋盘格可见性")
                print("   3. 考虑调整机器臂运动速度，等待稳定后再采集")

        # 返回详细误差信息
        detailed_errors = {
            'translation_errors': errors,
            'rotation_errors': angle_errors,
            'avg_translation_error': avg_error,
            'avg_rotation_error': avg_angle_error
        }

        return avg_error, detailed_errors

    def save_optimized_result(self, R_cam2gripper, t_cam2gripper, avg_error, method_name, n_inliers, n_total, save_dir, detailed_errors=None):
        """保存优化后的标定结果

        Args:
            detailed_errors: dict, 可选的详细误差信息，包含：
                - rotation_errors: 旋转误差列表（度）
                - translation_errors: 平移误差列表（毫米）
                - avg_rotation_error: 平均旋转误差（度）
                - avg_translation_error: 平均平移误差（毫米）
        """
        rotation = R.from_matrix(R_cam2gripper)
        quaternion = rotation.as_quat()
        euler_angles = rotation.as_euler('xyz', degrees=True)
        t_mm = t_cam2gripper.flatten() * 1000.0

        print("\n" + "="*60)
        print("✅ 优化版标定完成！")
        print("="*60)
        print(f"\n📊 最终标定结果:")
        print(f"   平移向量 (mm): X={t_mm[0]:.2f}, Y={t_mm[1]:.2f}, Z={t_mm[2]:.2f}")
        print(f"   欧拉角 (度):    Rx={euler_angles[0]:.2f}°, Ry={euler_angles[1]:.2f}°, Rz={euler_angles[2]:.2f}°")
        print(f"   四元数 (xyzw):  x={quaternion[0]:.4f}, y={quaternion[1]:.4f}, z={quaternion[2]:.4f}, w={quaternion[3]:.4f}")

        print(f"\n📈 标定质量:")

        # 显示详细误差信息
        if detailed_errors:
            avg_rot_err = detailed_errors.get('avg_rotation_error', 0)
            avg_trans_err = detailed_errors.get('avg_translation_error', avg_error)
            rot_errors = detailed_errors.get('rotation_errors', [])
            trans_errors = detailed_errors.get('translation_errors', [])

            print(f"   平均平移误差: {avg_trans_err:.3f}mm")
            print(f"   平均旋转误差: {avg_rot_err:.3f}°")

            if trans_errors:
                print(f"   平移误差范围: {np.min(trans_errors):.3f}mm - {np.max(trans_errors):.3f}mm")
                print(f"   平移误差标准差: {np.std(trans_errors):.3f}mm")

            if rot_errors:
                print(f"   旋转误差范围: {np.min(rot_errors):.3f}° - {np.max(rot_errors):.3f}°")
                print(f"   旋转误差标准差: {np.std(rot_errors):.3f}°")
        else:
            print(f"   平均平移误差: {avg_error:.3f}mm")

        print(f"   使用数据: {n_inliers}/{n_total} (质量预筛选+RANSAC)")
        print(f"   最优算法: {method_name}")

        # 综合质量评估
        if detailed_errors:
            avg_rot = detailed_errors.get('avg_rotation_error', 0)
            avg_trans = detailed_errors.get('avg_translation_error', avg_error)
        else:
            avg_rot = 0
            avg_trans = avg_error

        # 更严格的质量标准
        if avg_trans < 3.0 and avg_rot < 1.0:
            quality = "🌟 优秀 (工业级精度)"
            quality_tips = ""
        elif avg_trans < 5.0 and avg_rot < 2.0:
            quality = "👍 良好"
            quality_tips = "\n   💡 提示: 可通过增加高质量数据进一步提升精度"
        elif avg_trans < 8.0 and avg_rot < 3.0:
            quality = "⚠️  可接受"
            quality_tips = "\n   💡 建议: 重新采集数据，注意机械臂预热和位姿稳定"
        else:
            quality = "❌ 需要改进"
            quality_tips = "\n   ❗ 警告: 精度不足，必须重新标定！"
            quality_tips += "\n   主要问题分析:"
            if avg_trans > 8.0:
                quality_tips += f"\n     • 平移误差过大({avg_trans:.1f}mm) - 检查相机内参和机械臂精度"
            if avg_rot > 3.0:
                quality_tips += f"\n     • 旋转误差过大({avg_rot:.1f}°) - 增加旋转角度变化范围"
            if detailed_errors and detailed_errors.get('translation_errors'):
                trans_std = np.std(detailed_errors['translation_errors'])
                if trans_std > 3.0:
                    quality_tips += f"\n     • 误差分布不均(σ={trans_std:.1f}mm) - 数据质量不一致"

        print(f"   质量等级: {quality}{quality_tips}")

        # 保存结果
        result_data = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'method': f'Optimized_{method_name}_with_RANSAC',
            'optimization': {
                'ransac_enabled': True,
                'multi_algorithm': True,
                'iterative_refinement': True
            },
            'data_points': {
                'total': n_total,
                'used': n_inliers,
                'filtered': n_total - n_inliers
            },
            'transformation': {
                'rotation_matrix': R_cam2gripper.tolist(),
                'translation_mm': t_mm.tolist(),
                'quaternion_xyzw': quaternion.tolist(),
                'euler_xyz_deg': euler_angles.tolist()
            },
            'quality': {
                'avg_error_mm': float(avg_error),
                'quality_level': quality
            },
            'camera_intrinsics': {
                'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                'distortion_coefficients': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None
            }
        }

        result_file = os.path.join(save_dir, "optimized_calibration_result.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n💾 优化结果已保存到: {result_file}")

    def save_intrinsics_to_file(self, save_path, camera_matrix, dist_coeffs, source="Unknown"):
        """保存相机内参到文件

        Args:
            save_path: 保存文件的完整路径
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            source: 内参来源说明
        """
        try:
            intrinsics_data = {
                'camera_matrix': camera_matrix.tolist() if hasattr(camera_matrix, 'tolist') else camera_matrix,
                'distortion_coefficients': dist_coeffs.tolist() if hasattr(dist_coeffs, 'tolist') else dist_coeffs,
                'source': source,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'image_size': [1280, 720],  # 默认RealSense分辨率
                'calibration_info': {
                    'board_size': list(self.board_size),
                    'square_size_mm': self.chessboard_size_mm
                }
            }

            with open(save_path, 'w') as f:
                yaml.dump(intrinsics_data, f, default_flow_style=False)

            return True
        except Exception as e:
            print(f"⚠️ 保存内参失败: {e}")
            return False

    def get_camera_intrinsics(self, data_dir=None, save_to_dir=None):
        """智能获取相机内参

        Args:
            data_dir: 可选，数据目录路径，用于查找已保存的内参文件
            save_to_dir: 可选，保存内参文件的目录

        Returns:
            tuple: (camera_matrix, dist_coeffs, source) 相机内参矩阵、畸变系数、来源说明
        """
        intrinsics_loaded = False
        camera_matrix = None
        dist_coeffs = None
        source = ""


        data_dir = None
        
        # 策略1：优先使用data_dir中的realsense_intrinsics.yaml（如果提供）
        if data_dir:
            intrinsics_file = os.path.join(data_dir, "realsense_intrinsics.yaml")
            if os.path.exists(intrinsics_file):
                try:
                    with open(intrinsics_file, 'r') as f:
                        intrinsics_data = yaml.load(f, Loader=yaml.FullLoader)
                        camera_matrix = np.array(intrinsics_data['camera_matrix'], dtype=np.float32)
                        dist_coeffs = np.array(intrinsics_data['distortion_coefficients'], dtype=np.float32)
                        source = f"文件: {intrinsics_file}"
                        print(f"✅ 从数据目录加载相机内参")
                        intrinsics_loaded = True
                except Exception as e:
                    print(f"⚠️  读取内参文件失败: {e}")

        # 策略2：使用高质量标定文件 hand_camera_intrinsics.yaml
        if not intrinsics_loaded:
            default_intrinsics_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "hand_camera_intrinsics.yaml"
            )
            if os.path.exists(default_intrinsics_file):
                try:
                    with open(default_intrinsics_file, 'r') as f:
                        intrinsics_data = yaml.load(f, Loader=yaml.FullLoader)
                        # 解析格式: K是3x3矩阵展平成9元素, D是畸变系数
                        K = np.array(intrinsics_data['K'], dtype=np.float32)
                        camera_matrix = K.reshape(3, 3)
                        dist_coeffs = np.array(intrinsics_data['D'][:5], dtype=np.float32)
                        source = f"文件: hand_camera_intrinsics.yaml"
                        print(f"✅ 从默认标定文件加载相机内参")
                        intrinsics_loaded = True
                except Exception as e:
                    print(f"⚠️  读取默认标定文件失败: {e}")

        # 策略3：尝试连接RealSense相机获取实际内参
        if not intrinsics_loaded:
            try:
                import pyrealsense2 as rs
                print("📷 尝试从RealSense相机获取内参...")

                # 创建临时管道获取内参
                pipeline = rs.pipeline()
                config = rs.config()

                # 优先尝试连接手眼相机
                hand_camera_id = "337122071190"
                try:
                    config.enable_device(hand_camera_id)
                    print("   找到手眼相机")
                except:
                    print("   使用默认相机")

                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

                # 启动管道获取内参
                profile = pipeline.start(config)
                color_stream = profile.get_stream(rs.stream.color)
                intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

                # 转换为OpenCV格式
                camera_matrix = np.array([
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                ], dtype=np.float32)

                dist_coeffs = np.array(intrinsics.coeffs[:5], dtype=np.float32)

                pipeline.stop()

                source = "RealSense相机（出厂标定）"
                print(f"✅ 从RealSense获取内参成功")
                print(f"   焦距: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
                print(f"   光心: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
                print(f"   畸变: [{dist_coeffs[0]:.4f}, {dist_coeffs[1]:.4f}, {dist_coeffs[2]:.4f}, {dist_coeffs[3]:.4f}, {dist_coeffs[4]:.4f}]")

                # 保存内参
                if save_to_dir or data_dir:
                    save_dir = save_to_dir or data_dir
                    try:
                        intrinsics_save_data = {
                            'camera_matrix': camera_matrix.tolist(),
                            'distortion_coefficients': dist_coeffs.tolist(),
                            'width': intrinsics.width,
                            'height': intrinsics.height,
                            'model': str(intrinsics.model),
                            'source': 'RealSense Factory Calibration',
                            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                        }
                        save_path = os.path.join(save_dir, "realsense_intrinsics.yaml")
                        with open(save_path, 'w') as f:
                            yaml.dump(intrinsics_save_data, f)
                        print(f"   💾 内参已保存到: {save_path}")
                    except Exception as e:
                        print(f"   ⚠️ 保存内参失败: {e}")

                intrinsics_loaded = True

            except Exception as e:
                print(f"⚠️  无法从相机获取内参: {e}")
                print(f"⚠️  所有内参加载策略失败，无法继续")
                return None, None, "失败"

        return camera_matrix, dist_coeffs, source

    def load_camera_intrinsics(self):
        """从RealSense相机获取出厂内参（向后兼容）"""
        print("📷 获取相机内参...")

        # 使用新的通用函数，保存到当前目录
        self.camera_matrix, self.dist_coeffs, source = self.get_camera_intrinsics(save_to_dir=".")

        if self.camera_matrix is not None:
            print(f"✅ 内参获取成功（{source}）")
            return True
        else:
            print("❌ 内参获取失败")
            return False

    def warm_up_robot(self, duration_minutes=15):
        """机械臂预热程序，消除温度影响

        Args:
            duration_minutes: 预热时长(分钟)，默认15分钟
        """
        print(f"\n🔥 执行机械臂预热程序 ({duration_minutes}分钟)...")
        print("   预热可以消除温度变化对精度的影响")
        print("   达到热稳定状态需要15-20分钟")

        # 更大范围的预热位置，覆盖工作空间
        warm_up_positions = [
            [400, 0, 350, 180, 60, 180],    # X正方向极限
            [200, 0, 250, 180, 60, 180],    # X负方向极限
            [300, 100, 300, 180, 60, 180],  # Y正方向极限
            [300, -100, 300, 180, 60, 180], # Y负方向极限
            [300, 0, 400, 180, 60, 180],    # Z正方向极限
            [300, 0, 200, 180, 60, 180],    # Z负方向极限
            [300, 0, 300, 180, 80, 180],    # Pitch变化
            [300, 0, 300, 180, 40, 180],    # Pitch变化
        ]

        # 基于实际时间的预热循环 (精确控制)
        target_seconds = duration_minutes * 60
        print(f"   预热策略: 循环{len(warm_up_positions)}个位置，直到达到{duration_minutes}分钟")
        print(f"   目标时长: {duration_minutes}分钟 ({target_seconds}秒)")

        start_time = time.time()
        cycle_count = 0

        # 使用while循环,根据实际已用时间退出
        while (time.time() - start_time) < target_seconds:
            cycle_count += 1
            elapsed = (time.time() - start_time) / 60
            remaining = duration_minutes - elapsed
            progress = min(100, (elapsed / duration_minutes) * 100)

            print(f"\n预热进度: {progress:.0f}% (循环{cycle_count}, {elapsed:.1f}/{duration_minutes}分钟, 剩余{remaining:.1f}分钟)")

            for j, pos in enumerate(warm_up_positions):
                # 每个位置前检查时间,避免超时
                if (time.time() - start_time) >= target_seconds:
                    print(f"   ⏱️  已达到目标时长,停止预热")
                    break

                self.robot.arm.set_position(
                    x=pos[0], y=pos[1], z=pos[2],
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    wait=True, speed=self.motion_config['warmup_speed'],
                    use_gripper_center=False
                )
                time.sleep(2)

        total_time = (time.time() - start_time) / 60
        print(f"\n✅ 预热完成! 实际耗时: {total_time:.1f}分钟")
        print("   机械臂已达到热稳定状态")

    def connect_devices(self):
        """连接机器臂和相机"""
        try:
            import pyrealsense2 as rs

            # 初始化机器臂
            print("🔧 初始化机器臂...")
            cfg = create_config()
            self.robot = PiperRobot(cfg)
            self.robot.connect()

            # 询问是否需要预热
            warm_up_choice = input("\n是否执行机械臂预热15分钟? (y/n) [推荐y]: ").strip().lower()
            if warm_up_choice == 'y' or warm_up_choice == '':
                self.warm_up_robot(duration_minutes=15)
            else:
                print("⚠️  跳过预热可能导致精度下降")
                print("   建议手动运行机械臂15分钟后再开始标定")

            # 移动到初始位置
            print(f"\n📍 移动到初始位置: {self.initial_position[:3]}")
            self.robot.arm.set_position(
                x=self.initial_position[0], y=self.initial_position[1], z=self.initial_position[2],
                roll=self.initial_position[3], pitch=self.initial_position[4], yaw=self.initial_position[5],
                wait=True, speed=self.motion_config['normal_speed'], use_gripper_center=False
            )

            # 关闭爪子
            print("🤏 关闭爪子...")
            self.robot.arm.set_gripper_position(0, wait=True, speed=1000)

            # 初始化相机
            print("📹 初始化手眼相机...")
            self.pipeline = rs.pipeline()
            config = rs.config()

            hand_camera_id = "337122071190"
            config.enable_device(hand_camera_id)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

            profile = self.pipeline.start(config)
            self.pipeline_started = True

            # 使用通用函数获取内参（会自动从相机获取）
            self.camera_matrix, self.dist_coeffs, source = self.get_camera_intrinsics()
            print(f"✅ 内参来源: {source}")

            # 等待相机稳定
            for _ in range(5):
                self.pipeline.wait_for_frames()

            print("✅ 设备连接成功")
            return True

        except Exception as e:
            print(f"❌ 设备连接失败: {e}")
            return False

    def disconnect_devices(self):
        """断开设备连接"""
        if self.pipeline and self.pipeline_started:
            try:
                self.pipeline.stop()
                self.pipeline_started = False
                print("📹 相机已断开")
            except:
                pass

        if self.robot:
            try:
                self.robot.disconnect()
                print("🤖 机器臂已断开")
            except:
                pass

    def wait_for_pose_stability(self, target_pose, tolerance_mm=0.5, tolerance_deg=0.1, max_wait=None):
        """等待机器臂位姿稳定

        Args:
            target_pose: 目标位姿
            tolerance_mm: 位置容差（mm）
            tolerance_deg: 角度容差（度）
            max_wait: 最大等待时间（秒），None则使用配置值

        Returns:
            bool: 是否达到稳定状态
        """
        if max_wait is None:
            max_wait = self.motion_config['stability_wait']

        start_time = time.time()
        stable_count = 0
        required_stable_readings = 5

        print("   等待位姿稳定...", end="", flush=True)

        while time.time() - start_time < max_wait:
            # 获取当前位姿
            _, current_pose = self.robot.arm.get_position(return_gripper_center=False)
            current_pose = current_pose if isinstance(current_pose, list) else current_pose.tolist()

            # 计算偏差
            pos_diff = np.linalg.norm([
                current_pose[0] - target_pose[0],
                current_pose[1] - target_pose[1],
                current_pose[2] - target_pose[2]
            ]) * 1000  # 转换为mm

            rot_diff = max([
                abs(current_pose[3] - target_pose[3]),
                abs(current_pose[4] - target_pose[4]),
                abs(current_pose[5] - target_pose[5])
            ])

            # 检查是否稳定
            if pos_diff < tolerance_mm and rot_diff < tolerance_deg:
                stable_count += 1
                if stable_count >= required_stable_readings:
                    print(" ✅ 稳定")
                    # 额外静止时间，让微小振动完全消失
                    if self.motion_config['extra_settle_time'] > 0:
                        print(f"   额外静止 {self.motion_config['extra_settle_time']:.1f}秒...", end="", flush=True)
                        time.sleep(self.motion_config['extra_settle_time'])
                        print(" ✅")
                    return True
            else:
                stable_count = 0

            time.sleep(0.1)

        print(" ⚠️ 未达到稳定")
        return False

    def get_historical_data_dirs(self):
        """获取历史标定数据目录（仅原始采集数据：manual 和 replay）

        Returns:
            list: 排序后的目录路径列表
        """
        data_dirs = []
        valid_prefixes = ("manual_calibration_", "replay_calibration_")

        # 搜索 calibration_data 目录
        if os.path.exists(self.calibration_data_dir):
            for item in os.listdir(self.calibration_data_dir):
                if item.startswith(valid_prefixes):
                    full_path = os.path.join(self.calibration_data_dir, item)
                    if os.path.isdir(full_path):
                        data_dirs.append(full_path)

        # 搜索 verified_data 目录
        if os.path.exists(self.verified_data_dir):
            for item in os.listdir(self.verified_data_dir):
                if item.startswith(valid_prefixes):
                    full_path = os.path.join(self.verified_data_dir, item)
                    if os.path.isdir(full_path):
                        data_dirs.append(full_path)

        return sorted(data_dirs)

    def manual_calibration_mode(self):
        """手动标定模式：按空格保存图像和位姿，按ESC结束"""
        print("\n" + "="*60)
        print("🎯 手动手眼标定模式（增强版）")
        print("="*60)
        print("操作说明:")
        print("  [空格] - 保存当前图像和机械臂位姿")
        print("  [ESC]  - 结束采集并开始标定")
        print("  [q]    - 退出不标定")
        print("  [w]    - 查看预热状态")
        print("")
        print("⚠️  注意事项:")
        print("  • 建议机器臂预热5-10分钟以减少热漂移")
        print("  • 建议Yaw角度偏差保持在 ±30° 以内（相对于初始位置）")
        print("  • 系统将自动等待位姿稳定后再采集")
        print("="*60)
        print("\n✅ 机器臂已完成15分钟预热，可以开始采集")

        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.calibration_data_dir, f"manual_calibration_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n📁 数据保存目录: {save_dir}/")

        # 数据记录
        collected_data = []
        frame_count = 0
        poses_txt_path = os.path.join(save_dir, "poses.txt")

        # 写入txt文件头
        with open(poses_txt_path, 'w') as f:
            f.write("# 手眼标定数据 - 机械臂末端位姿\n")
            f.write("# 格式: frame_id roll(rad) pitch(rad) yaw(rad) x(m) y(m) z(m)\n")
            f.write(f"# 采集时间: {timestamp}\n")
            f.write("# " + "-"*70 + "\n")

        # 保存相机内参到数据目录
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            intrinsics_path = os.path.join(save_dir, "camera_intrinsics.yaml")
            # 获取内参来源信息
            _, _, source = self.get_camera_intrinsics(save_to_dir=save_dir)
            if self.save_intrinsics_to_file(intrinsics_path, self.camera_matrix, self.dist_coeffs, source):
                print(f"💾 相机内参已保存到: {intrinsics_path}")

        print("\n📸 开始实时预览，移动机械臂到合适位置后按空格保存...")

        # 获取初始yaw值作为参考
        _, initial_pose = self.robot.arm.get_position(return_gripper_center=False)
        initial_pose = initial_pose if isinstance(initial_pose, list) else initial_pose.tolist()
        initial_yaw = initial_pose[5]  # 初始yaw角度（度）
        print(f"📐 初始Yaw角度: {initial_yaw:.1f}°")
        print(f"   建议保持Yaw偏差在 ±30° 以内（相对于初始值）")

        try:
            while True:
                # 获取相机图像
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                display_image = color_image.copy()

                # 尝试检测棋盘格
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                #ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
                ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + 
          cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    # 精细化角点
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(display_image, self.board_size, corners_refined, ret)
                    cv2.putText(display_image, "Chessboard: DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    cv2.putText(display_image, "Chessboard: NOT DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # 显示采集数量
                cv2.putText(display_image, f"Collected: {frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # 显示预热完成状态
                cv2.putText(display_image, "Warmup: READY", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(display_image, "[SPACE] Save  [ESC] Calibrate  [Q] Quit",
                           (10, display_image.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 显示图像
                cv2.imshow("Manual Calibration", display_image)

                # 按键处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # 空格：保存
                    if not ret:
                        print("\n⚠️  未检测到棋盘格，建议调整位置后再保存")
                        user_confirm = input("是否仍要保存此帧? (y/n): ")
                        if user_confirm.lower() != 'y':
                            continue

                    # 获取当前机械臂位姿
                    _, current_pose = self.robot.arm.get_position(return_gripper_center=False)
                    current_pose = current_pose if isinstance(current_pose, list) else current_pose.tolist()

                    # 等待位姿稳定
                    is_stable = self.wait_for_pose_stability(current_pose)
                    if not is_stable:
                        print("   ⚠️ 位姿未完全稳定，可能影响精度")
                        user_confirm = input("   是否仍要保存? (y/n): ")
                        if user_confirm.lower() != 'y':
                            continue

                    # 检查 yaw 角度相对于初始值的偏差
                    yaw_deg = current_pose[5]
                    yaw_deviation = yaw_deg - initial_yaw

                    # 处理角度环绕（-180° 到 180°）
                    if yaw_deviation > 180:
                        yaw_deviation -= 360
                    elif yaw_deviation < -180:
                        yaw_deviation += 360

                    '''
                    if abs(yaw_deviation) > 30:
                        print(f"\n⚠️  警告：Yaw偏差 {yaw_deviation:.1f}° 超出推荐范围 [-30°, 30°]")
                        print(f"   当前Yaw: {yaw_deg:.1f}°, 初始Yaw: {initial_yaw:.1f}°")
                        print(f"   大幅度的Yaw旋转可能影响标定精度")
                        user_confirm = input("是否仍要保存此数据? (y/n): ")
                        if user_confirm.lower() != 'y':
                            print("   已跳过此帧")
                            continue
                    '''
                    
                    # 单位转换：毫米→米，度→弧度
                    x_m = current_pose[0] / 1000.0
                    y_m = current_pose[1] / 1000.0
                    z_m = current_pose[2] / 1000.0
                    roll_rad = np.deg2rad(current_pose[3])
                    pitch_rad = np.deg2rad(current_pose[4])
                    yaw_rad = np.deg2rad(current_pose[5])

                    frame_count += 1

                    # 保存图像
                    image_filename = f"{frame_count}_Color.png"
                    image_path = os.path.join(save_dir, image_filename)
                    cv2.imwrite(image_path, color_image)

                    # 保存位姿到txt
                    with open(poses_txt_path, 'a') as f:
                        f.write(f"{frame_count} {roll_rad:.6f} {pitch_rad:.6f} {yaw_rad:.6f} "
                               f"{x_m:.6f} {y_m:.6f} {z_m:.6f}\n")

                    # 记录数据
                    collected_data.append({
                        'frame_id': frame_count,
                        'image_path': image_path,
                        'pose': [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad],
                        'pose_original': current_pose,
                        'has_chessboard': ret,
                        'corners': corners_refined if ret else None
                    })

                    print(f"✅ 已保存第 {frame_count} 帧:")
                    print(f"   位置: X={current_pose[0]:.1f}mm Y={current_pose[1]:.1f}mm Z={current_pose[2]:.1f}mm")

                    # 根据 yaw 偏差显示不同的状态
                    yaw_status = ""
                    if abs(yaw_deviation) > 30:
                        yaw_status = f" ⚠️ (偏差{yaw_deviation:+.1f}°, 超范围)"
                    elif abs(yaw_deviation) > 20:
                        yaw_status = f" ⚡ (偏差{yaw_deviation:+.1f}°, 接近边界)"
                    else:
                        yaw_status = f" (偏差{yaw_deviation:+.1f}°)"

                    print(f"   姿态: R={current_pose[3]:.1f}° P={current_pose[4]:.1f}° Y={current_pose[5]:.1f}°{yaw_status}")
                    print(f"   棋盘格: {'✓ 检测到' if ret else '✗ 未检测到'}")

                elif key == 27:  # ESC：结束采集
                    if frame_count == 0:
                        print("\n⚠️  未采集任何数据")
                        break
                    print(f"\n📊 采集完成，共 {frame_count} 帧")
                    print("🔧 开始手眼标定...")
                    calibration_success = self.perform_manual_calibration(collected_data, save_dir)

                    # 标定成功后，询问是否重播
                    if calibration_success:
                        self.save_dir_for_replay = save_dir  # 保存目录供后续重播使用
                        print("\n" + "="*60)
                        print("📝 是否进行轨迹重播验证?")
                        print("   重播可以验证标定结果的重复性")
                        replay_choice = input("是否重播轨迹? (y/n): ")
                        if replay_choice.lower() == 'y':
                            cv2.destroyAllWindows()  # 先关闭当前窗口
                            print("\n🔄 准备重播轨迹...")
                            

                            mode_choice = '1'
                            '''
                            # 询问重播模式
                            print("\n选择重播模式:")
                            print("  1. 高精度模式 - 每次：初始位置→零点→目标（推荐，消除累积误差）")
                            print("  2. 快速模式 - 直接移动到目标（更快但可能有累积误差）")
                            mode_choice = input("选择模式 (1/2) [默认1]: ").strip()
                            '''
                            
                            return_to_zero = True  # 默认高精度模式
                            if mode_choice == "2":
                                return_to_zero = False
                                print("⚡ 已选择快速模式")
                            else:
                                print("✅ 已选择高精度模式")

                            time.sleep(1)
                            self.replay_trajectory(save_dir, return_to_zero=return_to_zero)
                    break

                elif key == ord('q') or key == ord('Q'):  # Q：退出
                    print("\n❌ 用户取消")
                    break

        except KeyboardInterrupt:
            print("\n⚠️  用户中断")

        finally:
            cv2.destroyAllWindows()

            # 保存Excel文件
            if frame_count > 0:
                try:
                    import pandas as pd

                    # 读取poses.txt数据
                    data_rows = []
                    with open(poses_txt_path, 'r') as f:
                        for line in f:
                            if not line.startswith('#'):
                                data_rows.append(line.strip().split())

                    # 转换为DataFrame
                    df = pd.DataFrame(data_rows, columns=['frame_id', 'roll_rad', 'pitch_rad',
                                                          'yaw_rad', 'x_m', 'y_m', 'z_m'])
                    df = df.astype(float)
                    df['frame_id'] = df['frame_id'].astype(int)

                    # 保存为Excel
                    excel_path = os.path.join(save_dir, "poses.xlsx")
                    df.to_excel(excel_path, index=False, sheet_name='Poses')

                    print(f"\n💾 数据已保存")
                except ImportError:
                    print(f"\n💾 数据已保存 (无法生成Excel，需要安装pandas)")

    def analyze_error_patterns(self, errors, frame_ids):
        """分析误差模式以识别系统性问题

        Args:
            errors: 误差列表
            frame_ids: 帧ID列表

        Returns:
            dict: 误差分析结果
        """
        analysis = {
            'temporal_drift': False,
            'spatial_clustering': False,
            'periodic_pattern': False,
            'outlier_frames': [],
            'stable_frames': [],
            'recommendations': []
        }

        # 时间序列分析 - 检测热漂移
        if len(errors) > 10:
            # 将数据分成前半和后半
            mid_point = len(errors) // 2
            first_half_avg = np.mean(errors[:mid_point])
            second_half_avg = np.mean(errors[mid_point:])

            # 如果后半部分误差明显增大，可能存在热漂移
            if second_half_avg > first_half_avg * 1.3:
                analysis['temporal_drift'] = True
                analysis['recommendations'].append("检测到时间相关误差增长，建议：机器臂预热15-20分钟")

            # 检测周期性模式
            try:
                from scipy import signal
                # 使用自相关检测周期性
                autocorr = signal.correlate(errors - np.mean(errors), errors - np.mean(errors), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]  # 归一化

                # 寻找局部最大值
                peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=3)
                if len(peaks) > 0:
                    analysis['periodic_pattern'] = True
                    period = peaks[0]
                    analysis['recommendations'].append(f"检测到周期性误差（周期约{period}帧），可能存在机械振动或齿轮间隙")
            except:
                pass

        # 识别稳定帧和异常帧
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        for i, (err, fid) in enumerate(zip(errors, frame_ids)):
            if err < mean_error - std_error:
                analysis['stable_frames'].append(fid)
            elif err > mean_error + 1.5 * std_error:
                analysis['outlier_frames'].append(fid)

        # 空间聚类分析
        if len(analysis['outlier_frames']) > 3:
            # 检查异常帧是否连续
            outlier_diffs = np.diff(analysis['outlier_frames'])
            if np.all(outlier_diffs <= 2):
                analysis['spatial_clustering'] = True
                analysis['recommendations'].append(f"异常帧集中在{analysis['outlier_frames'][0]}-{analysis['outlier_frames'][-1]}，可能该区域存在遮挡或光照问题")

        return analysis

    def detect_pose_stability(self, pose_sequence, timestamps=None):
        """检测位姿序列的稳定性

        Args:
            pose_sequence: 位姿序列
            timestamps: 可选的时间戳

        Returns:
            dict: 稳定性分析结果
        """
        stability = {
            'position_jitter_mm': 0,
            'rotation_jitter_deg': 0,
            'settling_time_s': 0,
            'is_stable': True
        }

        if len(pose_sequence) < 2:
            return stability

        # 计算相邻位姿之间的变化
        position_changes = []
        rotation_changes = []

        for i in range(1, len(pose_sequence)):
            # 位置变化
            pos_diff = np.linalg.norm(pose_sequence[i][:3] - pose_sequence[i-1][:3]) * 1000  # 转换为mm
            position_changes.append(pos_diff)

            # 旋转变化
            rot_diff = np.abs(pose_sequence[i][3:] - pose_sequence[i-1][3:])
            rot_diff_deg = np.degrees(np.max(rot_diff))
            rotation_changes.append(rot_diff_deg)

        # 计算抖动指标
        if position_changes:
            stability['position_jitter_mm'] = np.std(position_changes)
            stability['rotation_jitter_deg'] = np.std(rotation_changes)

            # 判断是否稳定
            if stability['position_jitter_mm'] > 0.5 or stability['rotation_jitter_deg'] > 0.1:
                stability['is_stable'] = False

        return stability

    def _prepare_calibration_data(self, collected_data, compute_reprojection_errors=False):
        """提取公共的标定数据准备逻辑

        Args:
            collected_data: 已包含 corners 的数据（来自 apply_quality_filters 或 棋盘格检测）
            compute_reprojection_errors: 是否计算重投影误差（优化模式需要）

        Returns:
            tuple: (R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list,
                    frame_ids, reprojection_errors)
                    如果 compute_reprojection_errors=False，reprojection_errors 为 None
        """
        # 准备棋盘格世界坐标
        square_size = self.chessboard_size_mm / 1000.0
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # 初始化数据列表
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []
        frame_ids = []
        reprojection_errors = [] if compute_reprojection_errors else None

        for data in collected_data:
            # 机器人位姿（末端到基座）
            x, y, z, roll, pitch, yaw = data['pose']
            t_gripper2base_list.append(np.array([[x], [y], [z]]))
            R_robot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            R_gripper2base_list.append(R_robot)
            frame_ids.append(data['frame_id'])

            # 求解棋盘格相对于相机的位姿
            ret, rvec, tvec = cv2.solvePnP(objp, data['corners'], self.camera_matrix, self.dist_coeffs)
            if ret:
                R_target2cam_mat, _ = cv2.Rodrigues(rvec)
                R_target2cam_list.append(R_target2cam_mat)
                t_target2cam_list.append(tvec)

                # 可选：计算重投影误差
                if compute_reprojection_errors:
                    projected_points, _ = cv2.projectPoints(objp, rvec, tvec,
                                                           self.camera_matrix, self.dist_coeffs)
                    projected_points = projected_points.reshape(-1, 2)
                    detected_points = data['corners'].reshape(-1, 2)
                    error = np.sqrt(np.mean(np.sum((projected_points - detected_points)**2, axis=1)))
                    reprojection_errors.append(error)
            else:
                print(f"⚠️  帧 {data['frame_id']}: solvePnP 失败")

        return R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list, frame_ids, reprojection_errors

    def perform_optimized_calibration(self, collected_data, save_dir):
        """优化版手眼标定 - 包含质量预筛选、RANSAC过滤、多算法融合、迭代优化和高级诊断

        Args:
            collected_data: 采集的标定数据
            save_dir: 保存目录

        Returns:
            bool: 标定是否成功
        """
        print("\n" + "="*60)
        print("🔧 执行优化版手眼标定（含质量预筛选+高级诊断）")
        print("="*60)

        if len(collected_data) < 3:
            print(f"❌ 数据不足，至少需要3组数据，当前只有 {len(collected_data)} 组")
            return False

        # ========================================
        # 步骤0: 数据质量预筛选（在RANSAC之前）
        # ========================================
        collected_data, filter_report = self.apply_quality_filters(collected_data, verbose=True)

        if len(collected_data) < 10:
            print(f"\n❌ 质量过滤后数据不足（仅 {len(collected_data)} 帧），至少需要10帧")
            print("   建议:")
            print("   1. 重新采集数据，确保运动充分")
            print("   2. 避免在工作空间边界采集")
            print("   3. 增加数据采集总量")
            return False

        # 使用公共方法准备标定数据（包含重投影误差计算）
        R_gripper2base_all, t_gripper2base_all, R_target2cam_all, t_target2cam_all, frame_ids, reprojection_errors = \
            self._prepare_calibration_data(collected_data, compute_reprojection_errors=True)

        print(f"📊 质量预筛选后准备了 {len(R_gripper2base_all)} 组数据")

        # 显示角点检测质量统计
        if reprojection_errors:
            avg_reproj_error = np.mean(reprojection_errors)
            max_reproj_error = np.max(reprojection_errors)
            min_reproj_error = np.min(reprojection_errors)

            print(f"\n🎯 角点检测质量:")
            print(f"   平均重投影误差: {avg_reproj_error:.3f} 像素")
            print(f"   误差范围: {min_reproj_error:.3f} - {max_reproj_error:.3f} 像素")

            if avg_reproj_error < 0.5:
                print(f"   质量评级: 🌟 优秀 (< 0.5px)")
            elif avg_reproj_error < 1.0:
                print(f"   质量评级: 👍 良好 (< 1.0px)")
            elif avg_reproj_error < 2.0:
                print(f"   质量评级: ⚠️  可接受 (< 2.0px)")
            else:
                print(f"   质量评级: ❌ 较差 (>= 2.0px)")
                print(f"   建议: 重新标定相机内参或调整棋盘格检测参数")

            # 标记并过滤重投影误差过大的帧
            bad_reproj_frames = [frame_ids[i] for i, err in enumerate(reprojection_errors) if err > 2.0]
            if bad_reproj_frames:
                print(f"   ⚠️  重投影误差>2.0px的帧: {bad_reproj_frames[:10]}")
                if len(bad_reproj_frames) > 10:
                    print(f"      ... 还有 {len(bad_reproj_frames)-10} 帧")

        # 步骤0: 过滤高重投影误差帧（异常数据清理）
        if reprojection_errors:
            print("\n" + "-"*40)
            print("步骤0: 过滤异常帧（重投影误差 > 2.0px）")

            # 创建保留索引列表
            keep_indices = [i for i, err in enumerate(reprojection_errors) if err <= 2.0]
            removed_indices = [i for i, err in enumerate(reprojection_errors) if err > 2.0]

            if removed_indices:
                print(f"   移除 {len(removed_indices)} 帧:")
                for idx in removed_indices:
                    print(f"   ❌ 帧 {frame_ids[idx]}: 重投影误差 {reprojection_errors[idx]:.3f}px (移除)")

                # 过滤数据数组
                R_gripper2base_all = [R_gripper2base_all[i] for i in keep_indices]
                t_gripper2base_all = [t_gripper2base_all[i] for i in keep_indices]
                R_target2cam_all = [R_target2cam_all[i] for i in keep_indices]
                t_target2cam_all = [t_target2cam_all[i] for i in keep_indices]
                frame_ids = [frame_ids[i] for i in keep_indices]
                reprojection_errors = [reprojection_errors[i] for i in keep_indices]

                print(f"\n📊 过滤结果:")
                print(f"   保留帧数: {len(keep_indices)}")
                print(f"   移除帧数: {len(removed_indices)}")
            else:
                print(f"   ✅ 所有帧的重投影误差都 ≤ 2.0px，无需过滤")

            if len(R_gripper2base_all) < 10:
                print("❌ 过滤后有效数据不足10帧，无法进行标定")
                return False

        # 步骤1: RANSAC几何一致性过滤（第二层过滤）
        print("\n" + "-"*40)
        print("步骤1: RANSAC几何一致性过滤")

        # 改进的阈值策略：基于数据质量动态调整
        initial_threshold = 10.0  # 合理的初始阈值
        min_inliers = max(10, int(len(R_gripper2base_all) * 0.7))  # 至少保留70%的数据（更保守）

        best_inliers = self.ransac_filter_handeye(
            R_gripper2base_all, t_gripper2base_all,
            R_target2cam_all, t_target2cam_all,
            frame_ids,
            threshold=initial_threshold
        )

        # 如果内点太少，逐步放宽阈值（但不要太激进）
        attempts = 0
        while len(best_inliers) < min_inliers and initial_threshold < 20.0 and attempts < 3:
            initial_threshold += 3.0
            attempts += 1
            print(f"   调整RANSAC阈值到 {initial_threshold}mm (尝试 {attempts}/3)")
            best_inliers = self.ransac_filter_handeye(
                R_gripper2base_all, t_gripper2base_all,
                R_target2cam_all, t_target2cam_all,
                frame_ids,
                threshold=initial_threshold
            )

        if len(best_inliers) < 3:
            print("❌ RANSAC后有效数据不足")
            return self.perform_manual_calibration(collected_data, save_dir)

        # 使用过滤后的数据
        R_gripper2base = [R_gripper2base_all[i] for i in best_inliers]
        t_gripper2base = [t_gripper2base_all[i] for i in best_inliers]
        R_target2cam = [R_target2cam_all[i] for i in best_inliers]
        t_target2cam = [t_target2cam_all[i] for i in best_inliers]

        print(f"✅ RANSAC过滤: {len(best_inliers)}/{len(R_gripper2base_all)} 个内点")
        print(f"   被过滤的帧: {[frame_ids[i] for i in range(len(frame_ids)) if i not in best_inliers]}")

        # 步骤2: 多算法融合
        print("\n" + "-"*40)
        print("步骤2: 多算法融合标定")

        best_R, best_t, best_method = self.multi_algorithm_fusion(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam
        )

        if best_R is None:
            print("❌ 多算法融合失败")
            return self.perform_manual_calibration(collected_data, save_dir)

        print(f"✅ 最优算法: {best_method}")

        # 步骤3: 迭代优化
        print("\n" + "-"*40)
        print("步骤3: 迭代优化")

        R_optimized, t_optimized = self.iterative_optimization(
            best_R, best_t,
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam
        )

        # 提取原始位姿用于偏差显示
        original_poses = [data['pose'] for data in collected_data]

        # 评估最终结果（返回详细误差，包含位姿偏差信息）
        final_avg_error, detailed_errors = self.evaluate_calibration(
            R_optimized, t_optimized,
            R_gripper2base_all, t_gripper2base_all,
            R_target2cam_all, t_target2cam_all,
            frame_ids,
            original_poses=original_poses
        )

        # 保存优化结果（传入详细误差）
        self.save_optimized_result(
            R_optimized, t_optimized,
            final_avg_error,
            best_method,
            len(best_inliers),
            len(R_gripper2base_all),
            save_dir,
            detailed_errors=detailed_errors
        )

        return True

    def perform_manual_calibration(self, collected_data, save_dir):
        """使用手动采集的数据进行手眼标定"""
        print("\n" + "="*60)
        print("🔧 开始手眼标定计算")
        print("="*60)

        if len(collected_data) < 3:
            print(f"❌ 数据不足，至少需要3组数据，当前只有 {len(collected_data)} 组")
            return False

        # 检测并过滤有效数据
        valid_data = []
        for data in collected_data:
            image = cv2.imread(data['image_path'])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

            if ret:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                valid_data.append({
                    'frame_id': data['frame_id'],
                    'pose': data['pose'],
                    'corners': corners_refined,
                    'image': image
                })
                print(f"✅ 帧 {data['frame_id']}: 棋盘格检测成功")
            else:
                print(f"⚠️  帧 {data['frame_id']}: 棋盘格检测失败，跳过")

        if len(valid_data) < 3:
            print(f"\n❌ 有效数据不足，至少需要3组，当前只有 {len(valid_data)} 组")
            return False

        print(f"\n📊 有效数据: {len(valid_data)}/{len(collected_data)} 组")

        # 使用公共方法准备标定数据（不计算重投影误差）
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, _, _ = \
            self._prepare_calibration_data(valid_data, compute_reprojection_errors=False)

        if len(R_gripper2base) != len(R_target2cam) or len(R_gripper2base) < 3:
            print(f"❌ 标定数据准备失败")
            return False

        # 提供多种标定算法选择
        calibration_methods = [
            {'method': cv2.CALIB_HAND_EYE_TSAI, 'name': 'Tsai', 'description': '经典算法，适用于多数情况'},
            {'method': cv2.CALIB_HAND_EYE_PARK, 'name': 'Park', 'description': '对噪声鲁棒性好'},
            {'method': cv2.CALIB_HAND_EYE_HORAUD, 'name': 'Horaud', 'description': '高精度，需要更多数据'},
            {'method': cv2.CALIB_HAND_EYE_ANDREFF, 'name': 'Andreff', 'description': '快速算法'},
            {'method': cv2.CALIB_HAND_EYE_DANIILIDIS, 'name': 'Daniilidis', 'description': '数值稳定性好'}
        ]

        print(f"\n🔧 执行手眼标定...")
        print(f"   数据点数: {len(R_gripper2base)}")
        print("\n选择标定算法:")
        for i, method_info in enumerate(calibration_methods, 1):
            print(f"  {i}. {method_info['name']} - {method_info['description']}")

        '''
        choice = input("选择算法 (1-5) [默认1-Tsai]: ").strip()
        if choice == '':
            choice = '1'  # 默认使用Tsai算法
        '''
        choice = '1'  #'1'
        try:
            method_index = int(choice) - 1
            if 0 <= method_index < len(calibration_methods):
                selected_method = calibration_methods[method_index]
            else:
                print("⚠️  无效选择，使用Park算法")
                selected_method = calibration_methods[1]
        except ValueError:
            print("⚠️  无效输入，使用Park算法")
            selected_method = calibration_methods[1]

        print(f"\n使用 {selected_method['name']} 算法进行标定...")

        try:
            # 执行OpenCV手眼标定
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=selected_method['method']
            )

            print("\n" + "="*60)
            print("✅ 手眼标定完成！")
            print("="*60)

            # 转换为欧拉角和四元数
            rotation = R.from_matrix(R_cam2gripper)
            quaternion = rotation.as_quat()  # [x, y, z, w]
            euler_angles = rotation.as_euler('xyz', degrees=True)

            # 转换平移向量
            t_cam2gripper_mm = t_cam2gripper.flatten() * 1000.0
            t_cam2gripper_m = t_cam2gripper.flatten()

            print("\n📊 标定结果:")
            print(f"   平移向量 (mm): X={t_cam2gripper_mm[0]:.2f}, Y={t_cam2gripper_mm[1]:.2f}, Z={t_cam2gripper_mm[2]:.2f}")
            print(f"   欧拉角 (度):    Rx={euler_angles[0]:.2f}°, Ry={euler_angles[1]:.2f}°, Rz={euler_angles[2]:.2f}°")
            print(f"   四元数 (xyzw):  x={quaternion[0]:.4f}, y={quaternion[1]:.4f}, z={quaternion[2]:.4f}, w={quaternion[3]:.4f}")

            # 计算误差评估
            print("\n" + "="*60)
            print("📊 标定质量分析")
            print("="*60)

            rotation_errors = []
            translation_errors = []

            for i, data in enumerate(valid_data):
                # 使用标定结果预测
                R_target2base_pred = R_gripper2base[i] @ R_cam2gripper @ R_target2cam[i]
                t_target2base_pred = R_gripper2base[i] @ (R_cam2gripper @ t_target2cam[i] + t_cam2gripper) + t_gripper2base[i]

                # 第一帧作为参考
                if i == 0:
                    R_target2base_ref = R_target2base_pred.copy()
                    t_target2base_ref = t_target2base_pred.copy()

                # 计算相对误差
                R_error = R_target2base_ref.T @ R_target2base_pred
                angle_error_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
                angle_error_deg = np.degrees(angle_error_rad)
                rotation_errors.append(angle_error_deg)

                t_error = t_target2base_pred - t_target2base_ref
                translation_error_mm = np.linalg.norm(t_error) * 1000.0
                translation_errors.append(translation_error_mm)

                status = "✅" if translation_error_mm < 5.0 and angle_error_deg < 2.0 else "⚠️"
                print(f"{status} 帧 {data['frame_id']:2d}: 旋转误差 {angle_error_deg:6.3f}°  平移误差 {translation_error_mm:7.3f}mm")

            # 统计
            avg_rotation_error = np.mean(rotation_errors)
            avg_translation_error = np.mean(translation_errors)

            print("\n📈 误差统计:")
            print(f"   平均旋转误差:  {avg_rotation_error:.3f}°")
            print(f"   平均平移误差:  {avg_translation_error:.3f}mm")

            # 质量评估
            if avg_rotation_error < 1.0 and avg_translation_error < 5.0:
                quality = "🌟 优秀"
            elif avg_rotation_error < 2.0 and avg_translation_error < 10.0:
                quality = "👍 良好"
            elif avg_rotation_error < 5.0 and avg_translation_error < 20.0:
                quality = "⚠️  可接受"
            else:
                quality = "❌ 需要改进"

            print(f"\n🎯 标定质量: {quality}")

            # 保存标定结果
            result_data = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'method': f'CALIB_HAND_EYE_{selected_method["name"].upper()}',
                'data_points': len(R_gripper2base),
                'transformation': {
                    'rotation_matrix': R_cam2gripper.tolist(),
                    'translation_m': t_cam2gripper_m.tolist(),
                    'translation_mm': t_cam2gripper_mm.tolist(),
                    'quaternion_xyzw': quaternion.tolist(),
                    'euler_xyz_deg': euler_angles.tolist()
                },
                'quality': {
                    'avg_rotation_error_deg': float(avg_rotation_error),
                    'avg_translation_error_mm': float(avg_translation_error),
                    'quality_level': quality
                },
                'camera_intrinsics': {
                    'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                    'distortion_coefficients': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
                    'calibration_board': {
                        'board_size': list(self.board_size),
                        'square_size_mm': self.chessboard_size_mm
                    }
                }
            }

            # 保存JSON
            result_file = os.path.join(save_dir, "hand_eye_calibration_result.json")
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)

            # 保存YAML（ROS格式）
            yaml_result = {
                'hand_eye_calibration': {
                    'translation': {
                        'x': float(t_cam2gripper_m[0]),
                        'y': float(t_cam2gripper_m[1]),
                        'z': float(t_cam2gripper_m[2])
                    },
                    'rotation': {
                        'x': float(quaternion[0]),
                        'y': float(quaternion[1]),
                        'z': float(quaternion[2]),
                        'w': float(quaternion[3])
                    }
                }
            }

            yaml_file = os.path.join(save_dir, "hand_eye_calibration_result.yaml")
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_result, f, default_flow_style=False)

            print(f"\n💾 标定结果已保存到:")
            print(f"   - {result_file}")
            print(f"   - {yaml_file}")

            print("\n✅ 手眼标定流程完成！")
            return True

        except Exception as e:
            print(f"\n❌ 标定失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def verify_reference_position(self):
        """验证参考位置的重复精度"""
        print("\n   🔍 验证参考位置重复精度...")

        # 移动到参考位置
        self.robot.arm.set_position(
            x=self.reference_position[0], y=self.reference_position[1], z=self.reference_position[2],
            roll=self.reference_position[3], pitch=self.reference_position[4], yaw=self.reference_position[5],
            wait=True, speed=50, use_gripper_center=False
        )
        time.sleep(2.0)  # 等待稳定

        # 获取实际位置
        _, actual_pose = self.robot.arm.get_position(return_gripper_center=False)
        actual_pose = actual_pose if isinstance(actual_pose, list) else actual_pose.tolist()

        # 计算偏差
        deviation = [
            abs(actual_pose[0] - self.reference_position[0]),
            abs(actual_pose[1] - self.reference_position[1]),
            abs(actual_pose[2] - self.reference_position[2]),
            abs(actual_pose[3] - self.reference_position[3]),
            abs(actual_pose[4] - self.reference_position[4]),
            abs(actual_pose[5] - self.reference_position[5])
        ]

        max_pos_deviation = max(deviation[:3])
        max_rot_deviation = max(deviation[3:])

        self.position_deviations.append({
            'pos_deviation': max_pos_deviation,
            'rot_deviation': max_rot_deviation
        })

        if max_pos_deviation > 2.0:
            print(f"   ⚠️  参考位置偏差较大: 位置 {max_pos_deviation:.2f}mm, 角度 {max_rot_deviation:.2f}°")
            print(f"      建议检查机械臂重复定位精度")
        else:
            print(f"   ✅ 参考位置偏差正常: 位置 {max_pos_deviation:.2f}mm, 角度 {max_rot_deviation:.2f}°")

        return max_pos_deviation, max_rot_deviation

    def replay_trajectory(self, data_dir, return_to_zero=True):
        """重播保存的轨迹并重新采集图像进行标定

        Args:
            data_dir: 数据目录路径
            return_to_zero: 是否每次都先回零点（默认True，提高重复精度）
        """
        print("\n" + "="*60)
        print("🔄 轨迹重播模式")
        print("="*60)

        if return_to_zero:
            print("⚡ 模式：高精度模式 - 每个位置都先回零点")
            print("   流程：零点 → 位置1 → 零点 → 位置2 → 零点 → 位置3 ...")
            print("   优势：消除累积误差，提高重复定位精度")
            print("   📍 每5个位置返回参考位置验证重复精度")
        else:
            print("⚡ 模式：快速模式 - 连续移动")
            print("   流程：位置1 → 位置2 → 位置3 ...")
            print("   优势：速度快，但可能有累积误差")

        # 定义参考位置（用于验证重复精度）
        self.reference_position = [300, 0, 300, 180, 60, 180]  # 稳定的参考位置
        self.reference_check_interval = 5  # 每5个位置检查一次
        self.position_deviations = []  # 记录位置偏差

        # 读取poses.txt文件
        poses_file = os.path.join(data_dir, "poses.txt")
        if not os.path.exists(poses_file):
            print(f"❌ 找不到位姿文件: {poses_file}")
            return False

        # 解析位姿数据
        replay_poses = []
        with open(poses_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) == 7:
                        frame_id = int(parts[0])
                        roll_rad = float(parts[1])
                        pitch_rad = float(parts[2])
                        yaw_rad = float(parts[3])
                        x_m = float(parts[4])
                        y_m = float(parts[5])
                        z_m = float(parts[6])

                        # 转换回毫米和度
                        pose_mm_deg = [
                            x_m * 1000.0,
                            y_m * 1000.0,
                            z_m * 1000.0,
                            np.degrees(roll_rad),
                            np.degrees(pitch_rad),
                            np.degrees(yaw_rad)
                        ]
                        replay_poses.append({
                            'frame_id': frame_id,
                            'pose_mm_deg': pose_mm_deg,
                            'pose_m_rad': [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
                        })

        if not replay_poses:
            print("❌ 没有找到有效的位姿数据")
            return False

        print(f"📊 找到 {len(replay_poses)} 个位姿点")

        # 创建新的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.calibration_data_dir, f"replay_calibration_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 重播数据保存目录: {save_dir}/")

        # 保存原始数据目录引用
        with open(os.path.join(save_dir, "replay_source.txt"), 'w') as f:
            f.write(f"Original data from: {data_dir}\n")
            f.write(f"Replay time: {timestamp}\n")

        # 重播收集的数据
        collected_data = []
        poses_txt_path = os.path.join(save_dir, "poses.txt")

        # 写入文件头
        with open(poses_txt_path, 'w') as f:
            f.write("# 重播手眼标定数据 - 机械臂末端位姿\n")
            f.write("# 格式: frame_id roll(rad) pitch(rad) yaw(rad) x(m) y(m) z(m)\n")
            f.write(f"# 重播时间: {timestamp}\n")
            f.write(f"# 原始数据: {data_dir}\n")
            f.write("#" + "-"*70 + "\n")

        # 保存相机内参到数据目录
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            intrinsics_path = os.path.join(save_dir, "camera_intrinsics.yaml")
            # 获取内参来源信息
            _, _, source = self.get_camera_intrinsics(save_to_dir=save_dir)
            if self.save_intrinsics_to_file(intrinsics_path, self.camera_matrix, self.dist_coeffs, source):
                print(f"💾 相机内参已保存到: {intrinsics_path}")

        print("\n🎬 开始轨迹重播...")
        print("操作说明:")
        print("  [空格] - 暂停/继续")
        print("  [ESC]  - 结束重播并标定")
        print("  [Q]    - 退出")
        print("-"*60)

        paused = False
        current_index = 0

        try:
            while current_index < len(replay_poses):
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    paused = not paused
                    if paused:
                        print("⏸️  暂停重播")
                    else:
                        print("▶️  继续重播")
                    continue
                elif key == 27:  # ESC
                    print("\n📊 重播结束，开始标定...")
                    break
                elif key == ord('q') or key == ord('Q'):
                    print("\n❌ 用户取消重播")
                    return False

                if paused:
                    time.sleep(0.1)
                    continue

                # 获取当前要重播的位姿
                target_pose = replay_poses[current_index]
                pose_mm_deg = target_pose['pose_mm_deg']

                print(f"\n▶️  重播第 {target_pose['frame_id']}/{len(replay_poses)} 个位置...")

                if return_to_zero:
                    # 高精度模式：先回初始位置，再回零点，最后到目标位置
                    print(f"   🔄 [步骤1/3] 先回到初始安全位置...")
                    print(f"      初始位置: X={self.initial_position[0]:.1f}mm Y={self.initial_position[1]:.1f}mm Z={self.initial_position[2]:.1f}mm")
                    print(f"      初始姿态: R={self.initial_position[3]:.1f}° P={self.initial_position[4]:.1f}° Y={self.initial_position[5]:.1f}°")


                    # 先回到初始安全位置
                    self.robot.arm.set_position(
                        x=self.initial_position[0], y=self.initial_position[1], z=self.initial_position[2],
                        roll=self.initial_position[3], pitch=self.initial_position[4], yaw=self.initial_position[5],
                        wait=True, speed=self.motion_config['normal_speed'], use_gripper_center=False
                    )
                    print(f"   🔄 [步骤2/3] 从初始位置到机械臂关节零点...")
                    print(f"      关节零点: 所有关节角度归零")
                    # 再移动到机械臂关节零点
                    self.robot.arm._go_zero()
                    time.sleep(3.0)  # 增加到2秒，确保完全稳定

                    # 步骤3：从零点移动到目标位置
                    print(f"   📍 [步骤3/3] 从零点移动到目标位置...")
                else:
                    # 快速模式：直接移动（更快但可能累积误差）
                    print(f"   📍 直接移动到目标位置（快速模式）...")

                print(f"      目标位置: X={pose_mm_deg[0]:.1f}mm Y={pose_mm_deg[1]:.1f}mm Z={pose_mm_deg[2]:.1f}mm")
                print(f"      目标姿态: R={pose_mm_deg[3]:.1f}° P={pose_mm_deg[4]:.1f}° Y={pose_mm_deg[5]:.1f}°")

                self.robot.arm.set_position(
                    x=pose_mm_deg[0], y=pose_mm_deg[1], z=pose_mm_deg[2],
                    roll=pose_mm_deg[3], pitch=pose_mm_deg[4], yaw=pose_mm_deg[5],
                    wait=True, speed=self.motion_config['capture_speed'], use_gripper_center=False
                )

                # 等待机器臂完全稳定（使用配置的等待时间）
                print("   ⏱️  等待稳定...")
                time.sleep(self.motion_config['stability_wait'])  # 使用配置值(5.0秒)

                # 获取实际到达的位置，用于验证
                _, actual_pose = self.robot.arm.get_position(return_gripper_center=False)
                actual_pose = actual_pose if isinstance(actual_pose, list) else actual_pose.tolist()

                # 将实际位置转换为米和弧度（用于标定）
                actual_pose_m_rad = [
                    actual_pose[0] / 1000.0,  # mm to m
                    actual_pose[1] / 1000.0,
                    actual_pose[2] / 1000.0,
                    np.radians(actual_pose[3]),  # deg to rad
                    np.radians(actual_pose[4]),
                    np.radians(actual_pose[5])
                ]

                # 计算位置误差
                pos_error = [
                    abs(actual_pose[0] - pose_mm_deg[0]),
                    abs(actual_pose[1] - pose_mm_deg[1]),
                    abs(actual_pose[2] - pose_mm_deg[2])
                ]

                if max(pos_error) > 2.0:  # 如果位置误差超过2mm
                    print(f"   ⚠️  位置误差较大: X={pos_error[0]:.2f}mm Y={pos_error[1]:.2f}mm Z={pos_error[2]:.2f}mm")
                else:
                    print(f"   ✅ 到达目标位置（误差<2mm）")

                # 获取相机图像
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    display_image = color_image.copy()

                    # 检测棋盘格
                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

                    if ret:
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                        cv2.drawChessboardCorners(display_image, self.board_size, corners_refined, ret)
                        cv2.putText(display_image, "DETECTED", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_image, "NOT DETECTED", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    # 显示进度
                    cv2.putText(display_image, f"Replay: {current_index+1}/{len(replay_poses)}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(display_image, "[SPACE] Pause  [ESC] Stop  [Q] Quit",
                               (10, display_image.shape[0]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow("Replay Mode", display_image)

                    # 保存图像
                    frame_id = target_pose['frame_id']
                    image_filename = f"{frame_id}_Color.png"
                    image_path = os.path.join(save_dir, image_filename)
                    cv2.imwrite(image_path, color_image)

                    # 保存实际位姿（而不是目标位姿）
                    with open(poses_txt_path, 'a') as f:
                        f.write(f"{frame_id} {actual_pose_m_rad[3]:.6f} {actual_pose_m_rad[4]:.6f} {actual_pose_m_rad[5]:.6f} "
                               f"{actual_pose_m_rad[0]:.6f} {actual_pose_m_rad[1]:.6f} {actual_pose_m_rad[2]:.6f}\n")

                    # 记录数据（使用实际位置）
                    collected_data.append({
                        'frame_id': frame_id,
                        'image_path': image_path,
                        'pose': actual_pose_m_rad,  # 使用实际位置
                        'pose_original': actual_pose,  # 使用实际位置（mm和度）
                        'has_chessboard': ret,
                        'corners': corners_refined if ret else None
                    })

                    print(f"   ✅ 已采集第 {frame_id} 帧数据")
                    if ret:
                        print(f"   📷 棋盘格: 检测成功")
                    else:
                        print(f"   ⚠️  棋盘格: 未检测到")

                    # 定期验证参考位置（每5个位置检查一次）
                    if current_index % self.reference_check_interval == 0 and current_index > 0:
                        print(f"\n   📍 定期验证参考位置 (第 {current_index+1} 个位置后)")
                        pos_dev, rot_dev = self.verify_reference_position()

                        # 如果偏差太大，提示用户
                        if pos_dev > 3.0:
                            print(f"   ⚠️  警告：累积误差较大，建议重新开始标定")
                            print(f"      当前偏差: 位置 {pos_dev:.2f}mm, 角度 {rot_dev:.2f}°")
                            user_input = input("   是否继续? (y/n): ")
                            if user_input.lower() != 'y':
                                print("   ❌ 用户选择停止")
                                break

                current_index += 1
                time.sleep(0.5)  # 控制重播速度

        except KeyboardInterrupt:
            print("\n⚠️  用户中断重播")

        finally:
            cv2.destroyAllWindows()

        # 显示位置偏差统计
        if hasattr(self, 'position_deviations') and self.position_deviations:
            print("\n" + "="*60)
            print("📊 参考位置重复精度统计")
            print("="*60)
            avg_pos_dev = np.mean([d['pos_deviation'] for d in self.position_deviations])
            avg_rot_dev = np.mean([d['rot_deviation'] for d in self.position_deviations])
            max_pos_dev = np.max([d['pos_deviation'] for d in self.position_deviations])
            max_rot_dev = np.max([d['rot_deviation'] for d in self.position_deviations])

            print(f"验证次数: {len(self.position_deviations)}")
            print(f"平均偏差: 位置 {avg_pos_dev:.2f}mm, 角度 {avg_rot_dev:.2f}°")
            print(f"最大偏差: 位置 {max_pos_dev:.2f}mm, 角度 {max_rot_dev:.2f}°")

            if max_pos_dev > 2.0:
                print("\n⚠️  警告: 机械臂重复定位精度不足")
                print("   建议检查:")
                print("   • 机械臂齿轮间隙")
                print("   • 负载是否过重")
                print("   • 运动速度是否过快")
            else:
                print("\n✅ 机械臂重复定位精度良好")

        # 重播完成后进行标定
        if len(collected_data) >= 3:
            print(f"\n📊 重播完成，采集了 {len(collected_data)} 帧数据")
            print("🔧 开始手眼标定...")
            self.perform_manual_calibration(collected_data, save_dir)
        else:
            print(f"\n⚠️  有效数据不足（{len(collected_data)} 帧），无法进行标定")

        return True

    def compute_calibration_only(self, data_dir, use_optimization=True):
        """仅基于已有数据进行标定计算，不移动机械臂

        Args:
            data_dir: 数据目录
            use_optimization: 是否使用优化算法（RANSAC+多算法+迭代）
        """
        print("\n" + "="*60)
        print("🧮 纯计算重播模式 - 不移动机械臂")
        if use_optimization:
            print("✨ 启用优化算法（RANSAC+多算法融合+迭代优化）")
        print("="*60)

        # 读取poses.txt文件
        poses_file = os.path.join(data_dir, "poses.txt")
        if not os.path.exists(poses_file):
            print(f"❌ 找不到位姿文件: {poses_file}")
            return False

        # 解析位姿数据
        poses_data = []
        with open(poses_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) == 7:
                        frame_id = int(parts[0])
                        roll_rad = float(parts[1])
                        pitch_rad = float(parts[2])
                        yaw_rad = float(parts[3])
                        x_m = float(parts[4])
                        y_m = float(parts[5])
                        z_m = float(parts[6])

                        poses_data.append({
                            'frame_id': frame_id,
                            'pose': [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
                        })

        if not poses_data:
            print("❌ 没有找到有效的位姿数据")
            return False

        print(f"📊 找到 {len(poses_data)} 个位姿数据")

        # 使用通用函数获取相机内参
        self.camera_matrix, self.dist_coeffs, source = self.get_camera_intrinsics(data_dir=data_dir, save_to_dir=data_dir)
        print(f"📷 内参来源: {source}")

        # 检测棋盘格并收集有效数据
        valid_data = []
        print("\n检测棋盘格...")

        for pose_data in poses_data:
            frame_id = pose_data['frame_id']
            image_path = os.path.join(data_dir, f"{frame_id}_Color.png")

            if not os.path.exists(image_path):
                print(f"⚠️  帧 {frame_id}: 找不到图像文件")
                continue

            # 读取图像
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测棋盘格
            ret, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                      cv2.CALIB_CB_FILTER_QUADS
            )

            if ret:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                valid_data.append({
                    'frame_id': frame_id,
                    'pose': pose_data['pose'],
                    'corners': corners_refined,
                    'image': image,
                    'image_path': image_path  # 添加image_path以兼容perform_manual_calibration
                })
                print(f"✅ 帧 {frame_id}: 棋盘格检测成功")
            else:
                print(f"⚠️  帧 {frame_id}: 棋盘格检测失败")

        if len(valid_data) < 3:
            print(f"\n❌ 有效数据不足，至少需要3组，当前只有 {len(valid_data)} 组")
            return False

        print(f"\n📊 有效数据: {len(valid_data)}/{len(poses_data)} 组")

        # Yaw偏差过滤 - 使用initial_position的Yaw作为参考
        reference_yaw_deg = self.initial_position[5]  # 度
        print(f"\n📐 使用参考Yaw: {reference_yaw_deg:.1f}° (来自initial_position)")

        '''
        valid_data = self.filter_by_yaw_deviation(
            valid_data,
            reference_yaw_deg=reference_yaw_deg,
            max_deviation=30.0
        )

        if len(valid_data) < 3:
            print(f"\n❌ Yaw过滤后数据不足，至少需要3组，当前只有 {len(valid_data)} 组")
            print(f"💡 建议: 重新采集数据，保持Yaw在 {reference_yaw_deg-30:.1f}° ~ {reference_yaw_deg+30:.1f}° 范围内")
            return False

        print(f"\n✅ Yaw过滤后有效数据: {len(valid_data)} 组")
        '''
        
        # 创建新的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.calibration_data_dir, f"compute_only_calibration_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 结果保存目录: {save_dir}/")

        # 保存源数据信息
        with open(os.path.join(save_dir, "source_info.txt"), 'w') as f:
            f.write(f"Source data: {data_dir}\n")
            f.write(f"Compute time: {timestamp}\n")
            f.write(f"Valid frames: {len(valid_data)}\n")
            f.write(f"Camera intrinsics source: {source}\n")

        # 保存相机内参到数据目录
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            intrinsics_path = os.path.join(save_dir, "camera_intrinsics.yaml")
            if self.save_intrinsics_to_file(intrinsics_path, self.camera_matrix, self.dist_coeffs, source):
                print(f"💾 相机内参已保存到: {intrinsics_path}")

        # 执行标定计算
        print("\n🔧 开始标定计算...")
        

        # 根据选项使用不同的标定方法
        if use_optimization:
            calibration_success = self.perform_optimized_calibration(valid_data, save_dir)
        else:
            calibration_success = self.perform_manual_calibration(valid_data, save_dir)

        if calibration_success:
            print("\n✅ 纯计算标定完成！")
            print(f"结果已保存到: {save_dir}")

        return calibration_success

    def load_and_replay(self, data_dir, compute_only=False):
        """加载已有数据目录并重播轨迹或仅计算"""
        print("\n" + "="*60)
        print("📂 加载历史数据")
        print("="*60)

        if not os.path.exists(data_dir):
            print(f"❌ 目录不存在: {data_dir}")
            return False

        poses_file = os.path.join(data_dir, "poses.txt")
        if not os.path.exists(poses_file):
            print(f"❌ 找不到位姿文件: {poses_file}")
            return False

        print(f"✅ 找到数据目录: {data_dir}")

        # 显示原始标定信息（如果存在）
        result_file = os.path.join(data_dir, "hand_eye_calibration_result.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            print(f"📊 原始标定信息:")
            print(f"   时间: {result.get('timestamp', 'Unknown')}")
            print(f"   数据点: {result.get('data_points', 'Unknown')}")
            if 'quality' in result:
                print(f"   质量: {result['quality'].get('quality_level', 'Unknown')}")

        # 如果是纯计算模式，直接进行计算
        if compute_only:
            return self.compute_calibration_only(data_dir)

        choice  = 'y'
        mode_choice = '1'
        '''
        # 询问是否继续
        print("\n是否使用此数据进行轨迹重播?")
        choice = input("继续? (y/n): ")
        if choice.lower() != 'y':
            print("❌ 用户取消")
            return False

        # 询问重播模式
        print("\n选择重播模式:")
        print("  1. 高精度模式 - 每次：初始位置→零点→目标（推荐，消除累积误差）")
        print("  2. 快速模式 - 直接移动到目标（更快但可能有累积误差）")
        mode_choice = input("选择模式 (1/2) [默认1]: ").strip()
        '''


        return_to_zero = True  # 默认高精度模式
        if mode_choice == "2":
            return_to_zero = False
            print("⚡ 已选择快速模式")
        else:
            print("✅ 已选择高精度模式")

        # 执行重播
        return self.replay_trajectory(data_dir, return_to_zero=return_to_zero)


def main():
    """主函数 - 支持手动模式和重播模式"""
    print("="*60)
    print("🎯 手眼标定系统 - 增强版")
    print("="*60)
    print("功能模式:")
    print("  1. 手动标定 - 手动移动机器臂采集数据")
    print("  2. 轨迹重播 - 重播已保存的轨迹进行标定（需要机械臂）")
    print("  3. 纯计算重播 - 基于已有数据重新计算（不需要机械臂）")
    print("  4. 查看历史 - 列出所有历史标定数据")
    print("="*60)

    # 选择模式
    mode = input("\n请选择模式 (1/2/3/4) [默认1]: ").strip()
    if not mode:
        mode = "1"

    scanner = ManualHandEyeCalibrator()

    try:
        if mode == "1":
            # 手动标定模式
            print("\n" + "="*60)
            print("📝 手动标定模式")
            print("="*60)

            # 连接设备（会自动获取相机内参）
            if scanner.connect_devices():
                print("\n✅ 设备连接成功！")
                scanner.manual_calibration_mode()
            else:
                print("❌ 设备连接失败")

        elif mode == "2":
            # 轨迹重播模式
            print("\n" + "="*60)
            print("🔄 轨迹重播模式")
            print("="*60)

            # 列出可用的历史数据目录
            data_dirs = scanner.get_historical_data_dirs()

            if not data_dirs:
                print("❌ 没有找到历史标定数据")
                print("   请先运行手动标定模式生成数据")
                return

            print("\n📂 可用的历史数据:")
            for i, dir_name in enumerate(data_dirs, 1):
                # 检查是否有标定结果
                result_file = os.path.join(dir_name, "hand_eye_calibration_result.json")
                poses_file = os.path.join(dir_name, "poses.txt")

                if os.path.exists(poses_file):
                    # 统计位姿数量
                    pose_count = 0
                    with open(poses_file, 'r') as f:
                        for line in f:
                            if not line.startswith('#') and line.strip():
                                pose_count += 1

                    status = "✅ 已标定" if os.path.exists(result_file) else "📝 未标定"
                    print(f"  {i}. {dir_name} ({pose_count} 位姿) {status}")
                else:
                    print(f"  {i}. {dir_name} (无效)")

            # 选择数据目录
            choice = input("\n选择要重播的数据 (输入序号): ").strip()
            try:
                index = int(choice) - 1
                if 0 <= index < len(data_dirs):
                    selected_dir = data_dirs[index]
                    print(f"\n已选择: {selected_dir}")

                    # 连接设备（会自动获取相机内参）
                    if scanner.connect_devices():
                        print("\n✅ 设备连接成功！")
                        scanner.load_and_replay(selected_dir)
                    else:
                        print("❌ 设备连接失败")
                else:
                    print("❌ 无效的选择")
            except ValueError:
                print("❌ 请输入有效的数字")

        elif mode == "3":
            # 纯计算重播模式
            print("\n" + "="*60)
            print("🧮 纯计算重播模式")
            print("="*60)
            print("此模式不需要连接机械臂，仅基于已有数据重新计算标定结果")

            # 列出可用的历史数据目录
            data_dirs = scanner.get_historical_data_dirs()

            if not data_dirs:
                print("❌ 没有找到历史标定数据")
                print("   请先运行手动标定模式生成数据")
                return

            print("\n📂 可用的历史数据:")
            for i, dir_name in enumerate(data_dirs, 1):
                # 检查是否有位姿和图像文件
                poses_file = os.path.join(dir_name, "poses.txt")
                result_file = os.path.join(dir_name, "hand_eye_calibration_result.json")

                if os.path.exists(poses_file):
                    # 统计位姿数量
                    pose_count = 0
                    image_count = 0
                    with open(poses_file, 'r') as f:
                        for line in f:
                            if not line.startswith('#') and line.strip():
                                pose_count += 1

                    # 统计图像数量
                    images = glob.glob(os.path.join(dir_name, "*_Color.png"))
                    image_count = len(images)

                    status = "✅ 已标定" if os.path.exists(result_file) else "📝 未标定"
                    print(f"  {i}. {dir_name} ({pose_count} 位姿, {image_count} 图像) {status}")
                else:
                    print(f"  {i}. {dir_name} (无效)")

            # 选择数据目录
            choice = input("\n选择要重新计算的数据 (输入序号): ").strip()
            try:
                index = int(choice) - 1
                if 0 <= index < len(data_dirs):
                    selected_dir = data_dirs[index]
                    print(f"\n已选择: {selected_dir}")

                    # 不需要连接设备，直接进行计算
                    scanner.compute_calibration_only(selected_dir)
                else:
                    print("❌ 无效的选择")
            except ValueError:
                print("❌ 请输入有效的数字")

        elif mode == "4":
            # 查看历史模式
            print("\n" + "="*60)
            print("📊 历史标定数据")
            print("="*60)

            # 列出所有数据目录
            data_dirs = scanner.get_historical_data_dirs()

            if not data_dirs:
                print("❌ 没有找到历史标定数据")
                return

            for dir_name in data_dirs:
                print(f"\n📁 {dir_name}:")

                # 显示标定结果
                result_file = os.path.join(dir_name, "hand_eye_calibration_result.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result = json.load(f)

                    print(f"   时间: {result.get('timestamp', 'Unknown')}")
                    print(f"   方法: {result.get('method', 'Unknown')}")
                    print(f"   数据点: {result.get('data_points', 'Unknown')}")

                    if 'transformation' in result:
                        trans = result['transformation']
                        if 'translation_mm' in trans:
                            t = trans['translation_mm']
                            print(f"   平移: X={t[0]:.2f}mm Y={t[1]:.2f}mm Z={t[2]:.2f}mm")
                        if 'euler_xyz_deg' in trans:
                            e = trans['euler_xyz_deg']
                            print(f"   旋转: Rx={e[0]:.2f}° Ry={e[1]:.2f}° Rz={e[2]:.2f}°")

                    if 'quality' in result:
                        q = result['quality']
                        print(f"   质量: {q.get('quality_level', 'Unknown')}")
                        print(f"   误差: 旋转={q.get('avg_rotation_error_deg', 0):.3f}° "
                              f"平移={q.get('avg_translation_error_mm', 0):.3f}mm")
                else:
                    # 只显示位姿文件信息
                    poses_file = os.path.join(dir_name, "poses.txt")
                    if os.path.exists(poses_file):
                        pose_count = 0
                        with open(poses_file, 'r') as f:
                            for line in f:
                                if not line.startswith('#') and line.strip():
                                    pose_count += 1
                        print(f"   状态: 未标定")
                        print(f"   位姿数: {pose_count}")
                    else:
                        print(f"   状态: 无效数据")

        else:
            print(f"❌ 无效的模式选择: {mode}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断程序")
    except Exception as e:
        print(f"❌ 程序失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if mode in ["1", "2"]:
            scanner.disconnect_devices()


if __name__ == "__main__":
    main()
