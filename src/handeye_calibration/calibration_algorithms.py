#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_algorithms.py - 手眼标定算法模块

提供:
- 多算法融合 (Tsai, Park, Horaud, Daniilidis, Andreff)
- 迭代非线性优化
- 标定结果评估
- 误差分析

Design: Linus "Good Taste" 原则
- 每个算法独立,可单独测试
- 失败时优雅降级
- 清晰的误差度量
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as sst            

# ============================================================================
# 手眼标定算法
# ============================================================================

class HandEyeCalibration:
    """手眼标定算法集合"""

    # 类变量：保存完整的标定结果（包括副结果）
    _calibration_details = {}

    @staticmethod
    def get_calibration_quality(translation_error_mm, rotation_error_deg=None, 
                               mode='eye_in_hand'):
        """获取标定质量的emoji评级
        
        Args:
            translation_error_mm: 平移误差（毫米）
            rotation_error_deg: 旋转误差（度，可选）
            mode: 'eye_in_hand' 或 'eye_to_hand'
            
        Returns:
            str: 带emoji的质量评级字符串
        """
        if translation_error_mm < 3.0:
            return "🌟 优秀"
        elif translation_error_mm < 5.0:
            return "👍 良好"
        elif translation_error_mm < 10.0:
            return "⚠️  可接受"
        else:
            return "❌ 需要改进"
      

    @staticmethod
    def format_transformation_result(R, t, transform_name="变换"):
        """格式化变换矩阵结果输出
        
        Args:
            R: 旋转矩阵 (3x3)
            t: 平移向量 (3x1 或 3,)
            transform_name: 变换名称
        """
        # 确保平移向量是1D数组
        if t.ndim > 1:
            t = t.flatten()
        
        # 计算欧拉角
        from scipy.spatial.transform import Rotation as R_scipy
        r = R_scipy.from_matrix(R)
        euler_xyz = r.as_euler('xyz', degrees=True)
        quat_xyzw = r.as_quat()  # scipy返回 [x, y, z, w] 格式
        
        print(f"  {transform_name}:")
        print(f"    平移向量 (mm): X={t[0]*1000:.2f}, Y={t[1]*1000:.2f}, Z={t[2]*1000:.2f}")
        print(f"    欧拉角 (度):    Rx={euler_xyz[0]:.2f}°, Ry={euler_xyz[1]:.2f}°, Rz={euler_xyz[2]:.2f}°")
        print(f"    四元数 (xyzw):  x={quat_xyzw[0]:.4f}, y={quat_xyzw[1]:.4f}, z={quat_xyzw[2]:.4f}, w={quat_xyzw[3]:.4f}")

    @staticmethod
    def invert_rt(R, t):
        """计算变换的逆"""
        R_inv = R.T
        t_inv = -R_inv @ t
        return R_inv, t_inv
    
    @staticmethod
    def invert_transformations(R_list, t_list):
        """对变换列表求逆: T^(-1) = [R^T | -R^T*t]

        用于 eye-to-hand: 将 gripper2base 转换为 base2gripper

        Args:
            R_list: 旋转矩阵列表
            t_list: 平移向量列表

        Returns:
            (R_inv_list, t_inv_list): 逆变换列表
        """
        R_inv_list = []
        t_inv_list = []

        for R, t in zip(R_list, t_list):
            # T^(-1) = [R^T | -R^T * t]
            R_inv, t_inv = HandEyeCalibration.invert_rt(R,t)
            R_inv_list.append(R_inv)
            t_inv_list.append(t_inv)

        return R_inv_list, t_inv_list
    
 

    @staticmethod
    def multi_algorithm_fusion_eye_in_hand(R_gripper2base, t_gripper2base,
                               R_target2cam, t_target2cam, verbose=True):
        """多算法融合 - 选择最佳算法

        测试5种经典手眼标定算法,选择位姿重复性最好的

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_cam2gripper, t_cam2gripper, method_name) 或 (None, None, None)
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

                # 评估标定质量: 调用专门的评估函数
                eval_result = HandEyeCalibration.evaluate_calibration_eye_in_hand(
                    R_test, t_test, R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam, verbose=False
                )
                
                avg_t_error = eval_result['translation_error_mm']['mean']
                avg_r_error = eval_result['rotation_error_deg']['mean']

                # 综合评分: 归一化后加权求和
                # 平移: 5mm = 1.0, 旋转: 0.5° = 1.0
                score = (avg_t_error / 5.0) * 0.7 + (avg_r_error / 0.5) * 0.3

                if verbose:
                    print(f"   {method_name}: 平移={avg_t_error:.3f}mm, 旋转={avg_r_error:.3f}°, 综合={score:.3f}")
                    HandEyeCalibration.format_transformation_result(R_test, t_test, f"{method_name} Camera_to_Gripper")

                if score < best_score:
                    best_score = score
                    best_result = (R_test, t_test)
                    best_method = method_name

            except Exception as e:
                if verbose:
                    print(f"   {method_name}: 失败 ({e})")

        if best_result:
            return best_result[0], best_result[1], best_method

        return None, None, None

    @staticmethod
    def multi_algorithm_fusion_eye_to_hand(R_gripper2base, t_gripper2base,
                                               R_target2cam, t_target2cam, verbose=True):
        """Eye-to-Hand 多算法融合 - 新版本使用虚拟Eye-in-Hand方法
        将Eye-to-Hand问题转换为Eye-in-Hand问题，然后使用标准算法求解
        
        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息
            
        Returns:
            tuple: (R_cam2base, t_cam2base, method_name) 或 (None, None, None)
        """
        
        # 测试多种标准hand-eye算法
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
        
        # 数据预处理：转换为虚拟Eye-in-Hand格式
        R_base2gripper, t_base2gripper = HandEyeCalibration.invert_transformations(R_gripper2base, t_gripper2base)
        for method_id, method_name in methods:
            try:
                # 使用标准calibrateHandEye获取cam2base变换
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    R_base2gripper, t_base2gripper,
                    R_target2cam, t_target2cam,
                    method=method_id
                )
                
                # 计算 target2gripper 变换
                R_target2gripper_list = []
                t_target2gripper_list = []
                
                for i in range(len(R_base2gripper)):
                    # 构造变换矩阵
                    T_base2gripper_i = np.eye(4); T_base2gripper_i[:3,:3]=R_base2gripper[i]; T_base2gripper_i[:3,3:4]=t_base2gripper[i]
                    T_target2cam_i = np.eye(4); T_target2cam_i[:3,:3]=R_target2cam[i]; T_target2cam_i[:3,3:4]=t_target2cam[i]
                    T_cam2base = np.eye(4); T_cam2base[:3,:3]=R_cam2base; T_cam2base[:3,3:4]=t_cam2base
                    # 计算 ^eT_o = ^eT_b * ^bT_c * ^cT_o
                    T_target2gripper_i = T_base2gripper_i @ T_cam2base @ T_target2cam_i
                    R_target2gripper_list.append(T_target2gripper_i[:3,:3])
                    t_target2gripper_list.append(T_target2gripper_i[:3,3:4])

                # 旋转矩阵平均（四元数平均）
                
                quats_target2gripper = [sst.Rotation.from_matrix(R).as_quat() for R in R_target2gripper_list]
                mean_quat_target2gripper = np.mean(quats_target2gripper, axis=0)
                mean_quat_target2gripper = mean_quat_target2gripper / np.linalg.norm(mean_quat_target2gripper)
                R_target2gripper_avg = sst.Rotation.from_quat(mean_quat_target2gripper).as_matrix()
                t_target2gripper_avg = np.mean(t_target2gripper_list, axis=0)
                R_target2gripper, t_target2gripper =R_target2gripper_avg, t_target2gripper_avg

                # 评估标定质量
                eval_result = HandEyeCalibration.evaluate_calibration_eye_to_hand(
                    R_cam2base, t_cam2base, R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam, R_target2gripper, t_target2gripper,
                    verbose=False
                )

                avg_t_error = eval_result['translation_error_mm']['mean']
                avg_r_error = eval_result['rotation_error_deg']['mean']
                
                # 综合评分
                score = (avg_t_error / 5.0) * 1.0 + (avg_r_error / 0.5) * 0.0
                
                if verbose:
                    print(f"   {method_name}: 平移={avg_t_error:.3f}mm, 旋转={avg_r_error:.3f}°, 综合={score:.3f}")
                    HandEyeCalibration.format_transformation_result(R_cam2base, t_cam2base, f"{method_name} Camera_to_Base")
                    HandEyeCalibration.format_transformation_result(R_target2gripper, t_target2gripper, f"{method_name} Target_to_Gripper")
                
                if score < best_score:
                    best_score = score
                    best_result = (R_cam2base, t_cam2base, R_target2gripper, t_target2gripper)
                    best_method = method_name
                
            except Exception as e:
                if verbose:
                    print(f"   {method_name}: 失败 ({type(e).__name__}: {str(e)})")
        
        if best_result:
            # 保存完整的标定结果到类变量
            HandEyeCalibration._calibration_details = {
                'R_cam2base': best_result[0],
                't_cam2base': best_result[1],
                'R_target2gripper': best_result[2],
                't_target2gripper': best_result[3],
                'method': best_method,
                'mode': 'eye_to_hand'
            }
            return best_result[0], best_result[1], best_method
        
        return None, None, None


    @staticmethod
    def multi_algorithm_fusion_with_mode(R_gripper2base, t_gripper2base,
                                          R_target2cam, t_target2cam,
                                          mode='eye_in_hand', verbose=True):
        """多算法融合 - 支持 eye-in-hand 和 eye-to-hand

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            mode: 'eye_in_hand' 或 'eye_to_hand'
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_result, t_result, method_name) 或 (None, None, None)
        """
        if verbose:
            if mode == 'eye_in_hand':
                print(f"  标定模式: Eye-in-Hand (相机在末端)")
                print(f"  求解: camera_to_gripper")
            else:
                print(f"  标定模式: Eye-to-Hand (相机固定)")
                print(f"  求解: camera_to_base")

        if mode == 'eye_to_hand':
            return HandEyeCalibration.multi_algorithm_fusion_eye_to_hand(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam, verbose=verbose
            )
        else:
            # Eye-in-hand: 使用标准接口
            return HandEyeCalibration.multi_algorithm_fusion_eye_in_hand(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam, verbose=verbose
            )


    @staticmethod
    def levenberg_marquardt_optimization_eye_in_hand(R_initial, t_initial, R_gripper2base,
                                         t_gripper2base, R_target2cam, t_target2cam,
                                         verbose=True):
        """Levenberg-Marquardt 非线性优化

        使用 LM 算法优化标定结果，最小化位姿重复性残差

        Args:
            R_initial: 初始旋转矩阵
            t_initial: 初始平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_optimized, t_optimized)
        """
        from scipy.optimize import least_squares

        def residual_function(params):
            """残差函数: 计算所有帧的位姿偏差

            返回残差向量，LM 会最小化 sum(residuals^2)
            """
            # 解析参数
            rvec = params[:3]
            tvec = params[3:6]

            # 转换为旋转矩阵
            R_opt, _ = cv2.Rodrigues(rvec)
            t_opt = tvec.reshape(3, 1)

            # 计算每帧下 target 在 base 坐标系的位姿
            R_preds = []
            t_preds = []

            for i in range(len(R_gripper2base)):
                R_pred = R_gripper2base[i] @ R_opt @ R_target2cam[i]
                t_pred = R_gripper2base[i] @ (R_opt @ t_target2cam[i] + t_opt) + t_gripper2base[i]
                R_preds.append(R_pred)
                t_preds.append(t_pred)

            # 以第一帧为参考，计算所有帧的残差
            R_ref = R_preds[0]
            t_ref = t_preds[0]

            residuals = []
            for i in range(1, len(R_preds)):
                # 旋转误差 (度)
                R_error = R_ref.T @ R_preds[i]
                angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

                # 平移误差 (mm)
                t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000

                # 添加到残差向量（归一化：旋转权重 0.3，平移权重 0.7）
                residuals.append(angle_error * 10.0 * 0.5)  # 旋转: 1° = 10mm
                residuals.append(t_error * 0.5)               # 平移: mm

            return np.array(residuals)

        # 初始参数
        rvec_init, _ = cv2.Rodrigues(R_initial)
        tvec_init = t_initial.flatten()
        params_init = np.concatenate([rvec_init.flatten(), tvec_init])

        # 计算初始残差
        residuals_before = residual_function(params_init)
        error_before = np.sum(residuals_before**2)

        # 设置边界: 允许微调 (±5°, ±10mm)
        lower_bounds = []
        upper_bounds = []
        for i in range(3):
            # 旋转: ±5° ≈ ±0.087 rad
            lower_bounds.append(rvec_init[i] - 0.087)
            upper_bounds.append(rvec_init[i] + 0.087)
        for i in range(3):
            # 平移: ±10mm = ±0.01m
            lower_bounds.append(tvec_init[i] - 0.01)
            upper_bounds.append(tvec_init[i] + 0.01)

        # Levenberg-Marquardt 优化
        result = least_squares(
            residual_function,
            params_init,
            method='lm',  # Levenberg-Marquardt
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=100,
            verbose=0
        )

        # 解析结果
        rvec_opt = result.x[:3]
        tvec_opt = result.x[3:6]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        t_opt = tvec_opt.reshape(3, 1)

        # 计算优化后残差
        residuals_after = residual_function(result.x)
        error_after = np.sum(residuals_after**2)

        # 如果优化后更差，退回初始值
        if error_after > error_before:
            if verbose:
                print(f"   ⚠️  LM优化未改善结果，使用初始值")
                print(f"   优化前: {error_before:.3f}, 优化后: {error_after:.3f}")
            return R_initial, t_initial
        else:
            improvement = (error_before - error_after) / error_before * 100
            if verbose:
                print(f"   ✅ LM优化成功: {error_before:.3f} → {error_after:.3f} (改善{improvement:.1f}%)")
                print(f"   迭代次数: {result.nfev}, 状态: {result.status} ({result.message})")
            return R_opt, t_opt

    @staticmethod
    def levenberg_marquardt_optimization_eye_to_hand(R_initial, t_initial, R_gripper2base,
                                                      t_gripper2base, R_target2cam, t_target2cam,
                                                      verbose=True):
        """Eye-to-Hand Levenberg-Marquardt 非线性优化

        基于正确的Eye-to-Hand标定方程同时优化 cam2base 和 target2gripper 变换
        
        标定方程: T_gripper2base * T_target2gripper = T_cam2base * T_target2cam

        Args:
            R_initial: 初始旋转矩阵 (cam2base)
            t_initial: 初始平移向量 (cam2base)
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            R_target2gripper_initial: 初始target2gripper旋转矩阵（可选）
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_cam2base_optimized, t_cam2base_optimized)
        """
        from scipy.optimize import least_squares

        def residual_function(params):
            """
            残差函数：基于Eye-to-Hand标定方程
            
            标定方程: T_gripper2base * T_target2gripper = T_cam2base * T_target2cam
            
            params = [rvec_cam2base, t_cam2base, rvec_target2gripper, t_target2gripper] (12个参数)
            """
            # 解析参数
            rvec_cam2base = params[:3]
            t_cam2base = params[3:6].reshape(3, 1)
            rvec_target2gripper = params[6:9]
            t_target2gripper = params[9:12].reshape(3, 1)

            # 转换为旋转矩阵
            R_cam2base, _ = cv2.Rodrigues(rvec_cam2base)
            R_target2gripper, _ = cv2.Rodrigues(rvec_target2gripper)

            residuals = []
            for i in range(len(R_gripper2base)):
                # 左边：T_gripper2base * T_target2gripper (路径1: Target → Gripper → Base)
                T_gripper2base = np.eye(4)
                T_gripper2base[:3, :3] = R_gripper2base[i]
                T_gripper2base[:3, 3:4] = t_gripper2base[i]
                
                T_target2gripper = np.eye(4)
                T_target2gripper[:3, :3] = R_target2gripper
                T_target2gripper[:3, 3:4] = t_target2gripper
                
                left = T_gripper2base @ T_target2gripper  # Target → Gripper → Base
                
                # 右边：T_cam2base * T_target2cam (路径2: Target → Camera → Base)
                T_cam2base = np.eye(4)
                T_cam2base[:3, :3] = R_cam2base
                T_cam2base[:3, 3:4] = t_cam2base
                
                T_target2cam = np.eye(4)
                T_target2cam[:3, :3] = R_target2cam[i]
                T_target2cam[:3, 3:4] = t_target2cam[i]
                
                right = T_cam2base @ T_target2cam  # Target → Camera → Base
                
                # 计算变换误差：两条路径应该给出相同的Target在Base中的位姿
                error_T = left - right
                
                # 旋转误差 (度)
                R_error = error_T[:3, :3]
                angle_error = np.linalg.norm(R_error, 'fro') * 180 / np.pi
                
                # 平移误差 (mm)
                t_error = np.linalg.norm(error_T[:3, 3]) * 1000
                
                # 添加到残差向量
		# 添加到残差向量（归一化：旋转权重 0.3，平移权重 0.7）
                residuals.append(angle_error * 10.0 * 1.0)  # 旋转: 1° = 10mm
                residuals.append(t_error * 0.0)               # 平移: mm

                #residuals.append(angle_error)  # 旋转权重
                #residuals.append(t_error)       # 平移权重
            
            return np.array(residuals)

        # 从标定结果中提取target2gripper初始值
        R_target2gripper_initial = HandEyeCalibration._calibration_details['R_target2gripper']
        t_target2gripper_initial = HandEyeCalibration._calibration_details['t_target2gripper']

        # 初始参数
        rvec_cam2base_init, _ = cv2.Rodrigues(R_initial)
        rvec_target2gripper_init, _ = cv2.Rodrigues(R_target2gripper_initial)
        
        params_init = np.concatenate([
            rvec_cam2base_init.ravel(),
            t_initial.ravel(),
            rvec_target2gripper_init.ravel(),
            t_target2gripper_initial.ravel()
        ])

        # 计算初始残差
        residuals_before = residual_function(params_init)
        error_before = np.sum(residuals_before**2)

        # 设置边界: 允许微调
        lower_bounds = []
        upper_bounds = []
        # cam2base旋转边界 (±5°)
        for i in range(3):
            lower_bounds.append(rvec_cam2base_init[i] - 0.087)
            upper_bounds.append(rvec_cam2base_init[i] + 0.087)
        # cam2base平移边界 (±10mm)
        for i in range(3):
            lower_bounds.append(t_initial[i] - 0.01)
            upper_bounds.append(t_initial[i] + 0.01)
        # target2gripper旋转边界 (±10°)
        for i in range(3):
            lower_bounds.append(rvec_target2gripper_init[i] - 0.175)
            upper_bounds.append(rvec_target2gripper_init[i] + 0.175)
        # target2gripper平移边界 (±20mm)
        for i in range(3):
            lower_bounds.append(t_target2gripper_initial[i] - 0.02)
            upper_bounds.append(t_target2gripper_initial[i] + 0.02)

        # Levenberg-Marquardt 优化
        result = least_squares(
            residual_function,
            params_init,
            method='lm',  # Levenberg-Marquardt
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
            verbose=0
        )

        # 解析结果
        rvec_cam2base_opt = result.x[:3]
        t_cam2base_opt = result.x[3:6].reshape(3, 1)
        rvec_target2gripper_opt = result.x[6:9]
        t_target2gripper_opt = result.x[9:12].reshape(3, 1)
        
        R_cam2base_opt, _ = cv2.Rodrigues(rvec_cam2base_opt)
        R_target2gripper_opt, _ = cv2.Rodrigues(rvec_target2gripper_opt)

        # 计算优化后残差
        residuals_after = residual_function(result.x)
        error_after = np.sum(residuals_after**2)

        # 如果优化后更差，退回初始值
        if error_after > error_before:
            if verbose:
                print(f"   ⚠️  LM优化未改善结果，使用初始值")
                print(f"   优化前: {error_before:.3f}, 优化后: {error_after:.3f}")
            return R_initial, t_initial
        else:
            improvement = (error_before - error_after) / error_before * 100
            if verbose:
                print(f"   ✅ LM优化成功: {error_before:.3f} → {error_after:.3f} (改善{improvement:.1f}%)")
                print(f"   迭代次数: {result.nfev}, 状态: {result.status} ({result.message})")
                status = "收敛" if result.success else "未收敛"
                print(f"   优化状态: {status}")
            
            # 更新标定结果中的target2gripper变换
            HandEyeCalibration._calibration_details['R_target2gripper'] = R_target2gripper_opt
            HandEyeCalibration._calibration_details['t_target2gripper'] = t_target2gripper_opt
            
            return R_cam2base_opt, t_cam2base_opt

    @staticmethod
    def levenberg_marquardt_optimization_with_mode(R_initial, t_initial, R_gripper2base,
                                                    t_gripper2base, R_target2cam, t_target2cam,
                                                    mode='eye_in_hand', verbose=True):
        """Levenberg-Marquardt 优化 - 支持两种模式

        Args:
            R_initial: 初始旋转矩阵
            t_initial: 初始平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            mode: 'eye_in_hand' 或 'eye_to_hand'
            verbose: 是否显示详细信息

        Returns:
            tuple: (R_optimized, t_optimized)
        """
        # Eye-to-hand: 使用专用优化方法
        if mode == 'eye_to_hand':
            return HandEyeCalibration.levenberg_marquardt_optimization_eye_to_hand(
                R_initial, t_initial, R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam, verbose=verbose
            )
            #return R_initial, t_initial
        else:
            # Eye-in-hand: 使用标准优化方法
            return HandEyeCalibration.levenberg_marquardt_optimization_eye_in_hand(
                R_initial, t_initial, R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam, verbose=verbose
            )

    @staticmethod
    def evaluate_calibration_eye_to_hand(R_cam2base, t_cam2base, R_gripper2base,
                                          t_gripper2base, R_target2cam, t_target2cam,
                                          R_target2gripper, t_target2gripper,
                                          verbose=True, detail=False):
        """评估 Eye-to-Hand 标定结果质量
        
        基于标定方程 T_gripper2base * T_target2gripper = T_cam2base * T_target2cam 计算残差
        
        Args:
            R_cam2base: 相机到基座的旋转矩阵
            t_cam2base: 相机到基座的平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            R_target2gripper: 标靶到机器人末端的旋转矩阵
            t_target2gripper: 标靶到机器人末端的平移向量
            verbose: 是否显示详细信息
            
        Returns:
            dict: 评估结果
        """
        # 评估标定质量：基于Eye-to-Hand标定方程的残差
        t_errors = []
        r_errors = []
        
        for i in range(len(R_gripper2base)):
            # 左边：T_gripper2base * T_target2gripper (路径1: Target → Gripper → Base)
            R_gripper2base_i, t_gripper2base_i = R_gripper2base[i], t_gripper2base[i]
            T_gripper2base = np.eye(4)
            T_gripper2base[:3, :3] = R_gripper2base_i
            T_gripper2base[:3, 3:4] = t_gripper2base_i
            
            T_target2gripper = np.eye(4)
            T_target2gripper[:3, :3] = R_target2gripper
            T_target2gripper[:3, 3:4] = t_target2gripper
            
            left = T_gripper2base @ T_target2gripper  # Target → Gripper → Base
            
            # 右边：T_cam2base * T_target2cam (路径2: Target → Camera → Base)
            T_cam2base = np.eye(4)
            T_cam2base[:3, :3] = R_cam2base
            T_cam2base[:3, 3:4] = t_cam2base
            
            T_target2cam = np.eye(4)
            T_target2cam[:3, :3] = R_target2cam[i]
            T_target2cam[:3, 3:4] = t_target2cam[i]
            
            right = T_cam2base @ T_target2cam  # Target → Camera → Base
            
            # 计算变换误差：两条路径应该给出相同的Target在Base中的位姿
            error_T = left - right
            
            # 旋转误差 (度) - 使用Frobenius范数
            R_error = error_T[:3, :3]
            angle_error = np.linalg.norm(R_error, 'fro') * 180 / np.pi
            r_errors.append(angle_error)
            
            # 平移误差 (mm)
            t_error = np.linalg.norm(error_T[:3, 3]) * 1000
            t_errors.append(t_error)

        # 统计量
        result = {
            'translation_error_mm': {
                'mean': np.mean(t_errors),
                'std': np.std(t_errors),
                'max': np.max(t_errors),
                'median': np.median(t_errors)
            },
            'rotation_error_deg': {
                'mean': np.mean(r_errors),
                'std': np.std(r_errors),
                'max': np.max(r_errors),
                'median': np.median(r_errors)
            }
        }

        # 质量评级
        avg_t = result['translation_error_mm']['mean']
        avg_r = result['rotation_error_deg']['mean']
        quality_emoji = HandEyeCalibration.get_calibration_quality(avg_t, avg_r, mode='eye_to_hand')
        result['quality'] = quality_emoji

        if verbose:
            print(f"\n📊 标定质量评估 (Eye-to-Hand):")
            print(f"  验证方法: 前向运动学与相机观测比较")
            print(f"    方法1: 相机直接观测target位姿")
            print(f"    方法2: 前向运动学预测target位姿 (gripper2cam)")
            print(f"    比较两种方法在cam坐标系下的target位姿差异")
            print(f"  平移误差: {avg_t:.2f}±{result['translation_error_mm']['std']:.2f}mm (max={result['translation_error_mm']['max']:.2f}mm)")
            print(f"  旋转误差: {avg_r:.3f}±{result['rotation_error_deg']['std']:.3f}° (max={result['rotation_error_deg']['max']:.3f}°)")
            print(f"  质量评级: {quality_emoji}")

        # Detail mode: 打印每帧详细信息  
        if detail:
            print(f"\n============================================================")
            print(f"📊 最终标定质量分析")
            print(f"============================================================")
            
            # 首先计算所有帧的target在base坐标系下的位姿
            target_poses_in_base = []
            for i in range(len(R_gripper2base)):
                # 计算target在base坐标系下的位姿：T_gripper2base * T_target2gripper
                T_gb = np.eye(4)
                T_gb[:3, :3] = R_gripper2base[i]
                T_gb[:3, 3:4] = t_gripper2base[i]
                
                T_tg = np.eye(4)
                T_tg[:3, :3] = R_target2gripper
                T_tg[:3, 3:4] = t_target2gripper
                
                T_target_in_base = T_gb @ T_tg
                target_poses_in_base.append(T_target_in_base)
            
            for i in range(len(R_gripper2base)):
                # 计算当前帧target在base坐标系的位姿详细信息
                pos_mm = target_poses_in_base[i][:3, 3] * 1000  # 转换为毫米
                from scipy.spatial.transform import Rotation as R_scipy
                euler_deg = R_scipy.from_matrix(target_poses_in_base[i][:3, :3]).as_euler('xyz', degrees=True)
                
                pose_info = f" | 位姿: X={float(pos_mm[0]):+6.1f}, Y={float(pos_mm[1]):+6.1f}, Z={float(pos_mm[2]):+6.1f}mm, R={float(euler_deg[0]):+5.1f}°, P={float(euler_deg[1]):+5.1f}°, Y={float(euler_deg[2]):+5.1f}°"
                
                if i == 0:
                    print(f"✅ 帧 {i+1:2d}: 旋转误差  0.000°  平移误差   0.000mm{pose_info}")
                else:
                    # 使用已计算的误差值
                    t_error = t_errors[i-1]  # 因为t_errors从第二帧开始
                    r_error = r_errors[i-1]  # 因为r_errors从第二帧开始
                    
                    # 状态指示器
                    quality_text = HandEyeCalibration.get_calibration_quality(t_error)
                    status = quality_text.split()[0]  # 提取emoji部分
                    
                    print(f"{status} 帧 {i+1:2d}: 旋转误差 {r_error:6.3f}°  平移误差 {t_error:7.3f}mm{pose_info}")

        return result
    
    
    @staticmethod
    def evaluate_calibration_with_mode(R_cam2gripper, t_cam2gripper, R_gripper2base,
                            t_gripper2base, R_target2cam, t_target2cam,
                            verbose=True, mode='eye_in_hand', detail=False):
        """评估标定结果质量 - 支持两种模式

        计算位姿重复性误差和稳定性

        Args:
            R_cam2gripper: 相机到夹爪的旋转矩阵 (eye-in-hand) 或相机到基座 (eye-to-hand)
            t_cam2gripper: 相机到夹爪的平移向量 (eye-in-hand) 或相机到基座 (eye-to-hand)
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息
            mode: 'eye_in_hand' 或 'eye_to_hand'
            detail: 是否显示每帧详细信息

        Returns:
            dict: 评估结果
        """
        if mode == 'eye_to_hand':
            # 从类变量中获取副结果
            details = HandEyeCalibration._calibration_details
            R_target2gripper = details.get('R_target2gripper')
            t_target2gripper = details.get('t_target2gripper')

            return HandEyeCalibration.evaluate_calibration_eye_to_hand(
                R_cam2gripper, t_cam2gripper, R_gripper2base,
                t_gripper2base, R_target2cam, t_target2cam,
                R_target2gripper, t_target2gripper,
                verbose=verbose, detail=detail
            )
        elif mode == 'eye_in_hand':
            return HandEyeCalibration.evaluate_calibration_eye_in_hand(
                R_cam2gripper, t_cam2gripper, R_gripper2base,
                t_gripper2base, R_target2cam, t_target2cam,
                verbose=verbose, detail=detail
            )
            

    @staticmethod
    def evaluate_calibration_eye_in_hand(R_cam2gripper, t_cam2gripper, R_gripper2base,
                            t_gripper2base, R_target2cam, t_target2cam,
                            verbose=True, detail=False):
        """评估标定结果质量

        计算位姿重复性误差和稳定性

        Args:
            R_cam2gripper: 相机到夹爪的旋转矩阵 (eye-in-hand) 或相机到基座 (eye-to-hand)
            t_cam2gripper: 相机到夹爪的平移向量 (eye-in-hand) 或相机到基座 (eye-to-hand)
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            verbose: 是否显示详细信息
            mode: 'eye_in_hand' 或 'eye_to_hand'
            detail: 是否显示每帧详细信息

        Returns:
            dict: 评估结果
        """

        # Eye-in-hand: 计算每帧下target在base坐标系的位姿
        R_preds = []
        t_preds = []

        for i in range(len(R_gripper2base)):
            R_pred = R_gripper2base[i] @ R_cam2gripper @ R_target2cam[i]
            t_pred = R_gripper2base[i] @ (R_cam2gripper @ t_target2cam[i] + t_cam2gripper) + t_gripper2base[i]
            R_preds.append(R_pred)
            t_preds.append(t_pred)

        # 计算重复性误差
        R_ref = R_preds[0]
        t_ref = t_preds[0]

        t_errors = []
        r_errors = []

        for i in range(1, len(R_preds)):
            # 平移误差 (mm)
            t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000
            t_errors.append(t_error)

            # 旋转误差 (度)
            R_error = R_ref.T @ R_preds[i]
            angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            r_error = np.degrees(angle)
            r_errors.append(r_error)

        # 统计量
        result = {
            'translation_error_mm': {
                'mean': np.mean(t_errors),
                'std': np.std(t_errors),
                'max': np.max(t_errors),
                'median': np.median(t_errors)
            },
            'rotation_error_deg': {
                'mean': np.mean(r_errors),
                'std': np.std(r_errors),
                'max': np.max(r_errors),
                'median': np.median(r_errors)
            }
        }

        # 质量评级
        avg_t = result['translation_error_mm']['mean']
        avg_r = result['rotation_error_deg']['mean']
        quality_emoji = HandEyeCalibration.get_calibration_quality(avg_t, avg_r, mode='eye_in_hand')
        result['quality'] = quality_emoji

        if verbose:
            print(f"\n📊 标定质量评估:")
            print(f"  位姿重复性:")
            print(f"    平移: {avg_t:.2f}±{result['translation_error_mm']['std']:.2f}mm (max={result['translation_error_mm']['max']:.2f}mm)")
            print(f"    旋转: {avg_r:.3f}±{result['rotation_error_deg']['std']:.3f}° (max={result['rotation_error_deg']['max']:.3f}°)")
            print(f"  质量评级: {quality_emoji}")

        # Detail mode: 打印每帧详细信息
        if detail:
            print(f"\n============================================================")
            print(f"📊 最终标定质量分析")
            print(f"============================================================")
            
            for i in range(len(R_preds)):
                # 计算当前帧的位姿详细信息
                pos_mm = t_preds[i] * 1000  # 转换为毫米
                from scipy.spatial.transform import Rotation as R_scipy
                euler_deg = R_scipy.from_matrix(R_preds[i]).as_euler('xyz', degrees=True)
                
                pose_info = f" | 位姿: X={float(pos_mm[0]):+6.1f}, Y={float(pos_mm[1]):+6.1f}, Z={float(pos_mm[2]):+6.1f}mm, R={float(euler_deg[0]):+5.1f}°, P={float(euler_deg[1]):+5.1f}°, Y={float(euler_deg[2]):+5.1f}°"
                
                if i == 0:
                    print(f"✅ 帧 {i+1:2d}: 旋转误差  0.000°  平移误差   0.000mm{pose_info}")
                else:
                    # 计算与第一帧的误差
                    t_error = np.linalg.norm(t_preds[i] - t_preds[0]) * 1000
                    R_error = R_preds[0].T @ R_preds[i]
                    angle_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))
                    
                    # 状态指示器
                    quality_text = HandEyeCalibration.get_calibration_quality(t_error)
                    status = quality_text.split()[0]  # 提取emoji部分
                    
                    print(f"{status} 帧 {i+1:2d}: 旋转误差 {angle_error:6.3f}°  平移误差 {t_error:7.3f}mm{pose_info}")

        return result


# ============================================================================
# 误差分析工具
# ============================================================================

class ErrorAnalyzer:
    """误差分析工具"""

    @staticmethod
    def analyze_error_patterns(R_cam2gripper, t_cam2gripper, R_gripper2base,
                               t_gripper2base, R_target2cam, t_target2cam,
                               frame_ids):
        """分析误差模式,识别异常帧

        Args:
            R_cam2gripper: 相机到夹爪的旋转矩阵
            t_cam2gripper: 相机到夹爪的平移向量
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            R_target2cam: 标靶到相机的旋转列表
            t_target2cam: 标靶到相机的平移列表
            frame_ids: 帧ID列表

        Returns:
            dict: 误差分析结果
        """
        # 计算每帧的位姿预测
        R_preds = []
        t_preds = []

        for i in range(len(R_gripper2base)):
            R_pred = R_gripper2base[i] @ R_cam2gripper @ R_target2cam[i]
            t_pred = R_gripper2base[i] @ (R_cam2gripper @ t_target2cam[i] + t_cam2gripper) + t_gripper2base[i]
            R_preds.append(R_pred)
            t_preds.append(t_pred)

        # 以第一帧为参考
        R_ref = R_preds[0]
        t_ref = t_preds[0]

        # 计算每帧误差
        frame_errors = []
        for i in range(1, len(R_preds)):
            t_error = np.linalg.norm(t_preds[i] - t_ref) * 1000
            R_error = R_ref.T @ R_preds[i]
            r_error = np.degrees(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))

            frame_errors.append({
                'frame_id': frame_ids[i],
                'translation_error_mm': t_error,
                'rotation_error_deg': r_error,
                'total_error': t_error + r_error * 10.0  # 1° = 10mm
            })

        # 按误差排序
        frame_errors.sort(key=lambda x: x['total_error'], reverse=True)

        # 识别异常帧 (超过中位数 + 2倍MAD)
        errors = np.array([fe['total_error'] for fe in frame_errors])
        median = np.median(errors)
        mad = np.median(np.abs(errors - median))
        threshold = median + 2.0 * mad

        outliers = [fe for fe in frame_errors if fe['total_error'] > threshold]

        return {
            'frame_errors': frame_errors,
            'outliers': outliers,
            'outlier_threshold': threshold,
            'worst_frames': frame_errors[:5]  # 最差的5帧
        }

    @staticmethod
    def check_motion_diversity(R_gripper2base, t_gripper2base, verbose=True):
        """检查运动多样性

        Args:
            R_gripper2base: 机器人末端到基座的旋转列表
            t_gripper2base: 机器人末端到基座的平移列表
            verbose: 是否显示详细信息

        Returns:
            dict: 运动多样性分析结果
        """
        # 计算位移范围
        positions = np.array([t.flatten() for t in t_gripper2base])
        pos_range = positions.max(axis=0) - positions.min(axis=0)
        pos_range_mm = pos_range * 1000

        # 计算旋转范围
        euler_angles = []
        for R_mat in R_gripper2base:
            r = R.from_matrix(R_mat)
            euler = r.as_euler('xyz', degrees=True)
            euler_angles.append(euler)

        euler_angles = np.array(euler_angles)
        rot_range = euler_angles.max(axis=0) - euler_angles.min(axis=0)

        result = {
            'position_range_mm': pos_range_mm,
            'rotation_range_deg': rot_range,
            'sufficient_motion': (pos_range_mm > 50).all() and (rot_range > 10).all()
        }

        if verbose:
            print(f"\n🎯 运动多样性:")
            print(f"  位移范围: X={pos_range_mm[0]:.1f}mm, Y={pos_range_mm[1]:.1f}mm, Z={pos_range_mm[2]:.1f}mm")
            print(f"  旋转范围: Roll={rot_range[0]:.1f}°, Pitch={rot_range[1]:.1f}°, Yaw={rot_range[2]:.1f}°")
            print(f"  运动充分: {'✅' if result['sufficient_motion'] else '❌'}")

        return result
