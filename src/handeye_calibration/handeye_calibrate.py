#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handeye_calibrate.py - Hand-Eye Calibration Computation Module

Responsibilities:
- Load collected calibration data from disk
- Perform calibration computation (NO hardware interaction)
- Save calibration results

Design: Pure Computation
- This module ONLY handles calibration computation
- No hardware dependencies (robot or camera)
- Can be run offline with historical data
"""

import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime

# Import calibration modules
from calibration_common import (
    prepare_calibration_data,
    CameraIntrinsicsManager
)
from improve_data_quality import DataQualityFilter, RANSACFilter
from calibration_algorithms import HandEyeCalibration, ErrorAnalyzer


class HandEyeCalibrator:
    """Hand-eye calibration computer

    Pure computation - no hardware dependencies
    """

    VERSION = "3.5.4"  # 支持 Eye-in-Hand 和 Eye-to-Hand

    def __init__(self, config=None):
        """Initialize calibrator

        Args:
            config: Optional, configuration dict
        """
        print(f"HandEyeCalibrator v{self.VERSION}")
        print("="*60)

        # Directory configuration
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_data_dir = os.path.join(self.script_dir, "calibration_data")
        self.verified_data_dir = os.path.join(self.script_dir, "verified_data")
        self.config_dir = os.path.join(self.script_dir, "config")  # Fixed: removed dirname

        # Load configuration from file
        config_file = os.path.join(self.config_dir, "calibration_config.yaml")
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
        else:
            file_config = {}

        # Store the full config for later use
        self.file_config = file_config

        # Default configuration (without specific board_size yet)
        self.config = {
            'mode': file_config.get('calibration_mode', 'eye_in_hand'),
            'board_size': None,  # Will be set based on actual mode
            'chessboard_size_mm': None,  # Will be set based on actual mode
            'quality_filter': file_config.get('quality_filter', {
                'min_motion_mm': 5.0,
                'min_rotation_deg': 2.0,
                'boundary_margin_ratio': 0.10,
                'max_pitch_deg': 70.0
            }),
            'ransac': file_config.get('ransac', {
                'threshold_mm': 6.0,
                'min_inlier_ratio': 0.3
            }),
            'min_frames': file_config.get('calibration', {}).get('min_frames', 10)
        }

        # Override with user config
        if config:
            self.config.update(config)

        # Don't set board_size and chessboard_size_mm yet - will be set in calibrate()
        # based on the actual calibration mode
        self.board_size = None
        self.chessboard_size_mm = None

    # ========================================================================
    # Load Data
    # ========================================================================

    def load_calibration_data(self, data_dir, redetect_corners=False, calibration_mode=None):
        """Load calibration data from directory

        Args:
            data_dir: Directory containing calibration data
            redetect_corners: If True, re-detect corners from images instead of using saved corners
            calibration_mode: Calibration mode ('eye_in_hand' or 'eye_to_hand'), used for selecting intrinsics

        Supports both new format (calibration_data.json) and legacy format (poses.txt + images)

        Args:
            data_dir: Data directory path

        Returns:
            tuple: (collected_data, camera_matrix, dist_coeffs, source, metadata)
                   or (None, None, None, None, None) if failed
        """
        print(f"\nLoading calibration data from: {data_dir}")

        # Try to load metadata (contains calibration mode info)
        metadata = {}
        metadata_file = os.path.join(data_dir, "calibration_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'calibration_mode' in metadata:
                        print(f"  检测到标定模式: {metadata['calibration_mode']}")
                        # If calibration_mode not provided, use metadata
                        if calibration_mode is None:
                            calibration_mode = metadata['calibration_mode']
            except Exception as e:
                print(f"  Warning: Failed to load metadata: {e}")

        # Try new format first (calibration_data.json)
        data_file = os.path.join(data_dir, "calibration_data.json")
        poses_file = os.path.join(data_dir, "poses.txt")
        
        
        
        # Load camera intrinsics based on calibration mode
        # Default to config mode if calibration_mode is None
        if calibration_mode is None:
            calibration_mode = self.config.get('mode', 'eye_in_hand')

        # Select the appropriate intrinsics file based on calibration mode
        print(f"  🔍 调试：当前file_config中的key: {list(self.file_config.keys())}")
        if calibration_mode == 'eye_in_hand':
            # 从配置中获取eye_in_hand模式的相机内参文件
            intrinsics_file = self.file_config.get('eye_in_hand', {}).get('camera', {}).get('intrinsics_file')
            if not intrinsics_file:
                raise ValueError(f"❌ 配置错误：'{calibration_mode}' 模式的相机内参文件未配置，请在config中设置 eye_in_hand.camera.intrinsics_file")
            print(f"  📸 使用 Eye-in-Hand 模式相机内参: {intrinsics_file}")
        elif calibration_mode == 'eye_to_hand':
            # 从配置中获取eye_to_hand模式的相机内参文件
            print(f"  🔍 调试：eye_to_hand配置: {self.file_config.get('eye_to_hand', {})}")
            intrinsics_file = self.file_config.get('eye_to_hand', {}).get('camera', {}).get('intrinsics_file')
            if not intrinsics_file:
                raise ValueError(f"❌ 配置错误：'{calibration_mode}' 模式的相机内参文件未配置，请在config中设置 eye_to_hand.camera.intrinsics_file")
            print(f"  📸 使用 Eye-to-Hand 模式相机内参: {intrinsics_file}")
        else:
            raise ValueError(f"❌ 配置错误：未知的标定模式 '{calibration_mode}'，支持的模式: 'eye_in_hand', 'eye_to_hand'")

        # Load the selected intrinsics file
        intrinsics_path = os.path.join(self.config_dir, intrinsics_file)
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"❌ 相机内参文件不存在: {intrinsics_path}")
        
        camera_matrix, dist_coeffs = CameraIntrinsicsManager.load_from_file(intrinsics_path)
        source = f"File: {intrinsics_file}"

        if camera_matrix is None:
            raise ValueError(f"❌ 相机内参加载失败: {intrinsics_file}")

        print(f"  ✅ 内参加载成功 ({source})")
        print(f"     焦距: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
        print(f"     主点: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")



        collected_data = []

        if os.path.exists(data_file):
            # New format: calibration_data.json
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                # Load camera intrinsics if redetecting corners
                if redetect_corners:
                    print("🔄 重新检测角点模式")
                # Process each frame
                for item in data:
                    # Skip frames where chessboard was not detected (if not redetecting)
                    if not redetect_corners and not item.get('chessboard_detected', True):
                        continue

                    frame_id = item['frame_id']

                    # Re-detect corners from image if requested
                    if redetect_corners:
                        image_file = os.path.join(data_dir, f"{frame_id}_Color.png")

                        if not os.path.exists(image_file):
                            print(f"⚠️  帧 {frame_id}: 找不到图像文件")
                            continue

                        # Read image
                        image = cv2.imread(image_file)
                        if image is None:
                            print(f"⚠️  帧 {frame_id}: 无法读取图像文件")
                            continue

                        # Detect chessboard corners
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        ret, corners = cv2.findChessboardCorners(
                            gray,
                            self.board_size,
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
                        )

                        if not ret:
                            print(f"⚠️  帧 {frame_id}: 棋盘格检测失败")
                            continue

                        # Refine corners
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                        # Calculate reprojection error
                        reprojection_error = self._calculate_reprojection_error(
                            corners, camera_matrix, dist_coeffs
                        )

                        collected_data.append({
                            'frame_id': frame_id,
                            'pose': item['pose'],
                            'corners': corners,
                            'reprojection_error_px': reprojection_error
                        })

                        print(f"✅ 帧 {frame_id}: 棋盘格重新检测成功 (重投影误差: {reprojection_error:.3f}px)")

                    else:
                        # Use saved corners
                        # Skip if corners key is missing (shouldn't happen with chessboard_detected=true)
                        if 'corners' not in item:
                            print(f"  Warning: Frame {item['frame_id']} marked as detected but missing corners")
                            continue

                        corners = np.array(item['corners'], dtype=np.float32)
                        collected_data.append({
                            'frame_id': item['frame_id'],
                            'pose': item['pose'],
                            'corners': corners
                        })

                if redetect_corners:
                    print(f"  重新检测: {len(collected_data)}/{len(data)} 帧成功")
                else:
                    print(f"  Loaded {len(collected_data)}/{len(data)} frames with valid chessboard detection")

            except Exception as e:
                print(f"  Error loading data: {e}")
                return None, None, None, None, None

        elif os.path.exists(poses_file):
            # Legacy format: poses.txt + image files
            print(f"  Using legacy format (poses.txt + images)")

            try:
                # Read poses
                poses = []
                with open(poses_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) >= 7:
                            frame_id = int(parts[0])
                            roll, pitch, yaw = float(parts[1]), float(parts[2]), float(parts[3])
                            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                            poses.append({
                                'frame_id': frame_id,
                                'pose': [x, y, z, roll, pitch, yaw]
                            })

                print(f"📊 找到 {len(poses)} 个位姿数据")

                # Process each image to detect corners
                print("\n检测棋盘格...")
                for pose_data in poses:
                    frame_id = pose_data['frame_id']
                    image_file = os.path.join(data_dir, f"{frame_id}_Color.png")

                    if not os.path.exists(image_file):
                        print(f"⚠️  帧 {frame_id}: 找不到图像文件")
                        continue

                    # Read image
                    image = cv2.imread(image_file)
                    if image is None:
                        print(f"⚠️  帧 {frame_id}: 无法读取图像文件")
                        continue

                    # Detect chessboard corners
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray,
                        self.board_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
                    )

                    if not ret:
                        print(f"⚠️  帧 {frame_id}: 棋盘格检测失败")
                        continue

                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # Calculate reprojection error
                    reprojection_error = self._calculate_reprojection_error(
                        corners, camera_matrix, dist_coeffs
                    )

                    collected_data.append({
                        'frame_id': frame_id,
                        'pose': pose_data['pose'],
                        'corners': corners,
                        'reprojection_error_px': reprojection_error
                    })

                    print(f"✅ 帧 {frame_id}: 棋盘格检测成功 (重投影误差: {reprojection_error:.3f}px)")

                print(f"\n📊 有效数据: {len(collected_data)}/{len(poses)} 组")

                if len(collected_data) == 0:
                    print(f"  Error: No valid frames found")
                    return None, None, None, None, None

                return collected_data, camera_matrix, dist_coeffs, source, metadata

            except Exception as e:
                print(f"  Error processing legacy format: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, None, None
        else:
            print(f"  Error: No calibration data found (neither calibration_data.json nor poses.txt)")
            return None, None, None, None, None

        return collected_data, camera_matrix, dist_coeffs, source, metadata

    def _calculate_reprojection_error(self, corners, camera_matrix, dist_coeffs):
        """计算棋盘格角点的重投影误差

        Args:
            corners: 检测到的角点 (Nx1x2)
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数

        Returns:
            float: 平均重投影误差（像素）
        """
        # 准备棋盘格3D点
        objpoints = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        objpoints *= self.chessboard_size_mm / 1000.0  # 转为米

        # 使用PnP求解棋盘格位姿
        ret, rvec, tvec = cv2.solvePnP(
            objpoints, corners,
            camera_matrix, dist_coeffs
        )

        if not ret:
            return float('inf')  # PnP失败返回无穷大

        # 重投影3D点到图像平面
        projected_points, _ = cv2.projectPoints(
            objpoints, rvec, tvec,
            camera_matrix, dist_coeffs
        )

        # 计算平均重投影误差
        error = np.linalg.norm(
            corners.reshape(-1, 2) - projected_points.reshape(-1, 2),
            axis=1
        ).mean()

        return error

    # ========================================================================
    # Calibration Computation
    # ========================================================================

    def calibrate(self, data_dir, calibration_mode, save_results=True, redetect_corners=False):
        """Perform hand-eye calibration computation

        Args:
            data_dir: Data directory path
            mode: 'eye_in_hand' or 'eye_to_hand' (None = use config default)
            save_results: Whether to save results to data_dir
            redetect_corners: If True, re-detect corners from images instead of using saved corners

        Returns:
            tuple: (R_result, t_result, quality_result) or (None, None, None)
        """
        # Step 0: Load board configuration FIRST (needed for corner redetection)
        print("\nStep 0: Load calibration configuration")

        # Load board_size from config based on calibration_mode
        if calibration_mode not in self.file_config:
            raise ValueError(f"❌ 配置错误：标定模式 '{calibration_mode}' 在配置文件中未找到")

        if 'chessboard' not in self.file_config[calibration_mode]:
            raise ValueError(f"❌ 配置错误：'{calibration_mode}' 模式的棋盘格参数未配置，请在config中设置 {calibration_mode}.chessboard")

        chessboard_config = self.file_config[calibration_mode]['chessboard']

        if 'board_size' not in chessboard_config:
            raise ValueError(f"❌ 配置错误：'{calibration_mode}' 模式的 board_size 未配置")
        if 'square_size_mm' not in chessboard_config:
            raise ValueError(f"❌ 配置错误：'{calibration_mode}' 模式的 square_size_mm 未配置")

        # Set board_size and chessboard_size_mm BEFORE loading data
        self.board_size = tuple(chessboard_config['board_size'])
        self.chessboard_size_mm = chessboard_config['square_size_mm']

        # Update config with actual values
        self.config['board_size'] = self.board_size
        self.config['chessboard_size_mm'] = self.chessboard_size_mm

        print(f"  模式: {calibration_mode}")
        print(f"  棋盘格尺寸: {self.board_size}")
        print(f"  方格大小: {self.chessboard_size_mm}mm")

        # Now load data (with board_size already set for redetection)
        print("\nStep 1: Load calibration data")

        # 加载数据以获取metadata
        collected_data, camera_matrix, dist_coeffs, source, data_metadata = \
            self.load_calibration_data(data_dir, redetect_corners=redetect_corners, calibration_mode=calibration_mode)

        if collected_data is None:
            print("Failed to load data")
            return None, None, None
        
        # 强制要求data_metadata包含calibration_mode
        if 'calibration_mode' not in data_metadata:
            raise ValueError("❌ 数据错误：数据metadata中缺少calibration_mode，请确保数据收集时正确设置了标定模式")
        
        metadata_mode = data_metadata['calibration_mode']

        # 校验传入的calibration_mode参数与data_metadata的一致性
        if calibration_mode and calibration_mode != metadata_mode:
            raise ValueError(f"❌ 模式不一致：指定的标定模式是'{calibration_mode}'，但数据metadata中是'{metadata_mode}'，请确保一致性")
        
        # 校验camera serial_id一致性
        config_serial_id = self.file_config.get(metadata_mode, {}).get('camera', {}).get('serial_id')
        metadata_camera_id = data_metadata.get('camera_id')
        
        if not config_serial_id:
            raise ValueError(f"❌ 配置错误：'{metadata_mode}' 模式的相机serial_id未配置，请在config中设置 {metadata_mode}.camera.serial_id")
        
        if not metadata_camera_id:
            raise ValueError(f"❌ 数据错误：数据metadata中缺少camera_id，请确保数据收集时正确记录了相机ID")
        
        if config_serial_id != metadata_camera_id:
            raise ValueError(f"❌ 相机不匹配：配置文件中的serial_id是'{config_serial_id}'，但数据metadata中的camera_id是'{metadata_camera_id}'，请确保使用相同的相机")

        # Print header
        print("\n" + "="*60)
        print(f"Hand-Eye Calibration Computation ({calibration_mode.upper().replace('_', '-')})")
        print("="*60)


        # Step 2: Data quality filtering
        print("\nStep 2: Data quality filtering")
        filtered_data, filter_report = DataQualityFilter.apply_all_filters(
            collected_data, config=self.config.get('quality_filter', {}), verbose=True
        )

        if len(filtered_data) < self.config['min_frames']:
            print(f"\nInsufficient data after filtering: {len(filtered_data)} < {self.config['min_frames']}")
            return None, None, None





        # Step 3: Data preparation
        print("\nStep 3: Data preparation")
        R_g, t_g, R_t, t_t, frame_ids, reprojection_errors = \
            prepare_calibration_data(
                filtered_data,
                camera_matrix,
                dist_coeffs,
                self.board_size,
                self.chessboard_size_mm,
                compute_reprojection_errors=True
            )

        if len(R_g) < self.config['min_frames']:
            print(f"\nInsufficient valid poses: {len(R_g)} < {self.config['min_frames']}")
            return None, None, None

        print(f"  {len(R_g)} valid poses")
        print(f"  Reprojection error: {np.mean(reprojection_errors):.2f}±{np.std(reprojection_errors):.2f}px")

        # Step 4: Filter high reprojection error frames
        print("\nStep 4: Filter high reprojection error frames")

        # 保存过滤前的数据用于对比
        original_frame_ids = frame_ids.copy()
        original_errors = reprojection_errors.copy()

        R_g, t_g, R_t, t_t, frame_ids, reprojection_errors, removed = \
            DataQualityFilter.filter_reprojection_error(
                R_g, t_g, R_t, t_t, frame_ids, reprojection_errors,
                max_error_px=2.0
            )

        # 逐帧打印分析结果
        print(f"  过滤阈值: 重投影误差 ≤ 2.0px")
        print(f"\n  逐帧分析:")
        for i, (fid, err) in enumerate(zip(original_frame_ids, original_errors)):
            if err <= 2.0:
                print(f"    ✅ 帧 {fid}: 重投影误差 {err:.3f}px (保留)")
            else:
                print(f"    ❌ 帧 {fid}: 重投影误差 {err:.3f}px (移除)")

        # 统计总结
        print(f"\n  📊 过滤结果:")
        print(f"    原始帧数: {len(original_frame_ids)}")
        print(f"    保留帧数: {len(frame_ids)}")
        print(f"    移除帧数: {len(removed)}")
        if removed:
            removed_ids = [item['frame_id'] for item in removed]
            print(f"    移除的帧: {removed_ids}")

        if len(R_g) < self.config['min_frames']:
            print(f"\nInsufficient data after reprojection filtering: {len(R_g)} < {self.config['min_frames']}")
            return None, None, None

        # Step 5: RANSAC filtering
        print("\nStep 5: RANSAC geometric consistency filtering")
        print(f"  阈值: {self.config['ransac']['threshold_mm']}mm")
        best_inliers = RANSACFilter.ransac_filter_handeye(
            R_g, t_g, R_t, t_t, frame_ids,
            threshold=self.config['ransac']['threshold_mm'],
            min_inlier_ratio=self.config['ransac']['min_inlier_ratio']
        )

        if len(best_inliers) < self.config['min_frames']:
            print(f"\nInsufficient inliers after RANSAC: {len(best_inliers)} < {self.config['min_frames']}")
            return None, None, None

        # 逐帧打印 RANSAC 结果
        print(f"\n  逐帧分析:")
        inlier_set = set(best_inliers)
        for i, fid in enumerate(frame_ids):
            if i in inlier_set:
                print(f"    ✅ 帧 {fid}: 几何一致性良好 (inlier)")
            else:
                print(f"    ❌ 帧 {fid}: 几何异常 (outlier)")

        # 统计总结
        print(f"\n  📊 RANSAC 结果:")
        print(f"    原始帧数: {len(frame_ids)}")
        print(f"    Inliers: {len(best_inliers)}")
        print(f"    Outliers: {len(frame_ids) - len(best_inliers)}")
        print(f"    Inlier比例: {len(best_inliers)/len(frame_ids)*100:.1f}%")
        print(f"    保留的帧: {[frame_ids[i] for i in best_inliers]}")

        # Extract inliers
        R_g_inliers = [R_g[i] for i in best_inliers]
        t_g_inliers = [t_g[i] for i in best_inliers]
        R_t_inliers = [R_t[i] for i in best_inliers]
        t_t_inliers = [t_t[i] for i in best_inliers]
        inlier_frame_ids = [frame_ids[i] for i in best_inliers]

        # Step 6: Multi-algorithm fusion
        print("\nStep 6: Multi-algorithm fusion")
        R_result, t_result, best_method = HandEyeCalibration.multi_algorithm_fusion_with_mode(
            R_g_inliers, t_g_inliers, R_t_inliers, t_t_inliers,
            mode=calibration_mode,  # 传入标定模式
            verbose=True
        )

        if R_result is None:
            print("\nAll algorithms failed")
            return None, None, None

        print(f"  Best algorithm: {best_method}")

        # Step 7: Levenberg-Marquardt optimization
        print("\nStep 7: Levenberg-Marquardt non-linear optimization")
        R_optimized, t_optimized = HandEyeCalibration.levenberg_marquardt_optimization_with_mode(
            R_result, t_result, R_g_inliers, t_g_inliers,
            R_t_inliers, t_t_inliers,
            mode=calibration_mode,  # 传入标定模式
            verbose=True
        )

        # Step 8: Evaluation
        print("\nStep 8: Calibration result evaluation")
        quality_result = HandEyeCalibration.evaluate_calibration_with_mode(
            R_optimized, t_optimized, R_g_inliers, t_g_inliers,
            R_t_inliers, t_t_inliers, mode=calibration_mode, verbose=True, detail=True
        )

        # Print detailed calibration result (Legacy-style format)
        from scipy.spatial.transform import Rotation as R_scipy
        rotation = R_scipy.from_matrix(R_optimized)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        euler_angles = rotation.as_euler('xyz', degrees=True)
        t_mm = t_optimized.flatten() * 1000.0

        print("\n" + "="*60)
        print("📊 最终标定结果:")
        print("="*60)
        print(f"  平移向量 (mm): X={t_mm[0]:.2f}, Y={t_mm[1]:.2f}, Z={t_mm[2]:.2f}")
        print(f"  欧拉角 (度):    Rx={euler_angles[0]:.2f}°, Ry={euler_angles[1]:.2f}°, Rz={euler_angles[2]:.2f}°")
        print(f"  四元数 (xyzw):  x={quaternion[0]:.4f}, y={quaternion[1]:.4f}, z={quaternion[2]:.4f}, w={quaternion[3]:.4f}")


        print(f"\n📈 标定质量:")
        avg_t = quality_result['translation_error_mm']['mean']
        avg_r = quality_result['rotation_error_deg']['mean']
        max_t = quality_result['translation_error_mm']['max']
        max_r = quality_result['rotation_error_deg']['max']

        print(f"  平移重复性: 平均 {avg_t:.2f}mm (最大 {max_t:.2f}mm)")
        print(f"  旋转重复性: 平均 {avg_r:.3f}° (最大 {max_r:.3f}°)")
       

        # Quality assessment
        quality_emoji = HandEyeCalibration.get_calibration_quality(avg_t, avg_r, mode=calibration_mode)

        print(f"  总体质量: {quality_emoji}")

        print(f"\n  使用数据: {len(best_inliers)}/{len(collected_data)} 帧")
        print(f"  算法: Optimized_{best_method}_with_RANSAC")

        # Step 9: Save results
        if save_results:
            print("\nStep 9: Save calibration results")
            self._save_calibration_result(
                R_optimized, t_optimized, data_dir,
                {
                    'mode': calibration_mode,  # 保存标定模式
                    'method': f'Optimized_{best_method}_with_RANSAC',
                    'data_used': f'{len(best_inliers)}/{len(collected_data)}',
                    'quality': quality_result,
                    'inlier_frame_ids': inlier_frame_ids,
                    'filter_report': filter_report
                }
            )

        print("\n" + "="*60)
        print("Calibration completed successfully!")
        print("="*60)

        return R_optimized, t_optimized, quality_result

    def _save_calibration_result(self, R_cam2gripper, t_cam2gripper, save_dir, metadata):
        """Save calibration result to disk

        Args:
            R_cam2gripper: Camera to gripper rotation matrix
            t_cam2gripper: Camera to gripper translation vector
            save_dir: Save directory
            metadata: Metadata dict
        """
        from scipy.spatial.transform import Rotation as R
        import yaml

        # Convert to quaternion
        r = R.from_matrix(R_cam2gripper)
        quat = r.as_quat()  # [x, y, z, w]

        # 根据模式确定结果名称
        calibration_mode = metadata.get('mode', 'eye_in_hand')
        if calibration_mode == 'eye_to_hand':
            result_key = 'camera_to_base'
        else:
            result_key = 'camera_to_flan'
	
	
        result = {
            'version': self.VERSION,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'mode': calibration_mode,
            'method': metadata.get('method', 'Unknown'),
            result_key: {
                'translation': {
                    'x': float(t_cam2gripper[0][0]),
                    'y': float(t_cam2gripper[1][0]),
                    'z': float(t_cam2gripper[2][0])
                },
                'rotation_matrix': R_cam2gripper.tolist(),
                'quaternion': {
                    'x': float(quat[0]),
                    'y': float(quat[1]),
                    'z': float(quat[2]),
                    'w': float(quat[3])
                }
            },
            'quality': metadata.get('quality', {}),
            'data_statistics': {
                'frames_used': metadata.get('data_used', 'N/A'),
                'inlier_frame_ids': metadata.get('inlier_frame_ids', [])
            }
        }
        print(result)

        print("")
        print("")
        print("q = [%s,%s,%s,%s]"%tuple(quat))
        print("cfg.T_cam2flan = [%s,%s,%s]"%(float(t_cam2gripper[0][0]),float(t_cam2gripper[1][0]),float(t_cam2gripper[2][0])))
        print("")

        # Generate timestamped filenames
        timestamp = result['timestamp']
        json_file = os.path.join(save_dir, f"hand_eye_calibration_result_{timestamp}.json")
        yaml_file = os.path.join(save_dir, f"hand_eye_calibration_result_{timestamp}.yaml")

        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"  Results saved: {json_file}")

        # Save YAML (ROS friendly)
        with open(yaml_file, 'w') as f:
            yaml.dump(result, f)
        print(f"  Results saved: {yaml_file}")

        # Also save as latest (without timestamp) for easy access
        json_latest = os.path.join(save_dir, "hand_eye_calibration_result.json")
        yaml_latest = os.path.join(save_dir, "hand_eye_calibration_result.yaml")

        with open(json_latest, 'w') as f:
            json.dump(result, f, indent=4)

        with open(yaml_latest, 'w') as f:
            yaml.dump(result, f)

        print(f"  Latest results also saved to: hand_eye_calibration_result.json/yaml")

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def list_calibration_data(self):
        """List available calibration data directories

        Returns:
            list: List of data directory paths
        """
        print("\nAvailable calibration data:")

        data_dirs = []
        # 支持新格式（带模式）和旧格式（不带模式）
        valid_prefixes = (
            "manual_calibration_eyeinhand_", "manual_calibration_eyetohand_",
            "replay_calibration_eyeinhand_", "replay_calibration_eyetohand_",
            "manual_calibration_", "replay_calibration_"  # 兼容旧格式
        )

        # Search calibration_data directory
        if os.path.exists(self.calibration_data_dir):
            for item in os.listdir(self.calibration_data_dir):
                if item.startswith(valid_prefixes):
                    full_path = os.path.join(self.calibration_data_dir, item)
                    if os.path.isdir(full_path):
                        data_dirs.append(full_path)

        # Search verified_data directory
        if os.path.exists(self.verified_data_dir):
            for item in os.listdir(self.verified_data_dir):
                if item.startswith(valid_prefixes):
                    full_path = os.path.join(self.verified_data_dir, item)
                    if os.path.isdir(full_path):
                        data_dirs.append(full_path)

        # Sort by timestamp (extract from directory name)
        # Format: manual_calibration_YYYYMMDD_HHMMSS or replay_calibration_YYYYMMDD_HHMMSS
        def extract_timestamp(dir_path):
            """Extract timestamp from directory name for sorting"""
            dir_name = os.path.basename(dir_path)
            # Split by underscore and get last two parts (date and time)
            parts = dir_name.split('_')
            if len(parts) >= 3:
                # Get YYYYMMDD_HHMMSS
                timestamp_str = '_'.join(parts[-2:])
                return timestamp_str
            return dir_name  # Fallback to full name

        data_dirs = sorted(data_dirs, key=extract_timestamp, reverse=True)  # Newest first

        if not data_dirs:
            print("  (None)")
            return []

        for i, dir_path in enumerate(data_dirs, 1):
            dir_name = os.path.basename(dir_path)

            # Check files
            data_file = os.path.join(dir_path, "calibration_data.json")
            poses_file = os.path.join(dir_path, "poses.txt")
            result_file = os.path.join(dir_path, "hand_eye_calibration_result.json")

            try:
                # Count poses and images
                pose_count = 0
                image_count = 0

                # Try new format first (calibration_data.json)
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    pose_count = len(data)
                    data_format = "New"
                # Fallback to legacy format (poses.txt)
                elif os.path.exists(poses_file):
                    with open(poses_file, 'r') as f:
                        lines = f.readlines()
                    # Count non-empty, non-comment lines
                    pose_count = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
                    data_format = "Legacy"
                else:
                    print(f"  {i}. {dir_name} (No calibration data)")
                    continue

                # Count images (*.png files)
                image_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
                image_count = len(image_files)

                # Check calibration status
                status = "✅ Calibrated" if os.path.exists(result_file) else "📝 Uncalibrated"

                # Display detailed information in one line
                print(f"  {i}. {dir_name} - Poses: {pose_count} | Images: {image_count} | {status}")

            except Exception as e:
                print(f"  {i}. {dir_name} (Error: {e})")

        return data_dirs


# ============================================================================
# Main Function - Interactive Menu
# ============================================================================

def main():
    """Main function - Interactive calibration computation"""
    print("="*60)
    print("Hand-Eye Calibration Computer v3.3.0")
    print("="*60)
    print("This tool performs calibration computation from collected data")
    print("No hardware required - pure computation")
    print("="*60)

    calibrator = HandEyeCalibrator()

    # List available data
    data_dirs = calibrator.list_calibration_data()

    if not data_dirs:
        print("\nNo calibration data found")
        print("Please run handeye_data_collect.py first to collect data")
        return

    # Select data to calibrate
    print("\nSelect data to calibrate:")
    choice = input("Enter number (or 'all' to calibrate all uncalibrated data): ").strip()

    # 选择标定模式
    print("\n标定模式:")
    print("  1. Eye-in-Hand (相机在末端)")
    print("  2. Eye-to-Hand (相机固定)")
    mode_choice = input("选择模式 (1/2): ").strip()

    if mode_choice == "1":
        calibration_mode = "eye_in_hand"
    elif mode_choice == "2":
        calibration_mode = "eye_to_hand"
    else:
        raise ValueError("❌ 输入错误：请选择 1 或 2")
    

    # 选择是否重新检测角点
    print("\n角点检测:")
    print("  1. 使用已保存的角点（快速）")
    print("  2. 重新从图像检测角点（更精确，但较慢）")
    redetect_choice = input("选择方式 (1/2) [默认 1]: ").strip()

    redetect_corners = (redetect_choice == "2")

    if choice.lower() == 'all':
        # Calibrate all uncalibrated data
        uncalibrated = []
        for dir_path in data_dirs:
            result_file = os.path.join(dir_path, "hand_eye_calibration_result.json")
            if not os.path.exists(result_file):
                uncalibrated.append(dir_path)

        if not uncalibrated:
            print("\nAll data already calibrated")
            return

        print(f"\nFound {len(uncalibrated)} uncalibrated datasets")
        confirm = input("Proceed to calibrate all? (y/n): ").strip().lower()

        if confirm != 'y':
            print("Cancelled")
            return

        for dir_path in uncalibrated:
            print(f"\n{'='*60}")
            print(f"Calibrating: {os.path.basename(dir_path)}")
            print(f"{'='*60}")

            # 使用指定模式或自动检测（在 calibrate 函数内部）
            R, t, quality = calibrator.calibrate(dir_path, calibration_mode=calibration_mode, save_results=True, redetect_corners=redetect_corners)

            if R is not None:
                print(f"✅ Success")
            else:
                print(f"❌ Failed")

    else:
        # Calibrate selected data
        try:
            index = int(choice) - 1
        except ValueError:
            print("Please enter a valid number or 'all'")
            return
            
        if 0 <= index < len(data_dirs):
            selected_dir = data_dirs[index]
            print(f"\nSelected: {os.path.basename(selected_dir)}")

            # 使用指定模式或自动检测（在 calibrate 函数内部）
            R, t, quality = calibrator.calibrate(selected_dir, calibration_mode=calibration_mode, save_results=True, redetect_corners=redetect_corners)

            if R is not None:
                print(f"\n✅ Calibration successful!")
            else:
                print(f"\n❌ Calibration failed")
        else:
            print("Invalid selection")


if __name__ == "__main__":
    main()
