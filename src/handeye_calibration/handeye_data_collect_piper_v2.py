#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handeye_data_collect.py - Hand-Eye Calibration Data Collection Module

Responsibilities:
- Connect to hardware (robot arm + RealSense camera)
- Manual data collection mode
- Replay trajectory collection mode
- Save collected data to disk

Design: Separation of Concerns
- This module ONLY handles data collection (hardware interaction)
- Calibration computation is in handeye_calibrate.py (no hardware dependency)
"""

import sys
import os

import cv2
import numpy as np
import json
import time
from datetime import datetime

# Robot control - using piper_api_v2
sys.path.insert(0, '/data/workspace/MobileManipulator2/src/piper_grasp/scripts')
from piper_api_v2 import PiperAPI

# Camera intrinsics
from calibration_common import CameraIntrinsicsManager


class HandEyeDataCollector:
    """Hand-eye calibration data collector

    Handles all hardware interactions for data collection
    """

    VERSION = "5.0.0"  # V2 API + Joint control replay (100% reproducible)

    def __init__(self, config_dir=None, mode='eye_in_hand'):
        """Initialize data collector

        Args:
            config_dir: Optional, config directory path
            mode: 'eye_in_hand' or 'eye_to_hand'
        """
        self.calibration_mode = mode

        print(f"HandEyeDataCollector v{self.VERSION} ({mode.upper().replace('_', '-')})")
        print("="*60)

        if mode == 'eye_to_hand':
            print("📷 模式: Eye-to-Hand")
            print("   - 相机固定在基座附近")
            print("   - 棋盘格固定在机械臂末端")
            print("⚠️  请确保:")
            print("   1. 相机已固定在基座附近")
            print("   2. 棋盘格已固定在机械臂末端（使用示教模式固定）")
            print("="*60)
        else:
            print("📷 模式: Eye-in-Hand")
            print("   - 相机固定在机械臂末端")
            print("   - 棋盘格固定在外部")
            print("="*60)

        # Hardware handles
        self.arm = None  # PiperAPI instance
        self.pipeline = None
        self.pipeline_started = False

        # Directory configuration
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_data_dir = os.path.join(self.script_dir, "calibration_data")
        self.verified_data_dir = os.path.join(self.script_dir, "verified_data")
        self.config_dir = config_dir or os.path.join(self.script_dir, "config")

        os.makedirs(self.calibration_data_dir, exist_ok=True)
        os.makedirs(self.verified_data_dir, exist_ok=True)

        # Robot configuration
        self.initial_position = [300, 0, 300, 180, 60, 180]  # Safe initial position

        # Chessboard parameters
        self.board_size = (11, 8)  # 6x4 chessboard
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.chessboard_size_mm = 30.0  # 50mm per square

        # Camera configuration (根据模式选择相机)
        if mode == 'eye_to_hand':
            self.camera_id = "318122302992"  # RealSense top camera (fixed)
        else:
            self.camera_id = "337122071190"  # RealSense hand camera (on gripper)

        self.camera_matrix = None
        self.dist_coeffs = None

        # Motion configuration for stability
        self.motion_config = {
            'warmup_speed': 40,          # Warmup speed
            'normal_speed': 25,          # Normal motion speed
            'capture_speed': 20,         # Capture speed (minimize vibration)
            'stability_wait': 5.0,       # Stability wait time
            'extra_settle_time': 2.0,    # Extra settle time
            'warmup_duration': 900       # Warmup duration (15 min)
        }

    # ========================================================================
    # Hardware Connection
    # ========================================================================

    def connect_devices(self):
        """连接机器臂和相机"""
        try:
            import pyrealsense2 as rs

            # 初始化机器臂 (using PiperAPI v2)
            print("🔧 初始化机器臂...")
            self.arm = PiperAPI(can_name="can0")
            self.arm.connect()

            # 移动到初始位置
            print(f"\n📍 移动到初始位置: {self.initial_position[:3]}")
            self.arm.set_position(
                x=self.initial_position[0], y=self.initial_position[1], z=self.initial_position[2],
                roll=self.initial_position[3], pitch=self.initial_position[4], yaw=self.initial_position[5],
                wait=True, speed=self.motion_config['normal_speed'], use_gripper_center=False
            )

            # 关闭爪子
            print("🤏 关闭爪子...")
            self.arm.set_gripper_position(0, wait=True, speed=1000)

            # 初始化相机
            camera_name = "顶部相机" if self.calibration_mode == 'eye_to_hand' else "手眼相机"
            print(f"📹 初始化{camera_name} (ID: {self.camera_id})...")
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_device(self.camera_id)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

            # Start pipeline without custom queue (simpler, more reliable)
            profile = self.pipeline.start(config)
            self.pipeline_started = True

            # Warm up camera FIRST: discard first 30 frames to ensure stability
            print("⏳ 相机预热中...")
            for _ in range(30):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=1000)
                except:
                    pass

            # Get camera intrinsics with proper fallback
            # Strategy 1: Try from config file
            self.camera_matrix, self.dist_coeffs, source = \
                CameraIntrinsicsManager.get_camera_intrinsics(
                    config_dir=self.config_dir,
                    camera_id=None  # Don't pass camera_id to avoid double pipeline
                )

            # Strategy 2: If failed, get from already-started pipeline
            if self.camera_matrix is None:
                print("Config file not found, extracting from running pipeline...")
                try:
                    color_stream = profile.get_stream(rs.stream.color)
                    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

                    self.camera_matrix = np.array([
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]
                    ], dtype=np.float32)

                    self.dist_coeffs = np.array(intrinsics.coeffs[:5], dtype=np.float32)

                    source = "RealSense (from current pipeline)"
                    print(f"✅ 成功从pipeline获取内参")
                    print(f"   Focal: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")

                    # Save to config for future use
                    config_file = os.path.join(self.config_dir, "hand_camera_intrinsics.yaml")
                    CameraIntrinsicsManager.save_to_file(
                        self.camera_matrix, self.dist_coeffs, config_file
                    )
                except Exception as e:
                    print(f"❌ 无法获取相机内参: {e}")
                    return False

            print(f"✅ 内参来源: {source}")

            # Final validation
            if self.camera_matrix is None:
                print("❌ 相机内参加载失败")
                return False

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

        if self.arm:
            try:
                self.arm.disconnect()  # PiperAPI v2 uses safe disconnect by default
                print("🤖 机器臂已断开")
            except:
                pass

    # ========================================================================
    # Image Capture and Chessboard Detection
    # ========================================================================

    def reset_camera_pipeline(self):
        """Reset camera pipeline if stuck

        Returns:
            bool: True if reset successful
        """
        try:
            import pyrealsense2 as rs

            print("  Resetting camera pipeline...")

            # Stop existing pipeline
            if self.pipeline_started:
                self.pipeline.stop()
                self.pipeline_started = False

            time.sleep(1)

            # Restart pipeline (use same FPS as main pipeline!)
            config = rs.config()
            config.enable_device(self.camera_id)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            self.pipeline.start(config)
            self.pipeline_started = True

            # Warm up: discard first 10 frames
            for _ in range(10):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=1000)
                except:
                    pass

            print("  Camera pipeline reset complete")
            return True

        except Exception as e:
            print(f"  Error resetting camera: {e}")
            return False

    def capture_image(self, retries=3):
        """Capture image from camera with retry mechanism

        Args:
            retries: Number of retry attempts (default: 3)

        Returns:
            np.ndarray: Captured color image or None
        """
        if not self.pipeline_started:
            print("Error: Camera not started")
            return None

        import pyrealsense2 as rs

        # Buffer cleanup: poll 5 times to discard old frames
        # Typical queue holds 3-4 frames, polling 5 times gets us close to latest
        # Delay: <100ms (vs poll×1: ~150ms)
        try:
            for _ in range(5):
                self.pipeline.poll_for_frames()
        except:
            pass  # Queue may become empty, that's fine

        for attempt in range(retries):
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()

                if not color_frame:
                    print("Warning: No color frame received")
                    if attempt < retries - 1:
                        print(f"  Retry {attempt+1}/{retries-1}...")
                        time.sleep(0.5)
                        continue
                    return None

                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                return color_image

            except Exception as e:
                print(f"Error capturing image: {e}")

                if attempt < retries - 1:
                    print(f"  Retry {attempt+1}/{retries-1}...")
                    time.sleep(0.5)
                elif attempt == retries - 1:
                    # Last attempt: try resetting pipeline
                    print("  All retries failed, attempting pipeline reset...")
                    if self.reset_camera_pipeline():
                        # One more try after reset
                        try:
                            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                            color_frame = frames.get_color_frame()
                            if color_frame:
                                return np.asanyarray(color_frame.get_data())
                        except:
                            pass
        return None

    def detect_chessboard(self, image, visualize=False):
        """Detect chessboard in image

        Args:
            image: Color image
            visualize: Whether to display detection result

        Returns:
            tuple: (success, corners) or (False, None)
        """
        if image is None:
            return False, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

        if ret:
            # Refine corner positions
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            if visualize:
                # Draw and display
                img_with_corners = image.copy()
                cv2.drawChessboardCorners(img_with_corners, self.board_size, corners, ret)
                cv2.imshow('Chessboard Detection', img_with_corners)
                cv2.waitKey(100)

        return ret, corners

    # ========================================================================
    # Manual Data Collection Mode
    # ========================================================================

    def manual_collection_mode(self):
        """Manual data collection mode with real-time visualization

        User manually moves robot arm to different poses, system captures data
        """
        print("\n" + "="*60)
        print("Manual Data Collection Mode")
        print("="*60)
        print("Instructions:")
        print("  - Manually move robot arm to different poses")
        print("  - Ensure chessboard is visible to camera")
        print("  - Press SPACE to capture at current pose")
        print("  - Press ESC to finish collection")
        print("  - Press Q to quit without saving")
        print("="*60)

        collected_data = []
        frame_count = 0

        try:
            while True:
                # Capture image
                image = self.capture_image()
                if image is None:
                    print("Warning: Failed to capture image")
                    time.sleep(0.1)
                    continue

                display_image = image.copy()

                # Detect chessboard
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, self.board_size,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
                )

                if ret:
                    # Refine corners
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(display_image, self.board_size, corners, ret)
                    cv2.putText(display_image, "Chessboard: DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    cv2.putText(display_image, "Chessboard: NOT DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Display collected count
                cv2.putText(display_image, f"Collected: {frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Display instructions
                cv2.putText(display_image, "[SPACE] Save  [ESC] Finish  [Q] Quit",
                           (10, display_image.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show image
                cv2.imshow("Manual Calibration", display_image)

                # Key handling
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # SPACE: Save
                    if not ret:
                        print("\n⚠️  未检测到棋盘格，建议调整位置后再保存")
                        print("   是否仍要保存此帧? (按 SPACE 确认，其他键取消)")
                        confirm_key = cv2.waitKey(0) & 0xFF
                        if confirm_key != ord(' '):
                            print("   已取消保存")
                            continue

                    # Get current robot pose and joint angles
                    _, current_pose = self.arm.get_position(return_gripper_center=False)
                    if current_pose is None:
                        print("\n⚠️  无法获取机器臂位姿")
                        continue

                    current_pose = current_pose if isinstance(current_pose, list) else current_pose.tolist()
                    current_joints = self.arm.get_joint_degrees()  # Get joint angles

                    # Convert to meters and radians
                    x_m = current_pose[0] / 1000.0
                    y_m = current_pose[1] / 1000.0
                    z_m = current_pose[2] / 1000.0
                    roll_rad = np.radians(current_pose[3])
                    pitch_rad = np.radians(current_pose[4])
                    yaw_rad = np.radians(current_pose[5])

                    current_pose_m_rad = [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]

                    frame_count += 1

                    # Save data (including joint angles)
                    collected_data.append({
                        'frame_id': frame_count,
                        'pose': current_pose_m_rad,
                        'joints': current_joints,  # Save joint angles in degrees
                        'corners': corners.reshape(-1, 2).tolist() if ret else None,
                        'image': image
                    })

                    print(f"\n✅ 已采集第 {frame_count} 帧")
                    print(f"   位姿: X={current_pose[0]:.1f}mm Y={current_pose[1]:.1f}mm Z={current_pose[2]:.1f}mm")
                    print(f"   棋盘格: {'检测成功' if ret else '未检测到'}")

                elif key == 27:  # ESC: Finish
                    print("\n📊 数据采集完成")
                    break

                elif key == ord('q') or key == ord('Q'):  # Q: Quit
                    print("\n❌ 用户取消采集")
                    cv2.destroyAllWindows()

            cv2.destroyAllWindows()

            # Save collected data
            if collected_data:
                print(f"\n收集了 {len(collected_data)} 帧数据")
                return self._save_collected_data(collected_data, mode="manual")
            else:
                print("\n没有采集到数据")
                return None

        except KeyboardInterrupt:
            print("\n\n被用户中断")
            cv2.destroyAllWindows()
            return None

    # ========================================================================
    # Motion Control Methods
    # ========================================================================

    def _move_with_joint_control(self, joints_deg, speed=20, timeout=8.0):
        """Move robot using joint control (100% reproducible)

        Args:
            joints_deg: List of 6 joint angles in degrees [j1, j2, j3, j4, j5, j6]
            speed: Motion speed (1-100)
            timeout: Timeout for reaching target

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to SDK units
            joints_sdk = [int(deg * self.arm.FACTOR) for deg in joints_deg]

            # Execute joint control
            self.arm.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # Joint mode
            self.arm.piper.JointCtrl(joints_sdk[0], joints_sdk[1], joints_sdk[2],
                                     joints_sdk[3], joints_sdk[4], joints_sdk[5])

            # Wait for joints to reach target
            if not self.arm._wait_joint_reached(joints_sdk, timeout=timeout, tolerance_deg=2.0):
                actual_joints = self.arm.get_joint_degrees()
                print(f"  ⚠️  关节运动超时")
                print(f"     目标: [{', '.join([f'{j:.1f}' for j in joints_deg])}]°")
                print(f"     实际: [{', '.join([f'{j:.1f}' for j in actual_joints])}]°")
                return False

            print(f"  ✅ 关节控制成功")
            return True

        except Exception as e:
            print(f"  ❌ 关节控制异常: {e}")
            return False

    def _move_with_pose_control(self, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, speed=20):
        """Move robot using pose control (fallback for old data without joint angles)

        Args:
            x_mm, y_mm, z_mm: Target position in mm
            roll_deg, pitch_deg, yaw_deg: Target orientation in degrees
            speed: Motion speed

        Returns:
            bool: True if successful (even with error), False if failed
        """
        try:
            # Use SDK set_position
            self.arm.set_position(
                x=x_mm, y=y_mm, z=z_mm,
                roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
                wait=True, speed=speed, use_gripper_center=False
            )

            # Check actual position
            _, actual = self.arm.get_position(return_gripper_center=False)
            actual = actual if isinstance(actual, list) else actual.tolist()

            pos_error = max(
                abs(actual[0] - x_mm),
                abs(actual[1] - y_mm),
                abs(actual[2] - z_mm)
            )

            if pos_error <= 50.0:
                print(f"  ✅ 位姿控制成功 (误差: {pos_error:.1f}mm)")
                return True
            else:
                print(f"  ⚠️  位姿控制误差较大: {pos_error:.1f}mm（接受此位置）")
                return True  # Accept even with larger error

        except Exception as e:
            print(f"  ❌ 位姿控制异常: {e}")
            return False

    # ========================================================================
    # IK-based Motion (using Pinocchio) - DEPRECATED, kept for reference
    # ========================================================================

    def _init_ik_solver(self):
        """Initialize Pinocchio IK solver (lazy initialization)"""
        if hasattr(self, '_ik_solver') and self._ik_solver is not None:
            return True

        try:
            # DEPRECATED: Pinocchio IK not available on current system
            # Kept for reference only - joint control is the preferred method
            from piper_pinocchio import Arm_IK

            print("  🔧 初始化 Pinocchio IK 求解器...")
            self._ik_solver = Arm_IK(headless=True)
            print("  ✅ IK 求解器初始化成功")
            return True
        except Exception as e:
            print(f"  ❌ IK 求解器初始化失败: {e}")
            self._ik_solver = None
            return False

    def _move_with_ik(self, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg,
                      speed=20, error_threshold=50.0):
        """Move to target position with SDK first, then IK fallback if failed

        Strategy:
        1. Try SDK set_position first
        2. If position error > threshold, try Pinocchio IK + joint control
        3. Return result (no further retries)

        Args:
            x_mm, y_mm, z_mm: Target position in mm (flange frame)
            roll_deg, pitch_deg, yaw_deg: Target orientation in degrees
            speed: Motion speed
            error_threshold: Position error threshold (mm) to trigger IK fallback

        Returns:
            bool: True if position reached within threshold, False otherwise
        """
        # Step 1: Try SDK set_position
        self.arm.set_position(
            x=x_mm, y=y_mm, z=z_mm,
            roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
            wait=True, speed=speed, use_gripper_center=False
        )

        # Check actual position
        _, actual = self.arm.get_position(return_gripper_center=False)
        actual = actual if isinstance(actual, list) else actual.tolist()

        pos_error = [
            abs(actual[0] - x_mm),
            abs(actual[1] - y_mm),
            abs(actual[2] - z_mm)
        ]
        max_error = max(pos_error)

        # Success: within threshold
        if max_error <= error_threshold:
            return True

        # Step 2: SDK failed, try Pinocchio IK
        print(f"  ⚠️  SDK 位置偏差过大: {max_error:.1f}mm，尝试 IK 方案...")

        if not self._init_ik_solver():
            print(f"  ❌ IK 求解器初始化失败，跳过此位置")
            return False

        try:
            import numpy as np
            import math
            import pinocchio as pin

            # Quaternion from Euler helper
            def quaternion_from_euler(roll, pitch, yaw):
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                cp = math.cos(pitch * 0.5)
                sp = math.sin(pitch * 0.5)
                cr = math.cos(roll * 0.5)
                sr = math.sin(roll * 0.5)
                w = cr * cp * cy + sr * sp * sy
                x = sr * cp * cy - cr * sp * sy
                y = cr * sp * cy + sr * cp * sy
                z = cr * cp * sy - sr * sp * cy
                return [x, y, z, w]

            # Convert to meters and radians for IK
            x_m = x_mm / 1000.0
            y_m = y_mm / 1000.0
            z_m = z_mm / 1000.0
            roll_rad = math.radians(roll_deg)
            pitch_rad = math.radians(pitch_deg)
            yaw_rad = math.radians(yaw_deg)

            # Create target pose as 4x4 homogeneous matrix
            q = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
            target_pose = pin.SE3(
                pin.Quaternion(q[3], q[0], q[1], q[2]),
                np.array([x_m, y_m, z_m]),
            )

            # Get current joint angles as initial guess
            current_joints = self.arm.get_joint_degrees()
            init_q = [math.radians(j) for j in current_joints]

            # Solve IK
            sol_q, tau_ff, success = self._ik_solver.ik_fun(
                target_pose.homogeneous,
                gripper=0,
                motorstate=np.array(init_q)
            )

            if not success:
                print(f"  ❌ IK 求解失败，跳过此位置")
                return False

            # Convert solution to degrees for logging
            sol_deg = [math.degrees(q) for q in sol_q]
            print(f"  ✅ IK 求解成功: [{', '.join([f'{d:.1f}' for d in sol_deg])}]°")

            # Convert to SDK units (0.001 degrees as integer)
            sol_sdk = [int(d * self.arm.FACTOR) for d in sol_deg]

            # Get current joint state before move
            current_joints_deg = self.arm.get_joint_degrees()
            print(f"  当前关节: [{', '.join([f'{d:.1f}' for d in current_joints_deg])}]°")

            # Move using joint control (SDK units)
            self.arm.piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)  # Joint mode
            self.arm.piper.JointCtrl(sol_sdk[0], sol_sdk[1], sol_sdk[2], sol_sdk[3], sol_sdk[4], sol_sdk[5])

            # Wait for joints to reach target (relaxed tolerance for boundary joints)
            joint_reached = self.arm._wait_joint_reached(sol_sdk, timeout=8.0, tolerance_deg=5.0)

            # Read actual joint angles after movement
            actual_joints_deg = self.arm.get_joint_degrees()

            if not joint_reached:
                print(f"  ⚠️  关节运动超时（容差5°）")
                print(f"     目标: [{', '.join([f'{d:.1f}' for d in sol_deg])}]°")
                print(f"     实际: [{', '.join([f'{d:.1f}' for d in actual_joints_deg])}]°")
                return False

            # Joint reached target, IK solution accepted
            print(f"  ✅ IK 关节控制成功")
            print(f"     目标: [{', '.join([f'{d:.1f}' for d in sol_deg])}]°")
            print(f"     实际: [{', '.join([f'{d:.1f}' for d in actual_joints_deg])}]°")
            time.sleep(1.0)  # Extra settle time
            return True  # Success - actual position will be read by get_position() later

        except Exception as e:
            print(f"  ❌ IK 异常: {e}，跳过此位置")
            return False

    # ========================================================================
    # Replay Trajectory Collection Mode
    # ========================================================================

    def replay_trajectory_collection(self, trajectory_file, return_to_zero=True):
        """Replay trajectory and collect data

        Args:
            trajectory_file: Path to trajectory file (poses.txt)
            return_to_zero: Whether to return to zero between poses (default True for stability)

        Returns:
            str: Saved data directory or None
        """
        print("\n" + "="*60)
        print("Replay Trajectory Collection Mode")
        print("="*60)

        if return_to_zero:
            print("Mode: High precision - return to zero between poses")
        else:
            print("Mode: Fast - continuous movement")

        # Load trajectory
        trajectory = self._load_trajectory(trajectory_file)
        if not trajectory:
            print("Failed to load trajectory")
            return None

        # Check if using joint control or pose control
        use_joint_control = trajectory[0]['joints'] is not None
        if use_joint_control:
            print("✅ Using joint control (100% reproducible)")
        else:
            print("⚠️  Using pose control (fallback for old data)")

        # Create save directory FIRST (save images immediately, not in memory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 目录名格式：replay_calibration_eyeinhand_*, replay_calibration_eyetohand_*
        calibration_mode_short = self.calibration_mode.replace('_', '')  # eyeinhand / eyetohand
        save_dir = os.path.join(
            self.calibration_data_dir,
            f"replay_calibration_{calibration_mode_short}_{timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)
        print(f"Data will be saved to: {save_dir}")

        collected_data = []  # Only store metadata, NOT images

        try:
            for idx, traj_point in enumerate(trajectory):
                print(f"\n[{idx+1}/{len(trajectory)}] Moving to pose...")

                # Unpack trajectory point
                pose_data = traj_point['pose']
                joints_data = traj_point['joints']

                # Unpack pose (in meters and radians from file)
                x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad = pose_data

                # Convert to mm and degrees (robot API units)
                x_mm = x_m * 1000.0
                y_mm = y_m * 1000.0
                z_mm = z_m * 1000.0
                roll_deg = np.degrees(roll_rad)
                pitch_deg = np.degrees(pitch_rad)
                yaw_deg = np.degrees(yaw_rad)

                print(f"  Target: X={x_mm:.1f}mm Y={y_mm:.1f}mm Z={z_mm:.1f}mm")
                print(f"          Roll={roll_deg:.1f}° Pitch={pitch_deg:.1f}° Yaw={yaw_deg:.1f}°")

                # Optional: Return to zero first (for high precision)
                if return_to_zero and idx > 0:
                    print("  Returning to zero position...")
                    self.arm.go_zero()
                    time.sleep(self.motion_config['stability_wait'])

                # Move robot to target
                move_success = False

                if use_joint_control and joints_data is not None:
                    # Use joint control (new method, 100% reproducible)
                    print(f"  Joints: [{', '.join([f'{j:.1f}' for j in joints_data])}]°")
                    move_success = self._move_with_joint_control(joints_data, speed=self.motion_config['capture_speed'])
                else:
                    # Fallback to pose control (old method)
                    move_success = self._move_with_pose_control(
                        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg,
                        speed=self.motion_config['capture_speed']
                    )

                if not move_success:
                    print(f"  ❌ 跳过第 {idx+1} 帧（无法到达目标位置）")
                    continue

                # Wait for stability
                print(f"  Waiting {self.motion_config['stability_wait']}s for stability...")
                time.sleep(self.motion_config['stability_wait'])

                # Get ACTUAL arrived position and joints (not target, for accuracy)
                _, actual_pose = self.arm.get_position(return_gripper_center=False)
                actual_pose = actual_pose if isinstance(actual_pose, list) else actual_pose.tolist()
                actual_joints = self.arm.get_joint_degrees()  # Get actual joint angles

                # Display Actual position for diagnostics
                print(f"  Actual: X={actual_pose[0]:.1f}mm Y={actual_pose[1]:.1f}mm Z={actual_pose[2]:.1f}mm")

                # Convert actual position to meters and radians (for calibration)
                actual_pose_m_rad = [
                    actual_pose[0] / 1000.0,  # mm to m
                    actual_pose[1] / 1000.0,
                    actual_pose[2] / 1000.0,
                    np.radians(actual_pose[3]),  # deg to rad
                    np.radians(actual_pose[4]),
                    np.radians(actual_pose[5])
                ]

                # Check position error (for logging only)
                pos_error = [
                    abs(actual_pose[0] - x_mm),
                    abs(actual_pose[1] - y_mm),
                    abs(actual_pose[2] - z_mm)
                ]
                max_error = max(pos_error)

                if max_error > 2.0:
                    print(f"  ⚠️  Position error {max_error:.2f}mm > 2mm")
                else:
                    print(f"  ✅ Arrived at target (error < 2mm)")

                # Capture image
                image = self.capture_image()
                if image is None:
                    print("  Warning: Failed to capture image")
                    continue

                display_image = image.copy()

                # Detect chessboard
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, self.board_size,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
                )

                if ret:
                    # Refine corners
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(display_image, self.board_size, corners, ret)
                    cv2.putText(display_image, "DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    cv2.putText(display_image, "NOT DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # Display progress
                cv2.putText(display_image, f"Replay: {idx+1}/{len(trajectory)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_image, "[SPACE] Pause  [ESC] Stop  [Q] Quit",
                           (10, display_image.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show image
                cv2.imshow("Replay Mode", display_image)
                cv2.waitKey(1)

                # CRITICAL: Save image to disk IMMEDIATELY (not in memory)
                frame_id = idx + 1
                image_file = os.path.join(save_dir, f"{frame_id}_Color.png")
                cv2.imwrite(image_file, image)

                # Free memory immediately
                del image
                del display_image

                # Save metadata only (no image in memory)
                data_entry = {
                    'frame_id': frame_id,
                    'pose': actual_pose_m_rad,
                    'joints': actual_joints,  # Save actual joint angles
                    'chessboard_detected': ret
                }

                if ret:
                    # Add corners if chessboard detected
                    data_entry['corners'] = corners.reshape(-1, 2).tolist()
                    collected_data.append(data_entry)
                    print(f"  ✅ 已采集第 {idx+1} 帧数据 (图像已保存)")
                    print(f"  📷 棋盘格: 检测成功")
                else:
                    # Still save metadata even without chessboard
                    collected_data.append(data_entry)
                    print(f"  ⚠️  已采集第 {idx+1} 帧数据（棋盘格未检测到，图像已保存）")

            cv2.destroyAllWindows()

            # Save metadata (images already saved incrementally)
            if collected_data:
                print(f"\nCollected {len(collected_data)}/{len(trajectory)} frames")
                return self._save_metadata_only(collected_data, save_dir)
            else:
                print("\nNo data collected")
                return None

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            cv2.destroyAllWindows()
            return None

    def _load_trajectory(self, trajectory_file):
        """Load trajectory from file (supports both old and new format)

        Args:
            trajectory_file: Path to poses.txt

        Returns:
            list: List of dicts with 'pose' and optional 'joints'
                  pose: [x, y, z, roll, pitch, yaw] in meters and radians
                  joints: [j1, j2, j3, j4, j5, j6] in degrees (if available)
        """
        if not os.path.exists(trajectory_file):
            print(f"Error: Trajectory file not found: {trajectory_file}")
            return []

        trajectory = []

        try:
            with open(trajectory_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Parse pose
                    # Old format (7 fields): frame_id roll pitch yaw x y z
                    # New format (13 fields): frame_id roll pitch yaw x y z j1 j2 j3 j4 j5 j6
                    parts = line.split()

                    if len(parts) not in [7, 13]:
                        print(f"Warning: Invalid line format (expected 7 or 13 fields): {line}")
                        continue

                    frame_id = int(parts[0])
                    roll_rad = float(parts[1])
                    pitch_rad = float(parts[2])
                    yaw_rad = float(parts[3])
                    x_m = float(parts[4])
                    y_m = float(parts[5])
                    z_m = float(parts[6])

                    # pose as [x, y, z, roll, pitch, yaw]
                    pose = [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]

                    # Check if joint angles available (new format)
                    joints = None
                    if len(parts) == 13:
                        joints = [
                            float(parts[7]),   # j1
                            float(parts[8]),   # j2
                            float(parts[9]),   # j3
                            float(parts[10]),  # j4
                            float(parts[11]),  # j5
                            float(parts[12])   # j6
                        ]

                    trajectory.append({
                        'pose': pose,
                        'joints': joints
                    })

            if not trajectory:
                print("Error: No valid poses found in trajectory file")
            else:
                has_joints = sum(1 for t in trajectory if t['joints'] is not None)
                if has_joints > 0:
                    print(f"✅ Loaded {len(trajectory)} poses with joint angles (new format)")
                else:
                    print(f"✅ Loaded {len(trajectory)} poses without joint angles (old format)")

            return trajectory

        except Exception as e:
            print(f"Error loading trajectory: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ========================================================================
    # Data Saving
    # ========================================================================

    def _save_metadata_only(self, collected_data, save_dir):
        """Save metadata only (images already saved incrementally)

        Args:
            collected_data: List of collected metadata (NO images)
            save_dir: Directory where images were already saved

        Returns:
            str: Save directory path
        """
        # 保存标定模式元数据
        timestamp = os.path.basename(save_dir).split('_')[-2] + '_' + os.path.basename(save_dir).split('_')[-1]
        metadata = {
            'calibration_mode': self.calibration_mode,
            'collection_mode': 'replay',
            'camera_id': self.camera_id,
            'timestamp': timestamp,
            'version': self.VERSION
        }

        metadata_file = os.path.join(save_dir, "calibration_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save calibration data (JSON)
        data_file = os.path.join(save_dir, "calibration_data.json")
        with open(data_file, 'w') as f:
            json.dump(collected_data, f, indent=4)

        print(f"\n✅ Metadata saved to: {save_dir}")
        print(f"  - calibration_data.json: {len(collected_data)} frames")

        # Count saved images
        image_count = len([f for f in os.listdir(save_dir) if f.endswith('_Color.png')])
        print(f"  - Images: {image_count} files")

        # Save trajectory (poses.txt) for replay
        timestamp = os.path.basename(save_dir).split('_')[-2] + '_' + os.path.basename(save_dir).split('_')[-1]
        poses_file = os.path.join(save_dir, "poses.txt")
        with open(poses_file, 'w') as f:
            f.write("# Trajectory for replay - Robot arm end-effector poses\n")
            f.write("# Format: frame_id roll(rad) pitch(rad) yaw(rad) x(m) y(m) z(m) j1(deg) j2(deg) j3(deg) j4(deg) j5(deg) j6(deg)\n")
            f.write(f"# Collection time: {timestamp}\n")
            f.write(f"# Collection mode: replay\n")
            f.write("#" + "-"*70 + "\n")
            for data in collected_data:
                frame_id = data['frame_id']
                pose = data['pose']  # [x, y, z, roll, pitch, yaw] in meters and radians
                joints = data.get('joints', None)  # Joint angles in degrees

                # Write pose
                f.write(f"{frame_id} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} "
                       f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f}")

                # Write joint angles if available
                if joints is not None:
                    f.write(f" {joints[0]:.2f} {joints[1]:.2f} {joints[2]:.2f} "
                           f"{joints[3]:.2f} {joints[4]:.2f} {joints[5]:.2f}")

                f.write("\n")

        print(f"  - poses.txt: Trajectory for replay")

        # Save camera intrinsics
        if self.camera_matrix is not None:
            intrinsics_file = os.path.join(save_dir, "realsense_intrinsics.yaml")
            CameraIntrinsicsManager.save_to_file(
                self.camera_matrix,
                self.dist_coeffs,
                intrinsics_file
            )
            print(f"  - realsense_intrinsics.yaml: Camera intrinsics")

        return save_dir

    def _save_collected_data(self, collected_data, mode="manual"):
        """Save collected data to disk

        Args:
            collected_data: List of collected data
            mode: Collection mode ("manual" or "replay")

        Returns:
            str: Saved data directory path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 目录名格式：manual_calibration_eyeinhand_*, replay_calibration_eyetohand_*
        calibration_mode_short = self.calibration_mode.replace('_', '')  # eyeinhand / eyetohand
        save_dir = os.path.join(
            self.calibration_data_dir,
            f"{mode}_calibration_{calibration_mode_short}_{timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # 保存标定模式元数据
        metadata = {
            'calibration_mode': self.calibration_mode,
            'collection_mode': mode,
            'camera_id': self.camera_id,
            'timestamp': timestamp,
            'version': self.VERSION
        }

        metadata_file = os.path.join(save_dir, "calibration_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save images FIRST (before JSON serialization)
        image_count = 0
        for data in collected_data:
            if 'image' in data and data['image'] is not None:
                frame_id = data['frame_id']
                image_file = os.path.join(save_dir, f"{frame_id}_Color.png")
                cv2.imwrite(image_file, data['image'])
                image_count += 1
                # Remove image from data to avoid JSON serialization issues
                data.pop('image', None)

        # Save calibration data (JSON) - now safe, images already removed
        data_file = os.path.join(save_dir, "calibration_data.json")
        with open(data_file, 'w') as f:
            json.dump(collected_data, f, indent=4)

        print(f"\nData saved to: {save_dir}")
        print(f"  - calibration_data.json: {len(collected_data)} frames")
        if image_count > 0:
            print(f"  - Images: {image_count} files saved")

        # Save trajectory (poses.txt) for replay
        # Format: includes joint angles for precise replay
        poses_file = os.path.join(save_dir, "poses.txt")
        with open(poses_file, 'w') as f:
            f.write("# Trajectory for replay - Robot arm end-effector poses\n")
            f.write("# Format: frame_id roll(rad) pitch(rad) yaw(rad) x(m) y(m) z(m) j1(deg) j2(deg) j3(deg) j4(deg) j5(deg) j6(deg)\n")
            f.write(f"# Collection time: {timestamp}\n")
            f.write(f"# Collection mode: {mode}\n")
            f.write("#" + "-"*70 + "\n")
            for data in collected_data:
                frame_id = data['frame_id']
                pose = data['pose']  # [x, y, z, roll, pitch, yaw] in meters and radians
                joints = data.get('joints', None)  # Joint angles in degrees (may not exist in old data)

                # Write pose
                f.write(f"{frame_id} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} "
                       f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f}")

                # Write joint angles if available
                if joints is not None:
                    f.write(f" {joints[0]:.2f} {joints[1]:.2f} {joints[2]:.2f} "
                           f"{joints[3]:.2f} {joints[4]:.2f} {joints[5]:.2f}")

                f.write("\n")

        print(f"  - poses.txt: Trajectory for replay")

        # Save camera intrinsics
        if self.camera_matrix is not None:
            intrinsics_file = os.path.join(save_dir, "realsense_intrinsics.yaml")
            CameraIntrinsicsManager.save_to_file(
                self.camera_matrix,
                self.dist_coeffs,
                intrinsics_file
            )
            print(f"  - realsense_intrinsics.yaml: Camera intrinsics")

        return save_dir

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def list_historical_data(self):
        """List available historical calibration data

        Returns:
            list: List of data directory paths
        """
        print("\nAvailable historical calibration data:")

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

        data_dirs = sorted(data_dirs)

        if not data_dirs:
            print("  (None)")
            return []

        for i, dir_path in enumerate(data_dirs, 1):
            dir_name = os.path.basename(dir_path)

            # Check if has calibration data
            data_file = os.path.join(dir_path, "calibration_data.json")
            poses_file = os.path.join(dir_path, "poses.txt")

            if os.path.exists(data_file):
                # Count frames
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        frame_count = len(data)
                    print(f"  {i}. {dir_name} ({frame_count} frames)")
                except:
                    print(f"  {i}. {dir_name}")
            elif os.path.exists(poses_file):
                # Count poses
                pose_count = 0
                with open(poses_file, 'r') as f:
                    for line in f:
                        if not line.startswith('#') and line.strip():
                            pose_count += 1
                print(f"  {i}. {dir_name} ({pose_count} poses)")
            else:
                print(f"  {i}. {dir_name}")

        return data_dirs


# ============================================================================
# Main Function - Interactive Menu
# ============================================================================

def main():
    """Main function - Interactive data collection"""
    print("="*60)
    print("Hand-Eye Calibration Data Collector v3.3.0")
    print("="*60)

    # 选择标定模式
    print("\n标定模式:")
    print("  1. Eye-in-Hand (相机在末端，棋盘格固定)")
    print("  2. Eye-to-Hand (相机固定，棋盘格在末端)")
    mode_choice = input("选择模式 (1/2) [默认 1]: ").strip()

    if mode_choice == "2":
        calibration_mode = "eye_to_hand"
    else:
        calibration_mode = "eye_in_hand"

    print("\n采集模式:")
    print("  1. Manual collection - Manually move robot and capture")
    print("  2. Replay trajectory - Replay saved trajectory and capture")
    print("  3. List historical data")
    print("="*60)

    mode = input("\nSelect mode (1/2/3) [default 1]: ").strip()
    if not mode:
        mode = "1"

    collector = HandEyeDataCollector(mode=calibration_mode)

    try:
        if mode == "1":
            # Manual collection mode
            print("\n" + "="*60)
            print("Manual Collection Mode")
            print("="*60)

            if collector.connect_devices():
                print("\nDevices connected successfully!")
                save_dir = collector.manual_collection_mode()

                if save_dir:
                    print(f"\nData collection completed!")
                    print(f"Data saved to: {save_dir}")
                    print(f"\nNext step: Run handeye_calibrate.py to compute calibration")
            else:
                print("Failed to connect devices")

        elif mode == "2":
            # Replay trajectory mode
            print("\n" + "="*60)
            print("Replay Trajectory Mode")
            print("="*60)

            # List historical data
            data_dirs = collector.list_historical_data()

            if not data_dirs:
                print("\nNo historical data found")
                print("Please run manual collection mode first")
                return

            # Select trajectory
            choice = input("\nSelect trajectory to replay (enter number): ").strip()
            try:
                index = int(choice) - 1
                if 0 <= index < len(data_dirs):
                    selected_dir = data_dirs[index]
                    trajectory_file = os.path.join(selected_dir, "poses.txt")

                    if not os.path.exists(trajectory_file):
                        print(f"\nError: No trajectory file found in {selected_dir}")
                        return

                    print(f"\nSelected: {os.path.basename(selected_dir)}")

                    if collector.connect_devices():
                        print("\nDevices connected successfully!")
                        save_dir = collector.replay_trajectory_collection(trajectory_file)

                        if save_dir:
                            print(f"\nData collection completed!")
                            print(f"Data saved to: {save_dir}")
                            print(f"\nNext step: Run handeye_calibrate.py to compute calibration")
                    else:
                        print("Failed to connect devices")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")

        elif mode == "3":
            # List historical data
            collector.list_historical_data()

        else:
            print("Invalid mode")

    finally:
        collector.disconnect_devices()


if __name__ == "__main__":
    main()
