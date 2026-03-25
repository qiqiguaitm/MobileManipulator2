#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
PiperAPI V2 - Refactored following official SDK V2 examples

Key improvements over original:
1. Safety-first design: clean_error() won't cause arm to fall by default
2. Simplified code following official V2 demo patterns
3. Removed over-engineered damped_stop (100 lines -> 10 lines)
4. Kept useful features: gripper center coordinate transform

Reference: /data1/workspace/piper_sdk_demo/V2/
"""

import time
import math
import numpy as np
from typing import Optional, List, Tuple
from scipy.spatial.transform import Rotation as R

from piper_sdk import C_PiperInterface_V2


class PiperAPI:
    """
    Piper API V2 - Safety-first, following official SDK patterns

    Units (all interfaces use human-readable units):
        - Position: mm (interface) <-> 0.001mm (SDK internally)
        - Angle: degrees (interface) <-> 0.001deg (SDK internally)
        - Gripper: mm (interface) <-> 0.001mm (SDK internally)

    All unit conversions are handled internally. Callers always use mm/degrees.
    """

    FACTOR = 1000  # mm/deg to 0.001mm/0.001deg (SDK units)

    # Natural hanging position (measured with arm disabled)
    # This is where the arm naturally rests under gravity
    # Key: Joint 5 = 30.4° (NOT zero!)
    NATURAL_HANG_DEG = [0, -2.2, 0.2, 0.7, 30.4, 0]

    # Default ready position (gripper_center coordinates)
    DEFAULT_READY_POS = {
        'x': 400,
        'y': 0,
        'z': 150,
        'roll': 180,
        'pitch': 30,
        'yaw': 180
    }

    # Pre-computed joint configurations for Z-axis trajectory at X=400mm, pitch=30°
    # These bypass IK singularities by using known-good configurations
    # Keys are Z in mm, values are joint angles in degrees [j1, j2, j3, j4, j5, j6]
    # Note: These are SDK-convention angles (not URDF)
    Z_KEYPOINT_CONFIGS = {
        200:  [0.0,  79.4, -37.1, 0.0,  52.7, 0.0],
        150:  [0.0,  80.7, -27.2, 0.0,  41.4, 0.0],
        100:  [0.0,  84.1, -19.2, 0.0,  30.0, 0.0],
        50:   [0.0,  89.4, -13.0, 0.0,  18.6, 0.0],
        0:    [0.0,  96.7,  -8.8, 0.0,   7.1, 0.0],
        -50:  [0.0, 105.5,  -6.7, 0.0,  -3.8, 0.0],
        -100: [0.0, 115.5,  -6.8, 0.0, -13.7, 0.0],
        -150: [0.0, 126.1,  -9.1, 0.0, -22.0, 0.0],
        -200: [0.0, 136.7, -13.5, 0.0, -28.2, 0.0],
        -250: [0.0, 147.1, -19.8, 0.0, -32.3, 0.0],
    }

    def __init__(self,
                 can_name: str = "can0",
                 gripper_max_mm: float = 90.0,
                 gripper_offset_mm: float = 135.03):
        """
        Initialize PiperAPI

        Args:
            can_name: CAN bus device name
            gripper_max_mm: Maximum gripper opening (mm), clamped to 90mm
            gripper_offset_mm: Flange to gripper center offset (mm)
        """
        self.can_name = can_name
        self.piper = None
        self.is_connected = False

        # Gripper config (official V2 uses 50mm, we allow up to 90mm)
        self.gripper_max_mm = min(gripper_max_mm, 90.0)
        self.gripper_max_units = int(self.gripper_max_mm * self.FACTOR)

        # Coordinate transform
        self.GRIPPER_OFFSET_MM = gripper_offset_mm

        # State tracking
        self._last_gripper_position = 0
        self._last_valid_pose = [0, 0, 0, 0, 0, 0]

        print(f"[PiperAPI] Initialized: {can_name}, gripper: 0-{self.gripper_max_mm}mm, "
              f"offset: {gripper_offset_mm}mm")

    # ==================== Connection ====================

    def connect(self):
        """
        Connect to Piper arm - following official pattern
        """
        try:
            print(f"[PiperAPI] Connecting to {self.can_name}...")

            # Step 1: Create interface
            self.piper = C_PiperInterface_V2(self.can_name)

            # Step 2: Connect port
            self.piper.ConnectPort()
            time.sleep(0.1)

            # Mark as connected (needed for motion_enable to work)
            self.is_connected = True

            # Step 3: Enable using official enable_fun() pattern
            if not self.motion_enable(enable=True):
                self.is_connected = False
                raise RuntimeError("Enable timeout")

            # Step 4: Initialize gripper
            self._init_gripper()
            print("[PiperAPI] ✓ Connected successfully")

        except Exception as e:
            print(f"[PiperAPI] ✗ Connection failed: {e}")
            self.is_connected = False
            raise

    def _get_enable_status_official(self) -> list:
        """
        Get enable status using official V2 method
        (6 individual motor status checks, exactly like piper_enable.py)
        """
        try:
            msg = self.piper.GetArmLowSpdInfoMsgs()
            return [
                msg.motor_1.foc_status.driver_enable_status,
                msg.motor_2.foc_status.driver_enable_status,
                msg.motor_3.foc_status.driver_enable_status,
                msg.motor_4.foc_status.driver_enable_status,
                msg.motor_5.foc_status.driver_enable_status,
                msg.motor_6.foc_status.driver_enable_status,
            ]
        except Exception as e:
            print(f"[PiperAPI] ⚠ Cannot read enable status: {e}")
            return [False] * 6

    def _init_gripper(self):
        """Initialize gripper (official pattern)"""
        self.piper.GripperCtrl(0, 1000, 0x02, 0)  # Reset
        time.sleep(0.1)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)  # Enable
        time.sleep(0.1)

    def disconnect(self, safe: bool = True):
        """
        Disconnect from Piper arm

        Args:
            safe: If True, slowly return to zero before disable (default)
                  If False, immediate disable (arm will fall!)
        """
        if self.piper and self.is_connected:
            try:
                if safe:
                    self._safe_disable()
                else:
                    print("[PiperAPI] ⚠ 直接 disable (机械臂会掉落!)")
                    self.piper.DisableArm(7)
                    self.piper.GripperCtrl(0, 1000, 0x02, 0)

                self.piper.DisconnectPort()
                self.is_connected = False
                print("[PiperAPI] ✓ Disconnected")
            except Exception as e:
                print(f"[PiperAPI] ⚠ Disconnect warning: {e}")

    # ==================== Motion Control ====================

    def motion_enable(self, enable: bool = True, go_zero: bool = False) -> bool:
        """
        Enable/disable motion

        Args:
            enable: True to enable, False to disable (will damped_stop first!)
            go_zero: If True, go to zero position after enabling

        Returns:
            True if operation successful, False if timeout
        """
        if not self.is_connected:
            return False

        if not enable:
            # DISABLE: Use damped_stop to prevent falling
            print("[PiperAPI] motion_enable(False) -> damped_stop first")
            self._damped_stop()
            # damped_stop already calls DisableArm, just verify
            enable_list = self._get_enable_status_official()
            if not any(enable_list):
                print("[PiperAPI] ✓ Motion disabled (via damped_stop)")
                return True
            else:
                print(f"[PiperAPI] ⚠ Disable incomplete: {enable_list}")
                return False

        # ENABLE: Official enable_fun() logic
        enable_flag = False
        loop_flag = False
        timeout = 5
        start_time = time.time()

        while not loop_flag:
            elapsed_time = time.time() - start_time
            print("--------------------")

            enable_list = self._get_enable_status_official()
            enable_flag = all(enable_list)

            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)

            print(f"[PiperAPI] Enable status: {enable_flag} ({enable_list})")
            print("--------------------")

            if enable_flag:
                loop_flag = True

            if elapsed_time > timeout:
                print("[PiperAPI] ⚠ Timeout...")
                enable_flag = False
                loop_flag = True
                break

            time.sleep(0.5)

        print(f"[PiperAPI] Returning: {enable_flag}")

        if enable_flag and go_zero:
            self._go_zero()

        if not enable_flag:
            print("[PiperAPI] ⚠ Enable failed!")
        else:
            print("[PiperAPI] ✓ Enabled")

        return enable_flag

    def set_position(self,
                     x: float = None, y: float = None, z: float = None,
                     roll: float = None, pitch: float = None, yaw: float = None,
                     speed: int = 30, wait: bool = True,
                     use_gripper_center: bool = True,
                     linear: bool = False,
                     timeout: float = 5.0,
                     tolerance: int = 5000) -> bool:
        """
        Set end effector position

        Args:
            x, y, z: Position in mm (None = keep current)
            roll, pitch, yaw: Orientation in degrees (None = keep current)
            speed: Motion speed 1-100%
            wait: Wait for motion to complete
            use_gripper_center: If True, coordinates are gripper center (default)
                               If False, coordinates are flange
            linear: If True, use MoveL (linear interpolation in Cartesian space)
                   If False, use MoveJ (joint interpolation, curved path) - default

        Returns:
            True if position reached (when wait=True), else True

        Note:
            MoveL (linear=True): End effector moves in a straight line.
                Use for: approaching objects, descending to grasp, precise placement.
            MoveJ (linear=False): Each joint interpolates independently, curved path.
                Use for: fast repositioning, obstacle-free movement.
        """
        if not self.is_connected:
            return False

        try:
            # Get current pose
            current = self._get_current_pose()  # In SDK units (0.001mm/0.001deg)

            # Build target in mm/deg
            if use_gripper_center:
                # Get current gripper center position
                _, gc_pos = self.get_position(return_gripper_center=True)
                target_gc = [
                    x if x is not None else gc_pos[0],
                    y if y is not None else gc_pos[1],
                    z if z is not None else gc_pos[2],
                    roll if roll is not None else gc_pos[3],
                    pitch if pitch is not None else gc_pos[4],
                    yaw if yaw is not None else gc_pos[5],
                ]
                # Convert to flange coordinates
                target = self._gripper_center_to_flange(target_gc)
            else:
                # Direct flange coordinates
                target = [
                    int((x if x is not None else current[0] / self.FACTOR) * self.FACTOR),
                    int((y if y is not None else current[1] / self.FACTOR) * self.FACTOR),
                    int((z if z is not None else current[2] / self.FACTOR) * self.FACTOR),
                    int((roll if roll is not None else current[3] / self.FACTOR) * self.FACTOR),
                    int((pitch if pitch is not None else current[4] / self.FACTOR) * self.FACTOR),
                    int((yaw if yaw is not None else current[5] / self.FACTOR) * self.FACTOR),
                ]

            # Clamp speed
            safe_speed = max(1, min(100, speed))

            # Select motion mode: MoveL (0x02) for linear, MoveJ (0x00) for curved
            move_mode = 0x02 if linear else 0x00

            # MoveL speed limit: linear motion is more sensitive, cap at 50%
            if linear and safe_speed > 50:
                print(f"[PiperAPI] MoveL speed capped: {safe_speed}% -> 50%")
                safe_speed = 50

            # Log motion mode for debugging
            mode_name = "MoveL" if linear else "MoveJ"
            # print(f"[PiperAPI] {mode_name} to target, speed={safe_speed}%")

            # Send command (official pattern: MotionCtrl_2 + EndPoseCtrl)
            self.piper.MotionCtrl_2(0x01, move_mode, safe_speed, 0x00)  # Cartesian mode
            self.piper.EndPoseCtrl(*target)

            if wait:
                return self._wait_position_reached(target, timeout=timeout, speed=safe_speed,
                                                   move_mode=move_mode, tolerance=tolerance)
            else:
                time.sleep(0.1)
                return True

        except Exception as e:
            print(f"[PiperAPI] ✗ Set position failed: {e}")
            return False

    def _is_can_ok(self) -> bool:
        """Check CAN receive thread health via SDK's built-in FPS monitor."""
        try:
            return self.piper.isOk()
        except Exception:
            return False

    def _verify_at_target(self, target_gc: dict, tolerance: float = 30.0) -> Tuple[bool, float]:
        """Verify arm is at target position (gripper center).
        Returns (at_target, distance_mm).
        """
        _, gc = self.get_position(return_gripper_center=True)
        tx, ty, tz = target_gc.get('x', 0), target_gc.get('y', 0), target_gc.get('z', 0)
        dist = math.sqrt((gc[0]-tx)**2 + (gc[1]-ty)**2 + (gc[2]-tz)**2)
        return dist <= tolerance, dist

    def _wait_position_reached(self, target, timeout: float = 5.0,
                                tolerance: int = 5000, speed: int = 30,
                                move_mode: int = 0x00) -> bool:
        """Wait for arm to reach target using dual-track detection.

        Uses GetArmStatus().motion_status (CAN 0x2A1 Byte4):
            0x00 = reached target position
            0x01 = not yet reached

        Detection strategy (dual-track, parallel):
            1. motion_status transition: track 0x01→0x00, require 2 consecutive
               0x00 reads to filter stale CAN data from previous command
            2. Position check: independent of motion_status, runs after grace
               period — catches firmware bugs where motion_status stays 0x01
            Either path returning True is sufficient.

        Safety:
            - Error detection: arm_status 0x02/0x03/0x04, joint limit bits
            - Resend once at 1s if firmware never started moving
            - CAN thread health check via SDK isOk()

        Args:
            target: Target position in SDK units
            timeout: Maximum wait time in seconds
            tolerance: Position tolerance in SDK units (fallback check)
            speed: Motion speed 1-100%
            move_mode: Motion mode (0x00=MoveJ, 0x02=MoveL)
        """
        start = time.time()
        resent = False
        seen_moving = False  # True once we observe motion_status == 0x01
        reached_count = 0  # consecutive 0x00 reads after seen_moving
        REACHED_CONFIRM = 2  # require N consecutive 0x00 to confirm arrival
        ERROR_GRACE = 0.3  # ignore transient TARGET_LIMIT right after command
        RESEND_DEADLINE = 1.0  # firmware mode switch takes ~100ms; 1s is safe margin
        STALE_DEADLINE = 2.0  # check CAN health after this

        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                break

            status = self.piper.GetArmStatus()
            arm_st = status.arm_status.arm_status
            motion_st = status.arm_status.motion_status

            # Error detection — fail fast
            if arm_st == 0x04 and elapsed >= ERROR_GRACE:  # TARGET_POS_EXCEEDS_LIMIT
                print(f"[PiperAPI] ✗ Target limit (t={elapsed:.2f}s)")
                return False
            if arm_st in (0x02, 0x03):  # NO_SOLUTION, SINGULARITY
                print(f"[PiperAPI] ✗ Arm error: 0x{int(arm_st):02X} (t={elapsed:.2f}s)")
                return False
            if status.arm_status.err_code & 0x3F00:  # Joint limit bits
                joints = [f"J{i+1}" for i in range(6)
                          if status.arm_status.err_code & (1 << (8 + i))]
                print(f"[PiperAPI] ✗ Joint limit: {','.join(joints)}")
                return False

            if motion_st == 0x01:
                seen_moving = True
                reached_count = 0  # reset: arm still moving

            # Reached target: require N consecutive 0x00 reads after seen_moving
            if motion_st == 0x00 and seen_moving:
                reached_count += 1
                if reached_count >= REACHED_CONFIRM:
                    return True

            # Position check: independent of motion_status
            if elapsed > ERROR_GRACE:
                current = self._get_current_pose()
                if all(abs(current[i] - target[i]) < tolerance for i in range(3)):
                    return True

            # CAN thread dead → break early instead of wasting timeout
            if elapsed > STALE_DEADLINE and not self._is_can_ok():
                print(f"[PiperAPI] CAN thread dead, breaking early (t={elapsed:.2f}s)")
                break

            # Command lost / mode-switch drop: resend once after deadline
            if elapsed > RESEND_DEADLINE and not resent and not seen_moving:
                print(f"[PiperAPI] ⚠ No motion after {RESEND_DEADLINE}s, resending command")
                self.piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
                self.piper.EndPoseCtrl(*target)
                resent = True

            time.sleep(0.05)

        # Timeout — log position for diagnosis
        try:
            current = self._get_current_pose()
            diff = [abs(current[i] - target[i]) / self.FACTOR for i in range(3)]
            print(f"[PiperAPI] ✗ Timeout ({timeout}s), motion_status=0x{int(motion_st):02X}, "
                  f"seen_moving={seen_moving}, can_ok={self._is_can_ok()}, "
                  f"diff=[{diff[0]:.1f},{diff[1]:.1f},{diff[2]:.1f}]mm")
        except Exception:
            print(f"[PiperAPI] ✗ Timeout ({timeout}s), motion_status=0x{int(motion_st):02X}, "
                  f"seen_moving={seen_moving}")
        return False

    def move_z_segmented(self, target_z: float, max_step: float = 60,
                         speed: int = 20, use_gripper_center: bool = True,
                         lock_xy: bool = True) -> bool:
        """
        Move Z axis in segments to approximate straight-line motion with MoveJ.

        This solves the dilemma:
        - MoveL may fail (retract) near singularities
        - MoveJ over long distances causes curved paths (collision risk)

        By using small MoveJ segments with XY locked, we get:
        - Reliability of MoveJ (no singularity issues)
        - Straight vertical path (XY locked at each segment)

        Args:
            target_z: Target Z position in mm
            max_step: Maximum Z change per segment (mm), default 60mm
            speed: Motion speed 1-100%
            use_gripper_center: Use gripper center coordinates
            lock_xy: If True, explicitly set XY at each segment to prevent drift

        Returns:
            True if target reached, False otherwise
        """
        if not self.is_connected:
            return False

        _, current_pos = self.get_position(return_gripper_center=use_gripper_center)
        current_z = current_pos[2]
        # Lock XY and orientation from initial state.
        # Orientation MUST be locked: roll near ±180° causes sign-flip between segments
        # (179.9° → -179.9°), and firmware interprets the flip as a 360° J4 rotation.
        locked_x = current_pos[0]
        locked_y = current_pos[1]
        locked_roll  = current_pos[3]
        locked_pitch = current_pos[4]
        locked_yaw   = current_pos[5]
        delta_z = target_z - current_z

        # Per-segment timeout: shorter than global 10s because each segment is small.
        # Arm completes a 60mm step in ~3s at 20% speed. Post-timeout (3s) confirms if main
        # loop missed due to CAN resend cache stall. Total ~5s reliable per segment.
        # Tolerance 8mm (flange): handles pitch-induced coordinate conversion offset
        # where GC error ~5mm maps to flange error ~5.5mm, just over 5mm strict threshold.
        seg_timeout = 5.0
        seg_tolerance = 8000  # 8mm in SDK units (0.001mm)

        # If distance is small, direct move
        if abs(delta_z) <= max_step:
            if lock_xy:
                return self.set_position(x=locked_x, y=locked_y, z=target_z,
                                         roll=locked_roll, pitch=locked_pitch, yaw=locked_yaw,
                                         speed=speed, use_gripper_center=use_gripper_center,
                                         linear=False, timeout=seg_timeout, tolerance=seg_tolerance)
            else:
                return self.set_position(z=target_z, speed=speed,
                                         use_gripper_center=use_gripper_center, linear=False,
                                         timeout=seg_timeout, tolerance=seg_tolerance)

        # Calculate segments
        num_segments = int(abs(delta_z) / max_step) + 1
        step_size = delta_z / num_segments

        print(f"[PiperAPI] Segmented Z move: {current_z:.1f} -> {target_z:.1f}mm "
              f"({num_segments} segments, step={abs(step_size):.1f}mm, XY locked={lock_xy})")

        for i in range(num_segments):
            if i == num_segments - 1:
                # Last segment: go exactly to target
                segment_z = target_z
            else:
                segment_z = current_z + step_size * (i + 1)

            # Use MoveJ for each segment, with XY and orientation locked to prevent drift
            if lock_xy:
                success = self.set_position(x=locked_x, y=locked_y, z=segment_z,
                                            roll=locked_roll, pitch=locked_pitch, yaw=locked_yaw,
                                            speed=speed, use_gripper_center=use_gripper_center,
                                            linear=False, timeout=seg_timeout, tolerance=seg_tolerance)
            else:
                success = self.set_position(z=segment_z, speed=speed,
                                            use_gripper_center=use_gripper_center, linear=False,
                                            timeout=seg_timeout, tolerance=seg_tolerance)

            if not success:
                print(f"[PiperAPI] ✗ Segment {i+1}/{num_segments} failed at Z={segment_z:.1f}mm")
                return False

        return True

    def _angle_diff(self, a1, a2) -> int:
        """Compute angle difference with wrap-around (in 0.001deg units)"""
        diff = abs(a1 - a2) % (360 * self.FACTOR)
        if diff > 180 * self.FACTOR:
            diff = 360 * self.FACTOR - diff
        return diff

    def get_position(self, return_gripper_center: bool = True) -> Tuple[int, List[float]]:
        """
        Get current position

        Args:
            return_gripper_center: Return gripper center position (default: True)

        Returns:
            (status_code, [x, y, z, roll, pitch, yaw])
            Position in mm, angles in degrees
        """
        if not self.is_connected:
            return -1, [0, 0, 0, 0, 0, 0]

        # 重试机制：最多重试3次获取最新位置
        last_pos = None
        for retry in range(3):
            try:
                msg = self.piper.GetArmEndPoseMsgs()

                # Flange position
                pos = [
                    msg.end_pose.X_axis / self.FACTOR,
                    msg.end_pose.Y_axis / self.FACTOR,
                    msg.end_pose.Z_axis / self.FACTOR,
                    msg.end_pose.RX_axis / self.FACTOR,
                    msg.end_pose.RY_axis / self.FACTOR,
                    msg.end_pose.RZ_axis / self.FACTOR,
                ]

                # 如果位置变化了，说明获取到了新数据
                if last_pos is not None:
                    diff = sum(abs(pos[i] - last_pos[i]) for i in range(3))
                    if diff > 0.5:  # 位置有变化，使用新数据
                        last_pos = pos
                        break

                last_pos = pos
                if retry < 2:
                    time.sleep(0.01)  # 短暂等待后重试
            except Exception:
                break

        if last_pos is None:
            return -1, [0, 0, 0, 0, 0, 0]

        pos = last_pos

        if return_gripper_center:
            pos = self._flange_to_gripper_center(pos)

        return 0, pos

    # ==================== Coordinate Transform ====================

    def _flange_to_gripper_center(self, flange_pos: List[float]) -> List[float]:
        """Convert flange position to gripper center position"""
        # Rotation matrix from euler angles (xyz extrinsic)
        angles_rad = [math.radians(a) for a in flange_pos[3:6]]
        R_mat = R.from_euler('xyz', angles_rad).as_matrix()

        # Offset in base frame
        offset = R_mat @ np.array([0, 0, self.GRIPPER_OFFSET_MM])

        return [
            flange_pos[0] + offset[0],
            flange_pos[1] + offset[1],
            flange_pos[2] + offset[2],
            flange_pos[3],
            flange_pos[4],
            flange_pos[5],
        ]

    def _gripper_center_to_flange(self, gc_pos: List[float]) -> List[int]:
        """Convert gripper center position to flange position (SDK units)"""
        # Rotation matrix from euler angles
        angles_rad = [math.radians(a) for a in gc_pos[3:6]]
        R_mat = R.from_euler('xyz', angles_rad).as_matrix()

        # Offset in base frame
        offset = R_mat @ np.array([0, 0, self.GRIPPER_OFFSET_MM])

        # Flange = gripper_center - offset
        return [
            int((gc_pos[0] - offset[0]) * self.FACTOR),
            int((gc_pos[1] - offset[1]) * self.FACTOR),
            int((gc_pos[2] - offset[2]) * self.FACTOR),
            int(gc_pos[3] * self.FACTOR),
            int(gc_pos[4] * self.FACTOR),
            int(gc_pos[5] * self.FACTOR),
        ]

    # ==================== Gripper ====================

    def set_gripper_position(self, pos_mm: float, speed: int = 1000, wait: bool = True):
        """
        Set gripper position

        Args:
            pos_mm: Position in mm (0 = closed, max = gripper_max_mm)
            speed: Gripper speed 1-1000
            wait: Wait for gripper to reach position
        """
        if not self.is_connected:
            return

        # Convert mm to SDK units (0.001mm)
        pos_units = int(pos_mm * self.FACTOR)
        safe_pos = max(0, min(self.gripper_max_units, abs(pos_units)))
        safe_speed = max(1, min(1000, speed))

        self.piper.GripperCtrl(safe_pos, safe_speed, 0x01, 0)
        self._last_gripper_position = safe_pos

        if wait:
            time.sleep(0.3)

    def get_gripper_position(self) -> Tuple[int, float]:
        """
        Get gripper position

        Returns:
            (status_code, position_in_mm)
        """
        if not self.is_connected:
            return -1, 0.0

        try:
            msg = self.piper.GetArmGripperMsgs()
            if msg and hasattr(msg, 'gripper_state'):
                pos_units = msg.gripper_state.grippers_angle
                self._last_gripper_position = pos_units
                # Convert SDK units (0.001mm) to mm
                return 0, pos_units / self.FACTOR
        except Exception as e:
            print(f"[PiperAPI] ⚠ Get gripper position error: {e}")

        # Return last known position in mm
        return 0, self._last_gripper_position / self.FACTOR

    # ==================== Safety Operations ====================

    def emergency_stop(self):
        """
        Emergency stop - safe, arm will NOT fall (controlled deceleration)
        Uses official MotionCtrl_1(0x01) pattern

        IMPORTANT: After stop, you must call resume() then enable TWICE
        to regain control (official V2 requirement)
        """
        if self.piper:
            self.piper.MotionCtrl_1(0x01, 0, 0)
            print("[PiperAPI] ⚠ Emergency stop executed")
            print("[PiperAPI] ⚠ To recover: call resume() then enable twice")

    def stop(self):
        """Alias for emergency_stop"""
        self.emergency_stop()

    def resume(self):
        """
        Resume from stop state - following official reset pattern

        Official V2 (piper_reset.py):
            1. MotionCtrl_1(0x02, 0, 0) - 恢复
            2. MotionCtrl_2(0, 0, 0, 0x00) - 位置速度模式
        """
        if self.piper:
            self.piper.MotionCtrl_1(0x02, 0, 0)  # Resume/Reset
            self.piper.MotionCtrl_2(0, 0, 0, 0x00)  # Set position-velocity mode
            print("[PiperAPI] ✓ Resumed (position-velocity mode)")

    def recover_from_stop(self):
        """
        Full recovery from stop state - following official V2 pattern

        Official requirement: after stop, need reset + enable TWICE
        """
        if not self.is_connected:
            return False

        print("[PiperAPI] Recovering from stop...")

        # Step 1: Resume/Reset
        self.resume()
        time.sleep(0.5)

        # Step 2: Enable twice (official requirement)
        for i in range(2):
            print(f"[PiperAPI] Enable attempt {i+1}/2...")
            timeout = 5.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                time.sleep(1.0)

                if all(self._get_enable_status_official()):
                    print(f"[PiperAPI] ✓ Enable {i+1}/2 successful")
                    break
            else:
                print(f"[PiperAPI] ⚠ Enable {i+1}/2 timeout")
                return False

            # Small delay between enable attempts
            if i == 0:
                time.sleep(0.5)

        print("[PiperAPI] ✓ Full recovery complete")
        return True

    def clean_error(self, go_zero: bool = True, force: bool = False,
                    allow_disable: bool = False) -> bool:
        """
        Two-tier error handler:
          Tier 1: JointConfig(7, clear_err=0xAE) — safe, arm holds position
          Tier 2: Manual intervention (ResetPiper cuts current, too dangerous)

        TARGET_LIMIT and COLLISION are safe to auto-clear (Tier 1):
        the arm holds position, only the planner rejected the target.

        EMERGENCY_STOP, NO_SOLUTION, SINGULARITY, JOINT_COMM_ERR,
        BRAKE_NOT_RELEASED require manual intervention (Tier 2).

        Returns:
            True if error cleared, False if manual intervention needed
        """
        if not self.is_connected:
            return False

        has_error, error_type = self._get_error_type()
        if not has_error and not force:
            return True

        error_info = self._get_error_info()
        print(f"[PiperAPI] ❌ Error detected: {error_type} ({error_info})")

        # Tier 1: safe auto-clear for planner rejections
        TIER1_ERRORS = ("TARGET_LIMIT", "OTHER:0x04", "COLLISION")
        if error_type in TIER1_ERRORS or error_type.startswith("ERR_CODE:"):
            print(f"[PiperAPI] Tier1: clearing error via JointConfig...")
            self.piper.JointConfig(7, 0x00, 0x00, 500, 0xAE)  # clear_err=0xAE
            time.sleep(0.3)  # firmware needs time to process

            has_error_after, error_type_after = self._get_error_type()
            if not has_error_after:
                print(f"[PiperAPI] ✓ Error cleared")
                return True
            else:
                print(f"[PiperAPI] ⚠ Error persists after Tier1: {error_type_after}")

        # Tier 2: manual intervention required
        print("[PiperAPI] ⚠ Arm will HOLD current position (no auto-recovery)")
        print("[PiperAPI] ⚠ Please manually drag arm to safe position, then re-enable")
        return False

    def clean_gripper_error(self):
        """Clear gripper error using 'Enable and clear error' (gripper_code=0x03)"""
        if not self.is_connected:
            return

        self.piper.GripperCtrl(0, 1000, 0x03, 0)  # Enable and clear error
        time.sleep(0.1)
        print("[PiperAPI] ✓ Gripper error cleared")

    def _is_in_error(self) -> bool:
        """Check if arm is in error state"""
        try:
            status = self.piper.GetArmStatus()
            return status.arm_status.arm_status != 0 or status.arm_status.err_code != 0
        except Exception as e:
            print(f"[PiperAPI] ⚠ Cannot check error state: {e}")
            return True

    def _get_error_type(self) -> Tuple[bool, str]:
        """
        Get detailed error type - Good Taste: return what you need, not just True/False

        Returns:
            (has_error, error_type):
                - (False, "NORMAL"): No error
                - (True, "TARGET_LIMIT"): Target position exceeds limit (planning error)
                - (True, "JOINT_LIMIT"): Physical joint limit (hardware limit)
                - (True, "OTHER"): Other errors
        """
        try:
            status = self.piper.GetArmStatus()

            # Check target position limit (planning error)
            if status.arm_status.arm_status == 0x04:  # TARGET_POS_EXCEEDS_LIMIT
                return (True, "TARGET_LIMIT")

            # Check physical joint limits (hardware limit)
            if status.arm_status.err_code & 0x3F00:  # bit[8:13] = joint limits
                limited_joints = []
                for i in range(6):
                    if status.arm_status.err_code & (1 << (8 + i)):
                        limited_joints.append(f"J{i+1}")
                return (True, f"JOINT_LIMIT:{','.join(limited_joints)}")

            # Check other errors
            if status.arm_status.arm_status != 0:
                return (True, f"OTHER:0x{status.arm_status.arm_status:02X}")

            if status.arm_status.err_code != 0:
                return (True, f"ERR_CODE:0x{status.arm_status.err_code:04X}")

            return (False, "NORMAL")

        except Exception as e:
            return (True, f"EXCEPTION:{e}")

    def _get_error_info(self) -> str:
        """Get error description"""
        try:
            status = self.piper.GetArmStatus()
            return f"arm_status={status.arm_status.arm_status}, err_code={status.arm_status.err_code}"
        except Exception as e:
            return f"Cannot read status: {e}"

    # ==================== Zero Position ====================

    def go_zero(self):
        """Go to zero position (public method)"""
        self._go_zero()

    def _go_zero(self) -> bool:
        """Go to zero position with active position verification.

        Returns:
            True if zero reached, False on timeout/error
        """
        if not self.is_connected:
            return False

        print("[PiperAPI] Moving to zero position...")

        # Official pattern: MotionCtrl_2 + JointCtrl
        self.piper.MotionCtrl_2(0x01, 0x01, 50, 0)  # Joint mode, 50% speed
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)

        # Close gripper
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self._last_gripper_position = 0

        # Active wait with joint angle verification
        ok = self._wait_joint_reached([0, 0, 0, 0, 0, 0], timeout=5.0, tolerance_deg=2.5)
        if ok:
            print("[PiperAPI] ✓ Zero position reached")
        else:
            print("[PiperAPI] ⚠ Zero position not confirmed (see above)")
        return ok

    def go_ready(self, ready_pos: dict = None, open_gripper: bool = True,
                 speed: int = 30) -> bool:
        """
        Move to ready position (gripper_center coordinates)

        Args:
            ready_pos: Ready position dict with keys: x, y, z, roll, pitch, yaw
                      If None, uses DEFAULT_READY_POS
            open_gripper: Whether to open gripper before moving (default True)
            speed: Movement speed 1-100% (default 30)

        Returns:
            True if position reached, False otherwise
        """
        if not self.is_connected:
            return False

        # Use default if not specified
        pos = ready_pos or self.DEFAULT_READY_POS

        print(f"[PiperAPI] Moving to ready position...")
        print(f"[PiperAPI]   Target: x={pos.get('x')}, y={pos.get('y')}, z={pos.get('z')}, "
              f"roll={pos.get('roll')}, pitch={pos.get('pitch')}, yaw={pos.get('yaw')}")

        try:
            # Open gripper first if requested
            if open_gripper:
                print("[PiperAPI]   Opening gripper...")
                self.set_gripper_position(self.gripper_max_mm, speed=500, wait=True)

            target_kw = dict(
                x=pos.get('x', self.DEFAULT_READY_POS['x']),
                y=pos.get('y', self.DEFAULT_READY_POS['y']),
                z=pos.get('z', self.DEFAULT_READY_POS['z']),
                roll=pos.get('roll', self.DEFAULT_READY_POS['roll']),
                pitch=pos.get('pitch', self.DEFAULT_READY_POS['pitch']),
                yaw=pos.get('yaw', self.DEFAULT_READY_POS['yaw']),
                wait=True,
                speed=speed,
                use_gripper_center=True,
            )
            result = self.set_position(**target_kw)

            if not result:
                # Step 1: Hardware error → fast fail
                has_err, err_type = self._get_error_type()
                if has_err:
                    print(f"[PiperAPI] go_ready: hardware error ({err_type})")
                else:
                    # Step 2: CAN stale → wait for arm to physically settle
                    if not self._is_can_ok():
                        print("[PiperAPI] go_ready: CAN feedback stale, waiting 3s for arm to settle")
                        time.sleep(3.0)

                    # Step 3: Verify position (works whether CAN was stale or not)
                    at_target, dist = self._verify_at_target(pos)
                    if at_target:
                        print(f"[PiperAPI] go_ready: arm at target (dist={dist:.1f}mm)")
                        result = True
                    else:
                        # Step 4: One retry
                        print(f"[PiperAPI] go_ready: dist={dist:.1f}mm, retrying once")
                        time.sleep(0.5)
                        result = self.set_position(**target_kw)
                        if not result:
                            at_target, dist = self._verify_at_target(pos)
                            if at_target:
                                print(f"[PiperAPI] go_ready: retry OK (dist={dist:.1f}mm)")
                                result = True

            if result:
                print("[PiperAPI] ✓ Ready position reached")
            else:
                print("[PiperAPI] ⚠ Ready position not fully reached")

            return result

        except Exception as e:
            print(f"[PiperAPI] ✗ go_ready failed: {e}")
            return False

    # ==================== Safe Disable (Damped) ====================

    def get_joint_degrees(self) -> List[float]:
        """
        Get current joint angles in degrees

        Returns:
            List of 6 joint angles [j1, j2, j3, j4, j5, j6] in degrees
        """
        msg = self.piper.GetArmJointMsgs()
        return [
            msg.joint_state.joint_1 / self.FACTOR,
            msg.joint_state.joint_2 / self.FACTOR,
            msg.joint_state.joint_3 / self.FACTOR,
            msg.joint_state.joint_4 / self.FACTOR,
            msg.joint_state.joint_5 / self.FACTOR,
            msg.joint_state.joint_6 / self.FACTOR,
        ]

    def _damped_stop(self, speed_percent: int = 16):
        """
        Damped stop: Move slowly to natural hanging position, then disable.

        Natural hang position [0, -2.2, 0.2, 0.7, 30.4, 0]° is where the arm
        naturally rests under gravity. By moving there before disable,
        the arm won't fall after DisableArm.

        Args:
            speed_percent: Movement speed 1-10% (default 2% for high damping)
        """
        if not self.is_connected:
            return

        print("[PiperAPI] ------ DAMPED STOP ------")

        try:
            # Current position
            joints_before = self.get_joint_degrees()
            print(f"[PiperAPI] Current:  [{', '.join(f'{j:6.1f}' for j in joints_before)}]°")
            print(f"[PiperAPI] Target:   [{', '.join(f'{j:6.1f}' for j in self.NATURAL_HANG_DEG)}]°")

            # Calculate distance
            diff = [abs(joints_before[i] - self.NATURAL_HANG_DEG[i]) for i in range(6)]
            print(f"[PiperAPI] Distance: max {max(diff):.1f}° (Joint 5 key)")

            # Convert to SDK units
            target = [int(d * self.FACTOR) for d in self.NATURAL_HANG_DEG]

            # Speed (1-10%)
            speed = max(1, min(10, speed_percent))
            print(f"[PiperAPI] Moving at {speed}% speed...")

            # Move to natural hang
            self.piper.MotionCtrl_2(0x01, 0x01, speed, 0)  # Joint mode
            self.piper.JointCtrl(*target)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)

            # Wait for movement (longer for slower speed)
            wait_time = max(10.0, 120.0 / speed)
            start_time = time.time()

            while time.time() - start_time < wait_time:
                joints = self.get_joint_degrees()
                diff_now = [abs(joints[i] - self.NATURAL_HANG_DEG[i]) for i in range(6)]
                max_diff = max(diff_now)

                elapsed = time.time() - start_time
                print(f"[PiperAPI] [{elapsed:4.1f}s] max_diff={max_diff:5.1f}° J5={joints[4]:6.1f}°")

                if max_diff < 2.5:
                    print("[PiperAPI] ✓ Reached natural hang position")
                    time.sleep(0.5)
                    break

                # Resend command
                self.piper.MotionCtrl_2(0x01, 0x01, speed, 0)
                self.piper.JointCtrl(*target)
                time.sleep(1.0)

            # Check if arm actually reached near natural hang before disabling
            joints_pre = self.get_joint_degrees()
            diff_pre = [abs(joints_pre[i] - self.NATURAL_HANG_DEG[i]) for i in range(6)]
            max_diff_pre = max(diff_pre)
            print(f"[PiperAPI] Before disable: [{', '.join(f'{j:6.1f}' for j in joints_pre)}]°")
            print(f"[PiperAPI] Distance to hang: {max_diff_pre:.1f}°")

            # Safety gate: refuse to disable if arm is far from natural hang
            SAFE_DISABLE_THRESHOLD = 15.0  # degrees
            if max_diff_pre > SAFE_DISABLE_THRESHOLD:
                print(f"[PiperAPI] ❌ Arm NOT at natural hang ({max_diff_pre:.1f}° > {SAFE_DISABLE_THRESHOLD}°)")
                print("[PiperAPI] ❌ Skipping DisableArm to prevent fall! Arm stays enabled.")
                print("[PiperAPI] ------ DAMPED STOP (ABORTED) ------")
                return

            # Disable
            print("[PiperAPI] DisableArm(7)...")
            self.piper.DisableArm(7)
            time.sleep(1.0)

            # Measure drift
            joints_post = self.get_joint_degrees()
            drift = [abs(joints_post[i] - joints_pre[i]) for i in range(6)]
            max_drift = max(drift)

            print(f"[PiperAPI] After disable:  [{', '.join(f'{j:6.1f}' for j in joints_post)}]°")
            print(f"[PiperAPI] Drift: {max_drift:.2f}° (should be < 3°)")

            if max_drift < 3.0:
                print("[PiperAPI] ✓ Damped stop successful")
            else:
                print(f"[PiperAPI] ⚠ Drift {max_drift:.1f}° - natural hang may need calibration")

            print("[PiperAPI] ------ DAMPED STOP ------")

        except Exception as e:
            print(f"[PiperAPI] ⚠ Damped stop error: {e}")

    def _safe_disable(self):
        """
        Safe disable: go_zero → damped_stop (natural hang) → disable

        Flow:
        1. Go to zero position (50% speed)
        2. Move to natural hang position (2% speed)
        3. DisableArm (at natural hang, no fall)
        """
        if not self.is_connected:
            return

        print("[PiperAPI] Safe disable sequence...")

        # Step 1: Go to zero first (faster)
        print("[PiperAPI] Step 1: Go to zero (50% speed)...")
        self.piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(3.0)

        # Step 2: Damped stop (slow move to natural hang, then disable)
        print("[PiperAPI] Step 2: Damped stop...")
        self._damped_stop(speed_percent=8)

        # Verify disable
        time.sleep(0.5)
        if not any(self._get_enable_status_official()):
            print("[PiperAPI] ✓ Safe disable complete")
        else:
            print("[PiperAPI] ⚠ Disable verification timeout")

    # ==================== Helper Methods ====================

    def _get_current_pose(self) -> List[int]:
        """Get current pose in SDK units (0.001mm/0.001deg)"""
        msg = self.piper.GetArmEndPoseMsgs()
        if msg is None or not hasattr(msg, 'end_pose'):
            if self._last_valid_pose:
                print("[PiperAPI] Using cached pose (CAN read failed)")
                return self._last_valid_pose
            raise RuntimeError("Cannot read pose")

        pose = [
            int(msg.end_pose.X_axis),
            int(msg.end_pose.Y_axis),
            int(msg.end_pose.Z_axis),
            int(msg.end_pose.RX_axis),
            int(msg.end_pose.RY_axis),
            int(msg.end_pose.RZ_axis),
        ]
        self._last_valid_pose = pose
        return pose

    # ==================== IK Trajectory Execution ====================

    # Move mode constants
    MOVE_MODE_MOVEJ = 0x00  # Joint interpolation (curved path)
    MOVE_MODE_JOINT = 0x01  # Direct joint control
    MOVE_MODE_MOVEL = 0x02  # Linear interpolation (straight line)

    def _switch_move_mode(self, mode: int, speed: int = 30):
        """
        Switch motion mode explicitly.

        Args:
            mode: Motion mode
                - MOVE_MODE_MOVEJ (0x00): Joint interpolation, curved path
                - MOVE_MODE_JOINT (0x01): Direct joint control
                - MOVE_MODE_MOVEL (0x02): Linear interpolation, straight line
            speed: Motion speed 1-100%

        Note:
            This sets the control mode via MotionCtrl_2. The actual motion
            command (EndPoseCtrl or JointCtrl) should follow.
        """
        if not self.is_connected:
            return

        safe_speed = max(1, min(100, speed))

        # ctrl_mode: 0x01 = Cartesian (for MoveJ/MoveL), 0x01 = Joint (for direct joint)
        if mode == self.MOVE_MODE_JOINT:
            ctrl_mode = 0x01
        else:
            ctrl_mode = 0x01  # Cartesian mode for MoveJ/MoveL

        self.piper.MotionCtrl_2(ctrl_mode, mode, safe_speed, 0x00)

    def execute_ik_trajectory(self,
                               joint_waypoints: List[List[float]],
                               speed: int = 20,
                               xy_tolerance_mm: float = 5.0,
                               lock_xy: Tuple[float, float] = None,
                               use_joint_mode: bool = True) -> Tuple[bool, str]:
        """
        Execute IK-planned trajectory with XY drift monitoring.

        This is the core method for safe straight-line Pick motion:
        - Takes joint angle waypoints from IK solver
        - Executes each waypoint sequentially
        - Monitors XY drift at each step
        - Stops immediately if XY drift exceeds tolerance

        Args:
            joint_waypoints: List of joint angle arrays [j1..j6] in radians
            speed: Motion speed 1-100%
            xy_tolerance_mm: Maximum allowed XY drift (mm), default 5mm
            lock_xy: Expected (x, y) position in mm. If None, uses first waypoint's XY.
            use_joint_mode: If True, use direct joint mode (more reliable)
                           If False, use MoveJ mode

        Returns:
            (success: bool, message: str)
            - success: True if all waypoints executed without XY drift
            - message: Status or error description

        Example:
            # IK solver returns waypoints in radians
            waypoints = [
                [0.1, 0.5, -0.3, 0.0, -0.8, 1.2],  # waypoint 1
                [0.1, 0.6, -0.4, 0.0, -0.9, 1.2],  # waypoint 2
                ...
            ]
            success, msg = arm.execute_ik_trajectory(waypoints, speed=15)
        """
        if not self.is_connected:
            return False, "Not connected"

        if not joint_waypoints or len(joint_waypoints) == 0:
            return False, "Empty waypoints"

        num_waypoints = len(joint_waypoints)
        print(f"[PiperAPI] Executing IK trajectory: {num_waypoints} waypoints, speed={speed}%")

        # Get initial XY position for drift monitoring
        _, current_pos = self.get_position(return_gripper_center=True)
        if lock_xy is None:
            lock_xy = (current_pos[0], current_pos[1])
        locked_x, locked_y = lock_xy
        print(f"[PiperAPI] XY locked at: ({locked_x:.1f}, {locked_y:.1f})mm, tolerance={xy_tolerance_mm}mm")

        # Execute each waypoint
        for i, waypoint in enumerate(joint_waypoints):
            # Convert radians to SDK units (0.001 degrees)
            joint_deg = [math.degrees(rad) for rad in waypoint]
            joint_sdk = [int(deg * self.FACTOR) for deg in joint_deg]

            # Execute motion
            if use_joint_mode:
                # Direct joint mode (more reliable near singularities)
                self.piper.MotionCtrl_2(0x01, self.MOVE_MODE_JOINT, speed, 0x00)
                self.piper.JointCtrl(*joint_sdk)
            else:
                # MoveJ mode
                self.piper.MotionCtrl_2(0x01, self.MOVE_MODE_MOVEJ, speed, 0x00)
                self.piper.JointCtrl(*joint_sdk)

            # Wait for waypoint to be reached
            if not self._wait_joint_reached(joint_sdk, timeout=3.0):
                return False, f"Waypoint {i+1}/{num_waypoints} timeout"

            # XY drift check
            _, pos_now = self.get_position(return_gripper_center=True)
            x_drift = abs(pos_now[0] - locked_x)
            y_drift = abs(pos_now[1] - locked_y)
            xy_drift = math.sqrt(x_drift**2 + y_drift**2)

            if xy_drift > xy_tolerance_mm:
                error_msg = (f"XY drift exceeded at waypoint {i+1}/{num_waypoints}: "
                            f"drift={xy_drift:.1f}mm > {xy_tolerance_mm}mm")
                print(f"[PiperAPI] ✗ {error_msg}")
                return False, error_msg

            # Progress log (every 5 waypoints or last)
            if (i + 1) % 5 == 0 or i == num_waypoints - 1:
                print(f"[PiperAPI] Waypoint {i+1}/{num_waypoints}: "
                      f"Z={pos_now[2]:.1f}mm, XY_drift={xy_drift:.2f}mm")

        print(f"[PiperAPI] ✓ IK trajectory complete ({num_waypoints} waypoints)")
        return True, f"Completed {num_waypoints} waypoints"

    def _wait_joint_reached(self, target_sdk: List[int], timeout: float = 3.0,
                            tolerance_deg: float = 2.0) -> bool:
        """Wait for joint angles to reach target using dual-track detection.

        Detection strategy (mirrors _wait_position_reached):
            1. motion_status 0x01→0x00 transition: trust immediately
            2. Joint angle check: independent of motion_status, runs after
               grace period — catches firmware bugs where motion_status
               stays 0x01

        Args:
            target_sdk: Target joint angles in SDK units (0.001deg)
            timeout: Maximum wait time
            tolerance_deg: Position tolerance in degrees

        Returns:
            True if target reached, False if timeout/error
        """
        tolerance_sdk = int(tolerance_deg * self.FACTOR)
        start = time.time()
        seen_moving = False
        resent = False
        ERROR_GRACE = 0.3
        RESEND_DEADLINE = 1.0

        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                break

            status = self.piper.GetArmStatus()
            arm_st = status.arm_status.arm_status
            motion_st = status.arm_status.motion_status

            # Error detection — fail fast
            if arm_st == 0x04 and elapsed >= ERROR_GRACE:
                print(f"[PiperAPI] ✗ Joint target limit (t={elapsed:.2f}s)")
                return False
            if arm_st in (0x02, 0x03):
                print(f"[PiperAPI] ✗ Arm error: 0x{int(arm_st):02X} (t={elapsed:.2f}s)")
                return False
            if status.arm_status.err_code & 0x3F00:
                joints = [f"J{i+1}" for i in range(6)
                          if status.arm_status.err_code & (1 << (8 + i))]
                print(f"[PiperAPI] ✗ Joint limit: {','.join(joints)}")
                return False

            if motion_st == 0x01:
                seen_moving = True

            # Reached: motion_status 0x01→0x00 transition
            if motion_st == 0x00 and seen_moving:
                return True

            # Joint angle check: independent of motion_status
            if elapsed > ERROR_GRACE:
                msg = self.piper.GetArmJointMsgs()
                current = [
                    msg.joint_state.joint_1, msg.joint_state.joint_2,
                    msg.joint_state.joint_3, msg.joint_state.joint_4,
                    msg.joint_state.joint_5, msg.joint_state.joint_6,
                ]
                if all(abs(current[i] - target_sdk[i]) < tolerance_sdk for i in range(6)):
                    return True

            # Command lost: resend once after deadline
            if elapsed > RESEND_DEADLINE and not resent and not seen_moving:
                print(f"[PiperAPI] ⚠ No joint motion after {RESEND_DEADLINE}s, resending")
                self.piper.JointCtrl(*target_sdk)
                resent = True

            # CAN thread dead → break early
            if elapsed > 2.0 and not self._is_can_ok():
                print(f"[PiperAPI] CAN thread dead during joint wait, breaking early")
                break

            time.sleep(0.05)

        # Timeout — log joint diff for diagnosis
        try:
            msg = self.piper.GetArmJointMsgs()
            current = [
                msg.joint_state.joint_1, msg.joint_state.joint_2,
                msg.joint_state.joint_3, msg.joint_state.joint_4,
                msg.joint_state.joint_5, msg.joint_state.joint_6,
            ]
            diff = [abs(current[i] - target_sdk[i]) / self.FACTOR for i in range(6)]
            print(f"[PiperAPI] ✗ Joint timeout ({timeout}s), seen_moving={seen_moving}, "
                  f"diff={[f'{d:.1f}°' for d in diff]}")
        except Exception:
            print(f"[PiperAPI] ✗ Joint timeout ({timeout}s), seen_moving={seen_moving}")
        return False

    def move_z_with_ik(self,
                       target_z: float,
                       ik_solver=None,
                       speed: int = 15,
                       num_waypoints: int = 10,
                       xy_tolerance_mm: float = 5.0,
                       use_gripper_center: bool = True) -> Tuple[bool, str]:
        """
        Move Z axis using IK-planned trajectory (straight line).

        This is the preferred method for Pick descent/ascent:
        - Plans a straight-line trajectory using IK solver
        - Executes with XY drift monitoring
        - Stops if XY drift exceeds tolerance

        Args:
            target_z: Target Z position in mm
            ik_solver: IK solver instance (Arm_IK from piper_pinocchio)
                       If None, falls back to move_z_segmented
            speed: Motion speed 1-100%
            num_waypoints: Number of intermediate waypoints
            xy_tolerance_mm: Maximum XY drift allowed (mm)
            use_gripper_center: Use gripper center coordinates

        Returns:
            (success: bool, message: str)
        """
        if not self.is_connected:
            return False, "Not connected"

        # Get current position
        _, current_pos = self.get_position(return_gripper_center=use_gripper_center)
        current_z = current_pos[2]
        delta_z = target_z - current_z

        print(f"[PiperAPI] move_z_with_ik: {current_z:.1f} -> {target_z:.1f}mm (delta={delta_z:.1f}mm)")

        # If no IK solver, fall back to segmented move
        if ik_solver is None:
            print("[PiperAPI] No IK solver, using segmented MoveJ fallback")
            success = self.move_z_segmented(target_z, speed=speed,
                                            use_gripper_center=use_gripper_center)
            return success, "Segmented fallback" if success else "Segmented fallback failed"

        try:
            # Plan waypoints using IK
            waypoints = self._plan_z_trajectory_ik(
                current_pos, target_z, ik_solver, num_waypoints
            )

            if waypoints is None or len(waypoints) == 0:
                # IK failed - try joint-space interpolation as secondary fallback
                print("[PiperAPI] IK planning failed, trying joint-space interpolation")
                waypoints = self._plan_z_trajectory_joint_space(
                    current_z, target_z, num_waypoints
                )

                if waypoints is None or len(waypoints) == 0:
                    print("[PiperAPI] Joint-space planning also failed, using segmented fallback")
                    success = self.move_z_segmented(target_z, speed=speed,
                                                    use_gripper_center=use_gripper_center)
                    return success, "IK+joint-space failed, segmented fallback"

            # Execute trajectory
            lock_xy = (current_pos[0], current_pos[1])
            return self.execute_ik_trajectory(
                waypoints, speed=speed,
                xy_tolerance_mm=xy_tolerance_mm,
                lock_xy=lock_xy
            )

        except Exception as e:
            print(f"[PiperAPI] IK trajectory error: {e}, using segmented fallback")
            success = self.move_z_segmented(target_z, speed=speed,
                                            use_gripper_center=use_gripper_center)
            return success, f"Error: {e}, segmented fallback"

    def _plan_z_trajectory_ik(self,
                               start_pos: List[float],
                               target_z: float,
                               ik_solver,
                               num_waypoints: int) -> Optional[List[List[float]]]:
        """
        Plan Z trajectory waypoints using IK solver.

        Args:
            start_pos: Starting position [x, y, z, roll, pitch, yaw] in mm/deg
            target_z: Target Z position in mm
            ik_solver: Arm_IK instance
            num_waypoints: Number of waypoints

        Returns:
            List of joint angle arrays in radians, or None if planning fails
        """
        try:
            import pinocchio as pin
            from tf.transformations import quaternion_from_euler
        except ImportError:
            # Fallback quaternion implementation
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
            import sys
            sys.path.insert(0, '/usr/local/lib/python3.8/site-packages')
            import pinocchio as pin

        waypoints = []
        x, y = start_pos[0], start_pos[1]
        roll, pitch, yaw = start_pos[3], start_pos[4], start_pos[5]

        # Convert angles to radians for IK
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        start_z = start_pos[2]
        z_step = (target_z - start_z) / num_waypoints

        # Get current joint angles as initial guess
        current_joints = self.get_joint_degrees()
        init_q = [math.radians(j) for j in current_joints]

        for i in range(num_waypoints):
            z = start_z + z_step * (i + 1)

            # Convert to meters for IK (Pinocchio uses meters)
            x_m = x / 1000.0
            y_m = y / 1000.0
            z_m = z / 1000.0

            # Create target pose
            q = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
            target_pose = pin.SE3(
                pin.Quaternion(q[3], q[0], q[1], q[2]),
                np.array([x_m, y_m, z_m]),
            )

            # Solve IK
            try:
                sol_q, tau_ff, success = ik_solver.ik_fun(
                    target_pose.homogeneous,
                    gripper=0,
                    motorstate=np.array(init_q)
                )

                if not success:
                    print(f"[PiperAPI] IK failed at waypoint {i+1}/{num_waypoints} (Z={z:.1f}mm)")
                    return None

                waypoints.append(list(sol_q))
                init_q = sol_q  # Use solution as next initial guess

            except Exception as e:
                print(f"[PiperAPI] IK error at waypoint {i+1}: {e}")
                return None

        return waypoints

    def _plan_z_trajectory_joint_space(self,
                                        start_z: float,
                                        target_z: float,
                                        num_waypoints: int,
                                        current_joints: Optional[List[float]] = None) -> Optional[List[List[float]]]:
        """
        Plan Z trajectory using pre-computed keypoint configurations and joint-space interpolation.

        This method bypasses IK singularities by interpolating between known-good joint configurations.
        Use when IK-based planning fails due to singularities.

        Args:
            start_z: Starting Z position in mm
            target_z: Target Z position in mm
            num_waypoints: Number of waypoints to generate
            current_joints: Current joint angles in degrees (optional, for smoother start)

        Returns:
            List of joint angle arrays in radians, or None if planning fails
        """
        # Get sorted keypoint Z values
        keypoints = sorted(self.Z_KEYPOINT_CONFIGS.keys(), reverse=True)

        # Find bounding keypoints for start and target
        def find_bounding_keypoints(z_mm):
            """Find the two keypoints that bound a given Z value."""
            for i in range(len(keypoints) - 1):
                if keypoints[i] >= z_mm >= keypoints[i + 1]:
                    return keypoints[i], keypoints[i + 1]
            # Out of range - use nearest
            if z_mm > keypoints[0]:
                return keypoints[0], keypoints[0]
            if z_mm < keypoints[-1]:
                return keypoints[-1], keypoints[-1]
            return None, None

        def interpolate_config(z_mm):
            """Interpolate joint configuration for a given Z value."""
            z_high, z_low = find_bounding_keypoints(z_mm)
            if z_high is None:
                return None

            if z_high == z_low:
                return list(self.Z_KEYPOINT_CONFIGS[z_high])

            # Linear interpolation
            alpha = (z_high - z_mm) / (z_high - z_low)
            config_high = self.Z_KEYPOINT_CONFIGS[z_high]
            config_low = self.Z_KEYPOINT_CONFIGS[z_low]

            return [config_high[i] + alpha * (config_low[i] - config_high[i])
                    for i in range(6)]

        # Check if start and target are within keypoint range
        z_min = min(keypoints)
        z_max = max(keypoints)

        if start_z < z_min - 50 or start_z > z_max + 50:
            print(f"[PiperAPI] Start Z={start_z:.0f}mm outside keypoint range [{z_min}, {z_max}]mm")
            return None

        if target_z < z_min or target_z > z_max:
            print(f"[PiperAPI] Target Z={target_z:.0f}mm outside keypoint range [{z_min}, {z_max}]mm")
            return None

        # Generate waypoints
        waypoints = []
        z_values = np.linspace(start_z, target_z, num_waypoints + 1)[1:]  # Skip start point

        for z in z_values:
            config_deg = interpolate_config(z)
            if config_deg is None:
                print(f"[PiperAPI] Failed to interpolate at Z={z:.0f}mm")
                return None

            # Convert degrees to radians
            config_rad = [math.radians(j) for j in config_deg]
            waypoints.append(config_rad)

        print(f"[PiperAPI] Joint-space trajectory: {len(waypoints)} waypoints, "
              f"Z={start_z:.0f}->{target_z:.0f}mm")

        return waypoints

    # ==================== Context Manager ====================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        if hasattr(self, 'is_connected') and self.is_connected:
            try:
                self.disconnect()
            except Exception:
                pass  # Ignore errors during garbage collection


# ==================== Test ====================

if __name__ == "__main__":
    print("PiperAPI V2 Test")
    print("=" * 40)

    try:
        with PiperAPI("can0") as arm:
            arm.connect()

            # Test get position
            _, pos = arm.get_position(return_gripper_center=True)
            print(f"Current position (gripper center): {pos}")

            _, pos = arm.get_position(return_gripper_center=False)
            print(f"Current position (flange): {pos}")

            # Test small movement
            print("Moving +50mm in X...")
            arm.set_position(x=pos[0] + 50, wait=True, speed=20)

            _, new_pos = arm.get_position()
            print(f"New position: {new_pos}")

            # Test gripper
            print("Opening gripper...")
            arm.set_gripper_position(30, wait=True)  # 30mm

            print("Closing gripper...")
            arm.set_gripper_position(0, wait=True)

            print("✓ Test completed")

    except Exception as e:
        print(f"✗ Test failed: {e}")
