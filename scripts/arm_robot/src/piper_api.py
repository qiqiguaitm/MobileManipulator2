#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
PiperAPI Simple - Simplified XArmAPI Compatible Wrapper for PiperSDK

Simplified version based on V2 examples, focusing on core functionality
with minimal complexity while maintaining XArm compatibility.
"""

import time
import math
import warnings
import numpy as np
from typing import Optional, List, Tuple, Union

# Import PiperSDK
import sys
import os
from piper_sdk import C_PiperInterface_V2
from piper_sdk.piper_msgs.msg_v2.feedback.arm_feedback_status import ArmMsgFeedbackStatusEnum
from common import debug_print

class PiperAPI:
    """
    Simplified PiperAPI - XArmAPI compatible wrapper for Piper arms
    
    Based on V2 examples, provides essential XArm compatibility with minimal overhead.
    Follows the simple patterns from piper_sdk/demo/V2/ examples.
    """
    
    def __init__(self, can_name: str = "can0", gripper_max_mm: float = 90.0, damping_ratio: float = 0.98, gripper_offset_mm: float = 135.03):
        """
        Initialize PiperAPI Simple
        
        Args:
            can_name: CAN bus device name (e.g., "can0")
            gripper_max_mm: Maximum gripper range in mm (default 90mm for safety)
            damping_ratio: Damping ratio for gravity stop (0.0-1.0, default 0.9 for high damping fall)
            gripper_offset_mm: Flange to gripper center offset in mm (default 135.03mm)
        """
        self.can_name = can_name
        self.piper = None
        self.is_connected = False
        
        # Gripper parameters (based on V2 examples: 0.05m = 50mm range)
        self.gripper_max_mm = gripper_max_mm
        self.gripper_max_units = int(gripper_max_mm * 1000)  # Convert to 0.001mm units
        self.gripper_min_units = 0  # Fully closed
        
        # Gripper position tracking (since SDK read is unreliable)
        self._last_gripper_position = 0  # Track last commanded position
        
        # Damped gravity stop configuration (high damping for controlled fall)
        self.damping_ratio = max(0.0, min(1.0, damping_ratio))  # Clamp to [0,1]
        
        # Unit conversion and hardware constants (following V2 examples)
        self.FACTOR = 1000  # mm/degrees to 0.001mm/0.001degrees
        self.GRIPPER_OFFSET_MM = gripper_offset_mm  # Flange to gripper center offset (mm) along Z-axis 
        
        print(f"PiperAPI Simple initialized: {can_name}, gripper: 0-{gripper_max_mm}mm, gripper offset: {self.GRIPPER_OFFSET_MM}mm, gravity damping: {damping_ratio}")
    
    def connect(self):
        """
        Connect to Piper arm - simplified V2 style with enable state check
        
        Following the pattern from piper_ctrl_enable.py:
        1. Create interface
        2. Connect port
        3. Check if already enabled, go to zero if so
        4. Enable with simple retry loop if not enabled
        """
        try:
            print(f"Connecting to {self.can_name}...")
            
            # Step 1: Create interface (V2 pattern)
            self.piper = C_PiperInterface_V2(self.can_name)
            
            # Step 2: Connect port (V2 pattern)
            self.piper.ConnectPort()
            time.sleep(0.1)  # V2 standard delay
            # Configure gripper parameters
            # self.piper.GripperTeachingPendantParamConfig(100, 92, 1)
            # self.piper.ArmParamEnquiryAndConfig(4)
            
            # Step 3: Check if already enabled
            already_enabled = False
            try:
                # Try to get current status - if this succeeds, robot might already be enabled
                msg = self.piper.GetArmEndPoseMsgs()
                if msg:  # If we can get position data, robot is likely enabled
                    print("⚠ Robot appears to be already enabled")
                    already_enabled = True
                    self.is_connected = True
            except:
                # If we can't get position, robot is likely not enabled
                already_enabled = False
            
            # Step 4: If already enabled, go to zero position first
            if already_enabled:
                print("Moving to zero position before continuing...")
                self._go_zero()
            else:
                # Step 5: Enable with simple retry if not already enabled
                print("Enabling robot...")
                while not self.piper.EnablePiper():
                    time.sleep(0.01)  # V2 retry delay
                self._go_zero()
                self.is_connected = True
            
            # Perform gripper homing first (important!)
            #print("\nPerforming gripper homing...")
            #self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            #self.piper.GripperCtrl(0, 1000, 0x03, 0)  # 0x03 = Homing command
            #time.sleep(3)  # Wait for homing to complete
            
            # Initialize gripper (following SDK pattern: 0x02 init + 0x01 enable)
            print("Initializing gripper...")
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            self.piper.GripperCtrl(0, 1000, 0x02, 0)  # 0x02 = Initialize
            time.sleep(0.1)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)  # 0x01 = Enable (必须紧跟初始化)
            time.sleep(0.1)
            
    
            print("✓ Connected and enabled successfully")
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise e
    
    def disconnect(self):
        """Disconnect from Piper arm with go_zero, damped stop, then disable"""
        if self.piper and self.is_connected:
            try:
                # First go to zero position
                self._go_zero()
                
                # Execute damped stop before disable
                self._damped_stop()
                
                # Then disable and disconnect
                self.piper.DisablePiper()
                self.piper.DisconnectPort()
                self.is_connected = False
                print("✓ Disconnected")
            except Exception as e:
                print(f"⚠ Disconnect warning: {e}")
    
    def _go_zero(self):
        """
        Go to zero position (home position) based on piper_ctrl_go_zero.py

        Sets all joints to zero position before shutdown
        """
        if not self.is_connected:
            return

        try:
            print(f"------------------GO ZERO------------------")
            print(f"Moving to zero position...")
            print(f"------------------GO ZERO------------------")

            # Set joint control mode with moderate speed (30% speed for safety)
            self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # Joint mode

            # Move joints to zero position (all zeros)
            self.piper.JointCtrl(0, 0, 0, 0, 0, 0)

            # Close gripper
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            self._last_gripper_position = 0

            # Wait for joints to reach zero position
            tolerance = 2.0  # degrees
            timeout = 10.0  # seconds
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    joint_msg = self.piper.GetArmJointMsgs()
                    joints_deg = [
                        joint_msg.joint_state.joint_1 * 0.001,
                        joint_msg.joint_state.joint_2 * 0.001,
                        joint_msg.joint_state.joint_3 * 0.001,
                        joint_msg.joint_state.joint_4 * 0.001,
                        joint_msg.joint_state.joint_5 * 0.001,
                        joint_msg.joint_state.joint_6 * 0.001,
                    ]
                    max_error = max(abs(j) for j in joints_deg)
                    if max_error < tolerance:
                        print(f"✓ Zero position reached (max error: {max_error:.2f}°)")
                        break
                except:
                    pass
                time.sleep(0.2)
            else:
                print(f"⚠ Go zero timeout, continuing anyway")

            print("✓ Zero position completed")

        except Exception as e:
            print(f"⚠ Go zero error: {e}")
            # Continue with disconnect sequence even if go_zero fails

    def go_zero(self, wait: bool = True):
        """
        Public method to go to zero position (home position)

        Args:
            wait: Whether to wait for movement completion (default True)
        """
        self._go_zero()
        if wait:
            # Additional wait to ensure movement is complete
            time.sleep(0.5)

    def _damped_stop(self):
        """
        Improved damped stop with controlled descent and proper error handling
        
        Implements proper damped stopping by:
        1. First stopping current motion
        2. Setting low-speed linear motion mode for controlled descent
        3. Executing gradual downward movement with damping
        4. Proper reset sequence for clean state
        """
        if not self.is_connected:
            return
            
        try:
            print(f"------------------DAMPED STOP------------------")
            print(f"Damping ratio: {self.damping_ratio:.2f}")
            print(f"------------------DAMPED STOP------------------")
            
            # Step 1: Stop current motion immediately
            try:
                self.piper.EmergencyStop(0x01)  # Emergency stop
                time.sleep(0.1)  # Short wait for stop to take effect
            except:
                pass  # Continue even if emergency stop fails
            
            # Step 2: Set controlled descent mode
            # Calculate speed: higher damping_ratio = lower speed = more damping
            # damping_ratio 0.98 -> speed 1 (maximum damping, extremely slow)
            # damping_ratio 0.95 -> speed 1 (maximum damping, very slow)
            # damping_ratio 0.9 -> speed 1 (maximum damping, very slow)
            # damping_ratio 0.5 -> speed 5 (medium damping)
            damping_speed = max(1, int(10 * (1.0 - self.damping_ratio)))
            
            print(f"Setting damped descent mode with speed {damping_speed}%...")
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,              # CAN control mode
                move_mode=0x02,              # MOVE L (Linear motion for smooth descent)
                move_spd_rate_ctrl=damping_speed,  # Low speed for damping effect
                is_mit_mode=0x00             # Position-velocity mode
            )
            
            # Step 3: Execute controlled descent
            try:
                current = self._get_current_pose()
                print(f"Current Z position: {current[2] / self.FACTOR:.1f}mm")
                
                # Calculate descent distance based on damping ratio
                # Higher damping = smaller descent for safety
                descent_distance = int(self.FACTOR * 5.0 * (1.0 - self.damping_ratio))  # 0-5mm descent
                target_z = current[2] - descent_distance
                
                print(f"Executing controlled descent of {descent_distance/self.FACTOR:.1f}mm...")
                
                # Execute gradual descent with current position maintained for X,Y,rotations
                self.piper.EndPoseCtrl(
                    current[0], current[1], target_z,  # Maintain X,Y, lower Z
                    current[3], current[4], current[5]  # Maintain orientation
                )
                
                # Wait for controlled descent with timeout
                # Increased wait time for high damping ratios (0.98 -> ~4 seconds)
                descent_time = 1.0 + (self.damping_ratio * 3.0)  # 1.0-4.0 seconds based on damping
                time.sleep(descent_time)
                
            except Exception as e:
                print(f"Position-based descent failed: {e}, using time-based damping...")
                # Fallback: just wait with low speed setting (increased time for high damping)
                fallback_time = 2.0 + (self.damping_ratio * 2.0)  # 2.0-4.0 seconds
                time.sleep(fallback_time)
            
            # Step 4: Gradual stop sequence
            print("Executing gradual stop sequence...")
            
            # Resume from emergency stop first
            try:
                self.piper.EmergencyStop(0x02)  # Resume
                time.sleep(0.1)
            except:
                pass
            
            # Reset to clean state (based on piper_ctrl_reset.py)
            print("Resetting arm state for next session...")
            try:
                self.piper.MotionCtrl_1(0x02, 0, 0)  # Reset/Restore command
                time.sleep(0.1)
                self.piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)  # Back to standard joint mode
                time.sleep(0.1)
            except:
                pass
            
            print("✓ Damped stop completed successfully")
            
        except Exception as e:
            print(f"⚠ Damped stop error: {e}")
            # Fallback: immediate emergency stop
            try:
                self.piper.EmergencyStop(0x01)
                time.sleep(0.1)
                self.piper.EmergencyStop(0x02)
            except:
                pass
            print("✓ Fallback emergency stop executed")
    
    def clean_error(self, go_zero: bool = False, force: bool = False):
        """
        智能错误清除 - 基于 GetArmStatus 的状态检测
        
        Args:
            go_zero: 是否回到零点 (default False)
            force: 强制执行清除，跳过状态检测 (default False)
        """
        if not self.is_connected:
            return
            
        try:
            # 除非强制执行，否则先检测错误状态
            if not force:
                needs_clearing, error_desc = self._check_robot_error_state()
                severity = self._get_error_severity()
                
                if not needs_clearing:
                    print("ℹ Robot status normal, no error clearing needed")
                    if go_zero:
                        print("Executing go_zero as requested...")
                        self._go_zero()
                    return
                else:
                    print(f"🚨 Robot error detected [{severity}]: {error_desc}")
                    
                    # 对于严重错误，跳过 go_zero 操作
                    if severity == "CRITICAL":
                        print("⚠ Critical error detected - skipping go_zero for safety")
                        go_zero = False
            
            print("Performing error clearing sequence...")
            
            # 安全操作序列
            if go_zero:
                try:
                    print("Moving to zero position before error clearing...")
                    self._go_zero()
                except Exception as e:
                    print(f"⚠ Go zero failed: {e}, continuing with error clearing...")
            
            # 受控停止
            try:
                print("Executing controlled stop...")
                self._damped_stop()
            except Exception as e:
                print(f"⚠ Damped stop failed: {e}, continuing...")
            
            # 执行 disable/enable 循环
            print("Executing disable/enable cycle...")
            self.piper.DisablePiper()
            time.sleep(0.2)
            
            retry_count = 0
            max_retries = 10
            while not self.piper.EnablePiper() and retry_count < max_retries:
                time.sleep(0.1)
                retry_count += 1
                
            if retry_count >= max_retries:
                raise Exception(f"Failed to re-enable robot after {max_retries} attempts")
            
            # 验证错误是否清除
            time.sleep(0.2)  # 等待状态更新
            verification_needed, final_status = self._check_robot_error_state()
            
            if verification_needed:
                print(f"⚠ Some errors may remain: {final_status}")
            else:
                print("✓ Error clearing completed successfully")
                
        except Exception as e:
            print(f"✗ Error clearing failed: {e}")
            raise
    
    def clean_gripper_error(self):
        """Clear gripper error - simple reset"""
        if not self.is_connected:
            return
        try:
            self.piper.GripperCtrl(0, 1000, 0x02, 0)  # Reset command
            time.sleep(0.1)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)  # Enable command
            print("✓ Gripper error cleared")
        except Exception as e:
            print(f"⚠ Gripper error clearing warning: {e}")
    
    def motion_enable(self, enable: bool = True, go_zero: bool = False, force: bool = False):
        """Enable/disable motion with enable state check
        
        Args:
            enable: Whether to enable motion (default True)
            go_zero: Whether to return to zero position when already enabled (default False)
            force: Force enable even if already enabled - needed after ctrl_stop/damped_stop (default False)
        """
        if not self.is_connected:
            return
        try:
            if enable:
                if force:
                    # Force enable - skip state checks, directly enable
                    print("🔄 Force enabling motion (post ctrl_stop/damped_stop)")
                    while not self.piper.EnablePiper():
                        time.sleep(0.01)
                    print("✓ Motion force enabled")
                else:
                    # Check if already enabled before enabling
                    already_enabled = False
                    try:
                        # Try to get current status - if this succeeds, robot might already be enabled
                        msg = self.piper.GetArmEndPoseMsgs()
                        if msg:  # If we can get position data, robot is likely enabled
                            print("⚠ Robot appears to be already enabled during motion_enable")
                            already_enabled = True
                    except:
                        # If we can't get position, robot is likely not enabled
                        already_enabled = False
                    
                    # If already enabled, go to zero position first (if requested)
                    if already_enabled:
                        if go_zero:
                            # 先检查是否有严重错误
                            if self._get_error_severity() == "CRITICAL":
                                print("⚠ Critical error detected - skipping go_zero for safety")
                            else:
                                print("Moving to zero position before continuing...")
                                self._go_zero()
                    else:
                        # Enable if not already enabled
                        while not self.piper.EnablePiper():
                            time.sleep(0.01)
                    
                    print("✓ Motion enabled")
            else:
                self.piper.DisablePiper()
                print("✓ Motion disabled")
        except Exception as e:
            print(f"⚠ Motion enable warning: {e}")
    
    def set_gripper_enable(self, enable: bool = True):
        """Enable/disable gripper - integrated with motion in Piper"""
        print("ℹ Gripper enable is integrated with motion enable in Piper")
    
    def set_mode(self, mode: int = 0):
        """Set control mode - simplified"""
        if not self.is_connected:
            return
        # Mode 0: Position control (default for Piper)
        print(f"✓ Mode {mode} set (position control)")
    
    def set_state(self, state: int = 0):
        """Set arm state - simplified"""
        if not self.is_connected:
            return
        # State 0: Ready state (default)
        print(f"✓ State {state} set (ready)")
    
    def set_position(self, x: float = None, y: float = None, z: float = None,
                    roll: float = None, pitch: float = None, yaw: float = None,
                    wait: bool = False, speed: int = 30, use_gripper_center: bool = True):
        """
        Set end effector position - V2 style with gripper center support
        
        Following piper_ctrl_end_pose.py pattern:
        1. Convert gripper center to flange coordinates if needed
        2. Set motion control with speed parameter
        3. Convert units (mm/degrees * 1000)
        4. Call EndPoseCtrl
        
        Args:
            x, y, z: Position in mm
            roll, pitch, yaw: Orientation in degrees
            wait: Whether to wait for completion
            speed: Motion speed percentage (1-100, default 30 for low speed)
            use_gripper_center: If True, x,y,z represents gripper center position (default False for flange)
        """
        if not self.is_connected:
            return
        
        try:
            # Get current pose as base
            current = self._get_current_pose()
            
            # Convert gripper center to flange coordinates if needed
            if use_gripper_center:
                # Get current gripper center position (not flange!)
                _, current_gripper_pos = self.get_position(return_gripper_center=True)
                
                # Build target gripper position (mix current gripper center and new values)
                gripper_pos = [
                    x if x is not None else current_gripper_pos[0],
                    y if y is not None else current_gripper_pos[1], 
                    z if z is not None else current_gripper_pos[2],
                    roll if roll is not None else current_gripper_pos[3],
                    pitch if pitch is not None else current_gripper_pos[4],
                    yaw if yaw is not None else current_gripper_pos[5]
                ]
                #debug_print("Set Position, Gripper:", x, y, z, roll, pitch, yaw)
                # Convert gripper center to flange position
                # flange = gripper_center - R_gripper @ [0,0,offset]
                angles_rad = [math.radians(angle) for angle in gripper_pos[3:6]]
                from scipy.spatial.transform import Rotation as R
                R_gripper = R.from_euler('xyz', angles_rad).as_matrix()
                
                gripper_offset_flange = np.array([0, 0, self.GRIPPER_OFFSET_MM])  # mm
                gripper_offset_base = R_gripper @ gripper_offset_flange
                
                # Calculate flange position
                flange_pos = np.array(gripper_pos[:3]) - gripper_offset_base
                
                # Update x, y, z with flange coordinates
                x, y, z = flange_pos[0], flange_pos[1], flange_pos[2]
                #debug_print("Set Position, Flange:", x, y, z, roll, pitch, yaw)
            
            # Update specified coordinates (convert to 0.001mm/0.001degree units)
            if x is not None:
                current[0] = round(x * self.FACTOR)
            if y is not None:
                current[1] = round(y * self.FACTOR)
            if z is not None:
                current[2] = round(z * self.FACTOR)
            if roll is not None:
                current[3] = round(roll * self.FACTOR)
            if pitch is not None:
                current[4] = round(pitch * self.FACTOR)
            if yaw is not None:
                current[5] = round(yaw * self.FACTOR)
            
            # Clamp speed to safe range (1-100%)
            safe_speed = max(1, min(100, speed))
            
            # Send position command multiple times like the original demo
            # The original demo sends commands in a continuous loop
            for attempt in range(10):  # Send command multiple times
                self.piper.MotionCtrl_2(0x01, 0x00, safe_speed, 0x00)
                self.piper.EndPoseCtrl(current[0], current[1], current[2],
                                     current[3], current[4], current[5])
                time.sleep(0.05)  # Small delay between attempts
            
            
            if wait:
                # Dynamic wait with position checking
                max_wait = 2.0  # Maximum wait time
                check_interval = 0.5
                start_time = time.time()
                
                while (time.time() - start_time) < max_wait:
                    time.sleep(check_interval)
                    try:
                        # Check if position reached (within tolerance)
                        current_pose = self._get_current_pose()
                        tolerance = 5 * self.FACTOR  # 5mm tolerance
                        
                        reached = True
                        for i in range(3):  # Check XYZ only
                            if abs(current_pose[i] - current[i]) > tolerance:
                                reached = False
                                break
                        
                        if reached:
                            break
                    except:
                        pass  # Continue waiting if check fails
                
                # Final wait for stabilization
                time.sleep(0.5)
            else:
                time.sleep(0.5)  # Longer minimum wait for command processing
            
            
            print(f"✓ Flange Position set: [{current[0]/self.FACTOR:.1f}, {current[1]/self.FACTOR:.1f}, {current[2]/self.FACTOR:.1f}]mm at {safe_speed}% speed")
            
        except Exception as e:
            print(f"✗ Set position failed: {e}")
    
    def get_position(self, is_radian: bool = False, return_gripper_center: bool = True) -> Tuple[int, List[float]]:
        """
        Get current position - V2 style with gripper center option
        
        Following piper_read_end_pose.py pattern
        
        Args:
            is_radian: Return angles in radians (default False, returns degrees)
            return_gripper_center: Return gripper center position instead of flange (default False)
        
        Returns:
            tuple: (status_code, [x,y,z,roll,pitch,yaw])
                   positions in mm, orientations in degrees/radians
                   If return_gripper_center=True, position is gripper center
        """
        if not self.is_connected:
            return -1, [0, 0, 0, 0, 0, 0]
        
        try:
            # Get end pose message (V2 pattern)
            msg = self.piper.GetArmEndPoseMsgs()
            
            # Extract and convert units (0.001mm/0.001degree to mm/degree)
            flange_position = [
                float(msg.end_pose.X_axis) / self.FACTOR,   # x in mm
                float(msg.end_pose.Y_axis) / self.FACTOR,   # y in mm  
                float(msg.end_pose.Z_axis) / self.FACTOR,   # z in mm
                float(msg.end_pose.RX_axis) / self.FACTOR,  # roll in degrees
                float(msg.end_pose.RY_axis) / self.FACTOR,  # pitch in degrees
                float(msg.end_pose.RZ_axis) / self.FACTOR,  # yaw in degrees
            ]
            
            # Convert to radians if requested
            position = flange_position.copy()
            if is_radian:
                position[3:6] = [math.radians(angle) for angle in position[3:6]]
            
            # Calculate gripper center if requested
            if return_gripper_center:
                # Gripper offset in flange frame: [0, 0, offset_mm]
                gripper_offset_flange = np.array([0, 0, self.GRIPPER_OFFSET_MM])  # mm
                
                # Get rotation matrix from flange orientation
                angles_rad = [math.radians(angle) for angle in flange_position[3:6]]
                from scipy.spatial.transform import Rotation as R
                R_flange = R.from_euler('xyz', angles_rad).as_matrix()
                
                # Transform offset to base frame
                gripper_offset_base = R_flange @ gripper_offset_flange
                
                # Add offset to flange position
                position[0] += gripper_offset_base[0]  # x
                position[1] += gripper_offset_base[1]  # y  
                position[2] += gripper_offset_base[2]  # z
                # Orientation remains same (gripper parallel to flange)
            
            return 0, position  # 0 = success
            
        except Exception as e:
            print(f"⚠ Get position warning: {e}")
            return -1, [0, 0, 0, 0, 0, 0]
    
    def set_gripper_position(self, pos: int, wait: bool = False, speed: int = 1000):
        """
        Set gripper position - V2 style with speed control

        Following piper_ctrl_gripper.py pattern

        Args:
            pos: Gripper position in 0.001mm units (0 = closed, max = open)
            wait: Whether to wait for completion
            speed: Gripper speed (1-1000, default 500 for moderate speed)
        """
        if not self.is_connected:
            return

        try:
            # Clamp position to safe range
            safe_pos = max(self.gripper_min_units,
                          min(self.gripper_max_units, abs(pos)))

            # Clamp speed to safe range (1-1000)
            safe_speed = max(1, min(1000, speed))

            # Set motion control mode and send gripper command
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            self.piper.GripperCtrl(safe_pos, safe_speed, 0x01, 0)

            # Track commanded position
            self._last_gripper_position = safe_pos

            pos_mm = safe_pos / self.FACTOR

            if wait:
                # Wait for gripper to reach target position
                tolerance = 3000  # 3mm tolerance in 0.001mm units
                timeout = 5.0  # seconds
                start_time = time.time()

                while time.time() - start_time < timeout:
                    _, actual_pos = self.get_gripper_position()
                    if abs(actual_pos - safe_pos) <= tolerance:
                        break
                    time.sleep(0.1)

                _, final_pos = self.get_gripper_position()
                final_mm = final_pos / self.FACTOR
                print(f"✓ Gripper position: {final_mm:.1f}mm (target: {pos_mm:.1f}mm) at speed {safe_speed}")
            else:
                print(f"✓ Gripper command sent: {pos_mm:.1f}mm at speed {safe_speed}")

        except Exception as e:
            print(f"✗ Set gripper failed: {e}")
    
    def get_gripper_position(self) -> Tuple[int, int]:
        """
        Get current gripper position from hardware feedback
        
        Returns:
            tuple: (status_code, position_in_units)
                   position in 0.001mm units (0 = closed, max = open)
        """
        if not self.is_connected:
            return -1, 0
        
        try:
            # Get actual gripper position from hardware feedback
            msg = self.piper.GetArmGripperMsgs()
            if msg and hasattr(msg, 'gripper_state') and hasattr(msg.gripper_state, 'grippers_angle'):
                # Convert grippers_angle to position (already in 0.001mm units)
                actual_position = msg.gripper_state.grippers_angle
                
                # Update tracked position with actual value
                self._last_gripper_position = actual_position
                
                return 0, actual_position
            else:
                # Fallback to last commanded position if hardware read fails
                print("⚠ Failed to read gripper hardware state, using last commanded position")
                return 0, self._last_gripper_position
                
        except Exception as e:
            print(f"⚠ Get gripper position error: {e}, using last commanded position")
            # Fallback to last commanded position
            return 0, self._last_gripper_position
    
    def set_joint_position(self, angles: List[float], wait: bool = False, speed: int = 10):
        """
        Set joint angles with speed control
        
        Args:
            angles: 6 joint angles in degrees
            wait: Whether to wait for completion
            speed: Motion speed percentage (1-100, default 30 for low speed)
        """
        if not self.is_connected or len(angles) != 6:
            return
        
        try:
            # Clamp speed to safe range (1-100%)
            safe_speed = max(1, min(100, speed))
            
            # Set joint control mode with speed limit
            self.piper.MotionCtrl_2(0x01, 0x01, safe_speed, 0x00)  # Joint mode
            
            # Convert to 0.001degree units
            joint_units = [round(angle * self.FACTOR) for angle in angles]
            
            # Send joint command
            self.piper.JointCtrl(joint_units[0], joint_units[1], joint_units[2],
                               joint_units[3], joint_units[4], joint_units[5])
            
            if wait:
                time.sleep(1.0)
            
            print(f"✓ Joint angles set: {angles} at {safe_speed}% speed")
            
        except Exception as e:
            print(f"✗ Set joint position failed: {e}")
    
    # XArm compatibility - no-op implementations
    def set_tcp_load(self, mass: float, center: List[float]):
        """Set TCP load - not supported by Piper"""
        warnings.warn("set_tcp_load not supported by Piper", UserWarning)
    
    def set_tcp_offset(self, offset: List[float]):
        """Set TCP offset - not supported by Piper"""  
        warnings.warn("set_tcp_offset not supported by Piper", UserWarning)
    
    # Error detection methods
    def _check_robot_error_state(self) -> Tuple[bool, str]:
        """
        基于 GetArmStatus 的智能错误状态检测 - 兼容不同SDK版本
        
        Returns:
            Tuple[bool, str]: (needs_error_clearing, error_description)
        """
        try:
            # 获取机器人状态
            status = self.piper.GetArmStatus()
            if not status:
                return True, "Cannot get robot status"
            
            error_conditions = []
            needs_clearing = False
            
            # 1. 检查机械臂状态 (最重要的状态)
            if hasattr(status, 'arm_status'):
                try:
                    if status.arm_status != ArmMsgFeedbackStatusEnum.ArmStatus.NORMAL:
                        needs_clearing = True
                        error_conditions.append(f"Arm status: {status.arm_status}")
                        
                        # 特定严重错误需要立即清除
                        serious_errors = [
                            ArmMsgFeedbackStatusEnum.ArmStatus.EMERGENCY_STOP,
                            ArmMsgFeedbackStatusEnum.ArmStatus.JOINT_COMMUNICATION_ERR, 
                            ArmMsgFeedbackStatusEnum.ArmStatus.JOINT_BRAKE_NOT_RELEASED,
                            ArmMsgFeedbackStatusEnum.ArmStatus.COLLISION_OCCURRED,
                            ArmMsgFeedbackStatusEnum.ArmStatus.JOINT_STATUS_ERR,
                            ArmMsgFeedbackStatusEnum.ArmStatus.OTHER_ERR
                        ]
                        
                        if status.arm_status in serious_errors:
                            error_conditions.append("(SERIOUS - requires immediate clearing)")
                except Exception as e:
                    error_conditions.append(f"Arm status check failed: {e}")
            
            # 2. 检查错误码 (关节限位和通信错误) - 使用安全的属性检查
            if hasattr(status, 'err_code'):
                try:
                    if status.err_code != 0:
                        needs_clearing = True
                        error_details = []
                        
                        # 关节角度限位错误
                        joint_limit_errors = []
                        for i in range(6):
                            if status.err_code & (1 << (8 + i)):
                                joint_limit_errors.append(f"Joint{i+1}")
                        if joint_limit_errors:
                            error_details.append(f"Angle limits: {', '.join(joint_limit_errors)}")
                        
                        # 关节通信错误
                        comm_errors = []
                        for i in range(6):
                            if status.err_code & (1 << i):
                                comm_errors.append(f"Joint{i+1}")
                        if comm_errors:
                            error_details.append(f"Communication: {', '.join(comm_errors)}")
                            
                        error_conditions.append(f"Error code 0x{status.err_code:04X}: {'; '.join(error_details)}")
                except Exception as e:
                    error_conditions.append(f"Error code check failed: {e}")
            
            # 3. 检查控制模式 (应该是 CAN_CTRL) - 使用安全检查
            if hasattr(status, 'ctrl_mode'):
                try:
                    if status.ctrl_mode != ArmMsgFeedbackStatusEnum.CtrlMode.CAN_CTRL:
                        # 不一定需要清除错误，但需要报告
                        error_conditions.append(f"Control mode: {status.ctrl_mode} (expected CAN_CTRL)")
                except Exception as e:
                    error_conditions.append(f"Control mode check failed: {e}")
            
            # 4. 检查运动状态 - 使用安全检查
            if hasattr(status, 'motion_status'):
                try:
                    if status.motion_status == ArmMsgFeedbackStatusEnum.MotionStatus.REACH_TARGET_POS_FAILED:
                        error_conditions.append("Motion: Failed to reach target position")
                        # 这个不一定需要清除错误，可能只是运动失败
                except Exception as e:
                    error_conditions.append(f"Motion status check failed: {e}")
            
            # 如果无法获取任何状态信息，则认为需要清除错误
            if not error_conditions and not any(hasattr(status, attr) for attr in ['arm_status', 'err_code', 'ctrl_mode']):
                return True, "Status object missing expected attributes - may need error clearing"
            
            if error_conditions:
                return needs_clearing, "; ".join(error_conditions)
            else:
                return False, "Robot status normal"
                
        except Exception as e:
            return True, f"Status check failed: {e}"

    def _is_robot_in_error_state(self) -> bool:
        """
        快速检查机器人是否处于错误状态 - 兼容不同SDK版本
        
        Returns:
            bool: True if robot is in error state
        """
        try:
            status = self.piper.GetArmStatus()
            if not status:
                return True
            
            # 只检查最关键的错误状态，使用安全的属性检查
            has_arm_error = False
            has_err_code = False
            
            if hasattr(status, 'arm_status'):
                try:
                    has_arm_error = status.arm_status != ArmMsgFeedbackStatusEnum.ArmStatus.NORMAL
                except:
                    has_arm_error = True  # 如果无法检查，保守认为有错误
                    
            if hasattr(status, 'err_code'):
                try:
                    has_err_code = status.err_code != 0
                except:
                    has_err_code = False  # err_code检查失败不一定是错误
            
            return has_arm_error or has_err_code
                    
        except Exception:
            return True  # 如果无法获取状态，保守地认为有错误

    def _get_error_severity(self) -> str:
        """
        获取错误严重程度 - 兼容不同SDK版本
        
        Returns:
            "CRITICAL" | "WARNING" | "NORMAL"
        """
        try:
            status = self.piper.GetArmStatus()
            if not status:
                return "CRITICAL"
            
            # 检查严重错误，使用安全的属性检查
            if hasattr(status, 'arm_status'):
                try:
                    # 严重错误
                    critical_states = [
                        ArmMsgFeedbackStatusEnum.ArmStatus.EMERGENCY_STOP,
                        ArmMsgFeedbackStatusEnum.ArmStatus.JOINT_COMMUNICATION_ERR,
                        ArmMsgFeedbackStatusEnum.ArmStatus.COLLISION_OCCURRED,
                        ArmMsgFeedbackStatusEnum.ArmStatus.JOINT_STATUS_ERR
                    ]
                    
                    if status.arm_status in critical_states:
                        return "CRITICAL"
                    elif status.arm_status != ArmMsgFeedbackStatusEnum.ArmStatus.NORMAL:
                        severity = "WARNING"
                    else:
                        severity = "NORMAL"
                except:
                    severity = "WARNING"  # 如果无法检查arm_status，认为是警告级别
            else:
                severity = "WARNING"  # 如果没有arm_status属性，认为是警告
            
            # 检查错误码
            if hasattr(status, 'err_code'):
                try:
                    if status.err_code != 0:
                        if severity == "NORMAL":
                            severity = "WARNING"
                except:
                    pass  # err_code检查失败不影响整体严重程度判断
                        
            return severity
                
        except Exception:
            return "CRITICAL"

    # Private helper methods
    def _get_current_pose(self):
        """Get current pose in raw units for incremental control"""
        try:
            msg = self.piper.GetArmEndPoseMsgs()
            return [
                int(msg.end_pose.X_axis),
                int(msg.end_pose.Y_axis), 
                int(msg.end_pose.Z_axis),
                int(msg.end_pose.RX_axis),
                int(msg.end_pose.RY_axis),
                int(msg.end_pose.RZ_axis),
            ]
        except:
            return [0, 0, 200*self.FACTOR, 0, 0, 0]  # Default pose
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def __del__(self):
        if hasattr(self, 'is_connected') and self.is_connected:
            try:
                self.disconnect()
            except:
                pass


# Test code following V2 examples
if __name__ == "__main__":
    print("PiperAPI Simple Test")
    print("=" * 30)
    
    try:
        # Test basic functionality (V2 style)
        with PiperAPI("can0") as arm:
            arm.connect()
            
            # XArm compatibility test
            arm.clean_error(go_zero=True)
            arm.clean_gripper_error()
            #arm.motion_enable(True,go_zero=True) 
            
            # Position control test
            state, pos = arm.get_position()
            print(f"Current position: {pos}")
            
            # Test small movements with wait enabled
            target_x = pos[0] + 50  # 50mm movement
            target_z = pos[2] + 50  # 50mm upward
            print(f"Moving to: [{target_x:.1f}, {pos[1]:.1f}, {target_z:.1f}]")
            
            
            arm.set_position(x=target_x, y=pos[1], z=target_z, wait=True, speed=10)
            
            # Check position after movement
            state, new_pos = arm.get_position()
            print(f"Position after movement: {new_pos}")
            
            # Verify movement occurred
            moved_x = abs(new_pos[0] - pos[0])
            moved_z = abs(new_pos[2] - pos[2])
            print(f"Actual movement: X={moved_x:.1f}mm, Z={moved_z:.1f}mm")
            
            #init_pos = dict(x=270, y=0, z=307)
            #init_pos = dict(x=345, y=0, z=380, roll=180, pitch=40, yaw=180)
            init_pos =dict(x=431.78, y=0, z=276.58, roll=180, pitch=40, yaw=180) # gripper center
            arm.set_position(**init_pos, wait=True, speed=50)
            # Check position after movement
            state, new_pos = arm.get_position()
            print(f"Position after movement: {new_pos}")
            
            
            init_pos = dict(x=667.0,y=-102.9)
            arm.set_position(**init_pos, wait=True, speed=50)
            # Check position after movement
            state, new_pos = arm.get_position()
            print(f"Position after movement: {new_pos}")
            
            init_pos = dict(yaw=-18.7696764)
            arm.set_position(**init_pos, wait=True, speed=50)
            # Check position after movement
            state, new_pos = arm.get_position()
            print(f"Position after movement: {new_pos}")
        
            import time
            # Gripper test  
            arm.set_gripper_position(30000, wait=True)  # 30mm open
            time.sleep(2)
            arm.set_gripper_position(40000, wait=True)  # 30mm open
            time.sleep(2)
            arm.set_gripper_position(50000, wait=True)  # 30mm open
            time.sleep(2)
            arm.set_gripper_position(0, wait=True)      # Closed
            arm.disconnect()
            print("✓ Basic test completed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
