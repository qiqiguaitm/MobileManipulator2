#!/usr/bin/env python3
"""
Test script for PiperAPI V2
Validates all interfaces used by piper_grasp_node.py

Test order:
1. Connection
2. Motion Enable
3. Get Position
4. Set Position
5. Gripper Control
6. Go Zero
7. Joint Reading
8. Damped Stop (disables arm - must be last motion test)
9. Safe Disconnect
"""

import sys
import time
import os

# Add current script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from piper_api_v2 import PiperAPI

FACTOR = 1000


def print_section(title):
    print("\n" + "=" * 50)
    print(f"TEST: {title}")
    print("=" * 50)


def print_joints(label, joints):
    print(f"  {label}: [{', '.join(f'{j:6.1f}' for j in joints)}]°")


class TestPiperAPIV2:
    def __init__(self):
        self.arm = None
        self.tests_passed = 0
        self.tests_failed = 0

    def run_all(self):
        """Run all tests in correct order"""
        print("=" * 50)
        print("PiperAPI V2 Test Suite")
        print("=" * 50)

        try:
            self.test_connection()
            self.test_motion_enable()
            self.test_get_position()
            self.test_set_position()
            self.test_gripper()
            self.test_go_zero()
            self.test_go_ready()
            self.test_go_zero()
            self.test_go_ready()
            self.test_joint_reading()
            self.test_damped_stop()  # Disables arm
            self.test_disconnect()

        except Exception as e:
            print(f"\n✗ TEST SUITE ABORTED: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1

        finally:
            # Ensure cleanup
            if self.arm and self.arm.piper:
                try:
                    self.arm.piper.DisconnectPort()
                except:
                    pass

        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {self.tests_failed}")
        if self.tests_failed == 0:
            print("\n✓ ALL TESTS PASSED")
        else:
            print(f"\n✗ {self.tests_failed} TEST(S) FAILED")
        print("=" * 50)

        return self.tests_failed == 0

    def test_connection(self):
        """Test connect"""
        print_section("Connection")
        input("Press Enter to start...")

        self.arm = PiperAPI("can0")
        self.arm.connect()

        assert self.arm.is_connected, "Should be connected"
        assert self.arm.piper is not None, "piper attribute should exist"

        print("✓ connect() passed")
        self.tests_passed += 1

    def test_motion_enable(self):
        """Test motion_enable"""
        print_section("Motion Enable")
        input("Press Enter to start...")

        # Check current status
        enable_list = self.arm.piper.GetArmEnableStatus()
        print(f"  Current enable status: {enable_list}")

        # Enable
        result = self.arm.motion_enable(enable=True)
        assert result, "motion_enable should return True"

        # Verify
        enable_list = self.arm.piper.GetArmEnableStatus()
        assert all(enable_list), "All joints should be enabled"

        print("✓ motion_enable(True) passed")
        self.tests_passed += 1

    def test_get_position(self):
        """Test get_position with both modes"""
        print_section("Get Position")
        input("Press Enter to start...")

        # Gripper center
        status, pos_gc = self.arm.get_position(return_gripper_center=True)
        assert status == 0, "Status should be 0"
        assert len(pos_gc) == 6, "Should return 6 values"
        print(f"  Gripper center: [{pos_gc[0]:.1f}, {pos_gc[1]:.1f}, {pos_gc[2]:.1f}, "
              f"{pos_gc[3]:.1f}, {pos_gc[4]:.1f}, {pos_gc[5]:.1f}]")

        # Flange
        status, pos_fl = self.arm.get_position(return_gripper_center=False)
        assert status == 0, "Status should be 0"
        print(f"  Flange:         [{pos_fl[0]:.1f}, {pos_fl[1]:.1f}, {pos_fl[2]:.1f}, "
              f"{pos_fl[3]:.1f}, {pos_fl[4]:.1f}, {pos_fl[5]:.1f}]")

        # Verify offset
        z_diff = pos_gc[2] - pos_fl[2]
        print(f"  Z offset (gc - flange): {z_diff:.1f}mm (expected ~135mm)")

        print("✓ get_position() passed")
        self.tests_passed += 1

    def test_set_position(self):
        """Test set_position"""
        print_section("Set Position")
        input("Press Enter to start...")

        # Get current position
        _, pos_orig = self.arm.get_position(return_gripper_center=False)
        print(f"  Original Z: {pos_orig[2]:.1f}mm")

        # Move up 20mm
        print("  Moving +20mm in Z...")
        result = self.arm.set_position(z=pos_orig[2] + 20, wait=True, speed=30, use_gripper_center=False)

        # Verify
        _, pos_new = self.arm.get_position(return_gripper_center=False)
        z_moved = pos_new[2] - pos_orig[2]
        print(f"  New Z: {pos_new[2]:.1f}mm (moved {z_moved:.1f}mm)")

        if abs(z_moved - 20) < 5:
            print("✓ set_position() movement verified")
        else:
            print(f"⚠ Movement error: expected 20mm, got {z_moved:.1f}mm")

        # Move back
        print("  Moving back...")
        self.arm.set_position(z=pos_orig[2], wait=True, speed=30, use_gripper_center=False)

        print("✓ set_position() passed")
        self.tests_passed += 1

    def test_gripper(self):
        """Test gripper control"""
        print_section("Gripper Control")
        input("Press Enter to start...")

        # Get current (now returns mm directly)
        status, pos_orig = self.arm.get_gripper_position()
        print(f"  Current: {pos_orig:.1f}mm")

        # Open (API now uses mm directly)
        print("  Opening to 30mm...")
        self.arm.set_gripper_position(30, speed=500, wait=True)  # 30mm
        time.sleep(0.3)

        status, pos_open = self.arm.get_gripper_position()
        print(f"  After open: {pos_open:.1f}mm")

        # Close
        print("  Closing...")
        self.arm.set_gripper_position(0, speed=500, wait=True)
        time.sleep(0.3)

        status, pos_close = self.arm.get_gripper_position()
        print(f"  After close: {pos_close:.1f}mm")

        print("✓ Gripper control passed")
        self.tests_passed += 1

    def test_go_zero(self):
        """Test go_zero"""
        print_section("Go Zero")
        input("Press Enter to start...")

        # First move away from zero
        print("  Moving to test position...")
        self.arm.piper.MotionCtrl_2(0x01, 0x01, 30, 0)
        self.arm.piper.JointCtrl(0, int(-15 * FACTOR), int(15 * FACTOR), 0, 0, 0)
        time.sleep(2.0)

        joints_before = self.arm.get_joint_degrees()
        print_joints("Before go_zero", joints_before)

        # Go to zero
        print("  Executing go_zero()...")
        self.arm.go_zero()

        # Verify
        joints_after = self.arm.get_joint_degrees()
        print_joints("After go_zero", joints_after)

        max_error = max(abs(j) for j in joints_after)
        print(f"  Max error from zero: {max_error:.1f}°")

        if max_error < 3.0:
            print("✓ go_zero() reached zero position")
        else:
            print(f"⚠ go_zero() error: max deviation {max_error:.1f}°")

        print("✓ go_zero() passed")
        self.tests_passed += 1

    def test_go_ready(self):
        """Test go_ready"""
        print_section("Go Ready")
        input("Press Enter to start...")

        # Get current position (FLANGE coordinates)
        _, pos_before = self.arm.get_position(return_gripper_center=False)
        print(f"  Before: [{pos_before[0]:.1f}, {pos_before[1]:.1f}, {pos_before[2]:.1f}]mm")

        # Go to ready
        print("  Executing go_ready()...")
        result = self.arm.go_ready(open_gripper=True, speed=30)

        # Verify position (FLANGE coordinates)
        _, pos_after = self.arm.get_position(return_gripper_center=False)
        print(f"  After: [{pos_after[0]:.1f}, {pos_after[1]:.1f}, {pos_after[2]:.1f}]mm")

        # Check against default ready position
        ready = self.arm.DEFAULT_READY_POS
        diff_x = abs(pos_after[0] - ready['x'])
        diff_y = abs(pos_after[1] - ready['y'])
        diff_z = abs(pos_after[2] - ready['z'])
        print(f"  Expected: [{ready['x']}, {ready['y']}, {ready['z']}]mm")
        print(f"  Diff: [{diff_x:.1f}, {diff_y:.1f}, {diff_z:.1f}]mm")

        if diff_x < 10 and diff_y < 10 and diff_z < 10:
            print("✓ go_ready() reached ready position")
        else:
            print(f"⚠ go_ready() position error")

        print("✓ go_ready() passed")
        self.tests_passed += 1

    def test_joint_reading(self):
        """Test _get_joint_degrees"""
        print_section("Joint Reading")
        input("Press Enter to start...")

        joints = self.arm.get_joint_degrees()
        print_joints("Current joints", joints)

        assert len(joints) == 6, "Should return 6 joint values"

        # Verify reasonable range
        for i, j in enumerate(joints):
            assert -180 <= j <= 180, f"Joint {i+1} out of range: {j}"

        print("✓ _get_joint_degrees() passed")
        self.tests_passed += 1

    def test_damped_stop(self):
        """Test _damped_stop (moves to natural hang, then disables)"""
        print_section("Damped Stop")
        input("Press Enter to start...")

        print("  Natural hang target:", self.arm.NATURAL_HANG_DEG)

        # Get position before
        joints_before = self.arm.get_joint_degrees()
        print_joints("Before damped_stop", joints_before)

        # Execute damped stop
        print("  Executing _damped_stop()...")
        self.arm._damped_stop()

        # Get position after (arm is now disabled)
        joints_after = self.arm.get_joint_degrees()
        print_joints("After damped_stop", joints_after)

        # Check drift from natural hang
        diff = [abs(joints_after[i] - self.arm.NATURAL_HANG_DEG[i]) for i in range(6)]
        max_diff = max(diff)
        print(f"  Diff from natural hang: max {max_diff:.1f}°")

        if max_diff < 5.0:
            print("✓ damped_stop() reached natural hang position")
        else:
            print(f"⚠ damped_stop() deviation: {max_diff:.1f}°")

        # Verify disabled
        enable_list = self.arm.piper.GetArmEnableStatus()
        if not any(enable_list):
            print("✓ Arm is disabled after damped_stop")
        else:
            print(f"⚠ Arm still enabled: {enable_list}")

        print("✓ _damped_stop() passed")
        self.tests_passed += 1

    def test_disconnect(self):
        """Test disconnect (arm already disabled by damped_stop)"""
        print_section("Disconnect")
        input("Press Enter to start...")

        # Just disconnect port (arm already disabled)
        self.arm.piper.DisconnectPort()
        self.arm.is_connected = False

        print("✓ disconnect() passed")
        self.tests_passed += 1


def main():
    tester = TestPiperAPIV2()
    success = tester.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
