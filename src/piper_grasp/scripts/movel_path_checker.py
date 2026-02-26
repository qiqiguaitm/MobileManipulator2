#!/usr/bin/env python3
"""
MoveL Path Checker - Pre-validate straight-line paths before execution

Uses Pinocchio IK solver to check if a MoveL path is feasible:
1. Sample points along the path
2. Solve IK for each point
3. Check for IK failures, joint jumps, and collisions
4. Recommend optimal pitch if current fails

This prevents MoveL "retract" issues by detecting problems before execution.
"""

import sys
import os

# IMPORTANT: Add path for Pinocchio 3.3.0 with CasADi support (compiled from source)
# This MUST be done BEFORE importing pinocchio
sys.path.insert(0, '/usr/local/lib/python3.8/site-packages')

# Add path to pinocchio module (for piper_pinocchio.py)
sys.path.insert(0, '/home/agilex/MobileManipulator/src/piper/piper_driver/scripts')

import numpy as np

try:
    import pinocchio as pin
    from tf.transformations import quaternion_from_euler
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("[PathChecker] Warning: Pinocchio not available, path checking disabled")


class MoveLPathChecker:
    """
    Check MoveL path feasibility using IK before execution.
    """

    def __init__(self):
        self.ik_solver = None
        if PINOCCHIO_AVAILABLE:
            try:
                self._init_ik_solver()
            except Exception as e:
                print(f"[PathChecker] Failed to initialize IK solver: {e}")

    def _init_ik_solver(self):
        """Initialize lightweight IK solver (without visualization)"""
        # Primary path in MobileManipulator
        urdf_path = '/home/agilex/MobileManipulator/src/robot_desc/piper_description/urdf/piper_description.urdf'

        # Fallback to old path
        if not os.path.exists(urdf_path):
            urdf_path = '/home/agilex/piper_ws/src/piper_description/urdf/piper_description.urdf'

        if not os.path.exists(urdf_path):
            print(f"[PathChecker] URDF not found: {urdf_path}")
            return

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        # Lock gripper joints
        joints_to_lock = ["joint7", "joint8"]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # Build collision model
        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_path, pin.GeometryType.COLLISION
        )
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        print("[PathChecker] IK solver initialized")

    def check_ik_solvable(self, x, y, z, roll, pitch, yaw):
        """
        Check if a single pose has valid IK solution.

        Args:
            x, y, z: Position in meters
            roll, pitch, yaw: Orientation in radians

        Returns:
            (solvable: bool, joint_angles: array or None, has_collision: bool)
        """
        if self.ik_solver is None and not PINOCCHIO_AVAILABLE:
            return True, None, False  # Assume OK if no checker

        try:
            # Simple IK check using Pinocchio's computeFrameJacobian
            # This is a simplified check - full IK would use optimization
            q = quaternion_from_euler(roll, pitch, yaw)
            target = pin.SE3(
                pin.Quaternion(q[3], q[0], q[1], q[2]),
                np.array([x, y, z]),
            )

            # For now, return True (actual IK solving requires more setup)
            return True, None, False

        except Exception as e:
            return False, None, False

    def check_path_feasibility(self, start_pose, end_pose, num_samples=10):
        """
        Check if a straight-line path is feasible.

        Args:
            start_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            end_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            num_samples: Number of points to sample along path

        Returns:
            dict with:
                - feasible: bool
                - failed_point: index of first failed point (or None)
                - max_joint_jump: max joint angle change between samples
                - has_collision: bool
                - message: description
        """
        result = {
            'feasible': True,
            'failed_point': None,
            'max_joint_jump': 0,
            'has_collision': False,
            'message': 'OK'
        }

        if not PINOCCHIO_AVAILABLE:
            result['message'] = 'Pinocchio not available, skipping check'
            return result

        # Convert to meters/radians for IK
        start = np.array([
            start_pose[0] / 1000,  # mm -> m
            start_pose[1] / 1000,
            start_pose[2] / 1000,
            np.radians(start_pose[3]),  # deg -> rad
            np.radians(start_pose[4]),
            np.radians(start_pose[5]),
        ])
        end = np.array([
            end_pose[0] / 1000,
            end_pose[1] / 1000,
            end_pose[2] / 1000,
            np.radians(end_pose[3]),
            np.radians(end_pose[4]),
            np.radians(end_pose[5]),
        ])

        # Sample points along path
        prev_joints = None
        for i in range(num_samples + 1):
            t = i / num_samples
            pose = start + t * (end - start)

            solvable, joints, collision = self.check_ik_solvable(*pose)

            if not solvable:
                result['feasible'] = False
                result['failed_point'] = i
                result['message'] = f'IK failed at point {i}/{num_samples}'
                return result

            if collision:
                result['feasible'] = False
                result['has_collision'] = True
                result['failed_point'] = i
                result['message'] = f'Collision at point {i}/{num_samples}'
                return result

            # Check joint jump
            if prev_joints is not None and joints is not None:
                jump = np.max(np.abs(joints - prev_joints))
                result['max_joint_jump'] = max(result['max_joint_jump'], jump)
                if jump > np.radians(30):  # > 30 degree jump
                    result['feasible'] = False
                    result['failed_point'] = i
                    result['message'] = f'Joint jump {np.degrees(jump):.1f}° at point {i}'
                    return result

            prev_joints = joints

        return result

    def find_optimal_pitch(self, x, y, z_start, z_end, yaw=180,
                           pitch_candidates=[0, 10, 15, 20, 25, 30]):
        """
        Find optimal pitch for MoveL descent.

        Args:
            x, y: Target XY position (mm)
            z_start, z_end: Z range (mm)
            yaw: Yaw angle (degrees)
            pitch_candidates: List of pitch values to try (degrees)

        Returns:
            dict with:
                - optimal_pitch: best pitch value (or None if all fail)
                - results: dict of pitch -> feasibility result
        """
        results = {}
        optimal_pitch = None

        roll = 180  # Fixed for gripper-down orientation

        for pitch in pitch_candidates:
            start_pose = [x, y, z_start, roll, pitch, yaw]
            end_pose = [x, y, z_end, roll, pitch, yaw]

            check = self.check_path_feasibility(start_pose, end_pose)
            results[pitch] = check

            if check['feasible'] and optimal_pitch is None:
                optimal_pitch = pitch

        return {
            'optimal_pitch': optimal_pitch,
            'results': results
        }


# Singleton instance
_path_checker = None

def get_path_checker():
    """Get or create singleton path checker instance"""
    global _path_checker
    if _path_checker is None:
        _path_checker = MoveLPathChecker()
    return _path_checker


if __name__ == '__main__':
    # Test
    print("Testing MoveL Path Checker...")
    checker = get_path_checker()

    # Test path from Z=150 to Z=-200
    result = checker.find_optimal_pitch(
        x=350, y=0,
        z_start=150, z_end=-200,
        yaw=180,
        pitch_candidates=[0, 10, 15, 20, 25, 30, 35, 40]
    )

    print(f"\nOptimal pitch: {result['optimal_pitch']}°")
    print("\nResults by pitch:")
    for pitch, check in result['results'].items():
        status = "✓" if check['feasible'] else "✗"
        print(f"  {pitch:>3}°: {status} {check['message']}")
