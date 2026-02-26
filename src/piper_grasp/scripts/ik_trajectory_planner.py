#!/usr/bin/env python3
"""
IK-based Trajectory Planner for Piper Arm

Generates straight-line trajectories in Cartesian space using IK,
then executes via joint control - bypassing SDK's MoveL/MoveJ entirely.

This is the industrial-standard approach for precise path control:
1. Interpolate points in Cartesian space (guaranteed straight line)
2. Solve IK for each point (get joint angles)
3. Send joint angles directly (bypass motion planner)

Advantages:
- Perfect straight-line motion (Cartesian interpolation)
- No singularity "retract" (IK failure detected early)
- Full control over velocity profile
- No cumulative drift
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """Single point in trajectory"""
    position: np.ndarray      # [x, y, z] in mm
    orientation: np.ndarray   # [roll, pitch, yaw] in degrees
    joint_angles: Optional[np.ndarray] = None  # [j1, j2, j3, j4, j5, j6] in radians
    timestamp: float = 0.0    # Time from start (seconds)


class IKTrajectoryPlanner:
    """
    Plan and execute Cartesian straight-line trajectories using IK.
    """

    def __init__(self, ik_solver=None, joint_controller=None):
        """
        Args:
            ik_solver: IK solver instance (e.g., from piper_pinocchio.py)
            joint_controller: Joint control interface
        """
        self.ik_solver = ik_solver
        self.joint_controller = joint_controller

    def interpolate_linear(self, start: np.ndarray, end: np.ndarray,
                           num_points: int) -> List[np.ndarray]:
        """
        Linear interpolation between two poses.

        Args:
            start: [x, y, z, roll, pitch, yaw]
            end: [x, y, z, roll, pitch, yaw]
            num_points: Number of interpolation points (including start and end)

        Returns:
            List of interpolated poses
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            point = start + t * (end - start)
            points.append(point)
        return points

    def plan_linear_trajectory(self,
                               start_pose: List[float],
                               end_pose: List[float],
                               velocity: float = 50.0,
                               acceleration: float = 100.0) -> Tuple[List[TrajectoryPoint], str]:
        """
        Plan a straight-line trajectory from start to end.

        Args:
            start_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            end_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            velocity: Max Cartesian velocity (mm/s)
            acceleration: Max acceleration (mm/s^2)

        Returns:
            (trajectory_points, status_message)
        """
        start = np.array(start_pose)
        end = np.array(end_pose)

        # Calculate distance
        distance = np.linalg.norm(end[:3] - start[:3])

        if distance < 1.0:  # Less than 1mm
            return [TrajectoryPoint(end[:3], end[3:])], "OK (trivial move)"

        # Calculate time profile (trapezoidal velocity)
        # Simplified: assume constant velocity
        duration = distance / velocity

        # Determine number of points (at least 10Hz control rate)
        control_rate = 50  # Hz
        num_points = max(int(duration * control_rate), 10)

        # Interpolate in Cartesian space
        poses = self.interpolate_linear(start, end, num_points)

        # Build trajectory
        trajectory = []
        for i, pose in enumerate(poses):
            t = i * duration / (num_points - 1) if num_points > 1 else 0
            point = TrajectoryPoint(
                position=pose[:3],
                orientation=pose[3:],
                timestamp=t
            )
            trajectory.append(point)

        # Solve IK for each point (if solver available)
        if self.ik_solver is not None:
            prev_joints = None
            for i, point in enumerate(trajectory):
                # Convert to IK solver format (meters, radians)
                x, y, z = point.position / 1000.0  # mm -> m
                roll, pitch, yaw = np.radians(point.orientation)

                try:
                    # Call IK solver
                    joints, _, success = self.ik_solver.ik_fun(
                        self._pose_to_matrix(x, y, z, roll, pitch, yaw),
                        gripper=0,
                        motorstate=prev_joints
                    )

                    if not success:
                        return trajectory[:i], f"IK failed at point {i}/{num_points}"

                    # Check for large joint jumps
                    if prev_joints is not None:
                        max_jump = np.max(np.abs(joints - prev_joints))
                        if max_jump > np.radians(30):  # > 30 degree jump
                            return trajectory[:i], f"Joint jump {np.degrees(max_jump):.1f}° at point {i}"

                    point.joint_angles = joints
                    prev_joints = joints

                except Exception as e:
                    return trajectory[:i], f"IK error at point {i}: {e}"

        return trajectory, "OK"

    def _pose_to_matrix(self, x, y, z, roll, pitch, yaw):
        """Convert pose to 4x4 transformation matrix"""
        try:
            import pinocchio as pin
            from tf.transformations import quaternion_from_euler

            q = quaternion_from_euler(roll, pitch, yaw)
            T = pin.SE3(
                pin.Quaternion(q[3], q[0], q[1], q[2]),
                np.array([x, y, z]),
            )
            return T.homogeneous
        except ImportError:
            # Return identity if pinocchio not available
            return np.eye(4)

    def execute_trajectory(self, trajectory: List[TrajectoryPoint],
                          control_rate: float = 50.0) -> bool:
        """
        Execute planned trajectory via joint control.

        Args:
            trajectory: List of TrajectoryPoint with joint_angles
            control_rate: Control loop rate (Hz)

        Returns:
            True if successful
        """
        if not trajectory:
            return False

        if self.joint_controller is None:
            print("[IKPlanner] No joint controller, cannot execute")
            return False

        dt = 1.0 / control_rate
        start_time = time.time()

        for i, point in enumerate(trajectory):
            if point.joint_angles is None:
                print(f"[IKPlanner] No joint angles for point {i}")
                return False

            # Send joint command
            try:
                self.joint_controller.joint_control_piper(
                    point.joint_angles[0],
                    point.joint_angles[1],
                    point.joint_angles[2],
                    point.joint_angles[3],
                    point.joint_angles[4],
                    point.joint_angles[5],
                    0  # gripper
                )
            except Exception as e:
                print(f"[IKPlanner] Joint control failed: {e}")
                return False

            # Wait for next point
            target_time = start_time + point.timestamp
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        return True

    def move_linear(self,
                    current_pose: List[float],
                    target_pose: List[float],
                    velocity: float = 50.0) -> Tuple[bool, str]:
        """
        High-level API: Plan and execute linear move.

        Args:
            current_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            target_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            velocity: Cartesian velocity (mm/s)

        Returns:
            (success, message)
        """
        # Plan trajectory
        trajectory, status = self.plan_linear_trajectory(
            current_pose, target_pose, velocity
        )

        if "OK" not in status:
            return False, f"Planning failed: {status}"

        # Execute
        if self.ik_solver is not None and self.joint_controller is not None:
            success = self.execute_trajectory(trajectory)
            if not success:
                return False, "Execution failed"
        else:
            return False, "No IK solver or joint controller"

        return True, "OK"


def create_planner_with_piper():
    """
    Create planner integrated with Piper arm.

    Returns:
        IKTrajectoryPlanner instance (or None if setup fails)
    """
    try:
        import sys
        sys.path.insert(0, '/home/agilex/MobileManipulator/src/piper/piper_driver/scripts')

        from piper_pinocchio import Arm_IK
        from piper_control import PIPER

        # Initialize IK solver (without visualization)
        # Note: Arm_IK opens Meshcat visualizer by default, may need modification
        ik_solver = Arm_IK()

        # Initialize joint controller
        joint_controller = PIPER()

        planner = IKTrajectoryPlanner(
            ik_solver=ik_solver,
            joint_controller=joint_controller
        )

        print("[IKPlanner] Initialized with Piper arm")
        return planner

    except Exception as e:
        print(f"[IKPlanner] Failed to initialize: {e}")
        return None


# Example usage
if __name__ == '__main__':
    print("IK Trajectory Planner Demo")
    print("=" * 50)

    # Create planner (without actual hardware for demo)
    planner = IKTrajectoryPlanner()

    # Plan a descent trajectory
    start = [350, 0, 150, 180, 30, 180]  # Ready position
    end = [350, 0, -200, 180, 30, 180]    # Target position

    trajectory, status = planner.plan_linear_trajectory(
        start, end, velocity=50.0
    )

    print(f"Status: {status}")
    print(f"Trajectory points: {len(trajectory)}")

    if trajectory:
        print(f"Duration: {trajectory[-1].timestamp:.2f}s")
        print(f"Start: {trajectory[0].position}")
        print(f"End: {trajectory[-1].position}")
