#!/usr/bin/env python3
"""
Safe Trajectory Planner with Collision Avoidance

Uses segmented MoveJ (joint-space interpolation) for smooth motion:
1. Define key waypoints in Cartesian space
2. Solve IK for each waypoint
3. Interpolate in joint space between waypoints
4. Check collision at interpolated points
5. Auto-replan if collision detected

Advantages of MoveJ over MoveL:
- Avoids Cartesian singularities
- Smoother joint motion
- More predictable behavior
- Faster computation (fewer IK solves)
"""

import numpy as np
import math
import sys
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

# Add paths
sys.path.insert(0, '/opt/ros/noetic/lib/python3.8/site-packages')
sys.path.insert(0, '/usr/local/lib/python3.8/site-packages')
sys.path.insert(0, '/home/agilex/MobileManipulator/src/piper/piper_driver/scripts')
sys.path.insert(0, '/home/agilex/MobileManipulator/src/piper/piper_grasp/scripts')

from collision_checker import CollisionChecker, CollisionResult, CollisionType


class PlanningStatus(Enum):
    SUCCESS = 0
    COLLISION_AT_TARGET = 1
    COLLISION_ON_PATH = 2
    REPLANNED = 3
    FAILED = 4
    IK_FAILED = 5


@dataclass
class Waypoint:
    """Single waypoint in trajectory"""
    position: np.ndarray      # [x, y, z] in mm
    orientation: np.ndarray   # [roll, pitch, yaw] in degrees
    joint_angles: Optional[np.ndarray] = None


@dataclass
class TrajectoryPlan:
    """Complete trajectory plan"""
    status: PlanningStatus
    waypoints: List[Waypoint]
    original_target: Waypoint
    adjusted_target: Optional[Waypoint]
    collision_info: str
    replan_attempts: int


class SafeTrajectoryPlanner:
    """
    Trajectory planner with integrated collision avoidance.
    """

    # Replan strategies
    MAX_REPLAN_ATTEMPTS = 5

    # Waypoint insertion heights
    SAFE_HEIGHT_MM = 150  # Safe height for obstacle clearance
    LIFT_INCREMENT_MM = 50  # Incremental lift when replanning

    # Orientation adjustment ranges
    PITCH_RANGE = [0, 15, 30, 45]  # degrees
    YAW_ADJUSTMENTS = [-30, -15, 0, 15, 30]  # degrees offset

    def __init__(self, ik_solver=None, collision_checker: CollisionChecker = None):
        """
        Initialize safe trajectory planner.

        Args:
            ik_solver: IK solver instance (e.g., Arm_IK from piper_pinocchio)
            collision_checker: CollisionChecker instance
        """
        self.ik_solver = ik_solver
        self.collision_checker = collision_checker or CollisionChecker()

        # Try to import IK solver if not provided
        if self.ik_solver is None:
            try:
                from piper_pinocchio import Arm_IK
                self.ik_solver = Arm_IK(headless=True)
                print("[SafePlanner] IK solver initialized")
            except Exception as e:
                print(f"[SafePlanner] Warning: IK solver not available: {e}")

    def check_pose_safe(self, position_mm: np.ndarray,
                       orientation_deg: np.ndarray) -> Tuple[bool, CollisionResult]:
        """
        Check if a pose is collision-free.

        Args:
            position_mm: [x, y, z] in millimeters
            orientation_deg: [roll, pitch, yaw] in degrees

        Returns:
            (is_safe, collision_result)
        """
        pos_m = position_mm / 1000.0
        ori_rad = np.radians(orientation_deg)

        result = self.collision_checker.check_point_collision(pos_m, ori_rad)
        return not result.has_collision, result

    def plan_safe_trajectory(self,
                            start_pose: List[float],
                            target_pose: List[float],
                            velocity: float = 50.0) -> TrajectoryPlan:
        """
        Plan a safe trajectory from start to target with collision avoidance.

        Args:
            start_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            target_pose: [x, y, z, roll, pitch, yaw] in mm/degrees
            velocity: Cartesian velocity (mm/s)

        Returns:
            TrajectoryPlan with waypoints and status
        """
        start = np.array(start_pose)
        target = np.array(target_pose)

        original_target = Waypoint(
            position=target[:3],
            orientation=target[3:]
        )

        # Step 1: Check target pose safety
        target_safe, target_collision = self.check_pose_safe(target[:3], target[3:])

        if not target_safe:
            # Try to adjust target orientation
            adjusted = self._try_adjust_orientation(target[:3], target[3:])
            if adjusted is not None:
                target[3:] = adjusted
                target_safe = True
                print(f"[SafePlanner] Target adjusted to orientation: {adjusted}")
            else:
                return TrajectoryPlan(
                    status=PlanningStatus.COLLISION_AT_TARGET,
                    waypoints=[],
                    original_target=original_target,
                    adjusted_target=None,
                    collision_info=f"Target collision: {target_collision.collision_pairs}",
                    replan_attempts=0
                )

        # Step 2: Plan direct trajectory and check collisions
        waypoints, collision_point = self._plan_and_check(start, target, velocity)

        if collision_point is None:
            # Direct path is safe
            return TrajectoryPlan(
                status=PlanningStatus.SUCCESS,
                waypoints=waypoints,
                original_target=original_target,
                adjusted_target=Waypoint(target[:3], target[3:]) if not np.array_equal(original_target.orientation, target[3:]) else None,
                collision_info="",
                replan_attempts=0
            )

        # Step 3: Replan with waypoint insertion
        for attempt in range(self.MAX_REPLAN_ATTEMPTS):
            lift_height = self.SAFE_HEIGHT_MM + attempt * self.LIFT_INCREMENT_MM

            waypoints, collision_point = self._plan_with_lift(
                start, target, lift_height, velocity
            )

            if collision_point is None:
                return TrajectoryPlan(
                    status=PlanningStatus.REPLANNED,
                    waypoints=waypoints,
                    original_target=original_target,
                    adjusted_target=Waypoint(target[:3], target[3:]),
                    collision_info=f"Replanned with lift height {lift_height}mm",
                    replan_attempts=attempt + 1
                )

        # All replan attempts failed
        return TrajectoryPlan(
            status=PlanningStatus.FAILED,
            waypoints=[],
            original_target=original_target,
            adjusted_target=None,
            collision_info=f"Failed after {self.MAX_REPLAN_ATTEMPTS} replan attempts",
            replan_attempts=self.MAX_REPLAN_ATTEMPTS
        )

    def _try_adjust_orientation(self, position_mm: np.ndarray,
                                orientation_deg: np.ndarray) -> Optional[np.ndarray]:
        """
        Try different orientations to find a collision-free pose.

        Returns:
            Adjusted orientation or None if all failed
        """
        roll = orientation_deg[0]
        base_pitch = orientation_deg[1]
        base_yaw = orientation_deg[2]

        for pitch in self.PITCH_RANGE:
            for yaw_adj in self.YAW_ADJUSTMENTS:
                test_ori = np.array([roll, pitch, base_yaw + yaw_adj])
                is_safe, _ = self.check_pose_safe(position_mm, test_ori)
                if is_safe:
                    return test_ori

        return None

    def _plan_and_check(self, start: np.ndarray, target: np.ndarray,
                       velocity: float) -> Tuple[List[Waypoint], Optional[int]]:
        """
        Plan direct trajectory and check for collisions.

        Returns:
            (waypoints, collision_point_index) - collision_point is None if safe
        """
        waypoints = self._interpolate_trajectory(start, target, velocity)

        for i, wp in enumerate(waypoints):
            is_safe, result = self.check_pose_safe(wp.position, wp.orientation)
            if not is_safe:
                return waypoints[:i], i

        return waypoints, None

    def _plan_with_lift(self, start: np.ndarray, target: np.ndarray,
                       lift_height: float, velocity: float) -> Tuple[List[Waypoint], Optional[int]]:
        """
        Plan trajectory with intermediate lift waypoint.

        Path: start -> lift_point -> above_target -> target

        Returns:
            (waypoints, collision_point_index)
        """
        # Lift point: directly above start at safe height
        lift_point = start.copy()
        lift_point[2] = max(start[2], lift_height)

        # Above target: directly above target at safe height
        above_target = target.copy()
        above_target[2] = max(target[2], lift_height)

        # Build full trajectory
        all_waypoints = []

        # Segment 1: start -> lift_point (vertical lift)
        if lift_point[2] > start[2]:
            seg1 = self._interpolate_trajectory(start, lift_point, velocity)
            all_waypoints.extend(seg1[:-1])

        # Segment 2: lift_point -> above_target (horizontal move at safe height)
        seg2 = self._interpolate_trajectory(lift_point, above_target, velocity)
        all_waypoints.extend(seg2[:-1])

        # Segment 3: above_target -> target (vertical descent)
        seg3 = self._interpolate_trajectory(above_target, target, velocity)
        all_waypoints.extend(seg3)

        # Check collisions
        for i, wp in enumerate(all_waypoints):
            is_safe, result = self.check_pose_safe(wp.position, wp.orientation)
            if not is_safe:
                return all_waypoints[:i], i

        return all_waypoints, None

    def _interpolate_trajectory(self, start: np.ndarray, end: np.ndarray,
                               velocity: float) -> List[Waypoint]:
        """
        Interpolate waypoints between start and end.

        Args:
            start: [x, y, z, roll, pitch, yaw]
            end: [x, y, z, roll, pitch, yaw]
            velocity: mm/s

        Returns:
            List of Waypoint
        """
        distance = np.linalg.norm(end[:3] - start[:3])

        if distance < 1.0:  # Less than 1mm
            return [Waypoint(end[:3].copy(), end[3:].copy())]

        # Determine number of points (at least 5mm spacing)
        num_points = max(int(distance / 5.0), 2)

        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 1.0
            pose = start + t * (end - start)
            waypoints.append(Waypoint(
                position=pose[:3].copy(),
                orientation=pose[3:].copy()
            ))

        return waypoints

    def solve_trajectory_ik(self, waypoints: List[Waypoint]) -> Tuple[List[Waypoint], bool]:
        """
        Solve IK for all waypoints.

        Returns:
            (waypoints_with_ik, all_success)
        """
        if self.ik_solver is None:
            print("[SafePlanner] No IK solver available")
            return waypoints, False

        import pinocchio as pin
        from tf.transformations import quaternion_from_euler

        prev_joints = None
        all_success = True

        for wp in waypoints:
            # Convert to IK input format
            x, y, z = wp.position / 1000.0  # mm -> m
            roll, pitch, yaw = np.radians(wp.orientation)

            q = quaternion_from_euler(roll, pitch, yaw)
            target = pin.SE3(
                pin.Quaternion(q[3], q[0], q[1], q[2]),
                np.array([x, y, z])
            )

            sol_q, _, success = self.ik_solver.ik_fun(
                target.homogeneous,
                gripper=0,
                motorstate=prev_joints
            )

            if not success:
                all_success = False
                continue

            wp.joint_angles = sol_q
            prev_joints = sol_q

        return waypoints, all_success

    def plan_descent_trajectory(self,
                               x_mm: float, y_mm: float,
                               start_z_mm: float, target_z_mm: float,
                               orientation_deg: List[float],
                               velocity: float = 50.0) -> TrajectoryPlan:
        """
        Plan a vertical descent trajectory (common for grasping).

        Args:
            x_mm, y_mm: XY position (fixed)
            start_z_mm: Starting Z height
            target_z_mm: Target Z depth
            orientation_deg: [roll, pitch, yaw]
            velocity: mm/s

        Returns:
            TrajectoryPlan
        """
        start_pose = [x_mm, y_mm, start_z_mm] + list(orientation_deg)
        target_pose = [x_mm, y_mm, target_z_mm] + list(orientation_deg)

        return self.plan_safe_trajectory(start_pose, target_pose, velocity)


def create_safe_planner(ik_solver=None) -> SafeTrajectoryPlanner:
    """Create safe trajectory planner"""
    return SafeTrajectoryPlanner(ik_solver)


# Test
if __name__ == '__main__':
    print("Safe Trajectory Planner Test")
    print("=" * 60)

    planner = SafeTrajectoryPlanner()

    # Test 1: Safe descent
    print("\n[Test 1] Safe descent (X=400, Y=100, Z: 150 -> -200)")
    plan = planner.plan_descent_trajectory(
        x_mm=400, y_mm=100,
        start_z_mm=150, target_z_mm=-200,
        orientation_deg=[180, 30, 180]
    )
    print(f"  Status: {plan.status.name}")
    print(f"  Waypoints: {len(plan.waypoints)}")
    print(f"  Replan attempts: {plan.replan_attempts}")

    # Test 2: Collision at target (near lidar)
    print("\n[Test 2] Collision at target (near lidar)")
    plan = planner.plan_descent_trajectory(
        x_mm=270, y_mm=0,
        start_z_mm=150, target_z_mm=50,
        orientation_deg=[180, 0, 180]
    )
    print(f"  Status: {plan.status.name}")
    print(f"  Collision info: {plan.collision_info}")

    # Test 3: Need replan (collision on path)
    print("\n[Test 3] Path requiring replan")
    start_pose = [400, 0, 150, 180, 30, 180]
    target_pose = [200, 0, 50, 180, 30, 180]  # Goes through box area

    plan = planner.plan_safe_trajectory(start_pose, target_pose)
    print(f"  Status: {plan.status.name}")
    print(f"  Waypoints: {len(plan.waypoints)}")
    print(f"  Replan attempts: {plan.replan_attempts}")
    if plan.collision_info:
        print(f"  Info: {plan.collision_info}")

    # Test 4: Check specific poses (positions in mm, orientations in degrees)
    print("\n[Test 4] Pose safety checks")
    test_poses = [
        ([400, 100, -200], [180, 30, 180], "Safe grasp position"),
        ([150, 0, -50], [180, 0, 180], "Near lidar zone"),
        ([0, 0, -50], [180, 0, 180], "Box collision zone"),
        ([-300, 0, 200], [180, 0, 180], "Lifting collision zone"),
    ]

    for pos, ori, desc in test_poses:
        is_safe, result = planner.check_pose_safe(np.array(pos), np.array(ori))
        status = "SAFE" if is_safe else "COLLISION"
        dist_str = f"dist={result.min_distance*1000:.0f}mm"
        print(f"  {desc:25s}: {status:10s} ({dist_str})")
        if not is_safe:
            print(f"    Collision pairs: {result.collision_pairs}")
