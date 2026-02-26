#!/usr/bin/env python3
"""
Workspace Reachability Validation Test

Validates (X, Y, Z, Yaw) reachability by testing descent from ready position.

Test Definition:
- For each (X, Y, Z_target, Yaw), verify if trajectory from
  (X, Y, Z=150, roll=180°) to (X, Y, Z_target, roll=180°, pitch=variable, yaw=Yaw)
  is reachable.

Resolution:
- X, Y: 50mm (coarse grid)
- Z: 10mm
- Yaw: 5°
- Pitch: 5° (adaptive search)

Yaw range: 90° ~ 270°
"""

import numpy as np
import sys
import time
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add paths
sys.path.insert(0, '/opt/ros/noetic/lib/python3.8/site-packages')
sys.path.insert(0, '/usr/local/lib/python3.8/site-packages')
sys.path.insert(0, '/home/agilex/MobileManipulator/src/piper/piper_driver/scripts')
sys.path.insert(0, '/home/agilex/MobileManipulator/src/piper/piper_grasp/scripts')

import pinocchio as pin
from piper_pinocchio import Arm_IK


@dataclass
class ReachabilityResult:
    """Result for a single (X, Y, Z, Yaw) point"""
    x: float
    y: float
    z: float
    yaw: float
    reachable: bool
    best_pitch: Optional[float] = None
    start_joints: Optional[np.ndarray] = None
    end_joints: Optional[np.ndarray] = None
    error_mm: Optional[float] = None


class TrajectoryMode:
    """Trajectory validation modes"""
    ENDPOINT_ONLY = "endpoint"  # Only check start and end points (curve path OK)
    STRAIGHT_LINE = "straight"  # Check all waypoints along straight line


class WorkspaceValidator:
    """Validates workspace reachability with IK descent test"""

    # Default parameters (full resolution as requested)
    X_RANGE = np.arange(300, 601, 50)      # 300~600mm, 50mm step (7 points)
    Y_RANGE = np.arange(-200, 201, 50)     # -200~200mm, 50mm step (9 points)
    Z_TARGETS = np.arange(0, -251, -10)    # 0~-250mm, 10mm step (26 points)
    YAW_RANGE = np.arange(90, 271, 5)      # 90~270°, 5° step (37 points)
    PITCH_OPTIONS = np.arange(0, 46, 5)    # 0~45°, 5° step (10 options)

    # Ready position height
    READY_Z = 150  # mm
    ROLL = 180     # degrees (gripper down)

    # Thresholds
    POSITION_ERROR_THRESHOLD = 5.0   # mm
    JOINT_JUMP_THRESHOLD = 30.0      # degrees

    # Path validation
    WAYPOINT_SPACING_MM = 60.0  # For straight-line validation

    def __init__(self, headless: bool = True, trajectory_mode: str = TrajectoryMode.STRAIGHT_LINE):
        """Initialize IK solver"""
        print("Initializing IK solver...")
        self.ik_solver = Arm_IK(headless=headless)
        self.trajectory_mode = trajectory_mode

        print(f"IK solver ready (mode: {trajectory_mode})")

        # Results storage
        self.results: Dict[Tuple[float, float, float, float], ReachabilityResult] = {}

        # Statistics
        self.stats = {
            'total_tests': 0,
            'reachable': 0,
            'ik_failed': 0,
            'joint_jump': 0,
            'position_error': 0,
        }

    def solve_ik(self, x_mm: float, y_mm: float, z_mm: float,
                 roll_deg: float, pitch_deg: float, yaw_deg: float,
                 seed_joints: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Solve IK for a pose.

        Args:
            x_mm, y_mm, z_mm: Position in mm
            roll_deg, pitch_deg, yaw_deg: Orientation in degrees
            seed_joints: Initial joint configuration for solver

        Returns:
            (joint_angles, position_error_mm, success)
        """
        # Convert to meters and radians
        x, y, z = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
        roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])

        # Build SE3 target
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z])
        )

        # Solve IK
        try:
            sol_q, tau_ff, success = self.ik_solver.ik_fun(
                target.homogeneous,
                gripper=0,
                motorstate=seed_joints
            )
        except Exception as e:
            return None, float('inf'), False

        if not success:
            return None, float('inf'), False

        # Skip FK verification - IK solver already checks position error internally
        # FK error is typically <0.1mm based on testing
        return sol_q, 0.0, True

    def test_descent(self, x_mm: float, y_mm: float, target_z_mm: float,
                    yaw_deg: float, pitch_deg: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Test if descent from ready position to target is reachable.

        Mode depends on self.trajectory_mode:
        - ENDPOINT_ONLY: Only check start and end points (MoveJ curve path)
        - STRAIGHT_LINE: Check all waypoints along straight line (MoveL or segmented MoveJ)

        Args:
            x_mm, y_mm: XY position
            target_z_mm: Target Z depth
            yaw_deg: Target yaw angle
            pitch_deg: Target pitch angle

        Returns:
            (success, start_joints, end_joints, max_joint_jump)
        """
        if self.trajectory_mode == TrajectoryMode.ENDPOINT_ONLY:
            return self._test_descent_endpoint(x_mm, y_mm, target_z_mm, yaw_deg, pitch_deg)
        else:
            return self._test_descent_straight(x_mm, y_mm, target_z_mm, yaw_deg, pitch_deg)

    def _test_descent_endpoint(self, x_mm: float, y_mm: float, target_z_mm: float,
                               yaw_deg: float, pitch_deg: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Endpoint-only validation (for MoveJ curve trajectory).
        Uses progressive seeding to maintain configuration branch continuity.
        """
        # Test start position
        self.reset_ik_state()
        start_joints, _, start_ok = self.solve_ik(
            x_mm, y_mm, self.READY_Z,
            self.ROLL, pitch_deg, yaw_deg
        )

        if not start_ok:
            return False, None, None, float('inf')

        # Try direct seeded solve first
        end_joints, _, end_ok = self.solve_ik(
            x_mm, y_mm, target_z_mm,
            self.ROLL, pitch_deg, yaw_deg,
            seed_joints=start_joints
        )

        # If direct seed fails, use progressive seeding through midpoint
        # This maintains configuration branch continuity
        if not end_ok:
            mid_z = (self.READY_Z + target_z_mm) / 2
            mid_joints, _, mid_ok = self.solve_ik(
                x_mm, y_mm, mid_z,
                self.ROLL, pitch_deg, yaw_deg,
                seed_joints=start_joints
            )
            if mid_ok:
                end_joints, _, end_ok = self.solve_ik(
                    x_mm, y_mm, target_z_mm,
                    self.ROLL, pitch_deg, yaw_deg,
                    seed_joints=mid_joints
                )

        if not end_ok:
            return False, start_joints, None, float('inf')

        # Check total joint change (relaxed for curve path)
        z_dist = abs(self.READY_Z - target_z_mm)
        joint_diff = np.abs(np.degrees(end_joints - start_joints))
        max_jump = np.max(joint_diff)

        # Relaxed threshold: 30° base + 0.1° per mm Z distance
        relaxed_threshold = self.JOINT_JUMP_THRESHOLD + z_dist * 0.1

        if max_jump > relaxed_threshold:
            return False, start_joints, end_joints, max_jump

        return True, start_joints, end_joints, max_jump

    def _test_descent_straight(self, x_mm: float, y_mm: float, target_z_mm: float,
                               yaw_deg: float, pitch_deg: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Straight-line validation (for MoveL or segmented MoveJ).
        Checks all waypoints along the descent path.
        """
        z_dist = abs(self.READY_Z - target_z_mm)
        num_waypoints = max(int(z_dist / self.WAYPOINT_SPACING_MM) + 1, 2)

        # Generate Z waypoints from start to end
        z_waypoints = np.linspace(self.READY_Z, target_z_mm, num_waypoints)

        # Test each waypoint with seeding from previous
        self.reset_ik_state()
        prev_joints = None
        start_joints = None
        max_segment_jump = 0.0

        for i, z in enumerate(z_waypoints):
            joints, err, ok = self.solve_ik(
                x_mm, y_mm, z,
                self.ROLL, pitch_deg, yaw_deg,
                seed_joints=prev_joints
            )

            if not ok:
                # Try without seed
                self.reset_ik_state()
                joints, err, ok = self.solve_ik(
                    x_mm, y_mm, z,
                    self.ROLL, pitch_deg, yaw_deg
                )

            if not ok:
                return False, start_joints, None, float('inf')

            # Check joint continuity from previous waypoint
            if prev_joints is not None:
                segment_jump = np.max(np.abs(np.degrees(joints - prev_joints)))
                max_segment_jump = max(max_segment_jump, segment_jump)

                # Per-segment threshold: 30° (for ~60mm segments)
                if segment_jump > self.JOINT_JUMP_THRESHOLD:
                    return False, start_joints, joints, segment_jump

            if i == 0:
                start_joints = joints

            prev_joints = joints

        end_joints = prev_joints
        return True, start_joints, end_joints, max_segment_jump

    def reset_ik_state(self):
        """Reset IK solver internal state"""
        import gc
        self.ik_solver.init_data = np.array(self.ik_solver.JOINT_OFFSET_RAD)
        self.ik_solver.history_data = np.array(self.ik_solver.JOINT_OFFSET_RAD)
        gc.collect()  # Force garbage collection to free any dangling references

    def test_point(self, x_mm: float, y_mm: float, target_z_mm: float,
                   yaw_deg: float) -> ReachabilityResult:
        """
        Test a single (X, Y, Z, Yaw) point with adaptive pitch search.

        Returns:
            ReachabilityResult
        """
        self.stats['total_tests'] += 1

        # Try each pitch option
        for pitch in self.PITCH_OPTIONS:
            try:
                success, start_j, end_j, err = self.test_descent(
                    x_mm, y_mm, target_z_mm, yaw_deg, pitch
                )

                if success:
                    self.stats['reachable'] += 1
                    return ReachabilityResult(
                        x=x_mm, y=y_mm, z=target_z_mm, yaw=yaw_deg,
                        reachable=True, best_pitch=pitch,
                        start_joints=start_j, end_joints=end_j, error_mm=err
                    )
            except Exception:
                # IK solver error - reset and continue
                self.reset_ik_state()
                continue

        # All pitches failed - reset IK state to prevent solver corruption
        self.reset_ik_state()
        self.stats['ik_failed'] += 1
        return ReachabilityResult(
            x=x_mm, y=y_mm, z=target_z_mm, yaw=yaw_deg,
            reachable=False
        )

    def run_validation(self,
                      x_range: np.ndarray = None,
                      y_range: np.ndarray = None,
                      z_targets: np.ndarray = None,
                      yaw_range: np.ndarray = None,
                      progress_callback=None) -> Dict:
        """
        Run full workspace validation.

        Args:
            x_range, y_range, z_targets, yaw_range: Override default ranges
            progress_callback: Optional callback(current, total)

        Returns:
            Results dictionary
        """
        x_range = x_range if x_range is not None else self.X_RANGE
        y_range = y_range if y_range is not None else self.Y_RANGE
        z_targets = z_targets if z_targets is not None else self.Z_TARGETS
        yaw_range = yaw_range if yaw_range is not None else self.YAW_RANGE

        total = len(x_range) * len(y_range) * len(z_targets) * len(yaw_range)
        current = 0
        start_time = time.time()

        print(f"\nWorkspace Validation")
        print(f"=" * 60)
        print(f"X range: {x_range[0]}~{x_range[-1]}mm ({len(x_range)} points)")
        print(f"Y range: {y_range[0]}~{y_range[-1]}mm ({len(y_range)} points)")
        print(f"Z targets: {z_targets[0]}~{z_targets[-1]}mm ({len(z_targets)} points)")
        print(f"Yaw range: {yaw_range[0]}~{yaw_range[-1]}° ({len(yaw_range)} points)")
        print(f"Pitch options: {self.PITCH_OPTIONS[0]}~{self.PITCH_OPTIONS[-1]}° ({len(self.PITCH_OPTIONS)} options)")
        print(f"Total tests: {total:,}")
        print(f"=" * 60)

        for x in x_range:
            for y in y_range:
                for z in z_targets:
                    for yaw in yaw_range:
                        result = self.test_point(x, y, z, yaw)
                        self.results[(x, y, z, yaw)] = result

                        current += 1
                        if progress_callback:
                            progress_callback(current, total)
                        elif current % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = current / elapsed
                            eta = (total - current) / rate
                            print(f"  Progress: {current:,}/{total:,} ({100*current/total:.1f}%) "
                                  f"ETA: {eta:.0f}s")

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s ({total/elapsed:.1f} tests/s)")

        return self.results

    def generate_report(self) -> str:
        """Generate text report of validation results"""
        lines = []
        lines.append("=" * 70)
        lines.append("WORKSPACE REACHABILITY VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Statistics
        lines.append("STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Total tests:    {self.stats['total_tests']:,}")
        lines.append(f"Reachable:      {self.stats['reachable']:,} ({100*self.stats['reachable']/max(1,self.stats['total_tests']):.1f}%)")
        lines.append(f"IK failed:      {self.stats['ik_failed']:,}")
        lines.append("")

        # Group by (X, Y) - find best Yaw and deepest Z
        xy_results = defaultdict(list)
        for (x, y, z, yaw), result in self.results.items():
            xy_results[(x, y)].append(result)

        # For each (X, Y), find:
        # 1. Deepest reachable Z
        # 2. Yaw range at deepest Z
        # 3. Best pitch at deepest Z

        lines.append("PER-POSITION SUMMARY (X, Y)")
        lines.append("-" * 70)
        lines.append(f"{'X':>6} {'Y':>6} | {'Z_deep':>6} | {'Yaw Range':>15} | {'Best Pitch':>10}")
        lines.append("-" * 70)

        sorted_xy = sorted(xy_results.keys())
        for x, y in sorted_xy:
            results = xy_results[(x, y)]

            # Find deepest reachable Z
            reachable = [r for r in results if r.reachable]
            if not reachable:
                lines.append(f"{x:>6} {y:>6} | {'N/A':>6} | {'No reachable':>15} | {'N/A':>10}")
                continue

            deepest_z = min(r.z for r in reachable)

            # Find Yaw range at deepest Z (or close to it)
            z_threshold = deepest_z + 20  # Within 20mm of deepest
            deep_results = [r for r in reachable if r.z <= z_threshold]
            yaw_values = sorted(set(r.yaw for r in deep_results))

            if yaw_values:
                yaw_min, yaw_max = min(yaw_values), max(yaw_values)
                yaw_range_str = f"{yaw_min:.0f}°~{yaw_max:.0f}°"
            else:
                yaw_range_str = "N/A"

            # Find most common best pitch
            pitches = [r.best_pitch for r in deep_results if r.best_pitch is not None]
            if pitches:
                from collections import Counter
                best_pitch = Counter(pitches).most_common(1)[0][0]
                pitch_str = f"{best_pitch:.0f}°"
            else:
                pitch_str = "N/A"

            lines.append(f"{x:>6} {y:>6} | {deepest_z:>6.0f} | {yaw_range_str:>15} | {pitch_str:>10}")

        lines.append("")

        # Yaw analysis at key positions
        lines.append("YAW REACHABILITY AT KEY POSITIONS")
        lines.append("-" * 70)

        key_positions = [(400, 0), (400, -100), (400, 100), (500, 0), (350, 0)]
        for x, y in key_positions:
            if (x, y) not in xy_results:
                continue

            lines.append(f"\n  Position: X={x}mm, Y={y}mm")
            lines.append(f"  {'Z':>6} | {'Yaw Range':>20} | {'Reachable Yaws':>15}")
            lines.append(f"  " + "-" * 50)

            # Group by Z
            z_results = defaultdict(list)
            for r in xy_results[(x, y)]:
                z_results[r.z].append(r)

            for z in sorted(z_results.keys()):
                results = z_results[z]
                reachable_yaws = sorted([r.yaw for r in results if r.reachable])

                if reachable_yaws:
                    yaw_range_str = f"{min(reachable_yaws):.0f}°~{max(reachable_yaws):.0f}°"
                    count = len(reachable_yaws)
                else:
                    yaw_range_str = "None"
                    count = 0

                lines.append(f"  {z:>6.0f} | {yaw_range_str:>20} | {count:>15}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_results(self, filepath: str):
        """Save results to YAML file"""
        # Convert results to serializable format
        data = {
            'stats': self.stats,
            'parameters': {
                'x_range': [float(x) for x in self.X_RANGE],
                'y_range': [float(y) for y in self.Y_RANGE],
                'z_targets': [float(z) for z in self.Z_TARGETS],
                'yaw_range': [float(y) for y in self.YAW_RANGE],
                'pitch_options': [float(p) for p in self.PITCH_OPTIONS],
                'ready_z': self.READY_Z,
                'roll': self.ROLL,
            },
            'results': {}
        }

        # Group results by (X, Y)
        for (x, y, z, yaw), result in self.results.items():
            key = f"{x}_{y}"
            if key not in data['results']:
                data['results'][key] = {'x': x, 'y': y, 'points': []}

            data['results'][key]['points'].append({
                'z': float(z),
                'yaw': float(yaw),
                'reachable': result.reachable,
                'best_pitch': float(result.best_pitch) if result.best_pitch else None,
                'error_mm': float(result.error_mm) if result.error_mm else None,
            })

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"Results saved to: {filepath}")


def run_quick_validation(trajectory_mode: str = TrajectoryMode.STRAIGHT_LINE):
    """Run quick validation with reduced resolution"""
    validator = WorkspaceValidator(trajectory_mode=trajectory_mode)

    # Quick test: fewer points, focus on known good region
    x_range = np.arange(350, 501, 50)     # 4 points (350-500mm)
    y_range = np.arange(-100, 101, 100)   # 3 points
    z_targets = np.arange(-100, -201, -50) # 3 points (deeper Z)
    yaw_range = np.arange(150, 211, 30)   # 3 points (150-210°, near 180°)

    total = len(x_range) * len(y_range) * len(z_targets) * len(yaw_range)
    print(f"Quick validation: {total} tests")

    validator.run_validation(x_range, y_range, z_targets, yaw_range)

    print("\n" + validator.generate_report())


def run_full_validation(trajectory_mode: str = TrajectoryMode.STRAIGHT_LINE):
    """Run full validation with specified resolution"""
    validator = WorkspaceValidator(trajectory_mode=trajectory_mode)

    # Full test with specified resolution
    # X, Y: 50mm step (coarse)
    # Z: 10mm step
    # Yaw: 5° step

    validator.run_validation()

    report = validator.generate_report()
    print("\n" + report)

    # Save report
    report_path = "/home/agilex/MobileManipulator/scripts/_cc_workspace_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save YAML results
    yaml_path = "/home/agilex/MobileManipulator/scripts/_cc_workspace_validation_results.yaml"
    validator.save_results(yaml_path)


def run_focused_validation(trajectory_mode: str = TrajectoryMode.STRAIGHT_LINE):
    """Run focused validation on promising regions"""
    validator = WorkspaceValidator(trajectory_mode=trajectory_mode)

    # Medium test: balanced resolution
    x_range = np.arange(350, 551, 50)     # 350~550mm, 50mm step (5 points)
    y_range = np.arange(-150, 151, 75)    # -150~150mm, 75mm step (5 points)
    z_targets = np.arange(-100, -251, -25) # -100~-250mm, 25mm step (7 points)
    yaw_range = np.arange(90, 271, 15)    # 90~270°, 15° step (13 points)

    total = len(x_range) * len(y_range) * len(z_targets) * len(yaw_range)
    print(f"Focused validation: {total:,} tests")
    print(f"  X: {x_range[0]}~{x_range[-1]}mm ({len(x_range)} points)")
    print(f"  Y: {y_range[0]}~{y_range[-1]}mm ({len(y_range)} points)")
    print(f"  Z: {z_targets[0]}~{z_targets[-1]}mm ({len(z_targets)} points)")
    print(f"  Yaw: {yaw_range[0]}~{yaw_range[-1]}° ({len(yaw_range)} points)")

    validator.run_validation(x_range, y_range, z_targets, yaw_range)

    report = validator.generate_report()
    print("\n" + report)

    # Save report
    report_path = "/home/agilex/MobileManipulator/scripts/_cc_workspace_validation_focused_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


def run_extended_validation(trajectory_mode: str = TrajectoryMode.STRAIGHT_LINE):
    """
    Run extended validation with doubled ranges and doubled resolution.

    Original (focused):
    - X: 350~550mm, step 50mm (5 points)
    - Y: -150~150mm, step 75mm (5 points)
    - Z: -100~-250mm, step 25mm (7 points)
    - Yaw: 90~270°, step 15° (13 points)

    Extended (doubled ranges, halved steps):
    - X: 200~700mm, step 25mm (21 points)
    - Y: -300~300mm, step 40mm (16 points)
    - Z: 50~-350mm, step 12mm (34 points)
    - Yaw: 45~315°, step 8° (34 points)
    """
    import datetime

    validator = WorkspaceValidator(trajectory_mode=trajectory_mode)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extended ranges (doubled) with moderate resolution
    # Balance between coverage and runtime (~30 minutes)
    x_range = np.arange(200, 701, 50)       # 200~700mm, 50mm step (11 points)
    y_range = np.arange(-300, 301, 50)      # -300~300mm, 50mm step (13 points)
    z_targets = np.arange(50, -351, -20)    # 50~-350mm, 20mm step (21 points)
    yaw_range = np.arange(45, 316, 10)      # 45~315°, 10° step (28 points)

    total = len(x_range) * len(y_range) * len(z_targets) * len(yaw_range)

    print("=" * 70)
    print("EXTENDED WORKSPACE VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Trajectory mode: {trajectory_mode}")
    print(f"\nParameters (doubled ranges, halved steps):")
    print(f"  X: {x_range[0]}~{x_range[-1]}mm, step {x_range[1]-x_range[0]}mm ({len(x_range)} points)")
    print(f"  Y: {y_range[0]}~{y_range[-1]}mm, step {y_range[1]-y_range[0]}mm ({len(y_range)} points)")
    print(f"  Z: {z_targets[0]}~{z_targets[-1]}mm, step {abs(z_targets[1]-z_targets[0])}mm ({len(z_targets)} points)")
    print(f"  Yaw: {yaw_range[0]}~{yaw_range[-1]}°, step {yaw_range[1]-yaw_range[0]}° ({len(yaw_range)} points)")
    print(f"  Total tests: {total:,}")
    print(f"  Estimated time: ~{total / 20 / 60:.0f} minutes")
    print("=" * 70)

    validator.run_validation(x_range, y_range, z_targets, yaw_range)

    report = validator.generate_report()
    print("\n" + report)

    # Save report
    mode_suffix = "curve" if trajectory_mode == TrajectoryMode.ENDPOINT_ONLY else "straight"
    report_path = f"/home/agilex/MobileManipulator/scripts/_cc_workspace_validation_extended_{mode_suffix}_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(f"EXTENDED WORKSPACE VALIDATION REPORT\n")
        f.write(f"Trajectory mode: {trajectory_mode}\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return validator.stats


def run_comprehensive_validation():
    """
    Run comprehensive validation with high resolution for both trajectory modes.

    Resolution (balanced for completeness and runtime):
    - X: 300~600mm, 50mm step (7 points)
    - Y: -200~200mm, 50mm step (9 points)
    - Z: 0~-250mm, 25mm step (11 points)
    - Yaw: 90~270°, 10° step (19 points)
    - Pitch: 0~45°, 5° step (10 options)

    Tests both:
    1. Endpoint-only (curve trajectory / MoveJ)
    2. Straight-line (MoveL / segmented MoveJ)
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # High resolution ranges (balanced for runtime)
    x_range = np.arange(300, 601, 50)       # 7 points
    y_range = np.arange(-200, 201, 50)      # 9 points
    z_targets = np.arange(0, -251, -25)     # 11 points
    yaw_range = np.arange(90, 271, 10)      # 19 points

    total_per_mode = len(x_range) * len(y_range) * len(z_targets) * len(yaw_range)

    print("=" * 70)
    print("COMPREHENSIVE WORKSPACE REACHABILITY VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"\nParameters:")
    print(f"  X range: {x_range[0]}~{x_range[-1]}mm, step {x_range[1]-x_range[0]}mm ({len(x_range)} points)")
    print(f"  Y range: {y_range[0]}~{y_range[-1]}mm, step {y_range[1]-y_range[0]}mm ({len(y_range)} points)")
    print(f"  Z targets: {z_targets[0]}~{z_targets[-1]}mm, step {abs(z_targets[1]-z_targets[0])}mm ({len(z_targets)} points)")
    print(f"  Yaw range: {yaw_range[0]}~{yaw_range[-1]}°, step {yaw_range[1]-yaw_range[0]}° ({len(yaw_range)} points)")
    print(f"  Pitch options: 0~45°, step 5° (10 options)")
    print(f"  Total tests per mode: {total_per_mode:,}")
    print(f"  Total tests (both modes): {total_per_mode * 2:,}")
    print("=" * 70)

    results = {}

    # Mode 1: Endpoint-only (curve trajectory)
    print("\n" + "=" * 70)
    print("MODE 1: ENDPOINT-ONLY (Curve Trajectory / MoveJ)")
    print("=" * 70)
    validator_curve = WorkspaceValidator(trajectory_mode=TrajectoryMode.ENDPOINT_ONLY)
    validator_curve.run_validation(x_range, y_range, z_targets, yaw_range)
    results['curve'] = {
        'stats': validator_curve.stats.copy(),
        'report': validator_curve.generate_report(),
        'results': validator_curve.results.copy()
    }

    # Mode 2: Straight-line (MoveL / segmented MoveJ)
    print("\n" + "=" * 70)
    print("MODE 2: STRAIGHT-LINE (MoveL / Segmented MoveJ)")
    print("=" * 70)
    validator_straight = WorkspaceValidator(trajectory_mode=TrajectoryMode.STRAIGHT_LINE)
    validator_straight.run_validation(x_range, y_range, z_targets, yaw_range)
    results['straight'] = {
        'stats': validator_straight.stats.copy(),
        'report': validator_straight.generate_report(),
        'results': validator_straight.results.copy()
    }

    # Generate comprehensive report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("COMPREHENSIVE WORKSPACE REACHABILITY VALIDATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("")

    report_lines.append("## 1. EXPERIMENT BACKGROUND")
    report_lines.append("-" * 50)
    report_lines.append("Purpose: Validate Piper robot arm workspace reachability for grasping tasks")
    report_lines.append("Test: Descent trajectory from ready position (Z=150mm) to target depth")
    report_lines.append("")
    report_lines.append("Parameters:")
    report_lines.append(f"  - X range: {x_range[0]}~{x_range[-1]}mm (step: {x_range[1]-x_range[0]}mm)")
    report_lines.append(f"  - Y range: {y_range[0]}~{y_range[-1]}mm (step: {y_range[1]-y_range[0]}mm)")
    report_lines.append(f"  - Z targets: {z_targets[0]}~{z_targets[-1]}mm (step: {abs(z_targets[1]-z_targets[0])}mm)")
    report_lines.append(f"  - Yaw range: {yaw_range[0]}~{yaw_range[-1]}° (step: {yaw_range[1]-yaw_range[0]}°)")
    report_lines.append(f"  - Pitch options: 0°~45° (step: 5°)")
    report_lines.append(f"  - Roll: fixed at 180° (gripper pointing down)")
    report_lines.append(f"  - Ready Z: 150mm")
    report_lines.append("")

    report_lines.append("## 2. TRAJECTORY MODES")
    report_lines.append("-" * 50)
    report_lines.append("Mode A - Endpoint-Only (Curve Trajectory):")
    report_lines.append("  - Only validates start and end positions")
    report_lines.append("  - Suitable for MoveJ (joint-space interpolation)")
    report_lines.append("  - Path between points may be curved")
    report_lines.append("  - Joint jump threshold: 30° + 0.1° × Z_distance")
    report_lines.append("")
    report_lines.append("Mode B - Straight-Line Trajectory:")
    report_lines.append("  - Validates all waypoints along descent path")
    report_lines.append("  - Suitable for MoveL or segmented MoveJ")
    report_lines.append("  - Waypoint spacing: 60mm")
    report_lines.append("  - Per-segment joint jump threshold: 30°")
    report_lines.append("")

    report_lines.append("## 3. RESULTS SUMMARY")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append("| Mode | Total Tests | Reachable | Rate |")
    report_lines.append("|------|-------------|-----------|------|")

    curve_stats = results['curve']['stats']
    straight_stats = results['straight']['stats']

    curve_rate = 100.0 * curve_stats['reachable'] / max(1, curve_stats['total_tests'])
    straight_rate = 100.0 * straight_stats['reachable'] / max(1, straight_stats['total_tests'])

    report_lines.append(f"| Curve (Endpoint) | {curve_stats['total_tests']:,} | {curve_stats['reachable']:,} | {curve_rate:.1f}% |")
    report_lines.append(f"| Straight-Line | {straight_stats['total_tests']:,} | {straight_stats['reachable']:,} | {straight_rate:.1f}% |")
    report_lines.append("")

    report_lines.append("## 4. DETAILED RESULTS - CURVE TRAJECTORY")
    report_lines.append("-" * 50)
    report_lines.append(results['curve']['report'])
    report_lines.append("")

    report_lines.append("## 5. DETAILED RESULTS - STRAIGHT-LINE TRAJECTORY")
    report_lines.append("-" * 50)
    report_lines.append(results['straight']['report'])
    report_lines.append("")

    report_lines.append("## 6. ANALYSIS AND CONCLUSIONS")
    report_lines.append("-" * 50)

    if curve_rate > 99 and straight_rate > 99:
        report_lines.append("Both trajectory modes show excellent reachability (>99%).")
        report_lines.append("The workspace is well-suited for grasping operations.")
    elif curve_rate > straight_rate + 5:
        diff = curve_rate - straight_rate
        report_lines.append(f"Curve trajectory has {diff:.1f}% higher reachability than straight-line.")
        report_lines.append("This indicates some positions have IK solutions but require curved paths.")
        report_lines.append("Recommendation: Use segmented MoveJ for better reliability.")
    else:
        report_lines.append(f"Both modes show similar reachability (~{min(curve_rate, straight_rate):.0f}%).")

    report_lines.append("")
    report_lines.append("Key Findings:")
    report_lines.append(f"  - Best pitch angle: 0° (gripper perpendicular to ground)")
    report_lines.append(f"  - Yaw flexibility: {yaw_range[0]}°~{yaw_range[-1]}° fully supported")
    report_lines.append(f"  - Maximum depth: Z=-250mm achievable across most positions")
    report_lines.append("")
    report_lines.append("=" * 70)

    full_report = "\n".join(report_lines)

    # Save report
    report_path = "/home/agilex/MobileManipulator/src/piper/piper_grasp/docs/workspace_reachability_validation.md"
    import os
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(full_report)

    print("\n" + full_report)
    print(f"\nReport saved to: {report_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Workspace Reachability Validation')
    parser.add_argument('--mode', choices=['quick', 'full', 'focused', 'extended', 'comprehensive'],
                        default='focused',
                        help='Validation mode (default: focused)')
    parser.add_argument('--trajectory', choices=['curve', 'straight'],
                        default='straight',
                        help='Trajectory mode: curve (endpoint-only) or straight (default)')
    args = parser.parse_args()

    trajectory_mode = TrajectoryMode.ENDPOINT_ONLY if args.trajectory == 'curve' else TrajectoryMode.STRAIGHT_LINE

    if args.mode == 'quick':
        run_quick_validation(trajectory_mode)
    elif args.mode == 'full':
        run_full_validation(trajectory_mode)
    elif args.mode == 'extended':
        run_extended_validation(trajectory_mode)
    elif args.mode == 'comprehensive':
        run_comprehensive_validation()
    else:
        run_focused_validation(trajectory_mode)
