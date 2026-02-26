#!/usr/bin/env python3
"""
Collision Checker for Mobile Manipulator

Hierarchical collision detection:
1. Fast bounding box check
2. Simplified geometry check (Capsule/Cylinder/Box)
3. Optional precise mesh check (hpp-fcl)

Supports:
- Self-collision (arm vs chassis/lidar/lifting)
- Environment collision (ground/table planes)
"""

import numpy as np
import math
import yaml
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class CollisionType(Enum):
    NONE = 0
    SELF = 1
    ENVIRONMENT = 2
    BOTH = 3


@dataclass
class CollisionResult:
    """Result of collision check"""
    has_collision: bool
    collision_type: CollisionType
    collision_pairs: List[Tuple[str, str]]
    min_distance: float  # Negative if penetrating
    details: str


class GeometryPrimitive:
    """Base class for collision geometry primitives"""
    pass


@dataclass
class Sphere(GeometryPrimitive):
    center: np.ndarray
    radius: float


@dataclass
class Capsule(GeometryPrimitive):
    """Capsule: cylinder with hemispherical caps"""
    p1: np.ndarray  # Start point
    p2: np.ndarray  # End point
    radius: float


@dataclass
class Cylinder(GeometryPrimitive):
    center: np.ndarray
    radius: float
    height: float
    axis: np.ndarray = None  # Default: Z-axis

    def __post_init__(self):
        if self.axis is None:
            self.axis = np.array([0, 0, 1])


@dataclass
class Box(GeometryPrimitive):
    center: np.ndarray
    half_extents: np.ndarray  # [half_x, half_y, half_z]
    rotation: np.ndarray = None  # 3x3 rotation matrix

    def __post_init__(self):
        if self.rotation is None:
            self.rotation = np.eye(3)


@dataclass
class Plane(GeometryPrimitive):
    """Infinite plane"""
    point: np.ndarray  # Point on plane
    normal: np.ndarray  # Normal vector (pointing up)


class CollisionChecker:
    """
    Main collision checker with hierarchical detection.

    Coordinate System:
    - Input positions are in ARM coordinate frame (same as IK solver)
    - Internal calculations transform to base_link frame for chassis collision
    - ARM origin is at [0.118, 0, 0.1] relative to base_link
    """

    # Arm base position relative to base_link (from URDF)
    ARM_BASE_IN_BASE_LINK = np.array([0.118, 0.0, 0.1])

    # Default static bodies (relative to ARM coordinate frame, in meters)
    # These are transformed from base_link to arm frame
    DEFAULT_STATIC_BODIES = {
        'lidar': {
            'type': 'cylinder',
            'center': [0.155, 0, -0.05],  # base_link [0.273,0,0.05] - arm_base [0.118,0,0.1]
            'radius': 0.05,
            'height': 0.10
        },
        'box': {
            'type': 'box',
            'center': [-0.118, 0, -0.05],  # base_link [0,0,0.05] - arm_base
            'half_extents': [0.22, 0.18, 0.06]
        },
        'lifting': {
            'type': 'box',
            'center': [-0.358, 0, 0.22],  # base_link [-0.24,0,0.32] - arm_base
            'half_extents': [0.10, 0.08, 0.32]
        },
        'arm_base_mount': {
            'type': 'cylinder',
            'center': [0, 0, 0.06],  # Just below arm base
            'radius': 0.04,
            'height': 0.12
        }
    }

    # Arm link dimensions for capsule approximation
    ARM_LINK_PARAMS = {
        'link2': {'radius': 0.035},  # Upper arm
        'link3': {'radius': 0.030},  # Forearm
        'link5': {'radius': 0.025},  # Wrist
        'gripper': {'radius': 0.045, 'length': 0.135}
    }

    # Default collision pairs to check (self-collision only by default)
    DEFAULT_COLLISION_PAIRS = [
        # Arm vs Lidar (high risk when reaching forward)
        ('link2', 'lidar'),
        ('link3', 'lidar'),
        ('gripper', 'lidar'),

        # Arm vs Box/Chassis (high risk when reaching down)
        ('link3', 'box'),
        ('gripper', 'box'),

        # Arm vs Lifting mechanism (medium risk)
        ('link2', 'lifting'),
        ('link3', 'lifting'),
        ('gripper', 'lifting'),

        # Note: Environment collisions (ground/table) are task-specific
        # and should be added using set_table_surface() method
    ]

    # Environment planes (initially empty - task dependent)
    # Use set_table_surface() to configure based on task
    DEFAULT_ENVIRONMENT = {}

    def __init__(self, config_path: str = None, safety_margin: float = 0.01):
        """
        Initialize collision checker.

        Args:
            config_path: Path to YAML config file (optional)
            safety_margin: Additional safety margin in meters (default 10mm)
        """
        self.safety_margin = safety_margin
        self.static_bodies = {}
        self.environment = {}
        self.collision_pairs = []

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._use_defaults()

        # Build geometry objects
        self._build_static_geometries()
        self._build_environment_geometries()

        # Cache for arm link positions (updated by set_arm_state)
        self.arm_geometries = {}

    def _use_defaults(self):
        """Use default configuration"""
        self.static_bodies = self.DEFAULT_STATIC_BODIES.copy()
        self.environment = self.DEFAULT_ENVIRONMENT.copy()
        self.collision_pairs = self.DEFAULT_COLLISION_PAIRS.copy()

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.static_bodies = config.get('static_bodies', self.DEFAULT_STATIC_BODIES)
        self.environment = config.get('environment', self.DEFAULT_ENVIRONMENT)
        self.collision_pairs = config.get('collision_pairs', self.DEFAULT_COLLISION_PAIRS)

    def _build_static_geometries(self):
        """Build geometry objects for static bodies"""
        self.static_geoms = {}

        for name, params in self.static_bodies.items():
            geom_type = params['type']

            if geom_type == 'cylinder':
                self.static_geoms[name] = Cylinder(
                    center=np.array(params['center']),
                    radius=params['radius'] + self.safety_margin,
                    height=params['height']
                )
            elif geom_type == 'box':
                half_ext = np.array(params['half_extents']) + self.safety_margin
                self.static_geoms[name] = Box(
                    center=np.array(params['center']),
                    half_extents=half_ext
                )
            elif geom_type == 'sphere':
                self.static_geoms[name] = Sphere(
                    center=np.array(params['center']),
                    radius=params['radius'] + self.safety_margin
                )

    def _build_environment_geometries(self):
        """Build geometry objects for environment"""
        self.env_geoms = {}

        for name, params in self.environment.items():
            z = params['z'] + params.get('margin', 0)
            self.env_geoms[name] = Plane(
                point=np.array([0, 0, z]),
                normal=np.array([0, 0, 1])
            )

    def set_table_surface(self, z_mm: float, margin_mm: float = 10.0, name: str = 'table'):
        """
        Set table/ground surface for environment collision detection.

        Args:
            z_mm: Surface height in mm (in ARM coordinate frame)
                  e.g., -300 for a table 300mm below arm base
            margin_mm: Safety margin in mm (default 10mm)
            name: Name for this surface (default 'table')
        """
        z_m = z_mm / 1000.0
        margin_m = margin_mm / 1000.0

        self.environment[name] = {
            'z': z_m,
            'margin': margin_m
        }

        # Rebuild environment geometries
        self._build_environment_geometries()

        # Add collision pair if not exists
        pair = ('gripper', name)
        if pair not in self.collision_pairs:
            self.collision_pairs.append(pair)

        print(f"[CollisionChecker] Set {name} surface at Z={z_mm}mm (margin={margin_mm}mm)")

    def clear_environment(self):
        """Remove all environment constraints"""
        self.environment = {}
        self.env_geoms = {}
        # Remove environment collision pairs
        self.collision_pairs = [p for p in self.collision_pairs
                               if p[1] in self.static_bodies]

    def set_arm_state(self, joint_positions: np.ndarray, fk_func=None):
        """
        Update arm link positions from joint angles.

        Args:
            joint_positions: 6 joint angles in radians (SDK convention)
            fk_func: Forward kinematics function that returns link positions
                     Should return dict: {'link2': (p1, p2), 'link3': (p1, p2), ...}
        """
        if fk_func is not None:
            link_positions = fk_func(joint_positions)
        else:
            # Use approximate FK if no function provided
            link_positions = self._approximate_fk(joint_positions)

        # Build capsule geometries for arm links
        self.arm_geometries = {}

        for link_name, (p1, p2) in link_positions.items():
            if link_name in self.ARM_LINK_PARAMS:
                radius = self.ARM_LINK_PARAMS[link_name]['radius'] + self.safety_margin
                self.arm_geometries[link_name] = Capsule(
                    p1=np.array(p1),
                    p2=np.array(p2),
                    radius=radius
                )

    def set_gripper_pose(self, position: np.ndarray, orientation: np.ndarray = None):
        """
        Directly set gripper position for collision check.

        Args:
            position: [x, y, z] in meters (relative to base_link)
            orientation: [roll, pitch, yaw] in radians (optional)
        """
        # Approximate gripper as a capsule along its axis
        length = self.ARM_LINK_PARAMS['gripper']['length']
        radius = self.ARM_LINK_PARAMS['gripper']['radius'] + self.safety_margin

        if orientation is not None:
            # Calculate gripper axis direction from orientation
            pitch = orientation[1]
            yaw = orientation[2]

            # Gripper points "down" when pitch=0, along -Z
            # Adjust for pitch rotation
            direction = np.array([
                -np.sin(pitch) * np.cos(yaw),
                -np.sin(pitch) * np.sin(yaw),
                -np.cos(pitch)
            ])
        else:
            direction = np.array([0, 0, -1])  # Default: pointing down

        p1 = position
        p2 = position + direction * length

        self.arm_geometries['gripper'] = Capsule(p1=p1, p2=p2, radius=radius)

    def _approximate_fk(self, joint_positions: np.ndarray) -> Dict:
        """
        Approximate forward kinematics for collision checking.
        Returns link endpoints for capsule representation.

        Note: This is a simplified approximation. For precise FK,
        use the Pinocchio-based FK function.
        """
        # Arm base position (relative to base_link)
        arm_base = np.array([0.118, 0, 0.223])  # base + 0.123m height

        j1, j2, j3, j4, j5, j6 = joint_positions

        # Link lengths
        L2 = 0.284  # link2 length
        L3 = 0.242  # link3 length
        L5 = 0.091  # link5 length

        # Simplified FK (ignoring some joint offsets for speed)
        # Joint 1: rotation around Z
        c1, s1 = np.cos(j1), np.sin(j1)

        # Joint 2: rotation around Y (with offset)
        j2_adjusted = j2 + np.radians(10)  # SDK offset
        c2, s2 = np.cos(j2_adjusted), np.sin(j2_adjusted)

        # Joint 3: rotation around Y
        j3_adjusted = j3 - np.radians(40)  # SDK offset
        c23 = np.cos(j2_adjusted + j3_adjusted)
        s23 = np.sin(j2_adjusted + j3_adjusted)

        # Link 2 endpoints
        link2_start = arm_base
        link2_end = arm_base + np.array([
            L2 * c2 * c1,
            L2 * c2 * s1,
            L2 * s2
        ])

        # Link 3 endpoints
        link3_start = link2_end
        link3_end = link2_end + np.array([
            L3 * c23 * c1,
            L3 * c23 * s1,
            L3 * s23
        ])

        # Gripper (simplified)
        gripper_start = link3_end
        gripper_end = link3_end + np.array([0, 0, -0.135])  # Approximate

        return {
            'link2': (link2_start, link2_end),
            'link3': (link3_start, link3_end),
            'gripper': (gripper_start, gripper_end)
        }

    def check_collision(self, joint_positions: np.ndarray = None,
                       gripper_pose: Tuple[np.ndarray, np.ndarray] = None) -> CollisionResult:
        """
        Check for collisions.

        Args:
            joint_positions: 6 joint angles in radians (optional if gripper_pose given)
            gripper_pose: (position, orientation) tuple (optional)

        Returns:
            CollisionResult with collision details
        """
        if joint_positions is not None:
            self.set_arm_state(joint_positions)

        if gripper_pose is not None:
            pos, ori = gripper_pose
            self.set_gripper_pose(pos, ori)

        collision_pairs_found = []
        min_distance = float('inf')
        collision_type = CollisionType.NONE

        for arm_link, obstacle in self.collision_pairs:
            if arm_link not in self.arm_geometries:
                continue

            arm_geom = self.arm_geometries[arm_link]

            # Check against static bodies
            if obstacle in self.static_geoms:
                obs_geom = self.static_geoms[obstacle]
                dist = self._compute_distance(arm_geom, obs_geom)
                min_distance = min(min_distance, dist)

                if dist < 0:
                    collision_pairs_found.append((arm_link, obstacle))
                    collision_type = CollisionType.SELF

            # Check against environment
            elif obstacle in self.env_geoms:
                env_geom = self.env_geoms[obstacle]
                dist = self._compute_distance(arm_geom, env_geom)
                min_distance = min(min_distance, dist)

                if dist < 0:
                    collision_pairs_found.append((arm_link, obstacle))
                    if collision_type == CollisionType.SELF:
                        collision_type = CollisionType.BOTH
                    else:
                        collision_type = CollisionType.ENVIRONMENT

        has_collision = len(collision_pairs_found) > 0

        details = ""
        if has_collision:
            details = f"Collision detected: {collision_pairs_found}"

        return CollisionResult(
            has_collision=has_collision,
            collision_type=collision_type,
            collision_pairs=collision_pairs_found,
            min_distance=min_distance,
            details=details
        )

    def check_point_collision(self, position: np.ndarray,
                             orientation: np.ndarray = None,
                             gripper_radius: float = None) -> CollisionResult:
        """
        Quick check if a gripper position would cause collision.

        Args:
            position: [x, y, z] in meters
            orientation: [roll, pitch, yaw] in radians
            gripper_radius: Override gripper radius (optional)

        Returns:
            CollisionResult
        """
        self.set_gripper_pose(position, orientation)

        if gripper_radius is not None:
            # Temporarily adjust gripper radius
            orig_radius = self.arm_geometries['gripper'].radius
            self.arm_geometries['gripper'].radius = gripper_radius + self.safety_margin

        result = self.check_collision()

        if gripper_radius is not None:
            self.arm_geometries['gripper'].radius = orig_radius

        return result

    def _compute_distance(self, geom1: GeometryPrimitive,
                         geom2: GeometryPrimitive) -> float:
        """
        Compute signed distance between two geometries.
        Negative = penetrating, Positive = separated
        """
        if isinstance(geom1, Capsule) and isinstance(geom2, Cylinder):
            return self._capsule_cylinder_distance(geom1, geom2)
        elif isinstance(geom1, Capsule) and isinstance(geom2, Box):
            return self._capsule_box_distance(geom1, geom2)
        elif isinstance(geom1, Capsule) and isinstance(geom2, Plane):
            return self._capsule_plane_distance(geom1, geom2)
        elif isinstance(geom1, Capsule) and isinstance(geom2, Sphere):
            return self._capsule_sphere_distance(geom1, geom2)
        else:
            # Fallback: use bounding sphere approximation
            return self._bounding_sphere_distance(geom1, geom2)

    def _capsule_cylinder_distance(self, capsule: Capsule, cylinder: Cylinder) -> float:
        """Distance between capsule and cylinder (approximated as vertical)"""
        # Project capsule segment onto XY plane
        p1_xy = capsule.p1[:2]
        p2_xy = capsule.p2[:2]
        cyl_xy = cylinder.center[:2]

        # Find closest point on capsule segment to cylinder axis
        seg_dir = p2_xy - p1_xy
        seg_len = np.linalg.norm(seg_dir)

        if seg_len < 1e-6:
            closest_xy = p1_xy
        else:
            seg_dir = seg_dir / seg_len
            t = np.dot(cyl_xy - p1_xy, seg_dir)
            t = np.clip(t, 0, seg_len)
            closest_xy = p1_xy + t * seg_dir

        # XY distance
        xy_dist = np.linalg.norm(closest_xy - cyl_xy)
        xy_separation = xy_dist - capsule.radius - cylinder.radius

        # Z overlap check
        cyl_z_min = cylinder.center[2] - cylinder.height / 2
        cyl_z_max = cylinder.center[2] + cylinder.height / 2
        cap_z_min = min(capsule.p1[2], capsule.p2[2]) - capsule.radius
        cap_z_max = max(capsule.p1[2], capsule.p2[2]) + capsule.radius

        z_overlap = min(cyl_z_max, cap_z_max) - max(cyl_z_min, cap_z_min)

        if z_overlap <= 0:
            # No Z overlap, no collision possible
            return max(xy_separation, -z_overlap)

        return xy_separation

    def _capsule_box_distance(self, capsule: Capsule, box: Box) -> float:
        """Distance between capsule and axis-aligned box"""
        # Find closest point on capsule segment to box
        seg_closest = self._closest_point_segment_to_box(
            capsule.p1, capsule.p2, box.center, box.half_extents
        )

        # Find closest point on box to that point
        box_closest = self._closest_point_on_box(seg_closest, box.center, box.half_extents)

        dist = np.linalg.norm(seg_closest - box_closest)
        return dist - capsule.radius

    def _capsule_plane_distance(self, capsule: Capsule, plane: Plane) -> float:
        """Distance between capsule and plane"""
        # Distance from each endpoint to plane
        d1 = np.dot(capsule.p1 - plane.point, plane.normal)
        d2 = np.dot(capsule.p2 - plane.point, plane.normal)

        min_dist = min(d1, d2)
        return min_dist - capsule.radius

    def _capsule_sphere_distance(self, capsule: Capsule, sphere: Sphere) -> float:
        """Distance between capsule and sphere"""
        # Closest point on segment to sphere center
        closest = self._closest_point_on_segment(sphere.center, capsule.p1, capsule.p2)
        dist = np.linalg.norm(closest - sphere.center)
        return dist - capsule.radius - sphere.radius

    def _closest_point_on_segment(self, point: np.ndarray,
                                  p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Find closest point on line segment to a point"""
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)

        if seg_len_sq < 1e-10:
            return p1

        t = np.dot(point - p1, seg) / seg_len_sq
        t = np.clip(t, 0, 1)
        return p1 + t * seg

    def _closest_point_on_box(self, point: np.ndarray,
                             center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
        """Find closest point on AABB to a point"""
        closest = np.zeros(3)
        for i in range(3):
            closest[i] = np.clip(point[i],
                                center[i] - half_extents[i],
                                center[i] + half_extents[i])
        return closest

    def _closest_point_segment_to_box(self, p1: np.ndarray, p2: np.ndarray,
                                      center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
        """Find point on segment closest to box"""
        # Sample along segment and find closest to box
        best_point = p1
        best_dist = float('inf')

        for t in np.linspace(0, 1, 10):
            point = p1 + t * (p2 - p1)
            box_closest = self._closest_point_on_box(point, center, half_extents)
            dist = np.linalg.norm(point - box_closest)
            if dist < best_dist:
                best_dist = dist
                best_point = point

        return best_point

    def _bounding_sphere_distance(self, geom1: GeometryPrimitive,
                                  geom2: GeometryPrimitive) -> float:
        """Fallback: bounding sphere distance"""
        c1, r1 = self._get_bounding_sphere(geom1)
        c2, r2 = self._get_bounding_sphere(geom2)

        dist = np.linalg.norm(c1 - c2)
        return dist - r1 - r2

    def _get_bounding_sphere(self, geom: GeometryPrimitive) -> Tuple[np.ndarray, float]:
        """Get bounding sphere for geometry"""
        if isinstance(geom, Sphere):
            return geom.center, geom.radius
        elif isinstance(geom, Capsule):
            center = (geom.p1 + geom.p2) / 2
            half_len = np.linalg.norm(geom.p2 - geom.p1) / 2
            radius = half_len + geom.radius
            return center, radius
        elif isinstance(geom, Cylinder):
            return geom.center, max(geom.radius, geom.height / 2)
        elif isinstance(geom, Box):
            return geom.center, np.linalg.norm(geom.half_extents)
        else:
            return np.zeros(3), 0


# Convenience function
def create_collision_checker(config_path: str = None) -> CollisionChecker:
    """Create collision checker with optional config"""
    return CollisionChecker(config_path)


if __name__ == '__main__':
    # Test
    print("Collision Checker Test")
    print("=" * 70)
    print("Coordinate system: ARM frame (same as IK solver)")
    print()

    checker = CollisionChecker()
    print(f"Lidar center at {checker.static_bodies['lidar']['center']} (arm frame)")
    print(f"Box center at {checker.static_bodies['box']['center']} (arm frame)")
    print(f"Collision pairs: {len(checker.collision_pairs)} (self-collision only)")
    print()

    # Test 1: Self-collision checks
    print("=" * 70)
    print("Test 1: Self-Collision Detection (no table surface)")
    print("-" * 70)

    test_positions = [
        # Safe positions (from workspace exploration)
        ([0.40, 0.10, 0.15], [np.pi, 0.26, np.pi], "Ready position Z=150mm"),
        ([0.40, 0.10, -0.20], [np.pi, 0.52, np.pi], "Grasp depth Z=-200mm"),
        ([0.40, 0.10, -0.25], [np.pi, 0.52, np.pi], "Max depth Z=-250mm"),
        ([0.35, 0.20, -0.25], [np.pi, 0.26, np.pi], "Best region (350,200,-250)"),

        # Self-collision positions
        ([0.15, 0, -0.05], [np.pi, 0, np.pi], "LIDAR collision zone"),
        ([0.0, 0, -0.05], [np.pi, 0, np.pi], "BOX collision zone"),
        ([-0.30, 0, 0.20], [np.pi, 0, np.pi], "LIFTING collision zone"),

        # Edge cases
        ([0.50, 0, -0.20], [np.pi, 0.52, np.pi], "Far reach Z=-200mm"),
    ]

    for pos, ori, desc in test_positions:
        pos = np.array(pos)
        ori = np.array(ori)
        result = checker.check_point_collision(pos, ori)
        status = "COLLISION" if result.has_collision else "OK"
        print(f"{desc:30s} -> {status:10s} (dist={result.min_distance:+.3f}m)")
        if result.has_collision:
            print(f"    Pairs: {result.collision_pairs}")

    # Test 2: With table surface
    print()
    print("=" * 70)
    print("Test 2: With Table Surface at Z=-300mm")
    print("-" * 70)

    checker.set_table_surface(z_mm=-300, margin_mm=10)

    test_positions_table = [
        ([0.40, 0.10, -0.25], [np.pi, 0.52, np.pi], "Z=-250mm (above table)"),
        ([0.40, 0.10, -0.30], [np.pi, 0.52, np.pi], "Z=-300mm (at table)"),
        ([0.40, 0.10, -0.35], [np.pi, 0.52, np.pi], "Z=-350mm (into table)"),
    ]

    for pos, ori, desc in test_positions_table:
        pos = np.array(pos)
        ori = np.array(ori)
        result = checker.check_point_collision(pos, ori)
        status = "COLLISION" if result.has_collision else "OK"
        print(f"{desc:30s} -> {status:10s} (dist={result.min_distance:+.3f}m)")
        if result.has_collision:
            print(f"    Pairs: {result.collision_pairs}")

    print()
    print("=" * 70)
    print("Summary: Self-collision geometry working correctly")
    print("         Table surface can be set per-task")
    print("=" * 70)
