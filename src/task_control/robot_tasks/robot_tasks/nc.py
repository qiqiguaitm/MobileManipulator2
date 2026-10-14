"""Navigation controller: semantic/mock front alignment or supplied grasp position."""

import math, time
import numpy as np
from rclpy.action import ActionServer, ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from robot_task_interfaces.action import Approach
from .ros_support import LocatedNode, Problem, pose_msg, run
from .geometry import (
    front_goal,
    inspection_goals,
    pose_errors,
    target_in_standoff_bounds,
    inspection_base_distance,
)
from .semantic_mock import SemanticMapMock
from .planning_debug import PlanningDebug


class NavigationController(LocatedNode):
    def __init__(self):
        super().__init__("navigation_controller")
        self.debug = PlanningDebug(self)
        self.allow = self.param("allow_execution", False)
        self.require_runtime = self.param("require_motion_runtime", False)
        self.runtime_fault = ""
        self.motion_requested = False
        self.runtime_check = self.create_client(
            Trigger, "/robot_tasks/runtime/check", callback_group=self.group
        )
        self.motion_permit = self.create_publisher(Bool, "/nc/motion_permit", 1)
        self.create_timer(0.1, self.publish_permit, callback_group=self.group)
        self.create_subscription(
            String,
            "/robot_tasks/runtime/fault",
            lambda m: setattr(self, "runtime_fault", m.data),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self.group,
        )
        self.allow_mock = self.param("allow_mock_front_execution", False)
        self.mock = self.param("washer_front_pose", [2.0, 0.0, 0.5, math.pi])
        self.semantic_map = SemanticMapMock(self.mock, self.map_frame)
        self.front_extent = self.param("front_extent", 0.434)
        self.clearance = self.param("front_clearance", 0.5)
        self.grasp_clearance = self.param("grasp_front_clearance", 0.5)
        self.grasp_distance = inspection_base_distance(
            self.front_extent, self.grasp_clearance
        )
        self.grasp_y = self.param("grasp_y_limit", 0.15)
        self.grasp_z = self.param("grasp_z_range", [-0.15, 0.9])
        self.angles = self.param("grasp_angles_deg", [0.0, 30.0, -30.0, 60.0, -60.0])
        self.timeout = self.param("navigation_timeout", 180.0)
        self.target_age = self.param("grasp_target_max_age", 45.0)
        self.xy_tol = self.param("position_tolerance", 0.05)
        self.yaw_tol = self.param("yaw_tolerance", 0.0174533)
        nav_name = self.param("nav_action", "/navigate_to_pose")
        self.nav = ActionClient(
            self, NavigateToPose, nav_name, callback_group=self.group
        )
        self.planner = ActionClient(
            self,
            ComputePathToPose,
            self.param("plan_action", "/compute_path_to_pose"),
            callback_group=self.group,
        )
        self.nav_active = False
        self.create_subscription(
            GoalStatusArray,
            nav_name + "/_action/status",
            self.nav_status,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self.group,
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/nc/navigation_goal", 1)
        self.server = ActionServer(
            self,
            Approach,
            "/nc/approach",
            execute_callback=self.execute,
            goal_callback=self.accept,
            cancel_callback=self.cancel,
            callback_group=self.group,
        )

    def debug_update(self, method, *args):
        try:
            getattr(self.debug, method)(*args)
        except Exception as error:
            self.get_logger().warning(
                "Planning visualization unavailable: " + str(error)
            )

    def nav_status(self, msg):
        self.nav_active = any(s.status in (1, 2, 3) for s in msg.status_list)

    def publish_permit(self):
        self.motion_permit.publish(
            Bool(data=self.motion_requested and not self.closing)
        )

    def navigation_tick(self):
        if self.require_runtime and self.runtime_fault:
            raise Problem("MOTION_GUARD", self.runtime_fault)
        return self.robot_pose()

    def feedback(self, h, state, message=""):
        f = Approach.Feedback()
        f.state = state
        f.message = message
        h.publish_feedback(f)

    def initial_robot_pose(self, h):
        # Action discovery can finish before this node receives its first TF.
        # Wait only before planning; a lost transform during motion still fails immediately.
        deadline = time.monotonic() + 5.0
        while True:
            self.check(h)
            try:
                return self.robot_pose()
            except Problem as e:
                if e.code != "LOCALIZATION" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def execute(self, h):
        result = Approach.Result()
        result.position_error = float("nan")
        result.yaw_error = float("nan")
        problem = None
        try:
            r = h.request
            if r.mode not in ("front_align", "grasp_ready"):
                raise Problem("INVALID_MODE", "Expected front_align or grasp_ready")
            if r.execute and not self.allow:
                raise Problem("EXECUTION_DISABLED", "NC execution is disabled")
            if r.execute and self.require_runtime:
                ready = self.call(self.runtime_check, Trigger.Request(), h, 3.0)
                if not ready.success:
                    raise Problem("MOTION_GUARD", ready.message)
            if self.nav_active:
                raise Problem("NAV_BUSY", "Another Nav2 goal is active")
            robot, robot_z = self.initial_robot_pose(h)
            target = None
            if r.mode == "front_align":
                try:
                    frame, front = self.semantic_map.lookup_front_pose(r.name)
                except KeyError as e:
                    raise Problem("TARGET_UNKNOWN", str(e)) from None
                except ValueError as e:
                    raise Problem("CONFIG", str(e)) from None
                if frame != self.map_frame:
                    raise Problem(
                        "TARGET_FRAME",
                        "Semantic pose frame must match the navigation map",
                    )
                if r.execute and not self.allow_mock:
                    raise Problem(
                        "MOCK_TARGET",
                        "Washer pose is a mock; real execution requires a verified configured pose and explicit enable",
                    )
                candidates = [
                    front_goal(front[:3], front[3], self.clearance, self.front_extent)
                ]
            else:
                if r.target.header.frame_id != self.map_frame:
                    raise Problem(
                        "TARGET_FRAME", "GC must supply map-frame object position"
                    )
                stamp = r.target.header.stamp.sec + r.target.header.stamp.nanosec * 1e-9
                age = self.get_clock().now().nanoseconds / 1e9 - stamp
                if not -0.1 <= age <= self.target_age:
                    raise Problem(
                        "TARGET_STALE", f"Object observation too old: {age:.1f}s"
                    )
                p = r.target.point
                target = (p.x, p.y, p.z)
                if (
                    not np.isfinite(target).all()
                    or not self.grasp_z[0] <= p.z - robot_z <= self.grasp_z[1]
                ):
                    raise Problem(
                        "UNREACHABLE_HEIGHT",
                        "Object height is outside configured inspection bounds",
                    )
                candidates = inspection_goals(
                    target,
                    robot,
                    self.grasp_distance,
                    [math.radians(v) for v in self.angles],
                )
            visual_target = target if target is not None else front[:3]
            self.debug_update("begin", robot, visual_target, candidates, r.name, h)
            self.feedback(h, "PLANNING")
            choices = []
            for candidate_index, candidate in enumerate(candidates):
                self.check(h)
                self.robot_pose()
                goal = pose_msg(
                    candidate, self.map_frame, self.get_clock().now().to_msg()
                )
                req = ComputePathToPose.Goal()
                req.goal = goal
                req.use_start = False
                try:
                    planned = self.action(self.planner, req, h, 10.0)
                except Problem as e:
                    if e.code == "CHILD_FAILED":
                        self.debug_update("candidate", candidate_index, "NO_PATH")
                        continue
                    raise
                points = planned.path.poses
                if not points:
                    self.debug_update("candidate", candidate_index, "EMPTY_PATH")
                    continue
                end = points[-1].pose.position
                if math.hypot(end.x - candidate[0], end.y - candidate[1]) > 0.1:
                    self.debug_update(
                        "candidate", candidate_index, "ENDPOINT_TOO_FAR", points
                    )
                    continue
                self.debug_update("candidate", candidate_index, "REACHABLE", points)
                length = sum(
                    math.hypot(
                        b.pose.position.x - a.pose.position.x,
                        b.pose.position.y - a.pose.position.y,
                    )
                    for a, b in zip(points, points[1:])
                )
                choices.append((length, candidate, goal))
            if not choices:
                self.debug_update("finish", "NO_PATH")
                raise Problem("NO_PATH", "No reachable stopping pose")
            _, candidate, goal = min(choices, key=lambda c: c[0])
            result.navigation_goal = goal
            self.goal_pub.publish(goal)
            self.debug_update("finish", "SELECTED", candidates.index(candidate))
            if not r.execute:
                result.success = True
                result.code = "PREVIEW"
                result.message = (
                    "Reachable stopping pose computed; no motion or arrival claimed"
                )
            else:
                if self.nav_active:
                    raise Problem(
                        "NAV_BUSY", "Another Nav2 goal started during planning"
                    )
                self.feedback(h, "NAVIGATING")
                req = NavigateToPose.Goal()
                req.pose = goal
                self.motion_requested = True
                self.publish_permit()
                try:
                    self.action(
                        self.nav, req, h, self.timeout, tick=self.navigation_tick
                    )
                except Problem as e:
                    if e.code == "CHILD_FAILED":
                        raise Problem("NAV_FAILED", str(e)) from None
                    raise
                self.feedback(h, "VERIFYING")
                actual, z = self.robot_pose()
                xy, yaw = pose_errors(actual, candidate)
                result.position_error = xy
                result.yaw_error = yaw
                if xy > self.xy_tol or yaw > self.yaw_tol:
                    raise Problem(
                        "ARRIVAL_ERROR",
                        f"Pose errors: distance={xy:.3f}m yaw={math.degrees(yaw):.2f}deg",
                    )
                # This checks navigation stand-off only, not the arm's physical reach.
                standoff_x = [
                    self.grasp_distance - self.xy_tol,
                    self.grasp_distance + self.xy_tol,
                ]
                if target is not None and not target_in_standoff_bounds(
                    (target[0], target[1], target[2] - z),
                    actual,
                    standoff_x,
                    self.grasp_y,
                    self.grasp_z,
                ):
                    raise Problem(
                        "STANDOFF_ERROR",
                        "Target is outside configured front-clearance tolerance",
                    )
                result.success = True
                result.code = "ARRIVED"
                result.message = "Map-relative approach verified; grasp feasibility belongs to manipulation"
        except Exception as e:
            problem = e
        finally:
            self.motion_requested = False
            self.publish_permit()
        return self.finish(h, result, problem)


def main():
    run(NavigationController)
