"""
PickStateMachine — blocking state machine for autonomous pick-and-place.

Runs in a dedicated daemon thread. Communicates with ROS2 services/actions
via call_async() + polling (never spin_until_future_complete).
"""

import time
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from geometry_msgs.msg import Point
from piper_msgs.srv import Observe, GoReady, InWorkingArea
from piper_msgs.action import PiperPick, PiperPlace


class PickState(IntEnum):
    IDLE = 0
    PLANNING = 1
    STOWING = 2
    NAVIGATING = 3
    DEPLOYING = 4
    SCANNING = 5
    PICKING = 6
    PLACING = 7
    ERROR = 8
    COMPLETED = 9


PICK_STATE_NAMES = {s: s.name for s in PickState}


@dataclass
class PickConfig:
    max_picks_per_nav: int = 10
    scan_stable_frames: int = 3
    scan_stable_timeout: float = 3.0
    pick_speed: int = 30
    lift_height: float = 200.0
    observe_prompt: str = "bottle.cup.box"
    observe_timeout: float = 10.0
    pick_timeout: float = 60.0
    place_timeout: float = 30.0
    nav_timeout: float = 120.0
    max_consecutive_failures: int = 3
    error_cooldown: float = 5.0
    max_error_retries: int = 3


@dataclass
class PickContext:
    """Mutable context shared across state handlers."""
    current_target: object = None           # TargetRecord
    workspace_targets: List = field(default_factory=list)
    workspace_idx: int = 0
    consecutive_nav_failures: int = 0
    consecutive_pick_failures: int = 0
    error_retries: int = 0
    last_nav_time_ms: float = 0.0
    last_pick_time_ms: float = 0.0
    error_message: str = ""
    active_goal_handle: object = None       # For abort cancellation


class PickStateMachine:
    """Blocking state machine — call run() from a daemon thread."""

    def __init__(self, node, pool, navigator, config: PickConfig):
        """
        Args:
            node: RobotManagerNode (provides clients, logger, TF, get_robot_pos)
            pool: TargetPool
            navigator: ApproachNavigator
            config: PickConfig
        """
        self._node = node
        self._pool = pool
        self._nav = navigator
        self._cfg = config
        self._log = node.get_logger()

        self._state = PickState.IDLE
        self._ctx = PickContext()
        self._state_lock = threading.Lock()

        self._handlers = {
            PickState.PLANNING:   self._do_planning,
            PickState.STOWING:    self._do_stowing,
            PickState.NAVIGATING: self._do_navigating,
            PickState.DEPLOYING:  self._do_deploying,
            PickState.SCANNING:   self._do_scanning,
            PickState.PICKING:    self._do_picking,
            PickState.PLACING:    self._do_placing,
            PickState.ERROR:      self._do_error,
        }

    @property
    def state(self) -> PickState:
        with self._state_lock:
            return self._state

    @property
    def state_name(self) -> str:
        return PICK_STATE_NAMES.get(self.state, "UNKNOWN")

    @property
    def context(self) -> PickContext:
        return self._ctx

    def run(self, abort_event: threading.Event) -> None:
        """Main loop — blocks until COMPLETED/IDLE or abort."""
        self._log.info("State machine started")
        self._set_state(PickState.PLANNING)
        self._ctx = PickContext()

        while not abort_event.is_set():
            state = self.state
            if state in (PickState.IDLE, PickState.COMPLETED):
                break

            handler = self._handlers.get(state)
            if handler is None:
                self._log.error(f"No handler for state {state}")
                break

            next_state = handler(abort_event)
            if abort_event.is_set():
                break
            self._set_state(next_state)

        if abort_event.is_set():
            self._log.warn("Aborted — stowing arm")
            self._cancel_active_goal()
            self._nav.cancel()
            self._pool.resume()
            self._safe_stow(abort_event=None)  # best-effort stow
            self._set_state(PickState.IDLE)

        self._log.info(f"State machine ended in {self.state_name}")

    def _set_state(self, s: PickState):
        with self._state_lock:
            if self._state != s:
                self._log.info(f"State: {PICK_STATE_NAMES.get(self._state)} -> {PICK_STATE_NAMES.get(s)}")
                self._state = s

    # ------------------------------------------------------------------
    # State handlers — each returns next PickState
    # ------------------------------------------------------------------

    def _do_planning(self, abort_event) -> PickState:
        robot_pos = self._node.get_robot_pos_map()
        if robot_pos is None:
            self._ctx.error_message = "Cannot get robot position"
            return PickState.ERROR

        target = self._pool.get_nav_target(robot_pos)
        if target is None:
            self._log.info("No more targets — done")
            return PickState.COMPLETED

        self._ctx.current_target = target
        self._ctx.workspace_targets = []
        self._ctx.workspace_idx = 0
        self._log.info(f"Next target: {target.category} @ ({target.position_map.x:.2f}, {target.position_map.y:.2f})")
        return PickState.STOWING

    def _do_stowing(self, abort_event) -> PickState:
        ok = self._safe_stow(abort_event)
        if not ok:
            self._ctx.error_message = "Stow failed"
            return PickState.ERROR
        return PickState.NAVIGATING

    def _do_navigating(self, abort_event) -> PickState:
        target = self._ctx.current_target
        self._pool.pause()

        nav_point = Point()
        nav_point.x = target.position_map.x
        nav_point.y = target.position_map.y
        nav_point.z = 0.0  # 2D navigation

        t0 = time.time()
        result = self._nav.approach_to_target(nav_point)
        self._ctx.last_nav_time_ms = (time.time() - t0) * 1000.0

        if abort_event.is_set():
            return PickState.IDLE

        if result.success:
            self._ctx.consecutive_nav_failures = 0
            return PickState.DEPLOYING

        self._log.warn(f"Nav failed: {result.error_message}")
        self._ctx.consecutive_nav_failures += 1
        self._pool.resume()

        if self._ctx.consecutive_nav_failures >= self._cfg.max_consecutive_failures:
            self._ctx.error_message = f"Nav failed {self._ctx.consecutive_nav_failures} times"
            return PickState.ERROR

        self._pool.mark_failed(target.position_map, f"nav: {result.error_message}")
        return PickState.PLANNING

    def _do_deploying(self, abort_event) -> PickState:
        ok = self._call_go_ready(self._cfg.pick_speed, open_gripper=True, abort_event=abort_event)
        if not ok:
            self._ctx.error_message = "Deploy (go_ready) failed"
            return PickState.ERROR
        return PickState.SCANNING

    def _do_scanning(self, abort_event) -> PickState:
        self._pool.resume()
        self._wait_perception_stable(abort_event)
        if abort_event.is_set():
            return PickState.IDLE

        robot_pos = self._node.get_robot_pos_map()
        if robot_pos is None:
            self._ctx.error_message = "Cannot get robot position during scan"
            return PickState.ERROR

        targets = self._pool.get_workspace_targets(
            in_working_area_fn=self._check_in_working_area,
            robot_pos_map=robot_pos,
        )

        if not targets:
            self._log.info("No targets in workspace — replan")
            return PickState.PLANNING

        self._ctx.workspace_targets = targets[:self._cfg.max_picks_per_nav]
        self._ctx.workspace_idx = 0
        self._log.info(f"Found {len(self._ctx.workspace_targets)} workspace targets")
        return PickState.PICKING

    def _do_picking(self, abort_event) -> PickState:
        targets = self._ctx.workspace_targets
        idx = self._ctx.workspace_idx

        if idx >= len(targets):
            self._log.info("Workspace queue exhausted — replan")
            return PickState.PLANNING

        target = targets[idx]
        self._ctx.current_target = target
        self._log.info(f"Picking [{idx+1}/{len(targets)}]: {target.category}")

        # Observe
        observe_ok = self._call_observe(self._cfg.observe_prompt, abort_event)
        if abort_event.is_set():
            return PickState.IDLE
        if not observe_ok:
            self._log.warn("Observe failed — skip target")
            self._pool.mark_failed(target.position_map, "observe failed")
            self._ctx.workspace_idx += 1
            self._ctx.consecutive_pick_failures += 1
            if self._ctx.consecutive_pick_failures >= self._cfg.max_consecutive_failures:
                self._ctx.error_message = "Pick failed consecutively"
                return PickState.ERROR
            return PickState.PICKING

        # Pick
        t0 = time.time()
        pick_ok = self._call_pick(abort_event)
        self._ctx.last_pick_time_ms = (time.time() - t0) * 1000.0
        if abort_event.is_set():
            return PickState.IDLE

        if pick_ok:
            self._ctx.consecutive_pick_failures = 0
            return PickState.PLACING

        self._log.warn("Pick failed — skip target")
        self._pool.mark_failed(target.position_map, "pick failed")
        self._ctx.workspace_idx += 1
        self._ctx.consecutive_pick_failures += 1
        if self._ctx.consecutive_pick_failures >= self._cfg.max_consecutive_failures:
            self._ctx.error_message = "Pick failed consecutively"
            return PickState.ERROR
        return PickState.PICKING

    def _do_placing(self, abort_event) -> PickState:
        place_ok = self._call_place(abort_event)
        if abort_event.is_set():
            return PickState.IDLE

        target = self._ctx.current_target
        if place_ok:
            self._pool.mark_picked(target.position_map)
        else:
            self._log.warn("Place failed — mark as failed")
            self._pool.mark_failed(target.position_map, "place failed")

        self._ctx.workspace_idx += 1
        if self._ctx.workspace_idx < len(self._ctx.workspace_targets):
            return PickState.PICKING
        return PickState.PLANNING

    def _do_error(self, abort_event) -> PickState:
        self._log.error(f"ERROR: {self._ctx.error_message}")
        self._ctx.error_retries += 1
        if self._ctx.error_retries >= self._cfg.max_error_retries:
            self._log.error("Max error retries — giving up")
            return PickState.COMPLETED

        self._log.info(f"Cooldown {self._cfg.error_cooldown}s (retry {self._ctx.error_retries}/{self._cfg.max_error_retries})")
        deadline = time.time() + self._cfg.error_cooldown
        while time.time() < deadline and not abort_event.is_set():
            time.sleep(0.1)

        self._ctx.consecutive_nav_failures = 0
        self._ctx.consecutive_pick_failures = 0
        return PickState.PLANNING

    # ------------------------------------------------------------------
    # Piper service/action wrappers (blocking via poll)
    # ------------------------------------------------------------------

    def _wait_future(self, future, timeout: float, abort_event: Optional[threading.Event]):
        deadline = time.time() + timeout
        while not future.done():
            if abort_event and abort_event.is_set():
                return None
            if time.time() > deadline:
                self._log.warn("Service call timed out")
                return None
            time.sleep(0.05)
        return future.result()

    def _safe_stow(self, abort_event) -> bool:
        return self._call_go_ready(self._cfg.pick_speed, open_gripper=False, abort_event=abort_event)

    def _call_go_ready(self, speed: int, open_gripper: bool, abort_event) -> bool:
        client = self._node.go_ready_client
        req = GoReady.Request()
        req.speed = speed
        req.open_gripper = open_gripper

        future = client.call_async(req)
        resp = self._wait_future(future, 30.0, abort_event)
        if resp is None:
            return False
        if not resp.success:
            self._log.warn(f"go_ready failed: {resp.message}")
        return resp.success

    def _call_observe(self, prompt: str, abort_event) -> bool:
        client = self._node.observe_client
        req = Observe.Request()
        req.prompt = prompt
        req.enable_cdm = True

        future = client.call_async(req)
        resp = self._wait_future(future, self._cfg.observe_timeout, abort_event)
        if resp is None:
            return False
        if not resp.success:
            self._log.warn(f"observe failed: {resp.error_message}")
        return resp.success

    def _call_pick(self, abort_event) -> bool:
        client = self._node.pick_client
        goal = PiperPick.Goal()
        goal.use_last_observe = True
        goal.speed = self._cfg.pick_speed
        goal.lift_height = self._cfg.lift_height
        goal.return_to_ready = False

        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 10.0, abort_event)
        if goal_handle is None or not goal_handle.accepted:
            return False

        self._ctx.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        wrapped = self._wait_future(result_future, self._cfg.pick_timeout, abort_event)
        self._ctx.active_goal_handle = None

        if wrapped is None:
            return False
        return wrapped.result.success

    def _call_place(self, abort_event) -> bool:
        client = self._node.place_client
        goal = PiperPlace.Goal()
        goal.use_default_place = True
        goal.speed = self._cfg.pick_speed
        goal.return_to_ready = True

        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 10.0, abort_event)
        if goal_handle is None or not goal_handle.accepted:
            return False

        self._ctx.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        wrapped = self._wait_future(result_future, self._cfg.place_timeout, abort_event)
        self._ctx.active_goal_handle = None

        if wrapped is None:
            return False
        return wrapped.result.success

    def _cancel_active_goal(self):
        gh = self._ctx.active_goal_handle
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            self._ctx.active_goal_handle = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_in_working_area(self, pos_map: Point) -> bool:
        """Check if a map-frame position is in Piper working area.
        This is the ONLY m->mm conversion point.
        """
        # Transform map -> base_link first
        try:
            tf = self._node.tf_buffer.lookup_transform(
                'base_link', 'map',
                self._node.get_clock().now().to_msg(),
                timeout=None
            )
        except Exception:
            return False

        from robot_manager.target_pool import _transform_point
        pos_base = _transform_point(pos_map, tf)

        client = self._node.in_working_area_client
        req = InWorkingArea.Request()
        req.point_in_base = [pos_base.x * 1000.0, pos_base.y * 1000.0, pos_base.z * 1000.0]
        req.yaw = float('nan')
        req.offset = []
        req.point3d_cam = []
        req.end_pose = []

        future = client.call_async(req)
        resp = self._wait_future(future, 5.0, None)
        if resp is None:
            return False
        return resp.in_area

    def _wait_perception_stable(self, abort_event):
        """Wait until tracked object count stabilizes."""
        stable_count = 0
        last_count = -1
        deadline = time.time() + self._cfg.scan_stable_timeout

        while time.time() < deadline and not abort_event.is_set():
            current = self._pool.get_active_count()
            if current == last_count:
                stable_count += 1
            else:
                stable_count = 0
            last_count = current

            if stable_count >= self._cfg.scan_stable_frames:
                return
            time.sleep(0.3)
