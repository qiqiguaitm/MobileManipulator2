"""
PickStateMachine — blocking state machine for autonomous pick-and-place.

Flow: PLANNING → NAVIGATING → PICKING (展臂→observe→pick→place→收臂) → PLANNING

Runs in a dedicated daemon thread. Communicates with ROS2 services/actions
via call_async() + polling (never spin_until_future_complete).
"""

import time
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from geometry_msgs.msg import Point
from piper_msgs.srv import Observe, GoReady, GoZero
from piper_msgs.action import PiperPick, PiperPlace


class PickState(IntEnum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    PICKING = 3       # 展臂 → observe → pick → place → 收臂
    ERROR = 4
    COMPLETED = 5


PICK_STATE_NAMES = {s: s.name for s in PickState}


@dataclass
class PickConfig:
    max_picks_per_nav: int = 10
    pick_speed: int = 30
    lift_height: float = 200.0
    observe_prompt: str = ""  # empty = use piper_grasp_node default_prompt
    observe_timeout: float = 10.0
    pick_timeout: float = 60.0
    place_timeout: float = 30.0
    nav_timeout: float = 120.0
    max_consecutive_failures: int = 3
    error_cooldown: float = 5.0
    max_error_retries: int = 3
    wait_for_first_target_timeout: float = 15.0


@dataclass
class PickContext:
    """Mutable context shared across state handlers."""
    current_target: object = None           # TargetRecord
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
            node: CleanerManagerNode (provides clients, logger, TF, get_robot_pos)
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
            PickState.NAVIGATING: self._do_navigating,
            PickState.PICKING:    self._do_picking,
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
        # TF 在导航刚结束后可能短暂中断，重试最多 3s
        robot_pos = None
        for _ in range(6):
            robot_pos = self._node.get_robot_pos_map()
            if robot_pos is not None:
                break
            if abort_event.is_set():
                return PickState.IDLE
            time.sleep(0.5)
        if robot_pos is None:
            self._ctx.error_message = "Cannot get robot position"
            return PickState.ERROR

        target = self._pool.get_nav_target(robot_pos)
        if target is None:
            stats = self._pool.stats
            if stats['active'] > 0 or stats['total'] == 0:
                self._log.info("No qualified target yet — waiting for perception...")
                if not self._wait_for_first_target(abort_event):
                    return PickState.COMPLETED
                return PickState.PLANNING
            self._log.info("No more targets — done")
            return PickState.COMPLETED

        self._ctx.current_target = target
        self._log.info(f"Next target: {target.category} @ ({target.position_map.x:.2f}, {target.position_map.y:.2f})")

        # 导航前收臂
        ok = self._safe_stow(abort_event)
        if not ok:
            self._ctx.error_message = "Stow failed"
            return PickState.ERROR
        return PickState.NAVIGATING

    def _wait_for_first_target(self, abort_event) -> bool:
        """Block until get_nav_target returns a qualified target or timeout."""
        deadline = time.time() + self._cfg.wait_for_first_target_timeout
        last_diag = 0.0
        while time.time() < deadline and not abort_event.is_set():
            robot_pos = self._node.get_robot_pos_map()
            if robot_pos is not None and self._pool.get_nav_target(robot_pos) is not None:
                return True
            now = time.time()
            if now - last_diag >= 3.0:
                diag = self._pool.get_filter_diagnostics(robot_pos)
                self._log.info(f"等待目标: {diag}")
                last_diag = now
            time.sleep(0.5)
        diag = self._pool.get_filter_diagnostics(
            self._node.get_robot_pos_map())
        self._log.warn(f"等待目标超时! {diag}")
        return False

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
            self._ctx.consecutive_pick_failures = 0
            return PickState.PICKING

        self._log.warn(f"Nav failed: {result.error_message}")
        self._ctx.consecutive_nav_failures += 1
        self._pool.resume()

        if self._ctx.consecutive_nav_failures >= self._cfg.max_consecutive_failures:
            self._ctx.error_message = f"Nav failed {self._ctx.consecutive_nav_failures} times"
            return PickState.ERROR

        self._pool.mark_failed(target.position_map, f"nav: {result.error_message}")
        return PickState.PLANNING

    def _do_picking(self, abort_event) -> PickState:
        """完整抓取流程：展臂 → (observe → pick → place)* → 收臂"""
        target = self._ctx.current_target

        # --- 1. 展臂 ---
        self._nav._stop_robot()
        self._log.info("展臂...")
        time.sleep(0.5)
        if abort_event.is_set():
            return PickState.IDLE

        ok = self._call_go_ready(self._cfg.pick_speed, open_gripper=True,
                                 abort_event=abort_event)
        if not ok:
            self._ctx.error_message = "Deploy (go_ready) failed"
            return PickState.ERROR

        self._pool.resume()

        # --- 2. observe → pick → place 循环 ---
        picks_done = 0
        gripper_holding = False  # 安全标志：pick成功后置True，place成功后置False
        while picks_done < self._cfg.max_picks_per_nav and not abort_event.is_set():
            self._log.info(f"Observe [{picks_done + 1}]: 手部相机检测...")
            observe_ok = self._call_observe(self._cfg.observe_prompt, abort_event)
            if abort_event.is_set():
                break

            if not observe_ok:
                if picks_done == 0:
                    self._log.warn("手部相机未检测到可抓取物体")
                    if target is not None:
                        self._pool.mark_failed(target.position_map, "no graspable object")
                    self._ctx.consecutive_pick_failures += 1
                else:
                    self._log.info(f"已抓取 {picks_done} 个，无更多目标")
                    if target is not None:
                        self._pool.mark_picked(target.position_map)
                break

            # Pick
            t0 = time.time()
            pick_ok = self._call_pick(abort_event)
            self._ctx.last_pick_time_ms = (time.time() - t0) * 1000.0
            if abort_event.is_set():
                break

            if not pick_ok:
                self._log.warn("Pick failed")
                self._ctx.consecutive_pick_failures += 1
                if self._ctx.consecutive_pick_failures >= self._cfg.max_consecutive_failures:
                    break
                continue  # 重试 observe

            self._ctx.consecutive_pick_failures = 0
            gripper_holding = True  # pick成功，夹爪夹住物体

            # Place — return_to_ready=False，由我们显式控制后续姿态
            place_ok = self._call_place(abort_event)
            if abort_event.is_set():
                break
            if not place_ok:
                self._log.error("Place failed")

            # Place 后显式回 ready 位置（开爪），确保手部相机能继续检测
            if not self._call_go_ready(self._cfg.pick_speed, open_gripper=True,
                                       abort_event=abort_event) or abort_event.is_set():
                break

            gripper_holding = False  # 已开爪
            picks_done += 1

        # --- 3. 收臂 ---
        # 安全检查：若夹爪仍夹持物体，先强制展臂开爪再收臂
        if gripper_holding:
            self._log.error("Safety: 夹爪仍持有物体！强制开爪后再收臂...")
            self._call_go_ready(self._cfg.pick_speed, open_gripper=True, abort_event=abort_event)
            time.sleep(0.3)

        self._log.info("收臂...")
        self._safe_stow(abort_event)

        if abort_event.is_set():
            return PickState.IDLE
        if self._ctx.consecutive_pick_failures >= self._cfg.max_consecutive_failures:
            self._ctx.error_message = "Pick failed consecutively"
            return PickState.ERROR
        return PickState.PLANNING

    def _do_error(self, abort_event) -> PickState:
        self._pool.resume()
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
        """收臂归零 — 用 go_zero 而非 go_ready (go_ready = 展臂)"""
        return self._call_go_zero(abort_event)

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

    def _call_go_zero(self, abort_event) -> bool:
        """收臂到零位 (安全导航姿态)"""
        client = self._node.go_zero_client
        req = GoZero.Request()
        req.is_mit_mode = False

        future = client.call_async(req)
        resp = self._wait_future(future, 30.0, abort_event)
        if resp is None:
            return False
        if not resp.status:
            self._log.warn(f"go_zero failed: code={resp.code}")
        return resp.status

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
        goal.return_to_ready = False  # 由循环显式调 go_ready 控制后续姿态

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
