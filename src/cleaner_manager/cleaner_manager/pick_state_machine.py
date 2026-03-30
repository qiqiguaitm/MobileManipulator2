"""
PickStateMachine — blocking state machine for autonomous pick-and-place.

Flow: PLANNING → NAVIGATING → PICKING (展臂→observe→pick→place→收臂) → PLANNING

Runs in a dedicated daemon thread. Communicates with ROS2 services/actions
via call_async() + polling (never spin_until_future_complete).
"""

import logging
import math
import os
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Optional

from rclpy.time import Time

def _point_dist_2d(a, b) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
from rclpy.duration import Duration
from geometry_msgs.msg import Point
from piper_msgs.srv import Observe, GoReady, GoZero
from piper_msgs.action import PiperPick, PiperPlace


def _create_session_logger() -> logging.Logger:
    """创建 pick session 文件日志，写到 /tmp/cleaner_pick_<timestamp>.log"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'/tmp/cleaner_pick_{ts}.log'
    logger = logging.getLogger(f'pick_session_{ts}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%H:%M:%S'))
        logger.addHandler(fh)
    return logger


class PickState(IntEnum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    PICKING = 3       # 展臂 → observe → pick → place → 收臂
    ERROR = 4
    COMPLETED = 5
    SCANNING = 6      # 收臂后 3s 观察，目标池仅此时接受新增


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
    scan_duration: float = 2.0             # s, SCANNING 观察时长
    pick_clear_max_dist: float = 0.25      # m, 抓取后清除目标池的最大匹配距离
    max_reapproach: int = 2                # 单次 PICKING 最大 re-approach 次数
    reapproach_max_distance_mm: float = 1000.0  # mm, 超过此距离不 re-approach
    reapproach_depth_timeout: float = 3.0  # s, 深度相机数据等待超时
    reapproach_xmin_mm: float = 350.0     # mm, base x < 此值视为太近，后退修正
    reapproach_xmax_mm: float = 450.0     # mm, base x > 此值视为太远，前进修正 (< working_area xmax, 留100mm余量)
    reapproach_ymax_mm: float = 250.0     # mm, |base y| > 此值侧向无法修正 (= working_area ymax)
    detect_approach_margin: float = 0.50   # m, 前进后目标在 base_link 中的 x 距离 (≈arm working area center)
    default_detect_prompt: str = "can.vegetable.bottle.box.food.Rubik's cube.tool.bread.objects"


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
        self._flog: Optional[logging.Logger] = None  # session file logger

        self._handlers = {
            PickState.PLANNING:   self._do_planning,
            PickState.NAVIGATING: self._do_navigating,
            PickState.PICKING:    self._do_picking,
            PickState.SCANNING:   self._do_scanning,
            PickState.ERROR:      self._do_error,
        }

    def _set_perception(self, paused: bool):
        """控制感知检测暂停/恢复"""
        self._node._set_detection_paused(paused)

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
        self._flog = _create_session_logger()
        self._flog.info("=" * 60)
        self._flog.info("PICK SESSION START")
        self._flog.info("=" * 60)
        self._log.info("State machine started")
        self._pool.pause()  # 仅 SCANNING / _wait_for_first_target 时开放
        self._set_perception(paused=True)
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
            self._flog.warning("ABORTED by user")
            self._cancel_active_goal()
            self._nav.cancel()
            self._safe_stow(abort_event=None)  # best-effort stow
            self._set_state(PickState.IDLE)

        self._pool.resume()  # 状态机结束，恢复
        self._set_perception(paused=False)

        # session summary
        stats = self._pool.stats
        summary = (f"SESSION END: state={self.state_name} "
                   f"picked={stats['picked']} failed={stats['failed']} "
                   f"remaining={stats['remaining']}")
        self._log.info(f"State machine ended in {self.state_name}")
        self._flog.info(summary)
        self._flog.info("=" * 60)
        # close file handler
        for h in self._flog.handlers[:]:
            self._log.info(f"Pick session log: {h.baseFilename}")
            h.close()
            self._flog.removeHandler(h)
        self._flog = None

    def _set_state(self, s: PickState):
        with self._state_lock:
            if self._state != s:
                old = PICK_STATE_NAMES.get(self._state)
                new = PICK_STATE_NAMES.get(s)
                self._log.info(f"State: {old} -> {new}")
                if self._flog:
                    self._flog.info(f"STATE {old} -> {new}")
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

        # 诊断：打印所有候选目标及距离（帮助确认选择了最近的）
        all_viable = self._pool.get_all_active(robot_pos)
        if len(all_viable) > 1:
            cands = " | ".join(
                f"{t.category}({_point_dist_2d(t.position_map, robot_pos):.2f}m)"
                for t in all_viable[:5]
            )
            self._log.info(f"候选目标(按距离): {cands}")
            if self._flog:
                self._flog.info(f"CANDIDATES {cands}")

        # 可达性预检：最多检查 5 个不同目标
        for _ in range(5):
            if target is None:
                break
            nav_point = Point(x=target.position_map.x, y=target.position_map.y, z=0.0)
            if self._nav.is_reachable(nav_point):
                break
            self._log.warn(
                f"目标 {target.category} @ ({target.position_map.x:.2f}, "
                f"{target.position_map.y:.2f}) 不可达，跳过")
            if self._flog:
                self._flog.warning(
                    f"UNREACHABLE {target.category} "
                    f"map=({target.position_map.x:.2f},{target.position_map.y:.2f})")
            self._pool.mark_failed(target.position_map, "unreachable")
            target = self._pool.get_nav_target(robot_pos)

        if target is None:
            self._log.info("所有候选目标均不可达 — done")
            return PickState.COMPLETED

        self._ctx.current_target = target
        self._log.info(
            f"Next target: {target.category} @ "
            f"({target.position_map.x:.2f}, {target.position_map.y:.2f})")
        if self._flog:
            self._flog.info(
                f"TARGET {target.category} "
                f"map=({target.position_map.x:.2f},{target.position_map.y:.2f}) "
                f"obs={target.observation_count} score={target.track_score:.3f}")

        # 导航前收臂
        ok = self._safe_stow(abort_event)
        if not ok:
            self._ctx.error_message = "Stow failed"
            return PickState.ERROR
        return PickState.NAVIGATING

    def _wait_for_first_target(self, abort_event) -> bool:
        """Block until get_nav_target returns a qualified target or timeout.

        After the first target qualifies, continues waiting for scan_duration
        to let more targets accumulate observations — ensuring the nearest
        target is selected, not just the first to qualify.

        目标池在等待期间开放，结束后关闭。
        """
        self._set_perception(paused=False)
        self._pool.resume()
        deadline = time.time() + self._cfg.wait_for_first_target_timeout
        last_diag = 0.0
        settle_deadline = None
        try:
            while time.time() < deadline and not abort_event.is_set():
                robot_pos = self._node.get_robot_pos_map()
                if robot_pos is not None and self._pool.get_nav_target(robot_pos) is not None:
                    if settle_deadline is None:
                        settle_time = self._cfg.scan_duration
                        settle_deadline = min(time.time() + settle_time, deadline)
                        self._log.info(
                            f"首个目标已出现，继续观察 {settle_time:.0f}s 积累更多候选...")
                        if self._flog:
                            self._flog.info(f"FIRST_TARGET_FOUND settle={settle_time:.0f}s")
                    if time.time() >= settle_deadline:
                        return True
                now = time.time()
                if now - last_diag >= 3.0:
                    diag = self._pool.get_filter_diagnostics(robot_pos)
                    self._log.info(f"等待目标: {diag}")
                    last_diag = now
                time.sleep(0.3)
            if settle_deadline is not None:
                return True
            diag = self._pool.get_filter_diagnostics(
                self._node.get_robot_pos_map())
            self._log.warn(f"等待目标超时! {diag}")
            return False
        finally:
            self._pool.pause()
            self._set_perception(paused=True)

    def _do_navigating(self, abort_event) -> PickState:
        target = self._ctx.current_target

        nav_point = Point()
        nav_point.x = target.position_map.x
        nav_point.y = target.position_map.y
        nav_point.z = 0.0  # 2D navigation

        t0 = time.time()
        # Nav2 only, skip depth-based stages 2,3
        result = self._nav.approach_to_target(nav_point, skip_stages={2, 3})
        nav_ms = (time.time() - t0) * 1000.0
        self._ctx.last_nav_time_ms = nav_ms

        if abort_event.is_set():
            return PickState.IDLE

        if not result.success:
            self._log.warn(f"Nav failed: {result.error_message}")
            if self._flog:
                self._flog.warning(
                    f"NAV FAIL {target.category}: {result.error_message} {nav_ms:.0f}ms")
            self._ctx.consecutive_nav_failures += 1

            if self._ctx.consecutive_nav_failures >= self._cfg.max_consecutive_failures:
                self._ctx.error_message = f"Nav failed {self._ctx.consecutive_nav_failures} times"
                return PickState.ERROR

            self._pool.mark_failed(target.position_map, f"nav: {result.error_message}")
            return PickState.PLANNING

        self._ctx.consecutive_nav_failures = 0
        self._ctx.consecutive_pick_failures = 0

        # Detection-based approach (替代 depth 聚类的 stages 2+3)
        prompt = self._cfg.observe_prompt or self._cfg.default_detect_prompt
        approach_ok = self._detect_align_and_approach(prompt, abort_event)
        if not approach_ok:
            self._log.warn("Detection approach 失败，Nav2 已到位，尝试直接抓取")

        if self._flog:
            self._flog.info(
                f"NAV OK {target.category} "
                f"detect_approach={'OK' if approach_ok else 'FAIL'} {nav_ms:.0f}ms")

        return PickState.PICKING

    def _do_picking(self, abort_event) -> PickState:
        """完整抓取流程：展臂 → (observe → pick → place)* → 收臂"""
        target = self._ctx.current_target

        # --- 1. 展臂 ---
        self._nav._stop_robot()
        self._log.info("展臂...")
        time.sleep(0.2)
        if abort_event.is_set():
            return PickState.IDLE

        ok = self._call_go_ready(self._cfg.pick_speed, open_gripper=True,
                                 abort_event=abort_event)
        if not ok:
            self._ctx.error_message = "Deploy (go_ready) failed"
            return PickState.ERROR

        # 目标池和感知在整个流程中保持活跃

        # --- 2. observe → pick → place 循环 ---
        picks_done = 0
        gripper_holding = False  # 安全标志：pick成功后置True，place成功后置False
        local_queue_remaining = 0  # 本地追踪后端队列状态
        observed_map_pos = None    # observe 时一次计算的 map 坐标
        reapproach_count = 0       # re-approach 计数（PICKING 内）
        while picks_done < self._cfg.max_picks_per_nav and not abort_event.is_set():
            # 队列有剩余时跳过 Observe（队列=0时自动重新 Observe = 漏检）
            if local_queue_remaining == 0:
                self._log.info(f"Observe [{picks_done + 1}]: 手部相机检测...")
                observe_ok, obs_pos_mm, obs_cat, too_far = self._call_observe(
                    self._cfg.observe_prompt, abort_event)
                if abort_event.is_set():
                    break

                if not observe_ok:
                    # --- Re-approach: 物体可见但超出工作区 ---
                    if too_far and obs_pos_mm:
                        if self._flog:
                            self._flog.info(
                                f"OBSERVE TOO_FAR {obs_cat} "
                                f"base_dist={sum(x**2 for x in obs_pos_mm[:2])**0.5:.0f}mm")
                        reapproach_ok = self._try_reapproach(
                            obs_pos_mm, obs_cat, reapproach_count, abort_event)
                        if reapproach_ok:
                            reapproach_count += 1
                            continue  # 重新 observe
                        # re-approach 不可行 → 走原有失败逻辑

                    if picks_done == 0:
                        self._log.warn("手部相机未检测到可抓取物体")
                        if self._flog:
                            self._flog.warning("OBSERVE FAIL: no graspable object")
                        if target is not None:
                            self._pool.mark_failed(target.position_map, "no graspable object")
                        self._ctx.consecutive_pick_failures += 1
                    else:
                        self._log.info(f"已抓取 {picks_done} 个，无更多目标")
                    break

                # observe 成功时一次计算 map 位置
                observed_map_pos = self._base_to_map(obs_pos_mm)
                if observed_map_pos:
                    self._log.info(
                        f"Observe {obs_cat} → map({observed_map_pos.x:.2f},{observed_map_pos.y:.2f})")

                # 位置 OK（working_area 已统一为 450mm），进入 pick → 重置计数
                reapproach_count = 0
            else:
                self._log.info(f"Batch pick [{picks_done + 1}]: 队列剩余 {local_queue_remaining}")

            # Pick
            t0 = time.time()
            pick_ok, local_queue_remaining, grasp_pos_mm, pick_cat = self._call_pick(abort_event)
            self._ctx.last_pick_time_ms = (time.time() - t0) * 1000.0
            if abort_event.is_set():
                break

            if not pick_ok:
                self._log.warn("Pick failed")
                if self._flog:
                    self._flog.warning(f"PICK FAIL [{picks_done+1}] {self._ctx.last_pick_time_ms:.0f}ms")
                local_queue_remaining = 0  # 失败时重置，下轮重新 Observe
                self._ctx.consecutive_pick_failures += 1
                if self._ctx.consecutive_pick_failures >= self._cfg.max_consecutive_failures:
                    break
                # 回 ready 位置再重新 observe（避免中间位置导致 WIDTH_REJECT）
                self._call_go_ready(self._cfg.pick_speed, open_gripper=True,
                                    abort_event=abort_event)
                if abort_event.is_set():
                    break
                continue

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

            # 机械臂已归位，更新目标池（不影响 pick→place→go_ready 流程）
            pick_map = observed_map_pos or self._base_to_map(grasp_pos_mm)
            observed_map_pos = None  # 已消费，后续 batch pick 走 fallback
            if pick_map:
                marked, dist, cat = self._pool.mark_picked_nearest(
                    pick_map, max_dist=self._cfg.pick_clear_max_dist)
                if marked:
                    self._log.info(
                        f"抓取 {pick_cat} → 清除池目标 {cat} "
                        f"(dist={dist:.3f}m, map={pick_map.x:.2f},{pick_map.y:.2f})")
                    if self._flog:
                        self._flog.info(
                            f"PICKED {pick_cat} "
                            f"map=({pick_map.x:.2f},{pick_map.y:.2f}) "
                            f"pool_match={cat} dist={dist:.3f}m "
                            f"{self._ctx.last_pick_time_ms:.0f}ms")
                else:
                    dist_str = f"{dist:.3f}" if dist < 1e6 else "无"
                    self._log.info(
                        f"抓取 {pick_cat} @ map({pick_map.x:.2f},{pick_map.y:.2f}): "
                        f"最近池目标 {dist_str}m > {self._cfg.pick_clear_max_dist}m，未清除")
                    if self._flog:
                        self._flog.info(
                            f"PICKED {pick_cat} "
                            f"map=({pick_map.x:.2f},{pick_map.y:.2f}) "
                            f"no_pool_match nearest={dist_str}m")

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
        return PickState.SCANNING

    def _do_scanning(self, abort_event) -> PickState:
        """收臂后静止观察，感知+目标池仅此时开放。"""
        duration = self._cfg.scan_duration
        self._log.info(f"静止观察 {duration:.0f}s (感知+目标池开放)...")
        if self._flog:
            self._flog.info(f"SCANNING {duration:.0f}s")
        self._set_perception(paused=False)
        self._pool.resume()
        deadline = time.time() + duration
        while time.time() < deadline and not abort_event.is_set():
            time.sleep(0.2)
        self._pool.pause()
        self._set_perception(paused=True)
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
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _base_to_map(self, grasp_pos_mm) -> Optional[Point]:
        """Transform grasp position from base_link (mm) to map frame (m)."""
        if not grasp_pos_mm or len(grasp_pos_mm) < 3:
            return None
        try:
            tf = self._node.tf_buffer.lookup_transform(
                'map', 'base_link', Time(),
                timeout=Duration(seconds=1.0))
            # mm → m
            gx = grasp_pos_mm[0] / 1000.0
            gy = grasp_pos_mm[1] / 1000.0
            gz = grasp_pos_mm[2] / 1000.0
            # Quaternion rotation: v' = v + 2w(q×v) + 2(q×(q×v))
            q = tf.transform.rotation
            t = tf.transform.translation
            cx = q.y * gz - q.z * gy
            cy = q.z * gx - q.x * gz
            cz = q.x * gy - q.y * gx
            rx = gx + 2.0 * (q.w * cx + q.y * cz - q.z * cy)
            ry = gy + 2.0 * (q.w * cy + q.z * cx - q.x * cz)
            rz = gz + 2.0 * (q.w * cz + q.x * cy - q.y * cx)
            p = Point()
            p.x = t.x + rx
            p.y = t.y + ry
            p.z = t.z + rz
            return p
        except Exception as e:
            self._log.warn(f"grasp→map变换失败: {e}")
            return None

    # ------------------------------------------------------------------
    # Re-approach logic
    # ------------------------------------------------------------------

    def _try_reapproach(self, obs_pos_mm, obs_cat, reapproach_count,
                        abort_event) -> bool:
        """尝试 re-approach: 基于手部相机观测位置直接旋转+前进。

        obs_pos_mm 是 arm_base_link 坐标 (来自 piper_grasp_node point_in_base)。
        arm_base_link 原点在 base_link 前方 118mm，需要转换后计算。
        """
        cfg = self._cfg
        if cfg.max_reapproach <= 0:
            return False
        if reapproach_count >= cfg.max_reapproach:
            self._log.warn(f"Re-approach 已达上限 ({cfg.max_reapproach})")
            return False

        # obs_pos_mm 是 arm_base_link 坐标，转 base_link
        ARM_BASE_OFFSET_X = 118.0  # mm, arm_base_link 在 base_link 前方
        ax, ay = obs_pos_mm[0], obs_pos_mm[1]
        bx = ax + ARM_BASE_OFFSET_X   # base_link x
        by = ay                        # y 相同
        dist_mm = math.sqrt(bx**2 + by**2)

        self._log.info(
            f"Re-approach [{reapproach_count + 1}/{cfg.max_reapproach}]: "
            f"{obs_cat} @ arm=({ax:.0f},{ay:.0f}) base=({bx:.0f},{by:.0f}) dist={dist_mm:.0f}mm")
        if self._flog:
            self._flog.info(
                f"REAPPROACH [{reapproach_count+1}] {obs_cat} "
                f"arm=({ax:.0f},{ay:.0f}) base=({bx:.0f},{by:.0f}) dist={dist_mm:.0f}")

        # 1. 收臂
        self._log.info("Re-approach: 收臂...")
        if not self._safe_stow(abort_event) or abort_event.is_set():
            return False

        # 2. 朝向修正: 用 base_link 坐标算偏角 (旋转中心是 base_link)
        angle = math.atan2(by, bx)
        if abs(angle) > 0.05:  # >~3° 才修正
            self._log.info(
                f"Re-approach: 朝向修正 {math.degrees(angle):.1f}°")
            ok, msg = self._nav.rotate_by_angle(angle)
            if not ok:
                self._log.warn(f"Re-approach: 旋转失败 — {msg}")
            time.sleep(0.15)

        # 3. 前进: 旋转后物体近似在正前方, x ≈ dist
        #    detect_approach_margin (default 0.50m) 是 base_link 目标距离
        dist_m = dist_mm / 1000.0
        travel = dist_m - cfg.detect_approach_margin
        if travel > 0.03:  # > 3cm 才前进
            self._log.info(
                f"Re-approach: 前进 {travel*100:.0f}cm "
                f"(base距离={dist_m*100:.0f}cm, margin={cfg.detect_approach_margin*100:.0f}cm)")
            self._nav.move_forward(travel, speed=0.08)
            time.sleep(0.15)
        else:
            self._log.info(f"Re-approach: 已足够近 (dist={dist_m*100:.0f}cm)")

        # 4. 展臂 → 回到循环重新 observe
        self._log.info("Re-approach: 展臂，准备重新 observe...")
        if not self._call_go_ready(cfg.pick_speed, open_gripper=True,
                                   abort_event=abort_event) or abort_event.is_set():
            return False

        return True

    # ------------------------------------------------------------------
    # Detection-based approach (底盘相机)
    # ------------------------------------------------------------------

    def _detect_chassis_target(self, prompt: str, abort_event,
                               max_retries: int = 3) -> Optional[tuple]:
        """调用底盘相机检测，返回 (distance, angle, category) 或 None。

        直接使用 position_optical (相机光学系) 和 distance (base_link 距离)，
        不依赖 map 坐标系或 TF 查询。

        optical 坐标系: x=右, y=下, z=前(深度)
        angle: 正=物体偏左=左转, 负=物体偏右=右转
        """
        from perception.srv import DetectObjects

        for attempt in range(max_retries):
            if abort_event and abort_event.is_set():
                return None

            req = DetectObjects.Request()
            req.prompt = prompt
            req.camera_id = "chassis"
            req.enable_lidar = False

            future = self._node.detect_client.call_async(req)
            resp = self._wait_future(future, 5.0, abort_event)

            if resp is not None and resp.success and resp.result.objects:
                # 选深度最近的前方目标 (optical.z > 0.1m)
                best = None
                for obj in resp.result.objects:
                    oz = obj.position_optical.z  # 前方深度
                    if oz <= 0.1:
                        continue
                    if best is None or oz < best.position_optical.z:
                        best = obj

                if best is not None:
                    oz = best.position_optical.z
                    ox = best.position_optical.x
                    # optical x=右 → atan2(-ox, oz): 物体偏右→负角→右转, 偏左→正角→左转
                    angle = math.atan2(-ox, oz)
                    dist = best.distance  # base_link 下的距离 (m)
                    self._log.info(
                        f"Detection: {best.category} "
                        f"dist={dist:.3f}m angle={math.degrees(angle):.1f}° "
                        f"optical=({ox:.3f},{oz:.3f}) score={best.score:.2f}")
                    return dist, angle, best.category

            if attempt < max_retries - 1:
                self._log.info(
                    f"Detection: 第{attempt+1}次未获得结果，重试...")
                time.sleep(0.5)

        return None

    def _detect_align_and_approach(self, prompt: str, abort_event) -> bool:
        """检测 → 旋转对准 → 再检测 → 前进"""
        cfg = self._cfg

        # 1. 检测目标
        result = self._detect_chassis_target(prompt, abort_event)
        if result is None:
            self._log.warn("Detection approach: 未检测到目标")
            return False
        dist, angle, cat = result

        # 2. 旋转对准（偏角 > 3° 才修正）
        if abs(angle) > 0.05:
            self._log.info(f"Detection approach: 旋转 {math.degrees(angle):.1f}°")
            ok, msg = self._nav.rotate_by_angle(angle)
            if not ok:
                self._log.warn(f"Detection approach: 旋转失败 — {msg}")
            time.sleep(0.15)

        # 3. 再次检测获取旋转后的距离（旋转后角度/深度都变了）
        result2 = self._detect_chassis_target(prompt, abort_event)
        if result2 is not None:
            dist, angle, cat = result2

        # 4. 计算前进距离
        travel = dist - cfg.detect_approach_margin
        if travel > 0.03:  # > 3cm 才前进
            self._log.info(
                f"Detection approach: 前进 {travel*100:.0f}cm "
                f"(目标距离={dist*100:.0f}cm, margin={cfg.detect_approach_margin*100:.0f}cm)")
            self._nav.move_forward(travel, speed=0.08)
            time.sleep(0.15)
        else:
            self._log.info(f"Detection approach: 已足够近 (dist={dist*100:.0f}cm)")

        return True

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

    def _call_observe(self, prompt: str, abort_event) -> tuple:
        """返回 (success, point3d_base_mm, category, too_far)"""
        client = self._node.observe_client
        req = Observe.Request()
        req.prompt = prompt
        req.enable_cdm = True

        future = client.call_async(req)
        resp = self._wait_future(future, self._cfg.observe_timeout, abort_event)
        if resp is None:
            return False, None, "", False
        if not resp.success:
            self._log.warn(f"observe failed: {resp.error_message}")
            if (resp.error_message and resp.error_message.startswith("TOO_FAR:")
                    and resp.point3d_base and len(resp.point3d_base) >= 3):
                return False, list(resp.point3d_base), resp.category, True
            return False, None, "", False
        pos = list(resp.point3d_base) if resp.point3d_base else None
        return True, pos, resp.category, False

    def _call_pick(self, abort_event) -> tuple:
        """返回 (success, queue_remaining, grasp_pos_mm, category)"""
        client = self._node.pick_client
        goal = PiperPick.Goal()
        goal.use_last_observe = True
        goal.speed = self._cfg.pick_speed
        goal.lift_height = self._cfg.lift_height
        goal.return_to_ready = False

        send_future = client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 10.0, abort_event)
        if goal_handle is None or not goal_handle.accepted:
            return False, 0, None, ""

        self._ctx.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        wrapped = self._wait_future(result_future, self._cfg.pick_timeout, abort_event)
        self._ctx.active_goal_handle = None

        if wrapped is None:
            return False, 0, None, ""
        r = wrapped.result
        gp = list(r.grasp_position) if r.grasp_position else None
        return r.success, r.queue_remaining, gp, r.category

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
