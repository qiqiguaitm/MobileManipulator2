#!/usr/bin/env python3
"""
manual_control_node.py — Manual control backend for interactive pick-and-place.

State machine:
  IDLE -> DETECTING -> WAITING_SELECT -> WAITING_CONFIRM -> NAVIGATING ->
  APPROACHING -> OBSERVING -> PICKING -> PLACING -> IDLE

Driven by ManualCommand service calls from ManualPanel GUI.
"""

import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient

import tf2_ros
from geometry_msgs.msg import Point, PointStamped
from tf2_geometry_msgs import do_transform_point

from perception.srv import DetectObjects
from piper_msgs.srv import Observe, GoReady, GetStatus, GoZero
from piper_msgs.action import PiperPick, PiperPlace

from cleaner_manager.msg import ManualStatus, Candidate
from cleaner_manager.srv import ManualCommand

from approach_navigator import ApproachNavigator
from approach_navigator.nav_types import NavStage


# State constants (mirror ManualStatus.msg)
STATE_IDLE = 0
STATE_DETECTING = 1
STATE_WAITING_SELECT = 2
STATE_WAITING_CONFIRM = 3
STATE_NAVIGATING = 4
STATE_APPROACHING = 5
STATE_OBSERVING = 6
STATE_PICKING = 7
STATE_PLACING = 8
STATE_ERROR = 9

_STATE_NAMES = {
    STATE_IDLE: "IDLE",
    STATE_DETECTING: "DETECTING",
    STATE_WAITING_SELECT: "WAITING_SELECT",
    STATE_WAITING_CONFIRM: "WAITING_CONFIRM",
    STATE_NAVIGATING: "NAVIGATING",
    STATE_APPROACHING: "APPROACHING",
    STATE_OBSERVING: "OBSERVING",
    STATE_PICKING: "PICKING",
    STATE_PLACING: "PLACING",
    STATE_ERROR: "ERROR",
}


class ManualControlNode(Node):

    def __init__(self):
        super().__init__('manual_control_node')
        self._log = self.get_logger()
        self._cb_group = ReentrantCallbackGroup()

        # === Parameters ===
        self.declare_parameter('observe_prompt', '')  # empty = use piper_grasp_node default
        self.declare_parameter('pick_speed', 30)
        self.declare_parameter('lift_height', 200.0)
        self.declare_parameter('observe_timeout', 10.0)
        self.declare_parameter('pick_timeout', 60.0)
        self.declare_parameter('place_timeout', 30.0)
        self.declare_parameter('nav_timeout', 120.0)
        self.declare_parameter('status_rate', 10.0)

        # === TF ===
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # === Service server ===
        self.create_service(
            ManualCommand, '~/command', self._command_cb,
            callback_group=self._cb_group)

        # === Publisher ===
        self._status_pub = self.create_publisher(ManualStatus, '~/status', 10)

        # === Service clients ===
        self._detect_cli = self.create_client(
            DetectObjects, '/multi_camera_perception/detect',
            callback_group=self._cb_group)
        self._observe_cli = self.create_client(
            Observe, '/piper/observe',
            callback_group=self._cb_group)
        self._go_ready_cli = self.create_client(
            GoReady, '/piper/go_ready',
            callback_group=self._cb_group)
        self._get_status_cli = self.create_client(
            GetStatus, '/piper/get_status',
            callback_group=self._cb_group)
        self._go_zero_cli = self.create_client(
            GoZero, '/go_zero_srv',
            callback_group=self._cb_group)

        # === Action clients ===
        self._pick_ac = ActionClient(
            self, PiperPick, '/piper/pick',
            callback_group=self._cb_group)
        self._place_ac = ActionClient(
            self, PiperPlace, '/piper/place',
            callback_group=self._cb_group)

        # === State ===
        self._lock = threading.Lock()
        self._state = STATE_IDLE
        self._message = ""
        self._prompt = self.get_parameter('observe_prompt').value
        self._candidates = []
        self._selected_index = -1
        self._target_category = ""
        self._target_object_id = ""
        self._target_distance = 0.0
        self._target_position = Point()
        self._robot_position = Point()
        self._grasp_step = ""
        self._grasp_progress = 0.0
        self._error_message = ""

        # === Async operation tracking ===
        self._detect_future = None
        self._cancel_requested = False
        self._worker_thread = None
        self._active_goal_handle = None
        self._navigator = None

        # === Timers ===
        self.create_timer(0.02, self._poll_cb, callback_group=self._cb_group)
        rate = self.get_parameter('status_rate').value
        self.create_timer(
            1.0 / rate, self._publish_status, callback_group=self._cb_group)

        self._log.info("ManualControlNode initialized")

    def set_navigator(self, nav: ApproachNavigator):
        self._navigator = nav

    # ------------------------------------------------------------------
    # State management (thread-safe)
    # ------------------------------------------------------------------

    def _set_state(self, state: int, message: str = ""):
        with self._lock:
            if self._state != state:
                self._log.info(
                    f"State: {_STATE_NAMES.get(self._state)} -> {_STATE_NAMES.get(state)}: {message}")
            self._state = state
            self._message = message

    def _thread_set_state(self, state: int, message: str = ""):
        """Same as _set_state, used from worker thread for clarity."""
        self._set_state(state, message)

    def _get_state(self) -> int:
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    def _command_cb(self, request, response):
        cmd = request.command
        state = self._get_state()

        if cmd == ManualCommand.Request.CMD_SET_PROMPT:
            with self._lock:
                self._prompt = request.prompt or self._prompt
            response.success = True
            response.message = f"提示词: {self._prompt}"
            return response

        if cmd == ManualCommand.Request.CMD_CANCEL:
            self._cancel_current()
            response.success = True
            response.message = "已取消"
            return response

        if cmd == ManualCommand.Request.CMD_DETECT:
            if state not in (STATE_IDLE, STATE_WAITING_SELECT):
                response.success = False
                response.message = f"当前状态 {_STATE_NAMES.get(state)} 不允许检测"
                return response
            self._start_detection()
            response.success = True
            response.message = "开始检测"
            return response

        if cmd == ManualCommand.Request.CMD_SELECT:
            if state != STATE_WAITING_SELECT:
                response.success = False
                response.message = "请先检测目标"
                return response
            idx = request.target_index
            with self._lock:
                if idx < 0 or idx >= len(self._candidates):
                    response.success = False
                    response.message = f"无效索引 {idx}"
                    return response
                self._selected_index = idx
                c = self._candidates[idx]
                self._target_category = c.category
                self._target_object_id = c.object_id
                self._target_distance = c.distance
                self._target_position = c.position
            self._set_state(STATE_WAITING_CONFIRM,
                            f"已选择: {c.category} ({c.object_id})")
            response.success = True
            response.message = f"已选择 {c.category}"
            return response

        if cmd == ManualCommand.Request.CMD_SELECT_OBJECT:
            if state not in (STATE_IDLE, STATE_WAITING_SELECT, STATE_WAITING_CONFIRM):
                response.success = False
                response.message = f"当前状态 {_STATE_NAMES.get(state)} 不允许选择"
                return response
            # Normalize to map frame
            pos_map = request.target_position
            src_frame = request.position_frame or 'map'
            if src_frame != 'map':
                try:
                    ps = PointStamped()
                    ps.header.frame_id = src_frame
                    ps.point = request.target_position
                    tf_transform = self._tf_buffer.lookup_transform(
                        'map', src_frame, rclpy.time.Time())
                    pos_map = do_transform_point(ps, tf_transform).point
                except Exception as e:
                    response.success = False
                    response.message = f"坐标转换失败 ({src_frame}->map): {e}"
                    return response
            with self._lock:
                self._target_category = request.category
                self._target_object_id = request.object_id
                self._target_distance = request.distance
                self._target_position = pos_map
                self._selected_index = -1
                self._candidates = []
            self._set_state(STATE_WAITING_CONFIRM,
                            f"已选择: {request.category} ({request.object_id})")
            response.success = True
            response.message = f"已选择 {request.category}"
            return response

        if cmd == ManualCommand.Request.CMD_GO:
            if state != STATE_WAITING_CONFIRM:
                response.success = False
                response.message = "请先选择目标"
                return response
            self._start_go_pipeline()
            response.success = True
            response.message = "开始执行"
            return response

        response.success = False
        response.message = f"未知命令: {cmd}"
        return response

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _start_detection(self):
        self._set_state(STATE_DETECTING, "正在检测...")
        req = DetectObjects.Request()
        with self._lock:
            req.prompt = self._prompt
        req.enable_lidar = False
        req.camera_id = "all"
        self._detect_future = self._detect_cli.call_async(req)

    def _poll_cb(self):
        """50Hz poll for async detection result."""
        if self._detect_future is not None and self._detect_future.done():
            future = self._detect_future
            self._detect_future = None
            try:
                result = future.result()
                self._handle_detect_result(result)
            except Exception as e:
                self._set_state(STATE_IDLE, f"检测异常: {e}")

    def _handle_detect_result(self, result):
        if not result.success:
            self._set_state(STATE_IDLE, f"检测失败: {result.error_message}")
            return

        objects = result.result.objects
        if len(objects) == 0:
            self._set_state(STATE_IDLE, "未检测到目标")
            return

        candidates = []
        for obj in objects:
            c = self._transform_to_map(obj)
            if c is not None:
                candidates.append(c)

        if not candidates:
            self._set_state(STATE_IDLE, "坐标转换失败，未获得有效候选")
            return

        # Sort by distance
        candidates.sort(key=lambda c: c.distance)

        with self._lock:
            self._candidates = candidates
            self._selected_index = -1
        self._set_state(STATE_WAITING_SELECT, f"检测到 {len(candidates)} 个目标")

    def _transform_to_map(self, obj) -> Optional[Candidate]:
        """Transform Object3D (base_link) -> Candidate (map)."""
        try:
            transform = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            ps = PointStamped()
            ps.header.frame_id = 'base_link'
            ps.point = obj.position
            pos_map = do_transform_point(ps, transform).point

            c = Candidate()
            c.object_id = obj.object_id
            c.category = obj.category
            c.score = obj.score
            c.distance = obj.distance
            c.position = pos_map
            return c
        except Exception as e:
            self._log.warn(f"TF base_link->map failed: {e}")
            return None

    # ------------------------------------------------------------------
    # GO pipeline (runs in worker thread)
    # ------------------------------------------------------------------

    def _start_go_pipeline(self):
        self._cancel_requested = False
        with self._lock:
            target_pos = Point(
                x=self._target_position.x,
                y=self._target_position.y,
                z=0.0)
        self._worker_thread = threading.Thread(
            target=self._run_pipeline, args=(target_pos,),
            daemon=True, name="manual_pipeline")
        self._worker_thread.start()

    def _run_pipeline(self, target_point: Point):
        """Nav -> observe -> pick -> place (blocking, runs in thread)."""
        try:
            self._do_pipeline(target_point)
        except Exception as e:
            self._log.error(f"Pipeline exception: {e}")
            import traceback
            self._log.error(traceback.format_exc())
            self._thread_set_state(STATE_ERROR, f"异常: {e}")
            self._safe_go_ready()
            time.sleep(2.0)
            self._thread_set_state(STATE_IDLE, "错误已恢复")

    def _do_pipeline(self, target_point: Point):
        # --- NAVIGATING + APPROACHING ---
        self._thread_set_state(STATE_NAVIGATING, "正在导航...")
        result = self._navigator.approach_to_target(
            target_point, status_callback=self._nav_status_cb)

        if self._cancel_requested:
            return
        if not result.success:
            self._thread_set_state(STATE_ERROR, f"导航失败: {result.error_message}")
            self._safe_go_ready()
            time.sleep(2.0)
            self._thread_set_state(STATE_IDLE, "导航失败，已恢复")
            return

        # --- OBSERVING ---
        self._thread_set_state(STATE_OBSERVING, "展臂观察中...")
        go_ready_ok = self._blocking_go_ready(
            speed=self.get_parameter('pick_speed').value, open_gripper=True)
        if self._cancel_requested:
            return
        if not go_ready_ok:
            self._thread_set_state(STATE_ERROR, "展臂失败")
            self._safe_go_ready()
            time.sleep(2.0)
            self._thread_set_state(STATE_IDLE, "展臂失败，已恢复")
            return

        with self._lock:
            prompt = self._prompt
        obs_req = Observe.Request()
        obs_req.prompt = prompt
        obs_req.enable_cdm = True
        obs_result = self._blocking_service_call(
            self._observe_cli, obs_req,
            timeout=self.get_parameter('observe_timeout').value)
        if self._cancel_requested:
            return
        if obs_result is None or not obs_result.success:
            err = obs_result.error_message if obs_result else "超时"
            self._thread_set_state(STATE_ERROR, f"观察失败: {err}")
            self._safe_go_ready()
            time.sleep(2.0)
            self._thread_set_state(STATE_IDLE, "观察失败，已恢复")
            return

        # --- PICKING ---
        self._thread_set_state(STATE_PICKING, "抓取中...")
        pick_goal = PiperPick.Goal()
        pick_goal.use_last_observe = True
        pick_goal.speed = self.get_parameter('pick_speed').value
        pick_goal.lift_height = float(self.get_parameter('lift_height').value)
        pick_goal.return_to_ready = False
        pick_result = self._blocking_action_call(
            self._pick_ac, pick_goal,
            feedback_cb=self._pick_feedback_cb,
            timeout=self.get_parameter('pick_timeout').value)
        if self._cancel_requested:
            return
        if pick_result is None or not pick_result.success:
            err = pick_result.error_message if pick_result else "超时"
            self._thread_set_state(STATE_ERROR, f"抓取失败: {err}")
            self._safe_go_ready()
            time.sleep(2.0)
            self._thread_set_state(STATE_IDLE, "抓取失败，已恢复")
            return

        # --- PLACING ---
        self._thread_set_state(STATE_PLACING, "放置中...")
        with self._lock:
            self._grasp_step = ""
            self._grasp_progress = 0.0
        place_goal = PiperPlace.Goal()
        place_goal.use_default_place = True
        place_goal.speed = self.get_parameter('pick_speed').value
        place_goal.return_to_ready = True
        place_result = self._blocking_action_call(
            self._place_ac, place_goal,
            feedback_cb=self._place_feedback_cb,
            timeout=self.get_parameter('place_timeout').value)
        if self._cancel_requested:
            return
        if place_result is None or not place_result.success:
            err = place_result.error_message if place_result else "超时"
            self._thread_set_state(STATE_ERROR, f"放置失败: {err}")
            self._safe_go_ready()
            time.sleep(2.0)
            self._thread_set_state(STATE_IDLE, "放置失败，已恢复")
            return

        self._thread_set_state(STATE_IDLE, "归零中...")
        try:
            req = GoZero.Request()
            req.is_mit_mode = False
            self._blocking_service_call(self._go_zero_cli, req, timeout=15.0)
        except Exception as e:
            self._log.warn(f'go_zero failed: {e}')
        self._thread_set_state(STATE_IDLE, "完成!")

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def _cancel_current(self):
        self._cancel_requested = True
        state = self._get_state()

        if state in (STATE_NAVIGATING, STATE_APPROACHING):
            if self._navigator:
                self._navigator.cancel()

        if self._active_goal_handle is not None:
            try:
                self._active_goal_handle.cancel_goal_async()
            except Exception:
                pass

        if self._detect_future is not None:
            self._detect_future = None

        # Only reset arm when it might have moved from ready position
        if state in (STATE_OBSERVING, STATE_PICKING, STATE_PLACING):
            self._safe_go_ready()

        self._set_state(STATE_IDLE, "已取消")

    # ------------------------------------------------------------------
    # Blocking helpers (for worker thread, poll-based)
    # ------------------------------------------------------------------

    def _wait_future(self, future, timeout: float):
        deadline = time.time() + timeout
        while not future.done():
            if self._cancel_requested:
                return None
            if time.time() > deadline:
                self._log.warn("Service/action call timed out")
                return None
            time.sleep(0.05)
        return future.result()

    def _blocking_service_call(self, client, request, timeout=10.0):
        future = client.call_async(request)
        return self._wait_future(future, timeout)

    def _blocking_go_ready(self, speed: int, open_gripper: bool = False) -> bool:
        req = GoReady.Request()
        req.speed = speed
        req.open_gripper = open_gripper
        resp = self._blocking_service_call(self._go_ready_cli, req, timeout=30.0)
        if resp is None:
            return False
        return resp.success

    def _blocking_action_call(self, action_client, goal,
                              feedback_cb=None, timeout=60.0):
        send_future = action_client.send_goal_async(
            goal, feedback_callback=feedback_cb)
        goal_handle = self._wait_future(send_future, 10.0)
        if goal_handle is None or not goal_handle.accepted:
            return None

        self._active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        wrapped = self._wait_future(result_future, timeout)
        self._active_goal_handle = None

        if wrapped is None:
            return None
        return wrapped.result

    def _safe_go_ready(self):
        try:
            self._blocking_go_ready(speed=30, open_gripper=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _nav_status_cb(self, stage: NavStage, message: str):
        if self._cancel_requested:
            return
        if stage == NavStage.NAVIGATING:
            self._thread_set_state(STATE_NAVIGATING, message)
        elif stage == NavStage.ALIGNING:
            self._thread_set_state(STATE_APPROACHING, f"对齐中: {message}")
        elif stage == NavStage.FINAL_APPROACH:
            self._thread_set_state(STATE_APPROACHING, f"接近中: {message}")

    def _pick_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        with self._lock:
            self._grasp_step = fb.step_name
            self._grasp_progress = fb.progress

    def _place_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        with self._lock:
            self._grasp_step = fb.step_name
            self._grasp_progress = fb.progress

    # ------------------------------------------------------------------
    # Status publishing (10Hz)
    # ------------------------------------------------------------------

    def _publish_status(self):
        msg = ManualStatus()
        msg.header.stamp = self.get_clock().now().to_msg()

        with self._lock:
            msg.state = self._state
            msg.state_name = _STATE_NAMES.get(self._state, "UNKNOWN")
            msg.message = self._message
            msg.current_prompt = self._prompt
            msg.candidates = list(self._candidates)
            msg.selected_index = self._selected_index
            msg.target_category = self._target_category
            msg.target_object_id = self._target_object_id
            msg.target_distance = self._target_distance
            msg.target_position = self._target_position
            msg.grasp_step = self._grasp_step
            msg.grasp_progress = self._grasp_progress
            msg.error_message = self._error_message

        # Robot position (best-effort, no lock needed for TF)
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            msg.robot_position.x = tf.transform.translation.x
            msg.robot_position.y = tf.transform.translation.y
        except Exception:
            pass

        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ManualControlNode()
    navigator = ApproachNavigator()
    node.set_navigator(navigator)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(navigator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
