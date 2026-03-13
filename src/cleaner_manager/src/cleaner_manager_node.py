#!/usr/bin/env python3
"""
cleaner_manager_node.py — thin orchestration node.

Coordinates SLAM (Nav2), Perception (ObjectTracker), and Manipulation (Piper)
via a blocking state machine running in a worker thread.
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

import time as _time

from perception.msg import Object3DArray, PerceptionConfig
from piper_msgs.srv import Observe, GoReady, GetStatus, GoZero
from piper_msgs.action import PiperPick, PiperPlace
from cleaner_manager.msg import CleanerManagerStatus, TargetEntry
from cleaner_manager.target_pool import TargetPool
from cleaner_manager.target_filter import TargetFilter
from approach_navigator import ApproachNavigator

from cleaner_manager.pick_state_machine import (
    PickStateMachine, PickConfig, PickState, PICK_STATE_NAMES
)


class CleanerManagerNode(Node):

    def __init__(self):
        super().__init__('cleaner_manager_node')
        self._log = self.get_logger()

        # Callback group — reentrant for all subscriptions/services/timers
        self._cb_group = ReentrantCallbackGroup()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Declare parameters ---
        self.declare_parameter('tracked_objects_topic', '/multi_camera_perception/fused/objects_3d')
        self.declare_parameter('min_distance', 0.3)
        self.declare_parameter('max_distance', 3.0)
        self.declare_parameter('max_attempts', 3)
        self.declare_parameter('max_picks_per_nav', 10)
        self.declare_parameter('pick_speed', 30)
        self.declare_parameter('lift_height', 200.0)
        self.declare_parameter('observe_prompt', '')  # empty = use piper_grasp_node default
        self.declare_parameter('observe_timeout', 10.0)
        self.declare_parameter('pick_timeout', 60.0)
        self.declare_parameter('place_timeout', 30.0)
        self.declare_parameter('nav_timeout', 120.0)
        self.declare_parameter('max_consecutive_failures', 3)
        self.declare_parameter('error_cooldown', 5.0)
        self.declare_parameter('max_error_retries', 3)
        self.declare_parameter('status_rate', 1.0)
        self.declare_parameter('target_ttl', 60.0)
        self.declare_parameter('match_threshold', 0.20)
        self.declare_parameter('min_observations', 2)
        self.declare_parameter('wait_for_first_target_timeout', 15.0)
        self.declare_parameter('grasp_z_min',            -0.30)
        self.declare_parameter('grasp_z_max',             0.50)
        self.declare_parameter('grasp_distance_max',      6.0)
        self.declare_parameter('grasp_physical_min_size', 0.02)
        self.declare_parameter('grasp_physical_max_size', 0.25)

        # --- TargetFilter ---
        self._filter = TargetFilter(
            z_min    = self.get_parameter('grasp_z_min').value,
            z_max    = self.get_parameter('grasp_z_max').value,
            dist_max = self.get_parameter('grasp_distance_max').value,
            size_min = self.get_parameter('grasp_physical_min_size').value,
            size_max = self.get_parameter('grasp_physical_max_size').value,
            logger   = self._log,
        )

        # --- TargetPool ---
        self._pool = TargetPool(
            logger=self._log,
            min_distance=self.get_parameter('min_distance').value,
            max_distance=self.get_parameter('max_distance').value,
            max_attempts=self.get_parameter('max_attempts').value,
            target_ttl=self.get_parameter('target_ttl').value,
            match_threshold=self.get_parameter('match_threshold').value,
            min_observations=self.get_parameter('min_observations').value,
        )

        # --- Localization status gate ---
        self._localization_ok = False
        self.create_subscription(
            Bool, '/localization_status',
            self._localization_status_cb, 5,
            callback_group=self._cb_group,
        )

        # --- Subscription: tracked objects ---
        topic = self.get_parameter('tracked_objects_topic').value
        self.create_subscription(
            Object3DArray, topic,
            self._tracked_objects_cb, 10,
            callback_group=self._cb_group,
        )

        # --- Subscription: grasp filter config ---
        self.create_subscription(
            PerceptionConfig, '/perception/config',
            lambda msg: self._filter.update_from_config(msg), 10,
            callback_group=self._cb_group,
        )

        # --- Service clients (Piper) ---
        self.observe_client = self.create_client(
            Observe, '/piper/observe', callback_group=self._cb_group)
        self.go_ready_client = self.create_client(
            GoReady, '/piper/go_ready', callback_group=self._cb_group)
        self.get_status_client = self.create_client(
            GetStatus, '/piper/get_status', callback_group=self._cb_group)
        self.go_zero_client = self.create_client(
            GoZero, '/go_zero_srv', callback_group=self._cb_group)

        # --- Action clients (Piper) ---
        self.pick_client = ActionClient(
            self, PiperPick, '/piper/pick', callback_group=self._cb_group)
        self.place_client = ActionClient(
            self, PiperPlace, '/piper/place', callback_group=self._cb_group)

        # --- Services: start / abort / reset_pool ---
        self.create_service(
            Trigger, '~/start', self._start_cb, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/abort', self._abort_cb, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/reset_pool', self._reset_pool_cb, callback_group=self._cb_group)

        # --- Status publisher ---
        self._status_pub = self.create_publisher(
            CleanerManagerStatus, '~/status', 10)
        rate = self.get_parameter('status_rate').value
        self.create_timer(
            1.0 / rate, self._publish_status, callback_group=self._cb_group)

        # --- State machine (created on start) ---
        self._navigator = None
        self._sm = None
        self._abort_event = threading.Event()
        self._start_event = threading.Event()
        self._worker_thread = None
        self._running = False

        self._log.info("CleanerManagerNode initialized — call ~/start to begin")

    def set_navigator(self, navigator: ApproachNavigator):
        self._navigator = navigator

    def get_robot_pos_map(self) -> Point:
        """Get robot position in map frame via TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            p = Point()
            p.x = tf.transform.translation.x
            p.y = tf.transform.translation.y
            p.z = tf.transform.translation.z
            return p
        except Exception as e:
            self._log.debug(f'get_robot_pos_map failed: {e}')
            return None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _localization_status_cb(self, msg):
        self._localization_ok = msg.data

    def _tracked_objects_cb(self, msg):
        if not self._localization_ok:
            return  # localization not ready — discard perception data
        msg.objects = self._filter.filter(msg.objects)
        self._pool.update_from_tracker(msg)

    def _start_cb(self, request, response):
        if self._running:
            response.success = False
            response.message = "Already running"
            return response

        ok, msg = self._startup_check()
        if not ok:
            response.success = False
            response.message = f"Startup check failed: {msg}"
            return response

        self._pool.reset()
        self._abort_event.clear()
        self._running = True

        cfg = PickConfig(
            max_picks_per_nav=self.get_parameter('max_picks_per_nav').value,
            pick_speed=self.get_parameter('pick_speed').value,
            lift_height=self.get_parameter('lift_height').value,
            observe_prompt=self.get_parameter('observe_prompt').value,
            observe_timeout=self.get_parameter('observe_timeout').value,
            pick_timeout=self.get_parameter('pick_timeout').value,
            place_timeout=self.get_parameter('place_timeout').value,
            nav_timeout=self.get_parameter('nav_timeout').value,
            max_consecutive_failures=self.get_parameter('max_consecutive_failures').value,
            error_cooldown=self.get_parameter('error_cooldown').value,
            max_error_retries=self.get_parameter('max_error_retries').value,
            wait_for_first_target_timeout=self.get_parameter('wait_for_first_target_timeout').value,
        )
        self._sm = PickStateMachine(self, self._pool, self._navigator, cfg)

        self._worker_thread = threading.Thread(
            target=self._worker_run, daemon=True, name="pick_worker")
        self._worker_thread.start()

        response.success = True
        response.message = "Started"
        return response

    def _reset_pool_cb(self, request, response):
        self._pool.reset()
        self._log.info("Target pool cleared by request")
        response.success = True
        response.message = "Pool cleared"
        return response

    def _abort_cb(self, request, response):
        if not self._running:
            response.success = False
            response.message = "Not running"
            return response

        # 防止重复取消 (ReentrantCallbackGroup 下可能并发调用)
        if self._abort_event.is_set():
            response.success = True
            response.message = "Abort already in progress"
            return response

        self._log.info("Abort requested — cancelling...")
        self._abort_event.set()
        if self._navigator:
            self._navigator.cancel()
        if self._sm:
            self._sm._cancel_active_goal()

        response.success = True
        response.message = "Abort requested"
        return response

    def _publish_status(self):
        msg = CleanerManagerStatus()
        msg.header.stamp = self.get_clock().now().to_msg()

        if self._sm:
            msg.state = int(self._sm.state)
            msg.state_name = self._sm.state_name
            ctx = self._sm.context
            msg.current_target_category = (
                ctx.current_target.category if ctx.current_target else "")
            msg.consecutive_failures = max(
                ctx.consecutive_nav_failures, ctx.consecutive_pick_failures)
            msg.last_nav_time_ms = ctx.last_nav_time_ms
            msg.last_pick_time_ms = ctx.last_pick_time_ms
            msg.error_message = ctx.error_message
        else:
            msg.state = int(PickState.IDLE)
            msg.state_name = "IDLE"

        stats = self._pool.stats
        msg.targets_total = stats['total']
        msg.targets_picked = stats['picked']
        msg.targets_failed = stats['failed']
        msg.targets_remaining = stats['remaining']

        now = _time.time()
        entries = []
        for t in self._pool.get_snapshot():
            e = TargetEntry()
            e.category = t.category
            e.pos_x = t.position_map.x
            e.pos_y = t.position_map.y
            e.score = t.track_score
            e.status = int(t.status)
            e.observations = t.observation_count
            e.attempts = t.attempts
            e.last_seen_age = float(now - t.last_seen)
            e.viable = self._pool.is_viable(t)
            entries.append(e)
        msg.pool_targets = entries

        self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_run(self):
        try:
            self._sm.run(self._abort_event)
        except Exception as e:
            self._log.error(f"State machine exception: {e}")
            import traceback
            self._log.error(traceback.format_exc())
        finally:
            self._running = False
            self._log.info("Worker thread finished")

    # ------------------------------------------------------------------
    # Startup health check
    # ------------------------------------------------------------------

    def _startup_check(self):
        """Verify all dependencies are available. Returns (ok, message)."""
        # TF
        try:
            self.tf_buffer.lookup_transform(
                'map', 'base_link', Time(),
                timeout=rclpy.duration.Duration(seconds=5.0))
        except Exception as e:
            return False, f"TF map->base_link unavailable: {e}"
        if self._navigator is None:
            return False, "Navigator not set"

        # Services (2s timeout each)
        for name, client in [
            ('/piper/observe', self.observe_client),
            ('/piper/go_ready', self.go_ready_client),
            ('/go_zero_srv', self.go_zero_client),
        ]:
            if not client.wait_for_service(timeout_sec=2.0):
                return False, f"Service {name} not available"

        # Actions (2s timeout each)
        if not self.pick_client.wait_for_server(timeout_sec=2.0):
            return False, "Action /piper/pick not available"
        if not self.place_client.wait_for_server(timeout_sec=2.0):
            return False, "Action /piper/place not available"

        return True, "OK"


def main(args=None):
    rclpy.init(args=args)

    manager = CleanerManagerNode()
    navigator = ApproachNavigator()
    manager.set_navigator(navigator)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(manager)
    executor.add_node(navigator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
