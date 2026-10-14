"""Bounded ROS action/service calls with cancel propagation and single-task ownership."""

import math, threading, time, signal
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import GoalResponse, CancelResponse
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from .geometry import rotation


class Problem(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Cancelled(Problem):
    def __init__(self):
        super().__init__("CANCELED", "Task canceled")


def plain_wait(future, seconds):
    end = time.monotonic() + seconds
    while not future.done() and time.monotonic() < end:
        time.sleep(0.02)
    if not future.done():
        raise Problem("TIMEOUT", "ROS response timed out")
    if future.exception():
        raise Problem("TRANSPORT", str(future.exception()))
    return future.result()


def pose_msg(goal, frame, stamp):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    msg.pose.position.x = float(goal[0])
    msg.pose.position.y = float(goal[1])
    msg.pose.orientation.z = math.sin(goal[2] / 2)
    msg.pose.orientation.w = math.cos(goal[2] / 2)
    return msg


class TaskNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.busy = False
        self.interlock = False
        self.closing = False
        self.active_child = None

    def param(self, name, default):
        return self.declare_parameter(name, default).value

    def accept(self, request):
        with self.lock:
            if self.busy or self.interlock or self.closing:
                return GoalResponse.REJECT
            self.busy = True
        return GoalResponse.ACCEPT

    def cancel(self, handle):
        return CancelResponse.ACCEPT

    def check(self, parent):
        if self.closing or (parent is not None and parent.is_cancel_requested):
            raise Cancelled()

    def wait(self, future, seconds, parent=None, tick=None):
        end = time.monotonic() + seconds
        while not future.done():
            self.check(parent)
            if time.monotonic() >= end:
                raise Problem("TIMEOUT", "ROS response timed out")
            if tick:
                tick()
            time.sleep(0.05)
        self.check(parent)
        if future.exception():
            raise Problem("TRANSPORT", str(future.exception()))
        return future.result()

    def available(self, client, parent, seconds=3.0):
        end = time.monotonic() + seconds
        ready = (
            client.server_is_ready
            if hasattr(client, "server_is_ready")
            else client.service_is_ready
        )
        while not ready():
            self.check(parent)
            if time.monotonic() >= end:
                raise Problem(
                    "UNAVAILABLE", "Required ROS service/action is unavailable"
                )
            time.sleep(0.05)

    def call(self, client, request, parent, seconds=30.0):
        self.available(client, parent)
        return self.wait(client.call_async(request), seconds, parent)

    def cancel_child(self, handle, result_future):
        try:
            plain_wait(handle.cancel_goal_async(), 3.0)
            result = plain_wait(result_future, 5.0)
            if result.status not in (
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED,
                GoalStatus.STATUS_SUCCEEDED,
            ):
                raise Problem("CANCEL_UNCONFIRMED", "Child action did not terminate")
        except Exception:
            self.interlock = True
            raise Problem(
                "CANCEL_UNCONFIRMED",
                "Child stop unconfirmed; controller locked until restart and external stop verification",
            ) from None

    def action(
        self, client, request, parent, seconds=180.0, tick=None, allow_failure=False
    ):
        self.available(client, parent)
        self.check(parent)
        send = client.send_goal_async(request)
        # Resolve the goal handle even if cancellation arrives during acceptance.
        try:
            handle = plain_wait(send, 5.0)
        except Exception:
            self.interlock = True

            def late(f):
                try:
                    h = f.result()
                    if h and h.accepted:
                        h.cancel_goal_async()
                except Exception:
                    pass

            send.add_done_callback(late)
            raise Problem(
                "GOAL_ACCEPTANCE_UNKNOWN",
                "Child goal acceptance timed out; stop unconfirmed, controller locked",
            ) from None
        if not handle.accepted:
            raise Problem("GOAL_REJECTED", "Child action rejected goal")
        result_future = handle.get_result_async()
        self.active_child = handle
        try:
            result = self.wait(result_future, seconds, parent, tick)
            if result.status != GoalStatus.STATUS_SUCCEEDED and not (
                allow_failure and result.status == GoalStatus.STATUS_ABORTED
            ):
                raise Problem(
                    "CHILD_FAILED",
                    f"Child action ended with status {result.status}: "
                    + getattr(result.result, "message", ""),
                )
            return result.result
        except Exception:
            if not result_future.done():
                self.cancel_child(handle, result_future)
            raise
        finally:
            self.active_child = None

    def finish(self, handle, result, problem=None):
        if problem:
            result.success = False
            result.code = getattr(problem, "code", "ERROR")
            result.message = str(problem)
            if result.code in ("CANCEL_UNCONFIRMED", "GOAL_ACCEPTANCE_UNKNOWN"):
                self.interlock = True
            if isinstance(problem, Cancelled):
                handle.canceled()
            else:
                handle.abort()
        elif handle.is_cancel_requested:
            result.success = False
            result.code = "CANCELED"
            result.message = "Task canceled"
            handle.canceled()
        elif result.success:
            handle.succeed()
        else:
            handle.abort()
        with self.lock:
            self.busy = False
        return result


class LocatedNode(TaskNode):
    def __init__(self, name):
        super().__init__(name)
        self.map_frame = self.param("map_frame", "map")
        self.base_frame = self.param("base_frame", "base_link")
        self.tf_age = self.param("tf_max_age", 0.8)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)

    def robot_pose(self):
        try:
            t = self.tf.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception:
            raise Problem(
                "LOCALIZATION", "map to base_link transform unavailable"
            ) from None
        age = (
            self.get_clock().now().nanoseconds
            - (t.header.stamp.sec * 10**9 + t.header.stamp.nanosec)
        ) / 1e9
        if not -0.1 <= age <= self.tf_age:
            raise Problem("LOCALIZATION", f"Robot pose is stale: {age:.2f}s")
        q = t.transform.rotation
        rotation([q.x, q.y, q.z, q.w])
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        p = t.transform.translation
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z, yaw)):
            raise Problem("LOCALIZATION", "Invalid robot pose")
        return (p.x, p.y, yaw), p.z


def run(factory):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    node = factory()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.closing = True
        if node.active_child:
            try:
                node.active_child.cancel_goal_async()
            except Exception:
                pass
        # Keep ROS alive while worker callbacks cancel and await child termination.
        deadline = time.monotonic() + 10
        while node.busy and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        executor.shutdown(timeout_sec=8)
        node.destroy_node()
        rclpy.try_shutdown()
