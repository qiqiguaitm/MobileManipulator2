"""Test-only Nav2, manipulation and scene backends. Never launches hardware."""

import math, os, time
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from tf2_ros import TransformBroadcaster
from robot_task_interfaces.action import OperateTarget
from robot_task_interfaces.srv import DetectScene
from robot_task_interfaces.msg import ObjectTarget
from .ros_support import TaskNode, run


class MockBackends(TaskNode):
    def __init__(self):
        if os.environ.get("ROS_DOMAIN_ID") != "166":
            raise RuntimeError("Mock backends require isolated ROS_DOMAIN_ID=166")
        super().__init__("mock_task_backends")
        self.pose = [0.0, 0.0, 0.0]
        self.objects = [("杯子", 2.0, 1.0, 0.3), ("瓶子", 2.0, -1.0, 0.35)]
        self.removed = set()
        self.failures = set()
        self.nav_calls = 0
        self.op_calls = []
        self.detect_calls = 0
        self.nav_delay = self.param("navigation_delay", 0.2)
        self.ignore_cancel = False
        self.scene_error = False
        self.empty = False
        self.fail_name = ""
        self.block_plans = False
        self.broadcaster = TransformBroadcaster(self)
        self.tf_timer = self.create_timer(0.05, self.tf_tick, callback_group=self.group)
        self.plan_server = ActionServer(
            self,
            ComputePathToPose,
            "/compute_path_to_pose",
            execute_callback=self.plan,
            callback_group=self.group,
        )
        self.nav_server = ActionServer(
            self,
            NavigateToPose,
            "/navigate_to_pose",
            execute_callback=self.navigate,
            cancel_callback=lambda h: CancelResponse.ACCEPT,
            callback_group=self.group,
        )
        self.op_server = ActionServer(
            self,
            OperateTarget,
            "/manipulation/operate",
            execute_callback=self.operate,
            cancel_callback=lambda h: CancelResponse.ACCEPT,
            callback_group=self.group,
        )
        self.create_service(
            DetectScene, "/gc/detect_scene", self.detect, callback_group=self.group
        )

    def tf_tick(self):
        t = TransformStamped()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.header.stamp = self.get_clock().now().to_msg()
        t.transform.translation.x = self.pose[0]
        t.transform.translation.y = self.pose[1]
        t.transform.rotation.z = math.sin(self.pose[2] / 2)
        t.transform.rotation.w = math.cos(self.pose[2] / 2)
        self.broadcaster.sendTransform(t)

    def plan(self, h):
        result = ComputePathToPose.Result()
        if self.block_plans:
            h.abort()
            return result
        result.path.header = h.request.goal.header
        start = PoseStamped()
        start.header = h.request.goal.header
        start.pose.orientation.w = 1.0
        start.pose.position.x = self.pose[0]
        start.pose.position.y = self.pose[1]
        result.path.poses = [start, h.request.goal]
        h.succeed()
        return result

    def navigate(self, h):
        self.nav_calls += 1
        end = time.monotonic() + self.nav_delay
        while time.monotonic() < end:
            if self.closing:
                h.abort()
                return NavigateToPose.Result()
            if h.is_cancel_requested and not self.ignore_cancel:
                h.canceled()
                return NavigateToPose.Result()
            time.sleep(0.02)
        p = h.request.pose.pose
        q = p.orientation
        self.pose = [p.position.x, p.position.y, 2 * math.atan2(q.z, q.w)]
        self.tf_tick()
        time.sleep(0.1)
        h.succeed()
        return NavigateToPose.Result()

    def operate(self, h):
        r = OperateTarget.Result()
        self.op_calls.append(h.request.name)
        if h.request.name == self.fail_name:
            r.code = "GRASP_FAILED"
            r.message = "Injected confirmed grasp failure"
            h.abort()
            return r
        if h.is_cancel_requested:
            h.canceled()
            return r
        self.removed.add(h.request.name)
        r.success = True
        r.code = "MOCK_COMPLETED"
        r.message = "Mock only; no arm moved"
        h.succeed()
        return r

    def detect(self, req, res):
        self.detect_calls += 1
        self.last_description = req.description
        if self.scene_error:
            res.error = "Injected detector error"
            return res
        res.success = True
        if self.empty:
            return res
        for i, (name, x, y, z) in enumerate(self.objects):
            if name in self.removed or (
                req.description and name not in req.description
            ):
                continue
            o = ObjectTarget()
            o.instance_id = str(i)
            o.name = name
            o.position.header.frame_id = "map"
            o.position.header.stamp = self.get_clock().now().to_msg()
            o.position.point.x = x
            o.position.point.y = y
            o.position.point.z = z
            res.objects.append(o)
        return res


def main():
    run(MockBackends)
