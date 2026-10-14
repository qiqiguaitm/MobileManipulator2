"""GC: one washer operation or repeated processing of objects in the current scene."""

import json, math, time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from rclpy.action import ActionServer, ActionClient
from robot_task_interfaces.action import RunTask, Approach, OperateTarget
from robot_task_interfaces.srv import DetectScene
from .ros_support import TaskNode, Problem, Cancelled, run
from .ledger import Ledger


class GlobalController(TaskNode):
    def __init__(self):
        super().__init__("global_controller")
        self.allow = self.param("allow_execution", False)
        self.skip_grasp = self.param("skip_grasp", False)
        self.max_attempts = self.param("target_attempts", 2)
        self.scene_attempts = self.param("scene_attempts", 3)
        self.empty_confirmations = self.param("empty_confirmations", 2)
        self.max_rounds = self.param("max_scene_rounds", 60)
        self.radius = self.param("identity_radius", 0.12)
        self.target_age = self.param("target_max_age", 45.0)
        self.map_frame = self.param("map_frame", "map")
        self.report_root = Path(
            self.param("report_dir", str(Path.home() / ".ros/robot_tasks"))
        ).expanduser()
        self.approach = ActionClient(
            self, Approach, "/nc/approach", callback_group=self.group
        )
        self.operation = ActionClient(
            self, OperateTarget, "/manipulation/operate", callback_group=self.group
        )
        self.scene = self.create_client(
            DetectScene, "/gc/detect_scene", callback_group=self.group
        )
        self.server = ActionServer(
            self,
            RunTask,
            "/gc/run_task",
            execute_callback=self.execute,
            goal_callback=self.accept,
            cancel_callback=self.cancel,
            callback_group=self.group,
        )

    def feedback(self, h, state, ledger=None, name=""):
        f = RunTask.Feedback()
        f.state = state
        f.target_name = name
        if ledger:
            f.completed, f.failed = ledger.counts()
        h.publish_feedback(f)

    def validate(self, obj):
        p = obj.position.point
        t = obj.position.header.stamp
        age = self.get_clock().now().nanoseconds / 1e9 - t.sec - t.nanosec * 1e-9
        if (
            not obj.name
            or obj.position.header.frame_id != self.map_frame
            or not all(math.isfinite(x) for x in (p.x, p.y, p.z))
        ):
            raise Problem(
                "PERCEPTION_INVALID",
                "Object requires a name and finite map-frame position",
            )
        if not -0.1 <= age <= self.target_age:
            raise Problem("PERCEPTION_STALE", f"Object observation age={age:.1f}s")

    def detect(self, h):
        last = ""
        for _ in range(self.scene_attempts):
            self.check(h)
            req = DetectScene.Request()
            req.description = h.request.description.strip()
            try:
                response = self.call(self.scene, req, h, 40.0)
                if not response.success:
                    raise Problem("PERCEPTION_FAILED", response.error)
                for obj in response.objects:
                    self.validate(obj)
                return response
            except Problem as e:
                if e.code in ("CANCELED", "UNAVAILABLE"):
                    raise
                last = str(e)
        raise Problem(
            "PERCEPTION_FAILED",
            "Scene observation failed after bounded retries: " + last,
        )

    def execute(self, h):
        result = RunTask.Result()
        ledger = Ledger(self.radius, self.max_attempts)
        events = []
        problem = None
        report = self.report_root / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
        )

        def event(state, **data):
            events.append(dict(state=state, monotonic=time.monotonic(), **data))

        try:
            if h.request.task not in ("operate_washer", "inspect_objects"):
                raise Problem(
                    "INVALID_TASK", "Expected operate_washer or inspect_objects"
                )
            if (
                h.request.task == "inspect_objects"
                and not 0 < len(h.request.description.strip()) <= 500
            ):
                raise Problem(
                    "INVALID_DESCRIPTION",
                    "Inspection requires an object description, e.g. 地上的水桶 (up to 500 characters)",
                )
            if h.request.execute and not self.allow:
                raise Problem("EXECUTION_DISABLED", "GC execution is disabled")
            # Fail before moving if the downstream manipulation implementation is absent.
            self.available(self.approach, h)
            navigation_only = h.request.task == "inspect_objects" and self.skip_grasp
            if h.request.execute and not navigation_only:
                self.available(self.operation, h)
            if h.request.task == "operate_washer":
                self.feedback(h, "APPROACHING", name="洗衣机")
                req = Approach.Goal()
                req.mode = "front_align"
                req.name = "洗衣机"
                req.execute = h.request.execute
                arrived = self.action(self.approach, req, h, 220.0, allow_failure=True)
                if not arrived.success:
                    raise Problem(arrived.code, arrived.message)
                event("washer_approach", code=arrived.code)
                if h.request.execute:
                    self.feedback(h, "OPERATING", name="洗衣机")
                    op = OperateTarget.Goal()
                    op.operation = "operate_washer"
                    op.name = "洗衣机"
                    op.execute = True
                    done = self.action(self.operation, op, h, 180.0, allow_failure=True)
                    if not done.success:
                        raise Problem(done.code, done.message)
                    result.completed = 1
                result.success = True
                result.code = "COMPLETED" if h.request.execute else "PREVIEW"
                result.message = (
                    "Washer operation completed"
                    if h.request.execute
                    else "Washer approach preview only; no operation performed"
                )
            else:
                idle = 0
                for round_id in range(self.max_rounds):
                    self.check(h)
                    self.feedback(h, "OBSERVING", ledger)
                    scene = self.detect(h)
                    event(
                        "observed",
                        round=round_id,
                        count=len(scene.objects),
                        evidence=scene.evidence_dir,
                    )
                    obj, record = ledger.select(scene.objects)
                    if obj is None:
                        idle += 1
                        if idle >= self.empty_confirmations:
                            break
                        continue
                    idle = 0
                    self.feedback(h, "APPROACHING", ledger, obj.name)
                    req = Approach.Goal()
                    req.mode = "grasp_ready"
                    req.name = obj.name
                    req.target = obj.position
                    req.execute = h.request.execute
                    try:
                        arrived = self.action(
                            self.approach, req, h, 240.0, allow_failure=True
                        )
                        if not arrived.success:
                            raise Problem(arrived.code, arrived.message)
                        if h.request.execute:
                            if navigation_only:
                                self.feedback(h, "GRASP_SKIPPED", ledger, obj.name)
                                event(
                                    "grasp_skipped", name=obj.name, id=obj.instance_id
                                )
                            else:
                                self.feedback(h, "GRASPING", ledger, obj.name)
                                op = OperateTarget.Goal()
                                op.operation = "grasp"
                                op.name = obj.name
                                op.instance_id = obj.instance_id
                                op.target = obj.position
                                op.execute = True
                                done = self.action(
                                    self.operation, op, h, 180.0, allow_failure=True
                                )
                                if not done.success:
                                    raise Problem(done.code, done.message)
                            ledger.finish(record, True)
                        else:
                            record.state = "planned"
                        event(
                            (
                                "target_completed"
                                if h.request.execute
                                else "target_planned"
                            ),
                            name=obj.name,
                            id=obj.instance_id,
                        )
                    except Problem as e:
                        # Geometry/navigation/confirmed grasp failures can be skipped.
                        # Unconfirmed stop, localization or system faults stop the whole task.
                        if e.code not in (
                            "NO_PATH",
                            "UNREACHABLE_HEIGHT",
                            "ARRIVAL_ERROR",
                            "WORKSPACE",
                            "STANDOFF_ERROR",
                            "NAV_FAILED",
                            "GRASP_FAILED",
                        ):
                            raise
                        ledger.finish(record, False)
                        event(
                            "target_failed",
                            name=obj.name,
                            id=obj.instance_id,
                            code=e.code,
                            error=str(e),
                        )
                else:
                    raise Problem(
                        "ROUND_LIMIT",
                        "Inspection observation limit reached; not declared complete",
                    )
                result.completed, result.failed = ledger.counts()
                result.success = result.failed == 0
                result.code = (
                    ("COMPLETED" if h.request.execute else "PREVIEW")
                    if result.success
                    else "PARTIAL"
                )
                if h.request.execute and navigation_only and result.success:
                    result.code = "NAVIGATION_ONLY_COMPLETED"
                result.message = (
                    "No remaining unprocessed objects in current view; no patrol-point movement. "
                    f'Completed={result.completed}, failed={result.failed}, previewed={sum(r.state=="planned" for r in ledger.records)}'
                )
                if navigation_only:
                    result.message += " Navigation-only test: completed counts arrivals; no grasp executed."
        except Exception as e:
            problem = e
        finally:
            if h.request.task == "inspect_objects":
                result.completed, result.failed = ledger.counts()
            try:
                report.parent.mkdir(parents=True, exist_ok=True)
                result.report_file = str(report)
                report.write_text(
                    json.dumps(
                        dict(
                            task=h.request.task,
                            execute=h.request.execute,
                            description=h.request.description,
                            skip_grasp=self.skip_grasp,
                            success=result.success and problem is None,
                            code=getattr(problem, "code", result.code),
                            error=str(problem) if problem else "",
                            completed=result.completed,
                            failed=result.failed,
                            records=[asdict(r) for r in ledger.records],
                            events=events,
                        ),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            except Exception as e:
                if problem is None:
                    problem = Problem("REPORT_FAILED", str(e))
        return self.finish(h, result, problem)


def main():
    run(GlobalController)
