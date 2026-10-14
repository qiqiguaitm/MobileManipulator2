"""Real GC/NC action lifecycle with isolated fake Nav2/perception/manipulation."""

import os, time, threading

os.environ["ROS_DOMAIN_ID"] = "166"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
import pytest
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from action_msgs.msg import GoalStatus
from robot_task_interfaces.action import RunTask, Approach, OperateTarget
from robot_tasks.gc import GlobalController
from robot_tasks.nc import NavigationController
from robot_tasks.mock_backends import MockBackends


def await_future(f, timeout=20):
    end = time.monotonic() + timeout
    while not f.done() and time.monotonic() < end:
        time.sleep(0.02)
    assert f.done(), "Future timed out"
    return f.result()


@pytest.fixture
def stack(tmp_path):
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "allow_execution:=true",
            "-p",
            "allow_mock_front_execution:=true",
            "-p",
            f"report_dir:={tmp_path}",
        ]
    )
    mock = MockBackends()
    nc = NavigationController()
    gc = GlobalController()
    client_node = Node("test_client")
    executor = MultiThreadedExecutor(num_threads=12)
    for n in (mock, nc, gc, client_node):
        executor.add_node(n)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    cli = ActionClient(client_node, RunTask, "/gc/run_task")
    assert cli.wait_for_server(timeout_sec=5)
    deadline = time.monotonic() + 5
    while True:
        try:
            nc.robot_pose()
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    time.sleep(0.2)
    yield mock, nc, gc, cli, client_node
    for n in (mock, nc, gc):
        n.closing = True
    time.sleep(0.5)  # Drain terminal results before destroying ROS waitables.
    executor.shutdown(timeout_sec=5)
    thread.join(timeout=5)
    for s in (mock.plan_server, mock.nav_server, mock.op_server, nc.server, gc.server):
        s.destroy()
    cli.destroy()
    for n in (client_node, gc, nc, mock):
        n.destroy_node()
    rclpy.shutdown()


def send(cli, task="inspect_objects", execute=True, description="杯子或瓶子"):
    req = RunTask.Goal()
    req.task = task
    req.execute = execute
    req.description = description
    h = await_future(cli.send_goal_async(req))
    assert h.accepted
    return h


def test_two_objects_loop(stack):
    mock, nc, gc, cli, _ = stack
    result = await_future(send(cli).get_result_async(), 30)
    assert result.status == GoalStatus.STATUS_SUCCEEDED
    assert result.result.completed == 2 and result.result.failed == 0
    assert mock.op_calls == ["杯子", "瓶子"] and mock.nav_calls == 2
    assert mock.detect_calls == 4


def test_description_is_passed_to_perception(stack):
    mock, nc, gc, cli, _ = stack
    result = await_future(
        send(cli, description="地上的杯子", execute=False).get_result_async()
    )
    assert result.result.success and mock.last_description == "地上的杯子"
    assert not mock.op_calls and mock.nav_calls == 0


def test_empty_description_is_rejected_before_actions(stack):
    mock, nc, gc, cli, _ = stack
    result = await_future(send(cli, description=" ").get_result_async())
    assert result.result.code == "INVALID_DESCRIPTION"
    assert mock.detect_calls == mock.nav_calls == 0 and not mock.op_calls


def test_navigation_only_needs_no_manipulation_server(stack):
    mock, nc, gc, cli, _ = stack
    gc.skip_grasp = True
    gc.operation = ActionClient(
        gc, OperateTarget, "/absent_manipulation", callback_group=gc.group
    )
    result = await_future(send(cli).get_result_async(), 30)
    assert result.result.success and result.result.code == "NAVIGATION_ONLY_COMPLETED"
    assert result.result.completed == 2 and mock.nav_calls == 2 and mock.op_calls == []


def test_skip_grasp_does_not_skip_washer_operation(stack):
    mock, nc, gc, cli, _ = stack
    gc.skip_grasp = True
    result = await_future(send(cli, "operate_washer").get_result_async())
    assert result.result.success and mock.op_calls == ["洗衣机"]


def test_washer_order(stack):
    mock, nc, gc, cli, _ = stack
    result = await_future(send(cli, "operate_washer").get_result_async())
    assert result.result.success and result.result.completed == 1
    assert mock.nav_calls == 1 and mock.op_calls == ["洗衣机"]


def test_same_nodes_survive_failure_and_next_task(stack):
    mock, nc, gc, cli, _ = stack
    mock.block_plans = True
    first = await_future(send(cli, "operate_washer").get_result_async())
    assert first.result.code == "NO_PATH" and not nc.motion_requested
    mock.block_plans = False
    second = await_future(send(cli, "operate_washer").get_result_async())
    assert second.result.success and mock.nav_calls == 1 and not nc.motion_requested
    third = await_future(send(cli, "operate_washer").get_result_async())
    assert third.result.success and mock.nav_calls == 2 and not nc.motion_requested


def test_runtime_fault_cancels_navigation_and_nodes_can_be_reused(stack):
    from std_srvs.srv import Trigger
    from std_msgs.msg import String

    mock, nc, gc, cli, n = stack
    nc.require_runtime = True
    mock.nav_delay = 10

    def ready(req, res):
        res.success = True
        return res

    service = n.create_service(Trigger, "/robot_tasks/runtime/check", ready)
    try:
        handle = send(cli, "operate_washer")
        deadline = time.monotonic() + 8
        while mock.nav_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert mock.nav_calls == 1 and nc.motion_requested
        nc.runtime_fault = "Injected obstacle stop"
        result = await_future(handle.get_result_async(), 10)
        assert result.result.code == "MOTION_GUARD" and not nc.motion_requested
        assert not mock.op_calls and not gc.interlock and not nc.interlock
        nc.runtime_fault = ""
        mock.nav_delay = 0.2
        result = await_future(send(cli, "operate_washer").get_result_async())
        assert result.result.success and mock.op_calls == ["洗衣机"]
    finally:
        n.destroy_service(service)


def test_initial_tf_discovery_waits_before_navigation(stack):
    from robot_tasks.ros_support import Problem

    mock, nc, gc, cli, _ = stack
    pose = nc.robot_pose
    ready_at = time.monotonic() + 0.4

    def delayed_pose():
        if time.monotonic() < ready_at:
            assert mock.nav_calls == 0
            raise Problem(
                "LOCALIZATION", "TF listener has not received its first transform"
            )
        return pose()

    nc.robot_pose = delayed_pose
    result = await_future(send(cli, "operate_washer").get_result_async())
    assert result.result.success and mock.nav_calls == 1


def test_preview_does_not_move_or_claim_completed(stack):
    mock, nc, gc, cli, _ = stack
    result = await_future(send(cli, execute=False).get_result_async(), 30)
    assert result.result.success and result.result.code == "PREVIEW"
    assert result.result.completed == 0 and not mock.op_calls and mock.nav_calls == 0


def test_detector_error_is_failure_not_empty(stack):
    mock, nc, gc, cli, _ = stack
    mock.scene_error = True
    result = await_future(send(cli).get_result_async())
    assert result.result.code == "PERCEPTION_FAILED" and not result.result.success
    assert mock.detect_calls == 3 and mock.nav_calls == 0


def test_skip_after_two_confirmed_grasp_failures(stack):
    mock, nc, gc, cli, _ = stack
    mock.fail_name = "杯子"
    result = await_future(send(cli).get_result_async(), 30)
    assert result.result.code == "PARTIAL"
    assert (result.result.completed, result.result.failed) == (1, 1)
    assert mock.op_calls == ["杯子", "杯子", "瓶子"]


def test_no_path_skips_without_operation(stack):
    mock, nc, gc, cli, _ = stack
    mock.block_plans = True
    result = await_future(send(cli).get_result_async(), 30)
    assert result.result.code == "PARTIAL" and result.result.failed == 2
    assert mock.nav_calls == 0 and not mock.op_calls


def test_cancel_propagates_and_busy_rejects(stack):
    mock, nc, gc, cli, _ = stack
    mock.nav_delay = 6
    h = send(cli)
    deadline = time.monotonic() + 10
    while not mock.nav_calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert mock.nav_calls == 1
    other = RunTask.Goal()
    other.task = "operate_washer"
    other.execute = True
    assert not await_future(cli.send_goal_async(other)).accepted
    await_future(h.cancel_goal_async())
    result = await_future(h.get_result_async(), 15)
    assert result.status == GoalStatus.STATUS_CANCELED and not mock.op_calls
    assert not gc.interlock and not nc.interlock


def test_mock_front_execution_guard(stack):
    mock, nc, gc, cli, _ = stack
    nc.allow_mock = False
    result = await_future(send(cli, "operate_washer").get_result_async())
    assert (
        result.result.code == "MOCK_TARGET" and not mock.nav_calls and not mock.op_calls
    )


def test_stale_target_rejected(stack):
    mock, nc, gc, cli, n = stack
    client = ActionClient(n, Approach, "/nc/approach")
    assert client.wait_for_server(timeout_sec=3)
    req = Approach.Goal()
    req.mode = "grasp_ready"
    req.target.header.frame_id = "map"
    req.target.point.x = 2.0
    req.target.point.z = 0.3
    h = await_future(client.send_goal_async(req))
    r = await_future(h.get_result_async())
    assert r.result.code == "TARGET_STALE" and not mock.nav_calls
    client.destroy()


def test_navigation_timeout_cancels_child(stack):
    mock, nc, gc, cli, _ = stack
    mock.nav_delay = 3
    nc.timeout = 0.1
    result = await_future(send(cli, "operate_washer").get_result_async(), 15)
    assert result.result.code == "TIMEOUT" and not mock.op_calls
    assert not nc.interlock


def test_localization_loss_stops_navigation(stack):
    mock, nc, gc, cli, _ = stack
    mock.nav_delay = 5
    h = send(cli, "operate_washer")
    deadline = time.monotonic() + 10
    while not mock.nav_calls and time.monotonic() < deadline:
        time.sleep(0.02)
    mock.tf_timer.cancel()
    result = await_future(h.get_result_async(), 15)
    assert result.result.code == "LOCALIZATION" and not mock.op_calls


def test_unconfirmed_cancel_locks_controllers(stack):
    mock, nc, gc, cli, _ = stack
    mock.nav_delay = 20
    mock.ignore_cancel = True
    nc.timeout = 0.1
    result = await_future(send(cli, "operate_washer").get_result_async(), 15)
    assert result.result.code == "CANCEL_UNCONFIRMED"
    assert nc.interlock and gc.interlock and not mock.op_calls


def test_no_targets_requires_two_successful_observations(stack):
    mock, nc, gc, cli, _ = stack
    mock.empty = True
    result = await_future(send(cli).get_result_async())
    assert (
        result.result.success
        and result.result.completed == 0
        and mock.detect_calls == 2
    )


def test_default_execution_gate(stack):
    mock, nc, gc, cli, _ = stack
    gc.allow = False
    result = await_future(send(cli).get_result_async())
    assert result.result.code == "EXECUTION_DISABLED" and mock.nav_calls == 0


def test_cli_sigint_cancels_promptly(stack, tmp_path):
    import subprocess, signal, sys

    mock, nc, gc, cli, _ = stack
    mock.nav_delay = 15
    output = tmp_path / "cli.log"
    with output.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from robot_tasks.cli import main; raise SystemExit(main())",
                "operate_washer",
                "--execute",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            until = time.monotonic() + 10
            while "APPROACHING" not in output.read_text() and time.monotonic() < until:
                time.sleep(0.05)
            assert "APPROACHING" in output.read_text()
            time.sleep(0.3)
            begin = time.monotonic()
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=10) == 130
            assert time.monotonic() - begin < 3
            assert not mock.op_calls
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_inspection_front_clearance_and_arrival_use_same_reference(stack):
    import math

    mock, nc, gc, cli, node = stack
    approach = ActionClient(node, Approach, "/nc/approach")
    assert approach.wait_for_server(timeout_sec=5)
    try:
        req = Approach.Goal()
        req.mode = "grasp_ready"
        req.name = "垃圾桶"
        req.execute = True
        req.target.header.frame_id = "map"
        req.target.header.stamp = node.get_clock().now().to_msg()
        req.target.point.x = 2.0
        req.target.point.z = 0.3
        handle = await_future(approach.send_goal_async(req))
        assert handle.accepted
        result = await_future(handle.get_result_async(), 15).result
        assert result.success and result.code == "ARRIVED"
        p = result.navigation_goal.pose.position
        assert math.hypot(2.0 - p.x, p.y) - nc.front_extent == pytest.approx(0.5)
        assert mock.nav_calls == 1 and not mock.op_calls
    finally:
        approach.destroy()
