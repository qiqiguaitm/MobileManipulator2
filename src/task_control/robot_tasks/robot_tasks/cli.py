"""Task client. Preview by default; Ctrl+C propagates cancellation."""

import argparse, json, time, signal
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from robot_task_interfaces.action import RunTask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["operate_washer", "inspect_objects"])
    p.add_argument(
        "description",
        nargs="?",
        default="",
        help="Inspection target description, e.g. 地上的水桶",
    )
    p.add_argument(
        "--description",
        "--name",
        dest="description_option",
        default="",
        help="Target description (--name is a legacy alias)",
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.description and a.description_option and a.description != a.description_option:
        p.error("Conflicting descriptions")
    description = (a.description or a.description_option).strip()
    if a.task == "inspect_objects" and not 0 < len(description) <= 500:
        p.error("inspect_objects requires an object description (1–500 characters)")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    n = Node("gc_task_client")
    client = ActionClient(n, RunTask, "/gc/run_task")
    handle = None
    future = None
    sent = None

    def spin_future(f, timeout=None):
        deadline = float("inf") if timeout is None else time.monotonic() + timeout
        while not f.done() and time.monotonic() < deadline:
            rclpy.spin_once(n, timeout_sec=0.1)

    try:
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError("GC is not running")
        goal = RunTask.Goal()
        goal.task = a.task
        goal.description = description
        goal.execute = a.execute

        def feedback(m):
            print(
                json.dumps(
                    dict(
                        state=m.feedback.state,
                        target=m.feedback.target_name,
                        completed=m.feedback.completed,
                        failed=m.feedback.failed,
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )

        sent = client.send_goal_async(goal, feedback_callback=feedback)
        spin_future(sent, 10)
        if not sent.done():
            # Keep the request observable briefly; if a late handle arrives, cancel it.
            sent.add_done_callback(
                lambda f: (
                    f.result().cancel_goal_async() if f.result().accepted else None
                )
            )
            spin_future(sent, 5)
            raise RuntimeError(
                "GC acceptance timed out; late cancel requested. Verify task status before retrying"
            )
        handle = sent.result()
        if not handle.accepted:
            raise RuntimeError("GC rejected task (busy or locked)")
        future = handle.get_result_async()
        spin_future(future)
        result = future.result()
        r = result.result
        print(
            json.dumps(
                dict(
                    status=result.status,
                    success=r.success,
                    code=r.code,
                    message=r.message,
                    completed=r.completed,
                    failed=r.failed,
                    report_file=r.report_file,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if r.success else 2
    except KeyboardInterrupt:
        if handle is None and sent is not None:
            spin_future(sent, 5)
            if sent.done():
                handle = sent.result()
        if handle and handle.accepted:
            cancel = handle.cancel_goal_async()
            spin_future(cancel, 5)
            if future:
                spin_future(future, 10)
            print("Cancellation requested; check GC result for confirmation.")
        return 130
    except Exception as e:
        print(str(e))
        return 2
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
