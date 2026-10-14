"""Direct front-align client for navigation-only washer tests; no runtime teardown."""

import argparse, json, signal, time
import rclpy
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
from robot_task_interfaces.action import Approach


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def interrupt(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    n = rclpy.create_node("nc_approach_client")
    client = ActionClient(n, Approach, "/nc/approach")
    sent = handle = result = None

    def wait(f, seconds):
        end = time.monotonic() + seconds
        while not f.done() and time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=0.1)
        if not f.done():
            raise RuntimeError("NC response timed out")
        return f.result()

    try:
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError("NC is not running; start the session first")
        req = Approach.Goal()
        req.mode = "front_align"
        req.name = args.name
        req.execute = args.execute
        sent = client.send_goal_async(
            req, feedback_callback=lambda m: print(m.feedback.state, flush=True)
        )
        handle = wait(sent, 10)
        if not handle.accepted:
            raise RuntimeError("NC rejected goal")
        result = handle.get_result_async()
        r = wait(result, 220).result
        print(
            json.dumps(
                dict(
                    success=r.success,
                    code=r.code,
                    message=r.message,
                    position_error=r.position_error,
                    yaw_error=r.yaw_error,
                ),
                ensure_ascii=False,
            )
        )
        return 0 if r.success else 2
    except (KeyboardInterrupt, Exception) as error:
        print(str(error) or "Interrupted", flush=True)
        confirmed = sent is None
        if handle is None and sent is not None:
            try:
                handle = wait(sent, 5)
            except Exception:
                sent.add_done_callback(
                    lambda f: (
                        f.result().cancel_goal_async() if f.result().accepted else None
                    )
                )
        if handle and handle.accepted:
            try:
                wait(handle.cancel_goal_async(), 4)
                wait(result if result is not None else handle.get_result_async(), 10)
                print("NC goal terminated", flush=True)
                confirmed = True
            except Exception as e:
                print("Stop not confirmed: " + str(e), flush=True)
        elif handle is not None:
            confirmed = True
        if not confirmed:
            from std_srvs.srv import Trigger

            hold = n.create_client(Trigger, "/robot_tasks/runtime/hold")
            try:
                if not hold.wait_for_service(timeout_sec=2):
                    raise RuntimeError("Runtime hold service unavailable")
                response = wait(hold.call_async(Trigger.Request()), 3)
                print(response.message, flush=True)
            except Exception as e:
                print("STOP UNCONFIRMED: " + str(e), flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
