"""Persistent robot02 runtime. Tasks do not own or tear down hardware processes."""

import argparse, fcntl, json, math, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / ".ros/robot_tasks/session"


def identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def read_state():
    try:
        return json.loads((ROOT / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def alive(state):
    return bool(
        state.get("pid")
        and state.get("identity")
        and identity(state["pid"]) == state["identity"]
    )


def write_state(state):
    p = ROOT / "state.tmp"
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    p.replace(ROOT / "state.json")


def stop_child(p):
    if p and p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=3)


def serve(args):
    # Lock lasts for the process lifetime, including shutdown. No second CAN owner.
    lock = (ROOT / "session.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    import numpy as np
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from ament_index_python.packages import (
        get_package_prefix,
        get_package_share_directory,
    )
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, String
    from std_srvs.srv import Trigger
    from rcl_interfaces.srv import GetParameters
    from board_approach.ros_io import RobotIO
    from board_approach.motion import (
        Motion,
        can_command,
        wait_for_inputs,
        wait_for_can_mode,
        wait_for_driver_subscription,
        parent_death_signal,
        configure_approach_controller,
        approach_controller_parameters,
    )
    from board_approach.safety import can_error
    from .motion_lease import MotionLease
    from .runtime_health import RuntimeHealth
    from .can_feedback import CanFeedbackGate, CanStatusReceiver, DrainedSocket

    folder = ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir()
    state = dict(
        pid=os.getpid(),
        identity=identity(os.getpid()),
        status="starting",
        log_dir=str(folder),
        skip_grasp=args.skip_grasp,
        processes={},
        fault="",
    )
    write_state(state)
    children = {}
    node = None
    old = {}
    owned = False
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    class RuntimeIO(RuntimeHealth, RobotIO):
        pass

    class Guard(RuntimeHealth, Motion):
        def __init__(self, target):
            self.lease = MotionLease()
            self.runtime_ready = False
            self.last_fault = None
            super().__init__(target, 0.10)
            self.can_gate = CanFeedbackGate()
            self.leg_active = False
            self.can_receiver = CanStatusReceiver(self.can)
            self.can = DrainedSocket(self.can)
            from collections import deque

            self.events = deque(maxlen=1000)
            self.fault_pub = self.create_publisher(
                String,
                "/robot_tasks/runtime/fault",
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            )
            self.create_subscription(
                Bool,
                "/nc/motion_permit",
                lambda m: self.lease.update(m.data, time.monotonic()),
                1,
            )
            self.create_service(
                Trigger, "/robot_tasks/runtime/check", self.check_runtime
            )
            self.create_service(
                Trigger, "/robot_tasks/runtime/reset", self.reset_runtime
            )
            self.create_service(Trigger, "/robot_tasks/runtime/hold", self.hold_runtime)
            self.publish_fault()

        def publish_fault(self):
            self.fault_pub.publish(String(data=self.failure or ""))

        def hold_runtime(self, req, res):
            self.failure = "Motion held by operator"
            self.enabled = False
            self.command = Twist()
            self.output.publish(Twist())
            self.publish_fault()
            res.success = True
            res.message = "Motion held; nodes remain running. Cancel/finish the active task, then use session reset."
            return res

        def check_runtime(self, req, res):
            try:
                if not self.runtime_ready:
                    raise RuntimeError("Runtime is starting or stopping")
                if self.failure:
                    raise RuntimeError(
                        self.failure + "; use session reset after checking the scene"
                    )
                self.health()
                error = can_error(self.can_state, time.monotonic() - self.can_time)
                if error:
                    raise RuntimeError(error)
                res.success = True
                res.message = "Ready"
            except Exception as e:
                res.message = str(e)
            return res

        def reset_runtime(self, req, res):
            try:
                if not self.runtime_ready:
                    raise RuntimeError("Runtime is not ready")
                if self.active_goals or self.lease.permits(time.monotonic()):
                    raise RuntimeError("Cancel active navigation before reset")
                self.health()
                error = can_error(self.can_state, time.monotonic() - self.can_time)
                if error:
                    raise RuntimeError(error)
                self.failure = None
                self.last_fault = None
                self.command = Twist()
                self.steering_w = 0.0
                self.lease = MotionLease()
                self.publish_fault()
                res.success = True
                res.message = "Motion guard reset; no task started"
            except Exception as e:
                res.message = str(e)
            return res

        def guard(self):
            now = time.monotonic()
            self.can_receiver.poll()
            self.can_state = self.can_receiver.state
            self.can_time = (
                self.can_receiver.stamp
                if self.can_receiver.stamp is not None
                else float("-inf")
            )
            if self.lease.expired(now) and self.active_goals and not self.failure:
                self.failure = "NC motion permit expired"
            requested = (
                self.runtime_ready
                and self.lease.permits(now)
                and bool(self.active_goals)
                and not self.failure
            )
            if requested and not self.leg_active:
                try:
                    self.health()
                    p = self.data["pose"].pose.pose.position
                    self.start_xy = np.array([p.x, p.y])
                    self.last_xy = self.start_xy.copy()
                    self.travel = 0.0
                    self.started = now
                    self.command = Twist()
                    self.command_time = now
                    self.steering_w = 0.0
                    from board_approach.safety import FreshDataGate

                    self.fresh_gate = FreshDataGate()
                    self.paused = False
                    self.can_gate.reset()
                except Exception as e:
                    self.failure = str(e)
                    requested = False
            self.leg_active = bool(requested)
            can_ready = False
            if requested:
                previous = self.can_gate.paused
                # Inspect every received fault frame, even if a normal frame followed it in the queue.
                fault_state = self.can_receiver.fault_in_batch
                can_ready, error = self.can_gate.update(
                    time.monotonic(),
                    fault_state if fault_state is not None else self.can_state,
                    self.can_receiver.stamp,
                )
                if error:
                    self.failure = error
                if previous != self.can_gate.paused:
                    self.command = Twist()
                    self.command_time = time.monotonic()
                    self.steering_w = 0.0
                    self.record(
                        dict(
                            stage=(
                                "can_feedback_pause"
                                if self.can_gate.paused
                                else "can_feedback_resumed"
                            ),
                            sequence=self.can_receiver.sequence,
                            feedback_age=time.monotonic() - self.can_time,
                            recovery_frames=self.can_gate.count,
                        )
                    )
            self.enabled = bool(requested and can_ready and not self.failure)
            if not self.enabled:
                self.command = Twist()
                self.command_time = time.monotonic()
                self.steering_w = 0.0
            # Legacy guard retains its strict CAN check, plus localization, obstacle and velocity checks.
            # It only sees enabled=True after the timestamped recovery gate permits motion.
            before_failure = self.failure
            super().guard()
            # The strict parent can cross 400 ms between our check and its own check.
            # It already sent zero. Treat only that new freshness failure as a pause.
            if (
                requested
                and before_failure is None
                and self.failure
                and self.failure.startswith("CAN feedback stale or incomplete")
            ):
                self.failure = None
                self.can_gate.paused = True
                self.can_gate.count = 0
                if self.can_gate.pause_since is None:
                    self.can_gate.pause_since = time.monotonic()
                self.enabled = False
                self.command = Twist()
                self.steering_w = 0.0
                self.output.publish(Twist())
                self.record(
                    dict(
                        stage="can_feedback_pause",
                        reason="strict freshness boundary",
                        sequence=self.can_receiver.sequence,
                    )
                )
            if self.failure != self.last_fault:
                self.last_fault = self.failure
                self.publish_fault()

    def spawn(name, command):
        p = subprocess.Popen(
            command,
            stdout=open(folder / (name + ".log"), "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=parent_death_signal,
        )
        children[name] = p
        state["processes"][name] = p.pid
        write_state(state)
        return p

    try:
        links = json.loads(
            subprocess.check_output(["/usr/sbin/ip", "-json", "link", "show", "can2"])
        )
        if "UP" in links[0]["flags"]:
            raise RuntimeError("can2 is already owned; will not take over")
        if (
            subprocess.run(
                ["pgrep", "-x", "tracer_base_nod"], capture_output=True
            ).returncode
            == 0
        ):
            raise RuntimeError("A chassis driver is already running")
        if (
            subprocess.run(
                ["pgrep", "-f", "^/.*realsense2_camera_node( |$)"], capture_output=True
            ).returncode
            == 0
        ):
            raise RuntimeError(
                "Camera already owned; stop its temporary owner before starting the session"
            )
        node = RuntimeIO(with_camera=False)
        node.pump(2)
        node.health()
        if node.active_goals:
            raise RuntimeError("Navigation already active")
        names = {name for name, _ in node.get_node_names_and_namespaces()}
        if names & {"global_controller", "navigation_controller", "scene_perception"}:
            raise RuntimeError(
                "Task controllers already running; will not create duplicates"
            )
        target = dict(
            scan_frame=node.data["scan"].header.frame_id,
            base_from_scan=node.transform(
                "base_link", node.data["scan"].header.frame_id
            ).tolist(),
        )
        node.destroy_node()
        node = None
        camera = spawn(
            "camera",
            ["bash", str(Path(args.workspace) / "scripts/run_top_camera_rsusb.sh")],
        )
        can_command("type", "can", "bitrate", "500000")
        owned = True
        can_command("up")
        node = Guard(target)
        wait_for_inputs(node)
        executable = str(
            Path(get_package_prefix("tracer_base")) / "lib/tracer_base/tracer_base_node"
        )
        driver = spawn(
            "driver",
            [
                executable,
                "--ros-args",
                "-r",
                "/cmd_vel:=/board_approach/safe_cmd_vel",
                "-p",
                "port_name:=can2",
                "-p",
                "publish_tf:=false",
                "-p",
                "odom_frame:=wheel_odom",
                "-p",
                "odom_topic_name:=/robot_tasks/wheel_odom",
                "-p",
                "command_timeout:=0.4",
            ],
        )
        wait_for_can_mode(node, driver)
        wait_for_driver_subscription(node, driver)
        if len(node.get_publishers_info_by_topic("/board_approach/safe_cmd_vel")) != 1:
            raise RuntimeError("Unexpected safe velocity publisher")
        endpoint = node.get_subscriptions_info_by_topic("/board_approach/safe_cmd_vel")[
            0
        ]
        check = node.create_client(
            GetParameters,
            endpoint.node_namespace.rstrip("/")
            + "/"
            + endpoint.node_name
            + "/get_parameters",
        )
        if not check.wait_for_service(timeout_sec=3):
            raise RuntimeError("Driver watchdog unavailable")
        values = node.await_result(
            check.call_async(GetParameters.Request(names=["command_timeout"])), 3
        ).values
        if len(values) != 1 or abs(values[0].double_value - 0.4) > 0.001:
            raise RuntimeError("Driver watchdog not configured")
        desired = approach_controller_parameters(0.10, True)
        desired.update(
            {
                "goal_checker.xy_goal_tolerance": 0.035,
                "goal_checker.yaw_goal_tolerance": 0.01,
            }
        )
        configure_approach_controller(node, desired, old)
        config_dir = Path(get_package_share_directory("robot_tasks")) / "config"
        config = str(config_dir / "controllers.yaml")
        profile = str(config_dir / "robot02.yaml")
        parameters = ["--params-file", config, "--params-file", profile]
        prefix = Path(get_package_prefix("robot_tasks")) / "lib/robot_tasks"
        spawn(
            "gc",
            [
                str(prefix / "gc"),
                "--ros-args",
                *parameters,
                "-p",
                "allow_execution:=true",
                "-p",
                "skip_grasp:=" + str(args.skip_grasp).lower(),
            ],
        )
        spawn(
            "nc",
            [
                str(prefix / "nc"),
                "--ros-args",
                *parameters,
                "-p",
                "allow_execution:=true",
                "-p",
                "require_motion_runtime:=true",
            ],
        )
        spawn(
            "perception",
            [
                str(prefix / "scene_perception"),
                "--ros-args",
                *parameters,
                "-p",
                "workspace:=" + str(Path(args.workspace).expanduser().resolve()),
            ],
        )
        # Camera startup can take 10+ seconds. Keep hardware output zero throughout.
        from sensor_msgs.msg import Image

        frames = {}
        for topic in (
            "/camera/top/color/image_raw",
            "/camera/top/aligned_depth_to_color/image_raw",
        ):
            node.create_subscription(
                Image,
                topic,
                lambda m, key=topic: frames.update({key: time.monotonic()}),
                rclpy.qos.qos_profile_sensor_data,
            )
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            node.pump(0.1)
            if any(p.poll() is not None for p in children.values()):
                raise RuntimeError("A runtime process exited at startup")
            if len(frames) == 2 and all(
                time.monotonic() - v < 0.8 for v in frames.values()
            ):
                break
        else:
            raise RuntimeError("Camera RGB/depth startup timeout")
        node.runtime_ready = True
        state["status"] = "ready"
        write_state(state)
        last = 0.0
        saved_fault = None
        while rclpy.ok():
            node.pump(0.05)
            if any(p.poll() is not None for p in children.values()):
                raise RuntimeError(
                    "A runtime process exited; shutting down owned hardware"
                )
            if time.monotonic() - last >= 1:
                last = time.monotonic()
                state.update(
                    status="fault" if node.failure else "ready",
                    fault=node.failure or "",
                    moving_enabled=node.enabled,
                    updated_monotonic=last,
                    travel=node.travel,
                    can_feedback_paused=node.leg_active and node.can_gate.paused,
                    can_feedback_age=(
                        last - node.can_time if math.isfinite(node.can_time) else None
                    ),
                    can_recovery_frames=node.can_gate.count,
                )
                if node.can_receiver.pending:
                    with (folder / "can_frames.jsonl").open("a") as log:
                        while node.can_receiver.pending:
                            log.write(
                                json.dumps(node.can_receiver.pending.popleft()) + "\n"
                            )
                write_state(state)
            if not node.failure:
                saved_fault = None
            if node.failure and node.failure != saved_fault:
                saved_fault = node.failure
                (folder / "fault.json").write_text(
                    json.dumps(
                        dict(
                            error=node.failure,
                            events=list(node.events),
                            can_frames=list(node.can_receiver.recent),
                        ),
                        indent=2,
                    )
                )
                if getattr(node, "obstacle_evidence", None):
                    from board_approach.obstacle_evidence import save_trigger

                    save_trigger(folder, node.obstacle_evidence)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        state["error"] = str(e)
        print(str(e), flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        state["status"] = "stopping"
        write_state(state)
        if node and hasattr(node, "output"):
            node.runtime_ready = False
            node.enabled = False
            node.output.publish(Twist())
        # Driver stops before waiting for task cancellation; CAN closes on every exit.
        stop_child(children.get("driver"))
        if node and hasattr(node, "guard_timer"):
            node.guard_timer.cancel()
        if owned:
            try:
                can_command("down")
            except Exception as e:
                state["cleanup_error"] = str(e)
        for name in ("gc", "nc", "perception", "camera"):
            stop_child(children.get(name))
        if node:
            if old:
                try:
                    node.parameters(old)
                except Exception as e:
                    state["restore_error"] = str(e)
            if hasattr(node, "can"):
                node.can.close()
            node.destroy_node()
        rclpy.try_shutdown()
        state["status"] = "stopped"
        write_state(state)
    return 1 if state.get("error") else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["start", "stop", "status", "reset", "hold", "serve"]
    )
    parser.add_argument(
        "--workspace", default=str(Path.home() / "junbin.zhang/MobileManipulator2")
    )
    parser.add_argument(
        "--skip-grasp",
        action="store_true",
        help="Navigation-only inspection; no manipulation requests",
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "serve":
        return serve(args)
    state = read_state()
    if args.command == "status":
        state["process_alive"] = alive(state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if alive(state) else 1
    if args.command == "stop":
        if not alive(state):
            print("Session is already stopped")
            return 0
        os.kill(state["pid"], signal.SIGINT)
        deadline = time.monotonic() + 35
        while alive(state) and time.monotonic() < deadline:
            time.sleep(0.1)
        current = read_state()
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return 1 if alive(current) or current.get("cleanup_error") else 0
    if args.command in ("reset", "hold"):
        if not alive(state):
            print("Session is not running")
            return 1
        import rclpy
        from std_srvs.srv import Trigger

        rclpy.init()
        n = rclpy.create_node("runtime_reset_client")
        try:
            client = n.create_client(Trigger, "/robot_tasks/runtime/" + args.command)
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError("Runtime unavailable")
            f = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(n, f, timeout_sec=3)
            if not f.done():
                raise RuntimeError("Reset timed out")
            r = f.result()
            print(r.message)
            return 0 if r.success else 1
        finally:
            n.destroy_node()
            rclpy.shutdown()
    if alive(state):
        if state.get("skip_grasp") != args.skip_grasp:
            print(
                "Session already running with different skip_grasp setting; no processes restarted. Stop explicitly to change it."
            )
            return 1
        if state.get("status") in ("ready", "fault"):
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0 if state["status"] == "ready" else 1
    else:
        command = [
            sys.executable,
            "-m",
            "robot_tasks.session",
            "serve",
            "--workspace",
            args.workspace,
        ]
        if args.skip_grasp:
            command.append("--skip-grasp")
        with (ROOT / "supervisor.log").open("a") as output:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    deadline = time.monotonic() + 55
    while time.monotonic() < deadline:
        current = read_state()
        if alive(current) and current.get("status") == "ready":
            print(json.dumps(current, ensure_ascii=False, indent=2))
            return 0
        if current.get("error") and current.get("pid") != state.get("pid"):
            print(json.dumps(current, ensure_ascii=False, indent=2))
            return 1
        time.sleep(0.2)
    print(
        "Session startup has not completed; inspect session status and supervisor.log"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
