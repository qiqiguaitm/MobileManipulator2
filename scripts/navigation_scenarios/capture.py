#!/usr/bin/env python3
"""Start a capture stack, gate on localization, record, then verify. No driving commands."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name", help="Scene name: letters, digits, underscore or hyphen"
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start immediately after readiness for automated tests; default waits for start confirmation",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Seconds after recording starts; default 0 waits for stop confirmation",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("maps/auto_explore/remap_20261013_1349/navigation"),
    )
    parser.add_argument(
        "--existing-stack",
        action="store_true",
        help="Use an already running stack, never stop it",
    )
    parser.add_argument("--ready-timeout", type=float, default=180)
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    parser.add_argument("--description", default="")
    args = parser.parse_args()
    import math

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        parser.error("Invalid scene name")
    if (
        not math.isfinite(args.duration)
        or args.duration < 0
        or not math.isfinite(args.ready_timeout)
        or args.ready_timeout <= 0
    ):
        parser.error(
            "Duration must be finite and non-negative; ready timeout must be finite and positive"
        )
    if args.goal and not all(math.isfinite(v) for v in args.goal):
        parser.error("Goal must be finite")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    log_root = root / "log/navigation_scenarios"
    log_root.mkdir(parents=True, exist_ok=True)
    lock = (log_root / "capture.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            "Another capture owns this workspace; do not start a second recorder"
        )
    map_dir = args.map_dir.resolve()
    for name in ("map.yaml", "map.pgm", "GlobalMap_hdl.pcd"):
        if not (map_dir / name).is_file():
            raise RuntimeError("Missing map file: " + str(map_dir / name))
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free disk space")
    # Detect hardware owners across domains, including launches still starting.
    if not args.existing_stack:
        processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        owners = [
            line
            for line in processes.splitlines()
            if re.search(
                r"(/lib/(?:rslidar_sdk|fast_lio|hdl_localization|hdl_global_localization|tracer_base|nav2_map_server|robot_state_publisher)/|ros2 launch slam hdl_navigation_launch)",
                line,
            )
        ]
        if owners:
            raise RuntimeError(
                "Existing/partial hardware stack detected. Reuse with --existing-stack after checking it; no duplicate launch:\n"
                + "\n".join(owners)
            )
        link = json.loads(
            subprocess.check_output(
                ["ip", "-j", "-details", "link", "show", "can2"], text=True
            )
        )[0]
        bitrate = (
            link.get("linkinfo", {})
            .get("info_data", {})
            .get("bittiming", {})
            .get("bitrate")
        )
        if "UP" in link.get("flags", []):
            if bitrate != 500000:
                raise RuntimeError(
                    "can2 is UP with unexpected bitrate; refusing to reconfigure an active bus"
                )
        else:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "ip",
                    "link",
                    "set",
                    "can2",
                    "type",
                    "can",
                    "bitrate",
                    "500000",
                ],
                check=True,
            )
            subprocess.run(
                ["sudo", "-n", "ip", "link", "set", "can2", "up"], check=True
            )
    folder = log_root / (
        args.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    folder.mkdir()
    output = root / "data/navigation_scenarios"
    before = set(output.glob(args.name + "_*")) if output.exists() else set()
    launch = None
    recorder = None
    command = [
        "ros2",
        "launch",
        "slam",
        "hdl_navigation_launch.py",
        f"map_dir:={map_dir}",
        "launch_chassis:=true",
        "use_odom_fusion:=true",
        "use_odom_fused:=true",
        "enable_rviz:=false",
    ]
    (folder / "startup.json").write_text(
        json.dumps(
            {
                "command": command if not args.existing_stack else None,
                "map_dir": str(map_dir),
                "domain": os.environ.get("ROS_DOMAIN_ID"),
                "existing_stack": args.existing_stack,
            },
            indent=2,
        )
        + "\n"
    )
    print("Startup log:", folder / "navigation.log", flush=True)
    print(
        "Wait for LOCALIZATION READY and recording READY before teleoperating.",
        flush=True,
    )
    try:
        if not args.existing_stack:
            with (folder / "navigation.log").open("w") as log:
                launch = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        cmd = [
            sys.executable,
            str(root / "scripts/navigation_scenarios/record.py"),
            args.name,
            "--duration",
            str(args.duration),
            "--ready-timeout",
            str(args.ready_timeout),
            "--expected-map",
            str(map_dir / "map.yaml"),
            "--output",
            str(output),
            "--description",
            args.description,
        ]
        if not args.auto_start:
            cmd += ["--wait-for-start"]
        if args.goal:
            cmd += ["--goal", *map(str, args.goal)]
        recorder = subprocess.Popen(cmd, start_new_session=True)
        while recorder.poll() is None:
            if launch is not None and launch.poll() is not None:
                raise RuntimeError("Navigation launch exited; see navigation.log")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_owned(recorder)
        stop_owned(launch)
    scenes = set(output.glob(args.name + "_*")) - before if output.exists() else set()
    if len(scenes) != 1:
        print(
            "No complete recording was created. Check readiness/startup output.",
            flush=True,
        )
        return 2
    scene = scenes.pop()
    shutil.copy2(folder / "startup.json", scene / "startup.json")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/navigation_scenarios/verify.py"),
            str(scene),
            "--expected-map",
            str(map_dir / "map.yaml"),
        ]
    )
    print("Scene:", scene, flush=True)
    print("Startup diagnostics:", folder, flush=True)
    return result.returncode or (0 if recorder.returncode == 0 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
