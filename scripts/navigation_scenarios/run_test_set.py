#!/usr/bin/env python3
"""Run selected enabled scenes, then publish one run directory and HTML report."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from metrics import write_metrics
from run_report import report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", action="append", help="Enabled scene ID; repeat for a subset; omit for all")
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("test_set.yaml"))
    p.add_argument("--params-dir", type=Path, required=True, help="Directory with controller_server.yaml and optional costmap snapshots")
    p.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--public-base-url", default="http://127.0.0.1:18770")
    p.add_argument("--desc", default="", help="Description of this simulation run")
    a = p.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "149":
        p.error("Requires isolated ROS_DOMAIN_ID=149")
    config = yaml.safe_load(a.manifest.read_text())
    enabled = [s for s in config["scenes"] if s.get("enabled")]
    selected = set(a.scene or [s["id"] for s in enabled])
    if not selected or selected - {s["id"] for s in enabled}:
        p.error("Select one or more enabled scene IDs")
    config["scenes"] = [s for s in enabled if s["id"] in selected]
    if not (a.params_dir / "controller_server.yaml").is_file():
        p.error("Missing controller_server.yaml")
    for s in config["scenes"]:
        if not s.get("detection_target"):
            p.error(f"Missing detection_target for {s['id']}")
    root = Path(config["data_root"])
    started = datetime.now(timezone.utc)
    run = root / "simulation_runs" / started.strftime("run_%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True)
    (run / "test_set.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    (run / "run.json").write_text(json.dumps({"desc": a.desc, "started_utc": started.isoformat(), "params_source": str(a.params_dir.resolve()), "ros_domain_id": 149, "scene_ids": sorted(selected)}, indent=2))
    print("RUN", run, flush=True)
    report(run, root, config, a.public_base_url)
    for s in config["scenes"]:
        out = run / s["id"]
        out.mkdir()
        for name in ["controller_server.yaml", "local_costmap__local_costmap.yaml", "global_costmap__global_costmap.yaml"]:
            if (a.params_dir / name).exists():
                shutil.copy2(a.params_dir / name, out / name)
        print("START", s["id"], flush=True)
        for script in ["prepare_test.py", "test_navigation.py"]:
            command = [sys.executable, str(Path(__file__).with_name(script)), str(root / s["scene_dir"]), str(out), "--workspace", str(a.workspace)]
            if script == "prepare_test.py":
                command += ["--target", s["detection_target"]]
            with (out / (script + ".log")).open("w") as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                break
        if not (out / "metrics.json").exists():
            if not (out / "perception.json").exists():
                (out / "perception.json").write_text('{}')
            if not (out / "result.json").exists():
                (out / "result.json").write_text(json.dumps({"status": "failed", "failure_stage": "infrastructure", "error": f"{script} exited {code}; inspect logs"}))
            write_metrics(out)
        summary = report(run, root, config, a.public_base_url)
        print("DONE", s["id"], flush=True)
    print("REPORT", a.public_base_url.rstrip('/') + '/simulation_runs/' + run.name + '/report.html', flush=True)
    return 0 if summary["simulation_completed_count"] == summary["scene_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
