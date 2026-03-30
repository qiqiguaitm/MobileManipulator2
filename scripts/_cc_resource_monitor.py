#!/usr/bin/env python3
"""
Resource monitor for MobileManipulator2 full-manual system.

Records CPU%, RSS, and per-module breakdown for all ROS2 nodes
launched by `make full-manual`, plus Jetson system-level stats
(GPU, total RAM, temperatures, power).

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/_cc_resource_monitor.py              # live dashboard
    python3 scripts/_cc_resource_monitor.py --log         # live + log to file
    python3 scripts/_cc_resource_monitor.py --interval 3  # sample every 3s
    python3 scripts/_cc_resource_monitor.py --duration 60 # run for 60s then quit
"""

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time

# ── Module definitions ──────────────────────────────────────────────
# Each module: (display_name, list of process-match patterns)
# Patterns are matched against the full cmdline (case-insensitive).
MODULES = [
    # --- Localization pipeline ---
    ("Localization", [
        "fast_lio",
        "hdl_localization",
        "hdl_global_localization",
        "odom_fusion",
        "odom_frame_converter",
        "odom_relay",
        "delayed_cloud_relay",
        "tf_republisher",
        "globalmap_publisher",
    ]),
    # --- Nav2 stack ---
    ("Nav2", [
        "controller_server",
        "planner_server",
        "bt_navigator",
        "behavior_server",
        "smoother_server",
        "waypoint_follower",
        "velocity_smoother",
        "map_server",
        "lifecycle_manager",
    ]),
    # --- Sensors (lidar, imu, chassis) ---
    ("Lidar", [
        "rslidar",
        "pointcloud_to_laserscan",
    ]),
    ("IMU", [
        "hipnuc",
    ]),
    ("Chassis", [
        "tracer_base",
    ]),
    # --- Cameras (split by device, match __node:=top/hand/chassis) ---
    ("Cam-Top", [
        r"realsense2_camera_node.*__node:=top",
    ]),
    ("Cam-Hand", [
        r"realsense2_camera_node.*__node:=hand",
    ]),
    ("Cam-Chassis", [
        r"realsense2_camera_node.*__node:=chassis",
    ]),
    ("Camera", [
        "realsense2_camera_node",  # fallback for unmatched cameras
    ]),
    # --- Arm ---
    ("Arm-Grasp", [
        "piper_grasp_node",
    ]),
    ("Arm-TF", [
        "static_transform_publisher.*link8",
        "static_transform_publisher.*gripper",
        "robot_state_publisher",
        "joint_state_relay",
        "joint_state_static",
    ]),
    # --- Perception (split by function) ---
    ("Percept-Multi", [
        "multi_camera_perception",
    ]),
    ("Percept-Grasp", [
        "perception_grasp_node",
    ]),
    ("Percept-Viz", [
        "perception_grasp_rviz",
        "multi_camera_rviz",
    ]),
    # --- Control ---
    ("ManualCtrl", [
        "manual_control_node",
    ]),
    ("CleanerMgr", [
        "cleaner_manager_node",
    ]),
    ("ObjSelector", [
        "object_selector_node",
    ]),
    # --- Visualization ---
    ("RViz", [
        "rviz2",
    ]),
]

# ── Jetson tegrastats parsing ───────────────────────────────────────

def parse_tegrastats_line(line):
    """Parse one line from tegrastats into a dict."""
    info = {}
    m = re.search(r'RAM (\d+)/(\d+)MB', line)
    if m:
        info['ram_used_mb'] = int(m.group(1))
        info['ram_total_mb'] = int(m.group(2))

    m = re.search(r'GR3D_FREQ (\d+)%', line)
    if m:
        info['gpu_pct'] = int(m.group(1))

    m = re.search(r'CPU \[([^\]]+)\]', line)
    if m:
        cores = m.group(1).split(',')
        pcts = []
        for c in cores:
            cm = re.match(r'(\d+)%', c.strip())
            if cm:
                pcts.append(int(cm.group(1)))
        info['cpu_pcts'] = pcts
        info['cpu_avg_pct'] = sum(pcts) / len(pcts) if pcts else 0

    for tag in ('cpu', 'gpu', 'tj', 'soc0', 'soc1', 'soc2'):
        m = re.search(rf'{tag}@([\d.]+)C', line)
        if m:
            info[f'temp_{tag}'] = float(m.group(1))

    for tag in ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0'):
        m = re.search(rf'{tag} (\d+)mW', line)
        if m:
            info[f'power_{tag.lower()}_mw'] = int(m.group(1))

    m = re.search(r'SWAP (\d+)/(\d+)MB', line)
    if m:
        info['swap_used_mb'] = int(m.group(1))
        info['swap_total_mb'] = int(m.group(2))

    return info


class TegrastatsReader:
    """Non-blocking tegrastats reader via subprocess."""
    def __init__(self, interval_ms=1000):
        self.proc = None
        self.interval_ms = interval_ms
        self.last_info = {}

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ['sudo', '-S', 'tegrastats', '--interval', str(self.interval_ms)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self.proc.stdin.write('didi\n')
            self.proc.stdin.flush()
        except Exception as e:
            print(f"[WARN] tegrastats failed: {e}", file=sys.stderr)

    def read(self):
        if not self.proc or self.proc.poll() is not None:
            return self.last_info
        import select
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0)
            if not ready:
                break
            line = self.proc.stdout.readline()
            if line:
                self.last_info = parse_tegrastats_line(line)
        return self.last_info

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ── Process scanning ────────────────────────────────────────────────

def get_all_procs():
    """Return list of (pid, cpu%, rss_kb, nice, cmdline) for all processes."""
    try:
        out = subprocess.check_output(
            ['ps', '-eo', 'pid,pcpu,rss,ni,args', '--no-headers'],
            text=True, timeout=5
        )
    except Exception:
        return []
    procs = []
    for line in out.strip().split('\n'):
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
            rss = int(parts[2])
            ni = parts[3].strip()
            ni = int(ni) if ni != '-' else 0
            cmdline = parts[4]
            procs.append((pid, cpu, rss, ni, cmdline))
        except (ValueError, IndexError):
            continue
    return procs


def short_name(cmdline):
    """Extract a readable short name from cmdline."""
    # ROS2 node name from __node:=xxx
    m = re.search(r'__node:=(\S+)', cmdline)
    if m:
        node_name = m.group(1)
        # For realsense, show as "realsense2 (top)" etc.
        if 'realsense2_camera' in cmdline:
            return f"realsense2 ({node_name})"
        return node_name
    # Fallback: executable name
    cmd = cmdline.split('--ros-args')[0].strip()
    name = cmd.split('/')[-1].strip()
    name = name.replace('.py', '')
    return name[:30]


def match_module(cmdline, patterns):
    for pat in patterns:
        if re.search(pat, cmdline, re.IGNORECASE):
            return True
    return False


def classify_procs(procs):
    """Group processes into modules. Returns {module_name: [(pid, cpu, rss, ni, short_name)]}."""
    result = {name: [] for name, _ in MODULES}
    result['Other ROS'] = []
    assigned = set()

    for pid, cpu, rss, ni, cmdline in procs:
        # Skip launch wrapper processes (they match too many patterns)
        if 'ros2 launch' in cmdline or 'ros2 run' in cmdline:
            continue
        for mod_name, patterns in MODULES:
            if match_module(cmdline, patterns):
                result[mod_name].append((pid, cpu, rss, ni, short_name(cmdline)))
                assigned.add(pid)
                break

    # Catch unmatched ROS2 nodes
    for pid, cpu, rss, ni, cmdline in procs:
        if pid not in assigned and ('ros' in cmdline.lower() or '__node:=' in cmdline or '__ns:=' in cmdline):
            result['Other ROS'].append((pid, cpu, rss, ni, short_name(cmdline)))

    return result


# ── Display ─────────────────────────────────────────────────────────

def fmt_mb(kb):
    return f"{kb / 1024:.0f}"


def render_dashboard(classified, sys_info, sample_num):
    """Render a terminal dashboard with per-process detail."""
    lines = []
    now = datetime.datetime.now().strftime('%H:%M:%S')
    lines.append(f"\033[2J\033[H")  # clear screen
    lines.append(f"═══════════════════════════════════════════════════════════════════════")
    lines.append(f"  MobileManipulator2 Resource Monitor   #{sample_num}  {now}")
    lines.append(f"═══════════════════════════════════════════════════════════════════════")

    # System overview
    ram_used = sys_info.get('ram_used_mb', '?')
    ram_total = sys_info.get('ram_total_mb', '?')
    gpu_pct = sys_info.get('gpu_pct', '?')
    cpu_avg = sys_info.get('cpu_avg_pct', '?')
    temp_cpu = sys_info.get('temp_cpu', '?')
    temp_gpu = sys_info.get('temp_gpu', '?')
    pwr_total = (sys_info.get('power_vin_sys_5v0_mw', 0)) / 1000
    pwr_gpu = sys_info.get('power_vdd_gpu_soc_mw', 0) / 1000
    pwr_cpu = sys_info.get('power_vdd_cpu_cv_mw', 0) / 1000
    swap_used = sys_info.get('swap_used_mb', 0)

    cpu_avg_s = f"{cpu_avg:.0f}" if isinstance(cpu_avg, (int, float)) else str(cpu_avg)

    lines.append(f"")
    lines.append(f"  System: RAM {ram_used}/{ram_total}MB  SWAP {swap_used}MB  "
                 f"CPU avg {cpu_avg_s}%  GPU {gpu_pct}%")
    lines.append(f"  Temp:   CPU {temp_cpu}\u00b0C  GPU {temp_gpu}\u00b0C  "
                 f"Power: Total {pwr_total:.1f}W  GPU {pwr_gpu:.1f}W  CPU {pwr_cpu:.1f}W")
    lines.append(f"")

    # Per-process table
    hdr = f"  {'Process':<30} {'PID':>6} {'NI':>4} {'CPU%':>6} {'RSS':>7}"
    lines.append(hdr)
    lines.append(f"  {'─'*30} {'─'*6} {'─'*4} {'─'*6} {'─'*7}")

    total_cpu = 0.0
    total_rss = 0

    module_order = [name for name, _ in MODULES] + ['Other ROS']

    for mod_name in module_order:
        procs = classified.get(mod_name, [])
        if not procs:
            continue
        mod_cpu = sum(p[1] for p in procs)
        mod_rss = sum(p[2] for p in procs)
        total_cpu += mod_cpu
        total_rss += mod_rss

        sorted_procs = sorted(procs, key=lambda x: x[1], reverse=True)

        if len(sorted_procs) == 1:
            # Single process — compact: module + process on one line
            pid, cpu, rss, ni, name = sorted_procs[0]
            ni_str = f"{ni:+d}" if ni != 0 else "0"
            label = f"{mod_name}: {name}"
            lines.append(f"  {label:<30} {pid:>6} {ni_str:>4} {cpu:>5.1f}% {fmt_mb(rss):>6}M")
        else:
            # Multi process — header + indented children
            lines.append(f"  \033[1m◆ {mod_name:<28} {'':>6} {'':>4} {mod_cpu:>5.0f}% {fmt_mb(mod_rss):>6}M\033[0m")
            for pid, cpu, rss, ni, name in sorted_procs:
                ni_str = f"{ni:+d}" if ni != 0 else "0"
                lines.append(f"    {name:<28} {pid:>6} {ni_str:>4} {cpu:>5.1f}% {fmt_mb(rss):>6}M")

    lines.append(f"  {'─'*30} {'─'*6} {'─'*4} {'─'*6} {'─'*7}")
    lines.append(f"  \033[1m{'TOTAL':<30} {'':>6} {'':>4} {total_cpu:>5.0f}% {fmt_mb(total_rss):>6}M\033[0m")
    lines.append(f"")

    # Per-core CPU
    cpu_pcts = sys_info.get('cpu_pcts', [])
    if cpu_pcts:
        core_strs = [f"{i}:{p:2d}%" for i, p in enumerate(cpu_pcts)]
        lines.append(f"  CPU cores: {' '.join(core_strs)}")
        lines.append(f"")

    lines.append(f"  Press Ctrl+C to stop")
    return '\n'.join(lines)


def build_log_record(classified, sys_info, sample_num):
    """Build a JSON-serializable record for logging."""
    rec = {
        'sample': sample_num,
        'ts': datetime.datetime.now().isoformat(),
        'system': sys_info,
        'modules': {},
    }
    for mod_name in [name for name, _ in MODULES] + ['Other ROS']:
        procs = classified.get(mod_name, [])
        if not procs:
            continue
        rec['modules'][mod_name] = {
            'n_procs': len(procs),
            'cpu_pct': round(sum(p[1] for p in procs), 1),
            'rss_mb': round(sum(p[2] for p in procs) / 1024, 1),
            'procs': [
                {'pid': p[0], 'cpu': p[1], 'rss_mb': round(p[2]/1024, 1), 'ni': p[3], 'name': p[4]}
                for p in sorted(procs, key=lambda x: x[1], reverse=True)
            ],
        }
    return rec


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MobileManipulator2 resource monitor')
    parser.add_argument('--interval', type=float, default=2.0, help='Sample interval (seconds)')
    parser.add_argument('--duration', type=float, default=0, help='Run duration (0=infinite)')
    parser.add_argument('--log', action='store_true', help='Log samples to JSONL file')
    parser.add_argument('--log-file', type=str, default='', help='Log file path (auto-generated if empty)')
    args = parser.parse_args()

    log_fp = None
    if args.log:
        if not args.log_file:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            args.log_file = f'/data/workspace/MobileManipulator2/scripts/resource_log_{ts}.jsonl'
        log_fp = open(args.log_file, 'w')
        print(f"[LOG] Writing to {args.log_file}")

    tegra = TegrastatsReader(interval_ms=int(args.interval * 1000))
    tegra.start()
    time.sleep(1)

    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    sample_num = 0
    start_time = time.time()

    try:
        while not stop:
            sample_num += 1
            sys_info = tegra.read()
            all_procs = get_all_procs()
            classified = classify_procs(all_procs)

            dashboard = render_dashboard(classified, sys_info, sample_num)
            print(dashboard, flush=True)

            if log_fp:
                rec = build_log_record(classified, sys_info, sample_num)
                log_fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
                log_fp.flush()

            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

            time.sleep(args.interval)
    finally:
        tegra.stop()
        if log_fp:
            log_fp.close()
            print(f"\n[LOG] Saved {sample_num} samples to {args.log_file}")

    print(f"\n[DONE] {sample_num} samples collected over {time.time() - start_time:.0f}s")


if __name__ == '__main__':
    main()
