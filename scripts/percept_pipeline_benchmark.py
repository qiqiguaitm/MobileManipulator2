#!/usr/bin/env python3
"""
Perception Pipeline End-to-End Latency Benchmark

Subscribes to /rosout to parse internal PERF/WORKER/FUSION logs from the
perception node, plus key topics for throughput rates. Treats dual-camera
parallel detection as a single unit.

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/_cc_percept_pipeline_benchmark.py --duration 30
"""

import argparse
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rcl_interfaces.msg import Log
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray

try:
    from perception.msg import Object3DArray
except ImportError:
    Object3DArray = None


# ---------------------------------------------------------------------------
# QoS profiles
# ---------------------------------------------------------------------------
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
)
DETECTION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
MARKER_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)
ROSOUT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST, depth=200,
    durability=QoSDurabilityPolicy.VOLATILE,
)


# ---------------------------------------------------------------------------
# Parsed log entries
# ---------------------------------------------------------------------------
@dataclass
class PerfEntry:
    camera: str
    wall_time: float    # time.time() when received
    parallel: int       # ms
    idle: int
    sam3_pre: int
    sam3_enc: int
    sam3_http: int
    sam3_mask: int
    sam3_total: int
    cdm_pre: int
    cdm_enc: int
    cdm_http: int
    cdm_post: int
    cdm_total: int
    post_dpub: int = 0
    post_3d: int = 0
    post_tf: int = 0
    post_build: int = 0
    post_total: int = 0
    wall: int = 0
    n_objs: int = 0


@dataclass
class FusionEntry:
    wall_time: float
    n_in: int
    n_fused: int
    n_tracked: int
    match_ms: int
    track_ms: int
    total_ms: int


@dataclass
class WorkerEntry:
    camera: str
    wall_time: float
    qwait: int
    detect: int
    pub: int
    fusion: int
    total: int


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
PERF_RE = re.compile(
    r'\[(?P<cam>top|chassis)\] PERF: '
    r'parallel=(?P<parallel>\d+)ms\(idle=(?P<idle>\d+)ms\) '
    r'A=\[pre:(?P<a_pre>\d+)\+enc:(?P<a_enc>\d+)\+http:(?P<a_http>\d+)\+mask:(?P<a_mask>\d+)=(?P<a_total>\d+)ms\] '
    r'B=\[pre:(?P<b_pre>\d+)\+enc:(?P<b_enc>\d+)\+http:(?P<b_http>\d+)\+post:(?P<b_post>\d+)=(?P<b_total>\d+)ms\]'
    r'(?P<rest>.*?)'
    r'wall=(?P<wall>\d+)ms'
)

POST_RE = re.compile(
    r'post=\[dpub:(?P<dpub>\d+)\+3d:(?P<t3d>\d+)\+tf:(?P<tf>\d+)\+build:(?P<build>\d+)=(?P<ptotal>\d+)ms\]'
)

OBJS_RE = re.compile(r'\|\s*(?P<n>\d+)\s+objs')

FUSION_RE = re.compile(
    r'\[FUSION\] in=(?P<n_in>\d+) fused=(?P<n_fused>\d+) tracked=(?P<n_tracked>\d+) '
    r'match=(?P<match>\d+)ms track=(?P<track>\d+)ms total=(?P<total>\d+)ms'
)

WORKER_RE = re.compile(
    r'\[(?P<cam>top|chassis)\] WORKER: '
    r'qwait=(?P<qwait>\d+)ms detect=(?P<detect>\d+)ms '
    r'pub=(?P<pub>\d+)ms fusion=(?P<fusion>\d+)ms total=(?P<total>\d+)ms'
)


# ---------------------------------------------------------------------------
# Topic rate counter
# ---------------------------------------------------------------------------
RATE_TOPICS = {
    "cam_top":      ("/camera/top/color/image_raw",                 Image,       SENSOR_QOS),
    "cam_chassis":  ("/camera/chassis/color/image_raw",             Image,       SENSOR_QOS),
    "fused":        ("/multi_camera_perception/fused/objects_3d",   "Object3DArray", DETECTION_QOS),
    "rviz_fused":   ("/multi_camera_rviz/fused/object_markers",    MarkerArray, MARKER_QOS),
}


# ---------------------------------------------------------------------------
# Benchmark node
# ---------------------------------------------------------------------------
class PipelineBenchmark(Node):

    def __init__(self, duration: float):
        super().__init__("percept_pipeline_benchmark")
        self.duration = duration
        self._lock = threading.Lock()

        # Internal log collections
        self.perfs: list[PerfEntry] = []
        self.fusions: list[FusionEntry] = []
        self.workers: list[WorkerEntry] = []

        # Topic rate counters
        self.topic_counts: dict[str, int] = {k: 0 for k in RATE_TOPICS}
        # RViz wall-time records for cross-node latency
        self.fused_walls: list[float] = []
        self.rviz_walls: list[float] = []

        # Subscribe to /rosout
        self.create_subscription(Log, "/rosout", self._rosout_cb, ROSOUT_QOS)
        self.get_logger().info("Subscribed: /rosout (for PERF/WORKER/FUSION)")

        # Subscribe to rate topics
        for key, (topic, msg_type, qos) in RATE_TOPICS.items():
            if msg_type == "Object3DArray":
                if Object3DArray is None:
                    continue
                msg_type = Object3DArray
            cb = self._make_rate_cb(key)
            self.create_subscription(msg_type, topic, cb, qos)
            self.get_logger().info(f"Subscribed: {topic}")

        self.start_time = time.time()
        self.create_timer(1.0, self._tick)

    # -- /rosout callback ----------------------------------------------------
    def _rosout_cb(self, msg: Log):
        if "multi_camera_perception" not in msg.name:
            return
        text = msg.msg
        wt = time.time()

        m = PERF_RE.search(text)
        if m:
            entry = PerfEntry(
                camera=m.group("cam"), wall_time=wt,
                parallel=int(m.group("parallel")), idle=int(m.group("idle")),
                sam3_pre=int(m.group("a_pre")), sam3_enc=int(m.group("a_enc")),
                sam3_http=int(m.group("a_http")), sam3_mask=int(m.group("a_mask")),
                sam3_total=int(m.group("a_total")),
                cdm_pre=int(m.group("b_pre")), cdm_enc=int(m.group("b_enc")),
                cdm_http=int(m.group("b_http")), cdm_post=int(m.group("b_post")),
                cdm_total=int(m.group("b_total")),
            )
            rest = m.group("rest")
            pm = POST_RE.search(rest)
            if pm:
                entry.post_dpub = int(pm.group("dpub"))
                entry.post_3d = int(pm.group("t3d"))
                entry.post_tf = int(pm.group("tf"))
                entry.post_build = int(pm.group("build"))
                entry.post_total = int(pm.group("ptotal"))
            entry.wall = int(m.group("wall"))
            om = OBJS_RE.search(text)
            if om:
                entry.n_objs = int(om.group("n"))
            with self._lock:
                self.perfs.append(entry)
            return

        m = FUSION_RE.search(text)
        if m:
            with self._lock:
                self.fusions.append(FusionEntry(
                    wall_time=wt,
                    n_in=int(m.group("n_in")),
                    n_fused=int(m.group("n_fused")),
                    n_tracked=int(m.group("n_tracked")),
                    match_ms=int(m.group("match")),
                    track_ms=int(m.group("track")),
                    total_ms=int(m.group("total")),
                ))
            return

        m = WORKER_RE.search(text)
        if m:
            with self._lock:
                self.workers.append(WorkerEntry(
                    camera=m.group("cam"), wall_time=wt,
                    qwait=int(m.group("qwait")),
                    detect=int(m.group("detect")),
                    pub=int(m.group("pub")),
                    fusion=int(m.group("fusion")),
                    total=int(m.group("total")),
                ))

    # -- rate counters -------------------------------------------------------
    def _make_rate_cb(self, key: str):
        def cb(msg):
            self.topic_counts[key] += 1
            if key == "fused":
                self.fused_walls.append(time.time())
            elif key == "rviz_fused":
                self.rviz_walls.append(time.time())
        return cb

    # -- tick ----------------------------------------------------------------
    def _tick(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            self.get_logger().info("Collection complete, generating report...")
            self._report()
            raise SystemExit(0)

        with self._lock:
            np_ = len(self.perfs)
            nf = len(self.fusions)
            nw = len(self.workers)
        self.get_logger().info(
            f"[{elapsed:.0f}/{self.duration:.0f}s] "
            f"PERF={np_}  FUSION={nf}  WORKER={nw}  "
            + "  ".join(f"{k}={v}" for k, v in self.topic_counts.items())
        )

    # -- report --------------------------------------------------------------
    def _report(self):
        elapsed = time.time() - self.start_time
        with self._lock:
            perfs = list(self.perfs)
            fusions = list(self.fusions)
            workers = list(self.workers)

        lines: list[str] = []
        W = 62
        sep = "=" * W

        lines.append("")
        lines.append(sep)
        lines.append(f"  PERCEPTION PIPELINE BENCHMARK ({elapsed:.1f}s)")
        lines.append(sep)

        # --- Topic Rates ---
        lines.append("")
        lines.append("[Topic Rates]")
        for key, (topic, _, _) in RATE_TOPICS.items():
            n = self.topic_counts[key]
            hz = n / elapsed if elapsed > 0 else 0.0
            lines.append(f"  {topic:<52s} {n:>5d} msgs  {hz:>5.1f} Hz")

        # --- Parallel Detection Cycles ---
        cycles = self._pair_detection_cycles(perfs)
        lines.append("")
        lines.append(f"[Parallel Detection Cycle]  (N={len(cycles)} cycles)")
        self._report_detection_cycles(cycles, lines)

        # --- Fusion & Tracking ---
        lines.append("")
        lines.append(f"[Fusion & Tracking]  (N={len(fusions)})")
        self._report_fusions(fusions, lines)

        # --- Worker Cycle ---
        lines.append("")
        lines.append(f"[Worker Cycle]  (N={len(workers)})")
        self._report_workers(workers, lines)

        # --- Cross-Node (RViz) ---
        lat_rviz = self._compute_rviz_latency()
        lines.append("")
        lines.append("[Cross-Node Latency]")
        self._stat_line(lines, "rviz (fused→markers)", lat_rviz)

        # --- Throughput Summary ---
        lines.append("")
        lines.append("[Throughput Summary]")
        n_fused = self.topic_counts.get("fused", 0)
        det_hz = len(cycles) / elapsed if elapsed > 0 else 0.0
        fused_hz = n_fused / elapsed if elapsed > 0 else 0.0
        rviz_hz = self.topic_counts.get("rviz_fused", 0) / elapsed if elapsed > 0 else 0.0
        lines.append(f"  Detection Cycle:  ~{det_hz:.1f} Hz")
        lines.append(f"  Fusion:           ~{fused_hz:.1f} Hz")
        lines.append(f"  RViz Markers:     ~{rviz_hz:.1f} Hz")
        if cycles:
            walls = [c["wall_max"] for c in cycles]
            lines.append(f"  Pipeline Period:  ~{np.median(walls):.0f}ms (p50 detect_wall)")

        lines.append(sep)
        report = "\n".join(lines)
        print(report)

    # -- detection cycle pairing ---------------------------------------------
    @staticmethod
    def _pair_detection_cycles(perfs: list[PerfEntry]) -> list[dict]:
        """Pair consecutive top/chassis PERF entries into detection cycles.
        A cycle = one top + one chassis entry within 500ms of each other."""
        tops = sorted([p for p in perfs if p.camera == "top"], key=lambda p: p.wall_time)
        chasses = sorted([p for p in perfs if p.camera == "chassis"], key=lambda p: p.wall_time)

        cycles = []
        ci = 0  # chassis index
        for t in tops:
            # find nearest chassis entry within ±500ms
            best = None
            best_gap = 999.0
            for j in range(ci, min(ci + 5, len(chasses))):
                gap = abs(chasses[j].wall_time - t.wall_time)
                if gap < best_gap:
                    best_gap = gap
                    best = j
                if chasses[j].wall_time > t.wall_time + 0.5:
                    break
            if best is not None and best_gap < 0.5:
                c = chasses[best]
                cycles.append({
                    "top": t, "chassis": c,
                    "wall_max": max(t.wall, c.wall),
                    "wall_top": t.wall, "wall_chassis": c.wall,
                    "bottleneck": "top" if t.wall >= c.wall else "chassis",
                })
                ci = best + 1
        return cycles

    def _report_detection_cycles(self, cycles: list[dict], lines: list[str]):
        if not cycles:
            lines.append("  (no paired cycles found)")
            return

        HDR = f"  {'':36s} {'n':>5s}  {'min':>6s}  {'mean':>6s}  {'p50':>6s}  {'p95':>6s}  {'max':>6s}"
        lines.append(HDR)

        # Helper: extract ms array from both cameras using max
        def _max_field(field: str):
            return np.array([max(getattr(c["top"], field), getattr(c["chassis"], field))
                             for c in cycles], dtype=float)

        def _field(cam: str, field: str):
            return np.array([getattr(c[cam], field) for c in cycles], dtype=float)

        # SAM3 sub-stages (parallel max)
        lines.append("  ── SAM3 API (parallel max) ──")
        for label, field in [
            ("preprocess", "sam3_pre"),
            ("encode (→JPEG)", "sam3_enc"),
            ("http (SAM3 call)", "sam3_http"),
            ("mask_decode", "sam3_mask"),
            ("sam3_subtotal", "sam3_total"),
        ]:
            self._stat_line(lines, label, _max_field(field))

        # CDM sub-stages (if available)
        cdm_fields = [("cdm_pre", "cdm_pre"), ("cdm_enc", "cdm_enc"),
                      ("cdm_http", "cdm_http"), ("cdm_post", "cdm_post"),
                      ("cdm_total", "cdm_total")]
        # 直接检查第一个 cycle 是否有 CDM 数据 (cycles[0] is a dict with "top"/"chassis" PerfEntry)
        has_cdm = False
        for _, field in cdm_fields:
            for cam in ("top", "chassis"):
                val = getattr(cycles[0][cam], field, None)
                if val is not None and val > 0:
                    has_cdm = True
                    break
            if has_cdm:
                break
        if has_cdm:
            lines.append("  ── CDM API (parallel max) ──")
            for label, field in cdm_fields:
                self._stat_line(lines, label, _max_field(field))

        # Parallel sync
        lines.append("  ── Parallel Sync ──")
        self._stat_line(lines, "parallel_total", _max_field("parallel"))
        self._stat_line(lines, "idle (sync wait)", _max_field("idle"))

        # 3D post-processing (parallel max)
        lines.append("  ── 3D Post-Processing (parallel max) ──")
        for label, field in [
            ("depth_publish", "post_dpub"),
            ("3d_computation", "post_3d"),
            ("tf_transform", "post_tf"),
            ("msg_build", "post_build"),
            ("post_subtotal", "post_total"),
        ]:
            self._stat_line(lines, label, _max_field(field))

        # Overall wall time
        lines.append("  ── Detection Wall Time ──")
        self._stat_line(lines, "top_wall", _field("top", "wall"))
        self._stat_line(lines, "chassis_wall", _field("chassis", "wall"))
        wall_max = np.array([c["wall_max"] for c in cycles], dtype=float)
        self._stat_line(lines, "parallel_wall (max)", wall_max)

        # Bottleneck analysis
        n_top = sum(1 for c in cycles if c["bottleneck"] == "top")
        n_chs = len(cycles) - n_top
        pct_top = 100 * n_top / len(cycles)
        pct_chs = 100 * n_chs / len(cycles)
        lines.append(f"  bottleneck: top={pct_top:.0f}% | chassis={pct_chs:.0f}%")

    def _report_fusions(self, fusions: list[FusionEntry], lines: list[str]):
        if not fusions:
            lines.append("  (no fusion data)")
            return
        HDR = f"  {'':36s} {'n':>5s}  {'min':>6s}  {'mean':>6s}  {'p50':>6s}  {'p95':>6s}  {'max':>6s}"
        lines.append(HDR)
        self._stat_line(lines, "hungarian_match",
                        np.array([f.match_ms for f in fusions], dtype=float))
        self._stat_line(lines, "byte_tracker_3d",
                        np.array([f.track_ms for f in fusions], dtype=float))
        self._stat_line(lines, "fusion_total",
                        np.array([f.total_ms for f in fusions], dtype=float))

    def _report_workers(self, workers: list[WorkerEntry], lines: list[str]):
        if not workers:
            lines.append("  (no worker data)")
            return
        HDR = f"  {'':36s} {'n':>5s}  {'min':>6s}  {'mean':>6s}  {'p50':>6s}  {'p95':>6s}  {'max':>6s}"
        lines.append(HDR)
        for label, field in [
            ("queue_wait", "qwait"),
            ("detect_total", "detect"),
            ("publish", "pub"),
            ("fusion_in_worker", "fusion"),
            ("worker_total", "total"),
        ]:
            arr = np.array([getattr(w, field) for w in workers], dtype=float)
            self._stat_line(lines, label, arr)

    def _compute_rviz_latency(self) -> np.ndarray:
        """Wall-clock latency from fused topic arrival to rviz marker arrival."""
        results = []
        fi = 0
        for rv_t in self.rviz_walls:
            # find nearest fused wall before rviz wall
            best = None
            for j in range(fi, len(self.fused_walls)):
                diff = rv_t - self.fused_walls[j]
                if diff < 0:
                    break
                if diff < 0.5:
                    best = diff
                    fi = j
            if best is not None:
                results.append(best * 1000.0)  # already in ms
        return np.array(results, dtype=float) if results else np.array([], dtype=float)

    @staticmethod
    def _stat_line(lines: list[str], label: str, arr: np.ndarray):
        if len(arr) == 0:
            lines.append(f"  {label:36s}     0     —       —       —       —       —")
            return
        n = len(arr)
        lines.append(
            f"  {label:36s} {n:>5d}  {arr.min():6.1f}  {arr.mean():6.1f}  "
            f"{np.percentile(arr, 50):6.1f}  {np.percentile(arr, 95):6.1f}  {arr.max():6.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Perception pipeline end-to-end latency benchmark"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Collection duration in seconds (default: 30)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = PipelineBenchmark(duration=args.duration)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
