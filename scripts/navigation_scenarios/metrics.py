#!/usr/bin/env python3
"""Generate per-run logsim metrics from saved results (no ROS required)."""
import argparse
import csv
import json
import math
from pathlib import Path


def motion_metrics(result, perception, samples):
    """Physical quantities, without arbitrary weighted scores or safety thresholds."""
    values = dict.fromkeys([
        "progress_percent", "progress_initial_goal_distance_m", "progress_remaining_goal_distance_m",
        "safety_min_clearance_m", "safety_collision_detected",
        "comfort_linear_accel_rms_mps2", "comfort_angular_accel_rms_radps2",
        "comfort_linear_accel_peak_mps2", "comfort_angular_accel_peak_radps2",
    ])
    values["comfort_valid_intervals"] = 0
    values["comfort_skipped_intervals"] = 0
    goal, start = result.get("goal"), perception.get("start")
    if samples and goal and start:
        end = result.get("final_pose")
        if end is None and all(k in samples[-1] for k in ("x", "y")):
            end = [samples[-1]["x"], samples[-1]["y"]]
        if end is not None:
            initial = math.hypot(start[0] - goal[0], start[1] - goal[1])
            remaining = math.hypot(end[0] - goal[0], end[1] - goal[1])
            values.update(progress_initial_goal_distance_m=initial, progress_remaining_goal_distance_m=remaining)
            if initial > 1e-6:
                values["progress_percent"] = 100 * (1 - remaining / initial)
    clearance = result.get("minimum_approximate_obstacle_clearance_m")
    if clearance is not None:
        values["safety_min_clearance_m"] = clearance
        values["safety_collision_detected"] = clearance <= 0
    durations, linear, angular = [], [], []
    for left, right in zip(samples, samples[1:]):
        if not all(k in row and math.isfinite(row[k]) for row in (left, right) for k in ("time", "v", "w")):
            values["comfort_skipped_intervals"] += 1
            continue
        dt = right["time"] - left["time"]
        if dt <= 0 or dt > 0.2:
            values["comfort_skipped_intervals"] += 1
            continue
        durations.append(dt)
        linear.append((right["v"] - left["v"]) / dt)
        angular.append((right["w"] - left["w"]) / dt)
    values["comfort_valid_intervals"] = len(durations)
    if durations:
        for kind, unit, accelerations in [("linear", "mps2", linear), ("angular", "radps2", angular)]:
            values[f"comfort_{kind}_accel_rms_{unit}"] = math.sqrt(sum(a*a*dt for a, dt in zip(accelerations, durations)) / sum(durations))
            values[f"comfort_{kind}_accel_peak_{unit}"] = max(abs(a) for a in accelerations)
    return values


def simulation_status(result, perception):
    """Separate expected algorithm outcomes from execution/data failures."""
    if result.get("simulation_status"):
        return result["simulation_status"]
    if result.get("failure_stage") in ("startup", "infrastructure"):
        return "failed"
    if result.get("status") in ("navigation_succeeded", "controller_failed", "TIMEOUT", "APPROXIMATE_FOOTPRINT_COLLISION"):
        return "completed"
    if result.get("error") in ("No reachable stopping pose", "Controller rejected path"):
        return "completed"
    if not result and perception.get("error") == "No matching object detected":
        return "completed"
    return "failed"


def build_metrics(result, perception, samples):
    candidates = result.get("candidates", [])
    succeeded = result.get("status") == "navigation_succeeded"
    stage = result.get("failure_stage")
    if not succeeded and not stage:
        if perception.get("status") != "prepared":
            stage = "perception"
        elif result.get("error") == "No reachable stopping pose":
            stage = "planning"
        elif result.get("action_status") is not None or samples:
            stage = "control"
        else:
            stage = "unknown"
    m = {
        "schema_version": 4,
        "simulation_status": simulation_status(result, perception),
        "algorithm_success": succeeded,
        "status": result.get("status", "not_run"),
        "success": succeeded,
        "failure_stage": None if succeeded else stage,
        "failure_reason": None if succeeded else result.get("error", perception.get("error", result.get("status"))),
        "perception_success": perception.get("status") == "prepared",
        "detected_object_count": len(perception.get("objects", [])),
        "detection_seconds": perception.get("detection_seconds"),
        "planning_candidate_count": len(candidates),
        "planning_success_count": sum(c.get("status") == 4 and c.get("poses", 0) > 0 for c in candidates),
        "planning_seconds": result.get("planning_seconds"),
        "navigation_started": bool(samples) or result.get("navigation_started", False),
        "navigation_duration_seconds": result.get("duration_seconds"),
        "simulation_wall_seconds": result.get("simulation_wall_seconds"),
        "distance_m": result.get("distance_m"),
        "planned_length_m": result.get("planned_length_m"),
        "goal_position_error_m": result.get("goal_position_error_m"),
        "goal_yaw_error_deg": math.degrees(result["goal_yaw_error_rad"]) if result.get("goal_yaw_error_rad") is not None else None,
        "reference_nearest_rmse_m": result.get("reference_nearest_rmse_m"),
        "reference_max_distance_m": result.get("reference_max_distance_m"),
        "reference_endpoint_error_m": result.get("reference_endpoint_error_m"),
        "minimum_approximate_obstacle_clearance_m": result.get("minimum_approximate_obstacle_clearance_m"),
        "untransformable_reference_scans": result.get("untransformable_reference_scans"),
        "max_scan_reference_distance_m": result.get("max_scan_reference_distance_m"),
        "max_scan_reference_yaw_deg": math.degrees(result["max_scan_reference_yaw_rad"]) if result.get("max_scan_reference_yaw_rad") is not None else None,
        "peak_abs_linear_speed_mps": max((abs(s["v"]) for s in samples), default=None),
        "peak_abs_angular_speed_radps": max((abs(s["w"]) for s in samples), default=None),
        "trajectory_sample_count": len(samples),
    }
    m.update(motion_metrics(result, perception, samples))
    planned, travelled = result.get("planned_length_m"), result.get("distance_m")
    m["progress_reached_goal"] = succeeded
    m["progress_path_efficiency_percent"] = (
        100 * planned / travelled
        if succeeded and planned is not None and travelled is not None
        and math.isfinite(planned) and math.isfinite(travelled)
        and planned > 0 and travelled > 1e-6 else None
    )
    m["progress_time_to_goal_seconds"] = result.get("duration_seconds") if succeeded else None
    m["progress_elapsed_seconds"] = result.get("duration_seconds")
    m["notes"] = [
        "Path efficiency = planned global path length / actual distance * 100, only for successful navigation. May exceed 100 when the controller takes a shorter path. Time to goal is control execution time, excluding detection and planning.",
        "Progress = 100*(1-final Euclidean goal distance/initial distance); negative means moving away. Not path completion; yaw is evaluated separately.",
        "Safety clearance is rectangle-footprint to observed/map points. Collision false means none detected in evaluated samples, not certified safe.",
        "Comfort uses time-weighted RMS finite differences of recorded v/w. Lower means smoother commands, not measured ride comfort. Intervals <=0 or >0.2s excluded; terminal stop beyond the recorded trajectory is not included.",
        "null means not evaluated or not recorded; it does not mean zero error.",
        "Reference errors use spatial nearest points, not time alignment or ground truth.",
        "Goal error is relative to the selected stopping pose, not the object center.",
        "Clearance is an approximate recorded-scan/map metric, not a real-world safety guarantee.",
        "Simulation wall time excludes perception; detection time covers the VLM request only.",
    ]
    return m


LABELS = {
    "simulation_status": "仿真执行状态",
    "algorithm_success": "算法任务成功",
    "reference_recording_duration_seconds": "参考 bag 录制时长（秒，包含停顿）",
    "progress_reached_goal": "Progress：已到达目标",
    "progress_path_efficiency_percent": "Progress：路径效率（%）",
    "progress_time_to_goal_seconds": "Progress：到达耗时（秒，仅控制执行）",
    "progress_elapsed_seconds": "Progress：已用控制执行时间（秒）",
    "progress_percent": "Progress：目标接近率（%）",
    "progress_initial_goal_distance_m": "初始目标距离（米）",
    "progress_remaining_goal_distance_m": "剩余目标距离（米）",
    "safety_min_clearance_m": "Safety：近似最小车体净距（米）",
    "safety_collision_detected": "近似碰撞检测触发",
    "comfort_linear_accel_rms_mps2": "Comfort：线加速度 RMS（m/s²）",
    "comfort_angular_accel_rms_radps2": "Comfort：角加速度 RMS（rad/s²）",
    "comfort_linear_accel_peak_mps2": "线加速度峰值（m/s²）",
    "comfort_angular_accel_peak_radps2": "角加速度峰值（rad/s²）",
    "comfort_valid_intervals": "平顺性有效采样区间数",
    "comfort_skipped_intervals": "平顺性跳过异常区间数",
    "success": "导航成功", "failure_stage": "失败阶段", "failure_reason": "失败原因",
    "perception_success": "目标检测与定位成功", "detected_object_count": "定位目标数",
    "detection_seconds": "检测耗时（秒）", "planning_candidate_count": "规划候选数",
    "planning_success_count": "规划成功候选数", "planning_seconds": "规划耗时（秒）",
    "navigation_started": "已开始控制", "navigation_duration_seconds": "运动时长（秒）",
    "simulation_wall_seconds": "仿真程序耗时（秒，不含感知）", "distance_m": "行驶距离（米）",
    "planned_length_m": "全局路径长度（米）", "goal_position_error_m": "停靠点位置误差（米）",
    "goal_yaw_error_deg": "停靠朝向误差（度）", "reference_nearest_rmse_m": "参考轨迹最近点 RMSE（米）",
    "reference_max_distance_m": "参考轨迹最大偏差（米）", "reference_endpoint_error_m": "参考终点偏差（米）",
    "minimum_approximate_obstacle_clearance_m": "近似最小障碍间距（米）",
    "untransformable_reference_scans": "缺少 TF 的参考激光帧数",
    "max_scan_reference_distance_m": "匹配参考扫描最大位置偏差（米）",
    "max_scan_reference_yaw_deg": "匹配参考扫描最大朝向偏差（度）",
    "peak_abs_linear_speed_mps": "最大线速度绝对值（米/秒）",
    "peak_abs_angular_speed_radps": "最大角速度绝对值（弧度/秒）",
    "trajectory_sample_count": "运动轨迹样本数",
}


def write_metrics(output):
    output = Path(output)
    result_file = output / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    perception = json.loads((output / "perception.json").read_text())
    trajectory = output / "trajectory.csv"
    samples = []
    if trajectory.exists():
        with trajectory.open() as stream:
            samples = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    m = build_metrics(result, perception, samples)
    m["reference_recording_duration_seconds"] = None
    if perception.get("scene"):
        scene_file = Path(perception["scene"]) / "scenario.json"
        if scene_file.is_file():
            scene = json.loads(scene_file.read_text())
            m["reference_recording_duration_seconds"] = scene.get("elapsed_wall_seconds")
    m.update(run_id=output.parent.name if output.parent.name.startswith("run_") else output.name,
             scene=perception.get("scene"))
    (output / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = [f"# Logsim metrics: {output.name}", "", "| 指标 | 结果 |", "|---|---|"]
    for key, label in LABELS.items():
        value = m[key]
        if value is None:
            value = "—（不适用或未记录）"
        elif isinstance(value, bool):
            value = "是" if value else "否"
        elif isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {label} | {str(value).replace('|', '/').replace(chr(10), ' ')} |")
    lines += ["", "参考轨迹偏差按空间最近点计算，并非时间对齐误差或真实定位误差。",
              "停靠点误差相对于检测生成的导航目标；参考终点仅供比较。",
              "障碍间距来自近似仿真，不代表实机安全验证。历史运行未记录的耗时不补造。", ""]
    (output / "metrics.md").write_text("\n".join(lines))
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", type=Path)
    for output in parser.parse_args().outputs:
        metrics = write_metrics(output)
        print(output / "metrics.json", metrics["status"])
