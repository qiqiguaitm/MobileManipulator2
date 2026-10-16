"""Publish compact run tables with a separate details page for each scenario."""
import html
import json
from pathlib import Path

from metrics import LABELS
from ui_theme import THEME, FAVICON
from catalog import artifact_url, foxglove_url, run_info

CSS = """
.overview{min-width:1100px}
.name{min-width:190px;max-width:290px;overflow-wrap:anywhere}.name small{display:block;margin-top:4px}
.overview th,.overview td{text-align:center;vertical-align:middle}.overview th:first-child,.overview td:first-child{text-align:left}
.num{white-space:nowrap}.sub{display:block;font-size:12px;color:var(--muted);margin-top:4px}.overview th small{font-weight:400}
.replay{white-space:nowrap;line-height:1.9;font-size:12px}
.links{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.button{background:#eaf0fc;padding:6px 10px;border-radius:6px;font-size:12px}
.detail td:last-child{overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.detail table th:first-child{width:50%}
.breadcrumb{font-size:12px;margin-bottom:18px}.metric-notes{margin:18px 0;color:var(--muted);font-size:12px;max-width:1000px}
.metric-notes summary{cursor:pointer;color:var(--blue)}
"""



def fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def page(title, body):
    return f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>{FAVICON}<style>{THEME}{CSS}</style></head><body><header class="topbar"><a class="brand" href="/"><span class="brand-icon">L</span>Logsim <span class="top-note">/ 导航仿真</span></a><span class="top-note">所有时间均为北京时间 · UTC+8</span></header><main>{body}</main></body></html>'


def report(run, root, config, base_url="http://127.0.0.1:18770"):
    run, root = Path(run), Path(root)
    display_time, run_desc = run_info(run)
    entries, table_rows = [], []
    for scene in config["scenes"]:
        output = run / scene["id"]
        metric_file = output / "metrics.json"
        if not metric_file.exists():
            continue
        m = json.loads(metric_file.read_text())
        replay = []
        for label, folder in [
            ("原始 bag", root / scene["scene_dir"] / "visualization_bag"),
            ("仿真 bag", output / "result_bag"),
        ]:
            files = sorted(folder.glob("*.mcap"))
            if len(files) == 1:
                url = foxglove_url(files[0], root, base_url)
                replay.append(f'<a target="_blank" rel="noopener" href="{html.escape(url)}">{label} ↗</a>')
            else:
                replay.append(f'<span class="muted">{label}：暂无</span>')
        links = []
        for filename, label in [("metrics.json", "完整 metrics"), ("result.json", "运行结果"), ("perception.json", "检测结果"), ("detections.jpg", "检测图像"), ("trajectory.csv", "轨迹 CSV"), ("test_params.yaml", "导航参数"), ("planner_server.log", "规划日志"), ("controller_server.log", "控制日志"), ("prepare_test.py.log", "检测日志"), ("test_navigation.py.log", "仿真日志")]:
            artifact = output / filename
            if artifact.is_file():
                links.append(f'<a class="button" href="{artifact_url(artifact, root)}">{label}</a>')
        name = html.escape(scene.get("name", scene["id"]))
        scene_id = html.escape(scene["id"])
        success = m.get("success", False)
        status = f'<span class="badge {"good" if success else "bad"}">{"成功" if success else "失败"}</span>'
        stage = {"planning": "全局规划", "perception": "感知", "control": "控制", "startup": "启动", "infrastructure": "运行环境"}.get(m.get("failure_stage"), m.get("failure_stage"))
        sim_completed = m.get('simulation_status') == 'completed'
        result_cell = '<span class="badge good">完成</span>' if sim_completed else '<span class="badge bad">异常</span>'
        algorithm_cell = status + (f'<span class="sub">{fmt(stage)}</span>' if stage else '')
        candidates = f'{fmt(m.get("planning_success_count"))}/{fmt(m.get("planning_candidate_count"))}'
        error = m.get('goal_position_error_m')
        error_cm = error * 100 if error is not None else None
        details_path = output / 'details.html'
        approach = m.get('progress_percent')
        approach_text = fmt(approach, 1) + ('%' if approach is not None else '')
        efficiency = m.get('progress_path_efficiency_percent')
        efficiency_text = fmt(efficiency, 1) + ('%' if efficiency is not None else '')
        sim_time = fmt(m.get('progress_elapsed_seconds'))
        reference_time = fmt(m.get('reference_recording_duration_seconds'))
        progress = f'<strong>接近率 {approach_text}</strong><span class="sub">路径效率 {efficiency_text}</span><span class="sub" title="仿真控制执行时间 / 参考完整录制时长（包含停顿）；终点可能不同，不作为速度效率比">仿真 / 参考：{sim_time} / {reference_time} s</span>'
        clearance = m.get('safety_min_clearance_m')
        collision = m.get('safety_collision_detected')
        safety_note = '未评测' if collision is None else ('检测到碰撞' if collision else '未检测到碰撞')
        safety = fmt(clearance, 3) + (' m' if clearance is not None else '')
        comfort_a, comfort_w = m.get('comfort_linear_accel_rms_mps2'), m.get('comfort_angular_accel_rms_radps2')
        comfort = fmt(comfort_a, 3) + ' / ' + fmt(comfort_w, 3)
        table_rows.append(f'<tr class="scenario-row"><td class="name"><strong>{name}</strong><small class="muted">{scene_id}</small></td><td>{result_cell}</td><td>{algorithm_cell}</td><td class="num">{progress}</td><td class="num">{safety}<span class="sub">{safety_note}</span></td><td class="num">{comfort}</td><td class="replay">{"<br>".join(replay)}</td><td><a href="{artifact_url(details_path, root)}">Details →</a></td></tr>')
        metric_rows = ''.join(f'<tr><td>{html.escape(label)}</td><td>{fmt(m.get(key), 4)}</td></tr>' for key, label in LABELS.items())
        description = html.escape(scene.get('description') or scene.get('task_description') or '')
        detail_body = f'<div class="detail"><a href="{artifact_url(run / "report.html", root)}">← 返回本次报告</a><h1>{name}</h1><div class="links"><span>算法结果 {status}</span><span>仿真状态 {result_cell}</span></div><p class="muted">{scene_id} / {html.escape(display_time)}（北京时间）</p><p>{description}</p><div class="links">{" · ".join(replay)}</div><div class="links">{"".join(links)}</div><details class="metric-notes"><summary>指标说明与计算方法</summary><p>Progress 展示接近率、路径效率和仿真 / 参考耗时。仿真为控制执行时间（不含检测、规划）；参考为完整录制时长，包含停顿、终点可能不同，不换算效率比例。路径效率 = 全局规划路径长度 / 实际行驶距离 × 100%，只对成功到达计算；局部优化走了更短路线时可超过 100%。接近率 = (1 − 剩余目标直线距离 / 初始目标直线距离) × 100%，不是路径完成比例，也不包含朝向；负数表示离目标更远。Safety 是车体矩形与记录激光/地图障碍点的近似最小间距，并非实机安全结论。Comfort 是记录速度差分的时间加权加速度 RMS，越小表示指令越平顺；不是实测乘坐舒适度。跳过非递增时间和超过 0.2 秒的区间，不包含轨迹记录结束后的停车变化。</p></details><div class="scroll"><table><thead><tr><th>指标</th><th>结果</th></tr></thead><tbody>{metric_rows}</tbody></table></div></div>'
        details_path.write_text(page(f'{scene["id"]} · Details', detail_body))
        entries.append({"scene_id": scene["id"], "result_dir": scene["id"], "metrics": m})
    total = len(entries)
    success = sum(e["metrics"].get("success", False) for e in entries)
    completed = sum(e["metrics"].get("simulation_status") == "completed" for e in entries)
    summary = {"simulation_completed_count": completed, "simulation_failed_count": total - completed, "algorithm_success_count": success, "run_id": run.name, "desc": run_desc, "started_beijing": display_time, "scene_count": total, "success_count": success,
               "success_rate": success / total if total else None, "results": entries}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    rate = f'{success/total:.0%}' if total else '—'
    body = f'<div class="breadcrumb"><a href="/">← 返回首页</a></div><div class="page-head"><div><h1>仿真测试报告</h1><p>{html.escape(display_time)} · 北京时间</p><p>{html.escape(run_desc or "未填写仿真说明")}</p></div><div class="stats"><div class="stat"><strong>{total}</strong><span>测试场景</span></div><div class="stat"><strong>{completed}/{total}</strong><span>仿真完成 · 异常 {total-completed}</span></div><div class="stat"><strong>{success}/{total}</strong><span>算法成功 · {rate}</span></div></div></div><section><div class="section-head"><h2>场景结果</h2><span class="count">{total}</span><p>查看关键指标与回放，完整信息见 Details</p></div>'

    if total:
        body += '<div class="scroll"><table class="overview"><thead><tr><th>场景</th><th>仿真状态</th><th>算法结果</th><th>Progress<br><small>接近率 · 效率 · 耗时</small></th><th>Safety<br><small>近似最小车体净距 ↑</small></th><th>Comfort<br><small>线 / 角加速度 RMS ↓<br>m/s² / rad/s²</small></th><th>Foxglove 回放</th><th>详情</th></tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div>'
    else:
        body += '<p>运行中，尚无完成场景。</p>'
    body += '</section><footer>— 表示未评测或未记录。Progress 显示接近率、路径效率及仿真 / 参考耗时。参考耗时为完整录制时长，包含停顿，且终点可能不同，不换算效率比例；Safety 为近似净距；Comfort 为指令变化的 RMS，不能作为实机安全/舒适性结论。停靠误差相对于导航目标。<br>Details 中包含完整指标、检测结果与图像、导航参数和日志。回放需要保持 SSH 端口转发。</footer>'
    (run / 'report.html').write_text(page('Logsim · ' + run.name, body))
    return summary
