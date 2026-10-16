"""Read-only data catalog and confined artifact URLs."""
import html
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote
import yaml
from ui_theme import THEME, FAVICON


def resolve_artifact(root, name):
    parts = name.split('/')
    if not name or any(p in ('', '.', '..') for p in parts) or '\\' in name:
        return None
    if parts[0] not in ('scenes', 'pending', 'simulation_runs'):
        return None
    if Path(name).suffix not in ('.mcap', '.html', '.json', '.md', '.yaml', '.csv', '.log', '.jpg', '.png'):
        return None
    try:
        target = (root / name).resolve(strict=True)
        target.relative_to(root.resolve())
        relative = target.relative_to(root.resolve())
        if relative.parts[0] not in ("scenes", "pending", "simulation_runs") or target.suffix != Path(name).suffix:
            return None
        if target.is_file():
            return target
    except (OSError, ValueError, RuntimeError):
        pass
    return None


def artifact_url(path, root):
    return '/' + quote(Path(path).relative_to(root).as_posix(), safe='/')


def foxglove_url(path, root, base):
    return 'https://app.foxglove.dev/~/view?ds=remote-file&ds.url=' + quote(base.rstrip('/') + artifact_url(path, root), safe='')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def run_info(run):
    metadata = read_json(run / 'run.json')
    stamp = metadata.get('started_utc')
    try:
        if stamp:
            dt = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            raw = run.name.removeprefix('run_')
            fmt = '%Y%m%dT%H%M%S_%fZ' if '_' in raw else '%Y%m%dT%H%M%SZ'
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        display = dt.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        display = run.name
    return display, str(metadata.get('desc') or '')


def anchor(url, title):
    return f'<a href="{html.escape(url)}">{html.escape(title)}</a>'


def catalog(root, manifest, base):
    config = yaml.safe_load(manifest.read_text())
    registered = {(root / s['scene_dir']).resolve(): s for s in config.get('scenes', [])}
    rows = []
    for run in sorted((root / 'simulation_runs').glob('run_*'), reverse=True):
        if not (run / 'report.html').is_file():
            continue
        summary = read_json(run / 'summary.json')
        count = summary.get('scene_count', 0)
        successes = summary.get('simulation_completed_count', 0)
        display, desc = run_info(run)
        report_url = artifact_url(run / 'report.html', root)
        completion = 'good' if count and successes == count else 'warn'
        algorithm_success = summary.get('algorithm_success_count', summary.get('success_count', 0))
        algorithm_class = 'good' if count and algorithm_success == count else 'bad'
        algorithm_text = f'{algorithm_success}/{count} 达成' + (f' · {count-algorithm_success} 未达成' if count > algorithm_success else '')
        rows.append('<tr><td class="report-time">' + anchor(report_url, display) + '</td><td>' + (html.escape(desc) if desc else '<span class="muted">未填写说明</span>') + f'</td><td class="num">{count}</td><td class="num"><span class="badge {completion}">{successes} / {count}</span></td><td class="num"><span class="badge {algorithm_class}">{algorithm_text}</span></td></tr>')
    runs = '<div class="table-scroll"><table><thead><tr><th>运行时间 · 北京时间</th><th>仿真说明</th><th class="num">场景数</th><th class="num">仿真完成</th><th class="num">算法结果</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>' if rows else '<p class="empty">暂无仿真运行。完成测试后，报告会显示在这里。</p>'


    def scenes(entries):
        cards = []
        for path, label in entries:
            info = read_json(path / 'scenario.json')
            verification = read_json(path / 'verification.json')
            entry = registered.get(path.resolve(), {})
            if path.parent.name == 'pending':
                badge, color = '待入集', '#8a631a'
            elif not entry:
                badge, color = '未登记', '#657389'
            elif entry.get('enabled', False):
                badge, color = '已启用', '#16734b'
            else:
                badge, color = '未启用', '#8a631a'
            description = entry.get('description') or entry.get('task_description') or info.get('task_description') or info.get('description') or '暂无描述'
            stamp = info.get('started_utc')
            try:
                stamp = datetime.fromisoformat(stamp.replace('Z', '+00:00')).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
            except (AttributeError, ValueError):
                stamp = '未记录'
            duration = entry.get('duration_seconds', info.get('elapsed_wall_seconds'))
            distance = info.get('estimated_travel_m', verification.get('estimated_travel_m'))
            duration_text = f'{duration:.2f} 秒' if isinstance(duration, (int, float)) else '未记录'
            distance_text = f'{distance:.2f} 米' if isinstance(distance, (int, float)) else '未记录'
            mode = entry.get('capture_mode', info.get('capture_mode', '未记录'))
            mode = {'teleoperation': '人工遥控', 'passive_subscription': '被动采集'}.get(mode, mode)
            links = []
            bags = sorted((path / 'visualization_bag').glob('*.mcap'))
            if bags:
                links.append(' · '.join(anchor(foxglove_url(bag, root, base), '原始 bag') for bag in bags))
            else:
                links.append('<span class="muted">原始 bag（未生成）</span>')
            for filename, title in [('scenario.json', '场景详情'), ('verification.json', '数据校验')]:
                if (path / filename).is_file():
                    links.append(anchor(artifact_url(path / filename, root), title))
                else:
                    links.append('<span class="muted">' + title + '（未生成）</span>')
            name = entry.get('name') or info.get('name') or label
            name_cell = html.escape(str(name))
            if name != path.name:
                name_cell += '<br><small class="muted">' + html.escape(path.name) + '</small>'
            status = verification.get('status', '未校验')
            status = {'passed_data_checks': '通过'}.get(status, status)
            badge_class = 'good' if badge == '已启用' else ('warn' if badge in ('待入集', '未启用') else 'neutral')
            cards.append('<tr><td class="scene-name">' + name_cell + '</td><td><span class="badge ' + badge_class + '">' + badge + '</span></td><td class="description">' + html.escape(str(description)) + '</td><td>' + html.escape(str(stamp)) + '<br><small class="muted">' + html.escape(str(mode)) + '</small></td><td>' + duration_text + '<br>' + distance_text + '</td><td>' + html.escape(str(status)) + '</td><td class="actions">' + '<br>'.join(links) + '</td></tr>')
        return '<div class="table-scroll"><table class="scene-table"><thead><tr><th>场景</th><th>启用状态</th><th>描述</th><th>录制时间 / 方式</th><th>时长 / 距离</th><th>校验</th><th>查看</th></tr></thead><tbody>' + ''.join(cards) + '</tbody></table></div>' if cards else '<p class="empty">暂无场景。</p>'

    pending = [(p, p.name) for p in sorted((root / 'pending').glob('*'), reverse=True) if p.is_dir()]
    enrolled = [(p, p.name) for p in sorted((root / 'scenes').glob('*'), reverse=True) if p.is_dir()]
    enabled_count = sum(bool(registered.get(p.resolve(), {}).get('enabled')) for p, _ in enrolled)
    heading = f'<header class="topbar"><div class="brand"><span class="brand-icon">L</span>Logsim <span class="top-note">/ 导航仿真</span></div><span class="top-note">所有时间均为北京时间 · UTC+8</span></header><main><div class="page-head"><div><h1>测试与场景</h1><p>查看每次运行，管理参考场景。</p></div><div class="stats"><div class="stat"><strong>{len(rows)}</strong><span>仿真运行</span></div><div class="stat"><strong>{len(enrolled)}</strong><span>场景 · {enabled_count} 个启用</span></div><div class="stat"><strong>{len(pending)}</strong><span>待入集</span></div></div></div>'
    toolbar = f'<div class="toolbar"><nav aria-label="页面导航"><a href="#runs">仿真报告<small>{len(rows)}</small></a><a href="#pending">待入集<small>{len(pending)}</small></a><a href="#scenes">场景库<small>{len(enrolled)}</small></a></nav><input class="search" type="search" aria-label="搜索报告或场景" placeholder="搜索时间、场景或描述…"></div>'
    sections = f'<section id="runs"><div class="section-head"><h2>历次仿真报告</h2><span class="count">{len(rows)}</span><p>按时间倒序排列</p></div>{runs}</section><section id="pending"><div class="section-head"><h2>待入集场景</h2><span class="count">{len(pending)}</span></div>{scenes(pending)}</section><section id="scenes"><div class="section-head"><h2>场景库</h2><span class="count">{len(enrolled)}</span><p>启用状态由测试集清单维护</p></div>{scenes(enrolled)}</section>'
    script = """<script>document.querySelector('.search').addEventListener('input',function(){const q=this.value.trim().toLocaleLowerCase();document.querySelectorAll('section').forEach(section=>{let visible=0;const rows=section.querySelectorAll('tbody tr');rows.forEach(row=>{row.hidden=!row.textContent.toLocaleLowerCase().includes(q);if(!row.hidden)visible++;});let note=section.querySelector('.filter-empty');if(!note){note=document.createElement('p');note.className='filter-empty';note.textContent='没有匹配的记录';section.appendChild(note);}note.hidden=!q||visible>0||rows.length===0;});});</script>"""
    return '<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Logsim · 测试与场景</title>' + FAVICON + '<style>' + THEME + '</style></head><body>' + heading + toolbar + sections + '<footer>报告与场景数据保存在机器人上 · 刷新页面获取最新列表</footer></main>' + script + '</body></html>'
