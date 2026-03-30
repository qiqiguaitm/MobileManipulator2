#!/usr/bin/env python3
"""
Track稳定性调试工具
订阅 /multi_camera_perception/fused/objects_3d，检测以下异常：
  DOUBLE_CALL    - 同tick两次publish（间隔 < 25ms），对比两次的source差异
  SOURCE_JUMP    - 同一track_id的source在 top ↔ fused 之间反复切换
  POSITION_JUMP  - 同一track_id单帧位置突变 > 阈值（部分遮挡深度污染）
  TRACK_LOST     - track_id 从输出消失
  TRACK_BORN     - track_id 首次出现
  ID_CHANGE      - 新track_id出现在已消失track的近邻（同一物体被重新分配ID）
  SNAPSHOT       - 每条消息的完整track状态（用于事后时间线重建）

--cost 模式额外输出（需感知节点 cost_debug=True）：
  MATCH_COST     - 每次匹配的代价分解: total cost, dist/IoU/cat 各项贡献

用法:
  python3 scripts/_cc_track_debug.py [--log FILE] [--target CATEGORY] [--cost]
"""

import argparse
import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from perception.msg import Object3DArray
from rcl_interfaces.msg import Log as RosoutLog

# ─── 阈值 ────────────────────────────────────────────────────────────────────
TOPIC                 = '/multi_camera_perception/fused/objects_3d'
DOUBLE_CALL_MS        = 25.0   # 两条消息间隔 < 此值 = 同tick double-call
ID_CHANGE_DIST_M      = 0.30   # 判定同一物体的距离阈值
ID_CHANGE_WINDOW_S    = 10.0   # dead track保留时间窗口
POSITION_JUMP_M       = 0.15   # 单帧位置突变阈值
SOURCE_JUMP_WINDOW    = 8      # 检测source跳变的滑动窗口大小
SOURCE_JUMP_MIN_TRANS = 2      # 滑动窗口内切换次数 ≥ 此值 = 跳变
SOURCE_JUMP_COOLDOWN  = 2.0    # 同一track两次SOURCE_JUMP报告的最小间隔(s)

# cost监控阈值
COST_WARN_THRESH      = 0.60   # cost > 此值时标记为WARN（接近gate=0.75）
COST_LOG_PREFIX       = '[COST]'  # byte_tracker_3d.py 打出的日志前缀

# ANSI
RED = '\033[91m'; YELLOW = '\033[93m'; GREEN = '\033[92m'
CYAN = '\033[96m'; RESET = '\033[0m'; BOLD = '\033[1m'; DIM = '\033[2m'


def dist3d(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def pos_of(obj) -> tuple:
    return (obj.position.x, obj.position.y, obj.position.z)


def fmt_pos(p) -> str:
    return f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"


class TrackEntry:
    def __init__(self, track_id: str, born_ts: float, category: str):
        self.track_id = track_id
        self.category = category
        self.born_ts = born_ts
        self.last_ts: float = born_ts
        self.last_pos: tuple = None
        self.last_source: str = None
        self.source_seq = deque(maxlen=SOURCE_JUMP_WINDOW)
        self.source_jump_last_report: float = 0.0

    def update(self, ts: float, source: str, pos: tuple):
        self.last_ts = ts
        self.last_pos = pos
        self.last_source = source
        self.source_seq.append((ts, source))

    def check_source_jump(self, now: float):
        """返回 (should_report, transition_count, recent_sources_list)"""
        if len(self.source_seq) < SOURCE_JUMP_WINDOW:
            return False, 0, []
        sources = [s for _, s in self.source_seq]
        transitions = sum(1 for i in range(1, len(sources)) if sources[i] != sources[i - 1])
        if transitions >= SOURCE_JUMP_MIN_TRANS:
            cooldown_ok = (now - self.source_jump_last_report) >= SOURCE_JUMP_COOLDOWN
            return cooldown_ok, transitions, sources
        return False, transitions, sources


class TrackDebugNode(Node):
    def __init__(self, log_file: str, target_category: str = None,
                 cost_mode: bool = False):
        super().__init__('track_debug')
        self.log_file = log_file
        self.target = target_category
        self.cost_mode = cost_mode

        self._lock = threading.Lock()
        self._active: dict[str, TrackEntry] = {}
        self._dead: dict[str, TrackEntry] = {}

        self._prev_msg_ts: float = None
        self._prev_snapshot: dict = {}  # track_id -> Object3D
        self._msg_count = 0
        self._stats = defaultdict(int)

        # cost监控：最近N条代价记录，用于heartbeat统计
        self._cost_recent: deque = deque(maxlen=200)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.sub = self.create_subscription(Object3DArray, TOPIC, self._cb, qos)
        self.get_logger().info(f'Subscribed to {TOPIC}')

        if cost_mode:
            rosout_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1000,
            )
            self._rosout_sub = self.create_subscription(
                RosoutLog, '/rosout', self._rosout_cb, rosout_qos)
            self.get_logger().info('Cost监控已启用，订阅 /rosout')

    def _emit(self, color: str, tag: str, line: str, data: dict):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        data.update({'type': tag, 'ts': ts})
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        self._stats[tag] += 1
        if tag != 'SNAPSHOT':
            print(f"{color}[{tag}]{RESET} {DIM}{ts}{RESET} {line}")

    # ── cost监控 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_cost_log(msg: str) -> dict:
        """
        解析 byte_tracker_3d 打出的 [COST] 日志行。
        格式: [COST] s=1 trk=track_3 tcat=box dcat=can src=fused
                     cost=0.320 d=0.183 c=0.127 age=0.010 hits=17 dist=5.0cm
        返回 dict，解析失败返回 {}。
        """
        try:
            idx = msg.find(COST_LOG_PREFIX)
            if idx < 0:
                return {}
            parts = msg[idx:].split()
            d = {}
            for p in parts[1:]:   # 跳过 '[COST]'
                if '=' in p:
                    k, v = p.split('=', 1)
                    d[k] = v
            return d
        except Exception:
            return {}

    def _rosout_cb(self, msg: RosoutLog):
        """订阅 /rosout，解析 [COST] 行并显示 MATCH_COST 事件。"""
        if COST_LOG_PREFIX not in msg.msg:
            return

        d = self._parse_cost_log(msg.msg)
        if not d:
            return

        try:
            stage   = d.get('s', '?')
            trk     = d.get('trk', '?')
            tcat    = d.get('tcat', '?')
            dcat    = d.get('dcat', '?')
            src     = d.get('src', '?')
            cost    = float(d.get('cost', 0))
            d_term  = float(d.get('d', 0))
            c_term  = float(d.get('c', 0))
            age_term = float(d.get('age', 0))
            hits    = d.get('hits', '?')
            # dist 字段带 'cm' 后缀，去掉再解析
            dist_raw = d.get('dist', '0')
            dist_cm = float(dist_raw.rstrip('cm'))
        except (ValueError, KeyError):
            return

        # 颜色：接近gate=红，label flip(tcat≠dcat)=黄，正常=绿
        cat_mismatch = (tcat != dcat)
        near_gate    = (cost >= COST_WARN_THRESH)

        if near_gate:
            color = RED
        elif cat_mismatch:
            color = YELLOW
        else:
            color = GREEN

        line = (
            f"s{stage}  {trk}({tcat}"
            + (f"{YELLOW}→{dcat}{RESET}" if cat_mismatch else f"→{dcat}")
            + f") {DIM}{src}{RESET}  "
            f"cost={color}{cost:.3f}{RESET}  "
            f"[d={d_term:.3f} c={c_term:.3f} age={age_term:.3f}]  "
            f"{dist_cm:.1f}cm hits={hits}"
            + (f"  {RED}⚠ near-gate{RESET}" if near_gate else "")
        )

        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"{color}[MATCH_COST]{RESET} {DIM}{ts}{RESET} {line}")

        # 写日志 + 统计
        self._stats['MATCH_COST'] += 1
        if near_gate:
            self._stats['COST_WARN'] += 1
        if cat_mismatch:
            self._stats['COST_FLIP'] += 1

        with self._lock:
            self._cost_recent.append({
                'ts': ts, 's': stage,
                'trk': trk, 'tcat': tcat, 'dcat': dcat,
                'cost': cost, 'd': d_term, 'c': c_term, 'age': age_term,
                'hits': hits, 'dist_cm': dist_cm,
            })
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({'type': 'MATCH_COST', **self._cost_recent[-1]}) + '\n')

    def _cb(self, msg: Object3DArray):
        now = time.time()
        with self._lock:
            self._msg_count += 1

            tracked = [o for o in msg.objects if o.object_id.startswith('track_')]
            if self.target:
                tracked = [o for o in tracked if self.target in o.category]

            current_snapshot = {o.object_id: o for o in tracked}
            current_ids = set(current_snapshot.keys())
            ts_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]

            # ── SNAPSHOT: 每条消息完整状态 ──────────────────────────────
            self._emit('', 'SNAPSHOT', '', {
                'msg_seq': self._msg_count,
                'tracks': [
                    {
                        'id': o.object_id,
                        'cat': o.category,
                        'src': o.source_camera,
                        'pos': [round(v, 3) for v in pos_of(o)],
                        'score': round(o.score, 3),
                        'depth': round(o.depth, 3),
                    }
                    for o in tracked
                ],
            })

            # ── DOUBLE_CALL ──────────────────────────────────────────────
            if self._prev_msg_ts is not None:
                gap_ms = (now - self._prev_msg_ts) * 1000.0
                if gap_ms < DOUBLE_CALL_MS:
                    src_prev = {k: v.source_camera for k, v in self._prev_snapshot.items()}
                    src_curr = {k: current_snapshot[k].source_camera for k in current_ids}
                    changed = {k: (src_prev.get(k, '?'), src_curr.get(k, '?'))
                               for k in current_ids if src_prev.get(k) != src_curr.get(k)}
                    self._emit(RED, 'DOUBLE_CALL',
                        f"gap={gap_ms:.1f}ms ids={list(current_ids)} src_changes={changed}",
                        {'gap_ms': round(gap_ms, 1), 'source_changes': changed,
                         'ids': list(current_ids)})

            # ── TRACK_BORN + ID_CHANGE ───────────────────────────────────
            new_ids = current_ids - set(self._active.keys())
            for tid in new_ids:
                obj = current_snapshot[tid]
                pos = pos_of(obj)
                entry = TrackEntry(tid, now, obj.category)
                entry.update(now, obj.source_camera, pos)
                self._active[tid] = entry

                # 在dead tracks中找最近邻
                best_dead, best_dist = None, ID_CHANGE_DIST_M
                for dead_id, dead in self._dead.items():
                    if dead.last_pos is None:
                        continue
                    if (now - dead.last_ts) > ID_CHANGE_WINDOW_S:
                        continue
                    d = dist3d(pos, dead.last_pos)
                    if d < best_dist:
                        best_dist, best_dead = d, (dead_id, dead)

                if best_dead:
                    dead_id, dead = best_dead
                    gap_s = now - dead.last_ts
                    self._emit(RED + BOLD, 'ID_CHANGE',
                        f"{dead_id}({dead.category}) → {tid}({obj.category}) "
                        f"dist={best_dist:.3f}m gap={gap_s:.1f}s "
                        f"pos={fmt_pos(pos)} src={obj.source_camera} "
                        f"old_src={dead.last_source}",
                        {'old_id': dead_id, 'new_id': tid,
                         'old_cat': dead.category, 'new_cat': obj.category,
                         'dist_m': round(best_dist, 3), 'gap_s': round(gap_s, 2),
                         'old_pos': [round(v, 3) for v in dead.last_pos],
                         'new_pos': [round(v, 3) for v in pos],
                         'new_src': obj.source_camera,
                         'old_last_src': dead.last_source})
                else:
                    self._emit(GREEN, 'TRACK_BORN',
                        f"{tid}({obj.category}) pos={fmt_pos(pos)} src={obj.source_camera}",
                        {'track_id': tid, 'category': obj.category,
                         'pos': [round(v, 3) for v in pos],
                         'source': obj.source_camera})

            # ── 更新active tracks + SOURCE_JUMP + POSITION_JUMP ─────────
            for tid, obj in current_snapshot.items():
                if tid in new_ids:
                    continue
                entry = self._active[tid]
                prev_pos = entry.last_pos
                cur_pos = pos_of(obj)
                entry.update(now, obj.source_camera, cur_pos)

                # POSITION_JUMP
                if prev_pos is not None:
                    d = dist3d(prev_pos, cur_pos)
                    if d > POSITION_JUMP_M:
                        self._emit(YELLOW, 'POSITION_JUMP',
                            f"{tid}({obj.category}) jump={d:.3f}m "
                            f"{fmt_pos(prev_pos)} → {fmt_pos(cur_pos)} "
                            f"src={obj.source_camera} score={obj.score:.2f}",
                            {'track_id': tid, 'category': obj.category,
                             'jump_m': round(d, 3),
                             'prev_pos': [round(v, 3) for v in prev_pos],
                             'cur_pos': [round(v, 3) for v in cur_pos],
                             'source': obj.source_camera,
                             'score': round(obj.score, 3),
                             'depth': round(obj.depth, 3)})

                # SOURCE_JUMP
                is_jump, n_trans, sources = entry.check_source_jump(now)
                if is_jump:
                    entry.source_jump_last_report = now
                    self._emit(YELLOW, 'SOURCE_JUMP',
                        f"{tid}({obj.category}) transitions={n_trans} seq={sources}",
                        {'track_id': tid, 'category': obj.category,
                         'transitions': n_trans, 'recent_sources': sources,
                         'pos': [round(v, 3) for v in cur_pos]})

            # ── TRACK_LOST ───────────────────────────────────────────────
            gone_ids = set(self._active.keys()) - current_ids
            for tid in gone_ids:
                entry = self._active.pop(tid)
                self._dead[tid] = entry
                age_s = now - entry.born_ts
                self._emit(YELLOW, 'TRACK_LOST',
                    f"{tid}({entry.category}) age={age_s:.1f}s "
                    f"last_src={entry.last_source} last_pos={fmt_pos(entry.last_pos)}",
                    {'track_id': tid, 'category': entry.category,
                     'age_s': round(age_s, 2),
                     'last_src': entry.last_source,
                     'last_pos': [round(v, 3) for v in entry.last_pos]})

            # 清理过期dead tracks
            self._dead = {k: v for k, v in self._dead.items()
                          if (now - v.last_ts) < ID_CHANGE_WINDOW_S}

            self._prev_msg_ts = now
            self._prev_snapshot = current_snapshot

            # ── 心跳 ─────────────────────────────────────────────────────
            if self._msg_count % 50 == 0:
                print(f"\n{CYAN}── msgs={self._msg_count} "
                      f"[BORN:{self._stats['TRACK_BORN']} "
                      f"LOST:{self._stats['TRACK_LOST']} "
                      f"IC:{self._stats['ID_CHANGE']} "
                      f"SJ:{self._stats['SOURCE_JUMP']} "
                      f"PJ:{self._stats['POSITION_JUMP']} "
                      f"DC:{self._stats['DOUBLE_CALL']}] "
                      f"active={len(self._active)} ──{RESET}")
                for tid, e in self._active.items():
                    print(f"  {GREEN}→{RESET} {tid}({e.category}) "
                          f"src={e.last_source} pos={fmt_pos(e.last_pos)}")
                # cost统计（仅--cost模式）
                if self.cost_mode and self._stats['MATCH_COST'] > 0:
                    recent = list(self._cost_recent)
                    avg_cost = sum(r['cost'] for r in recent) / len(recent) if recent else 0
                    s1 = sum(1 for r in recent if r['s'] == '1')
                    s2 = sum(1 for r in recent if r['s'] == '2')
                    flip = self._stats['COST_FLIP']
                    warn = self._stats['COST_WARN']
                    print(f"  {CYAN}cost: total={self._stats['MATCH_COST']} "
                          f"avg={avg_cost:.3f} "
                          f"s1={s1} s2={s2} "
                          f"flip={flip} warn(>{COST_WARN_THRESH})={warn}{RESET}")
                print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None)
    parser.add_argument('--target', default=None, help='只监控某类别, 如 box')
    parser.add_argument('--cost', action='store_true',
                        help='开启代价监控（需感知节点 cost_debug=True）')
    args = parser.parse_args()

    if args.log is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log = os.path.join('/home/didi/workspace/MobileManipulator2/scripts', f'track_debug_{ts}.jsonl')

    print(f"{BOLD}=== Track稳定性调试工具 ==={RESET}")
    print(f"订阅: {TOPIC}")
    print(f"日志: {args.log}")
    if args.target:
        print(f"过滤类别: {args.target}")
    print(f"\n检测项:")
    print(f"  {RED}DOUBLE_CALL{RESET}   间隔 < {DOUBLE_CALL_MS}ms 的两次publish")
    print(f"  {YELLOW}SOURCE_JUMP{RESET}   同track source反复切换")
    print(f"  {YELLOW}POSITION_JUMP{RESET} 单帧位置突变 > {POSITION_JUMP_M}m")
    print(f"  {YELLOW}TRACK_LOST{RESET}    track消失")
    print(f"  {GREEN}TRACK_BORN{RESET}    track首次出现")
    print(f"  {RED+BOLD}ID_CHANGE{RESET}     新ID出现在旧track近邻 (< {ID_CHANGE_DIST_M}m)")
    if args.cost:
        print(f"  {GREEN}MATCH_COST{RESET}    每次匹配代价分解（--cost模式）")
        print(f"    颜色: {GREEN}绿=正常{RESET} {YELLOW}黄=类别不一致(flip){RESET} "
              f"{CYAN}青=fallback(Lost){RESET} {RED}红=接近gate(>{COST_WARN_THRESH}){RESET}")
    print(f"\n每条消息均写入SNAPSHOT到日志，每50条消息打印心跳...\n")
    print("─" * 60)

    rclpy.init()
    node = TrackDebugNode(args.log, args.target, cost_mode=args.cost)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{BOLD}=== 最终统计 ==={RESET}")
        for k, v in sorted(node._stats.items()):
            print(f"  {k}: {v}")
        print(f"  总消息数: {node._msg_count}")
        if args.cost and node._cost_recent:
            recent = list(node._cost_recent)
            costs = [r['cost'] for r in recent]
            print(f"\n{BOLD}── 代价统计 (最近{len(recent)}条) ──{RESET}")
            print(f"  avg={sum(costs)/len(costs):.3f}  "
                  f"max={max(costs):.3f}  min={min(costs):.3f}")
            s1 = [r for r in recent if r['s'] == '1']
            s2 = [r for r in recent if r['s'] == '2']
            if s1: print(f"  stage1: n={len(s1)} avg={sum(r['cost'] for r in s1)/len(s1):.3f}")
            if s2: print(f"  stage2: n={len(s2)} avg={sum(r['cost'] for r in s2)/len(s2):.3f}")
        print(f"\n日志: {node.log_file}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
