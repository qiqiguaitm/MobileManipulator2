#!/usr/bin/env python3
"""
运行命令并过滤重复错误。只显示唯一错误，重复完全静默，退出后汇总。
用法: python3 _cc_run_filtered.py [命令...]
默认: make full-manual
"""

import sys
import re
import subprocess
import hashlib
from collections import defaultdict, OrderedDict

RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ROS2 标准日志: [LEVEL] [1234.567] [node]: msg
ROS2_RE = re.compile(
    r'\[(?P<level>ERROR|WARN|WARNING|FATAL|INFO|DEBUG)\]'
    r'(?:\s*\[[\d.]+\])?\s*\[(?P<node>[^\]]+)\]:\s*(?P<msg>.*)'
)

# launch 进程前缀的内部日志: [node-pid]  MM/DD HH:MM:SS,mmm LEVEL [threadid] (file:line) msg
PROC_RE = re.compile(
    r'\[(?P<proc>[^\]]+)\]\s+'
    r'\d{2}/\d{2} \d{2}:\d{2}:\d{2},\d{3}\s+'
    r'(?P<level>WARNING|ERROR|INFO|DEBUG)\s+'
    r'\[\d+\]\s*'
    r'(?:\([^)]+\)\s*)?'
    r'(?P<msg>.*)'
)

ERROR_KW = re.compile(r'\b(error|exception|traceback|abort|fatal|failed|failure)\b', re.I)
WARN_KW  = re.compile(r'\b(warn|warning|deprecated)\b', re.I)

# normalize: 剥离所有数字、地址、时间戳、线程ID，保留语义骨架
_STRIP = [
    re.compile(r'0x[0-9a-fA-F]+'),           # 16进制地址
    re.compile(r'\[\d{6,}\]'),                # 线程ID [281472xxx]
    re.compile(r'\d{2}/\d{2} \d{2}:\d{2}:\d{2},\d{3}'),  # 时间戳
    re.compile(r'\b\d+\.\d+\.\d+\b'),         # 版本号
    re.compile(r'\b\d+\b'),                   # 普通整数
    re.compile(r'/[^\s:,)]+'),                # 路径
]

LAUNCH_KW = ['FULL-MANUAL', 'MANUAL', 'Launching', 'process started', 'process has finished']


def normalize(msg):
    s = msg
    for pat in _STRIP:
        s = pat.sub('_', s)
    return re.sub(r'\s+', ' ', s).strip()


def dedup_key(level, node, msg):
    # node 去掉进程号后缀 (-25, -30 等)
    node_clean = re.sub(r'-\d+$', '', node)
    raw = level + '|' + node_clean + '|' + normalize(msg)
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def parse_line(line):
    """返回 (level, node, msg) 或 None"""
    m = ROS2_RE.search(line)
    if m:
        return m.group('level'), m.group('node'), m.group('msg').strip()
    m = PROC_RE.search(line)
    if m:
        lvl = m.group('level')
        lvl = 'WARN' if lvl == 'WARNING' else lvl
        return lvl, m.group('proc'), m.group('msg').strip()
    if ERROR_KW.search(line):
        return 'ERROR', '', line.strip()
    if WARN_KW.search(line):
        return 'WARN', '', line.strip()
    return None


def fmt_level(level):
    if level in ('ERROR', 'FATAL'):
        return RED + BOLD + '[' + level + ']' + RESET
    if level in ('WARN', 'WARNING'):
        return YELLOW + '[WARN ]' + RESET
    return CYAN + '[' + level + ']' + RESET


def run(cmd):
    # key -> [level, node, msg, count]
    seen = OrderedDict()

    print(CYAN + '=' * 64 + RESET)
    print(BOLD + 'CMD: ' + ' '.join(cmd) + RESET)
    print(CYAN + '=' * 64 + RESET)
    print(GRAY + 'INFO/DEBUG 已隐藏  |  重复完全静默  |  Ctrl-C 退出' + RESET + '\n')

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue

            # 打印 launch 启动/停止行
            if any(k in line for k in LAUNCH_KW):
                print(GRAY + line + RESET)
                continue

            result = parse_line(line)
            if result is None:
                continue

            level, node, msg = result

            if level in ('INFO', 'DEBUG'):
                continue

            dk = dedup_key(level, node, msg)
            if dk in seen:
                seen[dk][3] += 1
                continue

            # 第一次出现：打印
            seen[dk] = [level, node, msg, 1]
            node_str = ' ' + GRAY + '[' + node + ']' + RESET if node else ''
            # 消息截断到120字符
            display_msg = msg if len(msg) <= 120 else msg[:117] + '...'
            print(fmt_level(level) + node_str + ' ' + display_msg)

        proc.wait()

    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

    # 汇总
    errors   = [(v, k) for k, v in seen.items() if v[0] in ('ERROR', 'FATAL')]
    warnings = [(v, k) for k, v in seen.items() if v[0] in ('WARN', 'WARNING')]

    print('\n' + CYAN + '=' * 64 + RESET)
    print(BOLD + '退出汇总' + RESET)
    print('  唯一 ERROR/FATAL : ' + RED + str(len(errors)) + RESET)
    print('  唯一 WARN        : ' + YELLOW + str(len(warnings)) + RESET)
    total_suppressed = sum(v[3] - 1 for v in seen.values())
    print('  静默抑制重复行   : ' + GRAY + str(total_suppressed) + RESET)

    if errors:
        print('\n' + BOLD + 'TOP ERROR (按重复次数):' + RESET)
        top = sorted(errors, key=lambda x: -x[0][3])[:10]
        for v, _ in top:
            node_str = '[' + v[1] + '] ' if v[1] else ''
            msg_short = v[2][:80] + ('...' if len(v[2]) > 80 else '')
            print('  x' + str(v[3]).ljust(5) + ' ' + RED + node_str + RESET + msg_short)

    print(CYAN + '=' * 64 + RESET)


if __name__ == '__main__':
    cmd = sys.argv[1:] if len(sys.argv) > 1 else ['make', 'full-manual']
    run(cmd)
