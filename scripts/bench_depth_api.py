#!/usr/bin/env python3
"""
纯网络传输基准测试。
从磁盘读取已压缩图像，binary blob 发送，分离传输时间和服务端时间。

用法:
  python3 _cc_bench_depth_api.py --images-dir scripts/bench_images
  python3 _cc_bench_depth_api.py --images-dir scripts/bench_images --host 192.168.112.14
"""

import argparse
import os
import struct
import time
import numpy as np
import requests

FIELDS = ('rgb', 'dpt', 'ir_left', 'ir_right')


def load_from_dir(images_dir):
    """从磁盘加载已压缩图像字节，按分辨率目录组织。"""
    result = {}
    for name in sorted(os.listdir(images_dir)):
        sub = os.path.join(images_dir, name)
        if not os.path.isdir(sub):
            continue
        paths = {
            'rgb': os.path.join(sub, 'rgb.jpg'),
            'dpt': os.path.join(sub, 'depth.png'),
            'ir_left': os.path.join(sub, 'ir_left.jpg'),
            'ir_right': os.path.join(sub, 'ir_right.jpg'),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        raw = {k: open(p, 'rb').read() for k, p in paths.items()}
        result[name] = raw
    return result


def pack_binary(raw):
    """[4B len][data] x 4"""
    parts = []
    for key in FIELDS:
        d = raw[key]
        parts.append(struct.pack('<I', len(d)))
        parts.append(d)
    return b''.join(parts)


def send_once(url, blob, session):
    """发送一次，返回 (http_ms, server_ms, transfer_ms, recv_kb, ok)"""
    t0 = time.time()
    resp = session.post(url, data=blob,
                        headers={'Content-Type': 'application/octet-stream'},
                        timeout=60)
    http_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        return http_ms, 0, http_ms, 0, False

    server_ms = float(resp.headers.get('X-Server-Ms', 0))
    transfer_ms = http_ms - server_ms
    recv_kb = len(resp.content) / 1024

    return http_ms, server_ms, transfer_ms, recv_kb, True


def main():
    parser = argparse.ArgumentParser(description="纯网络传输基准测试")
    parser.add_argument('--host', default='192.168.112.14')
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('-n', '--rounds', type=int, default=10)
    parser.add_argument('--images-dir', default='scripts/bench_images')
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/api/predict_bin"

    # 加载图像
    loaded = load_from_dir(args.images_dir)
    if not loaded:
        print(f"未找到图像: {args.images_dir}")
        return

    # 连通性检查
    try:
        r = requests.get(f"http://{args.host}:{args.port}/api/health", timeout=3)
        print(f"服务: {url}  OK")
    except Exception:
        print(f"服务: {url}  尝试连接...")

    print(f"轮数: {args.rounds}  分辨率: {list(loaded.keys())}\n")

    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': None, 'https': None}

    all_stats = []

    for label, raw in loaded.items():
        blob = pack_binary(raw)
        up_kb = len(blob) / 1024

        transfer_times = []
        server_times = []
        recv_kbs = []

        for i in range(args.rounds):
            http_ms, srv_ms, xfer_ms, recv_kb, ok = send_once(url, blob, session)
            mark = "ok" if ok else "FAIL"
            print(f"  {label} [{i+1:>2}/{args.rounds}] {mark}  "
                  f"传输={xfer_ms:.1f}  服务={srv_ms:.1f}  "
                  f"总={http_ms:.1f} ms")
            if ok:
                transfer_times.append(xfer_ms)
                server_times.append(srv_ms)
                recv_kbs.append(recv_kb)

        if not transfer_times:
            continue

        recv_kb = np.mean(recv_kbs)
        total_kb = up_kb + recv_kb
        xfer_avg = np.mean(transfer_times)
        srv_avg = np.mean(server_times)
        speed_mbps = (total_kb / 1024) / (xfer_avg / 1000) * 8 if xfer_avg > 0 else 0

        all_stats.append({
            'label': label,
            'up_kb': up_kb,
            'down_kb': recv_kb,
            'total_kb': total_kb,
            'xfer_avg': xfer_avg,
            'xfer_p50': np.median(transfer_times),
            'xfer_p95': np.percentile(transfer_times, 95),
            'srv_avg': srv_avg,
            'speed_mbps': speed_mbps,
        })

    if not all_stats:
        return

    # 汇总
    all_stats.sort(key=lambda s: s['xfer_avg'])
    W = 95
    print(f"\n{'='*W}")
    print("  纯传输时间对比 (已扣除服务端处理)")
    print(f"{'='*W}")
    print(f"  {'分辨率':<24} {'上行':>6} {'下行':>5} {'传输avg':>8} {'p50':>7} {'p95':>7} {'服务端':>7} {'吞吐':>10} {'Hz':>6}")
    print(f"  {'':-<24} {'':-<6} {'':-<5} {'':-<8} {'':-<7} {'':-<7} {'':-<7} {'':-<10} {'':-<6}")

    for s in all_stats:
        hz = 1000 / s['xfer_avg'] if s['xfer_avg'] > 0 else 0
        print(f"  {s['label']:<24} {s['up_kb']:>5.0f}K {s['down_kb']:>4.0f}K "
              f"{s['xfer_avg']:>7.1f}ms {s['xfer_p50']:>6.1f}ms {s['xfer_p95']:>6.1f}ms "
              f"{s['srv_avg']:>6.1f}ms {s['speed_mbps']:>7.1f}Mbps {hz:>5.1f}")

    print(f"{'='*W}")

    best = max(all_stats, key=lambda s: s['total_kb'])
    print(f"  网络吞吐: {best['speed_mbps']:.0f} Mbps "
          f"(基于 {best['label']} {best['total_kb']:.0f}KB / {best['xfer_avg']:.0f}ms 传输)")

    worst_jitter = max(all_stats, key=lambda s: s['xfer_p95'] - s['xfer_p50'])
    jitter = worst_jitter['xfer_p95'] - worst_jitter['xfer_p50']
    if jitter > 10:
        print(f"  网络抖动: p95-p50={jitter:.0f}ms @ {worst_jitter['label']}")


if __name__ == '__main__':
    main()
