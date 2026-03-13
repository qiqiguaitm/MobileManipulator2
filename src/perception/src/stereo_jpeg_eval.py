#!/usr/bin/env python3
"""Stereo IR 编码格式评测: PNG vs JPEG q=95 深度精度对比

对 data/robot_captures 下所有帧，分别以 PNG 和 JPEG q=95 编码 IR 图发送到
FoundationStereo 服务，对比输出深度与 RealSense 原始深度(Ground Truth)的偏差。

用法:
    python3 stereo_jpeg_eval.py [--data-dir DIR] [--fs-url URL] [--jpeg-quality Q]
"""

import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np
import yaml
import requests


def send_stereo(url, ir_l, ir_r, intr_bytes, encode_fn):
    buf_l = encode_fn(ir_l)
    buf_r = encode_fn(ir_r)
    files = {
        'left_ir':  ('left_ir.jpg', buf_l, 'image/jpeg'),
        'right_ir': ('right_ir.jpg', buf_r, 'image/jpeg'),
        'intrinsics': ('intrinsics.yaml', intr_bytes, 'application/octet-stream'),
    }
    resp = requests.post(f'{url}/api/predict', files=files, timeout=60)
    resp.raise_for_status()
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)


def png_enc(img):
    _, b = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return b.tobytes()


def jpg_enc(img, quality=95):
    _, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return b.tobytes()


def load_frames(data_dir):
    captures = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    frames = []
    for cap in captures:
        cap_dir = os.path.join(data_dir, cap)
        rgb_files = sorted(glob.glob(os.path.join(cap_dir, 'rgb', '???.png')))
        intr_path = os.path.join(cap_dir, 'intrinsics.yaml')
        if not os.path.exists(intr_path):
            continue
        with open(intr_path) as f:
            intr = yaml.safe_load(f)
        intr_bytes = yaml.dump(intr).encode()
        for rgb_path in rgb_files:
            idx = os.path.splitext(os.path.basename(rgb_path))[0]
            ir_l = cv2.imread(os.path.join(cap_dir, 'ir_left', f'{idx}.png'), cv2.IMREAD_GRAYSCALE)
            ir_r = cv2.imread(os.path.join(cap_dir, 'ir_right', f'{idx}.png'), cv2.IMREAD_GRAYSCALE)
            raw_d = cv2.imread(os.path.join(cap_dir, 'depth', f'{idx}.png'), cv2.IMREAD_UNCHANGED)
            if ir_l is not None and ir_r is not None and raw_d is not None:
                frames.append((cap, idx, ir_l, ir_r, intr_bytes, raw_d))
    return frames


def main():
    parser = argparse.ArgumentParser(description='Stereo IR 编码格式精度评测')
    parser.add_argument('--data-dir', type=str,
                        default='/data/workspace/MobileManipulator2/data/robot_captures')
    parser.add_argument('--fs-url', type=str, default='http://192.168.112.14:8084')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    args = parser.parse_args()

    # 清除代理
    for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ.pop(k, None)

    frames = load_frames(args.data_dir)
    if not frames:
        print(f'没有可用帧: {args.data_dir}')
        sys.exit(1)

    q = args.jpeg_quality
    print(f'共 {len(frames)} 帧, JPEG quality={q}')

    # Warmup
    send_stereo(args.fs_url, frames[0][2], frames[0][3], frames[0][4], png_enc)

    ranges = [
        ('0-500mm',    0,  500),
        ('500mm-1m',   500, 1000),
        ('1-2m',       1000, 2000),
        ('2-3m',       2000, 3000),
        ('3-5m',       3000, 5000),
    ]

    png_vs_rs = {r[0]: [] for r in ranges}
    jpg_vs_rs = {r[0]: [] for r in ranges}
    jpg_vs_png = {r[0]: [] for r in ranges}

    # Per-frame results
    print()
    hdr = (f'{"frame":<25s} {"PNG→RS":>10s} {"JPG→RS":>10s} {"JPG-PNG":>10s} | '
           f'{"PNG≤10":>8s} {"JPG≤10":>8s} {"J-P≤10":>8s}')
    print(hdr)
    print('-' * len(hdr))

    jpg_fn = lambda img: jpg_enc(img, q)
    t_png_total = 0
    t_jpg_total = 0

    for cap, idx, ir_l, ir_r, intr_bytes, raw_d in frames:
        t0 = time.perf_counter()
        dp = send_stereo(args.fs_url, ir_l, ir_r, intr_bytes, png_enc).astype(np.float32)
        t_png_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        dj = send_stereo(args.fs_url, ir_l, ir_r, intr_bytes, jpg_fn).astype(np.float32)
        t_jpg_total += time.perf_counter() - t0

        dr = raw_d.astype(np.float32)

        valid = (dp > 0) & (dj > 0) & (dr > 0) & (dr <= 5000)
        if valid.sum() == 0:
            continue

        diff_pr = np.abs(dp - dr)
        diff_jr = np.abs(dj - dr)
        diff_jp = np.abs(dj - dp)

        v = valid
        n = v.sum()
        label = f'{cap}/{idx}'
        print(f'{label:<25s} '
              f'{diff_pr[v].mean():10.1f} {diff_jr[v].mean():10.1f} {diff_jp[v].mean():10.1f} | '
              f'{(diff_pr[v]<=10).sum()/n*100:7.1f}% '
              f'{(diff_jr[v]<=10).sum()/n*100:7.1f}% '
              f'{(diff_jp[v]<=10).sum()/n*100:7.1f}%')

        for rname, lo, hi in ranges:
            mask = valid & (dr > lo) & (dr <= hi)
            if mask.sum() > 0:
                png_vs_rs[rname].append(diff_pr[mask])
                jpg_vs_rs[rname].append(diff_jr[mask])
                jpg_vs_png[rname].append(diff_jp[mask])

    # Summary by depth range
    print()
    print('=' * 100)
    print(f'修改前后精度对比 — RealSense 原始深度为 Ground Truth (JPEG q={q})')
    print('=' * 100)
    print(f'{"深度段":>10s} | {"PNG→RS":>10s} {"JPG→RS":>10s} {"差值":>8s} | '
          f'{"PNG≤10mm":>10s} {"JPG≤10mm":>10s} | {"JPG vs PNG":>12s}')
    print('-' * 100)

    all_pr, all_jr, all_jp = [], [], []
    for rname, lo, hi in ranges:
        if png_vs_rs[rname]:
            ap = np.concatenate(png_vs_rs[rname])
            aj = np.concatenate(jpg_vs_rs[rname])
            ajp = np.concatenate(jpg_vs_png[rname])
            n = len(ap)
            delta = aj.mean() - ap.mean()
            print(f'{rname:>10s} | '
                  f'{ap.mean():10.1f} {aj.mean():10.1f} {delta:+8.1f} | '
                  f'{(ap<=10).sum()/n*100:9.1f}% {(aj<=10).sum()/n*100:9.1f}% | '
                  f'{ajp.mean():12.2f}mm')
            all_pr.append(ap)
            all_jr.append(aj)
            all_jp.append(ajp)
        else:
            print(f'{rname:>10s} |  no data')

    if all_pr:
        ap = np.concatenate(all_pr)
        aj = np.concatenate(all_jr)
        ajp = np.concatenate(all_jp)
        n = len(ap)
        delta = aj.mean() - ap.mean()
        print('-' * 100)
        print(f'{"0-5m":>10s} | '
              f'{ap.mean():10.1f} {aj.mean():10.1f} {delta:+8.1f} | '
              f'{(ap<=10).sum()/n*100:9.1f}% {(aj<=10).sum()/n*100:9.1f}% | '
              f'{ajp.mean():12.2f}mm')

    # Timing summary
    n_frames = len(frames)
    print()
    print(f'平均耗时: PNG={t_png_total/n_frames*1000:.0f}ms/帧  '
          f'JPEG q={q}={t_jpg_total/n_frames*1000:.0f}ms/帧  '
          f'加速={t_png_total/t_jpg_total:.2f}x')

    # Diff distribution (0-5m)
    if all_jp:
        ajp = np.concatenate(all_jp)
        print()
        print(f'JPG vs PNG 差异分布 (0-5m):')
        for t in [0, 1, 3, 5, 10, 50]:
            pct = (ajp > t).sum() / len(ajp) * 100
            print(f'  > {t:3d}mm: {pct:6.2f}%')


if __name__ == '__main__':
    main()
