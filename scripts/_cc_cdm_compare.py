#!/usr/bin/env python3
"""CDM 深度对比面板 — Raw vs CDM 深度可视化 + 差异图 + 统计。

生成 2x2 面板:
  左上: Raw Depth (turbo colormap)
  右上: CDM Depth (turbo colormap)
  左下: Diff (blue=CDM deeper, red=CDM shallower)
  右下: 统计信息

Usage:
    python3 scripts/_cc_cdm_compare.py captures/chassis_20260302_200838
    python3 scripts/_cc_cdm_compare.py captures/chassis_20260302_200838 --no-cdm-cache
    python3 scripts/_cc_cdm_compare.py captures/chassis_20260302_200838 --vmin 300 --vmax 4500
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ========== CDM 调用 ==========

def run_cdm(rgb, raw_depth, cache_path, force=False):
    """调用 CDM 并缓存结果。"""
    if not force and os.path.exists(cache_path):
        print(f'加载 CDM 缓存: {cache_path}')
        return cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)

    print('调用 CDM 服务...')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                    '..', 'install', 'perception', 'lib', 'python3.10', 'dist-packages'))
    from perception.percept import DepthOptimizerOnline
    from perception.scene_perception_core import SimpleConfig

    cfg = SimpleConfig(url='http://192.168.112.14:8082', chosen_policy='dn', warmup=0)
    cdm = DepthOptimizerOnline(cfg)
    result = cdm.forward(rgb, raw_depth, chosen_policy='dn')
    if not result.get('success'):
        print(f"CDM 失败: {result.get('error')}")
        sys.exit(1)

    cdm_depth = result['depth']
    cv2.imwrite(cache_path, cdm_depth)
    print(f'CDM 结果已缓存: {cache_path}')
    return cdm_depth


# ========== 可视化 ==========

def depth_to_turbo(depth_mm, vmin=300, vmax=4500):
    """深度图 → TURBO colormap (BGR)。"""
    d = np.clip((depth_mm.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth_mm == 0] = 0
    return colored


def make_diff_image(raw_mm, cdm_mm, max_delta=100):
    """发散色标差异图: 白=无差异, 红=CDM更浅, 蓝=CDM更深。"""
    diff = cdm_mm.astype(np.float64) - raw_mm.astype(np.float64)
    valid = (raw_mm > 100) & (cdm_mm > 100)
    norm = np.clip(diff / max_delta, -1.0, 1.0)

    img = np.full((*raw_mm.shape, 3), 255, dtype=np.uint8)

    neg = norm < 0
    intensity = (-norm * 255).astype(np.uint8)
    img[neg, 0] = np.clip(255 - intensity[neg], 0, 255).astype(np.uint8)
    img[neg, 1] = np.clip(255 - intensity[neg], 0, 255).astype(np.uint8)
    img[neg, 2] = 255

    pos = norm > 0
    intensity_pos = (norm * 255).astype(np.uint8)
    img[pos, 0] = 255
    img[pos, 1] = np.clip(255 - intensity_pos[pos], 0, 255).astype(np.uint8)
    img[pos, 2] = np.clip(255 - intensity_pos[pos], 0, 255).astype(np.uint8)

    img[~valid] = 40
    return img


def draw_text(img, text, pos, scale=0.5, color=(255, 255, 255), thickness=1):
    """带黑色描边的文字。"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def compute_stats(raw_mm, cdm_mm):
    """计算 CDM 偏差统计。"""
    valid = (raw_mm > 100) & (cdm_mm > 100)
    changed = valid & (raw_mm != cdm_mm)
    diff = (cdm_mm.astype(np.float64) - raw_mm.astype(np.float64))[valid]

    if len(diff) == 0:
        return None

    return {
        'changed': int(changed.sum()),
        'valid': int(valid.sum()),
        'changed_pct': changed.sum() / max(valid.sum(), 1) * 100,
        'mean': float(np.mean(diff)),
        'std': float(np.std(diff)),
        'median': float(np.median(diff)),
        'p5': float(np.percentile(diff, 5)),
        'p95': float(np.percentile(diff, 95)),
        'min': float(np.min(diff)),
        'max': float(np.max(diff)),
    }


def build_panel(raw_mm, cdm_mm, vmin, vmax, max_delta):
    """构建 2x2 对比面板。"""
    h, w = raw_mm.shape[:2]

    # 目标面板宽度 (每个子图)
    panel_w = min(w, 720)
    scale = panel_w / w
    panel_h = int(h * scale)

    # 三张可视化图
    raw_vis = cv2.resize(depth_to_turbo(raw_mm, vmin, vmax), (panel_w, panel_h))
    cdm_vis = cv2.resize(depth_to_turbo(cdm_mm, vmin, vmax), (panel_w, panel_h))
    diff_vis = cv2.resize(make_diff_image(raw_mm, cdm_mm, max_delta), (panel_w, panel_h))

    # 标签
    draw_text(raw_vis, f'Raw Depth ({vmin}-{vmax}mm)', (5, 20), 0.55, (0, 255, 255))
    draw_text(cdm_vis, f'CDM Depth ({vmin}-{vmax}mm)', (5, 20), 0.55, (0, 255, 255))
    draw_text(diff_vis,
              f'Diff (blue=CDM deeper, red=CDM shallower, +/- {max_delta}mm)',
              (5, 20), 0.45, (0, 255, 255))

    # 统计面板
    stats = compute_stats(raw_mm, cdm_mm)
    stats_img = np.full((panel_h, panel_w, 3), 40, dtype=np.uint8)

    if stats:
        y = 30
        lines = [
            (f'Changed: {stats["changed"]}/{stats["valid"]} '
             f'({stats["changed_pct"]:.1f}%)', (200, 200, 200)),
            (f'Mean:  {stats["mean"]:.1f}mm', (200, 200, 200)),
            (f'Std:  {stats["std"]:.1f}mm', (200, 200, 200)),
            (f'Median:  {stats["median"]:.1f}mm', (200, 200, 200)),
            (f'P5/P95:  {stats["p5"]:.1f} / {stats["p95"]:+.1f}mm', (200, 200, 200)),
            (f'Min/Max:  {stats["min"]:.0f} / {stats["max"]:+.0f}mm', (200, 200, 200)),
        ]
        for text, color in lines:
            draw_text(stats_img, text, (10, y), 0.55, color)
            y += 30

    # 组装 2x2
    top = np.hstack([raw_vis, cdm_vis])
    bottom = np.hstack([diff_vis, stats_img])
    return np.vstack([top, bottom])


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='CDM 深度对比面板')
    parser.add_argument('prefix', help='文件前缀, e.g. captures/chassis_20260302_200838')
    parser.add_argument('--no-cdm-cache', action='store_true', help='强制重新调用 CDM')
    parser.add_argument('--vmin', type=int, default=300, help='深度色标下限 mm (default: 300)')
    parser.add_argument('--vmax', type=int, default=4500, help='深度色标上限 mm (default: 4500)')
    parser.add_argument('--max-delta', type=int, default=100,
                        help='差异图色标范围 mm (default: 100)')
    args = parser.parse_args()

    prefix = args.prefix.rstrip('_')
    rgb_path = f'{prefix}_rgb.jpg'
    depth_path = f'{prefix}_depth.png'
    cdm_cache = f'{prefix}_cdm_cached.png'
    out_path = f'{prefix}_cdm_compare.png'

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f'找不到文件: {rgb_path} 或 {depth_path}')
        sys.exit(1)

    rgb = cv2.imread(rgb_path)
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    print(f'RGB: {rgb.shape}')
    print(f'Depth: {raw.shape} dtype={raw.dtype} '
          f'range=[{raw[raw > 0].min()}, {raw.max()}]mm')

    cdm = run_cdm(rgb, raw, cdm_cache, force=args.no_cdm_cache)
    print(f'CDM: range=[{cdm[cdm > 0].min()}, {cdm.max()}]mm')

    panel = build_panel(raw, cdm, args.vmin, args.vmax, args.max_delta)
    cv2.imwrite(out_path, panel)
    print(f'\n面板已保存: {out_path}')

    stats = compute_stats(raw, cdm)
    if stats:
        print(f'  Changed: {stats["changed"]}/{stats["valid"]} ({stats["changed_pct"]:.1f}%)')
        print(f'  Mean: {stats["mean"]:.1f}mm  Std: {stats["std"]:.1f}mm')
        print(f'  Median: {stats["median"]:.1f}mm')
        print(f'  P5/P95: {stats["p5"]:.1f} / {stats["p95"]:+.1f}mm')


if __name__ == '__main__':
    main()
