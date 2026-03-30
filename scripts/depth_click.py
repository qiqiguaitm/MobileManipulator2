#!/usr/bin/env python3
"""鼠标点击查看 Raw vs CDM 深度值差异。

Usage:
    python3 scripts/_cc_depth_click.py captures/chassis_20260302_200838
    python3 scripts/_cc_depth_click.py captures/chassis_20260302_200838 --no-cdm-cache
"""

import argparse
import os
import sys

import cv2
import numpy as np

# 全局状态
g = {}


def run_cdm(rgb, raw_depth, cache_path):
    """调用 CDM 并缓存结果。"""
    if os.path.exists(cache_path):
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


def build_display(rgb, raw, cdm):
    """构建显示图像：左=RGB，右=差异热力图。"""
    h, w = raw.shape[:2]
    vmin, vmax = 300, 5000

    # 差异图
    diff = cdm.astype(np.float32) - raw.astype(np.float32)
    both_valid = (raw > 0) & (cdm > 0)
    max_delta = 150

    diff_vis = np.full((h, w, 3), 40, dtype=np.uint8)
    diff_vis[both_valid] = 255

    pos = both_valid & (diff > 0)
    strength = np.clip(diff[pos] / max_delta, 0, 1)
    diff_vis[pos, 0] = 255
    diff_vis[pos, 1] = (255 * (1 - strength)).astype(np.uint8)
    diff_vis[pos, 2] = (255 * (1 - strength)).astype(np.uint8)

    neg = both_valid & (diff < 0)
    strength = np.clip(-diff[neg] / max_delta, 0, 1)
    diff_vis[neg, 2] = 255
    diff_vis[neg, 1] = (255 * (1 - strength)).astype(np.uint8)
    diff_vis[neg, 0] = (255 * (1 - strength)).astype(np.uint8)

    # 缩放到显示尺寸
    scale = 0.5
    rgb_small = cv2.resize(rgb, None, fx=scale, fy=scale)
    diff_small = cv2.resize(diff_vis, None, fx=scale, fy=scale)

    display = np.hstack([rgb_small, diff_small])
    return display, scale


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    scale = g['scale']
    h, w = g['raw'].shape[:2]
    disp_w = int(w * scale)

    # 判断点击在左半(RGB)还是右半(diff)
    if x < disp_w:
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        panel = 'RGB'
    else:
        orig_x = int((x - disp_w) / scale)
        orig_y = int(y / scale)
        panel = 'Diff'

    if orig_x < 0 or orig_x >= w or orig_y < 0 or orig_y >= h:
        return

    raw_v = int(g['raw'][orig_y, orig_x])
    cdm_v = int(g['cdm'][orig_y, orig_x])
    diff_v = cdm_v - raw_v

    raw_m = raw_v / 1000.0
    cdm_m = cdm_v / 1000.0

    tag = ''
    if raw_v == 0 and cdm_v == 0:
        tag = ' [both invalid]'
    elif raw_v == 0:
        tag = ' [raw invalid, CDM filled]'
    elif cdm_v == 0:
        tag = ' [CDM zeroed]'
    elif abs(diff_v) > 50:
        tag = ' *** LARGE ***'

    print(f'  ({orig_x:4d}, {orig_y:4d})  '
          f'Raw={raw_v:5d}mm ({raw_m:.3f}m)  '
          f'CDM={cdm_v:5d}mm ({cdm_m:.3f}m)  '
          f'Δ={diff_v:+5d}mm{tag}')

    # 在显示图上画十字标记
    display = g['display_base'].copy()
    color = (0, 255, 0)
    # 在两个面板上都标记
    for dx in [0, disp_w]:
        px = int(orig_x * scale) + dx
        py = int(orig_y * scale)
        cv2.drawMarker(display, (px, py), color, cv2.MARKER_CROSS, 20, 1)

    # 在图上显示信息
    info = f'Raw={raw_v}mm  CDM={cdm_v}mm  D={diff_v:+d}mm'
    cv2.putText(display, info, (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, info, (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('Depth Compare (click to inspect)', display)


def main():
    parser = argparse.ArgumentParser(description='鼠标点击查看 Raw vs CDM 深度差异')
    parser.add_argument('prefix', help='文件前缀, e.g. captures/chassis_20260302_200838')
    parser.add_argument('--no-cdm-cache', action='store_true', help='强制重新调用 CDM')
    args = parser.parse_args()

    prefix = args.prefix.rstrip('_')
    rgb_path = f'{prefix}_rgb.jpg'
    depth_path = f'{prefix}_depth.png'
    cdm_cache = f'{prefix}_cdm_cached.png'

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f'找不到文件: {rgb_path} 或 {depth_path}')
        sys.exit(1)

    if args.no_cdm_cache and os.path.exists(cdm_cache):
        os.remove(cdm_cache)

    rgb = cv2.imread(rgb_path)
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    print(f'RGB {rgb.shape}, Depth {raw.shape} range=[{raw[raw>0].min()}, {raw.max()}]mm')

    cdm = run_cdm(rgb, raw, cdm_cache)
    print(f'CDM depth loaded, range=[{cdm[cdm>0].min()}, {cdm.max()}]mm')

    display, scale = build_display(rgb, raw, cdm)

    g['raw'] = raw
    g['cdm'] = cdm
    g['scale'] = scale
    g['display_base'] = display.copy()

    print(f'\n左侧=RGB  右侧=差异图(蓝=CDM更深, 红=CDM更浅)')
    print(f'点击任意位置查看深度值, 按 q 退出\n')

    cv2.imshow('Depth Compare (click to inspect)', display)
    cv2.setMouseCallback('Depth Compare (click to inspect)', on_mouse)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
