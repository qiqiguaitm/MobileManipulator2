#!/usr/bin/env python3
"""
percept_benchmark.py - 感知服务性能评测脚本 (ROS2 版本)

测试 DinoX, SAM3, GraspAnything, CDM 四个服务的串行执行时间分布
支持可视化结果输出到 result 目录

Usage:
    ros2 run perception percept_benchmark --vis           # 可视化测试
    ros2 run perception percept_benchmark --benchmark     # 性能测试
    ros2 run perception percept_benchmark --benchmark --num-runs 20
"""

import sys
import os
import glob
import time
import statistics
import threading
import yaml
from datetime import datetime
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# 支持直接 python3 执行（无需 ROS2 环境）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from percept import (DinoXDetectorOnline, SAM3Online, GraspAnythingOnline, DepthOptimizerOnline,
                     CombinedPerceptionClient, DualPerceptionClient)


class SimpleConfig:
    """简单的配置类，用于初始化服务客户端"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_camera_data(base_dir):
    """Load all frames from a camera capture directory.

    Returns: (frames_list, intrinsics_dict)
      frames_list: [{'rgb': ndarray, 'depth': ndarray, 'ir_left': ndarray, 'ir_right': ndarray}, ...]
      intrinsics_dict: parsed intrinsics.yaml content
    """
    rgb_files = sorted(glob.glob(os.path.join(base_dir, 'rgb', '???.png')))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb frames found in {base_dir}/rgb/")

    frames = []
    for rgb_path in rgb_files:
        idx = os.path.splitext(os.path.basename(rgb_path))[0]  # e.g. '000'
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(os.path.join(base_dir, 'depth', f'{idx}.png'), cv2.IMREAD_UNCHANGED)
        ir_left = cv2.imread(os.path.join(base_dir, 'ir_left', f'{idx}.png'), cv2.IMREAD_GRAYSCALE)
        ir_right = cv2.imread(os.path.join(base_dir, 'ir_right', f'{idx}.png'), cv2.IMREAD_GRAYSCALE)

        if rgb is None or ir_left is None or ir_right is None:
            print(f"  Warning: skipping frame {idx} in {base_dir} (missing data)")
            continue

        frames.append({'rgb': rgb, 'depth': depth, 'ir_left': ir_left, 'ir_right': ir_right, 'idx': idx})

    intr_path = os.path.join(base_dir, 'intrinsics.yaml')
    with open(intr_path) as f:
        intrinsics = yaml.safe_load(f)

    if not frames:
        raise RuntimeError(f"No valid frames loaded from {base_dir}")

    print(f"  Loaded {len(frames)} frames from {base_dir}")
    return frames, intrinsics


# ========== 可视化函数 ==========

def ensure_result_dir():
    """确保 result 目录存在，返回带时间戳的子目录路径"""
    # 使用 /tmp 作为输出目录
    result_dir = '/tmp/perception_benchmark'
    os.makedirs(result_dir, exist_ok=True)

    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(result_dir, f'benchmark_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def vis_detection(rgb, result, service_name, categories=None):
    """可视化检测结果 (DinoX/SAM3)

    Args:
        rgb: 原始 BGR 图像
        result: LocalTaskResult.result，格式 {'objects': [...]}
        service_name: 服务名称，用于标注
        categories: 类别列表（可选）

    Returns:
        vis_img: 可视化后的图像
    """
    vis_img = rgb.copy()

    if not result or 'objects' not in result:
        return vis_img

    objects = result['objects']

    for i, obj in enumerate(objects):
        bbox = obj.get('bbox')
        score = obj.get('score', 0)
        category = obj.get('category', f'obj{i}')

        if not bbox:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 根据索引生成颜色
        color = ((i * 67 + 50) % 255, (i * 97 + 100) % 255, (i * 137 + 80) % 255)

        # 绘制边界框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label = f'{category}: {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # 绘制标签文字
        cv2.putText(vis_img, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 添加服务名称标注
    cv2.putText(vis_img, f'[{service_name}] {len(objects)} objects', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis_img


def vis_grasp(rgb, objs, padded_img=None):
    """可视化抓取检测结果 (GraspAnything)

    Args:
        rgb: 原始 BGR 图像
        objs: GraspAnything 返回的 objs 列表
        padded_img: padding 后的图像（可选）

    Returns:
        vis_img: 可视化后的图像
    """
    vis_img = rgb.copy()
    h, w = rgb.shape[:2]

    if not objs or len(objs) == 0 or len(objs[0]) == 0:
        cv2.putText(vis_img, '[GraspAnything] No objects', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis_img

    obj_list = objs[0]

    # 创建 mask 叠加层
    overlay = vis_img.copy()

    for i, obj in enumerate(obj_list):
        # 生成颜色
        color = ((i * 67 + 50) % 255, (i * 97 + 100) % 255, (i * 137 + 80) % 255)

        # 绘制 mask
        if 'dt_mask' in obj and obj['dt_mask'] is not None:
            mask = obj['dt_mask']
            if mask.shape[:2] == (h, w):
                overlay[mask] = color

        # 绘制边界框
        if 'dt_bbox' in obj:
            bbox = obj['dt_bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 绘制抓取点/抓取线
        if 'touching_points' in obj:
            for pts in obj['touching_points']:
                if len(pts) >= 2:
                    pt1, pt2 = pts[0], pts[1]
                    pt1 = (int(pt1[0]), int(pt1[1]))
                    pt2 = (int(pt2[0]), int(pt2[1]))
                    # 绘制抓取线
                    cv2.line(vis_img, pt1, pt2, (0, 0, 255), 3)
                    # 绘制端点
                    cv2.circle(vis_img, pt1, 5, (255, 0, 0), -1)
                    cv2.circle(vis_img, pt2, 5, (255, 0, 0), -1)

        # 绘制 affordance 中心点
        if 'affs' in obj:
            for aff in obj['affs']:
                cx, cy = int(aff[0]), int(aff[1])
                cv2.circle(vis_img, (cx, cy), 4, (0, 255, 255), -1)

    # 混合 mask 叠加层
    vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)

    # 添加服务名称标注
    cv2.putText(vis_img, f'[GraspAnything] {len(obj_list)} objects', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis_img


def vis_depth(depth_mm, depth_denoised=None):
    """可视化深度图 (CDM)

    Args:
        depth_mm: 原始深度图 (uint16, 单位 mm)
        depth_denoised: 去噪后的深度图 (uint16, 单位 mm)

    Returns:
        vis_img: 拼接的可视化图像 (原始 | 去噪)
    """
    def depth_to_colormap(depth):
        """深度图转伪彩色"""
        # 归一化到 0-255
        valid_mask = depth > 0
        if not np.any(valid_mask):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        d_min = np.min(depth[valid_mask])
        d_max = np.max(depth[valid_mask])

        if d_max == d_min:
            normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            normalized = ((depth.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # 无效区域设为 0
        normalized[~valid_mask] = 0

        # 应用 colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

        # 无效区域设为黑色
        colored[~valid_mask] = [0, 0, 0]

        return colored

    vis_orig = depth_to_colormap(depth_mm)
    cv2.putText(vis_orig, 'Original Depth', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if depth_denoised is not None:
        vis_denoised = depth_to_colormap(depth_denoised)
        cv2.putText(vis_denoised, 'CDM Denoised', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 水平拼接
        vis_img = np.hstack([vis_orig, vis_denoised])
    else:
        vis_img = vis_orig

    return vis_img


# ========== CDM 深度偏差对比函数 ==========

# 深度段配置 (mm)
CDM_DEPTH_BINS = [(500, 1000), (1000, 2000), (2000, 3000)]


def depth_to_turbo(depth_mm, vmin=200, vmax=4000):
    """深度图 → TURBO colormap (固定范围, BGR)。"""
    d = np.clip((depth_mm.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth_mm == 0] = 0
    return colored


def make_diff_image(raw_mm, cdm_mm, max_delta=100):
    """发散色标差异图: 白=0, 红=CDM更浅, 蓝=CDM更深。"""
    diff = cdm_mm.astype(np.float64) - raw_mm.astype(np.float64)
    valid = (raw_mm > 100) & (cdm_mm > 100)
    norm = np.clip(diff / max_delta, -1.0, 1.0)

    img = np.full((*raw_mm.shape, 3), 255, dtype=np.uint8)

    # CDM 更浅 (diff < 0) → 红色
    neg = norm < 0
    intensity = (-norm * 255).astype(np.uint8)
    img[neg, 0] = np.clip(255 - intensity[neg], 0, 255).astype(np.uint8)
    img[neg, 1] = np.clip(255 - intensity[neg], 0, 255).astype(np.uint8)
    img[neg, 2] = 255

    # CDM 更深 (diff > 0) → 蓝色
    pos = norm > 0
    intensity_pos = (norm * 255).astype(np.uint8)
    img[pos, 0] = 255
    img[pos, 1] = np.clip(255 - intensity_pos[pos], 0, 255).astype(np.uint8)
    img[pos, 2] = np.clip(255 - intensity_pos[pos], 0, 255).astype(np.uint8)

    img[~valid] = 128
    return img


def draw_text_outlined(img, text, pos, scale=0.5, color=(255, 255, 255), thickness=1):
    """带黑色描边的文字绘制。"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def make_histogram(diff_valid, width, height):
    """差异直方图 (BGR)。"""
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    if len(diff_valid) == 0:
        draw_text_outlined(canvas, 'No valid pixels', (10, height // 2), 0.6)
        return canvas

    bins = np.linspace(-200, 200, 81)
    hist, edges = np.histogram(diff_valid, bins=bins)
    max_count = max(hist.max(), 1)

    ml, mr, mt, mb = 60, 20, 20, 40
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    bar_w = max(plot_w // len(hist), 1)

    for i, count in enumerate(hist):
        bar_h = int(count / max_count * plot_h)
        x = ml + i * bar_w
        mid_val = (edges[i] + edges[i + 1]) / 2
        if mid_val < -5:
            color = (80, 80, 255)
        elif mid_val > 5:
            color = (255, 80, 80)
        else:
            color = (200, 200, 200)
        cv2.rectangle(canvas, (x, mt + plot_h - bar_h), (x + bar_w - 1, mt + plot_h), color, -1)

    for val in [-200, -100, 0, 100, 200]:
        idx = int((val - edges[0]) / (edges[-1] - edges[0]) * len(hist))
        x = ml + idx * bar_w
        draw_text_outlined(canvas, f'{val}', (x - 15, mt + plot_h + 25), 0.4, (180, 180, 180))

    draw_text_outlined(canvas, f'{max_count}', (5, mt + 15), 0.35, (180, 180, 180))
    draw_text_outlined(canvas, '0', (5, mt + plot_h + 5), 0.35, (180, 180, 180))

    zero_idx = int((0 - edges[0]) / (edges[-1] - edges[0]) * len(hist))
    cv2.line(canvas, (ml + zero_idx * bar_w, mt), (ml + zero_idx * bar_w, mt + plot_h), (0, 255, 0), 1)
    draw_text_outlined(canvas, 'CDM delta (mm)', (width // 2 - 60, height - 5), 0.4, (180, 180, 180))
    return canvas


def compute_cdm_stats(raw_mm, cdm_mm):
    """全图 + 分深度段统计。"""
    raw_f = raw_mm.astype(np.float64)
    cdm_f = cdm_mm.astype(np.float64)
    diff = cdm_f - raw_f

    valid = (raw_mm > 100) & (cdm_mm > 100)
    changed = valid & (raw_mm != cdm_mm)
    diff_valid = diff[valid]

    stats = {
        'valid_pixels': int(valid.sum()),
        'changed_pixels': int(changed.sum()),
        'changed_pct': changed.sum() / max(valid.sum(), 1) * 100,
        'mean': float(diff_valid.mean()) if len(diff_valid) else 0.0,
        'std': float(diff_valid.std()) if len(diff_valid) else 0.0,
        'bins': [],
    }
    for d_min, d_max in CDM_DEPTH_BINS:
        mask = valid & (raw_mm >= d_min) & (raw_mm < d_max)
        n = int(mask.sum())
        if n > 0:
            d = diff[mask]
            stats['bins'].append({
                'range': f'{d_min / 1000:.1f}-{d_max / 1000:.1f}m',
                'n': n,
                'mean': float(d.mean()),
                'std': float(d.std()),
            })
    return stats


def print_cdm_stats(stats):
    """终端打印 CDM 偏差统计。"""
    print(f'  全图 (有效像素 {stats["valid_pixels"]}):')
    print(f'    改变比例: {stats["changed_pct"]:.1f}%')
    direction = 'CDM偏浅' if stats['mean'] < 0 else 'CDM偏深'
    print(f'    均值偏差: {stats["mean"]:.1f} mm ({direction})')
    print(f'    标准差:   {stats["std"]:.1f} mm')
    if stats['bins']:
        print(f'    分段:')
        for b in stats['bins']:
            print(f'      {b["range"]}: mean={b["mean"]:+.1f}mm std={b["std"]:.1f}mm n={b["n"]}')


def build_cdm_compare_panel(raw_mm, cdm_mm, stats):
    """组装 CDM 对比可视化面板。"""
    h, w = raw_mm.shape[:2]
    panel_w = 400
    panel_h = int(panel_w * h / w)

    raw_vis = cv2.resize(depth_to_turbo(raw_mm), (panel_w, panel_h))
    cdm_vis = cv2.resize(depth_to_turbo(cdm_mm), (panel_w, panel_h))
    diff_vis = cv2.resize(make_diff_image(raw_mm, cdm_mm), (panel_w, panel_h))

    raw_valid = raw_mm[raw_mm > 100]
    cdm_valid = cdm_mm[cdm_mm > 100]
    raw_median = int(np.median(raw_valid)) if len(raw_valid) else 0
    cdm_median = int(np.median(cdm_valid)) if len(cdm_valid) else 0

    draw_text_outlined(raw_vis, f'Raw depth (median={raw_median}mm)', (5, 20), 0.45)
    draw_text_outlined(cdm_vis, f'CDM depth (median={cdm_median}mm)', (5, 20), 0.45)
    draw_text_outlined(diff_vis, f'Diff (mean={stats["mean"]:.1f}mm)', (5, 20), 0.45)
    draw_text_outlined(diff_vis, 'Red=CDM shallower  Blue=CDM deeper', (5, panel_h - 10), 0.35)

    top_row = np.hstack([raw_vis, cdm_vis, diff_vis])

    total_w = panel_w * 3
    bottom_h = 200
    valid = (raw_mm > 100) & (cdm_mm > 100)
    diff = (cdm_mm.astype(np.float64) - raw_mm.astype(np.float64))[valid]

    hist_w = total_w * 2 // 3
    hist_img = make_histogram(diff, hist_w, bottom_h)

    stats_w = total_w - hist_w
    stats_img = np.full((bottom_h, stats_w, 3), 40, dtype=np.uint8)
    y = 25
    draw_text_outlined(stats_img, 'CDM Depth Bias Stats', (5, y), 0.5, (0, 255, 255))
    y += 25
    draw_text_outlined(stats_img, f'Changed: {stats["changed_pct"]:.1f}%', (5, y), 0.4)
    y += 20
    draw_text_outlined(stats_img, f'Mean: {stats["mean"]:.1f} mm', (5, y), 0.4)
    y += 20
    draw_text_outlined(stats_img, f'Std: {stats["std"]:.1f} mm', (5, y), 0.4)
    y += 30
    for b in stats['bins']:
        draw_text_outlined(stats_img, f'{b["range"]}: {b["mean"]:+.1f} +/- {b["std"]:.1f} mm',
                           (5, y), 0.35, (180, 220, 180))
        y += 18

    bottom_row = np.hstack([hist_img, stats_img])
    return np.vstack([top_row, bottom_row])


def save_visualization_results(run_dir, data_name, results_dict):
    """保存单个数据集的所有可视化结果

    Args:
        run_dir: 输出目录
        data_name: 数据集名称 (e.g., '001')
        results_dict: 各服务的可视化结果 {'DinoX': vis_img, ...}
    """
    for service_name, vis_img in results_dict.items():
        if vis_img is not None:
            fp = os.path.join(run_dir, f'{data_name}_{service_name}.jpg')
            cv2.imwrite(fp, vis_img)


def run_visualization_test(all_data, run_dir, text_prompt_dinox, text_prompt_sam3, grasp_server_config):
    """运行可视化测试，为每个数据集生成各服务的可视化结果

    Args:
        all_data: 数据集列表 [{'name': '001', 'rgb': ..., 'depth': ...}, ...]
        run_dir: 输出目录
        text_prompt_dinox: DinoX 的 text prompt
        text_prompt_sam3: SAM3 的 text prompt
        grasp_server_config: GraspAnything 服务器配置文件路径
    """
    print(f"\n{'='*70}")
    print(f"开始可视化测试 (输出目录: {run_dir})")
    print(f"{'='*70}")

    # 初始化服务
    services = {}

    # DinoX
    try:
        cfg = SimpleConfig(
            url='http://192.168.112.14:10086',
            min_score=0.25,
            iou_threshold=0.5,
            warmup=0
        )
        services['DinoX'] = DinoXDetectorOnline(cfg)
        print("[DinoX] 服务初始化成功")
    except Exception as e:
        print(f"[DinoX] 服务初始化失败: {e}")

    # SAM3
    try:
        cfg = SimpleConfig(
            url='http://192.168.112.14:8081',
            min_score=0.30,
            return_mask=False,
            warmup=0
        )
        services['SAM3'] = SAM3Online(cfg)
        print("[SAM3] 服务初始化成功")
    except Exception as e:
        print(f"[SAM3] 服务初始化失败: {e}")

    # GraspAnything
    if grasp_server_config and os.path.exists(grasp_server_config):
        try:
            cfg = SimpleConfig(
                server_list=grasp_server_config,
                model_name='full',
                warmup=0
            )
            services['Grasp'] = GraspAnythingOnline(cfg)
            print("[GraspAnything] 服务初始化成功")
        except Exception as e:
            print(f"[GraspAnything] 服务初始化失败: {e}")
    else:
        print(f"[GraspAnything] 跳过 - 配置文件不存在: {grasp_server_config}")

    # CDM
    try:
        cfg = SimpleConfig(
            url='http://192.168.112.14:8082',
            chosen_policy='dn',
            warmup=0
        )
        services['CDM'] = DepthOptimizerOnline(cfg)
        print("[CDM] 服务初始化成功")
    except Exception as e:
        print(f"[CDM] 服务初始化失败: {e}")

    # 对每个数据集运行可视化
    for data in all_data:
        name = data['name']
        rgb = data['rgb']
        depth = data['depth']

        print(f"\n处理数据集: {name}")
        vis_results = {}

        # DinoX
        if 'DinoX' in services:
            try:
                result = services['DinoX'].forward(text_prompt_dinox, rgb)
                vis_results['DinoX'] = vis_detection(rgb, result.result, 'DinoX')
                print(f"  [DinoX] 检测到 {len(result.result.get('objects', []))} 个目标")
            except Exception as e:
                print(f"  [DinoX] 失败: {e}")

        # SAM3
        if 'SAM3' in services:
            try:
                result = services['SAM3'].forward(text_prompt_sam3, rgb)
                vis_results['SAM3'] = vis_detection(rgb, result.result, 'SAM3')
                print(f"  [SAM3] 检测到 {len(result.result.get('objects', []))} 个目标")
            except Exception as e:
                print(f"  [SAM3] 失败: {e}")

        # GraspAnything
        if 'Grasp' in services:
            try:
                objs, padded_img = services['Grasp'].forward(rgb, depth=None)
                vis_results['Grasp'] = vis_grasp(rgb, objs, padded_img)
                obj_count = len(objs[0]) if objs and len(objs) > 0 else 0
                print(f"  [GraspAnything] 检测到 {obj_count} 个目标")
            except Exception as e:
                print(f"  [GraspAnything] 失败: {e}")

        # CDM
        if 'CDM' in services:
            try:
                result = services['CDM'].forward(rgb, depth, chosen_policy='dn')
                if result.get('success') and 'depth' in result:
                    vis_results['CDM'] = vis_depth(depth, result['depth'])
                    print(f"  [CDM] 去噪成功")
                else:
                    vis_results['CDM'] = vis_depth(depth, None)
                    print(f"  [CDM] 去噪失败: {result.get('error', 'unknown')}")
            except Exception as e:
                print(f"  [CDM] 失败: {e}")

        # 保存原始 RGB 图像
        rgb_fp = os.path.join(run_dir, f'{name}_RGB.jpg')
        cv2.imwrite(rgb_fp, rgb)

        # 保存各服务可视化结果
        save_visualization_results(run_dir, name, vis_results)

    print(f"\n可视化结果已保存到: {run_dir}")
    return run_dir


def benchmark_with_timing(func, args_list, kwargs, name, num_runs=10, warmup_runs=3):
    """带详细时间分布的基准测试，支持多数据集轮换

    Args:
        func: 测试函数
        args_list: 参数列表，每个元素是一组 (data_name, args) 元组
        kwargs: 额外参数
        name: 测试名称
        num_runs: 每个数据集测试次数
        warmup_runs: 预热次数
    """
    times = []
    timing_stats = {}
    num_datasets = len(args_list)
    total_runs = num_runs * num_datasets

    print(f"\n{'='*70}")
    print(f"[{name}] 开始测试 ({num_datasets} 组数据, 每组 {num_runs} 次, 预热 {warmup_runs} 次)")
    print(f"{'='*70}")

    # 预热阶段 - 不计入统计，轮换数据集
    for i in range(warmup_runs):
        data_name, args = args_list[i % num_datasets]
        timing = {}
        kwargs['_timing'] = timing
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [预热] 第 {i+1:2d} 次 [{data_name}]: {elapsed*1000:7.1f}ms")

    # 正式测试 - 轮换数据集
    run_idx = 0
    for r in range(num_runs):
        for data_name, args in args_list:
            run_idx += 1
            timing = {}
            kwargs['_timing'] = timing

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # 收集时间分布
            for k, v in timing.items():
                if k not in timing_stats:
                    timing_stats[k] = []
                timing_stats[k].append(v)

            # 打印单次结果
            timing_str = ' | '.join([f"{k}:{v:.0f}" for k, v in timing.items()])
            print(f"  第 {run_idx:2d} 次 [{data_name}]: {elapsed*1000:7.1f}ms | {timing_str}")

    # 统计
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"\n[{name}] 统计结果:")
    print(f"  总耗时:   平均={avg_time*1000:.1f}ms  最小={min_time*1000:.1f}ms  最大={max_time*1000:.1f}ms  标准差={std_time*1000:.1f}ms")

    # 打印时间分布统计
    if timing_stats:
        print(f"\n  时间分布 (平均值):")
        local_total = 0
        http_total = 0
        for k, v in timing_stats.items():
            avg_v = statistics.mean(v)
            print(f"    {k:<20}: {avg_v:6.1f} ms")
            if 'http' in k.lower():
                http_total += avg_v
            else:
                local_total += avg_v
        print(f"    {'─'*35}")
        print(f"    {'本地处理':<20}: {local_total:6.1f} ms")
        print(f"    {'HTTP服务':<20}: {http_total:6.1f} ms")

    return {
        'name': name,
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'timing_stats': timing_stats
    }


def benchmark_concurrent_sam3_cdm_2way(sam3_service, cdm_service, all_data, text_prompt,
                                        num_runs=10, warmup_runs=3):
    """双路并发 benchmark: 模拟双相机 (SAM3 + CDM) × 2 同时调用

    Args:
        sam3_service: SAM3Online 实例
        cdm_service: DepthOptimizerOnline 实例
        all_data: 数据集列表
        text_prompt: SAM3 text prompt
        num_runs: 测试次数
        warmup_runs: 预热次数
    """
    n = len(all_data)
    times = []
    detail_times = {k: [] for k in ['SAM3-cam1', 'SAM3-cam2', 'CDM-cam1', 'CDM-cam2']}

    print(f"\n{'='*70}")
    print(f"[SAM3+CDM 双路] 2x SAM3 + 2x CDM 同时调用 ({num_runs} 次, 预热 {warmup_runs} 次)")
    print(f"{'='*70}")

    def call_sam3(rgb):
        t0 = time.perf_counter()
        sam3_service.forward(text_prompt, rgb)
        return time.perf_counter() - t0

    def call_cdm(rgb, depth):
        t0 = time.perf_counter()
        cdm_service.forward(rgb, depth, chosen_policy='dn')
        return time.perf_counter() - t0

    def run_once(d1, d2):
        """一次 4 路并发：cam1(SAM3+CDM) + cam2(SAM3+CDM)"""
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_sam1 = executor.submit(call_sam3, d1['rgb'])
            f_sam2 = executor.submit(call_sam3, d2['rgb'])
            f_cdm1 = executor.submit(call_cdm, d1['rgb'], d1['depth'])
            f_cdm2 = executor.submit(call_cdm, d2['rgb'], d2['depth'])

            r_sam1 = f_sam1.result(timeout=60)
            r_sam2 = f_sam2.result(timeout=60)
            r_cdm1 = f_cdm1.result(timeout=60)
            r_cdm2 = f_cdm2.result(timeout=60)

        wall = time.perf_counter() - t0
        return wall, r_sam1, r_sam2, r_cdm1, r_cdm2

    # 预热
    for i in range(warmup_runs):
        d1 = all_data[i % n]
        d2 = all_data[(i + 1) % n]
        wall, s1, s2, c1, c2 = run_once(d1, d2)
        print(f"  [预热] 第 {i+1} 次 [{d1['name']}+{d2['name']}]: "
              f"{wall*1000:.1f}ms | SAM3:{s1*1000:.0f}/{s2*1000:.0f} CDM:{c1*1000:.0f}/{c2*1000:.0f}")

    # 正式测试
    for i in range(num_runs):
        d1 = all_data[i % n]
        d2 = all_data[(i + 1) % n]
        wall, s1, s2, c1, c2 = run_once(d1, d2)

        times.append(wall)
        detail_times['SAM3-cam1'].append(s1)
        detail_times['SAM3-cam2'].append(s2)
        detail_times['CDM-cam1'].append(c1)
        detail_times['CDM-cam2'].append(c2)

        print(f"  第 {i+1:2d} 次 [{d1['name']}+{d2['name']}]: "
              f"{wall*1000:.1f}ms | SAM3:{s1*1000:.0f}/{s2*1000:.0f} CDM:{c1*1000:.0f}/{c2*1000:.0f}")

    # 统计
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"\n[SAM3+CDM 双路] 统计结果:")
    print(f"  总耗时: 平均={avg_time*1000:.1f}ms  最小={min_time*1000:.1f}ms  "
          f"最大={max_time*1000:.1f}ms  标准差={std_time*1000:.1f}ms")
    print(f"  理论最大频率: {1/(avg_time):.1f} Hz")

    print(f"\n  各服务耗时 (平均):")
    for label in ['SAM3-cam1', 'SAM3-cam2', 'CDM-cam1', 'CDM-cam2']:
        avg = statistics.mean(detail_times[label]) * 1000
        print(f"    {label:<12}: {avg:6.1f} ms")

    serial_sum = sum(statistics.mean(v) for v in detail_times.values()) * 1000
    print(f"    {'─'*28}")
    print(f"    {'串行理论':<12}: {serial_sum:6.1f} ms")
    print(f"    {'并行实际':<12}: {avg_time*1000:6.1f} ms")
    print(f"    {'加速比':<12}: {serial_sum / (avg_time * 1000):.2f}x")

    return {
        'name': 'SAM3+CDM 双路并发',
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'detail_times': detail_times,
    }


def benchmark_concurrent_sam3_stereo_2way(sam3_service, fs_service,
                                           top_frames, top_intrinsics,
                                           chassis_frames, chassis_intrinsics,
                                           text_prompt='pen,box,phone,bottle,toy',
                                           num_runs=10, warmup_runs=3):
    """双路并发 benchmark: 模拟双相机 (SAM3 + Stereo) × 2 同时调用

    Args:
        sam3_service: SAM3Online 实例
        fs_service: DepthOptimizerOnline 实例 (FoundationStereo)
        top_frames: top 相机帧列表
        top_intrinsics: top 相机内参
        chassis_frames: chassis 相机帧列表
        chassis_intrinsics: chassis 相机内参
        text_prompt: SAM3 text prompt
        num_runs: 测试次数
        warmup_runs: 预热次数
    """
    n_top = len(top_frames)
    n_chassis = len(chassis_frames)
    times = []
    detail_times = {k: [] for k in ['SAM3-top', 'SAM3-chassis', 'FS-top', 'FS-chassis']}

    print(f"\n{'='*70}")
    print(f"[SAM3+Stereo 双路] 2x SAM3 + 2x Stereo 同时调用 "
          f"({num_runs} 次, 预热 {warmup_runs} 次)")
    print(f"{'='*70}")

    def call_sam3(rgb):
        t0 = time.perf_counter()
        sam3_service.forward(text_prompt, rgb)
        return time.perf_counter() - t0

    def call_stereo(ir_left, ir_right, intrinsics):
        t0 = time.perf_counter()
        fs_service.forward_stereo(ir_left, ir_right, intrinsics)
        return time.perf_counter() - t0

    def run_once(d_top, d_chassis):
        """一次 4 路并发：top(SAM3+FS) + chassis(SAM3+FS)"""
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_sam_top = executor.submit(call_sam3, d_top['rgb'])
            f_sam_chassis = executor.submit(call_sam3, d_chassis['rgb'])
            f_fs_top = executor.submit(call_stereo,
                                       d_top['ir_left'], d_top['ir_right'], top_intrinsics)
            f_fs_chassis = executor.submit(call_stereo,
                                           d_chassis['ir_left'], d_chassis['ir_right'],
                                           chassis_intrinsics)

            r_sam_top = f_sam_top.result(timeout=60)
            r_sam_chassis = f_sam_chassis.result(timeout=60)
            r_fs_top = f_fs_top.result(timeout=60)
            r_fs_chassis = f_fs_chassis.result(timeout=60)

        wall = time.perf_counter() - t0
        return wall, r_sam_top, r_sam_chassis, r_fs_top, r_fs_chassis

    # 预热
    for i in range(warmup_runs):
        d_top = top_frames[i % n_top]
        d_chassis = chassis_frames[i % n_chassis]
        wall, s1, s2, f1, f2 = run_once(d_top, d_chassis)
        print(f"  [预热] 第 {i+1} 次 [top:{d_top['idx']}+chassis:{d_chassis['idx']}]: "
              f"{wall*1000:.1f}ms | SAM3:{s1*1000:.0f}/{s2*1000:.0f} "
              f"FS:{f1*1000:.0f}/{f2*1000:.0f}")

    # 正式测试
    for i in range(num_runs):
        d_top = top_frames[i % n_top]
        d_chassis = chassis_frames[i % n_chassis]
        wall, s1, s2, f1, f2 = run_once(d_top, d_chassis)

        times.append(wall)
        detail_times['SAM3-top'].append(s1)
        detail_times['SAM3-chassis'].append(s2)
        detail_times['FS-top'].append(f1)
        detail_times['FS-chassis'].append(f2)

        print(f"  第 {i+1:2d} 次 [top:{d_top['idx']}+chassis:{d_chassis['idx']}]: "
              f"{wall*1000:.1f}ms | SAM3:{s1*1000:.0f}/{s2*1000:.0f} "
              f"FS:{f1*1000:.0f}/{f2*1000:.0f}")

    # 统计
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"\n[SAM3+Stereo 双路] 统计结果:")
    print(f"  总耗时: 平均={avg_time*1000:.1f}ms  最小={min_time*1000:.1f}ms  "
          f"最大={max_time*1000:.1f}ms  标准差={std_time*1000:.1f}ms")
    print(f"  理论最大频率: {1/(avg_time):.1f} Hz")

    print(f"\n  各服务耗时 (平均):")
    for label in ['SAM3-top', 'SAM3-chassis', 'FS-top', 'FS-chassis']:
        avg = statistics.mean(detail_times[label]) * 1000
        print(f"    {label:<14}: {avg:6.1f} ms")

    serial_sum = sum(statistics.mean(v) for v in detail_times.values()) * 1000
    print(f"    {'─'*30}")
    print(f"    {'串行理论':<14}: {serial_sum:6.1f} ms")
    print(f"    {'并行实际':<14}: {avg_time*1000:6.1f} ms")
    print(f"    {'加速比':<14}: {serial_sum / (avg_time * 1000):.2f}x")

    return {
        'name': 'SAM3+Stereo 双路并发',
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'detail_times': detail_times,
    }


def benchmark_parallel_pipeline(sam3_service, fs_service,
                                top_frames, top_intrinsics,
                                chassis_frames, chassis_intrinsics,
                                text_prompt='pen,box,phone,bottle,toy',
                                num_cycles=20, warmup_cycles=3):
    """Dual-camera parallel SAM3+FoundationStereo benchmark.

    Simulates multi_camera_perception_node architecture: two cameras each
    running SAM3 + FS in parallel via a shared thread pool.
    """
    executor = ThreadPoolExecutor(max_workers=4)
    results_lock = threading.Lock()
    results = {'top': [], 'chassis': []}
    total_cycles = warmup_cycles + num_cycles

    print(f"\n{'='*80}")
    print(f"[Parallel Pipeline] 2 cameras × (SAM3 + FS) — "
          f"{num_cycles} cycles, {warmup_cycles} warmup")
    print(f"  Top: {len(top_frames)} frames, Chassis: {len(chassis_frames)} frames")
    print(f"  Prompt: {text_prompt}")
    print(f"{'='*80}")

    def camera_worker(cam_name, frames, intrinsics, res_list):
        n = len(frames)
        for i in range(total_cycles):
            frame = frames[i % n]
            det_timing, fs_timing = {}, {}
            is_warmup = i < warmup_cycles
            cycle_num = i - warmup_cycles + 1

            try:
                t0 = time.perf_counter()
                fut_det = executor.submit(sam3_service.forward, text_prompt, frame['rgb'],
                                          _timing=det_timing)
                fut_fs = executor.submit(fs_service.forward_stereo,
                                         frame['ir_left'], frame['ir_right'], intrinsics,
                                         _timing=fs_timing)
                det_result = fut_det.result(timeout=30)
                fs_result = fut_fs.result(timeout=60)
                wall_s = time.perf_counter() - t0
            except Exception as e:
                print(f"  [{cam_name:>7}] ERROR cycle {i}: {e}")
                continue
            wall_ms = wall_s * 1000

            sam3_pre = det_timing.get('sam3_preprocess', 0)
            sam3_enc = det_timing.get('sam3_encode', 0)
            sam3_http = det_timing.get('sam3_http', 0)
            sam3_total = sam3_pre + sam3_enc + sam3_http
            fs_enc = fs_timing.get('fs_encode', 0)
            fs_http = fs_timing.get('fs_http', 0)
            fs_total_ms = fs_enc + fs_http
            parallel_ms = wall_ms - max(sam3_total, fs_total_ms)

            tag = 'WARM' if is_warmup else 'PERF'
            num = i + 1 if is_warmup else cycle_num
            print(f"  [{cam_name:>7}] {tag} #{num:03d}: "
                  f"parallel={wall_ms:.0f}ms(idle={max(0,parallel_ms):.0f}ms) "
                  f"A=[pre:{sam3_pre:.0f}+enc:{sam3_enc:.0f}+http:{sam3_http:.0f}={sam3_total:.0f}ms] "
                  f"B=[enc:{fs_enc:.0f}+http:{fs_http:.0f}={fs_total_ms:.0f}ms] "
                  f"wall={wall_ms:.0f}ms")

            if not is_warmup:
                record = {
                    'wall_ms': wall_ms,
                    'sam3_preprocess': sam3_pre,
                    'sam3_encode': sam3_enc, 'sam3_http': sam3_http, 'sam3_total': sam3_total,
                    'fs_encode': fs_enc, 'fs_http': fs_http, 'fs_total': fs_total_ms,
                    'idle_ms': max(0, parallel_ms),
                }
                with results_lock:
                    res_list.append(record)

    # Run both cameras in parallel threads
    t_top = threading.Thread(target=camera_worker, name='cam-top',
                             args=('top', top_frames, top_intrinsics, results['top']))
    t_chassis = threading.Thread(target=camera_worker, name='cam-chassis',
                                 args=('chassis', chassis_frames, chassis_intrinsics,
                                       results['chassis']))

    t_global_start = time.perf_counter()
    t_top.start()
    t_chassis.start()
    t_top.join()
    t_chassis.join()
    total_wall = time.perf_counter() - t_global_start
    executor.shutdown(wait=False)

    # ========== Summary statistics ==========
    def stats_for(vals):
        if not vals:
            return {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0}
        s = sorted(vals)
        n = len(s)
        return {
            'avg': statistics.mean(s),
            'p50': s[n // 2],
            'p95': s[min(int((n - 1) * 0.95), n - 1)],
            'min': s[0],
            'max': s[-1],
        }

    print(f"\n{'='*80}")
    print(f"[Summary] Dual-Camera Parallel SAM3+FS  ({num_cycles} cycles each)")
    print(f"{'='*80}")

    stages = ['wall_ms', 'sam3_preprocess', 'sam3_encode', 'sam3_http', 'sam3_total',
              'fs_encode', 'fs_http', 'fs_total', 'idle_ms']
    stage_labels = {
        'wall_ms': 'Wall', 'sam3_preprocess': 'SAM3 pre', 'sam3_encode': 'SAM3 enc',
        'sam3_http': 'SAM3 http', 'sam3_total': 'SAM3 total',
        'fs_encode': 'FS enc', 'fs_http': 'FS http',
        'fs_total': 'FS total', 'idle_ms': 'Idle',
    }

    for cam_name in ['top', 'chassis']:
        recs = results[cam_name]
        print(f"\n  [{cam_name}] ({len(recs)} cycles)")
        print(f"    {'Stage':<12} {'avg':>7} {'p50':>7} {'p95':>7} {'min':>7} {'max':>7}")
        print(f"    {'─'*47}")
        for stage in stages:
            vals = [r[stage] for r in recs]
            s = stats_for(vals)
            print(f"    {stage_labels[stage]:<12} {s['avg']:7.1f} {s['p50']:7.1f} "
                  f"{s['p95']:7.1f} {s['min']:7.1f} {s['max']:7.1f}")

    # System-wide combined statistics (both cameras merged)
    all_recs = results['top'] + results['chassis']
    if all_recs:
        print(f"\n  [system] Both cameras combined ({len(all_recs)} cycles)")
        print(f"    {'Stage':<12} {'avg':>7} {'p50':>7} {'p95':>7} {'min':>7} {'max':>7}")
        print(f"    {'─'*47}")
        for stage in stages:
            vals = [r[stage] for r in all_recs]
            s = stats_for(vals)
            print(f"    {stage_labels[stage]:<12} {s['avg']:7.1f} {s['p50']:7.1f} "
                  f"{s['p95']:7.1f} {s['min']:7.1f} {s['max']:7.1f}")

    # Bottleneck analysis
    print(f"\n  Bottleneck analysis:")
    for cam_name in ['top', 'chassis']:
        recs = results[cam_name]
        sam3_wins = sum(1 for r in recs if r['sam3_total'] > r['fs_total'])
        fs_wins = len(recs) - sam3_wins
        print(f"    [{cam_name}] SAM3 slower: {sam3_wins}/{len(recs)} "
              f"({100*sam3_wins/max(1,len(recs)):.0f}%)  "
              f"FS slower: {fs_wins}/{len(recs)} "
              f"({100*fs_wins/max(1,len(recs)):.0f}%)")
    if all_recs:
        sam3_wins_all = sum(1 for r in all_recs if r['sam3_total'] > r['fs_total'])
        fs_wins_all = len(all_recs) - sam3_wins_all
        print(f"    [system] SAM3 slower: {sam3_wins_all}/{len(all_recs)} "
              f"({100*sam3_wins_all/len(all_recs):.0f}%)  "
              f"FS slower: {fs_wins_all}/{len(all_recs)} "
              f"({100*fs_wins_all/len(all_recs):.0f}%)")

    # Frequency analysis
    print(f"\n  Detection frequency:")
    for cam_name in ['top', 'chassis']:
        recs = results[cam_name]
        avg_wall = statistics.mean([r['wall_ms'] for r in recs])
        hz = 1000.0 / avg_wall if avg_wall > 0 else 0
        print(f"    [{cam_name}] avg={avg_wall:.1f}ms → {hz:.2f} Hz")

    if all_recs:
        avg_per_cam = statistics.mean([r['wall_ms'] for r in all_recs])
        interleaved_hz = 2 * 1000.0 / avg_per_cam if avg_per_cam > 0 else 0
        serial_sum = statistics.mean([r['sam3_total'] + r['fs_total'] for r in all_recs])
        speedup = serial_sum / avg_per_cam if avg_per_cam > 0 else 0
        print(f"    [system] per-cycle avg={avg_per_cam:.1f}ms → "
              f"per-cam {1000.0/avg_per_cam:.2f} Hz, "
              f"interleaved {interleaved_hz:.2f} Hz")
        print(f"    [system] serial would be {serial_sum:.1f}ms → "
              f"parallel speedup {speedup:.2f}x")

    # Total wall time
    print(f"\n  Total benchmark wall time: {total_wall:.1f}s")
    print(f"{'='*80}")


def benchmark_combined_dual(url_gpu0, url_gpu1,
                            top_frames, top_intrinsics,
                            chassis_frames, chassis_intrinsics,
                            text_prompt='pen,box,phone,bottle,toy',
                            num_runs=10, warmup_runs=3):
    """Combined server 双 GPU 并行 benchmark.

    每次迭代: top → GPU0, chassis → GPU1, 并行执行 SAM3+FS。
    衡量端到端 wall time, 判断是否达到 5Hz (≤200ms)。
    """
    n_top = len(top_frames)
    n_chassis = len(chassis_frames)

    cfg0 = SimpleConfig(url=url_gpu0, confidence=0.30, return_mask=False)
    cfg1 = SimpleConfig(url=url_gpu1, confidence=0.30, return_mask=False)
    dual = DualPerceptionClient(cfg0, cfg1)

    # health check
    health = dual.check_health()
    for i, h in enumerate(health):
        status = h.get('status', 'unknown')
        print(f"  [GPU{i}] health: {status}")
        if status != 'ok':
            print(f"  [GPU{i}] WARNING: server not ready — {h}")

    times = []
    detail_times = []

    print(f"\n{'='*70}")
    print(f"[Combined Dual] GPU0={url_gpu0}  GPU1={url_gpu1}")
    print(f"  {num_runs} runs, {warmup_runs} warmup, prompt={text_prompt}")
    print(f"  Top: {n_top} frames, Chassis: {n_chassis} frames")
    print(f"{'='*70}")

    for i in range(warmup_runs + num_runs):
        is_warmup = i < warmup_runs
        d_top = top_frames[i % n_top]
        d_chassis = chassis_frames[i % n_chassis]
        timing = {}

        t0 = time.perf_counter()
        r0, r1 = dual.forward(
            d_top['rgb'], d_top['ir_left'], d_top['ir_right'], top_intrinsics,
            d_chassis['rgb'], d_chassis['ir_left'], d_chassis['ir_right'], chassis_intrinsics,
            text_prompt=text_prompt, _timing=timing)
        wall = time.perf_counter() - t0
        wall_ms = wall * 1000

        # summary per GPU
        gpu0_http = timing.get('gpu0_combined_http', 0)
        gpu1_http = timing.get('gpu1_combined_http', 0)
        gpu0_enc = timing.get('gpu0_combined_encode', 0)
        gpu1_enc = timing.get('gpu1_combined_encode', 0)
        gpu0_dec = timing.get('gpu0_combined_decode', 0)
        gpu1_dec = timing.get('gpu1_combined_decode', 0)

        ok0 = r0.get('success', False)
        ok1 = r1.get('success', False)
        n_det0 = len(r0.get('detections', {}).get('objects', [])) if ok0 else -1
        n_det1 = len(r1.get('detections', {}).get('objects', [])) if ok1 else -1
        has_d0 = r0.get('depth') is not None if ok0 else False
        has_d1 = r1.get('depth') is not None if ok1 else False

        tag = 'WARM' if is_warmup else 'PERF'
        num = i + 1 if is_warmup else i - warmup_runs + 1

        # 网络分析字段
        g0_up = timing.get('gpu0_upload_kb', 0)
        g0_dn = timing.get('gpu0_download_kb', 0)
        g0_srv = timing.get('gpu0_server_total', 0)
        g1_up = timing.get('gpu1_upload_kb', 0)
        g1_dn = timing.get('gpu1_download_kb', 0)
        g1_srv = timing.get('gpu1_server_total', 0)
        g0_net = gpu0_http - g0_srv
        g1_net = gpu1_http - g1_srv

        print(f"  {tag} #{num:03d}: wall={wall_ms:.0f}ms | "
              f"GPU0[enc:{gpu0_enc:.0f} http:{gpu0_http:.0f}(srv:{g0_srv:.0f}+net:{g0_net:.0f}) "
              f"dec:{gpu0_dec:.0f} ↑{g0_up:.0f}KB↓{g0_dn:.0f}KB] "
              f"GPU1[enc:{gpu1_enc:.0f} http:{gpu1_http:.0f}(srv:{g1_srv:.0f}+net:{g1_net:.0f}) "
              f"dec:{gpu1_dec:.0f} ↑{g1_up:.0f}KB↓{g1_dn:.0f}KB]")

        if not is_warmup:
            times.append(wall)
            detail_times.append(timing)

    if not times:
        print("  No valid runs.")
        return

    # --- statistics ---
    sorted_times = sorted(times)
    n = len(sorted_times)
    avg_ms = statistics.mean(times) * 1000
    p50_ms = sorted_times[n // 2] * 1000
    p95_ms = sorted_times[min(int((n - 1) * 0.95), n - 1)] * 1000
    min_ms = sorted_times[0] * 1000
    max_ms = sorted_times[-1] * 1000

    print(f"\n{'='*70}")
    print(f"[Combined Dual] 统计 ({n} runs)")
    print(f"{'='*70}")
    print(f"  {'Metric':<8} {'avg':>7} {'P50':>7} {'P95':>7} {'min':>7} {'max':>7}")
    print(f"  {'─'*43}")
    print(f"  {'Wall':<8} {avg_ms:7.1f} {p50_ms:7.1f} {p95_ms:7.1f} {min_ms:7.1f} {max_ms:7.1f}")

    # Per-stage averages
    stage_keys = ['gpu0_combined_encode', 'gpu0_combined_http', 'gpu0_combined_decode',
                  'gpu1_combined_encode', 'gpu1_combined_http', 'gpu1_combined_decode']
    print(f"\n  分项耗时 (avg, ms):")
    for key in stage_keys:
        vals = [t.get(key, 0) for t in detail_times]
        avg_v = statistics.mean(vals)
        label = key.replace('combined_', '')
        print(f"    {label:<22}: {avg_v:6.1f}")

    # 网络分析
    net_keys = [
        ('gpu0_upload_kb',    'gpu0 upload (KB)'),
        ('gpu0_download_kb',  'gpu0 download (KB)'),
        ('gpu0_server_total', 'gpu0 server_total (ms)'),
        ('gpu1_upload_kb',    'gpu1 upload (KB)'),
        ('gpu1_download_kb',  'gpu1 download (KB)'),
        ('gpu1_server_total', 'gpu1 server_total (ms)'),
    ]
    print(f"\n  网络分析 (avg):")
    for key, label in net_keys:
        vals = [t.get(key, 0) for t in detail_times]
        avg_v = statistics.mean(vals) if vals else 0
        print(f"    {label:<26}: {avg_v:6.1f}")
    # 推算网络开销
    for gpu_label in ['gpu0', 'gpu1']:
        http_vals = [t.get(f'{gpu_label}_combined_http', 0) for t in detail_times]
        srv_vals = [t.get(f'{gpu_label}_server_total', 0) for t in detail_times]
        net_vals = [h - s for h, s in zip(http_vals, srv_vals)]
        avg_net = statistics.mean(net_vals) if net_vals else 0
        up_vals = [t.get(f'{gpu_label}_upload_kb', 0) for t in detail_times]
        dn_vals = [t.get(f'{gpu_label}_download_kb', 0) for t in detail_times]
        avg_up = statistics.mean(up_vals) if up_vals else 0
        avg_dn = statistics.mean(dn_vals) if dn_vals else 0
        print(f"    {gpu_label} net overhead     : {avg_net:6.1f} ms "
              f"(↑{avg_up:.0f}+↓{avg_dn:.0f} = {avg_up+avg_dn:.0f} KB)")

    hz = 1000.0 / p50_ms if p50_ms > 0 else 0
    target_ms = 200.0
    ok = p50_ms <= target_ms
    print(f"\n  频率 (P50): {hz:.2f} Hz")
    print(f"  5Hz 可达性: {'✅ 达标' if ok else '❌ 未达标'} "
          f"(P50={p50_ms:.1f}ms, 目标≤{target_ms:.0f}ms)")
    print(f"{'='*70}")


def compute_centroid_from_mask(depth_m, mask, bbox, intrinsics,
                               erode_kernel=5, iqr_factor=1.5,
                               depth_min=0.3, depth_max=10.0, min_pts=10):
    """用与 batch_camera_3d_percept 完全相同的算法计算单个物体的 3D 质心。

    Args:
        depth_m: 深度图 (H,W) float32 米
        mask: 二值 mask (H,W) uint8
        bbox: [x1, y1, x2, y2]
        intrinsics: {fx, fy, cx, cy}
        erode_kernel: mask 腐蚀核大小
        iqr_factor: IQR 异常值剔除因子
        depth_min/depth_max: 深度范围 (米)
        min_pts: 最少有效点数

    Returns:
        dict with centroid_optical (3D), depth_median, num_points, or None if invalid
    """
    H, W = depth_m.shape[:2]
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2]))
    y2 = min(H, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None

    mask_roi = mask[y1:y2, x1:x2]
    depth_roi = depth_m[y1:y2, x1:x2]

    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    eroded_roi = cv2.erode(mask_roi, kernel, iterations=1)
    if eroded_roi.sum() < min_pts:
        return None

    ys_local, xs_local = np.where(eroded_roi > 0)
    ds = depth_roi[ys_local, xs_local]
    valid = (ds > depth_min) & (ds < depth_max)
    if valid.sum() < min_pts:
        return None

    depth_values = ds[valid]
    q1, q3 = np.percentile(depth_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    valid2 = valid & (ds >= lower_bound) & (ds <= upper_bound)
    if valid2.sum() < min_pts:
        return None

    xs_img = xs_local[valid2].astype(np.float64) + x1
    ys_img = ys_local[valid2].astype(np.float64) + y1
    ds_valid = ds[valid2]

    X = (xs_img - cx) * ds_valid / fx
    Y = (ys_img - cy) * ds_valid / fy
    Z = ds_valid
    points = np.column_stack([X, Y, Z])
    centroid = np.median(points, axis=0)

    return {
        'centroid': centroid,
        'depth_median': float(np.median(ds_valid)),
        'depth_mean': float(np.mean(ds_valid)),
        'depth_std': float(np.std(ds_valid)),
        'num_points': int(valid2.sum()),
    }


def run_cdm_3d_compare(all_data, output_dir, text_prompt='pen,box,phone,bottle,toy',
                        cdm_url='http://192.168.112.14:8082',
                        sam3_url='http://192.168.112.14:8081'):
    """CDM 3D 质心偏差诊断：用真实 mask 对比 raw vs CDM 的 3D 定位差异。

    流程：SAM3 检测 → 解码 mask → 分别用 raw/CDM depth 计算 3D 质心 → 对比。
    使用与 pipeline (batch_camera_3d_percept) 完全相同的测量算法。
    """
    import pycocotools.mask as coco_mask

    print(f"\n{'='*70}")
    print(f"CDM 3D 质心偏差诊断")
    print(f"{'='*70}")

    # 初始化服务
    sam3_cfg = SimpleConfig(url=sam3_url, min_score=0.30, return_mask=True, warmup=0)
    sam3_service = SAM3Online(sam3_cfg)
    print(f"[SAM3] OK ({sam3_url})")

    cdm_cfg = SimpleConfig(url=cdm_url, chosen_policy='dn', warmup=0)
    cdm_service = DepthOptimizerOnline(cdm_cfg)
    print(f"[CDM] OK ({cdm_url})")

    # 典型 RealSense D435i @ 720x1280 内参 (用于相对比较，绝对值不影响 delta)
    intrinsics = {'fx': 640.0, 'fy': 640.0, 'cx': 640.0, 'cy': 360.0}
    print(f"[内参] fx={intrinsics['fx']}, fy={intrinsics['fy']}, "
          f"cx={intrinsics['cx']}, cy={intrinsics['cy']} (典型值，delta 不受影响)")

    all_deltas = []

    for data in all_data:
        name = data['name']
        rgb = data['rgb']
        raw_depth_mm = data['depth']  # uint16 mm

        print(f"\n--- {name} ---")

        # 1. SAM3 检测
        try:
            det_result = sam3_service.forward(text_prompt, rgb)
        except Exception as e:
            print(f"  [SAM3] 检测失败: {e}")
            continue

        objects = det_result.result.get('objects', [])
        if not objects:
            print(f"  [SAM3] 未检测到物体")
            continue

        # 解码 mask
        rle_masks = []
        obj_indices = []
        for i, obj in enumerate(objects):
            mask_rle = obj.get('mask')
            if mask_rle:
                rle_masks.append(mask_rle)
                obj_indices.append(i)

        if not rle_masks:
            print(f"  无可用 mask")
            continue

        masks_array = coco_mask.decode(rle_masks)
        if masks_array.ndim == 2:
            masks_array = masks_array[:, :, np.newaxis]

        decoded = {}
        for idx, obj_idx in enumerate(obj_indices):
            decoded[obj_idx] = masks_array[:, :, idx].astype(np.uint8)

        # 2. CDM 优化 (与 pipeline 相同: uint16 mm → forward → uint16 mm)
        cdm_result = cdm_service.forward(rgb, raw_depth_mm, chosen_policy='dn')
        if not cdm_result.get('success'):
            print(f"  [CDM] 失败: {cdm_result.get('error')}")
            continue

        cdm_depth_mm = cdm_result['depth']  # uint16 mm

        # 转 float32 m (与 pipeline 一致)
        raw_depth_m = raw_depth_mm.astype(np.float32) / 1000.0
        cdm_depth_m = cdm_depth_mm.astype(np.float32) / 1000.0

        print(f"  检测到 {len(objects)} 个物体, {len(rle_masks)} 个有 mask")
        print(f"  {'物体':<20} {'Raw Z(m)':<10} {'CDM Z(m)':<10} "
              f"{'ΔX(mm)':<9} {'ΔY(mm)':<9} {'ΔZ(mm)':<9} {'Δ3D(mm)':<9} {'点数':<6}")
        print(f"  {'─'*86}")

        for i, obj in enumerate(objects):
            mask = decoded.get(i)
            if mask is None or mask.sum() < 300:
                continue
            bbox = obj.get('bbox', [0, 0, 0, 0])
            category = obj.get('category', f'obj{i}')
            score = obj.get('score', 0)

            raw_meas = compute_centroid_from_mask(raw_depth_m, mask, bbox, intrinsics)
            cdm_meas = compute_centroid_from_mask(cdm_depth_m, mask, bbox, intrinsics)

            if raw_meas is None or cdm_meas is None:
                status = 'raw无效' if raw_meas is None else 'cdm无效'
                print(f"  {category}({score:.2f}){'':<10} {status}")
                continue

            rc = raw_meas['centroid']
            cc = cdm_meas['centroid']
            dx = (cc[0] - rc[0]) * 1000  # mm
            dy = (cc[1] - rc[1]) * 1000
            dz = (cc[2] - rc[2]) * 1000
            d3d = np.sqrt(dx**2 + dy**2 + dz**2)

            print(f"  {category}({score:.2f}){'':<8} "
                  f"{rc[2]:<10.4f} {cc[2]:<10.4f} "
                  f"{dx:<+9.1f} {dy:<+9.1f} {dz:<+9.1f} {d3d:<9.1f} "
                  f"{raw_meas['num_points']}")

            all_deltas.append({
                'sample': name, 'object': category, 'score': score,
                'raw_z': rc[2], 'cdm_z': cc[2],
                'dx_mm': dx, 'dy_mm': dy, 'dz_mm': dz, 'd3d_mm': d3d,
                'num_points': raw_meas['num_points'],
                'raw_depth_std': raw_meas['depth_std'],
                'cdm_depth_std': cdm_meas['depth_std'],
            })

    # 汇总
    if all_deltas:
        print(f"\n{'='*70}")
        print(f"汇总 ({len(all_deltas)} 个有效物体)")
        print(f"{'='*70}")

        dxs = [d['dx_mm'] for d in all_deltas]
        dys = [d['dy_mm'] for d in all_deltas]
        dzs = [d['dz_mm'] for d in all_deltas]
        d3ds = [d['d3d_mm'] for d in all_deltas]

        print(f"  ΔX: mean={np.mean(dxs):+.1f}mm, std={np.std(dxs):.1f}mm, "
              f"max_abs={max(abs(d) for d in dxs):.1f}mm")
        print(f"  ΔY: mean={np.mean(dys):+.1f}mm, std={np.std(dys):.1f}mm, "
              f"max_abs={max(abs(d) for d in dys):.1f}mm")
        print(f"  ΔZ: mean={np.mean(dzs):+.1f}mm, std={np.std(dzs):.1f}mm, "
              f"max_abs={max(abs(d) for d in dzs):.1f}mm")
        print(f"  3D: mean={np.mean(d3ds):.1f}mm, max={max(d3ds):.1f}mm")

        # 找出偏差最大的物体
        worst = max(all_deltas, key=lambda d: d['d3d_mm'])
        print(f"\n  最大偏差: {worst['sample']}/{worst['object']} "
              f"Δ3D={worst['d3d_mm']:.1f}mm "
              f"(ΔX={worst['dx_mm']:+.1f}, ΔY={worst['dy_mm']:+.1f}, ΔZ={worst['dz_mm']:+.1f})")

        # 按深度分段统计
        near = [d for d in all_deltas if d['raw_z'] < 1.0]
        mid = [d for d in all_deltas if 1.0 <= d['raw_z'] < 2.0]
        far = [d for d in all_deltas if d['raw_z'] >= 2.0]
        for label, group in [('近(<1m)', near), ('中(1-2m)', mid), ('远(>2m)', far)]:
            if group:
                g3d = [d['d3d_mm'] for d in group]
                gdz = [d['dz_mm'] for d in group]
                print(f"  {label}: n={len(group)}, "
                      f"ΔZ mean={np.mean(gdz):+.1f}mm, "
                      f"Δ3D mean={np.mean(g3d):.1f}mm max={max(g3d):.1f}mm")
    else:
        print("\n无有效测量数据")


def run_cdm_compare_test(all_data, output_dir, cdm_url='http://192.168.112.14:8082'):
    """遍历样本数据，调用 CDM，输出统计 + 可视化面板。

    Args:
        all_data: [{'name': '001', 'rgb': ndarray, 'depth': ndarray}, ...]
        output_dir: 输出目录
        cdm_url: CDM 服务地址
    """
    print(f"\n{'='*70}")
    print(f"CDM 深度偏差对比 (输出目录: {output_dir})")
    print(f"{'='*70}")

    cfg = SimpleConfig(url=cdm_url, chosen_policy='dn', warmup=0)
    cdm_service = DepthOptimizerOnline(cfg)
    print(f"[CDM] 服务初始化成功 ({cdm_url})")

    for data in all_data:
        name = data['name']
        rgb = data['rgb']
        raw_depth = data['depth']

        print(f"\n--- {name} ---")
        timing = {}
        result = cdm_service.forward(rgb, raw_depth, chosen_policy='dn', _timing=timing)

        if not result.get('success'):
            print(f"  [CDM] 调用失败: {result.get('error', 'unknown')}")
            continue

        cdm_depth = result['depth']
        timing_str = ' '.join(f'{k}={v:.0f}ms' for k, v in timing.items())
        print(f"  CDM 耗时: {timing_str}")

        stats = compute_cdm_stats(raw_depth, cdm_depth)
        print_cdm_stats(stats)

        panel = build_cdm_compare_panel(raw_depth, cdm_depth, stats)
        out_path = os.path.join(output_dir, f'{name}_cdm_compare.png')
        cv2.imwrite(out_path, panel)
        print(f"  面板已保存: {out_path}")

    print(f"\n全部完成，结果在: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Perception 服务 Benchmark 脚本 (ROS2)')
    parser.add_argument('--vis', action='store_true', help='运行可视化测试并输出到 result 目录')
    parser.add_argument('--benchmark', action='store_true', help='运行耗时 benchmark 测试')
    parser.add_argument('--cdm-compare', action='store_true', help='运行 CDM 深度偏差对比诊断')
    parser.add_argument('--cdm-3d', action='store_true', help='CDM 3D 质心偏差诊断 (SAM3 mask + raw/CDM depth 对比)')
    parser.add_argument('--num-runs', type=int, default=10, help='每组数据测试次数 (default: 10)')
    parser.add_argument('--warmup', type=int, default=3, help='预热次数 (default: 3)')
    parser.add_argument('--samples-dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'samples'),
                       help='样本数据目录')
    parser.add_argument('--grasp-config', type=str,
                       default='/data/workspace/MobileManipulator2/src/perception/config/server_grasp.json',
                       help='GraspAnything 服务器配置文件路径')
    parser.add_argument('--sam3-url', type=str, default='http://192.168.112.14:8081',
                       help='SAM3 service URL (default: http://192.168.112.14:8081)')
    parser.add_argument('--fs-url', type=str, default='http://192.168.112.14:8084',
                       help='FoundationStereo service URL (default: http://192.168.112.14:8084)')
    parser.add_argument('--data-dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '../../../data/robot_captures'),
                       help='Capture data directory (auto-discovers timestamped subdirs)')
    parser.add_argument('--text-prompt', type=str, default='pen,box,phone,bottle,toy',
                       help='SAM3 text prompt for parallel mode')
    parser.add_argument('--max-stereo-frames', type=int, default=20,
                       help='Stereo 测试最大帧数 (default: 20, 0=不限)')
    parser.add_argument('--combined-url-gpu0', type=str,
                       default='http://192.168.112.14:8090',
                       help='Combined server URL for GPU0')
    parser.add_argument('--combined-url-gpu1', type=str,
                       default='http://192.168.112.14:8091',
                       help='Combined server URL for GPU1')
    args = parser.parse_args()

    # 如果没有指定任何模式，默认运行 benchmark
    if not args.vis and not args.benchmark and not args.cdm_compare and not args.cdm_3d:
        args.benchmark = True

    num_runs = args.num_runs
    warmup_runs = args.warmup
    samples_dir = args.samples_dir
    grasp_server_config = args.grasp_config

    # 可用数据集
    datasets = ['001', '002', '666', '770']

    # 加载所有数据集
    all_data = []
    for ds in datasets:
        rgb_path = os.path.join(samples_dir, f'{ds}-rgb.jpg')
        depth_path = os.path.join(samples_dir, f'{ds}-dpt.png')

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"跳过数据集 {ds}: 文件不完整")
            continue

        rgb = cv2.imread(rgb_path)
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or depth_mm is None:
            print(f"跳过数据集 {ds}: 读取失败")
            continue

        all_data.append({'name': ds, 'rgb': rgb, 'depth': depth_mm})
        print(f"加载数据集 {ds}: RGB {rgb.shape}, Depth {depth_mm.shape}")

    if not all_data:
        print("没有可用的测试数据")
        sys.exit(1)

    print(f"\n共加载 {len(all_data)} 组数据")

    # 加载立体相机数据 (Stereo 测试需要)
    stereo_available = False
    camera_datasets = []  # list of (name, frames, intrinsics)
    if os.path.isdir(args.data_dir):
        subdirs = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        ])
        if subdirs:
            print(f"\n加载立体相机数据 ({args.data_dir})...")
            for subdir in subdirs:
                cap_dir = os.path.join(args.data_dir, subdir)
                try:
                    frames, intrinsics = load_camera_data(cap_dir)
                    camera_datasets.append((subdir, frames, intrinsics))
                except Exception as e:
                    print(f"  跳过 {subdir}: {e}")
            if camera_datasets:
                stereo_available = True
                print(f"  共加载 {len(camera_datasets)} 组相机数据")
        else:
            print(f"\n[Stereo] 跳过 - {args.data_dir} 下无子目录")
    else:
        print(f"\n[Stereo] 跳过 - 目录不存在: {args.data_dir}")

    text_prompt_dinox = 'pen.box.phone.bottle.toy'
    text_prompt_sam3 = 'pen,box,phone,bottle,toy'

    # ========== 可视化测试 ==========
    if args.vis:
        run_dir = ensure_result_dir()
        run_visualization_test(all_data, run_dir, text_prompt_dinox, text_prompt_sam3, grasp_server_config)

    # ========== CDM 深度偏差对比 ==========
    if args.cdm_compare:
        # samples_dir 来自用户参数，从它反推项目根目录
        project_root = os.path.normpath(os.path.join(samples_dir, '..', '..', '..'))
        output_dir = os.path.join(project_root, 'scripts', 'cdm_diag')
        os.makedirs(output_dir, exist_ok=True)
        run_cdm_compare_test(all_data, output_dir)

    # ========== CDM 3D 质心偏差诊断 ==========
    if args.cdm_3d:
        project_root = os.path.normpath(os.path.join(samples_dir, '..', '..', '..'))
        output_dir = os.path.join(project_root, 'scripts', 'cdm_diag')
        os.makedirs(output_dir, exist_ok=True)
        run_cdm_3d_compare(all_data, output_dir)

    # ========== Benchmark 测试 ==========
    if not args.benchmark:
        return

    print(f"\n每组测试 {num_runs} 次, 预热 {warmup_runs} 次")
    all_results = []

    # ========== 1. DinoX 测试 ==========
    try:
        cfg = SimpleConfig(
            url='http://192.168.112.14:10086',
            min_score=0.25,
            iou_threshold=0.5,
            warmup=0
        )
        dinox_service = DinoXDetectorOnline(cfg)

        args_list = [(d['name'], (text_prompt_dinox, d['rgb'])) for d in all_data]

        result = benchmark_with_timing(
            func=dinox_service.forward,
            args_list=args_list,
            kwargs={},
            name='DinoX',
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        all_results.append(result)
    except Exception as e:
        print(f"\n[DinoX] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 2. SAM3 测试 ==========
    try:
        cfg = SimpleConfig(
            url='http://192.168.112.14:8081',
            min_score=0.30,
            return_mask=False,
            warmup=0
        )
        sam3_service = SAM3Online(cfg)

        args_list = [(d['name'], (text_prompt_sam3, d['rgb'])) for d in all_data]

        result = benchmark_with_timing(
            func=sam3_service.forward,
            args_list=args_list,
            kwargs={},
            name='SAM3',
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        all_results.append(result)
    except Exception as e:
        print(f"\n[SAM3] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 3. GraspAnything 测试 ==========
    if grasp_server_config and os.path.exists(grasp_server_config):
        try:
            cfg = SimpleConfig(
                server_list=grasp_server_config,
                model_name='full',
                warmup=0
            )
            grasp_service = GraspAnythingOnline(cfg)

            args_list = [(d['name'], (d['rgb'],)) for d in all_data]

            result = benchmark_with_timing(
                func=grasp_service.forward,
                args_list=args_list,
                kwargs={'depth': None},
                name='GraspAnything',
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[GraspAnything] 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[GraspAnything] 跳过 - 配置文件不存在: {grasp_server_config}")

    # ========== 4. CDM 测试 ==========
    try:
        cfg = SimpleConfig(
            url='http://192.168.112.14:8082',
            chosen_policy='dn',
            warmup=0
        )
        cdm_service = DepthOptimizerOnline(cfg)

        args_list = [(d['name'], (d['rgb'], d['depth'])) for d in all_data]

        result = benchmark_with_timing(
            func=cdm_service.forward,
            args_list=args_list,
            kwargs={'chosen_policy': 'dn'},
            name='CDM',
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        all_results.append(result)
    except Exception as e:
        print(f"\n[CDM] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 5. Stereo (FoundationStereo) 测试 ==========
    fs_service = None
    if stereo_available:
        try:
            fs_service = DepthOptimizerOnline(SimpleConfig(fs_url=args.fs_url, warmup=0))

            stereo_args_list = []
            for cam_name, frames, intrinsics in camera_datasets:
                for f in frames:
                    stereo_args_list.append(
                        (f'{cam_name}:{f["idx"]}', (f['ir_left'], f['ir_right'], intrinsics)))

            if args.max_stereo_frames > 0 and len(stereo_args_list) > args.max_stereo_frames:
                print(f"  [Stereo] 帧数限制: {len(stereo_args_list)} → {args.max_stereo_frames}")
                stereo_args_list = stereo_args_list[:args.max_stereo_frames]

            # 固定 stereo 总测试次数 = 50 (num_runs × frames ≈ 50)
            stereo_total_target = 50
            stereo_num_runs = max(1, stereo_total_target // len(stereo_args_list))
            result = benchmark_with_timing(
                func=fs_service.forward_stereo,
                args_list=stereo_args_list,
                kwargs={},
                name='Stereo(FS)',
                num_runs=stereo_num_runs,
                warmup_runs=warmup_runs
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[Stereo] 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[Stereo] 跳过 - 无立体相机数据")

    # ========== 6. 并发 SAM3+CDM 双路测试 ==========
    cdm_concurrent_result = None
    try:
        # 复用已有的 SAM3 和 CDM 服务实例
        if 'sam3_service' not in dir():
            cfg = SimpleConfig(
                url='http://192.168.112.14:8081',
                min_score=0.30,
                return_mask=False,
                warmup=0
            )
            sam3_service = SAM3Online(cfg)

        if 'cdm_service' not in dir():
            cfg = SimpleConfig(
                url='http://192.168.112.14:8082',
                chosen_policy='dn',
                warmup=0
            )
            cdm_service = DepthOptimizerOnline(cfg)

        cdm_concurrent_result = benchmark_concurrent_sam3_cdm_2way(
            sam3_service=sam3_service,
            cdm_service=cdm_service,
            all_data=all_data,
            text_prompt=text_prompt_sam3,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )
    except Exception as e:
        print(f"\n[SAM3+CDM 双路] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 7. 并发 SAM3+Stereo 双路测试 ==========
    stereo_concurrent_result = None
    if stereo_available and fs_service is not None and len(camera_datasets) >= 2:
        try:
            # 复用已有的 SAM3 服务
            if 'sam3_service' not in dir():
                cfg = SimpleConfig(
                    url=args.sam3_url,
                    min_score=0.30,
                    return_mask=False,
                    warmup=0
                )
                sam3_service = SAM3Online(cfg)

            cam_a_name, cam_a_frames, cam_a_intrinsics = camera_datasets[0]
            cam_b_name, cam_b_frames, cam_b_intrinsics = camera_datasets[1]

            stereo_concurrent_result = benchmark_concurrent_sam3_stereo_2way(
                sam3_service=sam3_service,
                fs_service=fs_service,
                top_frames=cam_a_frames,
                top_intrinsics=cam_a_intrinsics,
                chassis_frames=cam_b_frames,
                chassis_intrinsics=cam_b_intrinsics,
                text_prompt=text_prompt_sam3,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
            )
        except Exception as e:
            print(f"\n[SAM3+Stereo 双路] 测试失败: {e}")
            import traceback
            traceback.print_exc()
    elif not stereo_available or len(camera_datasets) < 2:
        print(f"\n[SAM3+Stereo 双路] 跳过 - 需要至少2组相机数据 (当前 {len(camera_datasets)} 组)")

    # ========== 8. Combined Server 双 GPU 并行测试 ==========
    if args.combined_url_gpu0 and args.combined_url_gpu1:
        if stereo_available and len(camera_datasets) >= 2:
            try:
                cam_a_name, cam_a_frames, cam_a_intrinsics = camera_datasets[0]
                cam_b_name, cam_b_frames, cam_b_intrinsics = camera_datasets[1]

                benchmark_combined_dual(
                    url_gpu0=args.combined_url_gpu0,
                    url_gpu1=args.combined_url_gpu1,
                    top_frames=cam_a_frames,
                    top_intrinsics=cam_a_intrinsics,
                    chassis_frames=cam_b_frames,
                    chassis_intrinsics=cam_b_intrinsics,
                    text_prompt=text_prompt_sam3,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs,
                )
            except Exception as e:
                print(f"\n[Combined Dual] 测试失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[Combined Dual] 跳过 - 需要至少2组相机数据 (当前 {len(camera_datasets)} 组)")
    elif args.combined_url_gpu0 or args.combined_url_gpu1:
        print(f"\n[Combined Dual] 跳过 - 需要同时指定 --combined-url-gpu0 和 --combined-url-gpu1")

    # ========== 汇总对比 ==========
    if all_results:
        print(f"\n{'='*70}")
        print(f"串行执行性能汇总 ({len(all_data)} 组数据轮换测试)")
        print(f"{'='*70}")
        print(f"{'服务':<15} {'平均(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12} {'本地(ms)':<12} {'HTTP(ms)':<12}")
        print("-" * 75)

        total_local = 0
        total_http = 0
        total_time = 0

        for r in all_results:
            # 计算本地和HTTP时间
            local_time = 0
            http_time = 0
            for k, v in r['timing_stats'].items():
                avg_v = statistics.mean(v)
                if 'http' in k.lower():
                    http_time += avg_v
                else:
                    local_time += avg_v

            total_local += local_time
            total_http += http_time
            total_time += r['avg'] * 1000

            print(f"{r['name']:<15} {r['avg']*1000:<12.1f} {r['min']*1000:<12.1f} {r['max']*1000:<12.1f} {local_time:<12.1f} {http_time:<12.1f}")

        print("-" * 75)
        print(f"{'串行总计':<15} {total_time:<12.1f} {'':<12} {'':<12} {total_local:<12.1f} {total_http:<12.1f}")

        # 计算理论并行时间
        max_service_time = max(r['avg'] * 1000 for r in all_results)
        print(f"\n理论并行时间 (取最慢服务): {max_service_time:.1f} ms")
        print(f"串行 vs 并行加速比: {total_time / max_service_time:.2f}x")

    # 并发 vs 串行总结
    sam3_serial = next((r for r in all_results if r['name'] == 'SAM3'), None)
    cdm_serial = next((r for r in all_results if r['name'] == 'CDM'), None)
    stereo_serial = next((r for r in all_results if r['name'] == 'Stereo(FS)'), None)

    def print_concurrent_summary(label, conc_result, service_a, service_b, a_name, b_name):
        """打印并发测试 vs 串行对比"""
        print(f"\n{'='*70}")
        print(f"{label} vs 串行 对比")
        print(f"{'='*70}")

        if service_a and service_b:
            serial_total = (service_a['avg'] * 2 + service_b['avg'] * 2) * 1000
            print(f"  串行 2x{a_name}+2x{b_name}: {serial_total:.1f} ms")

        print(f"  并发实际:          {conc_result['avg']*1000:.1f} ms")
        print(f"  并发最小:          {conc_result['min']*1000:.1f} ms")
        print(f"  并发最大:          {conc_result['max']*1000:.1f} ms")

        if service_a and service_b:
            print(f"  加速比:            {serial_total / (conc_result['avg']*1000):.2f}x")

        print(f"\n  双相机融合频率:    {1/conc_result['avg']:.1f} Hz")
        target_hz = 10.0
        target_ms = 1000.0 / target_hz
        ok = conc_result['avg'] * 1000 < target_ms
        print(f"  {target_hz:.0f}Hz 可达性:        {'可达' if ok else '不可达'} "
              f"(需 <{target_ms:.0f}ms, 实际 {conc_result['avg']*1000:.1f}ms)")

    if cdm_concurrent_result:
        print_concurrent_summary('SAM3+CDM 双路并发', cdm_concurrent_result,
                                 sam3_serial, cdm_serial, 'SAM3', 'CDM')

    if stereo_concurrent_result:
        print_concurrent_summary('SAM3+Stereo 双路并发', stereo_concurrent_result,
                                 sam3_serial, stereo_serial, 'SAM3', 'Stereo')


if __name__ == '__main__':
    main()
