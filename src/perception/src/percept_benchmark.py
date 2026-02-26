#!/usr/bin/env python3
"""
percept.py 服务性能评测脚本
测试 DinoX, GraspAnything, CDM 三个服务的串行执行时间分布
"""

import sys
import os
import time
import statistics

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/agilex/MobileManipulator/src/perception/src')

import cv2
import numpy as np
from mmengine.config import Config as MMConfig
from percept import DinoXDetectorOnline, GraspAnythingOnline, DepthOptimizerOnline


def benchmark_with_timing(func, args, kwargs, name, num_runs=10):
    """带详细时间分布的基准测试"""
    times = []
    timing_stats = {}

    print(f"\n{'='*70}")
    print(f"[{name}] 开始测试 ({num_runs} 次)")
    print(f"{'='*70}")

    for i in range(num_runs):
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
        print(f"  第 {i+1:2d} 次: {elapsed*1000:7.1f}ms | {timing_str}")

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


def main():
    num_runs = 10
    warmup_runs = 3

    # 测试图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, './../samples/', 'dino_test.jpg')

    if not os.path.exists(img_path):
        print(f"测试图片不存在: {img_path}")
        sys.exit(1)

    # 加载测试图片
    rgb = cv2.imread(img_path)
    if rgb is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    rgb = cv2.resize(rgb, dsize=(1280, 720))
    # 生成模拟深度图 (用于 CDM 测试)
    depth_mm = np.random.randint(500, 2000, (720, 1280), dtype=np.uint16)

    print(f"测试图片: {img_path}")
    print(f"图片尺寸: {rgb.shape}")
    print(f"深度尺寸: {depth_mm.shape} (模拟数据)")
    print(f"测试次数: {num_runs}")
    print(f"预热次数: {warmup_runs}")

    all_results = []

    # ========== 1. DinoX 测试 ==========
    try:
        cfg = MMConfig()
        cfg.url = 'http://192.168.112.14:10086'
        cfg.min_score = 0.25
        cfg.iou_threshold = 0.5
        cfg.warmup = warmup_runs
        dinox_service = DinoXDetectorOnline(cfg)

        text_prompt = 'pen.box.phone.bottle.toy'

        result = benchmark_with_timing(
            func=dinox_service.forward,
            args=(text_prompt, rgb),
            kwargs={},
            name='DinoX',
            num_runs=num_runs
        )
        all_results.append(result)
    except Exception as e:
        print(f"\n[DinoX] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 2. GraspAnything 测试 ==========
    try:
        cfg = MMConfig()
        cfg.server_list = os.path.join(script_dir, 'config', 'server_grasp.json')
        cfg.model_name = 'full'
        cfg.warmup = warmup_runs
        grasp_service = GraspAnythingOnline(cfg)

        result = benchmark_with_timing(
            func=grasp_service.forward,
            args=(rgb,),
            kwargs={'depth': None},
            name='GraspAnything',
            num_runs=num_runs
        )
        all_results.append(result)
    except Exception as e:
        print(f"\n[GraspAnything] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 3. CDM 测试 ==========
    try:
        cfg = MMConfig()
        cfg.url = 'http://192.168.112.14:8086'
        cfg.chosen_policy = 'dn'
        cfg.warmup = warmup_runs
        cdm_service = DepthOptimizerOnline(cfg)

        result = benchmark_with_timing(
            func=cdm_service.forward,
            args=(rgb, depth_mm),
            kwargs={'chosen_policy': 'dn'},
            name='CDM',
            num_runs=num_runs
        )
        all_results.append(result)
    except Exception as e:
        print(f"\n[CDM] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 汇总对比 ==========
    if all_results:
        print(f"\n{'='*70}")
        print("串行执行性能汇总")
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


if __name__ == '__main__':
    main()
