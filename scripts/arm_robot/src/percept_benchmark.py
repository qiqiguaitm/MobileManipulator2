#!/usr/bin/env python3
"""
percept.py 服务性能评测脚本
测试 DinoX, GraspAnything, CDM 三个服务的串行和并行执行时间分布
"""

import sys
import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/agilex/MobileManipulator/src/perception/src')

import cv2
import numpy as np
from mmengine.config import Config as MMConfig
from percept import DinoXDetectorOnline, GraspAnythingOnline, DepthOptimizerOnline


def benchmark_serial(services, rgb, depth_mm, text_prompt, num_runs=10):
    """串行执行测试"""
    print(f"\n{'='*70}")
    print("【串行测试】依次执行 DinoX → GraspAnything → CDM")
    print(f"{'='*70}")

    all_times = []
    service_times = {'dinox': [], 'grasp': [], 'cdm': []}
    timing_stats = {}

    for i in range(num_runs):
        timing = {}
        run_times = {}

        # DinoX
        t0 = time.perf_counter()
        services['dinox'].forward(text_prompt, rgb, _timing=timing)
        run_times['dinox'] = (time.perf_counter() - t0) * 1000

        # GraspAnything
        t0 = time.perf_counter()
        services['grasp'].forward(rgb, depth=None, _timing=timing)
        run_times['grasp'] = (time.perf_counter() - t0) * 1000

        # CDM
        t0 = time.perf_counter()
        services['cdm'].forward(rgb, depth_mm, chosen_policy='dn', _timing=timing)
        run_times['cdm'] = (time.perf_counter() - t0) * 1000

        total = run_times['dinox'] + run_times['grasp'] + run_times['cdm']
        all_times.append(total)

        for k, v in run_times.items():
            service_times[k].append(v)

        for k, v in timing.items():
            if k not in timing_stats:
                timing_stats[k] = []
            timing_stats[k].append(v)

        print(f"  第{i+1:2d}次: 总={total:6.1f}ms | dinox={run_times['dinox']:.0f} | grasp={run_times['grasp']:.0f} | cdm={run_times['cdm']:.0f}")

    # 统计
    print(f"\n【串行统计】")
    print(f"  总耗时: 平均={statistics.mean(all_times):.1f}ms  最小={min(all_times):.1f}ms  最大={max(all_times):.1f}ms  σ={statistics.stdev(all_times):.1f}ms")
    print(f"  各服务平均: dinox={statistics.mean(service_times['dinox']):.1f}ms  grasp={statistics.mean(service_times['grasp']):.1f}ms  cdm={statistics.mean(service_times['cdm']):.1f}ms")

    # 时间分布
    print(f"\n  详细时间分布:")
    local_total = 0
    http_total = 0
    for k, v in sorted(timing_stats.items()):
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
        'mode': 'serial',
        'avg': statistics.mean(all_times),
        'min': min(all_times),
        'max': max(all_times),
        'std': statistics.stdev(all_times),
        'service_times': {k: statistics.mean(v) for k, v in service_times.items()},
        'local_total': local_total,
        'http_total': http_total
    }


def benchmark_parallel(services, rgb, depth_mm, text_prompt, num_runs=10):
    """并行执行测试 (类似ROS节点的ThreadPoolExecutor方式)"""
    print(f"\n{'='*70}")
    print("【并行测试】ThreadPoolExecutor 同时执行 DinoX + GraspAnything + CDM")
    print(f"{'='*70}")

    all_times = []
    service_times = {'dinox': [], 'grasp': [], 'cdm': []}
    timing_stats = {}

    # 创建持久线程池
    executor = ThreadPoolExecutor(max_workers=3)

    for i in range(num_runs):
        timing = {}
        run_times = {}

        t_start = time.perf_counter()

        # 并行提交三个任务
        futures = {}

        def call_dinox():
            t0 = time.perf_counter()
            result = services['dinox'].forward(text_prompt, rgb, _timing=timing)
            return ('dinox', (time.perf_counter() - t0) * 1000, result)

        def call_grasp():
            t0 = time.perf_counter()
            result = services['grasp'].forward(rgb, depth=None, _timing=timing)
            return ('grasp', (time.perf_counter() - t0) * 1000, result)

        def call_cdm():
            t0 = time.perf_counter()
            result = services['cdm'].forward(rgb, depth_mm, chosen_policy='dn', _timing=timing)
            return ('cdm', (time.perf_counter() - t0) * 1000, result)

        futures['dinox'] = executor.submit(call_dinox)
        futures['grasp'] = executor.submit(call_grasp)
        futures['cdm'] = executor.submit(call_cdm)

        # 等待所有完成
        for name, future in futures.items():
            try:
                svc_name, svc_time, _ = future.result(timeout=30)
                run_times[svc_name] = svc_time
            except Exception as e:
                print(f"    {name} 失败: {e}")
                run_times[name] = 0

        total = (time.perf_counter() - t_start) * 1000
        all_times.append(total)

        for k, v in run_times.items():
            service_times[k].append(v)

        for k, v in timing.items():
            if k not in timing_stats:
                timing_stats[k] = []
            timing_stats[k].append(v)

        print(f"  第{i+1:2d}次: 总={total:6.1f}ms | dinox={run_times.get('dinox', 0):.0f} | grasp={run_times.get('grasp', 0):.0f} | cdm={run_times.get('cdm', 0):.0f}")

    executor.shutdown(wait=False)

    # 统计
    print(f"\n【并行统计】")
    print(f"  总耗时: 平均={statistics.mean(all_times):.1f}ms  最小={min(all_times):.1f}ms  最大={max(all_times):.1f}ms  σ={statistics.stdev(all_times):.1f}ms")
    print(f"  各服务平均: dinox={statistics.mean(service_times['dinox']):.1f}ms  grasp={statistics.mean(service_times['grasp']):.1f}ms  cdm={statistics.mean(service_times['cdm']):.1f}ms")

    # 时间分布
    print(f"\n  详细时间分布:")
    local_total = 0
    http_total = 0
    for k, v in sorted(timing_stats.items()):
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
        'mode': 'parallel',
        'avg': statistics.mean(all_times),
        'min': min(all_times),
        'max': max(all_times),
        'std': statistics.stdev(all_times),
        'service_times': {k: statistics.mean(v) for k, v in service_times.items()},
        'local_total': local_total,
        'http_total': http_total
    }


def main():
    num_runs = 10
    warmup_runs = 3

    # 测试图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'samples', 'dino_test.jpg')

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

    print(f"{'='*70}")
    print("感知服务性能评测 - 串行 vs 并行对比")
    print(f"{'='*70}")
    print(f"测试图片: {img_path}")
    print(f"图片尺寸: {rgb.shape}")
    print(f"深度尺寸: {depth_mm.shape} (模拟数据)")
    print(f"测试次数: {num_runs}")
    print(f"预热次数: {warmup_runs}")

    text_prompt = 'pen.box.phone.bottle.toy'
    print(f"检测提示: {text_prompt}")

    # 初始化服务
    print(f"\n初始化服务...")
    services = {}

    try:
        cfg = MMConfig()
        cfg.url = 'http://192.168.112.14:10086'
        cfg.min_score = 0.25
        cfg.iou_threshold = 0.5
        cfg.warmup = warmup_runs
        services['dinox'] = DinoXDetectorOnline(cfg)
    except Exception as e:
        print(f"DinoX 初始化失败: {e}")
        sys.exit(1)

    try:
        cfg = MMConfig()
        cfg.server_list = os.path.join(script_dir, 'config', 'server_grasp.json')
        cfg.model_name = 'full'
        cfg.warmup = warmup_runs
        services['grasp'] = GraspAnythingOnline(cfg)
    except Exception as e:
        print(f"GraspAnything 初始化失败: {e}")
        sys.exit(1)

    try:
        cfg = MMConfig()
        cfg.url = 'http://192.168.112.14:8086'
        cfg.chosen_policy = 'dn'
        cfg.warmup = warmup_runs
        services['cdm'] = DepthOptimizerOnline(cfg)
    except Exception as e:
        print(f"CDM 初始化失败: {e}")
        sys.exit(1)

    print(f"服务初始化完成")

    # 执行测试
    results = []

    # 1. 串行测试
    try:
        serial_result = benchmark_serial(services, rgb, depth_mm, text_prompt, num_runs)
        results.append(serial_result)
    except Exception as e:
        print(f"串行测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 2. 并行测试
    try:
        parallel_result = benchmark_parallel(services, rgb, depth_mm, text_prompt, num_runs)
        results.append(parallel_result)
    except Exception as e:
        print(f"并行测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 汇总对比
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("【性能对比汇总】")
        print(f"{'='*70}")

        serial = results[0]
        parallel = results[1]

        print(f"\n{'模式':<10} {'总耗时(ms)':<12} {'DinoX(ms)':<12} {'Grasp(ms)':<12} {'CDM(ms)':<12}")
        print("-" * 58)
        print(f"{'串行':<10} {serial['avg']:<12.1f} {serial['service_times']['dinox']:<12.1f} {serial['service_times']['grasp']:<12.1f} {serial['service_times']['cdm']:<12.1f}")
        print(f"{'并行':<10} {parallel['avg']:<12.1f} {parallel['service_times']['dinox']:<12.1f} {parallel['service_times']['grasp']:<12.1f} {parallel['service_times']['cdm']:<12.1f}")

        speedup = serial['avg'] / parallel['avg']
        print(f"\n加速比: {speedup:.2f}x (串行{serial['avg']:.1f}ms → 并行{parallel['avg']:.1f}ms)")

        # 理论分析
        max_serial_service = max(serial['service_times'].values())
        print(f"\n理论分析:")
        print(f"  串行最慢服务: {max_serial_service:.1f}ms")
        print(f"  并行实际耗时: {parallel['avg']:.1f}ms")
        print(f"  并行额外开销: {parallel['avg'] - max_serial_service:.1f}ms ({(parallel['avg'] - max_serial_service) / max_serial_service * 100:.1f}%)")

        # 服务并行影响分析
        print(f"\n服务并行影响 (并行耗时 vs 串行耗时):")
        for svc in ['dinox', 'grasp', 'cdm']:
            s_time = serial['service_times'][svc]
            p_time = parallel['service_times'][svc]
            diff = p_time - s_time
            print(f"  {svc:<10}: {s_time:.1f}ms → {p_time:.1f}ms ({'+' if diff > 0 else ''}{diff:.1f}ms, {'+' if diff > 0 else ''}{diff/s_time*100:.1f}%)")


if __name__ == '__main__':
    main()
