#!/usr/bin/env python3
"""
深度排查感知系统10Hz瓶颈

测量各环节耗时：
1. DINO-X API 响应时间
2. CDM 深度优化 API 响应时间
3. 并发调用性能
"""

import time
import sys
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
DINOX_URL = "http://192.168.112.14:10086/api/predict"
CDM_URL = "http://192.168.112.14:8086/api/predict"
NUM_TESTS = 10


def create_test_images():
    """创建测试图像"""
    # 1280x720 RGB 图像
    rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # 深度图 (uint16, mm)
    depth = np.random.randint(500, 3000, (720, 1280), dtype=np.uint16)
    return rgb, depth


def test_dinox_single(rgb, session):
    """单次DINO-X调用测量"""
    timing = {}

    t0 = time.time()

    # 编码
    _, img_encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_bytes = img_encoded.tobytes()
    t1 = time.time()
    timing['encode'] = (t1 - t0) * 1000

    # HTTP请求
    files = {'images': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'text_prompt': 'bottle.cup.box',
        'min_score': 0.25,
        'iou_threshold': 0.5,
        'chosen_policy': 'det'
    }

    try:
        response = session.post(DINOX_URL, files=files, data=data, timeout=30)
        t2 = time.time()
        timing['http'] = (t2 - t1) * 1000
        timing['total'] = (t2 - t0) * 1000
        timing['status'] = response.status_code
        timing['success'] = True
    except Exception as e:
        t2 = time.time()
        timing['http'] = (t2 - t1) * 1000
        timing['total'] = (t2 - t0) * 1000
        timing['error'] = str(e)
        timing['success'] = False

    return timing


def test_cdm_single(rgb, depth, session):
    """单次CDM调用测量"""
    timing = {}

    t0 = time.time()

    # 编码 RGB
    _, rgb_encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rgb_bytes = rgb_encoded.tobytes()
    t1 = time.time()
    timing['encode_rgb'] = (t1 - t0) * 1000

    # 编码 Depth
    _, depth_encoded = cv2.imencode('.png', depth)
    depth_bytes = depth_encoded.tobytes()
    t2 = time.time()
    timing['encode_depth'] = (t2 - t1) * 1000

    # HTTP请求
    files = {
        'rgb': ('rgb.jpg', rgb_bytes, 'image/jpeg'),
        'dpt': ('depth.png', depth_bytes, 'image/png'),
    }
    data = {'chosen_policy': 'dn'}

    try:
        response = session.post(CDM_URL, files=files, data=data, timeout=60)
        t3 = time.time()
        timing['http'] = (t3 - t2) * 1000
        timing['total'] = (t3 - t0) * 1000
        timing['status'] = response.status_code
        timing['success'] = True
    except Exception as e:
        t3 = time.time()
        timing['http'] = (t3 - t2) * 1000
        timing['total'] = (t3 - t0) * 1000
        timing['error'] = str(e)
        timing['success'] = False

    return timing


def test_parallel(rgb, depth, session):
    """测试并行调用 DINO-X + CDM"""
    timing = {}

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dinox = executor.submit(test_dinox_single, rgb, session)
        future_cdm = executor.submit(test_cdm_single, rgb, depth, session)

        dinox_result = future_dinox.result(timeout=30)
        cdm_result = future_cdm.result(timeout=60)

    t1 = time.time()

    timing['dinox'] = dinox_result
    timing['cdm'] = cdm_result
    timing['parallel_total'] = (t1 - t0) * 1000
    timing['sequential_total'] = dinox_result['total'] + cdm_result['total']

    return timing


def main():
    print("=" * 60)
    print("感知系统 10Hz 瓶颈深度排查")
    print("=" * 60)

    # 创建 session
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': None, 'https': None}

    # 创建测试图像
    print("\n创建测试图像...")
    rgb, depth = create_test_images()
    print(f"  RGB: {rgb.shape}, Depth: {depth.shape}")

    # 检查服务可用性
    print("\n检查服务可用性...")
    try:
        r = session.get(DINOX_URL.replace('/api/predict', '/health'), timeout=5)
        print(f"  DINO-X: OK")
    except:
        print(f"  DINO-X: 尝试直接测试...")

    try:
        r = session.get(CDM_URL.replace('/api/predict', '/health'), timeout=5)
        print(f"  CDM: OK")
    except:
        print(f"  CDM: 尝试直接测试...")

    # ==================== 测试 1: DINO-X 单独调用 ====================
    print("\n" + "=" * 60)
    print("测试 1: DINO-X API 响应时间")
    print("=" * 60)

    dinox_times = []
    for i in range(NUM_TESTS):
        result = test_dinox_single(rgb, session)
        dinox_times.append(result)
        status = "✓" if result['success'] else "✗"
        print(f"  [{i+1}/{NUM_TESTS}] {status} encode={result['encode']:.1f}ms, http={result['http']:.1f}ms, total={result['total']:.1f}ms")

    successful = [t for t in dinox_times if t['success']]
    if successful:
        avg_total = np.mean([t['total'] for t in successful])
        avg_http = np.mean([t['http'] for t in successful])
        min_total = np.min([t['total'] for t in successful])
        max_total = np.max([t['total'] for t in successful])
        print(f"\n  DINO-X 统计 (n={len(successful)}):")
        print(f"    平均: {avg_total:.1f}ms (HTTP: {avg_http:.1f}ms)")
        print(f"    范围: {min_total:.1f}ms ~ {max_total:.1f}ms")
        print(f"    理论最大频率: {1000/avg_total:.1f} Hz")

    # ==================== 测试 2: CDM 单独调用 ====================
    print("\n" + "=" * 60)
    print("测试 2: CDM API 响应时间")
    print("=" * 60)

    cdm_times = []
    for i in range(NUM_TESTS):
        result = test_cdm_single(rgb, depth, session)
        cdm_times.append(result)
        status = "✓" if result['success'] else "✗"
        print(f"  [{i+1}/{NUM_TESTS}] {status} encode={result['encode_rgb']:.1f}+{result['encode_depth']:.1f}ms, http={result['http']:.1f}ms, total={result['total']:.1f}ms")

    successful = [t for t in cdm_times if t['success']]
    if successful:
        avg_total = np.mean([t['total'] for t in successful])
        avg_http = np.mean([t['http'] for t in successful])
        min_total = np.min([t['total'] for t in successful])
        max_total = np.max([t['total'] for t in successful])
        print(f"\n  CDM 统计 (n={len(successful)}):")
        print(f"    平均: {avg_total:.1f}ms (HTTP: {avg_http:.1f}ms)")
        print(f"    范围: {min_total:.1f}ms ~ {max_total:.1f}ms")
        print(f"    理论最大频率: {1000/avg_total:.1f} Hz")

    # ==================== 测试 3: 并行调用 ====================
    print("\n" + "=" * 60)
    print("测试 3: DINO-X + CDM 并行调用")
    print("=" * 60)

    parallel_times = []
    for i in range(NUM_TESTS):
        result = test_parallel(rgb, depth, session)
        parallel_times.append(result)
        print(f"  [{i+1}/{NUM_TESTS}] parallel={result['parallel_total']:.1f}ms (dinox={result['dinox']['total']:.1f}ms, cdm={result['cdm']['total']:.1f}ms)")

    avg_parallel = np.mean([t['parallel_total'] for t in parallel_times])
    avg_sequential = np.mean([t['sequential_total'] for t in parallel_times])
    min_parallel = np.min([t['parallel_total'] for t in parallel_times])
    max_parallel = np.max([t['parallel_total'] for t in parallel_times])

    print(f"\n  并行调用统计 (n={NUM_TESTS}):")
    print(f"    并行平均: {avg_parallel:.1f}ms")
    print(f"    串行平均: {avg_sequential:.1f}ms")
    print(f"    加速比: {avg_sequential/avg_parallel:.2f}x")
    print(f"    范围: {min_parallel:.1f}ms ~ {max_parallel:.1f}ms")
    print(f"    理论最大频率 (单相机): {1000/avg_parallel:.1f} Hz")

    # ==================== 测试 4: 双相机并发场景 ====================
    print("\n" + "=" * 60)
    print("测试 4: 双相机并发场景 (4个API同时调用)")
    print("=" * 60)

    dual_times = []
    for i in range(NUM_TESTS):
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            # 模拟双相机: 2x DINO-X + 2x CDM
            futures = [
                executor.submit(test_dinox_single, rgb, session),
                executor.submit(test_dinox_single, rgb, session),
                executor.submit(test_cdm_single, rgb, depth, session),
                executor.submit(test_cdm_single, rgb, depth, session),
            ]
            results = [f.result(timeout=60) for f in futures]

        t1 = time.time()
        total = (t1 - t0) * 1000
        dual_times.append(total)

        dinox1, dinox2, cdm1, cdm2 = results[0]['total'], results[1]['total'], results[2]['total'], results[3]['total']
        print(f"  [{i+1}/{NUM_TESTS}] total={total:.1f}ms (dinox: {dinox1:.0f}/{dinox2:.0f}ms, cdm: {cdm1:.0f}/{cdm2:.0f}ms)")

    avg_dual = np.mean(dual_times)
    min_dual = np.min(dual_times)
    max_dual = np.max(dual_times)

    print(f"\n  双相机并发统计 (n={NUM_TESTS}):")
    print(f"    平均: {avg_dual:.1f}ms")
    print(f"    范围: {min_dual:.1f}ms ~ {max_dual:.1f}ms")
    print(f"    理论最大融合频率: {1000/avg_dual:.1f} Hz")

    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("深度排查总结")
    print("=" * 60)

    dinox_avg = np.mean([t['total'] for t in dinox_times if t['success']])
    cdm_avg = np.mean([t['total'] for t in cdm_times if t['success']])

    print(f"\n单次API调用耗时:")
    print(f"  DINO-X: {dinox_avg:.1f}ms")
    print(f"  CDM:    {cdm_avg:.1f}ms")

    print(f"\n10Hz 需求分析 (100ms/帧):")
    print(f"  单相机并行 (DINO-X+CDM): {avg_parallel:.1f}ms {'✓ 可达10Hz' if avg_parallel < 100 else '✗ 无法达10Hz'}")
    print(f"  双相机并发 (4个API): {avg_dual:.1f}ms {'✓ 可达10Hz' if avg_dual < 100 else '✗ 无法达10Hz'}")

    print(f"\n瓶颈分析:")
    if avg_parallel > 100:
        slower = "DINO-X" if dinox_avg > cdm_avg else "CDM"
        print(f"  主要瓶颈: {slower} 服务响应时间")
        print(f"  需要 {slower} 响应时间 < {100 - min(dinox_avg, cdm_avg):.0f}ms 才能达到 10Hz")
    else:
        print(f"  单相机: 理论可达 10Hz")
        if avg_dual > 100:
            print(f"  双相机并发: 资源竞争导致性能下降")
            print(f"  建议: 降低频率或使用相位偏移")


if __name__ == '__main__':
    main()
