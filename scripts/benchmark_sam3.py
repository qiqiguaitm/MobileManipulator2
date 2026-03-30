#!/usr/bin/env python3
"""
SAM3 + CDM 频率测试
"""

import time
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

# 配置
SAM3_URL = "http://192.168.112.14:8080/api/predict"
CDM_URL = "http://192.168.112.14:8086/api/predict"
NUM_TESTS = 10


def create_test_images():
    """创建测试图像"""
    rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    depth = np.random.randint(500, 3000, (720, 1280), dtype=np.uint16)
    return rgb, depth


def test_sam3_single(rgb, session):
    """单次SAM3调用测量"""
    timing = {}
    t0 = time.time()

    # 编码
    _, img_encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_bytes = img_encoded.tobytes()
    t1 = time.time()
    timing['encode'] = (t1 - t0) * 1000

    # HTTP请求 (与 percept.py 中 SAM3Online 保持一致)
    files = {'images': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'text_prompt': 'bottle.cup.box.barrel.toy.cabinet',
        'confidence': '0.25',
        'return_mask': 'true',
        'tiled': 'false'
    }

    try:
        response = session.post(SAM3_URL, files=files, data=data, timeout=30)
        t2 = time.time()
        timing['http'] = (t2 - t1) * 1000
        timing['total'] = (t2 - t0) * 1000
        timing['status'] = response.status_code
        timing['success'] = response.status_code == 200

        if timing['success']:
            result = response.json()
            timing['num_objects'] = len(result.get('objects', []))
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
        timing['success'] = response.status_code == 200
    except Exception as e:
        t3 = time.time()
        timing['http'] = (t3 - t2) * 1000
        timing['total'] = (t3 - t0) * 1000
        timing['error'] = str(e)
        timing['success'] = False

    return timing


def test_parallel(rgb, depth, session):
    """测试并行调用 SAM3 + CDM"""
    timing = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_sam3 = executor.submit(test_sam3_single, rgb, session)
        future_cdm = executor.submit(test_cdm_single, rgb, depth, session)

        sam3_result = future_sam3.result(timeout=30)
        cdm_result = future_cdm.result(timeout=60)

    t1 = time.time()

    timing['sam3'] = sam3_result
    timing['cdm'] = cdm_result
    timing['parallel_total'] = (t1 - t0) * 1000
    timing['sequential_total'] = sam3_result['total'] + cdm_result['total']

    return timing


def main():
    print("=" * 60)
    print("SAM3 + CDM 频率测试")
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
        r = session.get(SAM3_URL.replace('/api/predict', '/api/health'), timeout=5)
        print(f"  SAM3: {r.json().get('status', 'OK')}")
    except Exception as e:
        print(f"  SAM3: 尝试直接测试... ({e})")

    try:
        r = session.get(CDM_URL.replace('/api/predict', '/health'), timeout=5)
        print(f"  CDM: OK")
    except:
        print(f"  CDM: 尝试直接测试...")

    # ==================== 测试 1: SAM3 单独调用 ====================
    print("\n" + "=" * 60)
    print("测试 1: SAM3 API 响应时间")
    print("=" * 60)

    sam3_times = []
    for i in range(NUM_TESTS):
        result = test_sam3_single(rgb, session)
        sam3_times.append(result)
        status = "✓" if result['success'] else "✗"
        objs = result.get('num_objects', 'N/A')
        print(f"  [{i+1}/{NUM_TESTS}] {status} encode={result['encode']:.1f}ms, http={result['http']:.1f}ms, total={result['total']:.1f}ms, objects={objs}")

    successful = [t for t in sam3_times if t['success']]
    if successful:
        avg_total = np.mean([t['total'] for t in successful])
        avg_http = np.mean([t['http'] for t in successful])
        min_total = np.min([t['total'] for t in successful])
        max_total = np.max([t['total'] for t in successful])
        print(f"\n  SAM3 统计 (n={len(successful)}):")
        print(f"    平均: {avg_total:.1f}ms (HTTP: {avg_http:.1f}ms)")
        print(f"    范围: {min_total:.1f}ms ~ {max_total:.1f}ms")
        print(f"    理论最大频率: {1000/avg_total:.1f} Hz")
    else:
        print("\n  SAM3 服务不可用或全部失败")
        return

    # ==================== 测试 2: CDM 单独调用 ====================
    print("\n" + "=" * 60)
    print("测试 2: CDM API 响应时间 (与DINO-X测试对照)")
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
        print(f"\n  CDM 统计 (n={len(successful)}):")
        print(f"    平均: {avg_total:.1f}ms (HTTP: {avg_http:.1f}ms)")
        print(f"    理论最大频率: {1000/avg_total:.1f} Hz")

    # ==================== 测试 3: SAM3 + CDM 并行调用 ====================
    print("\n" + "=" * 60)
    print("测试 3: SAM3 + CDM 并行调用")
    print("=" * 60)

    parallel_times = []
    for i in range(NUM_TESTS):
        result = test_parallel(rgb, depth, session)
        parallel_times.append(result)
        print(f"  [{i+1}/{NUM_TESTS}] parallel={result['parallel_total']:.1f}ms (sam3={result['sam3']['total']:.1f}ms, cdm={result['cdm']['total']:.1f}ms)")

    avg_parallel = np.mean([t['parallel_total'] for t in parallel_times])
    avg_sequential = np.mean([t['sequential_total'] for t in parallel_times])

    print(f"\n  并行调用统计 (n={NUM_TESTS}):")
    print(f"    并行平均: {avg_parallel:.1f}ms")
    print(f"    串行平均: {avg_sequential:.1f}ms")
    print(f"    加速比: {avg_sequential/avg_parallel:.2f}x")
    print(f"    理论最大频率 (单相机): {1000/avg_parallel:.1f} Hz")

    # ==================== 测试 4: 双相机并发场景 ====================
    print("\n" + "=" * 60)
    print("测试 4: 双相机并发场景 (4个API同时调用)")
    print("=" * 60)

    dual_times = []
    for i in range(NUM_TESTS):
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(test_sam3_single, rgb, session),
                executor.submit(test_sam3_single, rgb, session),
                executor.submit(test_cdm_single, rgb, depth, session),
                executor.submit(test_cdm_single, rgb, depth, session),
            ]
            results = [f.result(timeout=60) for f in futures]

        t1 = time.time()
        total = (t1 - t0) * 1000
        dual_times.append(total)

        sam3_1, sam3_2, cdm1, cdm2 = results[0]['total'], results[1]['total'], results[2]['total'], results[3]['total']
        print(f"  [{i+1}/{NUM_TESTS}] total={total:.1f}ms (sam3: {sam3_1:.0f}/{sam3_2:.0f}ms, cdm: {cdm1:.0f}/{cdm2:.0f}ms)")

    avg_dual = np.mean(dual_times)

    print(f"\n  双相机并发统计 (n={NUM_TESTS}):")
    print(f"    平均: {avg_dual:.1f}ms")
    print(f"    理论最大融合频率: {1000/avg_dual:.1f} Hz")

    # ==================== 对比总结 ====================
    print("\n" + "=" * 60)
    print("SAM3 vs DINO-X 对比总结")
    print("=" * 60)

    sam3_avg = np.mean([t['total'] for t in sam3_times if t['success']])
    cdm_avg = np.mean([t['total'] for t in cdm_times if t['success']])

    print(f"\n单次API调用耗时:")
    print(f"  SAM3:   {sam3_avg:.1f}ms (DINO-X 约 230ms)")
    print(f"  CDM:    {cdm_avg:.1f}ms")

    print(f"\n理论最大频率:")
    print(f"  SAM3 单独: {1000/sam3_avg:.1f} Hz")
    print(f"  SAM3 + CDM 并行: {1000/avg_parallel:.1f} Hz")
    print(f"  双相机并发: {1000/avg_dual:.1f} Hz")

    print(f"\n10Hz 可达性:")
    print(f"  SAM3 单独: {'✓' if sam3_avg < 100 else '✗'}")
    print(f"  SAM3 + CDM: {'✓' if avg_parallel < 100 else '✗'}")
    print(f"  双相机并发: {'✓' if avg_dual < 100 else '✗'}")


if __name__ == '__main__':
    main()
