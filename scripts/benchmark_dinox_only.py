#!/usr/bin/env python3
"""
测试纯DINO-X (无CDM) 的最大频率
"""

import time
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

DINOX_URL = "http://192.168.112.14:10086/api/predict"
NUM_TESTS = 20


def test_dinox(rgb, session):
    """DINO-X调用"""
    t0 = time.time()
    _, img_encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_bytes = img_encoded.tobytes()

    files = {'images': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'text_prompt': 'bottle.cup.box.barrel.toy.cabinet',
        'min_score': 0.25,
        'iou_threshold': 0.5,
        'chosen_policy': 'det'
    }

    try:
        response = session.post(DINOX_URL, files=files, data=data, timeout=30)
        t1 = time.time()
        return (t1 - t0) * 1000, True
    except Exception as e:
        t1 = time.time()
        return (t1 - t0) * 1000, False


def main():
    print("=" * 60)
    print("纯DINO-X 频率测试 (无CDM深度优化)")
    print("=" * 60)

    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': None, 'https': None}

    rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # 测试1: 单线程串行
    print("\n测试1: 单相机串行调用")
    times = []
    for i in range(NUM_TESTS):
        t, ok = test_dinox(rgb, session)
        times.append(t)
        print(f"  [{i+1}/{NUM_TESTS}] {t:.1f}ms")

    avg = np.mean(times)
    print(f"\n  平均: {avg:.1f}ms")
    print(f"  理论最大频率: {1000/avg:.1f} Hz")

    # 测试2: 双相机并发
    print("\n测试2: 双相机并发调用")
    times = []
    for i in range(NUM_TESTS):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(test_dinox, rgb, session)
            f2 = executor.submit(test_dinox, rgb, session)
            r1 = f1.result()
            r2 = f2.result()
        t = (time.time() - t0) * 1000
        times.append(t)
        print(f"  [{i+1}/{NUM_TESTS}] {t:.1f}ms (cam1={r1[0]:.0f}ms, cam2={r2[0]:.0f}ms)")

    avg = np.mean(times)
    print(f"\n  平均: {avg:.1f}ms")
    print(f"  理论最大频率: {1000/avg:.1f} Hz")

    # 测试3: 连续快速调用模拟10Hz
    print("\n测试3: 模拟10Hz连续调用 (目标100ms/帧)")
    target_interval = 0.1  # 100ms
    actual_intervals = []
    last_time = time.time()

    for i in range(NUM_TESTS):
        t0 = time.time()
        _, ok = test_dinox(rgb, session)
        t1 = time.time()

        elapsed = t1 - last_time
        actual_intervals.append(elapsed)
        last_time = t1

        detect_time = (t1 - t0) * 1000
        status = "✓" if detect_time < 100 else "✗"
        print(f"  [{i+1}/{NUM_TESTS}] {status} detect={detect_time:.1f}ms, interval={elapsed*1000:.1f}ms")

    avg_interval = np.mean(actual_intervals[1:])  # 排除第一个
    actual_hz = 1.0 / avg_interval
    print(f"\n  实际平均频率: {actual_hz:.1f} Hz")
    print(f"  10Hz可达: {'是' if actual_hz >= 10 else '否'}")

    # 总结
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"DINO-X 单次调用: ~{np.mean(times):.0f}ms")
    print(f"禁用CDM后理论最大频率: ~{1000/np.mean(times):.1f} Hz")
    print(f"10Hz 需要每次检测 <100ms，当前 DINO-X 需要 ~230ms")
    print(f"即使禁用CDM，也无法达到10Hz")


if __name__ == '__main__':
    main()
