#!/usr/bin/env python3
"""
测试不同分辨率对DINO-X响应时间的影响
"""

import time
import cv2
import numpy as np
import requests

DINOX_URL = "http://192.168.112.14:10086/api/predict"
NUM_TESTS = 5

RESOLUTIONS = [
    (1280, 720),   # 原始
    (960, 540),    # 75%
    (640, 480),    # VGA
    (640, 360),    # 50%
    (480, 270),    # 37.5%
    (320, 240),    # QVGA
]


def test_dinox(rgb, session):
    """DINO-X调用"""
    t0 = time.time()
    _, img_encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_bytes = img_encoded.tobytes()

    files = {'images': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'text_prompt': 'bottle.cup.box',
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
    print("DINO-X 分辨率 vs 响应时间测试")
    print("=" * 60)

    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': None, 'https': None}

    results = []

    for w, h in RESOLUTIONS:
        print(f"\n测试分辨率: {w}x{h}")
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        times = []
        for i in range(NUM_TESTS):
            t, ok = test_dinox(rgb, session)
            times.append(t)
            print(f"  [{i+1}/{NUM_TESTS}] {t:.1f}ms")

        avg = np.mean(times)
        max_hz = 1000 / avg
        results.append((w, h, avg, max_hz))
        print(f"  平均: {avg:.1f}ms, 最大频率: {max_hz:.1f}Hz")

    # 总结
    print("\n" + "=" * 60)
    print("分辨率 vs 响应时间总结")
    print("=" * 60)
    print(f"{'分辨率':<15} {'平均耗时':<12} {'最大频率':<12} {'达10Hz':<8}")
    print("-" * 47)
    for w, h, avg, max_hz in results:
        can_10hz = "✓" if max_hz >= 10 else "✗"
        print(f"{w}x{h:<10} {avg:>8.1f}ms {max_hz:>8.1f}Hz {can_10hz:^8}")

    print("\n结论:")
    for w, h, avg, max_hz in results:
        if max_hz >= 10:
            print(f"  使用 {w}x{h} 分辨率可以达到 10Hz")
            break
    else:
        print(f"  即使最小分辨率 {results[-1][0]}x{results[-1][1]} 也无法达到 10Hz")
        print(f"  瓶颈在 DINO-X 服务端，建议:")
        print(f"    1. 升级服务器GPU")
        print(f"    2. 使用更轻量的检测模型")
        print(f"    3. 降低检测频率目标")


if __name__ == '__main__':
    main()
