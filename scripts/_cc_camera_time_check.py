#!/usr/bin/env python3
"""
相机时钟 vs 系统时钟 同步性检测脚本

检测 RealSense 相机的 header.stamp 与系统 wall-clock 之间的时间偏差。
使用图像的 header.stamp 而非系统时间来评估实时同步状态。

用法:
    ros2 run perception _cc_camera_time_check.py
    或直接: python3 scripts/_cc_camera_time_check.py

同时检测:
    1. 相机stamp vs 系统wall-clock (绝对offset)
    2. 相机stamp的变化率 (是否单调递增)
    3. 相机stamp与图像实际帧率的匹配度
    4. 多相机之间的时钟一致性
"""

import sys
import time
import argparse
import threading
import queue
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class CameraTimeChecker(Node):
    def __init__(self, cameras: List[str], sample_count: int = 30, interval: float = 0.2):
        super().__init__('camera_time_checker')

        self.cameras = cameras
        self.sample_count = sample_count
        self.interval = interval
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=2,
        )

        # 每个相机的采样数据: (real_time_sec, camera_stamp_sec, wall_now_sec)
        self.samples: Dict[str, List[Tuple[float, float, float]]] = {cam: [] for cam in cameras}

        # 相机订阅
        self.subs: Dict[str, any] = {}
        for cam in cameras:
            topic = f'/camera/{cam}/color/image_raw'
            self.subs[cam] = self.create_subscription(
                Image, topic,
                lambda msg, c=cam: self._on_image(msg, c),
                self.qos
            )

        # 采样控制
        self._done = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        self.get_logger().info(f'已订阅相机: {cameras}')

    def _spin(self):
        while rclpy.ok() and not self._done.is_set():
            rclpy.spin_once(self, timeout_sec=0.05)

    def _on_image(self, msg: Image, camera: str):
        """图像回调：记录时间戳"""
        wall_now = time.time()
        cam_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.samples[camera].append((wall_now, cam_stamp, wall_now))

        # 采样够了就停止
        if all(len(v) >= self.sample_count for v in self.samples.values()):
            self._done.set()

    def run(self) -> Dict:
        """执行采样并返回结果"""
        self.get_logger().info(f'开始采样，每相机 {self.sample_count} 帧，间隔 {self.interval}s...')

        # 等待足够样本
        start = time.time()
        while not self._done.is_set():
            if time.time() - start > 10:
                self.get_logger().error('采样超时，未收到足够图像帧')
                break
            time.sleep(0.1)

        time.sleep(0.3)  # 再等一会儿确保最后几帧也收到
        self._done.set()
        self._spin_thread.join(timeout=2)

        return self._analyze()

    def _analyze(self) -> Dict:
        results = {}

        for cam, samples in self.samples.items():
            if not samples:
                results[cam] = {'error': '无采样数据'}
                continue

            offsets = []
            stamp_deltas = []
            wall_deltas = []

            for i in range(len(samples)):
                wall_now_i, cam_stamp_i, _ = samples[i]
                offset_i = cam_stamp_i - wall_now_i  # 正=相机超前，负=相机落后
                offsets.append(offset_i)

                if i > 0:
                    wall_now_prev, cam_stamp_prev, _ = samples[i - 1]
                    stamp_deltas.append(cam_stamp_i - cam_stamp_prev)
                    wall_deltas.append(wall_now_i - wall_now_prev)

            offsets_arr = offsets
            offset_avg = sum(offsets_arr) / len(offsets_arr)
            offset_min = min(offsets_arr)
            offset_max = max(offsets_arr)
            offset_std = self._std(offsets_arr)

            # 帧率估算
            if stamp_deltas:
                avg_dt = sum(stamp_deltas) / len(stamp_deltas)
                est_fps = 1.0 / avg_dt if avg_dt > 0 else 0
                fps_diff = abs(est_fps - 15.0)  # 期望15Hz
            else:
                avg_dt = 0
                est_fps = 0
                fps_diff = 0

            # 漂移检测：offset波动
            offset_drift = offset_max - offset_min

            # 单调性检查
            monotonic_ok = all(d > 0 for d in stamp_deltas) if stamp_deltas else False

            results[cam] = {
                'n_samples': len(samples),
                'offset_avg_ms': offset_avg * 1000,
                'offset_min_ms': offset_min * 1000,
                'offset_max_ms': offset_max * 1000,
                'offset_std_ms': offset_std * 1000,
                'offset_drift_ms': offset_drift * 1000,
                'est_fps': est_fps,
                'fps_diff_from_15Hz': fps_diff,
                'avg_frame_dt_ms': avg_dt * 1000,
                'monotonic': monotonic_ok,
                'offsets_ms': [o * 1000 for o in offsets_arr],
            }

        return results

    @staticmethod
    def _std(arr: List[float]) -> float:
        if len(arr) < 2:
            return 0.0
        mean = sum(arr) / len(arr)
        variance = sum((x - mean) ** 2 for x in arr) / len(arr)
        return variance ** 0.5

    def print_report(self, results: Dict):
        print()
        print("=" * 70)
        print("  相机时钟 vs 系统时钟 同步性检测报告")
        print("=" * 70)

        for cam, r in results.items():
            if 'error' in r:
                print(f"\n  [{cam}] 错误: {r['error']}")
                continue

            print(f"\n  [{cam}] ({r['n_samples']} 帧样本)")
            print(f"  {'─' * 60}")

            # 时钟偏差
            offset = r['offset_avg_ms']
            tag = ""
            if abs(offset) < 5:
                tag = "✓ 良好"
            elif abs(offset) < 50:
                tag = "⚠ 轻微偏差"
            elif abs(offset) < 200:
                tag = "⚠ 中等偏差"
            else:
                tag = "✗ 严重偏差"

            print(f"  时钟偏差 (相机 - 系统):")
            print(f"    平均: {offset:+8.1f} ms  {tag}")
            print(f"    范围: [{r['offset_min_ms']:+8.1f}, {r['offset_max_ms']:+8.1f}] ms")
            print(f"    标准差: {r['offset_std_ms']:.1f} ms")
            print(f"    漂移:   {r['offset_drift_ms']:+.1f} ms (采样期间)")

            # 帧率
            fps = r['est_fps']
            print(f"\n  帧率检查:")
            if fps > 0:
                diff = abs(fps - 15.0)
                fps_tag = "✓" if diff < 1 else "⚠"
                print(f"    估算帧率: {fps:.1f} Hz  (期望 15 Hz) {fps_tag}")
                print(f"    帧间隔:    {r['avg_frame_dt_ms']:.1f} ms")
            else:
                print(f"    无法估算")

            # 单调性
            mono = r['monotonic']
            print(f"\n  时间戳单调性: {'✓ 单调递增' if mono else '✗ 存在回退!'}")

            # 结论
            print(f"\n  诊断结论:")
            if abs(offset) < 5 and r['offset_drift_ms'] < 10:
                print(f"    ✓ 时钟同步良好，无明显偏差或漂移")
            elif abs(offset) < 50:
                print(f"    ⚠ 时钟有轻微偏差 (≈{abs(offset):.0f}ms)")
                print(f"      可能原因: RealSense 固件时钟漂移或初始化偏移")
                print(f"      影响: 感知层TF查询会有 ≈{abs(offset)*1:.0f}ms 额外误差")
            elif abs(offset) < 200:
                print(f"    ⚠ 时钟有明显偏差 (≈{abs(offset):.0f}ms)")
                print(f"      可能原因: RealSense 未使用系统时钟同步，固件时间漂移")
                print(f"      影响: 感知层TF查询误差 ≈{abs(offset):.0f}ms")
            else:
                print(f"    ✗ 时钟严重不同步 (≈{abs(offset):.0f}ms)")
                print(f"      可能原因: RealSense 使用独立硬件时钟，未与系统同步")
                print(f"      建议: 检查 realsense2_camera 的 use_system_time 参数")

        # 相机间一致性
        valid_cams = [cam for cam, r in results.items() if 'error' not in r and r.get('n_samples', 0) > 0]
        if len(valid_cams) > 1:
            print(f"\n  {'─' * 60}")
            print(f"  多相机时钟一致性:")
            offsets = [results[cam]['offset_avg_ms'] for cam in valid_cams]
            inter_offset_diff = max(offsets) - min(offsets)
            print(f"    各相机平均offset: " + ", ".join(
                f"{results[cam]['offset_avg_ms']:+.0f}ms ({cam})" for cam in valid_cams))
            print(f"    相机间最大差异: {inter_offset_diff:.0f} ms")
            if inter_offset_diff < 5:
                print(f"    ✓ 各相机时钟一致")
            else:
                print(f"    ⚠ 各相机时钟不同步!")

        print()
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='相机时钟同步检测')
    parser.add_argument('--cameras', '-c', nargs='+',
                        default=['top', 'chassis', 'hand'],
                        help='要检测的相机名称 (默认: top chassis hand)')
    parser.add_argument('--samples', '-n', type=int, default=30,
                        help='每相机采样帧数 (默认: 30)')
    parser.add_argument('--interval', '-i', type=float, default=0.2,
                        help='采样间隔秒数 (默认: 0.2)')
    args = parser.parse_args()

    rclpy.init()
    try:
        checker = CameraTimeChecker(
            cameras=args.cameras,
            sample_count=args.samples,
            interval=args.interval,
        )
        results = checker.run()
        checker.print_report(results)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
