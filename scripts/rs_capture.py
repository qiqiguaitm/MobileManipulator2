#!/usr/bin/env python3
"""从 chassis 相机 (D435) 直接采集一帧 RGB + aligned Depth 并保存。

Usage:
    python3 scripts/_cc_rs_capture.py
    python3 scripts/_cc_rs_capture.py --output /tmp/capture
    python3 scripts/_cc_rs_capture.py --serial 337122071540
"""

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(description='RealSense chassis 相机单帧采集')
    parser.add_argument('--serial', default='337122071540',
                        help='相机序列号 (default: chassis D435)')
    parser.add_argument('--output', default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), '..', 'captures'),
                        help='输出目录')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=30,
                        help='丢弃前 N 帧用于自动曝光稳定')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 配置 pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    # 启动
    profile = pipeline.start(config)

    # 开启 align (depth → color)
    align = rs.align(rs.stream.color)

    # 获取内参
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print(f"相机 {args.serial} 已启动: {intr.width}x{intr.height}")
    print(f"内参: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")

    # 自动曝光稳定
    print(f"等待自动曝光稳定 ({args.warmup} 帧)...")
    for i in range(args.warmup):
        pipeline.wait_for_frames()

    # 采集一帧
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    if not color_frame or not depth_frame:
        print("采集失败: 无有效帧")
        pipeline.stop()
        return

    rgb = np.asanyarray(color_frame.get_data())        # (H,W,3) uint8 BGR
    depth_mm = np.asanyarray(depth_frame.get_data())    # (H,W) uint16 mm

    pipeline.stop()

    # 保存
    ts = time.strftime('%Y%m%d_%H%M%S')
    rgb_path = os.path.join(args.output, f'chassis_{ts}_rgb.jpg')
    depth_path = os.path.join(args.output, f'chassis_{ts}_depth.png')
    intrinsics_path = os.path.join(args.output, f'chassis_{ts}_intrinsics.txt')

    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(depth_path, depth_mm)

    with open(intrinsics_path, 'w') as f:
        f.write(f"fx: {intr.fx}\nfy: {intr.fy}\ncx: {intr.ppx}\ncy: {intr.ppy}\n"
                f"width: {intr.width}\nheight: {intr.height}\n")

    # 统计
    valid = depth_mm > 0
    print(f"\nRGB: {rgb.shape} → {rgb_path}")
    print(f"Depth: {depth_mm.shape} uint16 mm → {depth_path}")
    print(f"  有效像素: {valid.sum()}/{depth_mm.size} ({valid.sum()/depth_mm.size*100:.1f}%)")
    print(f"  深度范围: {depth_mm[valid].min()}mm ~ {depth_mm[valid].max()}mm")
    print(f"内参 → {intrinsics_path}")


if __name__ == '__main__':
    main()
