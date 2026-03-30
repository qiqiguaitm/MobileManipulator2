#!/usr/bin/env python3
"""
从 top RealSense 相机采集各分辨率图像，保存供 bench 测试用。

用法: python3 scripts/_cc_capture_bench_images.py
输出: scripts/bench_images/{WxH}/ 下各含 rgb.jpg, depth.png, ir_left.jpg, ir_right.jpg
"""

import os
import sys
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("需要 pyrealsense2")

TOP_CAMERA_SERIAL = '318122302992'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'bench_images')

# (rgb_w, rgb_h, depth_w, depth_h) — depth/IR 上限 1280x720
RESOLUTIONS = [
    (424, 240, 424, 240),
    (480, 270, 480, 270),
    (640, 360, 640, 360),
    (640, 480, 640, 480),
    (848, 480, 848, 480),
    (1280, 720, 1280, 720),
]

# 额外：用 1280x720 采集后 resize RGB 模拟的高分辨率
UPSCALE_RESOLUTIONS = [
    (1920, 1080),
]


def capture(rgb_w, rgb_h, dw, dh):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(TOP_CAMERA_SERIAL)

    config.enable_stream(rs.stream.color, rgb_w, rgb_h, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, dw, dh, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, dw, dh, rs.format.y8, 30)

    pipeline.start(config)

    # 丢弃前 30 帧让自动曝光稳定
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()

    rgb = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data())
    ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())
    ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())

    pipeline.stop()
    return rgb, depth, ir_left, ir_right


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rgb_w, rgb_h, dw, dh in RESOLUTIONS:
        label = f"{rgb_w}x{rgb_h}"
        if (rgb_w, rgb_h) != (dw, dh):
            label += f"_{dw}x{dh}"

        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"采集 {label}  rgb={rgb_w}x{rgb_h}  depth/ir={dw}x{dh} ...", end=' ', flush=True)

        try:
            rgb, depth, ir_left, ir_right = capture(rgb_w, rgb_h, dw, dh)
        except Exception as e:
            print(f"SKIP: {e}")
            continue

        cv2.imwrite(os.path.join(out_dir, 'rgb.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(out_dir, 'depth.png'), depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(out_dir, 'ir_left.jpg'), ir_left, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(out_dir, 'ir_right.jpg'), ir_right, [cv2.IMWRITE_JPEG_QUALITY, 95])

        rgb_kb = os.path.getsize(os.path.join(out_dir, 'rgb.jpg')) / 1024
        dpt_kb = os.path.getsize(os.path.join(out_dir, 'depth.png')) / 1024
        irl_kb = os.path.getsize(os.path.join(out_dir, 'ir_left.jpg')) / 1024
        irr_kb = os.path.getsize(os.path.join(out_dir, 'ir_right.jpg')) / 1024
        total_kb = rgb_kb + dpt_kb + irl_kb + irr_kb

        print(f"OK  total={total_kb:.0f}KB (rgb={rgb_kb:.0f} dpt={dpt_kb:.0f} ir={irl_kb:.0f}+{irr_kb:.0f})")

    # 用 1280x720 采集数据 resize 出高分辨率版本
    if UPSCALE_RESOLUTIONS:
        src_dir = os.path.join(OUTPUT_DIR, '1280x720')
        if os.path.isdir(src_dir):
            rgb_src = cv2.imread(os.path.join(src_dir, 'rgb.jpg'), cv2.IMREAD_COLOR)
            dpt_src = cv2.imread(os.path.join(src_dir, 'depth.png'), cv2.IMREAD_UNCHANGED)
            irl_src = cv2.imread(os.path.join(src_dir, 'ir_left.jpg'), cv2.IMREAD_GRAYSCALE)
            irr_src = cv2.imread(os.path.join(src_dir, 'ir_right.jpg'), cv2.IMREAD_GRAYSCALE)

            for up_w, up_h in UPSCALE_RESOLUTIONS:
                label = f"{up_w}x{up_h}_upscaled"
                out_dir = os.path.join(OUTPUT_DIR, label)
                os.makedirs(out_dir, exist_ok=True)

                print(f"Resize {label}  rgb→{up_w}x{up_h}  depth/ir 保持 1280x720 ...", end=' ', flush=True)

                rgb_up = cv2.resize(rgb_src, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(out_dir, 'rgb.jpg'), rgb_up, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(out_dir, 'depth.png'), dpt_src, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(os.path.join(out_dir, 'ir_left.jpg'), irl_src, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(out_dir, 'ir_right.jpg'), irr_src, [cv2.IMWRITE_JPEG_QUALITY, 95])

                rgb_kb = os.path.getsize(os.path.join(out_dir, 'rgb.jpg')) / 1024
                dpt_kb = os.path.getsize(os.path.join(out_dir, 'depth.png')) / 1024
                irl_kb = os.path.getsize(os.path.join(out_dir, 'ir_left.jpg')) / 1024
                irr_kb = os.path.getsize(os.path.join(out_dir, 'ir_right.jpg')) / 1024
                total_kb = rgb_kb + dpt_kb + irl_kb + irr_kb
                print(f"OK  total={total_kb:.0f}KB (rgb={rgb_kb:.0f} dpt={dpt_kb:.0f} ir={irl_kb:.0f}+{irr_kb:.0f})")

    print(f"\n保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
