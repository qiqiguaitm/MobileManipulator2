#!/usr/bin/env python3
"""深度诊断脚本 — 从 chassis 相机抓一帧，分析深度精度和 pinhole 反投影。

用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_depth_diag.py                         # 交互式选 ROI
  python3 scripts/_cc_depth_diag.py --roi 400,300,500,400   # 指定 ROI (x1,y1,x2,y2)
  python3 scripts/_cc_depth_diag.py --center                # 使用图像中心 100x100 区域
"""

import argparse
import sys
import os
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Chassis 相机深度诊断')
    parser.add_argument('--roi', help='ROI 区域 x1,y1,x2,y2 (像素)')
    parser.add_argument('--center', action='store_true', help='使用图像中心 100x100 区域')
    parser.add_argument('--save', action='store_true', help='保存 RGB+depth 到 /tmp')
    parser.add_argument('--camera', default='chassis', help='相机名 (default: chassis)')
    args = parser.parse_args()

    # ROS2 初始化
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import cv2

    rclpy.init()
    node = rclpy.create_node('depth_diag')
    bridge = CvBridge()

    camera = args.camera
    color_topic = f'/camera/{camera}/color/image_raw'
    depth_topic = f'/camera/{camera}/aligned_depth_to_color/image_raw'
    info_topic = f'/camera/{camera}/aligned_depth_to_color/camera_info'

    print(f'=== 深度诊断: {camera} 相机 ===')
    print(f'  color: {color_topic}')
    print(f'  depth: {depth_topic}')
    print(f'  info:  {info_topic}')
    print()

    # 1. 获取内参
    print('等待相机内参...')
    intrinsics = None
    def info_cb(msg):
        nonlocal intrinsics
        if intrinsics is None:
            intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[4 + 2],  # k[5] is not cy, k[2] is cx, k[5] is cy
                'width': msg.width, 'height': msg.height,
            }
            # Fix: cy is k[5]
            intrinsics['cy'] = msg.k[5]

    sub_info = node.create_subscription(CameraInfo, info_topic, info_cb, 10)
    t0 = time.time()
    while intrinsics is None and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.5)

    if intrinsics is None:
        print('错误: 未收到相机内参')
        sys.exit(1)

    print(f'  内参: fx={intrinsics["fx"]:.2f} fy={intrinsics["fy"]:.2f} '
          f'cx={intrinsics["cx"]:.2f} cy={intrinsics["cy"]:.2f} '
          f'{intrinsics["width"]}x{intrinsics["height"]}')
    print()

    # 2. 获取 RGB + depth 帧
    print('等待 RGB + depth 帧...')
    rgb_frame = None
    depth_frame = None

    def color_cb(msg):
        nonlocal rgb_frame
        if rgb_frame is None:
            rgb_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_cb(msg):
        nonlocal depth_frame
        if depth_frame is None:
            depth_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    sub_color = node.create_subscription(Image, color_topic, color_cb, 10)
    sub_depth = node.create_subscription(Image, depth_topic, depth_cb, 10)

    t0 = time.time()
    while (rgb_frame is None or depth_frame is None) and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.5)

    if rgb_frame is None or depth_frame is None:
        print('错误: 未收到图像帧')
        sys.exit(1)

    H, W = depth_frame.shape[:2]
    print(f'  RGB:   {rgb_frame.shape}')
    print(f'  Depth: {depth_frame.shape} dtype={depth_frame.dtype}')

    # 转换深度到米
    if depth_frame.dtype == np.uint16:
        depth_m = depth_frame.astype(np.float32) / 1000.0
    else:
        depth_m = depth_frame.astype(np.float32)

    # 保存帧
    if args.save:
        cv2.imwrite('/tmp/diag_rgb.png', rgb_frame)
        np.save('/tmp/diag_depth_mm.npy', depth_frame)
        print(f'  已保存: /tmp/diag_rgb.png, /tmp/diag_depth_mm.npy')
    print()

    # 3. 全图深度统计
    valid_mask = depth_m > 0.1
    print('=== 全图深度统计 ===')
    print(f'  有效像素: {valid_mask.sum()}/{H*W} ({valid_mask.sum()/(H*W)*100:.1f}%)')
    if valid_mask.sum() > 0:
        vd = depth_m[valid_mask]
        print(f'  深度范围: [{vd.min():.3f}, {vd.max():.3f}] m')
        print(f'  中值: {np.median(vd):.4f} m')
        print(f'  均值: {np.mean(vd):.4f} m')
    print()

    # 4. 图像中心单点深度
    cx_px, cy_px = W // 2, H // 2
    center_depth = depth_m[cy_px, cx_px]
    print(f'=== 图像中心 ({cx_px}, {cy_px}) ===')
    print(f'  深度: {center_depth:.4f} m ({center_depth*1000:.1f} mm)')

    # pinhole 反投影
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx_i, cy_i = intrinsics['cx'], intrinsics['cy']
    if center_depth > 0.1:
        X = (cx_px - cx_i) * center_depth / fx
        Y = (cy_px - cy_i) * center_depth / fy
        Z = center_depth
        print(f'  光学坐标: ({X:.4f}, {Y:.4f}, {Z:.4f}) m')
    print()

    # 5. ROI 分析
    if args.center:
        roi = [W//2 - 50, H//2 - 50, W//2 + 50, H//2 + 50]
    elif args.roi:
        roi = list(map(int, args.roi.split(',')))
    else:
        # 交互式: 让用户输入或使用默认
        print('提示: 使用 --roi x1,y1,x2,y2 或 --center 指定分析区域')
        print('      使用默认: 图像中心 200x200 区域')
        roi = [W//2 - 100, H//2 - 100, W//2 + 100, H//2 + 100]

    x1, y1, x2, y2 = roi
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    roi_depth = depth_m[y1:y2, x1:x2]
    roi_valid = roi_depth[roi_depth > 0.1]

    print(f'=== ROI [{x1},{y1},{x2},{y2}] ({x2-x1}x{y2-y1}) ===')
    print(f'  有效像素: {len(roi_valid)}/{roi_depth.size}')
    if len(roi_valid) > 0:
        print(f'  深度范围: [{roi_valid.min():.4f}, {roi_valid.max():.4f}] m')
        print(f'  中值: {np.median(roi_valid):.4f} m ({np.median(roi_valid)*1000:.1f} mm)')
        print(f'  均值: {np.mean(roi_valid):.4f} m ({np.mean(roi_valid)*1000:.1f} mm)')
        print(f'  标准差: {np.std(roi_valid):.4f} m ({np.std(roi_valid)*1000:.1f} mm)')

        # 直方图
        print(f'\n  深度直方图 (ROI):')
        bins = np.linspace(roi_valid.min() - 0.01, roi_valid.max() + 0.01, 11)
        counts, edges = np.histogram(roi_valid, bins=bins)
        for i in range(len(counts)):
            bar = '█' * int(counts[i] / max(counts.max(), 1) * 30)
            print(f'    {edges[i]:.3f}-{edges[i+1]:.3f}m: {counts[i]:5d} {bar}')
    print()

    # 6. 模拟管线: 在 ROI 内做 pinhole 反投影 + median
    print('=== 模拟管线 pinhole 反投影 (ROI 内) ===')
    ys, xs = np.where(roi_depth > 0.1)
    if len(ys) > 0:
        xs_img = xs.astype(np.float64) + x1
        ys_img = ys.astype(np.float64) + y1
        ds = roi_depth[ys, xs]

        X = (xs_img - cx_i) * ds / fx
        Y = (ys_img - cy_i) * ds / fy
        Z = ds

        centroid_median = np.array([np.median(X), np.median(Y), np.median(Z)])
        centroid_mean = np.array([np.mean(X), np.mean(Y), np.mean(Z)])

        print(f'  3D点数: {len(ys)}')
        print(f'  中值质心 (opt): ({centroid_median[0]:.4f}, {centroid_median[1]:.4f}, {centroid_median[2]:.4f}) m')
        print(f'  均值质心 (opt): ({centroid_mean[0]:.4f}, {centroid_mean[1]:.4f}, {centroid_mean[2]:.4f}) m')
        print(f'  Z (depth): median={np.median(Z):.4f} mean={np.mean(Z):.4f} m')
        print(f'  像素 v: min={ys_img.min():.0f} max={ys_img.max():.0f} median={np.median(ys_img):.0f}')
        print(f'  像素 u: min={xs_img.min():.0f} max={xs_img.max():.0f} median={np.median(xs_img):.0f}')

        # Y 分析
        _v_med = np.median(ys_img)
        _Y_at_vmed = (_v_med - cy_i) * np.median(Z) / fy
        print(f'\n  Y 分解: v_median={_v_med:.0f}, cy={cy_i:.2f}, v-cy={_v_med - cy_i:.1f}px')
        print(f'          Y = (v-cy)*Z/fy = ({_v_med - cy_i:.1f})*{np.median(Z):.4f}/{fy:.2f} = {_Y_at_vmed:.4f}m')
    print()

    # 7. 深度沿垂直中线的剖面
    print('=== 垂直中线深度剖面 (u=中心, 每隔50行) ===')
    u_center = W // 2
    for v in range(0, H, 50):
        d = depth_m[v, u_center]
        if d > 0.1:
            Y_opt = (v - cy_i) * d / fy
            print(f'  v={v:4d}  depth={d:.4f}m  Y_opt={Y_opt:.4f}m')
        else:
            print(f'  v={v:4d}  depth=0 (invalid)')
    print()

    print('诊断完成。')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
