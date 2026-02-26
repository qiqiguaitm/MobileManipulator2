#!/usr/bin/env python3
"""
验证 LiDAR 点云投影到 Top Camera 图像是否正确

关键修正：外参文件是 TF 约定，需要转换为点变换形式
"""

import sys
import os

sys.path.insert(0, '/home/agilex/MobileManipulator/arm_robot/src')

import cv2
import numpy as np
import yaml
import time
from scipy.spatial.transform import Rotation

from ros_lidar import ROSLiDAR
from camera import RealSenseCamera


class SimpleConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_extrinsics_as_point_transform(yaml_path):
    """
    加载外参文件并转换为点变换形式

    YAML 文件存储的是 TF 约定：
    - t_tf: 相机原点在 lidar 坐标系中的位置
    - R_tf: 相机坐标轴在 lidar 坐标系中的方向

    点变换需要：
    - p_cam = R_point @ p_lidar + t_point
    - R_point = R_tf^T
    - t_point = -R_tf^T @ t_tf
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    trans = data['transform']['translation']
    t_tf = np.array([trans['x'], trans['y'], trans['z']])

    rot = data['transform']['rotation']
    quat = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
    R_tf = Rotation.from_quat(quat).as_matrix()

    # 转换为点变换形式
    R_point = R_tf.T
    t_point = -R_tf.T @ t_tf

    return R_point, t_point, R_tf, t_tf


def transform_points(points, R, t):
    """点变换: p_out = R @ p_in + t"""
    if len(points) == 0:
        return points
    xyz = points[:, :3]
    xyz_t = (R @ xyz.T).T + t
    if points.shape[1] > 3:
        return np.hstack([xyz_t, points[:, 3:]])
    return xyz_t


def project_to_image(points_optical, intrinsics):
    if len(points_optical) == 0:
        return np.empty((0, 2))
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    X, Y, Z = points_optical[:, 0], points_optical[:, 1], points_optical[:, 2]
    Z_safe = np.where(Z > 0.01, Z, 0.01)
    u = fx * X / Z_safe + cx
    v = fy * Y / Z_safe + cy
    return np.column_stack([u, v])


def main():
    print("=" * 70)
    print("LiDAR → Top Camera 投影验证 (使用正确的点变换)")
    print("=" * 70)

    extrinsics_path = '/home/agilex/MobileManipulator/src/cam_lidar_calibrate/results/calib_data_top_20251229_144945/conf/interactive_extrinsics.yaml'
    output_dir = '/home/agilex/MobileManipulator/scripts/lidar_projection_test'
    os.makedirs(output_dir, exist_ok=True)

    # 加载外参
    print(f"\n[外参文件] {extrinsics_path}")
    R_point, t_point, R_tf, t_tf = load_extrinsics_as_point_transform(extrinsics_path)

    print(f"\nTF 约定 (YAML 原始值):")
    print(f"  t_tf: [{t_tf[0]:.4f}, {t_tf[1]:.4f}, {t_tf[2]:.4f}]")
    euler_tf = Rotation.from_matrix(R_tf).as_euler('xyz', degrees=True)
    print(f"  R_tf euler: [{euler_tf[0]:.1f}, {euler_tf[1]:.1f}, {euler_tf[2]:.1f}]°")

    print(f"\n点变换形式 (p_cam = R @ p_lidar + t):")
    print(f"  t_point: [{t_point[0]:.4f}, {t_point[1]:.4f}, {t_point[2]:.4f}]")
    euler_point = Rotation.from_matrix(R_point).as_euler('xyz', degrees=True)
    print(f"  R_point euler: [{euler_point[0]:.1f}, {euler_point[1]:.1f}, {euler_point[2]:.1f}]°")

    # 分析坐标轴映射
    print(f"\n坐标轴映射 (点变换后):")
    print(f"  rslidar X(前) → optical: [{R_point[0,0]:.3f}, {R_point[1,0]:.3f}, {R_point[2,0]:.3f}]")
    print(f"  rslidar Y(左) → optical: [{R_point[0,1]:.3f}, {R_point[1,1]:.3f}, {R_point[2,1]:.3f}]")
    print(f"  rslidar Z(上) → optical: [{R_point[0,2]:.3f}, {R_point[1,2]:.3f}, {R_point[2,2]:.3f}]")

    # 检查主导轴
    print(f"\n图像 U (optical X) 主导:")
    u_contrib = np.abs(R_point[0, :])
    dominant_u = ['X(前)', 'Y(左)', 'Z(上)'][np.argmax(u_contrib)]
    print(f"    贡献: X={u_contrib[0]:.3f}, Y={u_contrib[1]:.3f}, Z={u_contrib[2]:.3f} → {dominant_u}")

    print(f"图像 V (optical Y) 主导:")
    v_contrib = np.abs(R_point[1, :])
    dominant_v = ['X(前)', 'Y(左)', 'Z(上)'][np.argmax(v_contrib)]
    print(f"    贡献: X={v_contrib[0]:.3f}, Y={v_contrib[1]:.3f}, Z={v_contrib[2]:.3f} → {dominant_v}")

    # 预期：rslidar Y(左右) 应主导图像 U，rslidar Z(上下) 应主导图像 V
    print(f"\n预期映射检查:")
    print(f"  图像 U 应由 rslidar Y(左右) 主导: {dominant_u == 'Y(左)'} ({'✓' if dominant_u == 'Y(左)' else '✗'})")
    print(f"  图像 V 应由 rslidar Z(上下) 主导: {dominant_v == 'Z(上)'} ({'✓' if dominant_v == 'Z(上)' else '✗'})")

    # 连接传感器
    print("\n" + "-" * 70)
    camera_cfg = SimpleConfig(
        device_id='318122302992', width=1280, height=720, fps=30,
        enable_depth=True, use_calibrated_intrinsics=False,
    )
    camera = RealSenseCamera(camera_cfg)
    camera.connect()
    intrinsics = {
        'fx': camera.intrinsics.fx, 'fy': camera.intrinsics.fy,
        'cx': camera.intrinsics.ppx, 'cy': camera.intrinsics.ppy,
        'width': camera.intrinsics.width, 'height': camera.intrinsics.height,
    }
    print(f"相机内参: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    lidar = ROSLiDAR(topic='/lidar/chassis/point_cloud', timeout=1.0)
    lidar.connect()

    # 采集并验证
    print("\n" + "-" * 70)
    print("采集验证 (3帧)")
    print("-" * 70)

    results = []

    try:
        for frame_num in range(3):
            bundle = camera.get_image_bundle()
            rgb = bundle['rgb'].copy()
            lidar_points = lidar.get_one_frame()

            if len(lidar_points) == 0:
                continue

            # 使用点变换形式
            points_optical = transform_points(lidar_points, R_point, t_point)

            # 过滤
            valid = (points_optical[:, 2] > 0.3) & (points_optical[:, 2] < 6.0)
            points_filtered = points_optical[valid]

            if len(points_filtered) == 0:
                continue

            uv = project_to_image(points_filtered, intrinsics)

            in_image = (uv[:, 0] >= 0) & (uv[:, 0] < intrinsics['width']) & \
                       (uv[:, 1] >= 0) & (uv[:, 1] < intrinsics['height'])
            uv_valid = uv[in_image]
            z_valid = points_filtered[in_image, 2]

            if len(uv_valid) == 0:
                continue

            u_range = uv_valid[:, 0].max() - uv_valid[:, 0].min()
            v_range = uv_valid[:, 1].max() - uv_valid[:, 1].min()
            ratio = u_range / max(v_range, 1)

            if ratio < 0.5:
                status = "垂直条纹 ✗"
            elif ratio < 0.8:
                status = "偏垂直"
            else:
                status = "正常 ✓"

            results.append({'u_range': u_range, 'v_range': v_range, 'ratio': ratio, 'status': status})

            print(f"\n[帧 {frame_num+1}] {len(uv_valid)} 点")
            print(f"  U: [{uv_valid[:, 0].min():.0f}, {uv_valid[:, 0].max():.0f}] 跨度 {u_range:.0f} px")
            print(f"  V: [{uv_valid[:, 1].min():.0f}, {uv_valid[:, 1].max():.0f}] 跨度 {v_range:.0f} px")
            print(f"  比例: {ratio:.2f} → {status}")

            # 绘制
            for i, (u, v) in enumerate(uv_valid.astype(int)):
                z = z_valid[i]
                t = min(1.0, (z - 0.3) / 5.0)
                color = (int(255 * (1 - t)), 50, int(255 * t))
                cv2.circle(rgb, (u, v), 2, color, -1)

            cv2.putText(rgb, f"Frame {frame_num+1}: {len(uv_valid)} pts | U:{u_range:.0f} V:{v_range:.0f} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            output_path = os.path.join(output_dir, f'projection_corrected_{frame_num+1}.png')
            cv2.imwrite(output_path, rgb)
            print(f"  保存: {output_path}")

            time.sleep(0.2)

    finally:
        lidar.disconnect()
        camera.pipeline.stop()

    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    if len(results) > 0:
        normal_count = sum(1 for r in results if '✓' in r['status'])
        avg_ratio = np.mean([r['ratio'] for r in results])
        print(f"正常帧: {normal_count}/{len(results)}, 平均 U/V 比例: {avg_ratio:.2f}")
        if normal_count >= len(results) * 0.6:
            print("✓ 投影正常")
        else:
            print("✗ 投影异常 - 请检查坐标系")
    print(f"\n图像: {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
