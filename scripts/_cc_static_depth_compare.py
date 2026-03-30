#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态 LiDAR vs RealSense 深度对比工具

采集 N 帧 RealSense 深度取中值去噪，与单帧 LiDAR 对比。
生成三张静态可视化图：
  1. LiDAR 投影叠加（深度着色）
  2. 深度差异热图（LiDAR - RealSense，绿=吻合/红=LiDAR远/蓝=LiDAR近）
  3. RealSense 中值深度图

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/_cc_static_depth_compare.py --camera chassis
    python3 scripts/_cc_static_depth_compare.py --camera chassis --n-frames 60
    python3 scripts/_cc_static_depth_compare.py --camera top --save
"""
import os
import signal
import subprocess
import sys
import argparse
import threading
import time
import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'cam_lidar_calibration'))
from lib.io_utils import load_extrinsics as _lib_load_extrinsics
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'src', 'perception', 'config')

CAMERAS = {
    'chassis': {
        'image_topic': '/camera/chassis/color/image_raw',
        'depth_topic': '/camera/chassis/aligned_depth_to_color/image_raw',
        'extrinsics': os.path.join(CONFIG_DIR, 'extrinsics_rslidar_to_chassis_camera_optical_frame.yaml'),
        'intrinsics': os.path.join(CONFIG_DIR, 'intrinsics_chassis_d435.yaml'),
    },
    'top': {
        'image_topic': '/camera/top/color/image_raw',
        'depth_topic': '/camera/top/aligned_depth_to_color/image_raw',
        'extrinsics': os.path.join(CONFIG_DIR, 'extrinsics_rslidar_to_top_camera_optical_frame.yaml'),
        'intrinsics': os.path.join(CONFIG_DIR, 'intrinsics_top_d455.yaml'),
    },
}
LIDAR_TOPIC = '/rslidar_points'


def load_intrinsics(yaml_path, resolution='1280x720'):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for profile in data['profiles']:
        if profile['resolution'] == resolution:
            K = np.array([[profile['fx'], 0, profile['cx']],
                          [0, profile['fy'], profile['cy']],
                          [0, 0, 1]])
            D = np.array(profile.get('distortion', [0, 0, 0, 0, 0]), dtype=np.float64)
            return K, D
    raise ValueError(f"Resolution {resolution} not found in {yaml_path}")


def pointcloud2_to_xyz(msg):
    offsets = {f.name: f.offset for f in msg.fields if f.name in ('x', 'y', 'z')}
    if len(offsets) < 3:
        return np.empty((0, 3), dtype=np.float32)
    n = msg.width * msg.height
    step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    pts = np.empty((n, 3), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        pts[:, i] = np.ndarray(shape=(n,), dtype=np.float32,
                               buffer=raw.data, offset=offsets[name], strides=(step,))
    return pts[np.isfinite(pts).all(axis=1)]


# ============================================================================
# 数据采集节点
# ============================================================================

class DataCollector(Node):
    def __init__(self, camera_name, n_frames):
        super().__init__('static_depth_compare')
        self.bridge = CvBridge()
        self.n_frames = n_frames
        self.cfg = CAMERAS[camera_name]

        self.depth_frames = []
        self.rgb_image = None
        self.lidar_points = None
        self.lock = threading.Lock()
        self.done = False

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, self.cfg['image_topic'], self._rgb_cb, qos)
        self.create_subscription(Image, self.cfg['depth_topic'], self._depth_cb, qos)
        self.create_subscription(PointCloud2, LIDAR_TOPIC, self._lidar_cb, qos)

        self.get_logger().info(f"采集中: 目标 {n_frames} 帧深度...")

    def _rgb_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.rgb_image = img
        except Exception:
            pass

    def _depth_cb(self, msg):
        if self.done:
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')  # uint16, mm
            with self.lock:
                self.depth_frames.append(depth.copy())
                n = len(self.depth_frames)
                if n % 10 == 0 or n == self.n_frames:
                    self.get_logger().info(f"  深度帧: {n}/{self.n_frames}")
                if n >= self.n_frames:
                    self.done = True
        except Exception:
            pass

    def _lidar_cb(self, msg):
        pts = pointcloud2_to_xyz(msg)
        with self.lock:
            self.lidar_points = pts

    def is_ready(self):
        with self.lock:
            return (self.done and self.rgb_image is not None
                    and self.lidar_points is not None)

    def get_data(self):
        with self.lock:
            return self.rgb_image.copy(), list(self.depth_frames), self.lidar_points.copy()


# ============================================================================
# 可视化
# ============================================================================

def compute_median_depth(depth_frames):
    """多帧中值滤波去噪。"""
    stack = np.stack(depth_frames, axis=0).astype(np.float32)
    # 将 0 (无效) 替换为 nan
    stack[stack == 0] = np.nan
    with np.errstate(all='ignore'):
        median = np.nanmedian(stack, axis=0)
    median = np.nan_to_num(median, nan=0.0)
    return median.astype(np.float32)  # mm, float32


def depth_to_pointcloud(depth_m, K):
    """深度图 → 3D 点云 (相机坐标系)。"""
    h, w = depth_m.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth_m > 0.1

    z = depth_m[valid]
    x = (u_grid[valid] - K[0, 2]) * z / K[0, 0]
    y = (v_grid[valid] - K[1, 2]) * z / K[1, 1]

    return np.stack([x, y, z], axis=1).astype(np.float32)


def make_lidar_overlay(image, pts_lidar, R, t, K, max_depth=10.0, point_size=2):
    """LiDAR 深度着色投影。"""
    pts_cam = (R @ pts_lidar.T).T + t
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return image.copy()

    z = pts_cam[:, 2]
    u = (K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).astype(int)
    v = (K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).astype(int)

    h, w = image.shape[:2]
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[valid], v[valid], z[valid]

    norm = np.clip(z / max_depth, 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_JET).reshape(-1, 3)

    vis = image.copy()
    for px, py, c in zip(u, v, colors):
        cv2.circle(vis, (px, py), point_size, (int(c[0]), int(c[1]), int(c[2])), -1)
    return vis


def make_depth_diff_map(image, pts_lidar, R, t, K, depth_median_m, max_diff=0.15, point_size=3):
    """深度差异热图 + 统计。"""
    pts_cam = (R @ pts_lidar.T).T + t
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return image.copy(), {}

    z_lidar = pts_cam[:, 2]
    u = (K[0, 0] * pts_cam[:, 0] / z_lidar + K[0, 2]).astype(int)
    v = (K[1, 1] * pts_cam[:, 1] / z_lidar + K[1, 2]).astype(int)

    h, w = image.shape[:2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z_lidar = u[in_img], v[in_img], z_lidar[in_img]

    z_rs = depth_median_m[v, u]
    valid = z_rs > 0.1
    u, v, z_lidar, z_rs = u[valid], v[valid], z_lidar[valid], z_rs[valid]

    if len(u) == 0:
        return image.copy(), {}

    diff = z_lidar - z_rs

    # 着色
    norm = np.clip(diff / max_diff, -1, 1)
    colors = np.zeros((len(norm), 3), dtype=np.uint8)
    colors[:, 1] = ((1.0 - np.abs(norm)) * 255).astype(np.uint8)  # green = 吻合
    colors[:, 2] = (np.clip(norm, 0, 1) * 255).astype(np.uint8)   # red = LiDAR远
    colors[:, 0] = (np.clip(-norm, 0, 1) * 255).astype(np.uint8)  # blue = LiDAR近

    vis = image.copy()
    for px, py, c in zip(u, v, colors):
        cv2.circle(vis, (px, py), point_size, (int(c[0]), int(c[1]), int(c[2])), -1)

    # 分段统计
    bins = [(0, 2, '0-2m'), (2, 4, '2-4m'), (4, 8, '4-8m'), (8, 999, '8m+')]
    stats = {
        'all': {'n': len(diff), 'mean': np.mean(diff), 'std': np.std(diff),
                'abs_mean': np.mean(np.abs(diff)), 'median': np.median(diff)},
    }
    for lo, hi, label in bins:
        mask_b = (z_lidar >= lo) & (z_lidar < hi)
        if np.sum(mask_b) > 0:
            d = diff[mask_b]
            stats[label] = {'n': int(np.sum(mask_b)), 'mean': np.mean(d),
                            'std': np.std(d), 'abs_mean': np.mean(np.abs(d))}

    return vis, stats


def make_rs_depth_vis(depth_median_m, max_depth=10.0):
    """RealSense 中值深度可视化。"""
    norm = np.clip(depth_median_m / max_depth, 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth_median_m < 0.1] = 0
    return vis


def print_stats(stats):
    """打印深度差异统计表。"""
    print(f"\n{'─'*60}")
    print(f"  深度差异统计 (LiDAR - RealSense)")
    print(f"{'─'*60}")
    print(f"  {'区段':<10} {'点数':>6} {'均值':>10} {'|均值|':>10} {'标准差':>10}")
    print(f"  {'─'*48}")
    for key in ['all', '0-2m', '2-4m', '4-8m', '8m+']:
        if key not in stats:
            continue
        s = stats[key]
        label = '总计' if key == 'all' else key
        print(f"  {label:<10} {s['n']:>6} {s['mean']*100:>+9.2f}cm {s['abs_mean']*100:>9.2f}cm {s['std']*100:>9.2f}cm")
    print(f"{'─'*60}")


def ros2_setup_env():
    env = os.environ.copy()
    env['PATH'] = '/usr/bin:' + env.get('PATH', '')
    env['ROS_DOMAIN_ID'] = env.get('ROS_DOMAIN_ID', '42')
    for key in ('LD_LIBRARY_PATH', 'PATH'):
        if key in env:
            env[key] = ':'.join(p for p in env[key].split(':') if 'miniconda' not in p)
    return env


def launch_sensors(camera_name):
    env = ros2_setup_env()
    procs = []

    def _launch(cmd, label):
        shell_cmd = (f'source /opt/ros/humble/setup.bash && '
                     f'source {PROJECT_ROOT}/install/setup.bash 2>/dev/null; {cmd}')
        print(f"  启动 {label}")
        p = subprocess.Popen(['bash', '-c', shell_cmd], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             preexec_fn=os.setsid)
        procs.append(p)

    _launch('ros2 launch rslidar_sdk start.py', 'LiDAR')
    top_en = 'true' if camera_name == 'top' else 'false'
    chassis_en = 'true' if camera_name == 'chassis' else 'false'
    _launch(f'ros2 launch camera_driver camera_driver.launch.py '
            f'top_enable:={top_en} chassis_enable:={chassis_en} hand_enable:=false',
            f'Camera ({camera_name})')
    return procs


def stop_sensors(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except OSError:
                pass
    time.sleep(2)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                pass


def wait_for_topics(camera_name, timeout=30):
    cfg = CAMERAS[camera_name]
    needed = {LIDAR_TOPIC, cfg['image_topic'], cfg['depth_topic']}
    print(f"  等待话题就绪 (超时 {timeout}s)...")
    t0 = time.time()
    ready = set()
    while time.time() - t0 < timeout:
        try:
            result = subprocess.run(
                ['bash', '-c', f'source /opt/ros/humble/setup.bash && '
                 f'source {PROJECT_ROOT}/install/setup.bash 2>/dev/null && '
                 f'ros2 topic list'],
                capture_output=True, text=True, timeout=5, env=ros2_setup_env())
            active = set(result.stdout.strip().split('\n'))
            for t in needed:
                if t in active and t not in ready:
                    ready.add(t)
                    print(f"    ✓ {t}")
            if ready >= needed:
                print("  全部话题就绪")
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        time.sleep(2)
    missing = needed - ready
    print(f"  ⚠ 超时，缺少: {missing}")
    return False


def main():
    parser = argparse.ArgumentParser(description='静态 LiDAR vs RealSense 深度对比')
    parser.add_argument('--camera', required=True, choices=['chassis', 'top'])
    parser.add_argument('--n-frames', type=int, default=30, help='深度帧数（中值去噪）')
    parser.add_argument('--max-depth', type=float, default=10.0)
    parser.add_argument('--max-diff', type=float, default=0.15, help='差异着色范围 (m)')
    parser.add_argument('--point-size', type=int, default=3)
    parser.add_argument('--save', action='store_true', help='自动保存结果图')
    parser.add_argument('--skip-launch', action='store_true', help='跳过传感器启动（已在运行）')
    args = parser.parse_args()

    cfg = CAMERAS[args.camera]
    R, t = _lib_load_extrinsics(cfg['extrinsics'], as_point_transform=True)
    K, _ = load_intrinsics(cfg['intrinsics'])

    sensor_procs = []
    try:
        if not args.skip_launch:
            print("启动传感器...")
            sensor_procs = launch_sensors(args.camera)
            wait_for_topics(args.camera)
            print()

        print(f"相机: {args.camera}, 采集 {args.n_frames} 帧深度取中值")
        print(f"外参: {cfg['extrinsics']}")

        # 采集数据
        rclpy.init()
        collector = DataCollector(args.camera, args.n_frames)

        print("等待数据...")
        t0 = time.time()
        while not collector.is_ready():
            rclpy.spin_once(collector, timeout_sec=0.1)
            if time.time() - t0 > 60:
                print("超时，退出")
                collector.destroy_node()
                rclpy.shutdown()
                return

        rgb, depth_frames, pts_lidar = collector.get_data()
        collector.destroy_node()
        rclpy.shutdown()

        print(f"采集完成: {len(depth_frames)} 帧深度, {len(pts_lidar)} LiDAR 点")

        # 中值滤波
        print("计算中值深度...")
        depth_median_mm = compute_median_depth(depth_frames)
        depth_median_m = depth_median_mm / 1000.0

        valid_pix = np.sum(depth_median_m > 0.1)
        total_pix = depth_median_m.size
        print(f"中值深度: {valid_pix}/{total_pix} 有效像素 ({valid_pix/total_pix*100:.1f}%)")

        # 生成可视化
        print("生成可视化...")
        vis_lidar = make_lidar_overlay(rgb, pts_lidar, R, t, K, args.max_depth, args.point_size)
        vis_diff, stats = make_depth_diff_map(rgb, pts_lidar, R, t, K, depth_median_m,
                                               args.max_diff, args.point_size)
        vis_rs = make_rs_depth_vis(depth_median_m, args.max_depth)

        if stats:
            print_stats(stats)

        # 在差异图上标注统计
        if 'all' in stats:
            s = stats['all']
            text = f"mean={s['mean']*100:+.1f}cm  |err|={s['abs_mean']*100:.1f}cm  std={s['std']*100:.1f}cm  n={s['n']}"
            cv2.putText(vis_diff, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis_diff, f"green=match  red=LiDAR far  blue=LiDAR near  (range=+/-{args.max_diff*100:.0f}cm)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(vis_lidar, f"LiDAR overlay ({args.camera})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_rs, f"RealSense median depth ({len(depth_frames)} frames)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 保存
        if args.save:
            ts = time.strftime('%Y%m%d_%H%M%S')
            base = os.path.join(PROJECT_ROOT, 'scripts')
            for name, img in [('lidar', vis_lidar), ('diff', vis_diff), ('rs_depth', vis_rs)]:
                path = os.path.join(base, f'depth_compare_{args.camera}_{name}_{ts}.png')
                cv2.imwrite(path, img)
                print(f"保存: {path}")

        # 显示
        print("\n按任意键切换, Q/ESC 退出")
        views = [('LiDAR Overlay', vis_lidar), ('Depth Diff', vis_diff), ('RS Depth', vis_rs)]
        idx = 1  # 默认显示 diff

        while True:
            title, img = views[idx]
            scale = min(1.0, 1200.0 / img.shape[1])
            show = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img
            cv2.imshow('Static Depth Compare', show)
            cv2.setWindowTitle('Static Depth Compare', f'{title} [{idx+1}/{len(views)}] - Any key to switch')

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                path = os.path.join(PROJECT_ROOT, 'scripts', f'depth_compare_{args.camera}_{ts}.png')
                cv2.imwrite(path, img)
                print(f"保存: {path}")
            else:
                idx = (idx + 1) % len(views)

        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n退出...")
    finally:
        if sensor_procs:
            print("清理传感器进程...")
            stop_sensors(sensor_procs)
            print("完成")


if __name__ == '__main__':
    main()
