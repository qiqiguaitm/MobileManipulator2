#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Camera 外参验证工具
将 LiDAR 点云投影到相机图像上，支持 RealSense 深度对比。

快捷键:
  1 - LiDAR 深度着色模式（默认）
  2 - LiDAR vs RealSense 深度差异模式
  3 - 仅原图（无叠加）
  Space - 闪烁切换（叠加/原图）
  S - 截图保存到 scripts/
  +/- - 调整点大小

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/verify_extrinsics.py --camera chassis --skip-launch
    python3 scripts/verify_extrinsics.py --camera both
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
from scipy.spatial.transform import Rotation

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

MODE_LIDAR_DEPTH = 1
MODE_DEPTH_DIFF = 2
MODE_IMAGE_ONLY = 3
MODE_NAMES = {MODE_LIDAR_DEPTH: 'LiDAR Depth', MODE_DEPTH_DIFF: 'Depth Diff', MODE_IMAGE_ONLY: 'Image Only'}


# ============================================================================
# 传感器启动 / 清理
# ============================================================================

def ros2_setup_env():
    env = os.environ.copy()
    env['PATH'] = '/usr/bin:' + env.get('PATH', '')
    env['ROS_DOMAIN_ID'] = env.get('ROS_DOMAIN_ID', '42')
    for key in ('LD_LIBRARY_PATH', 'PATH'):
        if key in env:
            env[key] = ':'.join(p for p in env[key].split(':') if 'miniconda' not in p)
    return env


def launch_sensor(cmd, label, env):
    shell_cmd = f'source /opt/ros/humble/setup.bash && source {PROJECT_ROOT}/install/setup.bash 2>/dev/null; {cmd}'
    print(f"  启动 {label}: {cmd}")
    proc = subprocess.Popen(
        ['bash', '-c', shell_cmd],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    return proc


def launch_sensors(camera_names):
    env = ros2_setup_env()
    procs = []
    procs.append(launch_sensor('ros2 launch rslidar_sdk start.py', 'LiDAR (rslidar)', env))
    top_en = 'true' if 'top' in camera_names else 'false'
    chassis_en = 'true' if 'chassis' in camera_names else 'false'
    cam_cmd = (f'ros2 launch camera_driver camera_driver.launch.py '
               f'top_enable:={top_en} chassis_enable:={chassis_en} hand_enable:=false')
    procs.append(launch_sensor(cam_cmd, f'Camera (top={top_en}, chassis={chassis_en})', env))
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


def wait_for_topics(camera_names, timeout=30):
    topics_needed = [LIDAR_TOPIC]
    for name in camera_names:
        topics_needed.append(CAMERAS[name]['image_topic'])
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
            for t in topics_needed:
                if t in active and t not in ready:
                    ready.add(t)
                    print(f"    ✓ {t}")
            if ready >= set(topics_needed):
                print("  全部话题就绪")
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        time.sleep(2)
    missing = set(topics_needed) - ready
    print(f"  ⚠ 超时，缺少: {missing}")
    return False


# ============================================================================
# 标定加载 / 投影
# ============================================================================

def load_extrinsics(yaml_path):
    return _lib_load_extrinsics(yaml_path, as_point_transform=True)


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


def project_lidar_to_pixels(points_lidar, R, t, K, img_shape):
    """LiDAR→像素坐标，返回 (u, v, z_cam) 全部在图像内的有效点。"""
    pts_cam = (R @ points_lidar.T).T + t
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    if len(pts_cam) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0)

    z = pts_cam[:, 2]
    u = (K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).astype(np.float32)
    v = (K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).astype(np.float32)

    h, w = img_shape[:2]
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[valid].astype(int), v[valid].astype(int), z[valid]


def overlay_lidar_depth(image, u, v, z, point_size, max_depth):
    """模式1: LiDAR 深度着色叠加。"""
    if len(u) == 0:
        return image.copy()
    norm = np.clip(z / max_depth, 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_JET).reshape(-1, 3)
    vis = image.copy()
    # 向量化绘制
    for ps in range(point_size, 0, -1):
        vis[np.clip(v, 0, vis.shape[0]-1), np.clip(u, 0, vis.shape[1]-1)] = colors
    if point_size >= 2:
        coords = np.stack([u, v], axis=1)
        for pt, c in zip(coords, colors):
            cv2.circle(vis, tuple(pt), point_size, (int(c[0]), int(c[1]), int(c[2])), -1)
    return vis


def overlay_depth_diff(image, u, v, z_lidar, depth_image, point_size):
    """模式2: LiDAR vs RealSense 深度差异。绿=吻合，红=偏大，蓝=偏小。"""
    if len(u) == 0 or depth_image is None:
        return image.copy(), None

    z_rs = depth_image[v, u].astype(np.float32) / 1000.0  # mm → m
    valid = z_rs > 0.1  # RealSense 有效深度
    if np.sum(valid) == 0:
        return image.copy(), None

    u_v, v_v, z_l, z_r = u[valid], v[valid], z_lidar[valid], z_rs[valid]
    diff = z_l - z_r  # 正=LiDAR更远, 负=LiDAR更近

    # 着色: 差异映射到颜色 [-0.2m .. 0 .. +0.2m] → [蓝 .. 绿 .. 红]
    max_diff = 0.2
    norm = np.clip(diff / max_diff, -1, 1)  # [-1, 1]
    colors = np.zeros((len(norm), 3), dtype=np.uint8)
    # 绿色通道: |diff|越小越亮
    colors[:, 1] = ((1.0 - np.abs(norm)) * 255).astype(np.uint8)
    # 红色通道: diff > 0 (LiDAR更远)
    colors[:, 2] = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    # 蓝色通道: diff < 0 (LiDAR更近)
    colors[:, 0] = (np.clip(-norm, 0, 1) * 255).astype(np.uint8)

    vis = image.copy()
    coords = np.stack([u_v, v_v], axis=1)
    for pt, c in zip(coords, colors):
        cv2.circle(vis, tuple(pt), point_size, (int(c[0]), int(c[1]), int(c[2])), -1)

    stats = {
        'n_valid': int(np.sum(valid)),
        'n_total': len(u),
        'mean': float(np.mean(diff[valid[valid]])) if len(diff) > 0 else 0,
        'std': float(np.std(diff[valid[valid]])) if len(diff) > 0 else 0,
        'abs_mean': float(np.mean(np.abs(diff))),
        'median': float(np.median(diff)),
    }
    # recompute on the valid-only subset
    diff_v = diff
    stats['mean'] = float(np.mean(diff_v))
    stats['std'] = float(np.std(diff_v))
    stats['abs_mean'] = float(np.mean(np.abs(diff_v)))
    stats['median'] = float(np.median(diff_v))

    return vis, stats


# ============================================================================
# ROS2 Node
# ============================================================================

class ExtrinsicsVerifier(Node):
    def __init__(self, camera_names, point_size, max_depth):
        super().__init__('extrinsics_verifier')
        self.bridge = CvBridge()
        self.point_size = point_size
        self.max_depth = max_depth
        self.mode = MODE_LIDAR_DEPTH
        self.flicker_off = False

        self.cams = {}
        for name in camera_names:
            cfg = CAMERAS[name]
            R, t = load_extrinsics(cfg['extrinsics'])
            K, _ = load_intrinsics(cfg['intrinsics'])
            self.cams[name] = {
                'R': R, 't': t, 'K': K,
                'image': None, 'depth': None, 'lock': threading.Lock(),
            }
            self.get_logger().info(f"[{name}] t={t.round(3)}, fx={K[0,0]:.1f}")

        self.lidar_points = None
        self.lidar_lock = threading.Lock()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(PointCloud2, LIDAR_TOPIC, self._lidar_cb, qos)
        for name in self.cams:
            cfg = CAMERAS[name]
            self.create_subscription(Image, cfg['image_topic'],
                                     lambda msg, n=name: self._image_cb(n, msg), qos)
            self.create_subscription(Image, cfg['depth_topic'],
                                     lambda msg, n=name: self._depth_cb(n, msg), qos)

        self.get_logger().info(f"Ready. cameras={list(self.cams)}, mode=1:LiDAR 2:DepthDiff 3:Image")
        self.create_timer(0.1, self._tick)

    def _lidar_cb(self, msg):
        pts = pointcloud2_to_xyz(msg)
        with self.lidar_lock:
            self.lidar_points = pts

    def _image_cb(self, name, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"[{name}] cv_bridge: {e}")
            return
        with self.cams[name]['lock']:
            self.cams[name]['image'] = img

    def _depth_cb(self, name, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().warn(f"[{name}] depth cv_bridge: {e}")
            return
        with self.cams[name]['lock']:
            self.cams[name]['depth'] = depth

    def _tick(self):
        with self.lidar_lock:
            pts = self.lidar_points
        if pts is None or len(pts) == 0:
            return

        for name, cam in self.cams.items():
            with cam['lock']:
                img = cam['image']
                depth = cam['depth']
            if img is None:
                continue

            u, v, z = project_lidar_to_pixels(pts, cam['R'], cam['t'], cam['K'], img.shape)

            # 渲染
            if self.mode == MODE_IMAGE_ONLY or self.flicker_off:
                vis = img.copy()
                info_text = f"{name} | Image Only"
            elif self.mode == MODE_DEPTH_DIFF:
                vis, stats = overlay_depth_diff(img, u, v, z, depth, self.point_size)
                if stats:
                    info_text = (f"{name} | DepthDiff | "
                                 f"mean={stats['mean']*100:+.1f}cm "
                                 f"std={stats['std']*100:.1f}cm "
                                 f"|err|={stats['abs_mean']*100:.1f}cm "
                                 f"({stats['n_valid']}/{stats['n_total']} pts)")
                else:
                    has_depth = depth is not None
                    info_text = f"{name} | DepthDiff | {'no overlap' if has_depth else 'no depth'}"
            else:
                vis = overlay_lidar_depth(img, u, v, z, self.point_size, self.max_depth)
                info_text = f"{name} | LiDAR Depth | {len(u)} pts"

            # HUD
            cv2.putText(vis, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, "[1]LiDAR [2]Diff [3]Image [Space]Flicker [S]Save [+/-]Size",
                        (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            scale = min(1.0, 960.0 / vis.shape[1])
            if scale < 1.0:
                vis = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.imshow(f"Verify: {name}", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('1'):
            self.mode = MODE_LIDAR_DEPTH
            self.get_logger().info("Mode: LiDAR Depth")
        elif key == ord('2'):
            self.mode = MODE_DEPTH_DIFF
            self.get_logger().info("Mode: Depth Diff (LiDAR vs RealSense)")
        elif key == ord('3'):
            self.mode = MODE_IMAGE_ONLY
            self.get_logger().info("Mode: Image Only")
        elif key == ord(' '):
            self.flicker_off = not self.flicker_off
        elif key == ord('s') or key == ord('S'):
            ts = time.strftime('%Y%m%d_%H%M%S')
            for name in self.cams:
                path = os.path.join(PROJECT_ROOT, 'scripts', f'verify_{name}_{ts}.png')
                # 从当前显示窗口截取不方便，重新渲染一帧保存
                with self.cams[name]['lock']:
                    img_s = self.cams[name]['image']
                    depth_s = self.cams[name]['depth']
                if img_s is None:
                    continue
                u_s, v_s, z_s = project_lidar_to_pixels(
                    pts, self.cams[name]['R'], self.cams[name]['t'],
                    self.cams[name]['K'], img_s.shape)
                if self.mode == MODE_DEPTH_DIFF:
                    save_img, _ = overlay_depth_diff(img_s, u_s, v_s, z_s, depth_s, self.point_size)
                else:
                    save_img = overlay_lidar_depth(img_s, u_s, v_s, z_s, self.point_size, self.max_depth)
                cv2.imwrite(path, save_img)
                self.get_logger().info(f"Saved: {path}")
        elif key == ord('+') or key == ord('='):
            self.point_size = min(self.point_size + 1, 8)
            self.get_logger().info(f"Point size: {self.point_size}")
        elif key == ord('-') or key == ord('_'):
            self.point_size = max(self.point_size - 1, 1)
            self.get_logger().info(f"Point size: {self.point_size}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LiDAR-Camera 外参验证（一键启动）')
    parser.add_argument('--camera', choices=['chassis', 'top', 'both'], default='both')
    parser.add_argument('--skip-launch', action='store_true', help='跳过传感器启动（已在运行）')
    parser.add_argument('--point-size', type=int, default=2)
    parser.add_argument('--max-depth', type=float, default=15.0)
    args = parser.parse_args()

    cameras = list(CAMERAS.keys()) if args.camera == 'both' else [args.camera]
    sensor_procs = []

    try:
        if not args.skip_launch:
            print("=" * 50)
            print(" 启动传感器")
            print("=" * 50)
            sensor_procs = launch_sensors(cameras)
            wait_for_topics(cameras)
            print()

        print("=" * 50)
        print(" 外参验证")
        print("  [1] LiDAR深度  [2] 深度差异  [3] 原图")
        print("  [Space] 闪烁  [S] 截图  [+/-] 点大小")
        print("  Ctrl+C 退出")
        print("=" * 50)
        rclpy.init()
        node = ExtrinsicsVerifier(cameras, args.point_size, args.max_depth)
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n退出...")
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        if sensor_procs:
            print("清理传感器进程...")
            stop_sensors(sensor_procs)
            print("完成")


if __name__ == '__main__':
    main()
