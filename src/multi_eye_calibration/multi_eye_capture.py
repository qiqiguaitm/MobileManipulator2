#!/usr/bin/env python3
"""
Multi-Eye Stereo Capture: Chassis Camera <-> Hand Camera

直接使用 RealSense SDK 采集双相机棋盘格图像对，无 ROS 依赖。
输出图像对 + 法兰位姿 + MATLAB 加载脚本，用于后续立体标定。

按键:
    SPACE - 保存一对图像
    P     - 重新读取法兰位姿
    Q     - 退出

用法:
    python3 multi_eye_capture.py
    python3 multi_eye_capture.py --no-arm
    python3 multi_eye_capture.py --square-size 0.030 --target 50
"""

import os
import sys
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


# ── RealSense 轻量封装 ─────────────────────────────────────────────

class RealsenseCapture:
    """轻量 RealSense RGB 采集，只开彩色流。"""

    def __init__(self, serial, width=1280, height=720, fps=15):
        self.serial = str(serial)
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.intrinsics = None

    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            print(f"[ERROR] Camera {self.serial} start failed: {e}")
            self.pipeline = None
            return False

        # 等待 5 帧稳定
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(2000)
            except RuntimeError:
                pass

        # 提取内参
        stream = profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()
        self.intrinsics = intr
        print(f"[OK] Camera {self.serial}: {self.width}x{self.height}@{self.fps} "
              f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
        return True

    def grab(self):
        try:
            frames = self.pipeline.wait_for_frames(3000)
        except RuntimeError:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        rgb = np.asanyarray(color.get_data())
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def get_intrinsics_dict(self):
        i = self.intrinsics
        return {
            'fx': float(i.fx), 'fy': float(i.fy),
            'cx': float(i.ppx), 'cy': float(i.ppy),
            'width': i.width, 'height': i.height,
            'dist_model': str(i.model),
            'dist_coeffs': [float(c) for c in i.coeffs],
        }

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None


# ── 机械臂位姿只读 ─────────────────────────────────────────────────

class ArmPoseReader:
    """只读连接 Piper，读法兰位姿。仅 ConnectPort，不 enable/不移动。"""

    FACTOR = 1000  # SDK 原始值单位: 0.001mm / 0.001deg

    def __init__(self, can_name='can0'):
        self.can_name = can_name
        self.piper = None

    def connect(self):
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError:
            print("[WARN] piper_sdk not found, arm pose unavailable")
            return False

        try:
            self.piper = C_PiperInterface_V2(self.can_name)
            self.piper.ConnectPort()
            time.sleep(0.5)

            # 验证位姿反馈
            pose = self._raw_pose()
            if all(v == 0 for v in pose):
                print("[WARN] Arm pose all zeros - is the arm enabled?")
            else:
                print(f"[OK] Arm pose: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} mm")
            print(f"[OK] Arm connected on {self.can_name}")
            return True
        except Exception as e:
            print(f"[WARN] Arm connection failed: {e}")
            self.piper = None
            return False

    def _raw_pose(self):
        msg = self.piper.GetArmEndPoseMsgs()
        p = msg.end_pose
        return [float(v) / self.FACTOR for v in
                (p.X_axis, p.Y_axis, p.Z_axis, p.RX_axis, p.RY_axis, p.RZ_axis)]

    def read_flange_pose(self):
        vals = self._raw_pose()
        return {
            'x_mm': vals[0], 'y_mm': vals[1], 'z_mm': vals[2],
            'roll_deg': vals[3], 'pitch_deg': vals[4], 'yaw_deg': vals[5],
        }

    def disconnect(self):
        self.piper = None


# ── MATLAB 脚本生成 ─────────────────────────────────────────────────

def generate_matlab_script(output_dir, square_size, chassis_intr, hand_intr):
    ci = chassis_intr
    hi = hand_intr
    script = f"""%% Multi-Eye Stereo Calibration - MATLAB Loader
% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
%
% Usage:
%   1. cd to this directory in MATLAB
%   2. Run this script
%   3. Stereo Camera Calibrator will open with images loaded

%% Image paths
chassisDir = fullfile(pwd, 'chassis');
handDir    = fullfile(pwd, 'hand');

chassisImages = dir(fullfile(chassisDir, '*.png'));
handImages    = dir(fullfile(handDir, '*.png'));

[~, idx] = sort({{chassisImages.name}});
chassisImages = chassisImages(idx);
[~, idx] = sort({{handImages.name}});
handImages = handImages(idx);

chassisImageFiles = fullfile(chassisDir, {{chassisImages.name}});
handImageFiles    = fullfile(handDir, {{handImages.name}});

fprintf('Found %d image pairs\\n', length(chassisImageFiles));

%% Camera intrinsics (from RealSense)
% Chassis camera
chassisIntrinsics = cameraIntrinsics( ...
    [{ci['fx']:.4f}, {ci['fy']:.4f}], ...
    [{ci['cx']:.4f}, {ci['cy']:.4f}], ...
    [{ci['height']}, {ci['width']}]);

% Hand camera
handIntrinsics = cameraIntrinsics( ...
    [{hi['fx']:.4f}, {hi['fy']:.4f}], ...
    [{hi['cx']:.4f}, {hi['cy']:.4f}], ...
    [{hi['height']}, {hi['width']}]);

%% Launch calibrator
squareSize = {square_size};  % meters

stereoCameraCalibrator(chassisImageFiles, handImageFiles, ...
    'SquareSize', squareSize);

fprintf('Stereo Camera Calibrator launched\\n');
"""
    path = os.path.join(output_dir, 'load_for_matlab.m')
    with open(path, 'w') as f:
        f.write(script)
    print(f"[OK] MATLAB script: {path}")


# ── 主程序 ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Multi-eye stereo capture (chassis <-> hand)')
    p.add_argument('--chassis-serial', default='337122071540')
    p.add_argument('--hand-serial', default='337122071190')
    p.add_argument('--no-arm', action='store_true', help='Skip arm connection')
    p.add_argument('--can', default='can0', help='CAN interface for arm')
    p.add_argument('--square-size', type=float, default=0.025, help='Chessboard square size in meters')
    p.add_argument('--target', type=int, default=80, help='Target number of image pairs')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 启动相机 ──
    chassis_cam = RealsenseCapture(args.chassis_serial, args.width, args.height, args.fps)
    hand_cam = RealsenseCapture(args.hand_serial, args.width, args.height, args.fps)

    if not chassis_cam.start():
        print("[FATAL] Chassis camera failed")
        return 1
    if not hand_cam.start():
        print("[FATAL] Hand camera failed")
        chassis_cam.stop()
        return 1

    # ── 连接机械臂（可选）──
    arm_reader = None
    if not args.no_arm:
        arm_reader = ArmPoseReader(args.can)
        if not arm_reader.connect():
            arm_reader = None

    # ── 创建输出目录 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'calibration_data',
        f'multi_eye_{timestamp}')
    output_dir = os.path.normpath(output_dir)
    chassis_dir = os.path.join(output_dir, 'chassis')
    hand_dir = os.path.join(output_dir, 'hand')
    os.makedirs(chassis_dir, exist_ok=True)
    os.makedirs(hand_dir, exist_ok=True)

    # ── 保存法兰位姿 ──
    if arm_reader:
        pose = arm_reader.read_flange_pose()
        pose_path = os.path.join(output_dir, 'arm_pose.yaml')
        with open(pose_path, 'w') as f:
            yaml.dump({'flange_pose': pose, 'timestamp': timestamp}, f, default_flow_style=False)
        print(f"[OK] Arm pose saved: {pose}")

    # ── 保存元数据 ──
    metadata = {
        'chassis_serial': args.chassis_serial,
        'hand_serial': args.hand_serial,
        'resolution': f'{args.width}x{args.height}',
        'fps': args.fps,
        'square_size_m': args.square_size,
        'chassis_intrinsics': chassis_cam.get_intrinsics_dict(),
        'hand_intrinsics': hand_cam.get_intrinsics_dict(),
        'timestamp': timestamp,
    }
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    # ── 显示循环 ──
    capture_count = 0
    target = args.target
    win_name = 'Multi-Eye Capture'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 520)

    print(f"\nOutput: {output_dir}")
    print(f"Target: {target} pairs")
    print("[SPACE] Capture  [P] Re-read pose  [Q] Quit\n")

    while True:
        chassis_frame = chassis_cam.grab()
        hand_frame = hand_cam.grab()

        if chassis_frame is None or hand_frame is None:
            display = np.zeros((520, 1280, 3), dtype=np.uint8)
            cv2.putText(display, 'Waiting for cameras...', (400, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            h, w = 360, 640
            chassis_disp = cv2.resize(chassis_frame, (w, h))
            hand_disp = cv2.resize(hand_frame, (w, h))

            cv2.putText(chassis_disp, 'CHASSIS', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(hand_disp, 'HAND', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            combined = np.hstack([chassis_disp, hand_disp])

            # 状态栏
            bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
            color = (0, 255, 0) if capture_count < target else (0, 165, 255)
            cv2.putText(bar, f'Count: {capture_count}/{target}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            arm_status = 'ARM:OK' if arm_reader else 'ARM:OFF'
            cv2.putText(bar, f'[SPACE] Capture  [P] Pose  [Q] Quit  |  {arm_status}',
                        (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            display = np.vstack([combined, bar])

        cv2.imshow(win_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and chassis_frame is not None and hand_frame is not None:
            if capture_count >= target:
                print(f"[INFO] Already reached target {target}")
                continue
            capture_count += 1
            fname = f'{capture_count:04d}.png'
            cv2.imwrite(os.path.join(chassis_dir, fname), chassis_frame)
            cv2.imwrite(os.path.join(hand_dir, fname), hand_frame)
            print(f"[{capture_count}/{target}] Saved {fname}")

            if capture_count == 1:
                generate_matlab_script(
                    output_dir, args.square_size,
                    chassis_cam.get_intrinsics_dict(),
                    hand_cam.get_intrinsics_dict())

            if capture_count >= target:
                print("[INFO] Target reached!")
                time.sleep(1)

        elif key == ord('p') and arm_reader:
            pose = arm_reader.read_flange_pose()
            pose_path = os.path.join(output_dir, 'arm_pose.yaml')
            with open(pose_path, 'w') as f:
                yaml.dump({'flange_pose': pose, 'timestamp': timestamp}, f, default_flow_style=False)
            print(f"[OK] Pose updated: {pose}")

        elif key == ord('q'):
            break

    # ── 清理 ──
    cv2.destroyAllWindows()
    chassis_cam.stop()
    hand_cam.stop()
    if arm_reader:
        arm_reader.disconnect()

    # ── 摘要 ──
    chassis_files = [f for f in os.listdir(chassis_dir) if f.endswith('.png')]
    hand_files = [f for f in os.listdir(hand_dir) if f.endswith('.png')]
    print('\n' + '=' * 50)
    print(f'Output:  {output_dir}')
    print(f'Pairs:   {capture_count}')
    print(f'Files:   chassis={len(chassis_files)}, hand={len(hand_files)}')
    if len(chassis_files) == len(hand_files) and len(chassis_files) > 0:
        print('Pairing: OK')
    elif capture_count == 0:
        print('Pairing: No images captured')
    else:
        print('Pairing: MISMATCH!')
    print('=' * 50)
    return 0


if __name__ == '__main__':
    sys.exit(main())
