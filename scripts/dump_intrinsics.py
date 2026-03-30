#!/usr/bin/env python3
"""
从 ROS2 camera_info 话题获取相机内参，自动生成 MATLAB 可用的 .m 文件

使用方法:
    # 相机启动后运行
    python3 scripts/_cc_dump_intrinsics.py

    # 指定输出目录 (默认: 最新的 calibration_data/stereo_chessboard_* 目录)
    python3 scripts/_cc_dump_intrinsics.py /path/to/calibration_data/stereo_chessboard_xxx
"""
import os
import sys
import glob
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class IntrinsicsDumper(Node):
    def __init__(self, topics, output_dir):
        super().__init__('intrinsics_dumper')
        self.results = {}
        self.targets = set(topics)
        self.output_dir = output_dir
        for topic in topics:
            self.create_subscription(
                CameraInfo, topic,
                lambda msg, t=topic: self._cb(t, msg), 10
            )

    def _cb(self, topic, msg):
        if topic in self.results:
            return
        K = msg.k  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.results[topic] = {
            'fx': K[0], 'fy': K[4], 'cx': K[2], 'cy': K[5],
            'width': msg.width, 'height': msg.height,
            'D': list(msg.d),
        }
        name = 'Top' if 'top' in topic else 'Chassis'
        print(f"\n  {name} Camera: fx={K[0]:.4f} fy={K[4]:.4f} cx={K[2]:.4f} cy={K[5]:.4f} [{msg.height}x{msg.width}]")

        if len(self.results) == len(self.targets):
            self._save_matlab_file()
            rclpy.shutdown()

    def _save_matlab_file(self):
        """生成 camera_intrinsics.m 文件"""
        top = self.results.get('/camera/top/color/camera_info', {})
        ch = self.results.get('/camera/chassis/color/camera_info', {})

        if not top or not ch:
            print("ERROR: 未获取到两个相机的内参")
            return

        content = f"""%% camera_intrinsics.m
% 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
% 来源: ROS2 camera_info 话题 (RealSense 工厂标定)
% 注意: 此文件由 _cc_dump_intrinsics.py 自动生成，请勿手动修改

% ---- Top 相机内参 (Camera1) ----
fx_top = {top['fx']:.6f};
fy_top = {top['fy']:.6f};
cx_top = {top['cx']:.6f};
cy_top = {top['cy']:.6f};
imageSize_top = [{top['height']}, {top['width']}];

% ---- Chassis 相机内参 (Camera2) ----
fx_ch = {ch['fx']:.6f};
fy_ch = {ch['fy']:.6f};
cx_ch = {ch['cx']:.6f};
cy_ch = {ch['cy']:.6f};
imageSize_ch = [{ch['height']}, {ch['width']}];
"""
        out_path = os.path.join(self.output_dir, 'camera_intrinsics.m')
        with open(out_path, 'w') as f:
            f.write(content)

        print(f"\n{'='*50}")
        print(f"  已保存: {out_path}")
        print(f"{'='*50}")


def find_latest_calibration_dir():
    """查找最新的标定数据目录"""
    base = '/home/didi/workspace/MobileManipulator2/calibration_data'
    pattern = os.path.join(base, 'stereo_chessboard_*')
    dirs = sorted(glob.glob(pattern))
    if dirs:
        return dirs[-1]
    # 如果没有标定数据目录，创建一个通用目录
    os.makedirs(base, exist_ok=True)
    return base


def main():
    # 确定输出目录
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = find_latest_calibration_dir()

    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print("等待相机数据...")

    rclpy.init()
    topics = ['/camera/top/color/camera_info', '/camera/chassis/color/camera_info']
    node = IntrinsicsDumper(topics, output_dir)
    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == '__main__':
    main()
