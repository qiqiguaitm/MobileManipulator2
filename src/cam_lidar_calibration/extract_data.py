#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标定数据提取工具
从 rosbag 文件中提取图像和点云数据，用于相机-激光雷达联合标定
"""

import os
import sys
import subprocess
import argparse
import datetime
import shutil
import glob

import rosbag2_py
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import cv2
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
import struct


# 相机配置 (与 collect_data.py 保持一致)
CAMERA_CONFIG = {
    'top': {
        'name': 'Top Camera',
        'image_topic': '/camera/top/color/image_raw',
    },
    'chassis': {
        'name': 'Chassis Camera',
        'image_topic': '/camera/chassis/color/image_raw',
    }
}

# LiDAR 配置
LIDAR_TOPIC = '/rslidar_points'


def read_messages_from_bag(bag_path, topic, msg_type):
    """从 ROS2 bag 目录读取指定 topic 的所有消息"""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    topic_filter = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(topic_filter)

    msgs = []
    while reader.has_next():
        (_, data, _) = reader.read_next()
        msg = deserialize_message(data, msg_type)
        msgs.append(msg)

    return msgs


def extract_image_from_bag(bag_path, image_topic, output_path, bridge):
    """从 bag 文件提取中间帧图像"""
    try:
        msgs = read_messages_from_bag(bag_path, image_topic, Image)
        if not msgs:
            print("  [警告] 在 bag 中未找到图像消息: %s" % image_topic)
            return False

        mid_idx = len(msgs) // 2
        msg = msgs[mid_idx]
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(output_path, cv_img)
        return True
    except Exception as e:
        print("  [错误] 提取图像失败: %s" % str(e))
        return False


def extract_pointcloud_from_bag(bag_path, lidar_topic, output_path):
    """从 bag 文件提取中间帧点云并保存为 PCD 格式"""
    try:
        msgs = read_messages_from_bag(bag_path, lidar_topic, PointCloud2)
        if not msgs:
            print("  [警告] 在 bag 中未找到点云消息: %s" % lidar_topic)
            return False

        # 选择中间帧
        mid_idx = len(msgs) // 2
        cloud_msg = msgs[mid_idx]

        # 提取点云数据
        points = list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")))
        if not points:
            print("  [警告] 点云数据为空")
            return False

        # 写入 PCD 文件（二进制格式）
        num_points = len(points)
        with open(output_path, 'wb') as f:
            # PCD 头部（ASCII）
            header = (
                "# .PCD v0.7 - Point Cloud Data file format\n"
                "VERSION 0.7\n"
                "FIELDS x y z\n"
                "SIZE 4 4 4\n"
                "TYPE F F F\n"
                "COUNT 1 1 1\n"
                "WIDTH %d\n"
                "HEIGHT 1\n"
                "VIEWPOINT 0 0 0 1 0 0 0\n"
                "POINTS %d\n"
                "DATA binary\n" % (num_points, num_points)
            )
            f.write(header.encode('ascii'))

            # 点云数据（二进制）
            for p in points:
                f.write(struct.pack('fff', p[0], p[1], p[2]))

        return True
    except Exception as e:
        print("  [错误] 提取点云失败: %s" % str(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description='标定数据提取工具 - 从 rosbag 提取图像和点云',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python extract_data.py top       # 提取 Top Camera 标定数据
  python extract_data.py chassis   # 提取 Chassis Camera 标定数据
  python extract_data.py top -i ./my_bags -o ./my_output
        '''
    )
    parser.add_argument(
        'camera',
        choices=['top', 'chassis'],
        help='选择相机类型: top (顶部相机) 或 chassis (底盘相机)'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='输入目录 (bag 文件目录), 默认为 ./data/calib_bags_<camera>'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出目录, 默认为 ./data/calib_data_<camera>'
    )

    args = parser.parse_args()

    # 获取相机配置
    cam_cfg = CAMERA_CONFIG[args.camera]
    image_topic = cam_cfg['image_topic']

    # 设置输入目录（默认查找最新的采集目录）
    if args.input:
        bag_dir = args.input
    else:
        pattern = "./data/calib_bags_%s_*" % args.camera
        candidates = sorted(glob.glob(pattern))
        if candidates:
            bag_dir = candidates[-1]  # 按字典序最新（即时间戳最大）
            print("[自动选择] 最新采集目录: %s" % bag_dir)
        else:
            bag_dir = "./data/calib_bags_%s" % args.camera  # 兼容旧目录

    # 设置输出目录（默认带时间戳，避免覆盖历史数据）
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = "./data/calib_data_%s_%s" % (args.camera, timestamp)

    img_out = os.path.join(output_dir, "image")
    pcd_out = os.path.join(output_dir, "pcd")

    # 检查输入目录
    if not os.path.exists(bag_dir):
        print("[错误] 输入目录不存在: %s" % bag_dir)
        print("       请先运行 collect_data.py %s 采集数据" % args.camera)
        sys.exit(1)

    # 创建输出目录
    for d in [img_out, pcd_out]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 打印配置
    print("=" * 60)
    print("     标定数据提取工具")
    print("=" * 60)
    print("")
    print("[相机配置]")
    print("  名称:     %s" % cam_cfg['name'])
    print("  图像话题: %s" % image_topic)
    print("")
    print("[LiDAR配置]")
    print("  点云话题: %s" % LIDAR_TOPIC)
    print("")
    print("[路径配置]")
    print("  输入目录: %s" % bag_dir)
    print("  输出目录: %s" % output_dir)
    print("")

    # 获取 bag 文件列表
    bag_files = sorted([
        d for d in os.listdir(bag_dir)
        if os.path.isdir(os.path.join(bag_dir, d))
        and os.path.exists(os.path.join(bag_dir, d, "metadata.yaml"))
    ])
    if not bag_files:
        print("[错误] 未找到 bag 数据目录: %s" % bag_dir)
        sys.exit(1)

    print("[处理] 找到 %d 个 bag 文件" % len(bag_files))
    print("-" * 60)

    bridge = CvBridge()
    success_count = 0

    for i, bag_name in enumerate(bag_files):
        bag_path = os.path.join(bag_dir, bag_name)
        print("\n[%d/%d] 处理: %s" % (i + 1, len(bag_files), bag_name))

        img_path = os.path.join(img_out, "%d.png" % i)
        pcd_path = os.path.join(pcd_out, "%d.pcd" % i)

        # 提取图像
        img_ok = extract_image_from_bag(bag_path, image_topic, img_path, bridge)
        if img_ok:
            print("  [OK] 图像: %s" % img_path)

        # 提取点云
        pcd_ok = extract_pointcloud_from_bag(bag_path, LIDAR_TOPIC, pcd_path)
        if pcd_ok:
            print("  [OK] 点云: %s" % pcd_path)

        if img_ok and pcd_ok:
            success_count += 1

    # 复制相机内参文件到输出目录
    intrinsic_src = os.path.join(bag_dir, "camera_intrinsics_%s.yaml" % args.camera)
    intrinsic_dst = os.path.join(output_dir, "camera_intrinsics_%s.yaml" % args.camera)
    if os.path.exists(intrinsic_src):
        shutil.copy2(intrinsic_src, intrinsic_dst)
        print("\n[OK] 已复制相机内参文件")
    else:
        print("\n[警告] 未找到相机内参文件: %s" % intrinsic_src)
        print("       标定时需要手动指定内参")

    print("\n" + "=" * 60)
    print("提取完成！")
    print("  成功: %d / %d 样本" % (success_count, len(bag_files)))
    print("")
    print("输出目录结构:")
    print("  %s/" % output_dir)
    print("    ├── image/   (PNG 图像)")
    print("    └── pcd/     (PCD 点云)")
    print("=" * 60)


if __name__ == "__main__":
    main()
