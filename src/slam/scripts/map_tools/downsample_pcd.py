#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCD 点云降采样 (简化版，使用pypcd4)
用法: python3 downsample_pcd_simple.py <input.pcd> <output.pcd> [voxel_size]
"""

import sys
import numpy as np
from pathlib import Path
from pypcd4 import PointCloud


def downsample_pcd(input_path, output_path, voxel_size=0.1):
    """对 PCD 进行体素降采样"""
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"加载: {input_path}")
    pc = PointCloud.from_path(str(input_path))
    points = pc.numpy()[:, :3]
    original_count = len(points)
    print(f"  原始点数: {original_count}")

    print(f"降采样: voxel_size={voxel_size}m")

    # 体素降采样
    voxel_dict = {}
    for p in points:
        key = (int(p[0]/voxel_size), int(p[1]/voxel_size), int(p[2]/voxel_size))
        if key not in voxel_dict:
            voxel_dict[key] = p

    down_points = np.array(list(voxel_dict.values()), dtype=np.float32)
    down_count = len(down_points)
    ratio = down_count / original_count * 100
    print(f"  降采样后: {down_count} ({ratio:.1f}%)")

    # 保存PCD
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(down_points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(down_points)}
DATA binary
"""

    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        down_points.tofile(f)

    print(f"保存: {output_path}")
    return down_count


def main():
    if len(sys.argv) < 3:
        print("用法: python3 downsample_pcd_simple.py <input.pcd> <output.pcd> [voxel_size]")
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    voxel_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    if not Path(input_path).exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        return 1

    downsample_pcd(input_path, output_path, voxel_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
