#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SC-PGO 地图合并工具 (简化版，使用pypcd4)
"""

import sys
import numpy as np
from pathlib import Path
from pypcd4 import PointCloud

def load_optimized_poses(poses_file):
    """加载 SC-PGO 优化后的位姿文件"""
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            values = [float(x) for x in line.split()]
            pose = np.array(values).reshape(3, 4)
            pose_4x4 = np.vstack([pose, [0, 0, 0, 1]])
            poses.append(pose_4x4)
    return poses


def merge_scans(map_dir, output_name="GlobalMap", voxel_size=0.02, near_removal=0.5):
    """合并所有扫描帧到全局地图"""
    map_dir = Path(map_dir)
    scans_dir = map_dir / "Scans"
    poses_file = map_dir / "optimized_poses.txt"

    if not scans_dir.exists():
        print(f"[ERROR] Scans 目录不存在: {scans_dir}")
        return False
    if not poses_file.exists():
        print(f"[ERROR] 位姿文件不存在: {poses_file}")
        return False

    print(f"[INFO] 加载位姿: {poses_file}")
    poses = load_optimized_poses(poses_file)
    print(f"[INFO] 共 {len(poses)} 个位姿")

    scan_files = sorted([f for f in scans_dir.iterdir() if f.suffix == '.pcd'])
    print(f"[INFO] 共 {len(scan_files)} 个扫描文件")

    if len(scan_files) != len(poses):
        print(f"[WARNING] 扫描文件数量 ({len(scan_files)}) 与位姿数量 ({len(poses)}) 不匹配")
        count = min(len(scan_files), len(poses))
        scan_files = scan_files[:count]
        poses = poses[:count]

    # 合并点云 - 使用字典做体素降采样
    print(f"[INFO] 开始合并点云 (体素大小: {voxel_size}m)")
    voxel_dict = {}

    for i, (scan_file, pose) in enumerate(zip(scan_files, poses)):
        try:
            pc = PointCloud.from_path(str(scan_file))
            points = pc.numpy()[:, :3]  # 只取xyz

            # 近距离点过滤
            if near_removal > 0:
                distances = np.linalg.norm(points, axis=1)
                points = points[distances > near_removal]

            # 变换到全局坐标系
            ones = np.ones((len(points), 1))
            points_h = np.hstack([points, ones])
            points_transformed = (pose @ points_h.T).T[:, :3]

            # 体素降采样 (用字典实现)
            for p in points_transformed:
                key = (int(p[0]/voxel_size), int(p[1]/voxel_size), int(p[2]/voxel_size))
                if key not in voxel_dict:
                    voxel_dict[key] = p

        except Exception as e:
            print(f"[WARN] 跳过 {scan_file.name}: {e}")
            continue

        if (i + 1) % 500 == 0 or i == len(scan_files) - 1:
            print(f"[INFO] 处理进度: {i + 1}/{len(scan_files)}, 当前点数: {len(voxel_dict)}")

    # 转换为数组
    all_points = np.array(list(voxel_dict.values()), dtype=np.float32)
    print(f"[INFO] 合并完成, 总点数: {len(all_points)}")

    # 保存PCD
    output_file = map_dir / f"{output_name}.pcd"

    # 创建PCD头
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(all_points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(all_points)}
DATA binary
"""

    with open(output_file, 'wb') as f:
        f.write(header.encode('ascii'))
        all_points.tofile(f)

    print(f"[SUCCESS] 地图已保存: {output_file}")
    return True


def main():
    default_map_dir = "/home/didi/workspace/MobileManipulator2/maps/sc_pgo"

    map_dir = sys.argv[1] if len(sys.argv) > 1 else default_map_dir
    output_name = sys.argv[2] if len(sys.argv) > 2 else "GlobalMap"
    voxel_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02

    print("=" * 60)
    print("SC-PGO 地图合并工具 (简化版)")
    print("=" * 60)
    print(f"地图目录: {map_dir}")
    print(f"输出名称: {output_name}")
    print(f"体素大小: {voxel_size}m")
    print("=" * 60)

    success = merge_scans(map_dir, output_name, voxel_size)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
