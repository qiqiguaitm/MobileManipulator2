#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 3D PCD 点云生成 2D 占据栅格地图 (简化版，使用pypcd4)
用法: python3 pcd_to_2dmap_simple.py <input.pcd> <output_dir> [resolution]
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from pypcd4 import PointCloud


def pcd_to_2dmap(pcd_file, output_dir, resolution=0.05,
                 min_height=0.05, max_height=1.5, padding=1.0):
    """从 PCD 生成 2D 占据栅格地图"""
    pcd_file = Path(pcd_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载点云: {pcd_file}")
    pc = PointCloud.from_path(str(pcd_file))
    points = pc.numpy()[:, :3]
    print(f"  总点数: {len(points)}")

    if len(points) == 0:
        print("[ERROR] 点云为空")
        return False

    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    print(f"  Z范围: [{z_min:.2f}, {z_max:.2f}]m")

    # 障碍物: 地面以上一定高度范围
    obstacle_mask = (points[:, 2] > min_height) & (points[:, 2] < max_height)
    obstacle_points = points[obstacle_mask]
    print(f"  障碍物点数 ({min_height}m < z < {max_height}m): {len(obstacle_points)}")

    if len(obstacle_points) == 0:
        print("[WARN] 没有障碍物点，使用全部点")
        obstacle_points = points

    # 计算地图边界
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    print(f"  地图尺寸: {width}x{height} pixels ({(x_max-x_min):.1f}x{(y_max-y_min):.1f}m)")

    # 创建地图 (初始化为未知=205)
    grid = np.full((height, width), 205, dtype=np.uint8)

    # 标记障碍物 (黑色=0)
    for p in obstacle_points:
        px = int((p[0] - x_min) / resolution)
        py = height - 1 - int((p[1] - y_min) / resolution)
        if 0 <= px < width and 0 <= py < height:
            grid[py, px] = 0

    # 标记自由空间: 使用地面点 (z < min_height)
    ground_mask = points[:, 2] <= min_height
    ground_points = points[ground_mask]
    print(f"  地面点数 (z <= {min_height}m): {len(ground_points)}")

    for p in ground_points:
        px = int((p[0] - x_min) / resolution)
        py = height - 1 - int((p[1] - y_min) / resolution)
        if 0 <= px < width and 0 <= py < height:
            if grid[py, px] == 205:
                grid[py, px] = 254

    # 统计
    free_count = np.sum(grid > 250)
    obstacle_count = np.sum(grid < 10)
    unknown_count = grid.size - free_count - obstacle_count

    print(f"  自由空间: {free_count} ({100*free_count/grid.size:.1f}%)")
    print(f"  障碍物:   {obstacle_count} ({100*obstacle_count/grid.size:.1f}%)")
    print(f"  未知:     {unknown_count} ({100*unknown_count/grid.size:.1f}%)")

    # 保存 PGM
    pgm_file = output_dir / "map.pgm"
    cv2.imwrite(str(pgm_file), grid)
    print(f"保存: {pgm_file}")

    # 保存 YAML
    yaml_file = output_dir / "map.yaml"
    yaml_content = f"""image: map.pgm
resolution: {resolution}
origin: [{x_min:.6f}, {y_min:.6f}, 0.0]
occupied_thresh: 0.65
free_thresh: 0.196
negate: 0
"""
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    print(f"保存: {yaml_file}")

    # 备份原始地图
    original_pgm = output_dir / "map_original.pgm"
    cv2.imwrite(str(original_pgm), grid)
    print(f"备份: {original_pgm}")

    return True


def main():
    if len(sys.argv) < 3:
        print("用法: python3 pcd_to_2dmap_simple.py <input.pcd> <output_dir> [resolution]")
        return 1

    pcd_file = sys.argv[1]
    output_dir = sys.argv[2]
    resolution = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    if not Path(pcd_file).exists():
        print(f"[ERROR] 文件不存在: {pcd_file}")
        return 1

    success = pcd_to_2dmap(pcd_file, output_dir, resolution)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
