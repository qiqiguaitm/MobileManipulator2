#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D地图改进工具 v3

设计原则：地图只保留两种值
  - 0   = 障碍物 (黑色)
  - 254 = 自由空间 (白色)

膨胀完全由 costmap 的 inflation_layer 统一处理，不在地图中预膨胀。
这样可以通过调整 inflation_radius 参数灵活控制安全边界。
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import yaml


def load_map(map_dir):
    """加载地图"""
    map_dir = Path(map_dir)
    yaml_file = map_dir / "map.yaml"

    with open(yaml_file, 'r') as f:
        map_info = yaml.safe_load(f)

    pgm_file = map_dir / map_info['image']
    img = cv2.imread(str(pgm_file), cv2.IMREAD_GRAYSCALE)

    return img, map_info


def analyze_map(img):
    """分析地图像素分布"""
    total = img.size
    free = np.sum(img > 250)
    obstacle = np.sum(img < 10)
    unknown = total - free - obstacle

    print(f"  自由空间 (>250): {free} ({100*free/total:.1f}%)")
    print(f"  障碍物 (<10):    {obstacle} ({100*obstacle/total:.1f}%)")
    print(f"  未知/其他:       {unknown} ({100*unknown/total:.1f}%)")


def main():
    map_dir = Path("/home/didi/workspace/MobileManipulator2/maps/sc_pgo")

    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            map_dir = Path(arg)

    print("=" * 60)
    print("2D地图改进工具 v3")
    print("=" * 60)
    print(f"地图目录: {map_dir}")
    print(f"\n设计原则: 地图只有 0(障碍物) 和 254(自由空间)")
    print(f"膨胀由 costmap inflation_layer 统一处理")

    # 加载原始地图
    original_pgm = map_dir / "map_original.pgm"
    if original_pgm.exists():
        img = cv2.imread(str(original_pgm), cv2.IMREAD_GRAYSCALE)
        map_info = yaml.safe_load(open(map_dir / "map.yaml"))
    else:
        img, map_info = load_map(map_dir)
        # 备份原始地图
        cv2.imwrite(str(map_dir / "map_original.pgm"), img)

    resolution = map_info.get('resolution', 0.05)

    print(f"\n地图尺寸: {img.shape[1]}x{img.shape[0]} pixels")
    print(f"分辨率: {resolution}m/pixel")

    print("\n原始地图:")
    analyze_map(img)

    # Step 1: 识别障碍物 (值 < 10)
    obstacle_mask = img < 10
    obstacle_uint8 = obstacle_mask.astype(np.uint8) * 255

    # Step 2: 移除孤立小障碍物 (噪点过滤)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle_uint8, connectivity=8)
    clean_obstacle_mask = np.zeros_like(obstacle_mask)
    small_removed = 0

    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 5:  # 保留 >= 5像素的障碍物
            clean_obstacle_mask |= (labels == i)
        else:
            small_removed += 1

    print(f"\n移除孤立小障碍物: {small_removed}个")

    # Step 3: 生成最终地图 (只有两种值)
    result = np.full_like(img, 254)  # 默认自由空间
    result[clean_obstacle_mask] = 0  # 障碍物

    print("\n改进后地图:")
    analyze_map(result)

    # 保存
    output_pgm = map_dir / "map.pgm"
    cv2.imwrite(str(output_pgm), result)
    print(f"\n已保存: {output_pgm}")

    # 生成对比图
    compare = np.zeros((img.shape[0], img.shape[1] * 2 + 10), dtype=np.uint8)
    compare[:, :img.shape[1]] = img
    compare[:, img.shape[1] + 10:] = result
    compare[:, img.shape[1]:img.shape[1] + 10] = 128

    compare_file = map_dir / "map_comparison.png"
    cv2.imwrite(str(compare_file), compare)
    print(f"对比图: {compare_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
