#!/usr/bin/env python3
"""
自适应地面过滤: 去除地面误检噪点，保持原始坐标
用于 process_map.sh 后处理
"""
import sys
import numpy as np
import open3d as o3d
from scipy import ndimage


def adaptive_ground_filter(input_pcd, output_pcd, grid_res=0.5, ground_margin=0.08,
                           obstacle_min=0.10, obstacle_max=1.8, denoise=True):
    """
    自适应地面过滤 (保持原始坐标)
    """
    print(f"[INFO] 加载点云: {input_pcd}")
    pcd = o3d.io.read_point_cloud(input_pcd)
    points = np.asarray(pcd.points)
    print(f"[INFO] 原始点数: {len(points):,}")

    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()
    grid_w = int(np.ceil((x_max - x_min) / grid_res))
    grid_h = int(np.ceil((y_max - y_min) / grid_res))

    print(f"[INFO] 自适应地面检测 (网格: {grid_w}x{grid_h})")

    # 每个网格取最低10%点的均值作为地面高度
    ground_grid = np.full((grid_h, grid_w), np.nan)
    for gy in range(grid_h):
        for gx in range(grid_w):
            mask = (points[:,0] >= x_min + gx*grid_res) & \
                   (points[:,0] < x_min + (gx+1)*grid_res) & \
                   (points[:,1] >= y_min + gy*grid_res) & \
                   (points[:,1] < y_min + (gy+1)*grid_res)
            cell_pts = points[mask, 2]
            if len(cell_pts) > 10:
                sorted_z = np.sort(cell_pts)
                n_low = max(1, len(sorted_z) // 10)
                ground_grid[gy, gx] = sorted_z[:n_low].mean()

    ground_mean = np.nanmean(ground_grid)
    ground_smooth = ndimage.uniform_filter(
        np.nan_to_num(ground_grid, nan=ground_mean), size=3)

    # 计算每个点的相对高度，同时保留原始坐标和相对高度
    # (x, y, z_original, z_relative)
    points_with_rel = []
    for p in points:
        gx = min(int((p[0] - x_min) / grid_res), grid_w - 1)
        gy = min(int((p[1] - y_min) / grid_res), grid_h - 1)
        local_ground = ground_smooth[gy, gx]
        relative_z = p[2] - local_ground

        # 保留地面和有效障碍物
        if relative_z < ground_margin:  # 地面
            points_with_rel.append([p[0], p[1], p[2], relative_z])
        elif obstacle_min < relative_z < obstacle_max:  # 障碍物
            points_with_rel.append([p[0], p[1], p[2], relative_z])

    points_with_rel = np.array(points_with_rel)
    print(f"[INFO] 地面过滤后: {len(points_with_rel):,}")

    # 去噪处理 (基于相对高度判断)
    if denoise and len(points_with_rel) > 0:
        res = 0.03
        w = int((x_max - x_min + 1) / res)
        h = int((y_max - y_min + 1) / res)

        # 障碍物mask基于相对高度
        obs_mask = points_with_rel[:,3] >= ground_margin
        obs_grid = np.zeros((h, w), dtype=np.int32)

        for p in points_with_rel[obs_mask]:
            px = int((p[0] - x_min) / res)
            py = int((p[1] - y_min) / res)
            if 0 <= px < w and 0 <= py < h:
                obs_grid[py, px] += 1

        obs_grid[obs_grid < 3] = 0
        kernel = np.ones((3, 3))
        obs_clean = ndimage.binary_opening((obs_grid > 0).astype(np.uint8), kernel)

        final_points = []
        for p in points_with_rel:
            if p[3] < ground_margin:  # 地面 (用相对高度判断)
                final_points.append([p[0], p[1], p[2]])  # 只保存原始坐标
            else:
                px = int((p[0] - x_min) / res)
                py = int((p[1] - y_min) / res)
                if 0 <= px < w and 0 <= py < h and obs_clean[py, px]:
                    final_points.append([p[0], p[1], p[2]])

        final_points = np.array(final_points)
        print(f"[INFO] 去噪后: {len(final_points):,}")
    else:
        final_points = points_with_rel[:, :3]

    # 保存
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(final_points)
    o3d.io.write_point_cloud(output_pcd, pcd_out)

    print(f"[INFO] 边界: X[{final_points[:,0].min():.2f}, {final_points[:,0].max():.2f}] "
          f"Y[{final_points[:,1].min():.2f}, {final_points[:,1].max():.2f}] "
          f"Z[{final_points[:,2].min():.2f}, {final_points[:,2].max():.2f}]")
    print(f"[SUCCESS] 已保存: {output_pcd}")

    return output_pcd


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 adaptive_ground_filter.py <input.pcd> <output.pcd>")
        sys.exit(1)

    adaptive_ground_filter(sys.argv[1], sys.argv[2])
