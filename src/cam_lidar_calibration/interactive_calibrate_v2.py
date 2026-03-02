#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式相机-LiDAR标定工具 v2

模块化重构版本：
- 从lib模块导入可复用函数
- 保留主流程和交互逻辑
- 功能与v1完全一致
"""

import os
import sys
import json
import datetime

# macOS环境变量设置
if sys.platform == 'darwin':
    os.environ['DISPLAY'] = ':0'

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

# ============================================================================
# 从lib模块导入可复用函数
# ============================================================================
from lib import (
    # IO
    load_extrinsics,
    load_intrinsics,
    load_pcd,
    save_extrinsics,
    # 变换
    invert_tf_extrinsics,
    transform_link_to_optical,
    project_camera_center_to_lidar,
    convert_optical_to_link_extrinsics,
    # 几何
    fit_rectangle_known_size,
    get_board_corners_3d,
    compute_lid_center_from_camera,
    validate_projected_rectangle,
    project_points,
    calculate_reprojection_error,
    draw_projection,
    # 平面拟合
    fit_ground_plane,
    fit_plane_ransac,
    # 质量评估
    evaluate_quality,
    print_quality_summary_table,
    print_frame_quality,
    # 棋盘检测
    detect_chessboard,
    draw_chessboard,
    # 标定求解
    solve_rotation_kabsch,
    solve_translation_plane_constraint,
    check_normal_diversity,
)


# ============================================================================
# ROI缓存管理
# ============================================================================

def get_roi_cache_path(data_dir, frame_idx):
    """获取ROI缓存文件路径"""
    cache_dir = os.path.join(data_dir, 'roi_cache')
    return os.path.join(cache_dir, f'frame_{frame_idx:03d}.json')


def load_roi_cache(data_dir, frame_idx):
    """加载ROI缓存"""
    cache_path = get_roi_cache_path(data_dir, frame_idx)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            print(f"  [缓存] 加载ROI: {os.path.basename(cache_path)}")
            return cache
        except Exception as e:
            print(f"  [缓存] 加载失败: {e}")
    return None


def save_roi_cache(data_dir, frame_idx, y_range, z_range):
    """保存ROI缓存"""
    cache_dir = os.path.join(data_dir, 'roi_cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = get_roi_cache_path(data_dir, frame_idx)
    cache = {
        'frame_idx': frame_idx,
        'y_range': list(y_range),
        'z_range': list(z_range),
    }

    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"  [缓存] 保存ROI: {os.path.basename(cache_path)}")


# ============================================================================
# 交互式ROI选择
# ============================================================================

# 全局变量用于鼠标回调
g_roi_start = None
g_roi_end = None
g_selecting = False
g_roi_confirmed = False


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数"""
    global g_roi_start, g_roi_end, g_selecting, g_roi_confirmed

    if event == cv2.EVENT_LBUTTONDOWN:
        g_roi_start = (x, y)
        g_roi_end = (x, y)
        g_selecting = True
        g_roi_confirmed = False
    elif event == cv2.EVENT_MOUSEMOVE and g_selecting:
        g_roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        g_roi_end = (x, y)
        g_selecting = False
        g_roi_confirmed = True


def auto_select_board_points(points_3d, cam_corners_3d, init_R, init_t, board_size, margin=0.05):
    """自动选择标定板区域的LiDAR点云"""
    R_inv = init_R.T
    t_inv = -init_R.T @ init_t
    lid_corners = []
    for corner_cam in cam_corners_3d:
        corner_lid = R_inv @ corner_cam + t_inv
        lid_corners.append(corner_lid)
    lid_corners = np.array(lid_corners)

    vec_h = lid_corners[1] - lid_corners[0]
    vec_v = lid_corners[3] - lid_corners[0]
    normal = np.cross(vec_h, vec_v)
    normal = normal / np.linalg.norm(normal)

    h_unit = vec_h / np.linalg.norm(vec_h)
    v_unit = vec_v / np.linalg.norm(vec_v)

    board_w, board_h = board_size
    h_half = board_w / 2 + margin
    v_half = board_h / 2 + margin
    depth_margin = 0.10

    center = np.mean(lid_corners, axis=0)

    selected_mask = np.zeros(len(points_3d), dtype=bool)
    for i, pt in enumerate(points_3d):
        vec = pt - center
        h_proj = np.dot(vec, h_unit)
        v_proj = np.dot(vec, v_unit)
        n_proj = np.dot(vec, normal)

        if (abs(h_proj) < h_half and abs(v_proj) < v_half and abs(n_proj) < depth_margin):
            selected_mask[i] = True

    selected_points = points_3d[selected_mask]

    print(f"  [自动选点] 投影中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  [自动选点] 筛选范围: h=±{h_half:.2f}m, v=±{v_half:.2f}m, depth=±{depth_margin:.2f}m")
    print(f"  [自动选点] 选中 {len(selected_points)}/{len(points_3d)} 个点")

    return selected_points, lid_corners


def select_board_roi_3d(points_3d, image, corners_2d, frame_idx, cam_distance=None, ground_z=None, ground_z_std=None,
                         data_dir=None, test_mode=False):
    """使用matplotlib 2D可视化选择LiDAR点云中的标定板区域"""
    # 测试模式：从缓存加载ROI
    if test_mode and data_dir:
        cache = load_roi_cache(data_dir, frame_idx)
        if cache:
            y1, y2 = cache['y_range']
            z1, z2 = cache['z_range']
            print(f"  [测试模式] 使用缓存ROI: Y[{y1:.2f}, {y2:.2f}], Z[{z1:.2f}, {z2:.2f}]")
            min_dist, max_dist = 1.0, 3.5
            margin = 0.01
            y1 -= margin
            y2 += margin
            z1 -= margin
            z2 += margin
            if ground_z is not None and ground_z_std is not None:
                z_lower_bound = ground_z + 3 * ground_z_std
            else:
                z_lower_bound = -0.35
            yz_mask = (points_3d[:, 1] >= y1) & (points_3d[:, 1] <= y2) & \
                      (points_3d[:, 2] >= max(z1, z_lower_bound)) & (points_3d[:, 2] <= z2)
            yz_candidates = points_3d[yz_mask]
            if len(yz_candidates) < 20:
                print(f"  [警告] YZ区域点数太少: {len(yz_candidates)}")
                return None
            x_median = np.median(yz_candidates[:, 0])
            x_std = np.std(yz_candidates[:, 0])
            x_min_auto = max(min_dist, x_median - 1.5 * max(x_std, 0.3))
            x_max_auto = min(max_dist, x_median + 1.5 * max(x_std, 0.3))
            z_min_final = max(z1, z_lower_bound)
            roi_mask = (points_3d[:, 0] >= x_min_auto) & (points_3d[:, 0] <= x_max_auto) & \
                       (points_3d[:, 1] >= y1) & (points_3d[:, 1] <= y2) & \
                       (points_3d[:, 2] >= z_min_final) & (points_3d[:, 2] <= z2)
            roi_points_3d = points_3d[roi_mask]
            print(f"  提取到 {len(roi_points_3d)} 个点")
            return roi_points_3d if len(roi_points_3d) >= 10 else None
        else:
            print(f"  [测试模式] 无缓存，跳过此帧")
            return None

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    print(f"\n=== Frame {frame_idx} 点云2D可视化选择 ===")
    print("操作说明:")
    print("  1. 左图：相机图像（参考）")
    print("  2. 右上：YZ侧视图（从右侧看）- 正对标定板")
    print("  3. 右下：XY俯视图")
    print("  4. 在YZ侧视图中拖动鼠标框选标定板区域")
    print("  5. 按Enter确认，按S跳过")

    print(f"  点云总数: {len(points_3d)}")
    print(f"  X范围: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] 中位数: {np.median(points_3d[:, 0]):.2f}")
    print(f"  Y范围: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] 中位数: {np.median(points_3d[:, 1]):.2f}")
    print(f"  Z范围: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] 中位数: {np.median(points_3d[:, 2]):.2f}")

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'Frame {frame_idx} - 共{len(points_3d)}个点', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(1, 2, 1)
    if corners_2d is not None and len(corners_2d) > 0:
        vis_img = image.copy()
        n_corners = len(corners_2d)
        if n_corners == 42:
            vis_pattern = (7, 6)
        elif n_corners == 48:
            vis_pattern = (8, 6)
        elif n_corners == 54:
            vis_pattern = (9, 6)
        else:
            vis_pattern = (7, 6)
        corners_draw = corners_2d.reshape(-1, 1, 2).astype(np.float32)
        cv2.drawChessboardCorners(vis_img, vis_pattern, corners_draw, True)
        ax1.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Camera Image (corners: {n_corners})', fontsize=12)
    else:
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Camera Image', fontsize=12)
    ax1.axis('off')

    ax2 = plt.subplot(2, 2, 2)

    if cam_distance is not None:
        min_dist = max(0.5, cam_distance - 1.5)
        max_dist = min(5.0, cam_distance + 1.5)
    else:
        min_dist = 1.0
        max_dist = 3.5

    x_mask = (points_3d[:, 0] >= min_dist) & (points_3d[:, 0] <= max_dist)
    display_points = points_3d[x_mask]
    print(f"  显示范围: x=[{min_dist:.1f}, {max_dist:.1f}]m, 显示{len(display_points)}个点")

    if len(display_points) > 0:
        colors = (display_points[:, 0] - display_points[:, 0].min()) / (display_points[:, 0].max() - display_points[:, 0].min() + 1e-6)
        ax2.scatter(display_points[:, 1], display_points[:, 2], c=colors, s=3, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('Y (m) - Left/Right')
    ax2.set_ylabel('Z (m) - Up/Down')
    ax2.set_title(f'YZ View (Side) - x in [{min_dist:.1f}, {max_dist:.1f}]m', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Ground Level')

    if ground_z is not None:
        ax2.axhline(y=ground_z, color='g', linestyle='--', alpha=0.8, linewidth=2, label=f'Fitted Ground (z={ground_z:.2f}m)')
        ax2.legend(loc='upper right', fontsize=8)

    ax3 = plt.subplot(2, 2, 4)
    if len(display_points) > 0:
        colors = (display_points[:, 2] - display_points[:, 2].min()) / (display_points[:, 2].max() - display_points[:, 2].min() + 1e-6)
        ax3.scatter(display_points[:, 0], display_points[:, 1], c=colors, s=3, cmap='plasma', alpha=0.7)
    ax3.set_xlabel('X (m) - Forward')
    ax3.set_ylabel('Y (m) - Left/Right')
    ax3.set_title('XY View (Top)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    selected_region = {'y_range': None, 'z_range': None}

    def onselect(eclick, erelease):
        y1, z1 = eclick.xdata, eclick.ydata
        y2, z2 = erelease.xdata, erelease.ydata
        y_min, y_max = min(y1, y2), max(y1, y2)
        z_min, z_max = min(z1, z2), max(z1, z2)
        selected_region['y_range'] = (y_min, y_max)
        selected_region['z_range'] = (z_min, z_max)
        print(f"  [选择] Y: [{y_min:.2f}, {y_max:.2f}], Z: [{z_min:.2f}, {z_max:.2f}]")

        yz_mask = (display_points[:, 1] >= y_min) & (display_points[:, 1] <= y_max) & \
                  (display_points[:, 2] >= z_min) & (display_points[:, 2] <= z_max)
        n_selected = np.sum(yz_mask)
        print(f"  [选择] 区域内点数: {n_selected}")

    selector = RectangleSelector(ax2, onselect, useblit=True,
                                  button=[1], minspanx=5, minspany=5,
                                  spancoords='pixels', interactive=True)

    result = {'action': None}

    def on_key(event):
        if event.key == 'enter':
            result['action'] = 'confirm'
            plt.close()
        elif event.key == 's':
            result['action'] = 'skip'
            plt.close()
        elif event.key == 'escape':
            result['action'] = 'skip'
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()

    if result['action'] == 'skip':
        print("  [跳过] 用户选择跳过此帧")
        return None

    if selected_region['y_range'] is None or selected_region['z_range'] is None:
        print("  [跳过] 未选择有效区域")
        return None

    y1, y2 = selected_region['y_range']
    z1, z2 = selected_region['z_range']

    if data_dir:
        save_roi_cache(data_dir, frame_idx, (y1, y2), (z1, z2))

    margin = 0.01
    y1 -= margin
    y2 += margin
    z1 -= margin
    z2 += margin

    if ground_z is not None and ground_z_std is not None:
        z_lower_bound = ground_z + 3 * ground_z_std
        if z1 < z_lower_bound:
            print(f"  [地面过滤] Z下限从 {z1:.3f}m 调整到 {z_lower_bound:.3f}m (ground_z + 3σ)")
            z1 = z_lower_bound
    else:
        z_lower_bound = -0.35

    yz_mask = (points_3d[:, 1] >= y1) & (points_3d[:, 1] <= y2) & \
              (points_3d[:, 2] >= z1) & (points_3d[:, 2] <= z2)
    yz_candidates = points_3d[yz_mask]

    if len(yz_candidates) < 20:
        print(f"  [警告] YZ区域点数太少: {len(yz_candidates)}")
        return None

    x_median = np.median(yz_candidates[:, 0])
    x_std = np.std(yz_candidates[:, 0])
    x_min_auto = max(min_dist, x_median - 1.5 * max(x_std, 0.3))
    x_max_auto = min(max_dist, x_median + 1.5 * max(x_std, 0.3))

    print(f"  [X范围] 自动估计: [{x_min_auto:.2f}, {x_max_auto:.2f}]m (median={x_median:.2f}, std={x_std:.3f})")

    roi_mask = (points_3d[:, 0] >= x_min_auto) & (points_3d[:, 0] <= x_max_auto) & \
               (points_3d[:, 1] >= y1) & (points_3d[:, 1] <= y2) & \
               (points_3d[:, 2] >= z1) & (points_3d[:, 2] <= z2)
    roi_points_3d = points_3d[roi_mask]

    print(f"  提取到 {len(roi_points_3d)} 个点")
    print(f"    X: [{roi_points_3d[:, 0].min():.2f}, {roi_points_3d[:, 0].max():.2f}]")
    print(f"    Y: [{roi_points_3d[:, 1].min():.2f}, {roi_points_3d[:, 1].max():.2f}]")
    print(f"    Z: [{roi_points_3d[:, 2].min():.2f}, {roi_points_3d[:, 2].max():.2f}]")

    if len(roi_points_3d) < 10:
        print(f"  [警告] 点数太少，返回None")
        return None

    return roi_points_3d


def auto_select_board_roi(points_2d, points_cam, points_3d, corners_2d, margin=50):
    """自动选择标定板区域"""
    if corners_2d is None or len(corners_2d) == 0:
        return None, None, None

    corners_min = corners_2d.min(axis=0)
    corners_max = corners_2d.max(axis=0)
    x_min, y_min = corners_min - margin
    x_max, y_max = corners_max + margin

    mask = (points_2d[:, 0] >= x_min) & (points_2d[:, 0] <= x_max) & \
           (points_2d[:, 1] >= y_min) & (points_2d[:, 1] <= y_max)

    roi_points_2d = points_2d[mask]
    roi_points_cam = points_cam[mask]
    roi_points_3d = points_3d[mask]

    return roi_points_2d, roi_points_cam, roi_points_3d


# ============================================================================
# 诊断与打印函数
# ============================================================================

def print_round_result(round_name, opt_R, opt_t, cost):
    """打印每轮优化结果的简要摘要"""
    opt_R_tf = opt_R.T
    opt_t_tf = -opt_R.T @ opt_t
    print(f"  {round_name} cost={cost:.6f}, t=[{opt_t_tf[0]:.4f}, {opt_t_tf[1]:.4f}, {opt_t_tf[2]:.4f}]")


def print_round_diagnostic(round_name, features_prev, features_curr, frame_ids, opt_R, opt_t):
    """打印每轮的诊断对比表格"""
    print(f"\n  [诊断] {round_name} 各帧对比:")
    print(f"  {'Frame':<5} {'lid_normal(前)':<22} {'lid_normal(后)':<22} {'Δn°':<5} "
          f"{'lid_center(前)':<26} {'lid_center(后)':<26} {'Δc_mm':<7} {'残差mm':<7}")
    print(f"  {'-'*150}")

    total_normal_changes = []
    total_center_corrections = []
    total_residuals = []

    for idx, (feat_prev, feat_curr, fid) in enumerate(zip(features_prev, features_curr, frame_ids)):
        _, _, lid_normal_prev, lid_center_prev = feat_prev
        cam_normal, cam_center, lid_normal_curr, lid_center_curr = feat_curr

        normal_dot = np.clip(np.dot(lid_normal_prev, lid_normal_curr), -1, 1)
        normal_change = np.degrees(np.arccos(abs(normal_dot)))
        total_normal_changes.append(normal_change)

        center_correction = np.linalg.norm(lid_center_curr - lid_center_prev) * 1000
        total_center_corrections.append(center_correction)

        lid_center_in_cam = opt_R @ lid_center_curr + opt_t
        residual = np.linalg.norm(lid_center_in_cam - cam_center) * 1000
        total_residuals.append(residual)

        print(f"  {fid:<5} "
              f"[{lid_normal_prev[0]:>5.2f},{lid_normal_prev[1]:>5.2f},{lid_normal_prev[2]:>5.2f}] "
              f"[{lid_normal_curr[0]:>5.2f},{lid_normal_curr[1]:>5.2f},{lid_normal_curr[2]:>5.2f}] "
              f"{normal_change:>4.1f} "
              f"[{lid_center_prev[0]:>6.3f},{lid_center_prev[1]:>6.3f},{lid_center_prev[2]:>6.3f}] "
              f"[{lid_center_curr[0]:>6.3f},{lid_center_curr[1]:>6.3f},{lid_center_curr[2]:>6.3f}] "
              f"{center_correction:>5.1f}   {residual:>5.1f}")

    print(f"  {'-'*150}")
    print(f"  统计: Δnormal={np.mean(total_normal_changes):.2f}±{np.std(total_normal_changes):.2f}°, "
          f"Δcenter={np.mean(total_center_corrections):.1f}±{np.std(total_center_corrections):.1f}mm, "
          f"残差={np.mean(total_residuals):.1f}±{np.std(total_residuals):.1f}mm")


# ============================================================================
# 外参优化
# ============================================================================

def solve_translation_hybrid(R, cam_centers, lid_normals, lid_plane_points, lid_centers,
                             plane_weight=0.3, verbose=True):
    """混合约束求解平移：平面约束 + 点约束"""
    t_estimates = []
    for cam_c, lid_c in zip(cam_centers, lid_centers):
        t_est = cam_c - R @ lid_c
        t_estimates.append(t_est)
    t_point = np.mean(t_estimates, axis=0)

    t_plane, _ = solve_translation_plane_constraint(R, cam_centers, lid_normals,
                                                     lid_plane_points, verbose=False)

    t_hybrid = plane_weight * t_plane + (1 - plane_weight) * t_point

    if verbose:
        print(f"\n  [混合约束] 权重: 平面={plane_weight:.1f}, 点={1-plane_weight:.1f}")
        print(f"    平面约束 t: [{t_plane[0]:.4f}, {t_plane[1]:.4f}, {t_plane[2]:.4f}]")
        print(f"    点约束 t:   [{t_point[0]:.4f}, {t_point[1]:.4f}, {t_point[2]:.4f}]")
        print(f"    混合 t:     [{t_hybrid[0]:.4f}, {t_hybrid[1]:.4f}, {t_hybrid[2]:.4f}]")
        diff = np.linalg.norm(t_plane - t_point)
        print(f"    两种估计差异: {diff*1000:.1f}mm")

    return t_hybrid


def optimize_extrinsics(features, K, init_R, init_t, verbose=True, fix_translation=False,
                        use_robust_cost=False):
    """标准外参优化（用于迭代refinement）"""
    def cost_function(x, return_details=False):
        rvec = x[:3]
        t = x[3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()

        total_cost = 0.0
        details = []

        for i, feat in enumerate(features):
            cam_n, cam_c, lid_n, lid_c = feat

            predicted_cam_n = R @ lid_n
            normal_cost = 1.0 - np.dot(predicted_cam_n, cam_n)

            predicted_cam_c = R @ lid_c + t
            center_error = np.linalg.norm(predicted_cam_c - cam_c)
            center_cost = center_error / 3.0

            if use_robust_cost:
                center_cost = np.sqrt(center_cost**2 + 0.01**2) - 0.01

            frame_cost = 2.0 * normal_cost + 1.0 * center_cost
            total_cost += frame_cost

            if return_details:
                details.append({
                    'frame': i,
                    'normal_cost': normal_cost,
                    'center_error': center_error,
                    'frame_cost': frame_cost
                })

        if return_details:
            return total_cost, details
        return total_cost

    init_rvec = Rotation.from_matrix(init_R).as_rotvec()
    init_x = np.concatenate([init_rvec, init_t])

    if fix_translation:
        def cost_fixed_t(rvec):
            return cost_function(np.concatenate([rvec, init_t]))
        result = minimize(cost_fixed_t, init_rvec, method='Powell',
                         options={'maxiter': 1000, 'ftol': 1e-7})
        opt_rvec = result.x
        opt_t = init_t.copy()
    else:
        result = minimize(cost_function, init_x, method='Powell',
                         options={'maxiter': 1000, 'ftol': 1e-7})
        opt_rvec = result.x[:3]
        opt_t = result.x[3:6]

    opt_R = Rotation.from_rotvec(opt_rvec).as_matrix()

    if verbose:
        init_cost, _ = cost_function(init_x, return_details=True)
        final_cost, final_details = cost_function(np.concatenate([opt_rvec, opt_t]), return_details=True)
        euler = Rotation.from_matrix(opt_R).as_euler('xyz', degrees=True)
        print(f"    初始 cost: {init_cost:.6f}")
        print(f"    最终 cost: {final_cost:.6f}")
        print(f"    旋转 (Euler XYZ): [{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]")
        print(f"    平移: [{opt_t[0]:.4f}, {opt_t[1]:.4f}, {opt_t[2]:.4f}]m")

    return opt_R, opt_t, result.fun


def optimize_extrinsics_v2(features_v2, K, init_R, init_t, verbose=True):
    """方案C优化：用相机角点引导LiDAR中心计算"""
    print("\n" + "="*70)
    print("方案C优化：相机角点引导LiDAR中心计算")
    print("="*70)

    precomputed_lid_centers = []
    for i, feat in enumerate(features_v2):
        cam_corners = feat['cam_corners_3d']
        cam_center = feat['cam_center']
        lid_normal = feat['lid_normal']
        lid_plane_point = feat['lid_plane_point']
        board_size = feat['board_size']

        lid_c, corners_on_plane, proj_errors = compute_lid_center_from_camera(
            cam_corners, cam_center, init_R, init_t, lid_normal, lid_plane_point
        )
        is_valid, metrics = validate_projected_rectangle(corners_on_plane, board_size)
        precomputed_lid_centers.append(lid_c)

    def cost_function(x, return_details=False):
        rvec = x[:3]
        t = x[3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()

        total_cost = 0.0
        details = []

        for i, feat in enumerate(features_v2):
            cam_c = feat['cam_center']
            cam_n = feat['cam_normal']
            lid_n = feat['lid_normal']
            lid_c = precomputed_lid_centers[i]

            predicted_cam_n = R @ lid_n
            normal_cost = 1.0 - np.dot(predicted_cam_n, cam_n)

            predicted_cam_c = R @ lid_c + t
            center_error = np.linalg.norm(predicted_cam_c - cam_c)
            center_cost = center_error / 3.0

            frame_cost = 2.0 * normal_cost + 1.0 * center_cost
            total_cost += frame_cost

            if return_details:
                details.append({
                    'frame': i,
                    'normal_cost': normal_cost,
                    'center_error': center_error,
                    'frame_cost': frame_cost,
                    'lid_center': lid_c.copy(),
                    'cam_dist': np.linalg.norm(cam_c),
                    'lid_dist': np.linalg.norm(lid_c)
                })

        if return_details:
            return total_cost, details
        return total_cost

    init_rvec = Rotation.from_matrix(init_R).as_rotvec()
    init_x = np.concatenate([init_rvec, init_t])

    init_cost, init_details = cost_function(init_x, return_details=True)
    print(f"\n  初始代价: {init_cost:.6f}")

    if verbose:
        print(f"\n  [逐帧分析 - 初始外参]")
        for d in init_details:
            status = "✓ OK" if d['center_error'] < 0.5 else "✗ BAD"
            print(f"    Frame {d['frame']}: normal_cost={d['normal_cost']:.3f}, "
                  f"center_err={d['center_error']*100:.1f}cm, "
                  f"cam_dist={d['cam_dist']:.2f}m, lid_dist={d['lid_dist']:.2f}m [{status}]")

    result = minimize(cost_function, init_x, method='Powell',
                     options={'maxiter': 2000, 'ftol': 1e-6})

    opt_x = result.x
    opt_R = Rotation.from_rotvec(opt_x[:3]).as_matrix()
    opt_t = opt_x[3:6]

    print(f"\n  最终代价: {result.fun:.6f} (初始: {init_cost:.6f})")
    print(f"  代价下降: {((init_cost - result.fun) / init_cost * 100):.1f}%")

    if verbose:
        final_cost, final_details = cost_function(opt_x, return_details=True)
        print(f"\n  [逐帧分析 - 优化后]")
        center_errors = []
        for d in final_details:
            status = "✓ OK" if d['center_error'] < 0.5 else "✗ BAD"
            print(f"    Frame {d['frame']}: normal_cost={d['normal_cost']:.3f}, "
                  f"center_err={d['center_error']*100:.1f}cm [{status}]")
            center_errors.append(d['center_error'])

        print(f"\n  [一致性统计]")
        print(f"    中心误差: 均值={np.mean(center_errors)*100:.2f}cm, "
              f"标准差={np.std(center_errors)*100:.2f}cm, "
              f"最大={np.max(center_errors)*100:.2f}cm")

        lid_centers = np.array([d['lid_center'] for d in final_details])
        lid_center_std = np.std(lid_centers, axis=0)
        print(f"    LiDAR中心标准差: X={lid_center_std[0]*1000:.1f}mm, "
              f"Y={lid_center_std[1]*1000:.1f}mm, Z={lid_center_std[2]*1000:.1f}mm")

    return opt_R, opt_t, result.fun


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 完整标定流程"""
    import argparse

    parser = argparse.ArgumentParser(description='交互式相机-LiDAR标定 v2')
    parser.add_argument('--data_dir', type=str,
                        default='./data/calib_data_top',
                        help='数据目录')
    parser.add_argument('--init_extrinsic', type=str,
                        default=None,
                        help='初始外参文件 (默认: ./config/init_extrinsics_lidar_to_<camera>.yaml)')
    parser.add_argument('--auto', action='store_true', help='自动模式(无GUI)')
    parser.add_argument('--test', action='store_true', help='测试模式（使用缓存ROI，无GUI交互）')
    args = parser.parse_args()

    # 测试模式
    test_mode = args.test
    if test_mode:
        print("[模式] 测试模式：使用缓存ROI区域，无GUI交互")

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载标定板配置
    board_config_file = os.path.join(script_dir, 'config', 'calibration_board.yaml')
    if not os.path.exists(board_config_file):
        print(f"[错误] 未找到标定板配置文件: {board_config_file}")
        sys.exit(1)

    with open(board_config_file, 'r') as f:
        board_config = yaml.safe_load(f)['chessboard']

    pattern_size = (board_config['pattern_cols'], board_config['pattern_rows'])
    square_size = board_config['square_size']
    board_width = board_config['board_width']
    board_height = board_config['board_height']

    # 推断相机类型
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    if 'top' in data_dir_name.lower():
        camera_type = 'top'
    elif 'chassis' in data_dir_name.lower():
        camera_type = 'chassis'
    else:
        camera_type = 'top'

    # 加载内参
    intrinsic_file = os.path.join(args.data_dir, f'camera_intrinsics_{camera_type}.yaml')
    if not os.path.exists(intrinsic_file):
        intrinsic_file = os.path.join(args.data_dir, 'conf', 'camera_intrinsic.yaml')

    if not os.path.exists(intrinsic_file):
        print(f"[错误] 未找到内参文件: {intrinsic_file}")
        sys.exit(1)

    print("[加载] 内参:", intrinsic_file)
    K, D = load_intrinsics(intrinsic_file)

    print(f"\n[标定板配置] {board_config_file}")
    print(f"  内角点: {pattern_size[0]}x{pattern_size[1]}")
    print(f"  方格边长: {square_size*100:.1f}cm")
    print(f"  物理尺寸: {board_width*100:.0f}cm × {board_height*100:.0f}cm")

    # 加载数据
    image_dir = os.path.join(args.data_dir, 'image')
    pcd_dir = os.path.join(args.data_dir, 'pcd')

    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    pcds = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    n_frames = min(len(images), len(pcds))

    print(f"[数据] 找到 {n_frames} 帧数据")

    # 加载初始外参
    init_extrinsic_file = args.init_extrinsic
    if init_extrinsic_file is None:
        init_extrinsic_file = os.path.join(script_dir, 'config',
                                          f'init_extrinsics_lidar_to_{camera_type}_camera.yaml')

    init_R, init_t = None, None
    if camera_type == 'top':
        output_child_frame = 'top_camera_optical_frame'
    else:
        output_child_frame = 'chassis_camera'

    if os.path.exists(init_extrinsic_file):
        print(f"\n[加载] 初始外参: {init_extrinsic_file}")
        R_tf, t_tf = load_extrinsics(init_extrinsic_file)
        init_R, init_t = invert_tf_extrinsics(R_tf, t_tf)

        with open(init_extrinsic_file, 'r') as f:
            data = yaml.safe_load(f)
        child_frame = data.get('child_frame_id', '')

        if child_frame:
            output_child_frame = child_frame

        if 'optical' not in child_frame.lower():
            print(f"  转换到 optical frame")
            from lib.transforms import transform_link_to_optical
            init_R, init_t = transform_link_to_optical(init_R, init_t)
            if not output_child_frame.endswith('_optical_frame'):
                output_child_frame = output_child_frame.replace('_camera', '_camera_optical_frame')

        print(f"  初始外参已加载，将用于投影引导选择")
    else:
        print(f"\n[警告] 初始外参不存在: {init_extrinsic_file}")
        init_R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        init_t = np.array([0.0, 0.46, 0.43])

    print("\n" + "="*60)
    print("3D投影引导标定 - 自动定位 + 交互确认")
    print("="*60 + "\n")

    # 收集特征
    features = []
    features_v2 = []
    frame_qualities = []
    accepted_frame_ids = []

    for i in range(n_frames):
        print(f"\n[Frame {i}] 处理中...")

        image_path = os.path.join(image_dir, images[i])
        pcd_path = os.path.join(pcd_dir, pcds[i])

        image = cv2.imread(image_path)
        points_3d = load_pcd(pcd_path)

        print(f"  加载点云: {len(points_3d)} 个点")

        # 检测棋盘格
        corners_2d, objp, _ = detect_chessboard(image, pattern_size, square_size)
        if corners_2d is None:
            print("  [跳过] 未检测到棋盘格")
            continue

        # 求解相机位姿
        ret, rvec, tvec = cv2.solvePnP(objp, corners_2d, K, D)
        if not ret:
            print("  [跳过] solvePnP 失败")
            continue

        reprojection_rms, reprojection_mean, reprojection_max, _ = \
            calculate_reprojection_error(objp, corners_2d, rvec, tvec, K, D)

        R_board, _ = cv2.Rodrigues(rvec)
        cam_center = tvec.flatten()
        cam_normal = R_board[:, 2]
        if cam_normal[2] > 0:
            cam_normal = -cam_normal

        cam_distance = np.linalg.norm(cam_center)

        ideal_normal = np.array([0, 0, -1])
        alignment_angle = np.degrees(np.arccos(np.clip(np.dot(cam_normal, ideal_normal), -1, 1)))
        tilt_angle = alignment_angle

        print(f"  相机检测: distance={cam_distance:.2f}m, 对齐角={tilt_angle:.1f}°")
        print(f"  重投影误差: RMS={reprojection_rms:.2f}px, Mean={reprojection_mean:.2f}px, Max={reprojection_max:.2f}px")

        # 拟合地面
        ground_result = fit_ground_plane(points_3d)
        ground_z_early, ground_z_std = (ground_result if ground_result else (None, None))

        # 手动框选标定板区域
        roi_points = select_board_roi_3d(points_3d, image, corners_2d, i, cam_distance=cam_distance,
                                          ground_z=ground_z_early, ground_z_std=ground_z_std,
                                          data_dir=args.data_dir, test_mode=test_mode)

        if roi_points is None or len(roi_points) == 0:
            print("  [跳过] 未选择有效区域")
            continue

        print(f"  选择了 {len(roi_points)} 个点")

        # 拟合平面
        lid_normal, lid_center, inliers = fit_plane_ransac(
            roi_points, board_size=(board_width, board_height), ground_z=ground_z_early, ground_z_std=ground_z_std,
            pattern_size=pattern_size, square_size=square_size, full_points=points_3d)

        if lid_normal is None:
            print("  [跳过] 平面拟合失败")
            continue

        inlier_ratio = len(inliers) / len(roi_points)
        distance_lid = np.linalg.norm(lid_center)

        # 计算距离一致性
        if init_R is not None and init_t is not None:
            cam_center_in_lidar = project_camera_center_to_lidar(cam_center, init_R, init_t)
            distance_diff = np.linalg.norm(cam_center_in_lidar - lid_center)
            lid_normal_in_cam = init_R @ lid_normal
            normal_diff = abs(np.dot(cam_normal, lid_normal_in_cam))
        else:
            cam_center_in_lidar = cam_center
            distance_diff = abs(cam_distance - distance_lid)
            normal_diff = abs(np.dot(cam_normal, lid_normal))

        grade, details, score = evaluate_quality(
            reprojection_rms, tilt_angle, inlier_ratio, distance_diff, normal_diff, len(roi_points))

        print_frame_quality(
            i, reprojection_rms, reprojection_mean, reprojection_max,
            tilt_angle, inlier_ratio, len(roi_points), len(inliers),
            distance_diff, cam_center_in_lidar, lid_center, normal_diff, grade, details, score)

        normal_angle = np.degrees(np.arccos(np.clip(normal_diff, -1, 1)))

        if normal_angle > 30:
            print(f"\n  [自动跳过] 法向量夹角过大({normal_angle:.1f}°>30°)")
            continue

        if grade in ['D']:
            print(f"\n  [警告] 帧质量过低({grade})")
            if test_mode:
                print("  [测试模式] 自动跳过D级帧")
                continue
            confirm = input("  仍要使用此帧？[y/N]: ").strip().lower()
            if confirm != 'y':
                continue

        features.append((cam_normal, cam_center, lid_normal, lid_center))
        accepted_frame_ids.append(i)

        cam_corners_3d = get_board_corners_3d(rvec, tvec, pattern_size, square_size, (board_width, board_height))
        lid_plane_point = np.mean(inliers, axis=0)
        features_v2.append({
            'cam_corners_3d': cam_corners_3d,
            'cam_center': cam_center,
            'cam_normal': cam_normal,
            'lid_normal': lid_normal,
            'lid_plane_point': lid_plane_point,
            'board_size': (board_width, board_height)
        })

        frame_qualities.append({
            'frame_id': i, 'reproj_rms': reprojection_rms, 'tilt_angle': tilt_angle,
            'inlier_ratio': inlier_ratio, 'distance_diff': distance_diff,
            'normal_angle': normal_angle, 'num_points': len(roi_points), 'grade': grade, 'score': score
        })

        print(f"\n  [✓ OK] 已添加特征 (共 {len(features)} 帧, 质量: {grade})")

    cv2.destroyAllWindows()

    if frame_qualities:
        print_quality_summary_table(frame_qualities)

    if len(features) < 3:
        print(f"\n[错误] 有效帧数不足: {len(features)} < 3")
        return

    # Kabsch旋转 + 迭代平移优化
    lid_plane_points = [f['lid_plane_point'] for f in features_v2]
    cam_normals = [f[0] for f in features]
    cam_centers = [f[1] for f in features]
    lid_normals = [f[2] for f in features]
    lid_centers = [f[3] for f in features]

    print("\n" + "="*70)
    print("组合策略：Kabsch旋转 + 迭代平移优化")
    print("="*70)

    # Kabsch求旋转
    print(f"\n[阶段1] Kabsch 算法求解旋转")
    R_kabsch, rot_residuals, inlier_mask = solve_rotation_kabsch(cam_normals, lid_normals, verbose=True)

    # 计算初始平移
    print(f"\n[阶段2] 用 Kabsch 旋转计算初始平移")
    t_estimates_point = []
    for cam_c, lid_c in zip(cam_centers, lid_centers):
        t_est = cam_c - R_kabsch @ lid_c
        t_estimates_point.append(t_est)
    t_point = np.mean(t_estimates_point, axis=0)

    R_T = R_kabsch.T
    A = np.array(lid_normals)
    b = []
    for cam_c, lid_c, lid_n in zip(cam_centers, lid_centers, lid_normals):
        rhs = np.dot(R_T @ cam_c, lid_n) - np.dot(lid_c, lid_n)
        b.append(rhs)
    b = np.array(b)
    u_plane, _, rank, s = np.linalg.lstsq(A, b, rcond=None)
    t_plane = R_kabsch @ u_plane

    lid_n_array = np.array(lid_normals)
    n_std = np.std(lid_n_array, axis=0)
    print(f"  法向量标准差: X={n_std[0]:.3f}, Y={n_std[1]:.3f}, Z={n_std[2]:.3f}")

    diff = np.linalg.norm(t_point - t_plane)
    if diff < 0.02:
        t_init_kabsch = 0.9 * t_point + 0.1 * t_plane
        print(f"  [策略] 两种约束一致(差异{diff*1000:.1f}mm)，轻微混合")
    else:
        t_init_kabsch = t_point.copy()
        print(f"  [策略] 两种约束差异大({diff*1000:.1f}mm)，使用点约束")

    print(f"  点约束平移:   [{t_point[0]:.4f}, {t_point[1]:.4f}, {t_point[2]:.4f}]m")
    print(f"  平面约束平移: [{t_plane[0]:.4f}, {t_plane[1]:.4f}, {t_plane[2]:.4f}]m")
    print(f"  最终初始化:   [{t_init_kabsch[0]:.4f}, {t_init_kabsch[1]:.4f}, {t_init_kabsch[2]:.4f}]m")

    def compute_cost(R, t, features):
        total = 0.0
        for cam_n, cam_c, lid_n, lid_c in features:
            normal_cost = 1.0 - np.dot(R @ lid_n, cam_n)
            center_error = np.linalg.norm(R @ lid_c + t - cam_c)
            total += 2.0 * normal_cost + center_error / 3.0
        return total

    kabsch_cost = compute_cost(R_kabsch, t_init_kabsch, features)
    init_cost = compute_cost(init_R, init_t, features)
    print(f"  Kabsch起点代价: {kabsch_cost:.6f}")
    print(f"  初始外参代价:   {init_cost:.6f}")

    if kabsch_cost < init_cost:
        print(f"  ✓ 使用 Kabsch 旋转作为起点")
    else:
        print(f"  ⚠ 初始外参更优，但仍使用 Kabsch 旋转（消除初始值依赖）")
    start_R, start_t = R_kabsch, t_init_kabsch

    print(f"\n" + "="*60)
    print(f"迭代优化 - 使用 {len(features)} 帧数据")
    print("="*60)

    # 第1轮优化
    print(f"\n[第1轮] 使用 Kabsch 旋转 + 点约束初始平移")
    opt_R_1, opt_t_1, cost_1 = optimize_extrinsics(features, K, start_R, start_t, verbose=False, use_robust_cost=False)
    print_round_result("第1轮", opt_R_1, opt_t_1, cost_1)

    # 第2轮：修正lid_center
    print(f"\n[第2轮] 用第1轮结果修正 lid_center")
    R_inv_1 = opt_R_1.T
    features_refined = []
    lid_center_corrections = []

    for i, feat in enumerate(features):
        cam_normal, cam_center, lid_normal, lid_center_old = feat
        cam_center_in_lid = R_inv_1 @ (cam_center - opt_t_1)
        dist_to_plane = np.dot(cam_center_in_lid - lid_center_old, lid_normal)
        lid_center_new = cam_center_in_lid - dist_to_plane * lid_normal
        correction = np.linalg.norm(lid_center_new - lid_center_old)
        lid_center_corrections.append(correction)
        features_refined.append((cam_normal, cam_center, lid_normal, lid_center_new))
        print(f"    Frame {accepted_frame_ids[i]}: lid_center 修正 {correction*1000:.1f}mm")

    avg_correction = np.mean(lid_center_corrections)
    print(f"  平均修正量: {avg_correction*1000:.1f}mm")

    opt_R_2, opt_t_2, cost_2 = optimize_extrinsics(features_refined, K, opt_R_1, opt_t_1, verbose=False)
    print_round_result("第2轮", opt_R_2, opt_t_2, cost_2)
    print_round_diagnostic("Round1→Round2", features, features_refined, accepted_frame_ids, opt_R_2, opt_t_2)

    # 第3轮（可选）
    if avg_correction > 0.005:
        print(f"\n[第3轮] 继续迭代（修正量 > 5mm）")
        R_inv_2 = opt_R_2.T
        features_refined_2 = []
        for i, feat in enumerate(features):
            cam_normal, cam_center, lid_normal, _ = feat
            _, _, _, lid_center_old = features_refined[i]
            cam_center_in_lid = R_inv_2 @ (cam_center - opt_t_2)
            dist_to_plane = np.dot(cam_center_in_lid - lid_center_old, lid_normal)
            lid_center_new = cam_center_in_lid - dist_to_plane * lid_normal
            correction = np.linalg.norm(lid_center_new - lid_center_old)
            features_refined_2.append((cam_normal, cam_center, lid_normal, lid_center_new))
            print(f"    Frame {accepted_frame_ids[i]}: lid_center 修正 {correction*1000:.1f}mm")
        opt_R, opt_t, final_cost = optimize_extrinsics(features_refined_2, K, opt_R_2, opt_t_2, verbose=False)
        print_round_result("第3轮", opt_R, opt_t, final_cost)
        print_round_diagnostic("Round2→Round3", features_refined, features_refined_2, accepted_frame_ids, opt_R, opt_t)
    else:
        opt_R, opt_t, final_cost = opt_R_2, opt_t_2, cost_2
        features_refined_2 = features_refined

    print(f"\n[迭代完成] 最佳 cost = {final_cost:.6f}")

    # 异常帧检测与剔除
    features_for_check = features_refined_2
    features_current = features_for_check
    frame_ids_current = accepted_frame_ids.copy()
    opt_R_current, opt_t_current = opt_R, opt_t

    total_removed = []
    iteration = 0
    max_iterations = 3

    while iteration < max_iterations and len(features_current) >= 5:
        iteration += 1
        residuals = []
        for feat in features_current:
            cam_n, cam_c, lid_n, lid_c = feat
            pred_cam_c = opt_R_current @ lid_c + opt_t_current
            residual = np.linalg.norm(pred_cam_c - cam_c) * 1000
            residuals.append(residual)

        residuals = np.array(residuals)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        threshold_sigma = mean_res + 1.5 * std_res
        threshold_abs = 18.0
        threshold = min(threshold_sigma, threshold_abs)

        outlier_mask = residuals > threshold
        n_outliers = np.sum(outlier_mask)

        if n_outliers == 0 or len(features_current) - n_outliers < 4:
            break

        outlier_ids = [frame_ids_current[i] for i in np.where(outlier_mask)[0]]

        if iteration == 1:
            print(f"\n[异常帧检测] 迭代剔除开始 (阈值=min(mean+1.5σ, 18mm))")
        print(f"  [迭代{iteration}] 阈值={threshold:.1f}mm, 异常帧: {outlier_ids}")

        features_clean = [f for i, f in enumerate(features_current) if not outlier_mask[i]]
        frame_ids_clean = [fid for i, fid in enumerate(frame_ids_current) if not outlier_mask[i]]

        opt_R_clean, opt_t_clean, cost_clean = optimize_extrinsics(features_clean, K, opt_R_current, opt_t_current, verbose=False)

        residuals_clean = []
        for feat in features_clean:
            cam_n, cam_c, lid_n, lid_c = feat
            pred_cam_c = opt_R_clean @ lid_c + opt_t_clean
            residual = np.linalg.norm(pred_cam_c - cam_c) * 1000
            residuals_clean.append(residual)

        new_mean = np.mean(residuals_clean)
        improvement = (1 - new_mean / mean_res) * 100
        print(f"  [迭代{iteration}] 剔除后: {len(features_clean)}帧, 均值={new_mean:.1f}mm (改善{improvement:.1f}%)")

        if new_mean < mean_res * 0.85 or new_mean < 10.0:
            total_removed.extend(outlier_ids)
            features_current = features_clean
            frame_ids_current = frame_ids_clean
            opt_R_current, opt_t_current = opt_R_clean, opt_t_clean
        else:
            print(f"  [迭代{iteration}] 改善不足，停止迭代")
            break

    if len(total_removed) > 0:
        final_residuals = []
        for feat in features_current:
            cam_n, cam_c, lid_n, lid_c = feat
            pred_cam_c = opt_R_current @ lid_c + opt_t_current
            residual = np.linalg.norm(pred_cam_c - cam_c) * 1000
            final_residuals.append(residual)
        final_mean = np.mean(final_residuals)

        orig_residuals = []
        for feat in features_for_check:
            cam_n, cam_c, lid_n, lid_c = feat
            pred_cam_c = opt_R @ lid_c + opt_t
            residual = np.linalg.norm(pred_cam_c - cam_c) * 1000
            orig_residuals.append(residual)
        orig_mean = np.mean(orig_residuals)

        total_improvement = (1 - final_mean / orig_mean) * 100
        print(f"  ✓ 共剔除 {len(total_removed)} 帧: {total_removed}")
        print(f"  ✓ 总改善: {orig_mean:.1f}mm → {final_mean:.1f}mm ({total_improvement:.1f}%)")

        opt_R, opt_t = opt_R_current, opt_t_current
        final_cost = optimize_extrinsics(features_current, K, opt_R, opt_t, verbose=False)[2]
        features_final = features_current
        accepted_frame_ids = frame_ids_current
    else:
        features_final = features_for_check

    # 输出结果
    print(f"\n" + "="*70)
    print("优化效果")
    print("="*70)
    print(f"  初始外参代价:     {init_cost:.6f}")
    print(f"  Kabsch起点代价:   {kabsch_cost:.6f}")
    print(f"  最终优化代价:     {final_cost:.6f}")
    print(f"  代价降低:         {(init_cost - final_cost) / init_cost * 100:.1f}%")

    # 最终诊断表格
    print(f"\n{'='*140}")
    print("最终诊断汇总")
    print(f"{'='*140}")

    total_residuals = []
    for i, (feat_final, fid) in enumerate(zip(features_final, accepted_frame_ids)):
        cam_normal, cam_center, lid_normal_final, lid_center_final = feat_final
        lid_center_in_cam = opt_R @ lid_center_final + opt_t
        center_residual = np.linalg.norm(lid_center_in_cam - cam_center) * 1000
        total_residuals.append(center_residual)

    # 质量评价
    mean_residual = np.mean(total_residuals)
    std_residual = np.std(total_residuals)
    max_residual = np.max(total_residuals)

    print(f"\n[精度指标]")
    print(f"  残差均值:     {mean_residual:>6.1f} mm")
    print(f"  残差标准差:   {std_residual:>6.1f} mm")
    print(f"  残差最大值:   {max_residual:>6.1f} mm")
    print(f"  使用帧数:     {len(features_final)}")

    # 评分
    if mean_residual < 5:
        residual_score = 100
    elif mean_residual < 10:
        residual_score = 90 + (10 - mean_residual) * 2
    elif mean_residual < 20:
        residual_score = 70 + (20 - mean_residual) * 2
    else:
        residual_score = max(0, 50 - (mean_residual - 20))

    if std_residual < 2:
        consistency_score = 100
    elif std_residual < 5:
        consistency_score = 90 + (5 - std_residual) * 3.33
    else:
        consistency_score = max(0, 70 - (std_residual - 5) * 4)

    n_frames_used = len(features)  # 使用原始帧数计算评分（与v1一致）
    if n_frames_used >= 9:
        frame_score = 100
    elif n_frames_used >= 6:
        frame_score = 90
    elif n_frames_used >= 4:
        frame_score = 75
    else:
        frame_score = 60

    total_score = residual_score * 0.5 + consistency_score * 0.3 + frame_score * 0.2

    print(f"\n[评分] 综合得分: {total_score:.1f}")

    if total_score >= 95:
        grade = "A+ (优秀)"
    elif total_score >= 85:
        grade = "A  (良好)"
    elif total_score >= 75:
        grade = "B  (合格)"
    elif total_score >= 60:
        grade = "C  (及格)"
    else:
        grade = "D  (不合格)"

    print(f"  等级: {grade}")

    # 转换为TF格式
    opt_R_tf = opt_R.T
    opt_t_tf = -opt_R.T @ opt_t
    opt_quat_tf = Rotation.from_matrix(opt_R_tf).as_quat()

    print(f"\n{'='*80}")
    print(f"标定结果 - TF 约定格式")
    print(f"{'='*80}")
    print(f"\n平移 (translation):")
    print(f"  x: {opt_t_tf[0]:>10.6f}")
    print(f"  y: {opt_t_tf[1]:>10.6f}")
    print(f"  z: {opt_t_tf[2]:>10.6f}")
    print(f"\n旋转 (quaternion xyzw):")
    print(f"  x: {opt_quat_tf[0]:>10.6f}")
    print(f"  y: {opt_quat_tf[1]:>10.6f}")
    print(f"  z: {opt_quat_tf[2]:>10.6f}")
    print(f"  w: {opt_quat_tf[3]:>10.6f}")

    # 保存结果（按时间戳区分）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    calib_label = '%s_camera2lidar' % camera_type
    output_dir = os.path.join(args.data_dir, 'config', 'results', '%s_%s' % (calib_label, timestamp))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'extrinsics_%s_%s.yaml' % (calib_label, timestamp))

    result = {
        'header': {
            'calibration_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'interactive_multi_frame_v2',
            'n_frames': len(features_final)
        },
        'frame_id': 'rs16_lidar',
        'child_frame_id': output_child_frame,
        'transform': {
            'translation': {'x': float(opt_t_tf[0]), 'y': float(opt_t_tf[1]), 'z': float(opt_t_tf[2])},
            'rotation': {'x': float(opt_quat_tf[0]), 'y': float(opt_quat_tf[1]),
                        'z': float(opt_quat_tf[2]), 'w': float(opt_quat_tf[3])}
        }
    }

    with open(output_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)

    print(f"\n[保存] TF 外参已保存到: {output_file}")

    # 转换optical_frame到camera_link坐标系（用于SLAM）
    output_file_apply = os.path.join(output_dir, 'extrinsics_apply_%s_%s.yaml' % (calib_label, timestamp))
    try:
        convert_optical_to_link_extrinsics(output_file, output_file_apply, camera_type)
        print(f"[保存] SLAM 外参已保存到: {output_file_apply}")
    except Exception as e:
        print(f"[警告] 坐标系转换失败: {e}")
        print(f"  请手动转换或检查camera_type设置")

    print("\n[完成]")


if __name__ == '__main__':
    main()
