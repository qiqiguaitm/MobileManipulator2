# -*- coding: utf-8 -*-
"""
Calibration optimization solvers for camera-LiDAR calibration.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from .geometry import compute_lid_center_from_camera, validate_projected_rectangle


def solve_rotation_kabsch(cam_normals, lid_normals, verbose=True):
    """Solve optimal rotation using Kabsch algorithm (SVD closed-form).

    R* = argmin Σ ||cam_n_i - R @ lid_n_i||²

    Args:
        cam_normals: camera frame normals [(3,), ...]
        lid_normals: LiDAR frame normals [(3,), ...]
        verbose: print debug info

    Returns:
        R: optimal rotation matrix (3,3)
        residuals: per-frame normal residuals (degrees)
        inlier_mask: inlier mask for outlier detection
    """
    n_frames = len(cam_normals)

    H = np.zeros((3, 3))
    for cam_n, lid_n in zip(cam_normals, lid_normals):
        H += np.outer(lid_n, cam_n)

    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    residuals_deg = []
    for cam_n, lid_n in zip(cam_normals, lid_normals):
        pred_cam_n = R @ lid_n
        cos_angle = np.clip(np.dot(pred_cam_n, cam_n), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        residuals_deg.append(angle_deg)
    residuals_deg = np.array(residuals_deg)

    mean_res = np.mean(residuals_deg)
    std_res = np.std(residuals_deg)
    threshold = max(mean_res + 3 * std_res, 15.0)
    inlier_mask = residuals_deg < threshold

    if verbose:
        print(f"\n  [Kabsch] SVD奇异值: [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.4f}]")
        print(f"  [Kabsch] 法向量残差: mean={mean_res:.2f}°, std={std_res:.2f}°, max={np.max(residuals_deg):.2f}°")

    return R, residuals_deg, inlier_mask


def solve_translation_plane_constraint(R, cam_centers, lid_normals, lid_plane_points, verbose=True):
    """Solve translation using plane constraint (linear least squares).

    Args:
        R: rotation matrix (3,3)
        cam_centers: board centers in camera frame [(3,), ...]
        lid_normals: plane normals in LiDAR frame [(3,), ...]
        lid_plane_points: reference points on LiDAR planes [(3,), ...]
        verbose: print debug info

    Returns:
        t: optimal translation (3,)
        plane_residuals: per-frame plane distance residuals (meters)
    """
    R_T = R.T

    N = []
    b = []
    for cam_c, lid_n, lid_p in zip(cam_centers, lid_normals, lid_plane_points):
        q = R_T @ cam_c - lid_p
        N.append(lid_n)
        b.append(np.dot(q, lid_n))

    N = np.array(N)
    b = np.array(b)

    u, residuals, rank, s = np.linalg.lstsq(N, b, rcond=None)
    t = R @ u

    plane_residuals = []
    for cam_c, lid_n, lid_p in zip(cam_centers, lid_normals, lid_plane_points):
        cam_c_in_lid = R_T @ (cam_c - t)
        dist = abs(np.dot(cam_c_in_lid - lid_p, lid_n))
        plane_residuals.append(dist)
    plane_residuals = np.array(plane_residuals)

    if verbose:
        cond = s[0]/s[-1] if s[-1] > 1e-10 else float('inf')
        print(f"\n  [平面约束] 矩阵秩: {rank}, 条件数: {cond:.1f}")
        print(f"  [平面约束] 平移向量: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]m")

    return t, plane_residuals


def check_normal_diversity(lid_normals, verbose=True):
    """Check normal vector diversity for constraint validity."""
    N = np.array(lid_normals)
    _, s, _ = np.linalg.svd(N)
    cond = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
    diversity_score = s[-1] / s[0] if s[0] > 0 else 0

    if verbose:
        std_xyz = np.std(N, axis=0)
        print(f"\n  [法向量多样性]")
        print(f"    各轴标准差: X={std_xyz[0]:.3f}, Y={std_xyz[1]:.3f}, Z={std_xyz[2]:.3f}")
        print(f"    条件数: {cond:.1f}, 多样性: {diversity_score:.3f}")

    return diversity_score, cond


def optimize_extrinsics_robust(features, lid_plane_points, K, init_R=None, init_t=None,
                               verbose=True, refine=True):
    """Robust extrinsics optimization: closed-form + adaptive constraints.

    Args:
        features: list of (cam_normal, cam_center, lid_normal, lid_center)
        lid_plane_points: reference points on LiDAR planes
        K: camera intrinsic matrix
        init_R, init_t: initial extrinsics (for comparison)
        verbose: print detailed info
        refine: whether to do local refinement

    Returns:
        opt_R: optimized rotation matrix
        opt_t: optimized translation vector
        cost: final cost
        diagnostics: diagnostic info dict
    """
    print("\n" + "="*70)
    print("鲁棒外参优化（闭式解 + 自适应约束）")
    print("="*70)

    cam_normals = [f[0] for f in features]
    cam_centers = [f[1] for f in features]
    lid_normals = [f[2] for f in features]
    lid_centers = [f[3] for f in features]

    n_frames = len(features)
    print(f"\n[输入] {n_frames} 帧数据")

    diversity, cond = check_normal_diversity(lid_normals, verbose=verbose)

    # Phase 1: Kabsch rotation
    print(f"\n[阶段1] Kabsch 算法求解旋转")
    R_kabsch, rot_residuals, inlier_mask = solve_rotation_kabsch(
        cam_normals, lid_normals, verbose=verbose
    )

    # Phase 2: Translation
    DIVERSITY_THRESHOLD = 0.15

    def compute_cost(R, t, features):
        total = 0.0
        for cam_n, cam_c, lid_n, lid_c in features:
            normal_cost = 1.0 - np.dot(R @ lid_n, cam_n)
            center_error = np.linalg.norm(R @ lid_c + t - cam_c)
            total += 2.0 * normal_cost + center_error / 3.0
        return total

    if diversity >= DIVERSITY_THRESHOLD:
        print(f"\n[阶段2] 平面约束求解平移")
        t_closed, plane_residuals = solve_translation_plane_constraint(
            R_kabsch, cam_centers, lid_normals, lid_plane_points, verbose=verbose
        )
    else:
        print(f"\n[阶段2] 混合约束求解平移（多样性不足）")
        t_plane, _ = solve_translation_plane_constraint(
            R_kabsch, cam_centers, lid_normals, lid_plane_points, verbose=False
        )
        t_estimates = [cam_c - R_kabsch @ lid_c for cam_c, lid_c in zip(cam_centers, lid_centers)]
        t_point = np.mean(t_estimates, axis=0)
        plane_weight = max(0.2, diversity / DIVERSITY_THRESHOLD * 0.5)
        t_closed = plane_weight * t_plane + (1 - plane_weight) * t_point
        plane_residuals = None

    closed_form_cost = compute_cost(R_kabsch, t_closed, features)
    print(f"\n[闭式解] 代价 = {closed_form_cost:.6f}")

    # Phase 3: Optional refinement
    if refine:
        print(f"\n[阶段3] 局部精细化")
        start_R, start_t = R_kabsch, t_closed

        if init_R is not None and init_t is not None:
            init_cost = compute_cost(init_R, init_t, features)
            if init_cost < closed_form_cost:
                start_R, start_t = init_R, init_t

        init_rvec = Rotation.from_matrix(start_R).as_rotvec()
        init_x = np.concatenate([init_rvec, start_t])

        def cost_function(x):
            rvec = x[:3]
            t = x[3:6]
            R = Rotation.from_rotvec(rvec).as_matrix()
            return compute_cost(R, t, features)

        result = minimize(cost_function, init_x, method='Powell',
                         options={'maxiter': 1000, 'ftol': 1e-7})

        opt_R = Rotation.from_rotvec(result.x[:3]).as_matrix()
        opt_t = result.x[3:6]
        final_cost = result.fun
    else:
        opt_R, opt_t, final_cost = R_kabsch, t_closed, closed_form_cost

    euler = Rotation.from_matrix(opt_R).as_euler('xyz', degrees=True)
    print(f"\n[最终结果]")
    print(f"  旋转 (Euler XYZ): [{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]")
    print(f"  平移: [{opt_t[0]:.4f}, {opt_t[1]:.4f}, {opt_t[2]:.4f}]m")

    diagnostics = {
        'kabsch_R': R_kabsch,
        'closed_t': t_closed,
        'closed_form_cost': closed_form_cost,
        'rot_residuals': rot_residuals,
        'inlier_mask': inlier_mask,
        'diversity': diversity
    }

    return opt_R, opt_t, final_cost, diagnostics


def optimize_extrinsics_v2(features_v2, K, init_R, init_t, verbose=True):
    """Method C optimization: use camera corners to guide LiDAR center.

    Args:
        features_v2: enhanced features with camera corners
        K: camera intrinsic matrix
        init_R, init_t: initial extrinsics (point transform form)
        verbose: print detailed info

    Returns:
        opt_R: optimized rotation matrix
        opt_t: optimized translation vector
        cost: final cost
    """
    print("\n" + "="*70)
    print("方案C优化：相机角点引导LiDAR中心计算")
    print("="*70)

    # Precompute lid_centers using camera corners
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
                    'lid_center': lid_c.copy()
                })

        if return_details:
            return total_cost, details
        return total_cost

    init_rvec = Rotation.from_matrix(init_R).as_rotvec()
    init_x = np.concatenate([init_rvec, init_t])

    result = minimize(cost_function, init_x, method='Powell',
                     options={'maxiter': 2000, 'ftol': 1e-6})

    opt_R = Rotation.from_rotvec(result.x[:3]).as_matrix()
    opt_t = result.x[3:6]

    print(f"\n  最终代价: {result.fun:.6f}")

    return opt_R, opt_t, result.fun
