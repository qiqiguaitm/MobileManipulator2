# -*- coding: utf-8 -*-
"""
Plane fitting utilities for camera-LiDAR calibration.
Extracted from interactive_calibrate.py - keeps exact same logic and output.
"""

import numpy as np


def fit_ground_plane(points_3d, x_range=(0.5, 3.0), y_range=(-1.5, 1.5)):
    """拟合地面平面，返回地面z坐标

    思路：地面是点云中最低的大面积水平面
    限制在有效范围内拟合，避免远处噪声干扰

    Args:
        points_3d: 完整点云 (N, 3)
        x_range: x方向有效范围 (min, max)，默认 (0.5, 3.0)m
        y_range: y方向有效范围 (min, max)，默认 (-1.5, 1.5)m

    Returns:
        ground_z: 地面z坐标（在LiDAR坐标系中，通常为负值）
    """
    import open3d as o3d

    # 0. 先过滤到有效范围内
    x_mask = (points_3d[:, 0] >= x_range[0]) & (points_3d[:, 0] <= x_range[1])
    y_mask = (points_3d[:, 1] >= y_range[0]) & (points_3d[:, 1] <= y_range[1])
    roi_points = points_3d[x_mask & y_mask]

    print(f"  [地面拟合] ROI范围: x=[{x_range[0]}, {x_range[1]}]m, y=[{y_range[0]}, {y_range[1]}]m")
    print(f"  [地面拟合] ROI内点数: {len(roi_points)}/{len(points_3d)} ({100*len(roi_points)/len(points_3d):.1f}%)")

    if len(roi_points) < 500:
        print(f"  [地面拟合] ROI内点太少({len(roi_points)}个)，使用全部点云")
        roi_points = points_3d

    # 1. 选择z坐标较低的点作为地面候选（取z最小的30%点）
    z_values = roi_points[:, 2]
    z_threshold = np.percentile(z_values, 30)  # z最小的30%

    ground_candidates = roi_points[z_values <= z_threshold]

    if len(ground_candidates) < 100:
        print(f"  [地面拟合] 候选点太少({len(ground_candidates)}个)")
        return None

    # 2. RANSAC拟合水平面（多次运行取最佳）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_candidates)

    best_inliers = None
    best_plane_model = None
    for trial in range(3):
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=500
            )
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane_model = plane_model
        except:
            continue

    if best_inliers is None:
        print(f"  [地面拟合] RANSAC失败")
        return None

    plane_model = best_plane_model
    inliers = best_inliers

    # 3. 检查是否为水平面（法向量接近[0,0,1]）
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        print(f"  [地面拟合] 拟合平面不是水平面 (normal_z={normal[2]:.3f})")
        return None

    # 4. 计算地面z坐标（使用内点z均值，更准确）
    inlier_points = ground_candidates[inliers]
    z_mean = inlier_points[:, 2].mean()
    z_std = inlier_points[:, 2].std()
    ground_z = z_mean  # 直接使用内点均值作为地面高度

    print(f"  [地面拟合] 成功! ground_z = {ground_z:.4f}m, z标准差={z_std*100:.2f}cm")
    print(f"  [地面拟合] 内点: {len(inliers)}个, Z下限阈值={ground_z + 3*z_std:.4f}m (ground_z + 3σ)")

    return ground_z, z_std


def fit_plane_ransac(points, threshold=0.02, check_vertical=True, board_size=None, ground_z=None, ground_z_std=None, pattern_size=None, square_size=None, full_points=None):
    """RANSAC平面拟合 + 动态点云扩展

    Args:
        points: 初始点云 (N, 3) - 人工框选区域
        threshold: RANSAC距离阈值
        check_vertical: 是否检查平面是否竖直
        board_size: 标定板物理尺寸 (width, height)，用于覆盖率验证
        ground_z: 地面z坐标，用于约束标定板下边缘
        ground_z_std: 地面z坐标的标准差，用于计算Z下限
        pattern_size: 内角点数 (cols, rows)，用于计算内角点区域
        square_size: 方格边长 (米)
        full_points: 完整点云 (M, 3) - 用于动态扩展，None则不扩展
    """
    import open3d as o3d
    from .geometry import fit_rectangle_known_size

    if len(points) < 10:
        return None, None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 多次运行RANSAC取最佳结果（解决Open3D内部随机性问题）
    best_inliers = None
    best_plane_model = None
    n_trials = 5

    for trial in range(n_trials):
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=threshold,
                ransac_n=3,
                num_iterations=1000
            )
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane_model = plane_model
        except:
            continue

    if best_inliers is None or len(best_inliers) < 5:
        return None, None, None

    plane_model = best_plane_model
    inliers = best_inliers

    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    d = plane_model[3]  # 平面方程: normal · p + d = 0

    # ========== 动态点云扩展 ==========
    # 思路：以 RANSAC 得到的平面为基准，在更大范围内搜索符合平面的点
    if full_points is not None and board_size is not None:
        board_width, board_height = board_size
        initial_inliers = points[inliers]
        initial_center = np.mean(initial_inliers, axis=0)
        initial_count = len(initial_inliers)

        # 构建搜索坐标系
        n_ground = np.array([0.0, 0.0, 1.0])
        h_dir = np.cross(normal, n_ground)
        h_norm = np.linalg.norm(h_dir)
        if h_norm > 0.1:
            h_dir = h_dir / h_norm
        else:
            h_dir = np.array([0, 1, 0])
        v_dir = np.cross(normal, h_dir)
        v_dir = v_dir / np.linalg.norm(v_dir)
        if v_dir[2] < 0:
            v_dir = -v_dir
            h_dir = -h_dir

        # 扩展搜索范围
        search_h = board_width / 2 + 0.10   # 水平扩展 10cm 余量
        search_v = board_height / 2 + 0.10  # 垂直扩展 10cm 余量
        search_depth = 0.03  # 深度方向严格 3cm

        # 动态扩展使用 ground_z + 3σ 约束剔除地面点
        if ground_z is not None and ground_z_std is not None:
            z_lower_bound = ground_z + 3 * ground_z_std
        else:
            z_lower_bound = -0.35  # 默认值

        # 在完整点云中搜索符合平面的点
        expanded_points = []
        for pt in full_points:
            # 地面约束：剔除低于地面的点
            if pt[2] < z_lower_bound:
                continue

            # 点到平面距离
            dist_to_plane = abs(np.dot(pt, normal) + d)
            if dist_to_plane > search_depth:
                continue

            # 检查是否在标定板范围内
            vec = pt - initial_center
            h_proj = abs(np.dot(vec, h_dir))
            v_proj = abs(np.dot(vec, v_dir))

            if h_proj < search_h and v_proj < search_v:
                expanded_points.append(pt)

        expanded_points = np.array(expanded_points) if expanded_points else np.array([]).reshape(0, 3)
        expanded_count = len(expanded_points)

        if expanded_count > initial_count:
            print(f"  [动态扩展] 初始内点: {initial_count} → 扩展后: {expanded_count} (+{expanded_count - initial_count})")
            # 使用扩展后的点云替代原始内点
            inlier_points = expanded_points
        else:
            print(f"  [动态扩展] 无需扩展 (初始={initial_count}, 扩展={expanded_count})")
            inlier_points = initial_inliers
    else:
        inlier_points = points[inliers]

    # 检查法向量是否合理
    # 注意：标定板可能前俯或后仰（倾斜角度可达45°甚至更大）
    # normal_z = sin(tilt_angle)，所以：
    #   - 倾斜30°: normal_z ≈ 0.5
    #   - 倾斜45°: normal_z ≈ 0.707
    #   - 倾斜60°: normal_z ≈ 0.866
    # 只有当 normal_z > 0.95 时才认为是水平面（倾斜>72°）
    # 对于16线雷达稀疏点云，当点数<100时，跳过水平面检测（避免误判）
    if check_vertical:
        vertical_component = abs(normal[2])  # Z分量

        # 点数少时，跳过水平面检测（16线雷达稀疏点云容易误判）
        if len(points) < 100:
            if vertical_component > 0.5:
                tilt_angle = np.degrees(np.arcsin(vertical_component))
                print(f"  [提示] 点数较少({len(points)}个)，标定板倾斜约 {tilt_angle:.1f}° (normal_z={vertical_component:.3f})")
        else:
            # 点数充足时，进行水平面检测
            if vertical_component > 0.95:  # 放宽阈值到0.95（倾斜>72°）
                print(f"  [警告] 检测到近似水平面 (normal_z={vertical_component:.3f})，可能选到了地面")
                print(f"         标定板倾斜角度超过72°，请检查选择区域")
                return None, None, None
            elif vertical_component > 0.5:
                # 标定板有明显倾斜，给出提示但继续处理
                tilt_angle = np.degrees(np.arcsin(vertical_component))
                print(f"  [提示] 标定板倾斜约 {tilt_angle:.1f}° (normal_z={vertical_component:.3f})")

    # 确保法向量指向LiDAR传感器方向
    # 约定：法向量从标定板指向传感器（与相机侧一致）
    # 由于标定板在LiDAR前方（X > 0），指向LiDAR的法向量X分量应为负
    if normal[0] > 0:
        normal = -normal

    # 注意：inlier_points 已在动态扩展部分设置

    # ========== 几何中心计算（地面约束 + 边界中点） ==========
    # 思路：
    # 1. 标定板竖直放在地面上，下边缘z = ground_z
    # 2. h方向（水平）：用边界min/max中点
    # 3. v方向（垂直）：用地面约束 + board_height/2

    # 1. 构建标定板坐标系（h: 水平方向, v: 垂直向上方向）
    n_ground = np.array([0.0, 0.0, 1.0])  # 地面法向量（向上）

    # h = 标定板平面与地面的交线方向（水平方向）
    h = np.cross(normal, n_ground)
    h_norm = np.linalg.norm(h)
    if h_norm < 0.1:
        # 标定板接近水平（不应该发生）
        print(f"  [警告] 标定板接近水平，法向量z分量过大")
        h = np.array([0, 1, 0])
    else:
        h = h / h_norm

    # v = 标定板上的"向上"方向（在标定板平面内，垂直于h）
    v = np.cross(normal, h)
    v = v / np.linalg.norm(v)

    # 确保v指向上方（z分量为正）
    if v[2] < 0:
        v = -v
        h = -h  # 保持右手系

    # 2. 参考点（内点平均）
    p0 = np.mean(inlier_points, axis=0)

    # 3. 将内点转换到标定板2D坐标系 (h, v)
    coords_2d = []
    for p in inlier_points:
        p_proj = p - np.dot(p - p0, normal) * normal
        coord_h = np.dot(p_proj - p0, h)
        coord_v = np.dot(p_proj - p0, v)
        coords_2d.append([coord_h, coord_v])
    coords_2d = np.array(coords_2d)

    # 统计边界信息（用于fallback和调试）
    min_h, max_h = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    min_v, max_v = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    range_h = max_h - min_h
    range_v = max_v - min_v
    center_h_boundary = (min_h + max_h) / 2
    center_v_boundary = (min_v + max_v) / 2
    center_h_mean = np.mean(coords_2d[:, 0])
    center_v_mean = np.mean(coords_2d[:, 1])

    # ========== 方案：矩形拟合（利用已知物理尺寸） ==========
    # 标定板是已知尺寸的矩形，拟合矩形位置比边界/重心更鲁棒
    rect_fit_success = False
    center_h_rect = None
    center_v_rect = None
    if board_size is not None:
        board_width, board_height = board_size
        print(f"  [矩形拟合] 使用已知尺寸 {board_width*100:.0f}cm × {board_height*100:.0f}cm")

        rect_cx, rect_cy, rect_theta, rect_fit_success = fit_rectangle_known_size(
            coords_2d, board_width, board_height, verbose=True)

        if rect_fit_success:
            # 矩形拟合成功，但需要转换：矩形中心是物理边界中心，需转为内角点中心
            # 如果边框对称，两者重合；如果不对称，需要偏移
            # 目前假设对称边框
            center_h_rect = rect_cx
            center_v_rect = rect_cy

            # 比较各方法的差异
            print(f"  [对比] h方向: 矩形={center_h_rect:.3f}m, 重心={center_h_mean:.3f}m, 边界={center_h_boundary:.3f}m")
            print(f"  [对比] h差异: 矩形-重心={((center_h_rect-center_h_mean)*1000):.1f}mm, 矩形-边界={((center_h_rect-center_h_boundary)*1000):.1f}mm")
            print(f"  [对比] v方向: 矩形={center_v_rect:.3f}m, 重心={center_v_mean:.3f}m, 边界={center_v_boundary:.3f}m")

    # ========== 确定最终中心 ==========
    # 策略改进：基于点云密度自适应选择
    # - 点云密集时：边界中点可靠
    # - 点云稀疏时：矩形拟合更稳定

    # 计算点云密度指标（点数/面积）
    n_points = len(inlier_points)
    if board_size is not None:
        expected_area = board_size[0] * board_size[1]  # m²
        point_density = n_points / expected_area  # 点/m²
        # 密度阈值：约1000点/m²认为是密集，低于500点/m²认为是稀疏
        density_factor = min(1.0, point_density / 1000.0)  # 0~1，越高越信任边界
    else:
        density_factor = 1.0

    # h方向：根据密度和矩形拟合质量加权
    if rect_fit_success:
        rect_inlier_ratio = 0.98  # 从矩形拟合结果获取（这里假设成功时>94%）
        rect_boundary_diff = abs(center_h_rect - center_h_boundary) * 1000  # mm

        # 如果矩形拟合质量高且与边界差异小，使用矩形结果
        # 如果差异大，根据密度加权
        if rect_boundary_diff < 5.0:
            # 差异<5mm，使用矩形（更稳定）
            center_h = center_h_rect
            h_method = "矩形拟合(差异小)"
        elif density_factor > 0.8:
            # 密度高，信任边界
            center_h = center_h_boundary
            h_method = "边界中点(密度高)"
        else:
            # 密度低，加权平均
            rect_weight = 1.0 - density_factor  # 密度越低，矩形权重越高
            center_h = rect_weight * center_h_rect + (1 - rect_weight) * center_h_boundary
            h_method = f"加权(矩形{rect_weight:.0%})"

        print(f"  [h方向] {h_method}: {center_h:.3f}m")
        print(f"  [h方向] 边界={center_h_boundary:.3f}m, 矩形={center_h_rect:.3f}m, 密度因子={density_factor:.2f}")
    else:
        center_h = center_h_boundary
        print(f"  [h方向] 使用边界中点: {center_h:.3f}m (重心={center_h_mean:.3f}m, 差{(center_h_boundary-center_h_mean)*1000:.1f}mm)")

    # v方向：优先使用地面约束（物理依据更强）
    if ground_z is not None and board_size is not None and pattern_size is not None and square_size is not None:
        board_width, board_height = board_size
        cols, rows = pattern_size
        pattern_height = (rows - 1) * square_size
        bottom_margin = (board_height - pattern_height) / 2

        if abs(v[2]) > 0.1:
            pattern_bottom_z = ground_z + bottom_margin
            v_pattern_bottom = (pattern_bottom_z - p0[2] - center_h * h[2]) / v[2]
            center_v_ground = v_pattern_bottom + pattern_height / 2

            if rect_fit_success:
                print(f"  [v方向] 地面约束: {center_v_ground:.3f}m, 矩形拟合: {center_v_rect:.3f}m, 差异: {abs(center_v_ground - center_v_rect)*1000:.1f}mm")

            center_v = center_v_ground
            print(f"  [v方向] 使用地面约束: {center_v:.3f}m")
        else:
            center_v = center_v_mean
            print(f"  [v方向] v的z分量小，使用重心: {center_v:.3f}m")
    else:
        center_v = center_v_mean
        print(f"  [v方向] 无地面约束，使用重心: {center_v:.3f}m")

    # 6. 转回3D坐标
    center = p0 + center_h * h + center_v * v

    # ========== 关键：强制约束 center.z（基于局部地面高度、法向量、物理尺寸）==========
    # 物理模型：
    # 1. 标定板下边缘在地面上：z_bottom = local_ground_z
    # 2. 标定板"向上"方向为 v 向量（依赖于法向量 normal）
    # 3. 中心相对于下边缘中点的位移 = (board_height/2) * v
    # 4. 因此：center.z = local_ground_z + (board_height/2) * v[2]
    #
    # 改进：在标定板XY位置附近重新拟合局部地面，避免全局ground_z偏差
    if ground_z is not None and board_size is not None and full_points is not None:
        board_width, board_height = board_size
        # 在标定板XY位置附近重新拟合局部地面
        local_x_range = (center[0] - 1.0, center[0] + 1.0)
        local_y_range = (center[1] - 1.0, center[1] + 1.0)
        local_result = fit_ground_plane(full_points, x_range=local_x_range, y_range=local_y_range)
        if local_result is not None:
            local_ground_z, local_ground_std = local_result
            print(f"  [z约束] 局部地面: ground_z={local_ground_z:.3f}m (全局={ground_z:.3f}m, 差={1000*(local_ground_z-ground_z):.1f}mm)")
        else:
            local_ground_z = ground_z
            print(f"  [z约束] 局部地面拟合失败，使用全局ground_z={ground_z:.3f}m")
        # v[2] 反映标定板的倾斜程度：竖直时 v[2]=1，向前倾斜时 v[2]<1
        center_z_constrained = local_ground_z + (board_height / 2) * v[2]
        center_z_diff = center[2] - center_z_constrained
        print(f"  [z约束] 物理约束: center.z = local_ground_z + (board_height/2)*v[2]")
        print(f"  [z约束]   = {local_ground_z:.3f} + ({board_height:.2f}/2)*{v[2]:.3f} = {center_z_constrained:.3f}m")
        print(f"  [z约束] 原计算值: {center[2]:.3f}m, 差异: {center_z_diff*1000:.1f}mm → 强制修正")
        center[2] = center_z_constrained
    elif ground_z is not None and board_size is not None:
        # 无 full_points，使用全局 ground_z
        board_width, board_height = board_size
        center_z_constrained = ground_z + (board_height / 2) * v[2]
        center_z_diff = center[2] - center_z_constrained
        print(f"  [z约束] 物理约束: center.z = ground_z + (board_height/2)*v[2]")
        print(f"  [z约束]   = {ground_z:.3f} + ({board_height:.2f}/2)*{v[2]:.3f} = {center_z_constrained:.3f}m")
        print(f"  [z约束] 原计算值: {center[2]:.3f}m, 差异: {center_z_diff*1000:.1f}mm → 强制修正")
        center[2] = center_z_constrained

    # 打印覆盖情况
    if board_size is not None:
        board_width, board_height = board_size
        coverage_h = range_h / board_width * 100
        coverage_v = range_v / board_height * 100 if 'range_v' in dir() else 0
        print(f"  [边界检测] 点云范围: h={range_h*100:.1f}cm, v={range_v*100:.1f}cm")
        print(f"  [边界检测] 标定板尺寸: {board_width*100:.1f}cm x {board_height*100:.1f}cm")
        print(f"  [边界检测] 覆盖率: h={coverage_h:.0f}%, v={coverage_v:.0f}%")

    # 打印对比信息
    center_mean = np.mean(inlier_points, axis=0)
    center_median = np.median(inlier_points, axis=0)
    print(f"  [中心计算] 地面约束中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  [中心计算] 3D平均值:     [{center_mean[0]:.3f}, {center_mean[1]:.3f}, {center_mean[2]:.3f}] (差 {np.linalg.norm(center-center_mean)*100:.1f}cm)")
    print(f"  [中心计算] 3D中位数:     [{center_median[0]:.3f}, {center_median[1]:.3f}, {center_median[2]:.3f}] (差 {np.linalg.norm(center-center_median)*100:.1f}cm)")

    return normal, center, inlier_points
