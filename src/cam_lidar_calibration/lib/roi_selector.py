#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D点云ROI选择工具 - 投影引导 + 交互确认
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def project_camera_to_lidar(cam_center, R_c2l, t_c2l):
    """
    将相机坐标系中的点投影到LiDAR坐标系

    Args:
        cam_center: 标定板中心在相机坐标系中的位置 (3,)
        R_c2l: 相机到LiDAR的旋转矩阵 (逆变换)
        t_c2l: 相机到LiDAR的平移向量 (逆变换)

    Returns:
        标定板中心在LiDAR坐标系中的估计位置
    """
    # cam_center 是在相机光学坐标系中的位置
    # 需要变换到LiDAR坐标系: p_lidar = R_c2l^-1 @ (p_cam - t_c2l)
    # 由于我们的外参是 p_cam = R @ p_lidar + t
    # 所以逆变换是 p_lidar = R^T @ (p_cam - t)
    R_inv = R_c2l.T
    lidar_center = R_inv @ (cam_center - t_c2l)
    return lidar_center


def auto_estimate_roi(points_3d, cam_center, R_c2l, t_c2l, board_size=(0.6, 0.8)):
    """
    基于相机检测结果自动估计LiDAR点云中的标定板区域

    Args:
        points_3d: 全部点云 (N, 3)
        cam_center: 相机检测到的标定板中心
        R_c2l: 当前外参（相机到LiDAR变换）
        t_c2l: 当前外参平移
        board_size: 标定板物理尺寸 (width, height)，单位米

    Returns:
        estimated_center: 估计的LiDAR中心位置
        roi_points: 自动选择的ROI点云
        bbox: 包围盒 (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    # 投影相机中心到LiDAR坐标系
    estimated_center = project_camera_to_lidar(cam_center, R_c2l, t_c2l)

    print(f"  [投影引导] 相机检测: {cam_center}")
    print(f"  [投影引导] LiDAR估计: {estimated_center}")

    # 根据标定板尺寸设置搜索范围
    # 标定板: 0.6m宽 × 0.8m高，设置1.5倍的搜索范围
    search_x = 0.3  # 前后0.3m（标定板厚度方向）
    search_y = board_size[0] * 0.75  # 左右1.5倍宽度
    search_z = board_size[1] * 0.75  # 上下1.5倍高度

    # 构建包围盒
    x_min = estimated_center[0] - search_x
    x_max = estimated_center[0] + search_x
    y_min = estimated_center[1] - search_y
    y_max = estimated_center[1] + search_y
    z_min = max(0.1, estimated_center[2] - search_z)  # 避免选到地面
    z_max = estimated_center[2] + search_z

    # 提取ROI点云
    roi_mask = (
        (points_3d[:, 0] >= x_min) & (points_3d[:, 0] <= x_max) &
        (points_3d[:, 1] >= y_min) & (points_3d[:, 1] <= y_max) &
        (points_3d[:, 2] >= z_min) & (points_3d[:, 2] <= z_max)
    )
    roi_points = points_3d[roi_mask]

    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)

    print(f"  [投影引导] 包围盒: X[{x_min:.2f}, {x_max:.2f}], "
          f"Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    print(f"  [投影引导] 提取到 {len(roi_points)} 个点")

    return estimated_center, roi_points, bbox


def visualize_3d_interactive(points_3d, roi_points, estimated_center, bbox, frame_idx):
    """
    使用plotly交互式3D可视化，用户确认自动选择的区域

    Args:
        points_3d: 全部点云
        roi_points: 自动选择的ROI点云
        estimated_center: 估计的中心点
        bbox: 包围盒
        frame_idx: 帧索引

    Returns:
        confirmed: 用户是否确认
    """
    # 为了性能，下采样显示
    if len(points_3d) > 10000:
        indices = np.random.choice(len(points_3d), 10000, replace=False)
        display_points = points_3d[indices]
    else:
        display_points = points_3d

    # 创建3D scatter plot
    fig = go.Figure()

    # 全部点云（灰色，半透明）
    fig.add_trace(go.Scatter3d(
        x=display_points[:, 0],
        y=display_points[:, 1],
        z=display_points[:, 2],
        mode='markers',
        marker=dict(size=1, color='gray', opacity=0.3),
        name='全部点云',
        hoverinfo='skip'
    ))

    # ROI点云（彩色，按高度着色）
    if len(roi_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=roi_points[:, 0],
            y=roi_points[:, 1],
            z=roi_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=roi_points[:, 2],  # 按Z高度着色
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z (m)")
            ),
            name='选中区域',
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
        ))

    # 估计中心点（红色大点）
    fig.add_trace(go.Scatter3d(
        x=[estimated_center[0]],
        y=[estimated_center[1]],
        z=[estimated_center[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='估计中心',
        hovertemplate='中心: (%{x:.2f}, %{y:.2f}, %{z:.2f})'
    ))

    # 绘制包围盒
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    # 包围盒的12条边
    edges = [
        # 底面
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_min, x_max], [y_max, y_max], [z_min, z_min]),
        ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        # 顶面
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_min, x_max], [y_max, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        # 竖直边
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
    ]

    for i, (xs, ys, zs) in enumerate(edges):
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=(i == 0),
            name='包围盒' if i == 0 else None,
            hoverinfo='skip'
        ))

    # 设置布局
    fig.update_layout(
        title=f'Frame {frame_idx} - 3D点云选择预览<br><sub>关闭窗口后在终端输入 y 确认 / n 重选</sub>',
        scene=dict(
            xaxis_title='X (forward, m)',
            yaxis_title='Y (left, m)',
            zaxis_title='Z (up, m)',
            aspectmode='data'
        ),
        width=1200,
        height=800,
        showlegend=True
    )

    # 显示（在浏览器中打开）
    fig.show()

    # 等待用户确认
    print("\n  [3D预览] 在浏览器中查看选择结果")
    print("  确认选择吗？(y/n) ", end='')
    response = input().strip().lower()

    return response == 'y'


def select_board_roi_guided(points_3d, cam_center, cam_normal, R_init, t_init,
                            frame_idx, board_size=(0.6, 0.8),
                            manual_fallback=True):
    """
    投影引导的智能ROI选择

    流程：
    1. 利用初始外参将相机检测结果投影到LiDAR坐标系
    2. 自动框选估计区域
    3. 3D可视化预览
    4. 用户确认或手动调整

    Args:
        points_3d: LiDAR点云
        cam_center: 相机检测的标定板中心
        cam_normal: 相机检测的标定板法向量
        R_init: 初始外参旋转矩阵
        t_init: 初始外参平移向量
        frame_idx: 帧索引
        board_size: 标定板尺寸
        manual_fallback: 如果自动选择失败，是否降级到手动选择

    Returns:
        roi_points: 选中的ROI点云，失败返回None
    """
    print(f"\n=== Frame {frame_idx} 投影引导选择 ===")

    # 1. 自动估计ROI
    estimated_center, roi_points, bbox = auto_estimate_roi(
        points_3d, cam_center, R_init, t_init, board_size
    )

    if len(roi_points) < 10:
        print(f"  [警告] 自动选择点数太少 ({len(roi_points)} < 10)")
        if manual_fallback:
            print(f"  [降级] 切换到手动选择模式")
            return None  # 调用者处理降级
        else:
            return None

    # 2. 3D可视化预览
    confirmed = visualize_3d_interactive(
        points_3d, roi_points, estimated_center, bbox, frame_idx
    )

    if confirmed:
        print(f"  [确认] 使用自动选择的区域 ({len(roi_points)} 个点)")
        return roi_points
    else:
        print(f"  [拒绝] 用户拒绝自动选择")
        if manual_fallback:
            print(f"  [降级] 请手动调整包围盒范围")
            # TODO: 实现手动调整功能
            return None
        else:
            return None


def manual_adjust_bbox(points_3d, initial_bbox):
    """
    手动调整包围盒（命令行输入）

    Args:
        points_3d: 点云
        initial_bbox: 初始包围盒

    Returns:
        roi_points: 调整后的ROI点云
    """
    x_min, x_max, y_min, y_max, z_min, z_max = initial_bbox

    print("\n  [手动调整] 当前包围盒:")
    print(f"    X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"    Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"    Z: [{z_min:.2f}, {z_max:.2f}]")
    print("\n  输入调整值（直接回车跳过）:")

    # 逐个轴调整
    for axis, (v_min, v_max) in [('X', (x_min, x_max)),
                                  ('Y', (y_min, y_max)),
                                  ('Z', (z_min, z_max))]:
        inp = input(f"    {axis}_min (当前 {v_min:.2f}): ").strip()
        if inp:
            if axis == 'X':
                x_min = float(inp)
            elif axis == 'Y':
                y_min = float(inp)
            else:
                z_min = float(inp)

        inp = input(f"    {axis}_max (当前 {v_max:.2f}): ").strip()
        if inp:
            if axis == 'X':
                x_max = float(inp)
            elif axis == 'Y':
                y_max = float(inp)
            else:
                z_max = float(inp)

    # 提取ROI
    roi_mask = (
        (points_3d[:, 0] >= x_min) & (points_3d[:, 0] <= x_max) &
        (points_3d[:, 1] >= y_min) & (points_3d[:, 1] <= y_max) &
        (points_3d[:, 2] >= z_min) & (points_3d[:, 2] <= z_max)
    )
    roi_points = points_3d[roi_mask]

    print(f"\n  [调整后] 提取到 {len(roi_points)} 个点")

    return roi_points
