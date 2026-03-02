#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_verify.py - Camera Data Verification Tool

从两个 RealSense 相机采集稳定后的数据用于验证标定结果
- hand_camera: 从配置文件读取序列号
- top_camera: 从配置文件读取序列号

采集1秒时长的RGB和深度数据，保存到 calibration_data 目录
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
import logging
import yaml
import math
from libs.log_setting import CommonLog
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

# 配置日志
logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

# 采集参数
STABILIZATION_TIME = 2.0  # 相机稳定时间（秒）
SAVE_DATA = False         # 默认不保存数据


def load_calibration_results(hand_pipeline=None, top_pipeline=None):
    """加载标定结果（内参和外参）

    Args:
        hand_pipeline: hand camera pipeline (用于获取depth scale)
        top_pipeline: top camera pipeline (用于获取depth scale)

    Returns:
        dict: 包含两个相机的内参和外参
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    # D435 (hand camera) - eye-in-hand
    hand_intrinsics_file = os.path.join(results_dir, "eyeinhand_intrinsics_xarm_d435.yaml")
    hand_extrinsics_file = os.path.join(results_dir, "eyeinhand_extrintics_xarm_d435.yaml")

    # D455 (top camera) - eye-to-hand
    top_intrinsics_file = os.path.join(results_dir, "eyetohand_intrintics_xarm_d455.yaml")
    top_extrinsics_file = os.path.join(results_dir, "eyetohand_extrintics_xarm_d455.yaml")

    calib_data = {}

    # 获取真实的深度尺度
    hand_depth_scale = 0.001  # 默认值
    top_depth_scale = 0.001

    if hand_pipeline is not None:
        try:
            profile = hand_pipeline.get_active_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            hand_depth_scale = depth_sensor.get_depth_scale()
            logger_.info(f"  Hand Camera depth scale: {hand_depth_scale:.6f} m/unit")
        except Exception as e:
            logger_.warning(f"  无法获取 Hand Camera depth scale，使用默认值: {e}")

    if top_pipeline is not None:
        try:
            profile = top_pipeline.get_active_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            top_depth_scale = depth_sensor.get_depth_scale()
            logger_.info(f"  Top Camera depth scale: {top_depth_scale:.6f} m/unit")
        except Exception as e:
            logger_.warning(f"  无法获取 Top Camera depth scale，使用默认值: {e}")

    calib_data['hand_depth_scale'] = hand_depth_scale
    calib_data['top_depth_scale'] = top_depth_scale

    try:
        # 检查文件是否存在
        files_to_check = [
            ('Hand intrinsics', hand_intrinsics_file),
            ('Hand extrinsics', hand_extrinsics_file),
            ('Top intrinsics', top_intrinsics_file),
            ('Top extrinsics', top_extrinsics_file)
        ]

        missing_files = []
        for name, filepath in files_to_check:
            if not os.path.exists(filepath):
                missing_files.append((name, filepath))

        if missing_files:
            logger_.error("✗ 缺少标定文件:")
            for name, filepath in missing_files:
                logger_.error(f"  - {name}: {filepath}")
            return None

        # 加载 hand camera 内参
        logger_.info(f"  加载 Hand Camera 内参...")
        with open(hand_intrinsics_file, 'r') as f:
            hand_int = yaml.safe_load(f)
            calib_data['hand_intrinsics'] = {
                'camera_matrix': np.array(hand_int['camera_matrix'], dtype=np.float32),
                'dist_coeffs': np.array(hand_int['distortion_coefficients'], dtype=np.float32)
            }

        # 加载 hand camera 外参 (camera_to_gripper) - 只加载我们需要的字段
        logger_.info(f"  加载 Hand Camera 外参 (camera_to_gripper)...")
        with open(hand_extrinsics_file, 'r') as f:
            # 读取文件内容，手动解析我们需要的部分
            content = f.read()
            # 找到camera_to_gripper部分并提取
            import re

            # 提取rotation_matrix (3x3矩阵)
            rot_match = re.search(r'rotation_matrix:\s*\n(.*?)\n\s*translation:', content, re.DOTALL)
            if rot_match:
                rot_str = rot_match.group(1)
                # 提取所有数字
                numbers = re.findall(r'-?[\d\.]+(?:e[+-]?\d+)?', rot_str)
                if len(numbers) >= 9:
                    R_mat = np.array([float(x) for x in numbers[:9]], dtype=np.float32).reshape(3, 3)
                else:
                    raise ValueError(f"Found only {len(numbers)} numbers in rotation_matrix")
            else:
                raise ValueError("Cannot parse rotation_matrix")

            # 提取translation
            t_match = re.search(r'translation:\s*\n\s*x:\s*([\d\.\-e]+)\s*\n\s*y:\s*([\d\.\-e]+)\s*\n\s*z:\s*([\d\.\-e]+)', content)
            if t_match:
                t_vec = np.array([float(t_match.group(1)), float(t_match.group(2)), float(t_match.group(3))], dtype=np.float32)
            else:
                raise ValueError("Cannot parse translation")

            calib_data['hand_extrinsics'] = {
                'R': R_mat,
                't': t_vec,
                'type': 'camera_to_gripper'
            }

        # 加载 top camera 内参
        logger_.info(f"  加载 Top Camera 内参...")
        with open(top_intrinsics_file, 'r') as f:
            top_int = yaml.safe_load(f)
            calib_data['top_intrinsics'] = {
                'camera_matrix': np.array(top_int['camera_matrix'], dtype=np.float32),
                'dist_coeffs': np.array(top_int['distortion_coefficients'], dtype=np.float32)
            }

        # 加载 top camera 外参 (camera_to_base)
        logger_.info(f"  加载 Top Camera 外参 (camera_to_base)...")
        with open(top_extrinsics_file, 'r') as f:
            content = f.read()
            import re

            # 提取rotation_matrix
            rot_match = re.search(r'rotation_matrix:\s*\n(.*?)\n\s*translation:', content, re.DOTALL)
            if rot_match:
                rot_str = rot_match.group(1)
                # 提取所有数字
                numbers = re.findall(r'-?[\d\.]+(?:e[+-]?\d+)?', rot_str)
                if len(numbers) >= 9:
                    R_mat = np.array([float(x) for x in numbers[:9]], dtype=np.float32).reshape(3, 3)
                else:
                    raise ValueError(f"Found only {len(numbers)} numbers in rotation_matrix")
            else:
                raise ValueError("Cannot parse rotation_matrix")

            # 提取translation
            t_match = re.search(r'translation:\s*\n\s*x:\s*([\d\.\-e]+)\s*\n\s*y:\s*([\d\.\-e]+)\s*\n\s*z:\s*([\d\.\-e]+)', content)
            if t_match:
                t_vec = np.array([float(t_match.group(1)), float(t_match.group(2)), float(t_match.group(3))], dtype=np.float32)
            else:
                raise ValueError("Cannot parse translation")

            calib_data['top_extrinsics'] = {
                'R': R_mat,
                't': t_vec,
                'type': 'camera_to_base'
            }

        logger_.info("✓ 标定结果加载成功")
        logger_.info(f"  Hand Camera fx={calib_data['hand_intrinsics']['camera_matrix'][0,0]:.1f}")
        logger_.info(f"  Top Camera fx={calib_data['top_intrinsics']['camera_matrix'][0,0]:.1f}")

        return calib_data

    except FileNotFoundError as e:
        logger_.error(f"✗ 文件未找到: {e}")
        return None
    except KeyError as e:
        logger_.error(f"✗ YAML 文件格式错误，缺少键: {e}")
        logger_.error(f"   请检查标定结果文件格式是否正确")
        return None
    except Exception as e:
        logger_.error(f"✗ 加载标定结果失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def deproject_to_pointcloud(color_image, depth_image, camera_matrix, depth_scale=0.001):
    """将RGB+深度图反投影为3D点云

    Args:
        color_image: RGB图像 (H, W, 3)
        depth_image: 深度图像 (H, W)，RealSense 原始深度值
        camera_matrix: 相机内参矩阵 (3, 3)
        depth_scale: 深度缩放因子 (depth_unit -> m)，从相机获取

    Returns:
        tuple: (points, colors)
               points: (N, 3) 3D点坐标 (米)
               colors: (N, 3) RGB颜色 [0-255]
    """
    h, w = depth_image.shape
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 获取有效深度点（过滤掉0和异常值）
    valid_mask = (depth_image > 0) & (depth_image < 65535)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_image[valid_mask] * depth_scale  # 转换为米

    # 反投影到3D（相机坐标系）
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    points = np.stack([x, y, z_valid], axis=1)

    # 获取对应的颜色
    colors = color_image[valid_mask]

    return points, colors


def transform_points_to_base(points, R, t):
    """将点云变换到base坐标系

    Args:
        points: (N, 3) 3D点
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量 (米)

    Returns:
        (N, 3) 变换后的3D点
    """
    # 点云变换: p_base = R @ p_camera + t
    points_base = (R @ points.T).T + t
    return points_base


def get_gripper_pose(xarm):
    """获取机械臂末端位姿

    Args:
        xarm: XArmAPI实例

    Returns:
        tuple: (success, R, t)
               R: (3, 3) 旋转矩阵
               t: (3,) 平移向量 (米)
    """
    try:
        code, pos = xarm.get_position()

        if code != 0:
            logger_.error(f"获取位姿失败，错误码: {code}")
            return False, None, None

        if not pos or len(pos) < 6:
            logger_.error(f"位姿数据无效: {pos}")
            return False, None, None

        # Convert from xArm units (mm, degrees) to standard units (m, radians)
        x, y, z, roll, pitch, yaw = pos
        x = x / 1000.0  # mm -> m
        y = y / 1000.0  # mm -> m
        z = z / 1000.0  # mm -> m
        roll = roll * math.pi / 180.0  # deg -> rad
        pitch = pitch * math.pi / 180.0
        yaw = yaw * math.pi / 180.0

        # 构建旋转矩阵和平移向量
        from scipy.spatial.transform import Rotation as Rot
        rot = Rot.from_euler('xyz', [roll, pitch, yaw])
        R_gripper_to_base = rot.as_matrix()
        t_gripper_to_base = np.array([x, y, z], dtype=np.float32)

        return True, R_gripper_to_base, t_gripper_to_base

    except Exception as e:
        logger_.error(f"获取机械臂位姿异常: {e}")
        return False, None, None


def connect_xarm(ip_address="192.168.1.236"):
    """连接机械臂

    Args:
        ip_address: 机械臂IP地址

    Returns:
        XArmAPI实例或None
    """
    try:
        logger_.info(f"连接机械臂: {ip_address}")
        xarm = XArmAPI(ip_address)

        # 设置模式和状态
        xarm.motion_enable(enable=True)
        xarm.set_mode(0)
        xarm.set_state(state=0)

        # 验证连接
        code, pos = xarm.get_position()
        if code == 0:
            logger_.info(f"✓ 机械臂连接成功")
            logger_.info(f"  当前位置: {pos}")
            return xarm
        else:
            logger_.error(f"✗ 机械臂状态异常，错误码: {code}")
            return None

    except Exception as e:
        logger_.error(f"✗ 连接机械臂失败: {e}")
        return None


def project_pointcloud_to_view(points, colors, x_idx, y_idx, view_size, camera_color, roi_limits=None):
    """将点云投影到指定视图

    Args:
        points: (N, 3) 点云坐标
        colors: (N, 3) 点云颜色
        x_idx: X轴索引
        y_idx: Y轴索引
        view_size: (width, height) 视图大小
        camera_color: 相机标识颜色 (B, G, R)
        roi_limits: 可选的ROI范围，格式为 (x_min, x_max, y_min, y_max)，单位为米

    Returns:
        np.ndarray: 渲染后的视图图像
    """
    # 创建视图画布（浅灰色背景）
    view_canvas = np.ones((view_size[1], view_size[0], 3), dtype=np.uint8) * 200

    if len(points) == 0:
        # 添加提示文字
        cv2.putText(view_canvas, "No Point Cloud Data", (view_size[0]//2 - 80, view_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return view_canvas

    # 获取投影坐标
    x_vals = points[:, x_idx]
    y_vals = points[:, y_idx]

    # 如果指定了ROI范围，使用ROI过滤并设置固定范围
    if roi_limits is not None:
        x_min, x_max, y_min, y_max = roi_limits
        # 创建ROI掩码
        roi_mask = (x_vals >= x_min) & (x_vals <= x_max) & (y_vals >= y_min) & (y_vals <= y_max)
        valid_mask = roi_mask
    else:
        # 过滤异常值 (使用百分位数)
        x_p1, x_p99 = np.percentile(x_vals, [1, 99])
        y_p1, y_p99 = np.percentile(y_vals, [1, 99])
        # 创建有效点掩码
        valid_mask = (x_vals >= x_p1) & (x_vals <= x_p99) & (y_vals >= y_p1) & (y_vals <= y_p99)
        x_min, x_max = x_p1, x_p99
        y_min, y_max = y_p1, y_p99

    if valid_mask.sum() == 0:
        cv2.putText(view_canvas, "No Valid Points in ROI", (view_size[0]//2 - 100, view_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return view_canvas

    # 如果没有指定ROI，添加边距
    if roi_limits is None:
        margin = 0.15
        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin

    # 计算缩放比例（保持纵横比，并留出边距）
    scale_x = (view_size[0] - 40) / (x_max - x_min)  # 留出20px边距
    scale_y = (view_size[1] - 40) / (y_max - y_min)
    scale = min(scale_x, scale_y)

    # 计算居中偏移
    total_width = (x_max - x_min) * scale
    total_height = (y_max - y_min) * scale
    offset_x = (view_size[0] - total_width) / 2
    offset_y = (view_size[1] - total_height) / 2

    # 绘制坐标轴网格
    draw_grid(view_canvas, x_min, x_max, y_min, y_max, scale, view_size, x_idx, y_idx)

    # 投影所有点云（不只是过滤后的点）- 使用实际RGB颜色
    point_size = 2
    rendered_count = 0

    for i, point in enumerate(points):
        # 计算投影坐标（相对于数据范围）
        x_normalized = (point[x_idx] - x_min) * scale
        y_normalized = (point[y_idx] - y_min) * scale

        # 添加居中偏移
        x_proj = int(x_normalized + offset_x)
        y_proj = int(y_normalized + offset_y)

        # 翻转Y轴（图像坐标系）
        y_proj = view_size[1] - y_proj

        # 检查是否在视图范围内
        if 0 <= x_proj < view_size[0] and 0 <= y_proj < view_size[1]:
            # 使用实际颜色（如果可用）
            if colors is not None and len(colors) > i:
                # RGB to BGR for OpenCV
                color = tuple(map(int, colors[i][::-1]))
            else:
                color = camera_color

            cv2.circle(view_canvas, (x_proj, y_proj), point_size, color, -1)
            rendered_count += 1

    # 添加统计信息（背景）
    info_bg_height = 35
    overlay = view_canvas.copy()
    cv2.rectangle(overlay, (0, view_size[1] - info_bg_height), (view_size[0], view_size[1]), (255, 255, 255), -1)
    view_canvas = cv2.addWeighted(view_canvas, 0.6, overlay, 0.4, 0)

    # 添加范围信息
    range_text = f"Range: X[{x_min:.2f}~{x_max:.2f}] Y[{y_min:.2f}~{y_max:.2f}]m"
    cv2.putText(view_canvas, range_text, (5, view_size[1] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

    # 添加渲染点数信息
    count_text = f"Points: {rendered_count}/{len(points)} ({100*rendered_count/len(points):.1f}%)"
    cv2.putText(view_canvas, count_text, (5, view_size[1] - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

    return view_canvas


def draw_grid(canvas, x_min, x_max, y_min, y_max, scale, view_size, x_idx, y_idx):
    """绘制坐标网格

    Args:
        canvas: 画布
        x_min, x_max, y_min, y_max: 数据范围
        scale: 缩放比例
        view_size: 视图大小
        x_idx, y_idx: 坐标轴索引
    """
    axis_names = ['X', 'Y', 'Z']

    # 绘制网格线
    grid_color = (180, 180, 180)

    # 水平网格线（5条）
    for i in range(1, 5):
        y = int(view_size[1] * i / 5)
        cv2.line(canvas, (0, y), (view_size[0], y), grid_color, 1)

    # 垂直网格线（5条）
    for i in range(1, 5):
        x = int(view_size[0] * i / 5)
        cv2.line(canvas, (x, 0), (x, view_size[1]), grid_color, 1)

    # 绘制坐标轴标签
    x_label = axis_names[x_idx]
    y_label = axis_names[y_idx]

    # X轴标签（底部中间）
    cv2.putText(canvas, x_label, (view_size[0] - 20, view_size[1] - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    # Y轴标签（左上角）
    cv2.putText(canvas, y_label, (5, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)


def create_xy_overlay_view(hand_points, hand_colors, top_points, top_colors, view_size=(600, 500), roi_limits=None):
    """创建XY视图的重叠比较图

    Args:
        hand_points: (N, 3) hand camera点云（base坐标系）
        hand_colors: (N, 3) hand camera颜色
        top_points: (M, 3) top camera点云（base坐标系）
        top_colors: (M, 3) top camera颜色
        view_size: (width, height) 视图大小
        roi_limits: ROI范围 (x_min, x_max, y_min, y_max)

    Returns:
        np.ndarray: 重叠视图图像
    """
    # 创建黑色背景
    canvas = np.zeros((view_size[1], view_size[0], 3), dtype=np.uint8)

    # 定义相机特征颜色 (用于标识,不是实际RGB)
    hand_marker_color = np.array([255, 100, 100], dtype=np.uint8)  # 浅红色
    top_marker_color = np.array([100, 255, 100], dtype=np.uint8)   # 浅绿色

    # XY视图: x_idx=0, y_idx=1
    x_idx, y_idx = 0, 1

    # 合并所有点以确定全局范围
    all_points = []
    if len(hand_points) > 0:
        all_points.append(hand_points)
    if len(top_points) > 0:
        all_points.append(top_points)

    if len(all_points) == 0:
        cv2.putText(canvas, "No Point Cloud Data", (view_size[0]//2 - 100, view_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas

    all_points = np.vstack(all_points)
    x_vals = all_points[:, x_idx]
    y_vals = all_points[:, y_idx]

    # 确定显示范围
    if roi_limits is not None:
        x_min, x_max, y_min, y_max = roi_limits
    else:
        x_min, x_max = np.percentile(x_vals, [1, 99])
        y_min, y_max = np.percentile(y_vals, [1, 99])
        # 添加边距
        margin = 0.15
        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)
        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin

    # 计算缩放比例
    margin_px = 60  # 留出更多边距用于标注
    scale_x = (view_size[0] - 2*margin_px) / (x_max - x_min)
    scale_y = (view_size[1] - 2*margin_px) / (y_max - y_min)
    scale = min(scale_x, scale_y)

    # 计算居中偏移
    total_width = (x_max - x_min) * scale
    total_height = (y_max - y_min) * scale
    offset_x = (view_size[0] - total_width) / 2
    offset_y = (view_size[1] - total_height) / 2

    # 绘制网格
    grid_color = (50, 50, 50)
    for i in range(1, 10):
        y = int(margin_px + (view_size[1] - 2*margin_px) * i / 10)
        cv2.line(canvas, (margin_px, y), (view_size[0]-margin_px, y), grid_color, 1)
        x = int(margin_px + (view_size[0] - 2*margin_px) * i / 10)
        cv2.line(canvas, (x, margin_px), (x, view_size[1]-margin_px), grid_color, 1)

    # 辅助函数：投影点云到视图
    def project_points(points, colors, marker_color, point_size=2, alpha=0.7):
        rendered = 0
        for i, point in enumerate(points):
            x_normalized = (point[x_idx] - x_min) * scale
            y_normalized = (point[y_idx] - y_min) * scale

            x_proj = int(x_normalized + offset_x)
            y_proj = int(y_normalized + offset_y)
            y_proj = view_size[1] - y_proj  # 翻转Y轴

            if 0 <= x_proj < view_size[0] and 0 <= y_proj < view_size[1]:
                # 使用实际颜色,但略微混合标识颜色以区分
                if colors is not None and len(colors) > i:
                    actual_color = colors[i][::-1]  # RGB to BGR
                    # 混合实际颜色和标识颜色
                    blended = (actual_color * alpha + marker_color * (1-alpha)).astype(np.uint8)
                else:
                    blended = marker_color

                cv2.circle(canvas, (x_proj, y_proj), point_size, tuple(map(int, blended)), -1)
                rendered += 1
        return rendered

    # 先绘制Top Camera (绿色调)
    top_rendered = 0
    if len(top_points) > 0:
        top_rendered = project_points(top_points, top_colors, top_marker_color, point_size=2, alpha=0.6)

    # 后绘制Hand Camera (红色调)
    hand_rendered = 0
    if len(hand_points) > 0:
        hand_rendered = project_points(hand_points, hand_colors, hand_marker_color, point_size=2, alpha=0.6)

    # 添加标题
    title = "XY Overlay Comparison (Base Frame)"
    cv2.putText(canvas, title, (view_size[0]//2 - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 添加图例 (右上角)
    legend_x = view_size[0] - 220
    legend_y = 60

    # Hand Camera 图例
    cv2.circle(canvas, (legend_x, legend_y), 8, tuple(map(int, hand_marker_color)), -1)
    cv2.putText(canvas, f"Hand Camera ({hand_rendered}/{len(hand_points)} pts)",
               (legend_x + 20, legend_y + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Top Camera 图例
    legend_y += 25
    cv2.circle(canvas, (legend_x, legend_y), 8, tuple(map(int, top_marker_color)), -1)
    cv2.putText(canvas, f"Top Camera ({top_rendered}/{len(top_points)} pts)",
               (legend_x + 20, legend_y + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 添加坐标轴标签和范围信息
    axis_color = (200, 200, 200)
    # X轴
    cv2.putText(canvas, "X (forward)", (view_size[0]//2 - 40, view_size[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)
    cv2.putText(canvas, f"{x_min:.2f}m", (margin_px, view_size[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)
    cv2.putText(canvas, f"{x_max:.2f}m", (view_size[0] - margin_px - 50, view_size[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)

    # Y轴 (旋转文字)
    cv2.putText(canvas, "Y (right)", (10, view_size[1]//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)
    cv2.putText(canvas, f"{y_max:.2f}m", (10, margin_px + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)
    cv2.putText(canvas, f"{y_min:.2f}m", (10, view_size[1] - margin_px),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)

    # 添加重叠分析信息 (左下角)
    info_y_start = view_size[1] - 80
    cv2.rectangle(canvas, (5, info_y_start - 5), (300, view_size[1] - 5), (40, 40, 40), -1)
    cv2.rectangle(canvas, (5, info_y_start - 5), (300, view_size[1] - 5), (100, 100, 100), 1)

    info_text = [
        "Overlap Analysis:",
        f"  Range: X[{x_min:.2f}~{x_max:.2f}]",
        f"         Y[{y_min:.2f}~{y_max:.2f}]m"
    ]

    for i, text in enumerate(info_text):
        cv2.putText(canvas, text, (10, info_y_start + 15 + i*18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return canvas


def visualize_dual_pointcloud(hand_points, hand_colors, top_points, top_colors):
    """可视化两个相机的点云投影到2D平面（2x3布局 + XY重叠视图）

    Args:
        hand_points: (N, 3) hand camera点云（base坐标系）
        hand_colors: (N, 3) hand camera颜色
        top_points: (M, 3) top camera点云（base坐标系）
        top_colors: (M, 3) top camera颜色

    Returns:
        np.ndarray: 可视化图像
    """
    # 2x3布局，每个视图大小
    view_size = (400, 300)
    canvas_width = view_size[0] * 3
    canvas_height = view_size[1] * 2

    # 白色背景
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 打印点云范围信息（用于调试）
    if len(hand_points) > 0:
        hand_min = hand_points.min(axis=0)
        hand_max = hand_points.max(axis=0)
        logger_.debug(f"Hand Camera (Base Frame): X[{hand_min[0]:.3f}, {hand_max[0]:.3f}] "
                     f"Y[{hand_min[1]:.3f}, {hand_max[1]:.3f}] Z[{hand_min[2]:.3f}, {hand_max[2]:.3f}]")

    if len(top_points) > 0:
        top_min = top_points.min(axis=0)
        top_max = top_points.max(axis=0)
        logger_.debug(f"Top Camera (Base Frame): X[{top_min[0]:.3f}, {top_max[0]:.3f}] "
                     f"Y[{top_min[1]:.3f}, {top_max[1]:.3f}] Z[{top_min[2]:.3f}, {top_max[2]:.3f}]")

    # 定义三个视角（Base坐标系：X前，Y右，Z上 - 右手系）
    # 每个视角都定义ROI范围，限制显示区域为base原点附近的工作区域
    views = [
        {
            'name': 'Top (X-Y)',
            'x_idx': 0, 'y_idx': 1,      # 俯视图：X前，Y右
            'roi': (0.0, 1.0, -0.5, 0.5)  # X: 0-100cm前方, Y: 左右各50cm
        },
        {
            'name': 'Front (X-Z)',
            'x_idx': 0, 'y_idx': 2,       # 正视图：X前，Z上
            'roi': (0.0, 1.0, 0.0, 1.0)   # X: 0-100cm前方, Z: 0-100cm高度
        },
        {
            'name': 'Side (Y-Z)',
            'x_idx': 1, 'y_idx': 2,       # 侧视图：Y右，Z上
            'roi': (-0.5, 0.5, 0.0, 1.0)  # Y: 左右各50cm, Z: 0-100cm高度
        }
    ]

    # 相机颜色（作为备用颜色）
    hand_color = (0, 0, 200)  # 深红色 (BGR)
    top_color = (200, 0, 0)   # 深蓝色 (BGR)

    # 第一行：Hand Camera 的3个视图
    for col, view in enumerate(views):
        view_img = project_pointcloud_to_view(
            hand_points, hand_colors,
            view['x_idx'], view['y_idx'],
            view_size, hand_color,
            roi_limits=view['roi']
        )

        # 添加标题
        title = f"Hand Camera - {view['name']}"
        cv2.putText(view_img, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 放置到画布
        x_start = col * view_size[0]
        y_start = 0
        canvas[y_start:y_start+view_size[1], x_start:x_start+view_size[0]] = view_img

    # 第二行：Top Camera 的3个视图
    for col, view in enumerate(views):
        view_img = project_pointcloud_to_view(
            top_points, top_colors,
            view['x_idx'], view['y_idx'],
            view_size, top_color,
            roi_limits=view['roi']
        )

        # 添加标题
        title = f"Top Camera - {view['name']}"
        cv2.putText(view_img, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 放置到画布
        x_start = col * view_size[0]
        y_start = view_size[1]
        canvas[y_start:y_start+view_size[1], x_start:x_start+view_size[0]] = view_img

    # 添加分隔线
    cv2.line(canvas, (0, view_size[1]), (canvas_width, view_size[1]), (150, 150, 150), 3)
    for col in range(1, 3):
        x = col * view_size[0]
        cv2.line(canvas, (x, 0), (x, canvas_height), (150, 150, 150), 2)

    # 添加坐标系说明和统计信息
    coord_text = "Base Frame: X(forward) Y(right) Z(up) [Right-hand]"
    cv2.putText(canvas, coord_text, (canvas_width - 400, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    info_text = f"Hand: {len(hand_points)} pts | Top: {len(top_points)} pts"
    cv2.putText(canvas, info_text, (10, canvas_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    return canvas


def visualize_depth(depth_image, camera_name="", frame_id=0):
    """高区分度深度图可视化，使用自适应直方图均衡化和对比度增强

    Args:
        depth_image: 深度图像（uint16）
        camera_name: 相机名称
        frame_id: 帧ID

    Returns:
        np.ndarray: 彩色深度图
    """
    # 过滤无效深度值（0值）
    valid_depth = depth_image[depth_image > 0]

    if len(valid_depth) == 0:
        # 如果没有有效深度值，返回黑色图像
        return np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

    # 使用更激进的百分位数来确定深度范围
    depth_min = np.percentile(valid_depth, 1)   # 1%分位数
    depth_max = np.percentile(valid_depth, 99)  # 99%分位数

    # 确保有有效范围
    if depth_max - depth_min < 10:
        depth_max = depth_min + 100

    # 创建掩码，标记有效深度区域
    valid_mask = depth_image > 0

    # 第一步：归一化到0-1范围
    depth_normalized = np.zeros_like(depth_image, dtype=np.float32)
    depth_normalized[valid_mask] = np.clip(
        (depth_image[valid_mask] - depth_min) / (depth_max - depth_min),
        0, 1
    )

    # 第二步：应用Gamma校正增强对比度（gamma < 1 增强细节）
    gamma = 0.7
    depth_normalized[valid_mask] = np.power(depth_normalized[valid_mask], gamma)

    # 第三步：转换为8位整数
    depth_8bit = (depth_normalized * 255).astype(np.uint8)

    # 第四步：对有效区域应用CLAHE（自适应直方图均衡化）增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    depth_enhanced = depth_8bit.copy()

    # 只对有效区域应用CLAHE
    if np.any(valid_mask):
        # 创建一个临时图像用于CLAHE
        temp_depth = depth_8bit.copy()
        temp_depth = clahe.apply(temp_depth)
        # 只更新有效区域
        depth_enhanced[valid_mask] = temp_depth[valid_mask]

    # 第五步：应用更好的颜色映射
    # 尝试使用不同的颜色映射以获得最佳效果
    try:
        # TURBO: 彩虹色，视觉效果好
        depth_colormap = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_TURBO)
    except:
        try:
            # VIRIDIS: 感知均匀，科学可视化常用
            depth_colormap = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_VIRIDIS)
        except:
            # 回退到JET
            depth_colormap = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_JET)

    # 将无效深度区域设为深灰色（更明显的区分）
    depth_colormap[~valid_mask] = [40, 40, 40]

    # 添加边缘检测增强视觉效果（可选）
    edges = cv2.Canny(depth_enhanced, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # 边缘用白色高亮
    depth_colormap[edges > 0] = (depth_colormap[edges > 0] * 0.7 + np.array([255, 255, 255]) * 0.3).astype(np.uint8)

    # 添加深度信息文本
    depth_info = f"{camera_name} Depth - Frame {frame_id}"
    range_info = f"Range: {depth_min:.0f}mm - {depth_max:.0f}mm"
    enhance_info = f"Enhanced (Gamma={gamma}, CLAHE)"

    # 添加半透明背景使文字更清晰
    overlay = depth_colormap.copy()
    cv2.rectangle(overlay, (5, 5), (450, 95), (0, 0, 0), -1)
    depth_colormap = cv2.addWeighted(depth_colormap, 0.7, overlay, 0.3, 0)

    # 添加文字
    cv2.putText(depth_colormap, depth_info, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(depth_colormap, range_info, (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(depth_colormap, enhance_info, (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return depth_colormap


def load_config():
    """加载配置文件

    Returns:
        dict: 配置数据，包含两个相机的设置
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config", "calibration_config.yaml")

    if not os.path.exists(config_file):
        logger_.error(f"配置文件不存在: {config_file}")
        return None

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # 提取两个相机的配置
        hand_camera_config = config.get('eye_in_hand', {}).get('camera', {})
        top_camera_config = config.get('eye_to_hand', {}).get('camera', {})

        # 获取分辨率和帧率，使用默认值如果未配置
        hand_resolution = hand_camera_config.get('resolution', [1280, 720])
        hand_fps = hand_camera_config.get('fps', 30)
        top_resolution = top_camera_config.get('resolution', [1280, 720])
        top_fps = top_camera_config.get('fps', 30)

        camera_configs = {
            'hand_camera': {
                'serial': hand_camera_config.get('serial_id', '344322074267'),
                'resolution': tuple(hand_resolution),
                'fps': hand_fps
            },
            'top_camera': {
                'serial': top_camera_config.get('serial_id', '318122302992'),
                'resolution': tuple(top_resolution),
                'fps': top_fps
            }
        }

        logger_.info("配置文件加载成功")
        return camera_configs

    except Exception as e:
        logger_.error(f"加载配置文件失败: {e}")
        return None


def initialize_camera(serial_number, camera_name, resolution, fps):
    """初始化指定序列号的 RealSense 相机

    Args:
        serial_number: 相机序列号
        camera_name: 相机名称（用于日志）
        resolution: 分辨率 (width, height)
        fps: 帧率

    Returns:
        tuple: (pipeline, align) 或 (None, None) 如果失败
    """
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # 指定特定相机
        config.enable_device(serial_number)

        # 配置流
        width, height = resolution
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # 启动相机
        profile = pipeline.start(config)

        # 创建对齐对象（将深度对齐到彩色）
        align = rs.align(rs.stream.color)

        # 获取相机信息
        device = profile.get_device()
        actual_serial = device.get_info(rs.camera_info.serial_number)
        device_name = device.get_info(rs.camera_info.name)

        logger_.info(f"✓ {camera_name} 初始化成功")
        logger_.info(f"  设备: {device_name}")
        logger_.info(f"  序列号: {actual_serial}")
        logger_.info(f"  分辨率: {width}x{height} @ {fps}fps")

        return pipeline, align

    except Exception as e:
        logger_.error(f"✗ {camera_name} 初始化失败: {e}")
        return None, None


def preview_cameras(hand_pipeline, hand_align, top_pipeline, top_align):
    """预览两个相机的实时画面

    Args:
        hand_pipeline: hand_camera pipeline
        hand_align: hand_camera align object
        top_pipeline: top_camera pipeline
        top_align: top_camera align object
    """
    logger_.info("="*60)
    logger_.info("预览模式 - 按 [SPACE] 开始采集, 按 [ESC] 退出")
    logger_.info("="*60)

    # 创建窗口
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 1280, 960)

    start_capture = False

    while True:
        # 获取hand_camera的帧
        hand_frames = hand_pipeline.wait_for_frames()
        hand_frames = hand_align.process(hand_frames)
        hand_color = np.asanyarray(hand_frames.get_color_frame().get_data())
        hand_depth = np.asanyarray(hand_frames.get_depth_frame().get_data())

        # 获取top_camera的帧
        top_frames = top_pipeline.wait_for_frames()
        top_frames = top_align.process(top_frames)
        top_color = np.asanyarray(top_frames.get_color_frame().get_data())
        top_depth = np.asanyarray(top_frames.get_depth_frame().get_data())

        # 创建优化的深度彩色映射
        hand_depth_colormap = visualize_depth(hand_depth, "Hand Camera", 0)
        top_depth_colormap = visualize_depth(top_depth, "Top Camera", 0)

        # 调整尺寸
        hand_color_resized = cv2.resize(hand_color, (640, 360))
        hand_depth_resized = cv2.resize(hand_depth_colormap, (640, 360))
        top_color_resized = cv2.resize(top_color, (640, 360))
        top_depth_resized = cv2.resize(top_depth_colormap, (640, 360))

        # 添加RGB标签
        cv2.putText(hand_color_resized, "Hand Camera - RGB", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(top_color_resized, "Top Camera - RGB", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 组合显示（2x2布局）
        top_row = np.hstack((hand_color_resized, hand_depth_resized))
        bottom_row = np.hstack((top_color_resized, top_depth_resized))
        display = np.vstack((top_row, bottom_row))

        # 添加底部提示
        cv2.putText(display, "[SPACE] Start Capture  [ESC] Exit", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Preview", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            logger_.info("用户取消采集")
            cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE
            logger_.info("开始采集...")
            cv2.destroyAllWindows()
            return True


def wait_for_stable_camera(pipeline, align, camera_name):
    """等待相机自动曝光稳定

    Args:
        pipeline: RealSense pipeline
        align: RealSense align object
        camera_name: 相机名称
    """
    logger_.info(f"{camera_name}: 等待相机稳定 ({STABILIZATION_TIME}秒)...")

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < STABILIZATION_TIME:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        frame_count += 1

    logger_.info(f"{camera_name}: 已稳定 (丢弃 {frame_count} 帧)")


def capture_data_dual(hand_pipeline, hand_align, top_pipeline, top_align, output_dir, calib_data=None, show_pointcloud=False, xarm=None, save_data=False):
    """同时采集两个相机的RGB和深度数据并持续可视化

    Args:
        hand_pipeline: hand_camera pipeline
        hand_align: hand_camera align object
        top_pipeline: top_camera pipeline
        top_align: top_camera align object
        output_dir: 输出目录
        calib_data: 标定数据（用于点云可视化）
        show_pointcloud: 是否显示点云可视化
        xarm: XArmAPI实例（用于获取末端位姿）
        save_data: 是否保存数据

    Returns:
        tuple: (hand_frames, top_frames) 采集的帧数
    """
    logger_.info("="*60)
    logger_.info("持续可视化模式")
    logger_.info("  按 [S] 保存当前帧")
    logger_.info("  按 [ESC] 退出")
    logger_.info("="*60)

    # 创建可视化窗口
    window_name_main = "Dual Camera Capture - [S]Save [ESC]Exit"
    cv2.namedWindow(window_name_main, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_main, 1280, 960)

    if show_pointcloud:
        window_name_pc = "Point Cloud Visualization (Base Frame)"
        cv2.namedWindow(window_name_pc, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name_pc, 1200, 600)

        # XY重叠视图窗口
        window_name_overlay = "XY Overlay Comparison"
        cv2.namedWindow(window_name_overlay, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name_overlay, 600, 500)

    start_time = time.time()
    hand_frame_id = 0
    top_frame_id = 0
    saved_frames = 0

    try:
        while True:  # 持续运行，直到用户按ESC
            # 获取hand_camera的帧
            hand_frames = hand_pipeline.wait_for_frames()
            hand_frames = hand_align.process(hand_frames)
            hand_color_frame = hand_frames.get_color_frame()
            hand_depth_frame = hand_frames.get_depth_frame()

            # 获取top_camera的帧
            top_frames = top_pipeline.wait_for_frames()
            top_frames = top_align.process(top_frames)
            top_color_frame = top_frames.get_color_frame()
            top_depth_frame = top_frames.get_depth_frame()

            if not hand_color_frame or not hand_depth_frame or not top_color_frame or not top_depth_frame:
                continue

            # 转换为numpy数组
            hand_color = np.asanyarray(hand_color_frame.get_data())
            hand_depth = np.asanyarray(hand_depth_frame.get_data())
            top_color = np.asanyarray(top_color_frame.get_data())
            top_depth = np.asanyarray(top_depth_frame.get_data())

            # 创建优化的深度彩色映射
            hand_depth_colormap = visualize_depth(hand_depth, "Hand Camera", hand_frame_id)
            top_depth_colormap = visualize_depth(top_depth, "Top Camera", top_frame_id)

            # 调整尺寸以便显示
            hand_color_resized = cv2.resize(hand_color, (640, 360))
            hand_depth_resized = cv2.resize(hand_depth_colormap, (640, 360))
            top_color_resized = cv2.resize(top_color, (640, 360))
            top_depth_resized = cv2.resize(top_depth_colormap, (640, 360))

            # 添加RGB标签
            cv2.putText(hand_color_resized, f"Hand Camera RGB - Frame {hand_frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(top_color_resized, f"Top Camera RGB - Frame {top_frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 组合显示（2x2布局）
            top_row = np.hstack((hand_color_resized, hand_depth_resized))
            bottom_row = np.hstack((top_color_resized, top_depth_resized))
            display = np.vstack((top_row, bottom_row))

            # 添加运行信息
            elapsed = time.time() - start_time
            info_text = f"Time: {elapsed:.1f}s | Frames: {hand_frame_id} | Saved: {saved_frames}"
            cv2.putText(display, info_text, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 可选：添加点云可视化
            if show_pointcloud and calib_data is not None:
                # 反投影到点云（使用正确的深度尺度）
                hand_points, hand_colors_pc = deproject_to_pointcloud(
                    hand_color, hand_depth,
                    calib_data['hand_intrinsics']['camera_matrix'],
                    depth_scale=calib_data['hand_depth_scale']
                )

                top_points, top_colors_pc = deproject_to_pointcloud(
                    top_color, top_depth,
                    calib_data['top_intrinsics']['camera_matrix'],
                    depth_scale=calib_data['top_depth_scale']
                )

                # Top camera 已经在 base 坐标系下 (camera_to_base)
                top_points_base = transform_points_to_base(
                    top_points,
                    calib_data['top_extrinsics']['R'],
                    calib_data['top_extrinsics']['t']
                )

                # Hand camera 需要两步变换: camera -> gripper -> base
                if xarm is not None:
                    # 获取当前gripper位姿
                    success, R_gripper_to_base, t_gripper_to_base = get_gripper_pose(xarm)

                    if success:
                        # 第一步: camera -> gripper
                        hand_points_gripper = transform_points_to_base(
                            hand_points,
                            calib_data['hand_extrinsics']['R'],
                            calib_data['hand_extrinsics']['t']
                        )

                        # 第二步: gripper -> base
                        hand_points_base = transform_points_to_base(
                            hand_points_gripper,
                            R_gripper_to_base,
                            t_gripper_to_base
                        )
                    else:
                        # 如果获取位姿失败，仅使用camera_to_gripper变换
                        logger_.warning("无法获取gripper位姿，hand camera点云可能不准确")
                        hand_points_base = transform_points_to_base(
                            hand_points,
                            calib_data['hand_extrinsics']['R'],
                            calib_data['hand_extrinsics']['t']
                        )
                else:
                    # 如果没有连接机械臂，仅使用camera_to_gripper变换
                    hand_points_base = transform_points_to_base(
                        hand_points,
                        calib_data['hand_extrinsics']['R'],
                        calib_data['hand_extrinsics']['t']
                    )

                # 下采样点云以加快渲染速度
                downsample_factor = 5  # 减少下采样，保留更多点
                hand_points_ds = hand_points_base[::downsample_factor]
                hand_colors_ds = hand_colors_pc[::downsample_factor]
                top_points_ds = top_points_base[::downsample_factor]
                top_colors_ds = top_colors_pc[::downsample_factor]

                # 首帧输出调试信息
                if hand_frame_id == 1:
                    logger_.info("="*60)
                    logger_.info("首帧点云统计 (Base 坐标系):")
                    if len(hand_points_base) > 0:
                        h_min = hand_points_base.min(axis=0)
                        h_max = hand_points_base.max(axis=0)
                        logger_.info(f"  Hand Camera: {len(hand_points_base)} points")
                        logger_.info(f"    X: [{h_min[0]:.3f}, {h_max[0]:.3f}] m (forward)")
                        logger_.info(f"    Y: [{h_min[1]:.3f}, {h_max[1]:.3f}] m (left)")
                        logger_.info(f"    Z: [{h_min[2]:.3f}, {h_max[2]:.3f}] m (up)")
                    if len(top_points_base) > 0:
                        t_min = top_points_base.min(axis=0)
                        t_max = top_points_base.max(axis=0)
                        logger_.info(f"  Top Camera: {len(top_points_base)} points")
                        logger_.info(f"    X: [{t_min[0]:.3f}, {t_max[0]:.3f}] m (forward)")
                        logger_.info(f"    Y: [{t_min[1]:.3f}, {t_max[1]:.3f}] m (left)")
                        logger_.info(f"    Z: [{t_min[2]:.3f}, {t_max[2]:.3f}] m (up)")
                    logger_.info("="*60)

                # 生成点云可视化（独立窗口）
                pointcloud_vis = visualize_dual_pointcloud(
                    hand_points_ds, hand_colors_ds,
                    top_points_ds, top_colors_ds
                )
                cv2.imshow(window_name_pc, pointcloud_vis)

                # 生成XY重叠比较视图
                # 使用与主视图相同的ROI范围
                roi_xy = (0.0, 1.0, -0.5, 0.5)  # X: 0-100cm, Y: 左右各50cm
                overlay_view = create_xy_overlay_view(
                    hand_points_ds, hand_colors_ds,
                    top_points_ds, top_colors_ds,
                    view_size=(600, 500),
                    roi_limits=roi_xy
                )
                cv2.imshow(window_name_overlay, overlay_view)

            # 显示主窗口（相机视图）
            cv2.imshow(window_name_main, display)

            # 检查键盘输入
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - 退出
                logger_.info("用户按ESC退出")
                break
            elif key == ord('s') or key == ord('S'):  # S - 保存当前帧
                # 保存hand_camera图像
                hand_color_filename = os.path.join(output_dir, f"hand_camera_frame_{saved_frames:03d}_color.png")
                hand_depth_filename = os.path.join(output_dir, f"hand_camera_frame_{saved_frames:03d}_depth.png")
                cv2.imwrite(hand_color_filename, hand_color)
                cv2.imwrite(hand_depth_filename, hand_depth)

                # 保存top_camera图像
                top_color_filename = os.path.join(output_dir, f"top_camera_frame_{saved_frames:03d}_color.png")
                top_depth_filename = os.path.join(output_dir, f"top_camera_frame_{saved_frames:03d}_depth.png")
                cv2.imwrite(top_color_filename, top_color)
                cv2.imwrite(top_depth_filename, top_depth)

                saved_frames += 1
                logger_.info(f"✓ 保存第 {saved_frames} 帧")

            hand_frame_id += 1
            top_frame_id += 1

    finally:
        cv2.destroyWindow(window_name_main)
        if show_pointcloud:
            cv2.destroyWindow(window_name_pc)
            cv2.destroyWindow(window_name_overlay)

    logger_.info(f"可视化结束:")
    logger_.info(f"  总帧数: {hand_frame_id}")
    logger_.info(f"  保存帧数: {saved_frames}")

    return hand_frame_id, top_frame_id


def save_camera_intrinsics(pipeline, camera_name, output_dir):
    """保存相机内参

    Args:
        pipeline: RealSense pipeline
        camera_name: 相机名称
        output_dir: 输出目录
    """
    try:
        profile = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        depth_stream = profile.get_stream(rs.stream.depth)

        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # 获取深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # 保存内参信息
        intrinsics_file = os.path.join(output_dir, f"{camera_name}_intrinsics.txt")
        with open(intrinsics_file, 'w', encoding='utf-8') as f:
            f.write(f"{camera_name} Camera Intrinsics\n")
            f.write(f"{'='*50}\n\n")

            f.write("Color Intrinsics:\n")
            f.write(f"  Resolution: {color_intrinsics.width} x {color_intrinsics.height}\n")
            f.write(f"  Focal Length: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}\n")
            f.write(f"  Principal Point: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}\n")
            f.write(f"  Distortion Model: {color_intrinsics.model}\n")
            f.write(f"  Distortion Coeffs: {color_intrinsics.coeffs}\n\n")

            f.write("Depth Intrinsics:\n")
            f.write(f"  Resolution: {depth_intrinsics.width} x {depth_intrinsics.height}\n")
            f.write(f"  Focal Length: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}\n")
            f.write(f"  Principal Point: cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}\n")
            f.write(f"  Distortion Model: {depth_intrinsics.model}\n")
            f.write(f"  Distortion Coeffs: {depth_intrinsics.coeffs}\n\n")

            f.write(f"Depth Scale: {depth_scale:.6f} (meters per unit)\n")

        logger_.info(f"  内参已保存: {intrinsics_file}")

    except Exception as e:
        logger_.warning(f"  保存内参失败: {e}")


def main():
    """主函数"""
    # 初始化变量
    hand_pipeline = None
    top_pipeline = None
    xarm = None

    # 加载配置
    camera_configs = load_config()
    if camera_configs is None:
        logger_.error("无法加载配置文件，程序退出")
        return

    # 创建输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "calibration_data", f"verify_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    hand_cfg = camera_configs['hand_camera']
    top_cfg = camera_configs['top_camera']

    logger_.info("="*60)
    logger_.info("相机标定验证 - 持续可视化模式")
    logger_.info("="*60)
    logger_.info(f"输出目录: {output_dir}")
    logger_.info(f"稳定时间: {STABILIZATION_TIME}秒")
    logger_.info(f"保存模式: {'自动保存' if SAVE_DATA else '手动保存(按S键)'}")
    logger_.info("")

    # 初始化两个相机
    logger_.info("初始化相机...")
    hand_pipeline, hand_align = initialize_camera(
        hand_cfg['serial'], "hand_camera",
        hand_cfg['resolution'], hand_cfg['fps']
    )
    top_pipeline, top_align = initialize_camera(
        top_cfg['serial'], "top_camera",
        top_cfg['resolution'], top_cfg['fps']
    )

    if hand_pipeline is None or top_pipeline is None:
        logger_.error("相机初始化失败，程序退出")
        return

    logger_.info("")

    try:
        # 保存相机内参
        logger_.info("保存相机内参...")
        save_camera_intrinsics(hand_pipeline, "hand_camera", output_dir)
        save_camera_intrinsics(top_pipeline, "top_camera", output_dir)
        logger_.info("")

        # 等待相机稳定
        wait_for_stable_camera(hand_pipeline, hand_align, "hand_camera")
        wait_for_stable_camera(top_pipeline, top_align, "top_camera")
        logger_.info("")

        # 预览相机画面
        should_capture = preview_cameras(hand_pipeline, hand_align, top_pipeline, top_align)
        if not should_capture:
            logger_.info("采集已取消")
            return

        logger_.info("")

        # 加载标定结果（传入pipeline以获取depth scale）
        logger_.info("加载标定结果...")
        calib_data = load_calibration_results(hand_pipeline=hand_pipeline, top_pipeline=top_pipeline)

        # 连接机械臂（用于获取末端位姿）
        logger_.info("")
        logger_.info("连接机械臂...")
        xarm = connect_xarm("192.168.1.236")
        if xarm is None:
            logger_.warning("⚠ 机械臂未连接，hand camera点云变换可能不准确")
        logger_.info("")

        # 采集数据（同时采集两个相机）
        show_pointcloud = calib_data is not None
        if show_pointcloud:
            logger_.info("✓ 启用点云可视化")
            if xarm is not None:
                logger_.info("✓ 使用实时gripper位姿进行hand camera变换")
            else:
                logger_.warning("⚠ 未连接机械臂，hand camera使用简化变换")
        else:
            logger_.info("✗ 未加载标定数据，禁用点云可视化")

        hand_frames, top_frames = capture_data_dual(
            hand_pipeline, hand_align,
            top_pipeline, top_align,
            output_dir,
            calib_data=calib_data,
            show_pointcloud=show_pointcloud,
            xarm=xarm,
            save_data=SAVE_DATA
        )
        logger_.info("")

        # 总结
        logger_.info("="*60)
        logger_.info("程序结束")
        if SAVE_DATA:
            logger_.info(f"数据保存在: {output_dir}")
        logger_.info("="*60)

    except KeyboardInterrupt:
        logger_.info("\n用户中断采集")

    except Exception as e:
        logger_.error(f"采集过程出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 停止相机
        if hand_pipeline:
            hand_pipeline.stop()
            logger_.info("hand_camera 已停止")
        if top_pipeline:
            top_pipeline.stop()
            logger_.info("top_camera 已停止")

        # 断开机械臂连接
        if xarm is not None:
            try:
                xarm.disconnect()
                logger_.info("机械臂已断开连接")
            except:
                pass


if __name__ == "__main__":
    main()
