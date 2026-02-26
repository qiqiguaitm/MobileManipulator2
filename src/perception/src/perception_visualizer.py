#!/usr/bin/python3
"""
Perception Visualizer - 感知结果可视化模块

从 depth_accuracy_analyzer.py 提取的可视化功能，用于 scene_perception_3d。
"""

import os
import cv2
import numpy as np
from datetime import datetime


class PerceptionVisualizer:
    """感知结果可视化器"""

    # 颜色列表 (BGR)
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
    ]

    def __init__(self, results_dir: str, transformer=None, intrinsics: dict = None):
        """
        初始化可视化器

        Args:
            results_dir: 结果保存目录
            transformer: CoordinateTransformer 实例 (用于 LiDAR 投影)
            intrinsics: 相机内参 {fx, fy, cx, cy, width, height}
        """
        self.results_dir = results_dir
        self.transformer = transformer
        self.intrinsics = intrinsics

        # 确保目录存在
        os.makedirs(results_dir, exist_ok=True)

    def save_overview(self, rgb: np.ndarray, depth: np.ndarray,
                      detections: list, object_results: list,
                      lidar_points: np.ndarray = None,
                      prompt: str = "") -> str:
        """
        保存多物体总览图 (1280x1440)

        上半部分: 检测结果 + LiDAR 投影
        下半部分: 深度图

        Args:
            rgb: BGR 图像 (H, W, 3)
            depth: 深度图 (H, W), 单位: 米
            detections: 检测结果列表, 每个包含 {object_id, category, score, bbox, mask}
            object_results: 测量结果列表, 每个包含 {camera: {distance}, lidar: {distance}}
            lidar_points: LiDAR 点云 (N, 3) 或 None
            prompt: 检测 prompt (用于文件名)

        Returns:
            保存的文件路径
        """
        h, w = rgb.shape[:2]

        # 输出图像尺寸 1280x1440 (上下各 720)
        OUT_W, OUT_H = 1280, 1440
        HALF_H = 720

        # ===== 上半部分: 检测结果 + LiDAR 投影 + Mask =====
        det_rgb = rgb.copy()

        # 1. 绘制 mask 叠加 (半透明)
        mask_count = 0
        for i, det in enumerate(detections):
            if 'mask' in det and det['mask'] is not None:
                mask = det['mask']
                if mask.sum() > 0:
                    color = self.COLORS[i % len(self.COLORS)]
                    self._draw_mask_overlay(det_rgb, mask, color, alpha=0.4)
                    mask_count += 1
        print(f"[Vis] 绘制 {mask_count} 个 mask")

        # 2. 绘制 LiDAR 投影点
        if lidar_points is not None and len(lidar_points) > 0 and self.transformer:
            self._draw_lidar_projection(det_rgb, lidar_points)

        # 3. 绘制检测框和距离标签
        for i, (det, res) in enumerate(zip(detections, object_results)):
            self._draw_detection_box(det_rgb, det, res, i)

        # 添加上半部分标题
        title = f"Detection + Mask + LiDAR | {len(detections)} objects"
        cv2.putText(det_rgb, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 缩放到 1280x720
        top_half = cv2.resize(det_rgb, (OUT_W, HALF_H))

        # ===== 下半部分: 深度图 =====
        depth_color = self._colorize_depth(depth)

        # 在深度图上绘制检测框
        for i, det in enumerate(detections):
            color = self.COLORS[i % len(self.COLORS)]
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)

        # 添加下半部分标题
        cv2.putText(depth_color, "Depth Map (0-5m)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 缩放到 1280x720
        bottom_half = cv2.resize(depth_color, (OUT_W, HALF_H))

        # ===== 合并 =====
        combined = np.vstack([top_half, bottom_half])

        # 分隔线
        cv2.line(combined, (0, HALF_H), (OUT_W, HALF_H), (255, 255, 255), 2)

        # 保存文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = prompt.replace(' ', '_').replace('.', '_')[:30] if prompt else "detect"
        filename = f"{timestamp}_{prefix}_overview.jpg"
        filepath = os.path.join(self.results_dir, filename)

        cv2.imwrite(filepath, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def save_detection_image(self, rgb: np.ndarray, detections: list,
                             object_results: list, prompt: str = "") -> str:
        """
        保存检测结果图像 (仅 bbox + mask + 测量结果)

        Args:
            rgb: BGR 图像
            detections: 检测结果列表
            object_results: 测量结果列表
            prompt: 检测 prompt

        Returns:
            保存的文件路径
        """
        det_rgb = rgb.copy()

        for i, (det, res) in enumerate(zip(detections, object_results)):
            color = self.COLORS[i % len(self.COLORS)]
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 绘制 bbox
            cv2.rectangle(det_rgb, (x1, y1), (x2, y2), color, 2)

            # 叠加 mask
            if 'mask' in det and det['mask'] is not None:
                mask_overlay = np.zeros_like(det_rgb)
                mask_overlay[:, :] = color
                mask_bool = det['mask'] > 0
                det_rgb[mask_bool] = cv2.addWeighted(
                    det_rgb[mask_bool], 0.7,
                    mask_overlay[mask_bool], 0.3, 0
                )

            # 绘制标签
            self._draw_detection_box(det_rgb, det, res, i)

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = prompt.replace(' ', '_').replace('.', '_')[:30] if prompt else "detect"
        filename = f"{timestamp}_{prefix}_detection.jpg"
        filepath = os.path.join(self.results_dir, filename)

        cv2.imwrite(filepath, det_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def save_combined(self, rgb: np.ndarray, depth: np.ndarray,
                      detections: list, object_results: list,
                      lidar_points: np.ndarray = None) -> str:
        """
        保存双拼图 (左: RGB+检测框, 右: 深度图)

        Args:
            rgb: BGR 图像
            depth: 深度图
            detections: 检测结果列表
            object_results: 测量结果列表
            lidar_points: LiDAR 点云

        Returns:
            保存的文件路径
        """
        h, w = rgb.shape[:2]

        # 左边: 检测结果 + LiDAR
        left = rgb.copy()
        if lidar_points is not None and len(lidar_points) > 0 and self.transformer:
            self._draw_lidar_projection(left, lidar_points)

        for i, (det, res) in enumerate(zip(detections, object_results)):
            color = self.COLORS[i % len(self.COLORS)]
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(left, (x1, y1), (x2, y2), color, 2)

        # 右边: 深度图
        right = self._colorize_depth(depth)

        # 添加测量结果文字
        y_offset = 30
        for det, res in zip(detections, object_results):
            cam_dist = res.get('camera', {}).get('distance')
            lidar_dist = res.get('lidar', {}).get('distance')

            if cam_dist:
                text = f"{det.get('object_id', 'obj')}: Cam {cam_dist:.3f}m"
                if lidar_dist:
                    text += f" / LiDAR {lidar_dist:.3f}m"
                cv2.putText(left, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        # 合并
        combined = np.hstack([left, right])

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_combined.jpg"
        filepath = os.path.join(self.results_dir, filename)

        cv2.imwrite(filepath, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def _draw_mask_overlay(self, img: np.ndarray, mask: np.ndarray,
                            color: tuple, alpha: float = 0.4):
        """在图像上绘制半透明 mask 叠加"""
        if mask is None or mask.sum() == 0:
            return

        # 创建彩色叠加层
        mask_bool = mask > 0
        overlay = np.zeros_like(img)
        overlay[mask_bool] = color

        # 使用 addWeighted 混合整个图像
        blended = cv2.addWeighted(img, 1.0, overlay, alpha, 0)

        # 只在 mask 区域应用混合结果
        img[mask_bool] = blended[mask_bool]

    def _draw_lidar_projection(self, img: np.ndarray, lidar_points: np.ndarray):
        """在图像上绘制 LiDAR 投影点

        与 depth_accuracy_analyzer.py 保持一致的实现
        """
        if self.transformer is None or self.intrinsics is None:
            return

        h, w = img.shape[:2]

        # 转换到光学坐标系 (与 depth_accuracy_analyzer.py:1285 一致)
        points_optical = self.transformer.rslidar_to_optical(lidar_points)

        # 投影到图像平面
        uv = self.transformer.project_to_image(points_optical, self.intrinsics)
        Z = points_optical[:, 2]

        # 只显示有效点 (Z > 0.1 确保在相机前方, Z < 10 限制最远距离)
        valid = (Z > 0.1) & (Z < 10) & \
                (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
                (uv[:, 1] >= 0) & (uv[:, 1] < h)

        valid_count = valid.sum()
        print(f"[Vis] LiDAR 投影: total={len(lidar_points)}, valid={valid_count}")

        # 绘制有效点 (近蓝远红)
        for idx in np.where(valid)[0]:
            u, v = int(uv[idx, 0]), int(uv[idx, 1])
            depth_ratio = min(Z[idx] / 5.0, 1.0)
            color = (int(255 * (1 - depth_ratio)), 0, int(255 * depth_ratio))
            cv2.circle(img, (u, v), 3, color, -1)

    def _draw_detection_box(self, img: np.ndarray, det: dict, res: dict, idx: int):
        """绘制检测框和标签"""
        color = self.COLORS[idx % len(self.COLORS)]
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 物体 ID
        obj_id = det.get('object_id', f"{det.get('category', 'obj')}_{idx}")
        if len(obj_id) > 20:
            obj_id = obj_id[:17] + "..."

        # 距离标签 (防御性检查)
        res = res or {}
        cam_info = res.get('camera') or {}
        lidar_info = res.get('lidar') or {}
        cam_dist = cam_info.get('distance')
        lidar_dist = lidar_info.get('distance')

        cam_str = f"C:{cam_dist:.2f}" if cam_dist else "C:N/A"
        lidar_str = f"L:{lidar_dist:.2f}" if lidar_dist else "L:N/A"

        label1 = obj_id
        label2 = f"{cam_str} {lidar_str}"

        # 计算标签尺寸
        (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        max_tw = max(tw1, tw2)

        # 标签背景
        cv2.rectangle(img, (x1, y1 - th1 - th2 - 10), (x1 + max_tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(img, label1, (x1 + 3, y1 - th2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, label2, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _colorize_depth(self, depth: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
        """将深度图转换为伪彩色图"""
        depth_vis = np.clip(depth, 0, max_depth) / max_depth * 255
        depth_vis = depth_vis.astype(np.uint8)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
