#!/usr/bin/env python3
"""
3D 深度性能分析工具

对比深度相机与 LiDAR 的测距精度，分析误差来源。
参考设计文档: docs/depth_accuracy_analyzer_design.md
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from pycocotools import mask as coco_mask

# 本地模块
from camera import RealSenseCamera
from ros_lidar import ROSLiDAR
from coordinate_transformer import CoordinateTransformer
from percept import DinoXDetectorOnline, DepthOptimizerOnline

# 简单的配置类 (不使用 mmengine.Config 因为接口不兼容)
class SimpleConfig:
    """简单的配置类"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MeasurementResult:
    """单次测量结果"""
    valid: bool = False
    confidence: float = 0.0
    centroid: Optional[np.ndarray] = None  # [x, y, z] in arm_base_link
    centroid_source: Optional[np.ndarray] = None  # [x, y, z] in source frame
    distance: float = 0.0  # 到 arm_base 原点的距离
    stats: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None


class DepthAccuracyAnalyzer:
    """3D 深度性能分析器"""

    # 相机配置
    CAMERA_DEVICES = {
        'top': '318122302992',
        'chassis': '337122071540',
    }

    # 算法参数 (来自设计文档 15.3)
    DEPTH_MIN = 0.3  # m
    DEPTH_MAX = 10.0  # m
    MASK_ERODE_KERNEL = 5
    IQR_FACTOR = 1.5
    LIDAR_DEPTH_TOLERANCE = 0.3  # m
    MIN_MASK_AREA = 300  # pixels
    MIN_DEPTH_POINTS = 10
    MIN_LIDAR_POINTS = 3

    def __init__(self, camera_name='top', use_depth_optimizer=True):
        """
        Args:
            camera_name: 'top' 或 'chassis'
            use_depth_optimizer: 是否使用深度优化服务
        """
        self.camera_name = camera_name
        self.use_depth_optimizer = use_depth_optimizer

        # 组件
        self.camera = None
        self.lidar = None
        self.transformer = None
        self.detector = None
        self.depth_optimizer = None

        # 相机内参 (运行时从相机获取)
        self.intrinsics = None

        # 结果保存目录
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def initialize(self):
        """初始化所有组件"""
        print("=" * 60)
        print("初始化 3D 深度性能分析器")
        print("=" * 60)

        # 1. 相机
        print(f"\n[1/5] 连接相机 ({self.camera_name})...")
        device_id = self.CAMERA_DEVICES.get(self.camera_name)
        if device_id is None:
            raise ValueError(f"未知相机: {self.camera_name}")

        camera_cfg = SimpleConfig(
            device_id=device_id,
            width=1280,
            height=720,
            fps=30,
            use_calibrated_intrinsics=False,  # 使用 RealSense 自动获取的内参
        )
        self.camera = RealSenseCamera(camera_cfg)
        self.camera.connect()

        # 获取内参 (从 rs.intrinsics 对象转换为 dict)
        rs_intr = self.camera.intrinsics
        self.intrinsics = {
            'fx': rs_intr.fx,
            'fy': rs_intr.fy,
            'cx': rs_intr.ppx,
            'cy': rs_intr.ppy,
            'width': rs_intr.width,
            'height': rs_intr.height,
        }
        print(f"  内参: fx={self.intrinsics['fx']:.1f}, fy={self.intrinsics['fy']:.1f}, "
              f"cx={self.intrinsics['cx']:.1f}, cy={self.intrinsics['cy']:.1f}")

        # 2. LiDAR (使用ROS话题，与标定数据源一致)
        print("\n[2/5] 连接 LiDAR...")
        self.lidar = ROSLiDAR(topic='/lidar/chassis/point_cloud', timeout=1.0)
        self.lidar.connect()

        # 3. 坐标变换器
        print("\n[3/5] 加载外参...")
        self.transformer = CoordinateTransformer()
        self.transformer.load_all_extrinsics()

        # 4. 检测服务
        print("\n[4/5] 初始化检测服务...")
        detector_cfg = SimpleConfig(
            url='http://192.168.112.14:10086',
            min_score=0.25,
            resize=(1280, 720),
        )
        self.detector = DinoXDetectorOnline(detector_cfg)

        # 5. 深度优化服务 (可选)
        if self.use_depth_optimizer:
            print("\n[5/5] 初始化深度优化服务...")
            optimizer_cfg = SimpleConfig(
                url='http://192.168.112.14:8086',
                chosen_policy='dn',
            )
            self.depth_optimizer = DepthOptimizerOnline(optimizer_cfg)
        else:
            print("\n[5/5] 跳过深度优化服务")

        print("\n" + "=" * 60)
        print("初始化完成")
        print("=" * 60)

    def shutdown(self):
        """关闭所有组件"""
        print("\n关闭组件...")
        if self.camera:
            try:
                if hasattr(self.camera, 'pipeline') and self.camera.pipeline:
                    self.camera.pipeline.stop()
                    print("  相机已关闭")
            except Exception as e:
                print(f"  相机关闭异常: {e}")
        if self.lidar:
            self.lidar.disconnect()

    def capture_synchronized(self):
        """
        同步采集相机和 LiDAR 数据

        Returns:
            rgb: (H, W, 3) BGR 图像
            depth: (H, W) 深度图，单位 m
            lidar_points: (N, 4) LiDAR 点云 [x, y, z, intensity]
            timestamp: 采集时间戳
        """
        # 采集相机数据
        bundle = self.camera.get_image_bundle()
        rgb = bundle['rgb']
        depth = bundle['depth']  # 已对齐的深度图

        # 采集 LiDAR 数据
        lidar_points = self.lidar.get_one_frame()

        timestamp = time.time()

        return rgb, depth, lidar_points, timestamp

    def detect_target(self, prompt: str, rgb: np.ndarray):
        """
        检测目标物体

        Args:
            prompt: 检测提示词，如 "box"
            rgb: BGR 图像

        Returns:
            dict: {'bbox': [x1,y1,x2,y2], 'mask': ndarray, 'score': float, 'category': str}
            或 None 如果未检测到
        """
        result = self.detector.forward(text=prompt, rgb=rgb)
        objects = result.result.get('objects', [])

        if not objects:
            return None

        # 选择置信度最高的目标
        best_obj = max(objects, key=lambda x: x.get('score', 0))

        # 解码 mask
        mask_rle = best_obj.get('mask')
        if mask_rle:
            binary_mask = coco_mask.decode(mask_rle).astype(np.uint8)
        else:
            # 如果没有 mask，使用 bbox 生成矩形 mask
            bbox = best_obj['bbox']
            binary_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            binary_mask[y1:y2, x1:x2] = 1

        # 检查 mask 面积
        mask_area = binary_mask.sum()
        if mask_area < self.MIN_MASK_AREA:
            print(f"[WARN] Mask 面积太小: {mask_area} < {self.MIN_MASK_AREA}")
            return None

        return {
            'bbox': best_obj['bbox'],
            'mask': binary_mask,
            'score': best_obj.get('score', 0),
            'category': best_obj.get('category', prompt),
        }

    def detect_all_targets(self, prompt: str, rgb: np.ndarray) -> List[dict]:
        """
        检测所有目标物体

        Args:
            prompt: 多物体提示词，如 "toy rubic.bottle.cup"
            rgb: BGR 图像

        Returns:
            List[dict]: 每个物体的 {bbox, mask, score, category}
        """
        result = self.detector.forward(text=prompt, rgb=rgb)
        objects = result.result.get('objects', [])

        detections = []
        for obj in objects:
            mask_rle = obj.get('mask')
            if mask_rle:
                binary_mask = coco_mask.decode(mask_rle).astype(np.uint8)
            else:
                bbox = obj['bbox']
                binary_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, bbox)
                binary_mask[y1:y2, x1:x2] = 1

            if binary_mask.sum() >= self.MIN_MASK_AREA:
                detections.append({
                    'bbox': obj['bbox'],
                    'mask': binary_mask,
                    'score': obj.get('score', 0),
                    'category': obj.get('category', ''),
                })

        return detections

    def _assign_object_ids(self, detections: List[dict]) -> List[dict]:
        """
        为检测结果分配唯一 ID，同类物体按置信度排序编号

        Args:
            detections: detect_all_targets 返回的检测列表

        Returns:
            List[dict]: 添加了 obj_id 和 index 的检测列表
        """
        # 按类别分组
        category_groups = {}
        for det in detections:
            cat = det['category']
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(det)

        # 每个类别内按置信度排序，分配编号
        result = []
        for cat, items in category_groups.items():
            # 按置信度降序排序
            items.sort(key=lambda x: x['score'], reverse=True)
            for idx, det in enumerate(items, start=1):
                det['index'] = idx
                det['obj_id'] = f"{cat}_{idx}"
                result.append(det)

        # 按原始置信度排序输出
        result.sort(key=lambda x: x['score'], reverse=True)
        return result

    def optimize_depth(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        使用 CDM 服务优化深度图

        Args:
            rgb: BGR 图像
            depth: 深度图 (m, float32)

        Returns:
            optimized_depth: 优化后的深度图 (m, float32)
        """
        if self.depth_optimizer is None:
            return depth

        # 转换为 mm uint16 (服务要求的格式)
        depth_mm = (depth * 1000).astype(np.uint16)

        result = self.depth_optimizer.forward(rgb, depth_mm)

        if result.get('success') and 'depth' in result:
            # 转回 m float32
            optimized = result['depth'].astype(np.float32) / 1000.0
            return optimized
        else:
            print(f"[WARN] 深度优化失败: {result.get('error', 'unknown')}")
            return depth

    def measure_camera(self, depth: np.ndarray, mask: np.ndarray) -> MeasurementResult:
        """
        Step 4A: 深度相机测距 (基于 Mask)

        Args:
            depth: 深度图 (m)
            mask: 二值 mask

        Returns:
            MeasurementResult
        """
        result = MeasurementResult()

        # 1. Mask 腐蚀 (去除边缘噪声)
        kernel = np.ones((self.MASK_ERODE_KERNEL, self.MASK_ERODE_KERNEL), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        # 检查腐蚀后的 mask
        if eroded_mask.sum() < self.MIN_DEPTH_POINTS:
            result.error_msg = f"腐蚀后 mask 太小: {eroded_mask.sum()}"
            return result

        # 2. 提取深度值
        depth_values = depth[eroded_mask > 0]

        # 3. 剔除无效值
        valid_mask = (depth_values > self.DEPTH_MIN) & (depth_values < self.DEPTH_MAX)
        depth_values = depth_values[valid_mask]

        if len(depth_values) < self.MIN_DEPTH_POINTS:
            result.error_msg = f"有效深度点太少: {len(depth_values)}"
            return result

        # 4. IQR 异常值剔除
        q1, q3 = np.percentile(depth_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.IQR_FACTOR * iqr
        upper_bound = q3 + self.IQR_FACTOR * iqr
        depth_values = depth_values[(depth_values >= lower_bound) & (depth_values <= upper_bound)]

        if len(depth_values) < self.MIN_DEPTH_POINTS:
            result.error_msg = f"IQR 后深度点太少: {len(depth_values)}"
            return result

        # 5. 生成 3D 点云
        ys, xs = np.where(eroded_mask > 0)
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']

        points = []
        for y, x in zip(ys, xs):
            d = depth[y, x]
            if lower_bound <= d <= upper_bound and self.DEPTH_MIN < d < self.DEPTH_MAX:
                X = (x - cx) * d / fx
                Y = (y - cy) * d / fy
                Z = d
                points.append([X, Y, Z])

        if len(points) < self.MIN_DEPTH_POINTS:
            result.error_msg = f"3D 点太少: {len(points)}"
            return result

        points = np.array(points)

        # 6. 计算质心 (使用中值更鲁棒)
        centroid_optical = np.median(points, axis=0)

        # 7. 变换到 arm_base_link
        centroid_arm = self.transformer.optical_to_arm(centroid_optical.reshape(1, 3))[0]

        # 计算到原点距离
        distance = np.linalg.norm(centroid_arm)

        # 计算置信度
        depth_std = np.std(depth_values)
        confidence = self._compute_confidence(len(points), depth_std)

        result.valid = True
        result.confidence = confidence
        result.centroid = centroid_arm
        result.centroid_source = centroid_optical
        result.distance = distance
        result.stats = {
            'depth_median': float(np.median(depth_values)),
            'depth_std': float(depth_std),
            'num_points': len(points),
            'valid_ratio': len(depth_values) / max(mask.sum(), 1),
        }

        return result

    def measure_lidar_guided(self, lidar_points: np.ndarray, bbox: List[float],
                             camera_depth: float) -> MeasurementResult:
        """
        Step 4B: LiDAR 相机引导模式

        Args:
            lidar_points: (N, 4) rslidar 坐标系点云
            bbox: [x1, y1, x2, y2] 检测框
            camera_depth: 相机测得的深度 (m)

        Returns:
            MeasurementResult
        """
        result = MeasurementResult()

        if len(lidar_points) == 0:
            result.error_msg = "无 LiDAR 点云"
            return result

        # 1. 变换到 optical frame
        points_optical = self.transformer.rslidar_to_optical(lidar_points)

        # 2. 投影到图像平面
        uv = self.transformer.project_to_image(points_optical, self.intrinsics)

        # 3. bbox + 深度范围筛选 (支持逐步扩大范围)
        x1, y1, x2, y2 = bbox
        depth_min = camera_depth - self.LIDAR_DEPTH_TOLERANCE
        depth_max = camera_depth + self.LIDAR_DEPTH_TOLERANCE

        Z = points_optical[:, 2]  # optical frame 中 Z 是深度
        in_depth = (Z >= depth_min) & (Z <= depth_max) & (Z > 0.1)

        # 逐步扩大 bbox 范围直到找到足够的点 (按比例扩展，无上限)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        expand_ratio = 0.0
        expand_step_ratio = 0.2  # 每次扩展 20%
        bbox_expanded = False
        img_w, img_h = self.intrinsics['width'], self.intrinsics['height']

        while True:
            expand_w = bbox_w * expand_ratio / 2  # 每边扩展一半
            expand_h = bbox_h * expand_ratio / 2
            ex1 = max(0, x1 - expand_w)
            ey1 = max(0, y1 - expand_h)
            ex2 = min(img_w, x2 + expand_w)
            ey2 = min(img_h, y2 + expand_h)

            in_bbox = (uv[:, 0] >= ex1) & (uv[:, 0] <= ex2) & \
                      (uv[:, 1] >= ey1) & (uv[:, 1] <= ey2)

            mask = in_bbox & in_depth
            filtered_indices = np.where(mask)[0]

            if len(filtered_indices) >= self.MIN_LIDAR_POINTS:
                if expand_ratio > 0:
                    bbox_expanded = True
                    print(f"    [INFO] bbox 扩展 {expand_ratio*100:.0f}% 后找到 {len(filtered_indices)} 个点")
                break

            # 检查是否已扩展到图像边界
            if ex1 <= 0 and ey1 <= 0 and ex2 >= img_w and ey2 >= img_h:
                break  # 已扩展到整个图像，无法继续

            expand_ratio += expand_step_ratio

        if len(filtered_indices) < self.MIN_LIDAR_POINTS:
            result.error_msg = f"筛选后 LiDAR 点太少: {len(filtered_indices)} (已扩展至图像边界)"
            return result

        # 获取筛选后的点 (rslidar 坐标系)
        filtered_points = lidar_points[filtered_indices, :3]

        # 4. IQR 异常值剔除 (基于到原点距离)
        distances = np.linalg.norm(filtered_points, axis=1)
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        valid_mask = (distances >= q1 - self.IQR_FACTOR * iqr) & \
                     (distances <= q3 + self.IQR_FACTOR * iqr)
        filtered_points = filtered_points[valid_mask]

        if len(filtered_points) < self.MIN_LIDAR_POINTS:
            result.error_msg = f"IQR 后 LiDAR 点太少: {len(filtered_points)}"
            return result

        # 5. 计算质心
        centroid_rslidar = np.median(filtered_points, axis=0)

        # 6. 变换到 arm_base_link (使用链式变换确保与相机测量一致)
        # rslidar → optical → arm (不使用直接的 rslidar → arm，因为 URDF 值可能有偏差)
        centroid_optical = self.transformer.rslidar_to_optical(centroid_rslidar.reshape(1, 3))
        centroid_arm = self.transformer.optical_to_arm(centroid_optical)[0]

        distance = np.linalg.norm(centroid_arm)

        # 计算置信度
        depth_std = np.std(np.linalg.norm(filtered_points, axis=1))
        confidence = self._compute_confidence(len(filtered_points), depth_std)

        result.valid = True
        result.confidence = confidence
        result.centroid = centroid_arm
        result.centroid_source = centroid_rslidar
        result.distance = distance
        result.stats = {
            'depth_median': float(np.median(distances[valid_mask])),
            'depth_std': float(depth_std),
            'num_points': len(filtered_points),
            'depth_range': [float(depth_min), float(depth_max)],
            'bbox_expand_ratio': expand_ratio if bbox_expanded else 0,
        }

        return result

    def measure_lidar_independent(self, lidar_points: np.ndarray,
                                   bbox: List[float]) -> MeasurementResult:
        """
        Step 4C: LiDAR 独立模式 (DBSCAN 聚类)

        Args:
            lidar_points: (N, 4) rslidar 坐标系点云
            bbox: [x1, y1, x2, y2] 检测框

        Returns:
            MeasurementResult
        """
        result = MeasurementResult()

        if len(lidar_points) == 0:
            result.error_msg = "无 LiDAR 点云"
            return result

        # 1. 变换到 optical frame
        points_optical = self.transformer.rslidar_to_optical(lidar_points)

        # 2. 投影到图像平面
        uv = self.transformer.project_to_image(points_optical, self.intrinsics)

        # 3. bbox 筛选 (不限制深度，支持逐步扩大范围)
        x1, y1, x2, y2 = bbox
        Z = points_optical[:, 2]
        in_range = (Z > 0.3) & (Z < 15.0)  # 基础深度范围

        # 逐步扩大 bbox 范围直到找到足够的点 (按比例扩展，无上限)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        expand_ratio = 0.0
        expand_step_ratio = 0.2  # 每次扩展 20%
        bbox_expanded = False
        img_w, img_h = self.intrinsics['width'], self.intrinsics['height']

        while True:
            expand_w = bbox_w * expand_ratio / 2  # 每边扩展一半
            expand_h = bbox_h * expand_ratio / 2
            ex1 = max(0, x1 - expand_w)
            ey1 = max(0, y1 - expand_h)
            ex2 = min(img_w, x2 + expand_w)
            ey2 = min(img_h, y2 + expand_h)

            in_bbox = (uv[:, 0] >= ex1) & (uv[:, 0] <= ex2) & \
                      (uv[:, 1] >= ey1) & (uv[:, 1] <= ey2)

            mask = in_bbox & in_range
            filtered_indices = np.where(mask)[0]

            if len(filtered_indices) >= self.MIN_LIDAR_POINTS:
                if expand_ratio > 0:
                    bbox_expanded = True
                    print(f"    [INFO] bbox 扩展 {expand_ratio*100:.0f}% 后找到 {len(filtered_indices)} 个点")
                break

            # 检查是否已扩展到图像边界
            if ex1 <= 0 and ey1 <= 0 and ex2 >= img_w and ey2 >= img_h:
                break  # 已扩展到整个图像，无法继续

            expand_ratio += expand_step_ratio

        if len(filtered_indices) < self.MIN_LIDAR_POINTS:
            result.error_msg = f"bbox 内 LiDAR 点太少: {len(filtered_indices)} (已扩展至图像边界)"
            return result

        # 获取筛选后的点 (rslidar 坐标系)
        filtered_points = lidar_points[filtered_indices, :3]

        # 4. DBSCAN 聚类
        # 自适应 eps
        distances = np.linalg.norm(filtered_points, axis=1)
        median_distance = np.median(distances)
        eps = self._compute_adaptive_eps(median_distance)

        clustering = DBSCAN(eps=eps, min_samples=self.MIN_LIDAR_POINTS).fit(filtered_points)
        labels = clustering.labels_

        # 找出所有有效聚类
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声标签

        if len(unique_labels) == 0:
            result.error_msg = "DBSCAN 未找到有效聚类"
            return result

        # 5. 选择最近的聚类 (前景目标)
        best_cluster = None
        best_depth = float('inf')
        for label in unique_labels:
            cluster_points = filtered_points[labels == label]
            cluster_depth = np.median(np.linalg.norm(cluster_points, axis=1))
            if cluster_depth < best_depth:
                best_depth = cluster_depth
                best_cluster = cluster_points

        if best_cluster is None or len(best_cluster) < self.MIN_LIDAR_POINTS:
            result.error_msg = f"最近聚类点数不足: {len(best_cluster) if best_cluster is not None else 0}"
            return result

        # 6. 计算质心
        centroid_rslidar = np.median(best_cluster, axis=0)

        # 7. 变换到 arm_base_link (使用链式变换确保与相机测量一致)
        # rslidar → optical → arm (不使用直接的 rslidar → arm，因为 URDF 值可能有偏差)
        centroid_optical = self.transformer.rslidar_to_optical(centroid_rslidar.reshape(1, 3))
        centroid_arm = self.transformer.optical_to_arm(centroid_optical)[0]

        distance = np.linalg.norm(centroid_arm)

        # 计算置信度
        cluster_distances = np.linalg.norm(best_cluster, axis=1)
        depth_std = np.std(cluster_distances)
        confidence = self._compute_confidence(len(best_cluster), depth_std)

        result.valid = True
        result.confidence = confidence
        result.centroid = centroid_arm
        result.centroid_source = centroid_rslidar
        result.distance = distance
        result.stats = {
            'depth_median': float(np.median(cluster_distances)),
            'depth_std': float(depth_std),
            'num_points': len(best_cluster),
            'num_clusters': len(unique_labels),
            'eps_used': float(eps),
            'bbox_expand_ratio': expand_ratio if bbox_expanded else 0,
        }

        return result

    def _compute_confidence(self, num_points: int, depth_std: float,
                            max_std: float = 0.3) -> float:
        """计算测量置信度"""
        point_score = min(num_points / 100.0, 1.0)
        std_score = max(0, 1.0 - depth_std / max_std)
        return 0.7 * point_score + 0.3 * std_score

    def _compute_adaptive_eps(self, estimated_depth: float) -> float:
        """计算自适应 DBSCAN eps"""
        # LiDAR 水平角度分辨率约 0.2°
        point_spacing = estimated_depth * np.tan(np.deg2rad(0.2))
        eps = np.clip(3 * point_spacing, 0.05, 0.3)
        return eps

    def analyze_single(self, prompt: str, sample_id: int = 0):
        """
        执行单次分析

        Args:
            prompt: 检测提示词
            sample_id: 样本 ID (用于保存文件)

        Returns:
            dict: 包含三种测量结果和元信息
        """
        print(f"\n{'='*60}")
        print(f"样本 {sample_id + 1}: 开始分析")
        print(f"{'='*60}")

        result = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'camera': None,
            'lidar_guided': None,
            'lidar_independent': None,
            'detection': None,
            'errors': [],
        }

        # 1. 同步采集
        print("\n[Step 1] 同步采集数据...")
        rgb, depth_raw, lidar_points, timestamp = self.capture_synchronized()
        print(f"  RGB: {rgb.shape}, Depth: {depth_raw.shape}, LiDAR: {len(lidar_points)} points")

        # 2. 深度优化
        if self.use_depth_optimizer:
            print("\n[Step 2] 优化深度图...")
            depth = self.optimize_depth(rgb, depth_raw)
        else:
            depth = depth_raw

        # 3. 目标检测
        print(f"\n[Step 3] 检测目标: '{prompt}'...")
        detection = self.detect_target(prompt, rgb)
        if detection is None:
            result['errors'].append(f"未检测到目标: {prompt}")
            print(f"  [ERROR] 未检测到目标")
            return result

        result['detection'] = {
            'bbox': detection['bbox'],
            'score': detection['score'],
            'category': detection['category'],
            'mask_area': int(detection['mask'].sum()),
        }
        print(f"  检测到: {detection['category']}, score={detection['score']:.3f}")
        print(f"  bbox: {[int(x) for x in detection['bbox']]}")

        # 4A. 深度相机测量
        print("\n[Step 4A] 深度相机测量...")
        camera_result = self.measure_camera(depth, detection['mask'])
        if camera_result.valid:
            result['camera'] = self._result_to_dict(camera_result)
            print(f"  质心 (arm_base): [{camera_result.centroid[0]:.3f}, "
                  f"{camera_result.centroid[1]:.3f}, {camera_result.centroid[2]:.3f}] m")
            print(f"  距离: {camera_result.distance:.3f} m")
        else:
            result['errors'].append(f"相机测量失败: {camera_result.error_msg}")
            print(f"  [ERROR] {camera_result.error_msg}")
            return result  # 相机测量是必须的

        # 4B. LiDAR 相机引导模式
        print("\n[Step 4B] LiDAR 相机引导模式...")
        camera_depth = camera_result.stats['depth_median']
        lidar_guided_result = self.measure_lidar_guided(
            lidar_points, detection['bbox'], camera_depth
        )
        if lidar_guided_result.valid:
            result['lidar_guided'] = self._result_to_dict(lidar_guided_result)
            print(f"  质心 (arm_base): [{lidar_guided_result.centroid[0]:.3f}, "
                  f"{lidar_guided_result.centroid[1]:.3f}, {lidar_guided_result.centroid[2]:.3f}] m")
            print(f"  距离: {lidar_guided_result.distance:.3f} m")
        else:
            result['errors'].append(f"LiDAR 引导模式失败: {lidar_guided_result.error_msg}")
            print(f"  [WARN] {lidar_guided_result.error_msg}")

        # 4C. LiDAR 独立模式
        print("\n[Step 4C] LiDAR 独立模式 (DBSCAN)...")
        lidar_indep_result = self.measure_lidar_independent(lidar_points, detection['bbox'])
        if lidar_indep_result.valid:
            result['lidar_independent'] = self._result_to_dict(lidar_indep_result)
            print(f"  质心 (arm_base): [{lidar_indep_result.centroid[0]:.3f}, "
                  f"{lidar_indep_result.centroid[1]:.3f}, {lidar_indep_result.centroid[2]:.3f}] m")
            print(f"  距离: {lidar_indep_result.distance:.3f} m")
            print(f"  聚类数: {lidar_indep_result.stats.get('num_clusters', 'N/A')}")
        else:
            result['errors'].append(f"LiDAR 独立模式失败: {lidar_indep_result.error_msg}")
            print(f"  [WARN] {lidar_indep_result.error_msg}")

        # 保存可视化
        self._save_visualization(rgb, depth, detection, lidar_points,
                                 camera_result, lidar_guided_result, lidar_indep_result,
                                 sample_id, prompt)

        return result

    def _result_to_dict(self, result: MeasurementResult) -> dict:
        """将 MeasurementResult 转换为可序列化的 dict"""
        return {
            'valid': result.valid,
            'confidence': result.confidence,
            'centroid': result.centroid.tolist() if result.centroid is not None else None,
            'centroid_source': result.centroid_source.tolist() if result.centroid_source is not None else None,
            'distance': result.distance,
            'stats': result.stats,
        }

    def _save_visualization(self, rgb, depth, detection, lidar_points,
                           camera_result, lidar_guided_result, lidar_indep_result,
                           sample_id, prompt: str):
        """保存可视化图像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用 prompt 作为文件名前缀，替换特殊字符
        prefix = prompt.replace(' ', '_').replace('/', '_')[:30]

        print(f"\n保存图像 (s{sample_id}_{prefix})...")

        # 1. 保存原始 RGB 图像
        raw_path = os.path.join(self.results_dir, f"s{sample_id}_{prefix}_1_raw.jpg")
        cv2.imwrite(raw_path, rgb)
        print(f"  原始图像: {raw_path}")

        # 2. 保存检测结果图像 (bbox + mask)
        det_rgb = rgb.copy()
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(det_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 叠加 mask
        mask_overlay = np.zeros_like(det_rgb)
        mask_overlay[:, :, 1] = detection['mask'] * 100  # 绿色
        det_rgb = cv2.addWeighted(det_rgb, 1.0, mask_overlay, 0.3, 0)

        # 添加检测信息
        cv2.putText(det_rgb, f"{detection['category']}: {detection['score']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        det_path = os.path.join(self.results_dir, f"s{sample_id}_{prefix}_2_detection.jpg")
        cv2.imwrite(det_path, det_rgb)
        print(f"  检测结果: {det_path}")

        # 3. 保存 LiDAR 投影图像
        lidar_rgb = rgb.copy()
        if len(lidar_points) > 0:
            points_optical = self.transformer.rslidar_to_optical(lidar_points)
            uv = self.transformer.project_to_image(points_optical, self.intrinsics)
            Z = points_optical[:, 2]

            # 只显示前方的点
            valid = (Z > 0.1) & (Z < 10) & \
                    (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])

            for i in np.where(valid)[0]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                # 根据深度着色
                depth_ratio = min(Z[i] / 5.0, 1.0)
                color = (int(255 * (1 - depth_ratio)), 0, int(255 * depth_ratio))
                cv2.circle(lidar_rgb, (u, v), 2, color, -1)

        # 添加检测框
        cv2.rectangle(lidar_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lidar_path = os.path.join(self.results_dir, f"s{sample_id}_{prefix}_3_lidar_projection.jpg")
        cv2.imwrite(lidar_path, lidar_rgb)
        print(f"  LiDAR投影: {lidar_path}")

        # 4. 保存深度伪彩色图
        depth_vis = depth.copy()
        depth_vis = np.clip(depth_vis, 0, 5) / 5 * 255
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        depth_path = os.path.join(self.results_dir, f"s{sample_id}_{prefix}_4_depth.jpg")
        cv2.imwrite(depth_path, depth_color)
        print(f"  深度图像: {depth_path}")

        # 5. 保存综合分析图 (检测 + LiDAR + 深度 + 测量结果)
        h, w = rgb.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = lidar_rgb  # 左: LiDAR 投影 + 检测框
        combined[:, w:] = depth_color  # 右: 深度图

        # 添加测量结果文字
        y_offset = 30
        if camera_result.valid:
            text = f"Camera: {camera_result.distance:.3f}m"
            cv2.putText(combined, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        if lidar_guided_result.valid:
            text = f"LiDAR Guided: {lidar_guided_result.distance:.3f}m"
            cv2.putText(combined, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        if lidar_indep_result.valid:
            text = f"LiDAR Indep: {lidar_indep_result.distance:.3f}m"
            cv2.putText(combined, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        combined_path = os.path.join(self.results_dir, f"s{sample_id}_{prefix}_5_combined.jpg")
        cv2.imwrite(combined_path, combined)
        print(f"  综合分析: {combined_path}")

    def analyze(self, prompt: str, num_samples: int = 3, interval: float = 1.5):
        """
        执行多次采样分析

        Args:
            prompt: 检测提示词
            num_samples: 采样次数
            interval: 采样间隔 (秒)

        Returns:
            dict: 汇总分析结果
        """
        all_results = []

        for i in range(num_samples):
            if i > 0:
                print(f"\n等待 {interval} 秒...")
                time.sleep(interval)

            result = self.analyze_single(prompt, sample_id=i)
            all_results.append(result)

        # 汇总统计
        summary = self._compute_summary(all_results)

        # 生成报告
        self._generate_report(all_results, summary, prompt)

        return {'samples': all_results, 'summary': summary}

    def _compute_summary(self, results: List[dict]) -> dict:
        """计算汇总统计"""
        camera_distances = []
        lidar_guided_distances = []
        lidar_indep_distances = []

        for r in results:
            if r.get('camera') and r['camera']['valid']:
                camera_distances.append(r['camera']['distance'])
            if r.get('lidar_guided') and r['lidar_guided']['valid']:
                lidar_guided_distances.append(r['lidar_guided']['distance'])
            if r.get('lidar_independent') and r['lidar_independent']['valid']:
                lidar_indep_distances.append(r['lidar_independent']['distance'])

        summary = {
            'num_samples': len(results),
            'camera': self._stats(camera_distances) if camera_distances else None,
            'lidar_guided': self._stats(lidar_guided_distances) if lidar_guided_distances else None,
            'lidar_independent': self._stats(lidar_indep_distances) if lidar_indep_distances else None,
        }

        # 计算误差
        if camera_distances and lidar_guided_distances:
            cam_mean = np.mean(camera_distances)
            lidar_guided_mean = np.mean(lidar_guided_distances)
            summary['error_cam_vs_guided'] = {
                'absolute': float(abs(cam_mean - lidar_guided_mean)),
                'relative': float(abs(cam_mean - lidar_guided_mean) / cam_mean * 100),
            }

        if camera_distances and lidar_indep_distances:
            cam_mean = np.mean(camera_distances)
            lidar_indep_mean = np.mean(lidar_indep_distances)
            summary['error_cam_vs_indep'] = {
                'absolute': float(abs(cam_mean - lidar_indep_mean)),
                'relative': float(abs(cam_mean - lidar_indep_mean) / cam_mean * 100),
            }

        return summary

    def _stats(self, values: List[float]) -> dict:
        """计算基本统计量"""
        arr = np.array(values)
        return {
            'count': len(arr),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
        }

    def _generate_report(self, results: List[dict], summary: dict, prompt: str):
        """生成分析报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 报告
        json_file = os.path.join(self.results_dir, f"depth_analysis_{timestamp}.json")
        report_data = {
            'timestamp': timestamp,
            'prompt': prompt,
            'camera': self.camera_name,
            'samples': results,
            'summary': summary,
        }
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 报告: {json_file}")

        # 文本报告
        txt_file = os.path.join(self.results_dir, f"depth_analysis_{timestamp}.txt")
        self._write_text_report(txt_file, results, summary, prompt, timestamp)
        print(f"文本报告: {txt_file}")

    def _write_text_report(self, filepath: str, results: List[dict],
                           summary: dict, prompt: str, timestamp: str):
        """写入文本格式报告"""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("                     3D 深度性能分析报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"相机: {self.camera_name}\n")
            f.write(f"检测目标: {prompt}\n")
            f.write(f"采样次数: {summary['num_samples']}\n")
            f.write("\n")

            # 汇总统计
            f.write("-" * 80 + "\n")
            f.write("【汇总统计】\n")
            f.write("-" * 80 + "\n")

            if summary.get('camera'):
                s = summary['camera']
                f.write(f"深度相机: {s['mean']:.4f} ± {s['std']:.4f} m "
                        f"(n={s['count']})\n")

            if summary.get('lidar_guided'):
                s = summary['lidar_guided']
                f.write(f"LiDAR 引导: {s['mean']:.4f} ± {s['std']:.4f} m "
                        f"(n={s['count']})\n")

            if summary.get('lidar_independent'):
                s = summary['lidar_independent']
                f.write(f"LiDAR 独立: {s['mean']:.4f} ± {s['std']:.4f} m "
                        f"(n={s['count']})\n")

            f.write("\n")

            # 误差分析
            f.write("-" * 80 + "\n")
            f.write("【误差分析】\n")
            f.write("-" * 80 + "\n")

            if summary.get('error_cam_vs_guided'):
                e = summary['error_cam_vs_guided']
                f.write(f"相机 vs LiDAR引导: {e['absolute']*1000:.1f} mm "
                        f"({e['relative']:.2f}%)\n")

            if summary.get('error_cam_vs_indep'):
                e = summary['error_cam_vs_indep']
                f.write(f"相机 vs LiDAR独立: {e['absolute']*1000:.1f} mm "
                        f"({e['relative']:.2f}%)\n")

            f.write("\n")
            f.write("=" * 80 + "\n")

    # ========== 多物体分析方法 ==========

    def analyze_multi(self, prompt: str, sample_id: int = 0) -> dict:
        """
        多物体深度分析

        Args:
            prompt: 多物体提示词，如 "toy rubic.bottle.cup"
            sample_id: 样本 ID

        Returns:
            dict: 包含所有物体的测量结果和统计
        """
        print(f"\n{'='*70}")
        print(f"多物体深度分析 | 样本 {sample_id}")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")

        result = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'objects': [],
            'summary': None,
            'errors': [],
        }

        # 1. 同步采集 (一次)
        print("\n[Step 1] 同步采集数据...")
        rgb, depth_raw, lidar_points, timestamp = self.capture_synchronized()
        print(f"  RGB: {rgb.shape}, Depth: {depth_raw.shape}, LiDAR: {len(lidar_points)} points")

        # 2. 深度优化 (一次)
        if self.use_depth_optimizer:
            print("\n[Step 2] 优化深度图...")
            depth = self.optimize_depth(rgb, depth_raw)
        else:
            depth = depth_raw

        # 3. 检测所有物体 (一次调用)
        print(f"\n[Step 3] 检测所有物体...")
        detections = self.detect_all_targets(prompt, rgb)

        if not detections:
            result['errors'].append(f"未检测到任何物体")
            print(f"  [ERROR] 未检测到任何物体")
            return result

        # 4. 为同类物体分配编号
        detections = self._assign_object_ids(detections)
        print(f"  检测到 {len(detections)} 个物体:")
        for det in detections:
            print(f"    - {det['obj_id']}: score={det['score']:.3f}, "
                  f"bbox={[int(x) for x in det['bbox']]}")

        # 5. 对每个物体执行三种测量
        print(f"\n[Step 4] 对每个物体执行测量...")
        object_results = []
        for i, det in enumerate(detections):
            print(f"\n  --- {det['obj_id']} ({i+1}/{len(detections)}) ---")
            obj_result = self._measure_single_object(det, depth, lidar_points)
            object_results.append(obj_result)

            # 保存该物体的中间结果
            self._save_object_visualization(rgb, depth, det, lidar_points,
                                            obj_result, sample_id)

        result['objects'] = object_results

        # 6. 保存总览图
        self._save_multi_overview(rgb, depth, detections, object_results,
                                  lidar_points, sample_id, prompt)

        # 7. 计算统计汇总
        result['summary'] = self._compute_multi_summary(object_results)

        # 8. 打印结果表格
        self._print_multi_results(object_results, result['summary'])

        # 9. 保存 JSON 结果
        self._save_multi_json(result, sample_id, prompt)

        return result

    def _measure_single_object(self, detection: dict, depth: np.ndarray,
                                lidar_points: np.ndarray) -> dict:
        """
        对单个物体执行三种测量方法

        Args:
            detection: 包含 obj_id, bbox, mask, score, category 的检测结果
            depth: 深度图
            lidar_points: LiDAR 点云

        Returns:
            dict: 包含三种测量结果和误差
        """
        obj_id = detection['obj_id']
        result = {
            'obj_id': obj_id,
            'category': detection['category'],
            'index': detection['index'],
            'bbox': detection['bbox'],
            'score': detection['score'],
            'camera': None,
            'lidar_guided': None,
            'lidar_independent': None,
            'error_guided': None,
            'error_independent': None,
        }

        # 4A. 深度相机测量
        camera_result = self.measure_camera(depth, detection['mask'])
        if camera_result.valid:
            result['camera'] = self._result_to_dict(camera_result)
            print(f"    相机: {camera_result.distance:.3f}m")
        else:
            print(f"    相机: [FAIL] {camera_result.error_msg}")
            return result  # 相机测量失败则跳过后续

        # 4B. LiDAR 相机引导模式
        camera_depth = camera_result.stats['depth_median']
        lidar_guided_result = self.measure_lidar_guided(
            lidar_points, detection['bbox'], camera_depth
        )
        if lidar_guided_result.valid:
            result['lidar_guided'] = self._result_to_dict(lidar_guided_result)
            result['error_guided'] = lidar_guided_result.distance - camera_result.distance
            print(f"    LiDAR引导: {lidar_guided_result.distance:.3f}m "
                  f"(Δ={result['error_guided']*1000:+.1f}mm)")
        else:
            print(f"    LiDAR引导: [FAIL] {lidar_guided_result.error_msg}")

        # 4C. LiDAR 独立模式
        lidar_indep_result = self.measure_lidar_independent(lidar_points, detection['bbox'])
        if lidar_indep_result.valid:
            result['lidar_independent'] = self._result_to_dict(lidar_indep_result)
            result['error_independent'] = lidar_indep_result.distance - camera_result.distance
            print(f"    LiDAR独立: {lidar_indep_result.distance:.3f}m "
                  f"(Δ={result['error_independent']*1000:+.1f}mm)")
        else:
            print(f"    LiDAR独立: [FAIL] {lidar_indep_result.error_msg}")

        return result

    def _save_object_visualization(self, rgb, depth, detection, lidar_points,
                                    obj_result, sample_id):
        """保存单个物体的可视化图像"""
        obj_id = detection['obj_id']
        # 替换空格为下划线
        obj_id_safe = obj_id.replace(' ', '_')

        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # 1. 检测结果图像
        det_rgb = rgb.copy()
        cv2.rectangle(det_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 叠加 mask
        mask_overlay = np.zeros_like(det_rgb)
        mask_overlay[:, :, 1] = detection['mask'] * 100
        det_rgb = cv2.addWeighted(det_rgb, 1.0, mask_overlay, 0.3, 0)

        # 添加信息
        label = f"{obj_id}: {detection['score']:.2f}"
        cv2.putText(det_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 添加测量结果
        y_offset = y2 + 20
        if obj_result.get('camera'):
            text = f"Cam: {obj_result['camera']['distance']:.3f}m"
            cv2.putText(det_rgb, text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 18
        if obj_result.get('lidar_guided'):
            text = f"LiDAR-G: {obj_result['lidar_guided']['distance']:.3f}m"
            cv2.putText(det_rgb, text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 18
        if obj_result.get('lidar_independent'):
            text = f"LiDAR-I: {obj_result['lidar_independent']['distance']:.3f}m"
            cv2.putText(det_rgb, text, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        det_path = os.path.join(self.results_dir, f"s{sample_id}_{obj_id_safe}_detection.jpg")
        cv2.imwrite(det_path, det_rgb)

        # 2. LiDAR 投影图像
        lidar_rgb = rgb.copy()
        if len(lidar_points) > 0:
            points_optical = self.transformer.rslidar_to_optical(lidar_points)
            uv = self.transformer.project_to_image(points_optical, self.intrinsics)
            Z = points_optical[:, 2]

            valid = (Z > 0.1) & (Z < 10) & \
                    (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])

            for i in np.where(valid)[0]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                depth_ratio = min(Z[i] / 5.0, 1.0)
                color = (int(255 * (1 - depth_ratio)), 0, int(255 * depth_ratio))
                cv2.circle(lidar_rgb, (u, v), 2, color, -1)

        cv2.rectangle(lidar_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lidar_path = os.path.join(self.results_dir, f"s{sample_id}_{obj_id_safe}_lidar.jpg")
        cv2.imwrite(lidar_path, lidar_rgb)

    def _save_multi_overview(self, rgb, depth, detections, object_results,
                              lidar_points, sample_id, prompt):
        """保存多物体总览图 (1280x1440)"""
        h, w = rgb.shape[:2]

        # 输出图像尺寸 1280x1440 (上下各720)
        OUT_W, OUT_H = 1280, 1440
        HALF_H = 720

        # 颜色列表
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
        ]

        # ===== 上半部分: 检测结果 + LiDAR投影 =====
        det_rgb = rgb.copy()

        # 绘制 LiDAR 投影点
        if len(lidar_points) > 0:
            points_optical = self.transformer.rslidar_to_optical(lidar_points)
            uv = self.transformer.project_to_image(points_optical, self.intrinsics)
            Z = points_optical[:, 2]

            valid = (Z > 0.1) & (Z < 10) & \
                    (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < h)

            for idx in np.where(valid)[0]:
                u, v = int(uv[idx, 0]), int(uv[idx, 1])
                depth_ratio = min(Z[idx] / 5.0, 1.0)
                pt_color = (int(255 * (1 - depth_ratio)), 0, int(255 * depth_ratio))
                cv2.circle(det_rgb, (u, v), 3, pt_color, -1)

        # 绘制检测框和距离标签
        for i, (det, res) in enumerate(zip(detections, object_results)):
            color = colors[i % len(colors)]
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(det_rgb, (x1, y1), (x2, y2), color, 2)

            # 物体ID
            obj_id = f"{det['category']}_{det['index']}"
            if len(obj_id) > 20:
                obj_id = f"{det['category'][:17]}.._{det['index']}"

            # 距离标签: 相机距离 / LiDAR距离
            cam_dist = f"C:{res['camera']['distance']:.2f}" if res.get('camera') else "C:N/A"
            lidar_dist = f"L:{res['lidar_guided']['distance']:.2f}" if res.get('lidar_guided') else "L:N/A"

            # 绘制标签背景
            label1 = f"{obj_id}"
            label2 = f"{cam_dist} {lidar_dist}"
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            max_tw = max(tw1, tw2)

            # 标签背景
            cv2.rectangle(det_rgb, (x1, y1 - th1 - th2 - 10), (x1 + max_tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(det_rgb, label1, (x1 + 3, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(det_rgb, label2, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 添加上半部分标题
        title = f"Detection + LiDAR Projection | {len(detections)} objects"
        cv2.putText(det_rgb, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 缩放到 1280x720
        top_half = cv2.resize(det_rgb, (OUT_W, HALF_H))

        # ===== 下半部分: 深度图 =====
        depth_vis = np.clip(depth, 0, 5) / 5 * 255
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 在深度图上绘制检测框
        for i, (det, res) in enumerate(zip(detections, object_results)):
            color = colors[i % len(colors)]
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

        prefix = prompt.replace(' ', '_').replace('.', '_')[:30]
        combined_path = os.path.join(self.results_dir,
                                      f"s{sample_id}_{prefix}_multi_overview.jpg")
        cv2.imwrite(combined_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n总览图: {combined_path}")

    def _compute_multi_summary(self, object_results: List[dict]) -> dict:
        """计算多物体统计汇总"""
        total = len(object_results)
        camera_ok = sum(1 for r in object_results if r.get('camera'))
        guided_ok = sum(1 for r in object_results if r.get('lidar_guided'))
        indep_ok = sum(1 for r in object_results if r.get('lidar_independent'))

        # 收集误差
        errors_guided = [r['error_guided'] for r in object_results
                         if r.get('error_guided') is not None]
        errors_indep = [r['error_independent'] for r in object_results
                        if r.get('error_independent') is not None]

        summary = {
            'total': total,
            'total_objects': total,
            'success_rate': {
                'camera': f"{camera_ok}/{total}",
                'lidar_guided': f"{guided_ok}/{total}",
                'lidar_independent': f"{indep_ok}/{total}",
            },
            # 保存原始误差数组用于多次采样汇总
            'errors_guided': errors_guided,
            'errors_independent': errors_indep,
        }

        if errors_guided:
            arr = np.array(errors_guided)
            summary['error_guided'] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'abs_mean': float(np.mean(np.abs(arr))),
                'max': float(np.max(np.abs(arr))),
            }
            summary['error_guided_mean'] = float(np.mean(arr))
            summary['error_guided_std'] = float(np.std(arr))

        if errors_indep:
            arr = np.array(errors_indep)
            summary['error_independent'] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'abs_mean': float(np.mean(np.abs(arr))),
                'max': float(np.max(np.abs(arr))),
            }
            summary['error_indep_mean'] = float(np.mean(arr))
            summary['error_indep_std'] = float(np.std(arr))

        return summary

    def _print_multi_results(self, object_results: List[dict], summary: dict):
        """打印多物体结果表格"""
        # 表格宽度
        TABLE_WIDTH = 145

        print(f"\n{'='*TABLE_WIDTH}")
        print(f"{'多物体深度分析结果':^{TABLE_WIDTH-20}}(共 {summary['total_objects']} 个物体)")
        print(f"{'='*TABLE_WIDTH}")

        # 表头
        header = (f"{'No.':<4} │ {'物体ID':<25} │ {'相机距离':>10} │ {'LiDAR深度ROI':>12} │ "
                  f"{'LiDAR聚类':>10} │ {'误差':>10} │ {'误差%':>8} │ "
                  f"{'相机点数':>8} │ {'LiDAR点数':>9} │ {'bbox扩展':>8}")
        print(f"\n{header}")
        print("─" * TABLE_WIDTH)

        # 数据行
        for i, r in enumerate(object_results, 1):
            # 物体ID：类别_编号，确保显示完整编号
            category = r['category']
            index = r['index']
            obj_id = f"{category}_{index}"
            # 截断类别名但保留编号
            if len(obj_id) > 23:
                obj_id = f"{category[:20]}.._{index}"

            # 距离
            cam_dist = f"{r['camera']['distance']:.3f}m" if r.get('camera') else "N/A"
            guided_dist = f"{r['lidar_guided']['distance']:.3f}m" if r.get('lidar_guided') else "N/A"
            indep_dist = f"{r['lidar_independent']['distance']:.3f}m" if r.get('lidar_independent') else "N/A"

            # 误差 (相机 - LiDAR深度ROI)
            if r.get('camera') and r.get('lidar_guided'):
                err_val = r['camera']['distance'] - r['lidar_guided']['distance']
                err_abs = f"{err_val*1000:+.1f}mm"
                err_pct = f"{err_val / r['lidar_guided']['distance'] * 100:+.2f}%"
            else:
                err_abs = "N/A"
                err_pct = "N/A"

            # 点数
            cam_pts = str(r['camera']['stats']['num_points']) if r.get('camera') else "N/A"
            if r.get('lidar_guided'):
                lidar_pts = str(r['lidar_guided']['stats']['num_points'])
            elif r.get('lidar_independent'):
                lidar_pts = str(r['lidar_independent']['stats']['num_points'])
            else:
                lidar_pts = "N/A"

            # bbox 扩展
            bbox_expand = ""
            if r.get('lidar_guided') and r['lidar_guided']['stats'].get('bbox_expand_ratio', 0) > 0:
                bbox_expand = f"{r['lidar_guided']['stats']['bbox_expand_ratio']*100:.0f}%"
            elif r.get('lidar_independent') and r['lidar_independent']['stats'].get('bbox_expand_ratio', 0) > 0:
                bbox_expand = f"{r['lidar_independent']['stats']['bbox_expand_ratio']*100:.0f}%"
            else:
                bbox_expand = "-"

            print(f"{i:<4} │ {obj_id:<25} │ {cam_dist:>10} │ {guided_dist:>12} │ "
                  f"{indep_dist:>10} │ {err_abs:>10} │ {err_pct:>8} │ "
                  f"{cam_pts:>8} │ {lidar_pts:>9} │ {bbox_expand:>8}")

        print("─" * TABLE_WIDTH)

        # 汇总
        print(f"\n成功率: 相机 {summary['success_rate']['camera']} | "
              f"LiDAR深度ROI {summary['success_rate']['lidar_guided']} | "
              f"LiDAR聚类 {summary['success_rate']['lidar_independent']}")

        if summary.get('error_guided'):
            e = summary['error_guided']
            print(f"LiDAR深度ROI误差: {e['mean']*1000:+.1f}mm ± {e['std']*1000:.1f}mm "
                  f"(|avg|={e['abs_mean']*1000:.1f}mm, max={e['max']*1000:.1f}mm)")

        if summary.get('error_independent'):
            e = summary['error_independent']
            print(f"LiDAR聚类误差: {e['mean']*1000:+.1f}mm ± {e['std']*1000:.1f}mm "
                  f"(|avg|={e['abs_mean']*1000:.1f}mm, max={e['max']*1000:.1f}mm)")

        print(f"{'='*TABLE_WIDTH}")

    def _save_multi_json(self, result: dict, sample_id: int, prompt: str):
        """保存多物体分析 JSON 结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = prompt.replace(' ', '_').replace('.', '_')[:30]
        json_path = os.path.join(self.results_dir,
                                  f"s{sample_id}_{prefix}_multi_result.json")

        # 移除不可序列化的 mask
        result_copy = result.copy()
        with open(json_path, 'w') as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False)
        print(f"JSON 结果: {json_path}")

    def _print_multi_sample_summary(self, all_results: List[dict], prompt: str):
        """打印多次采样汇总"""
        num_samples = len(all_results)
        MIN_LIDAR_POINTS_THRESHOLD = 10  # 点云不足阈值

        print("\n")
        print("=" * 120)
        print(f"{'多次采样汇总':^110} ({num_samples} 次采样)")
        print("=" * 120)

        # 收集所有物体的详细数据
        all_objects = []  # [{distance, error_guided, error_indep, lidar_points, obj_id, sample_id}]

        for i, result in enumerate(all_results):
            summary = result.get('summary', {})
            print(f"\n样本 {i}: 检测 {summary.get('total', 0)} 物体 | "
                  f"LiDAR深度ROI误差: {summary.get('error_guided_mean', 0)*1000:+.1f}mm ± {summary.get('error_guided_std', 0)*1000:.1f}mm | "
                  f"LiDAR聚类误差: {summary.get('error_indep_mean', 0)*1000:+.1f}mm ± {summary.get('error_indep_std', 0)*1000:.1f}mm")

            # 收集每个物体的详细数据
            for obj in result.get('objects', []):
                if obj.get('camera') and obj.get('lidar_guided'):
                    cam_dist = obj['camera']['distance']
                    lidar_points = obj['lidar_guided']['stats'].get('num_points', 0)
                    all_objects.append({
                        'sample_id': i,
                        'obj_id': obj['obj_id'],
                        'distance': cam_dist,
                        'error_guided': obj.get('error_guided'),
                        'error_indep': obj.get('error_independent'),
                        'lidar_points': lidar_points,
                    })

        # ==================== 分段统计 ====================
        def calc_stats(errors_mm):
            """计算统计量"""
            if not errors_mm:
                return None
            arr = np.array(errors_mm)
            return {
                'count': len(arr),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'abs_mean': float(np.mean(np.abs(arr))),
                'max': float(np.max(np.abs(arr))),
            }

        # 按距离分段
        ranges = [
            ('< 2.5m', lambda d: d < 2.5),
            ('2.5m - 3.5m', lambda d: 2.5 <= d < 3.5),
            ('> 3.5m', lambda d: d >= 3.5),
        ]

        print("\n" + "=" * 120)
        print(f"{'分段统计 (按相机距离)':^110}")
        print("=" * 120)
        print(f"\n{'距离范围':<15} │ {'数量':>6} │ {'LiDAR深度ROI误差':^40} │ {'LiDAR聚类误差':^40}")
        print("-" * 120)

        range_stats = {}
        for range_name, range_filter in ranges:
            objs_in_range = [o for o in all_objects if range_filter(o['distance'])]
            guided_errors = [o['error_guided']*1000 for o in objs_in_range if o['error_guided'] is not None]
            indep_errors = [o['error_indep']*1000 for o in objs_in_range if o['error_indep'] is not None]

            guided_stats = calc_stats(guided_errors)
            indep_stats = calc_stats(indep_errors)

            if guided_stats:
                guided_str = f"{guided_stats['mean']:+.1f}±{guided_stats['std']:.1f}mm (|avg|={guided_stats['abs_mean']:.1f}mm)"
            else:
                guided_str = "N/A"

            if indep_stats:
                indep_str = f"{indep_stats['mean']:+.1f}±{indep_stats['std']:.1f}mm (|avg|={indep_stats['abs_mean']:.1f}mm)"
            else:
                indep_str = "N/A"

            print(f"{range_name:<15} │ {len(objs_in_range):>6} │ {guided_str:^40} │ {indep_str:^40}")
            range_stats[range_name] = {'guided': guided_stats, 'indep': indep_stats, 'count': len(objs_in_range)}

        # ==================== 全部物体汇总 ====================
        all_guided_errors = [o['error_guided']*1000 for o in all_objects if o['error_guided'] is not None]
        all_indep_errors = [o['error_indep']*1000 for o in all_objects if o['error_indep'] is not None]

        print("-" * 120)
        all_guided_stats = calc_stats(all_guided_errors)
        all_indep_stats = calc_stats(all_indep_errors)
        if all_guided_stats:
            guided_str = f"{all_guided_stats['mean']:+.1f}±{all_guided_stats['std']:.1f}mm (|avg|={all_guided_stats['abs_mean']:.1f}mm)"
        else:
            guided_str = "N/A"
        if all_indep_stats:
            indep_str = f"{all_indep_stats['mean']:+.1f}±{all_indep_stats['std']:.1f}mm (|avg|={all_indep_stats['abs_mean']:.1f}mm)"
        else:
            indep_str = "N/A"
        print(f"{'全部':.<15} │ {len(all_objects):>6} │ {guided_str:^40} │ {indep_str:^40}")
        print("=" * 120)

        # ==================== 剔除点云不足的物体 ====================
        sufficient_objects = [o for o in all_objects if o['lidar_points'] >= MIN_LIDAR_POINTS_THRESHOLD]
        insufficient_objects = [o for o in all_objects if o['lidar_points'] < MIN_LIDAR_POINTS_THRESHOLD]

        print("\n" + "=" * 120)
        print(f"{'剔除点云不足物体后的统计 (LiDAR点数 >= ' + str(MIN_LIDAR_POINTS_THRESHOLD) + ')':^110}")
        print("=" * 120)
        print(f"\n剔除 {len(insufficient_objects)} 个点云不足物体，保留 {len(sufficient_objects)} 个")

        if insufficient_objects:
            print(f"\n被剔除的物体:")
            for o in insufficient_objects[:10]:  # 最多显示10个
                print(f"  - 样本{o['sample_id']} {o['obj_id']}: {o['distance']:.2f}m, {o['lidar_points']}点")
            if len(insufficient_objects) > 10:
                print(f"  ... 共 {len(insufficient_objects)} 个")

        print(f"\n{'距离范围':<15} │ {'数量':>6} │ {'LiDAR深度ROI误差':^40} │ {'LiDAR聚类误差':^40}")
        print("-" * 120)

        for range_name, range_filter in ranges:
            objs_in_range = [o for o in sufficient_objects if range_filter(o['distance'])]
            guided_errors = [o['error_guided']*1000 for o in objs_in_range if o['error_guided'] is not None]
            indep_errors = [o['error_indep']*1000 for o in objs_in_range if o['error_indep'] is not None]

            guided_stats = calc_stats(guided_errors)
            indep_stats = calc_stats(indep_errors)

            if guided_stats:
                guided_str = f"{guided_stats['mean']:+.1f}±{guided_stats['std']:.1f}mm (|avg|={guided_stats['abs_mean']:.1f}mm)"
            else:
                guided_str = "N/A"
            if indep_stats:
                indep_str = f"{indep_stats['mean']:+.1f}±{indep_stats['std']:.1f}mm (|avg|={indep_stats['abs_mean']:.1f}mm)"
            else:
                indep_str = "N/A"

            print(f"{range_name:<15} │ {len(objs_in_range):>6} │ {guided_str:^40} │ {indep_str:^40}")

        # 汇总行
        suff_guided = [o['error_guided']*1000 for o in sufficient_objects if o['error_guided'] is not None]
        suff_indep = [o['error_indep']*1000 for o in sufficient_objects if o['error_indep'] is not None]
        print("-" * 120)
        suff_guided_stats = calc_stats(suff_guided)
        suff_indep_stats = calc_stats(suff_indep)
        if suff_guided_stats:
            guided_str = f"{suff_guided_stats['mean']:+.1f}±{suff_guided_stats['std']:.1f}mm (|avg|={suff_guided_stats['abs_mean']:.1f}mm)"
        else:
            guided_str = "N/A"
        if suff_indep_stats:
            indep_str = f"{suff_indep_stats['mean']:+.1f}±{suff_indep_stats['std']:.1f}mm (|avg|={suff_indep_stats['abs_mean']:.1f}mm)"
        else:
            indep_str = "N/A"
        print(f"{'全部(剔除后)':.<15} │ {len(sufficient_objects):>6} │ {guided_str:^40} │ {indep_str:^40}")
        print("=" * 120)

        # 保存汇总 JSON
        prefix = prompt.replace(' ', '_').replace('.', '_')[:30]
        summary_path = os.path.join(self.results_dir, f"multi_summary_{prefix}_{num_samples}samples.json")
        summary_data = {
            'prompt': prompt,
            'num_samples': num_samples,
            'total_measurements': len(all_objects),
            'all_objects': {
                'lidar_guided': all_guided_stats,
                'lidar_independent': all_indep_stats,
            },
            'by_range': range_stats,
            'sufficient_points': {
                'threshold': MIN_LIDAR_POINTS_THRESHOLD,
                'count': len(sufficient_objects),
                'excluded': len(insufficient_objects),
                'lidar_guided': suff_guided_stats,
                'lidar_independent': suff_indep_stats,
            },
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n汇总结果: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='3D 深度性能分析工具')
    parser.add_argument('--camera', default='top', choices=['top', 'chassis'],
                        help='相机选择')
    parser.add_argument('--prompt', default='box', help='检测目标提示词')
    parser.add_argument('--samples', type=int, default=3, help='采样次数')
    parser.add_argument('--no-depth-optimize', action='store_true',
                        help='禁用深度优化')
    parser.add_argument('--interval', type=float, default=1.5, help='采样间隔 (秒)')
    parser.add_argument('--multi', action='store_true',
                        help='多物体模式 (一次检测多个物体)')

    args = parser.parse_args()

    analyzer = DepthAccuracyAnalyzer(
        camera_name=args.camera,
        use_depth_optimizer=not args.no_depth_optimize,
    )

    try:
        analyzer.initialize()

        if args.multi:
            # 多物体模式 (支持多次采样)
            all_results = []
            for i in range(args.samples):
                if i > 0:
                    print(f"\n等待 {args.interval} 秒...")
                    import time
                    time.sleep(args.interval)
                results = analyzer.analyze_multi(
                    prompt=args.prompt,
                    sample_id=i,
                )
                all_results.append(results)

            # 多次采样汇总
            if args.samples > 1:
                analyzer._print_multi_sample_summary(all_results, args.prompt)
        else:
            # 单物体模式 (多次采样)
            results = analyzer.analyze(
                prompt=args.prompt,
                num_samples=args.samples,
                interval=args.interval,
            )

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.shutdown()


if __name__ == '__main__':
    main()
