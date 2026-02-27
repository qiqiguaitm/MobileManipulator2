#!/usr/bin/env python3
"""
Scene Perception 3D - 核心测量算法

从 depth_accuracy_analyzer.py 提取的核心 3D 测量算法，
支持可配置的目标坐标系 (base_link / arm_base_link)。

这是一个纯 Python 模块，不依赖 ROS。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from perception.coordinate_transformer import CoordinateTransformer


@dataclass
class MeasurementResult:
    """单次测量结果"""
    valid: bool = False
    confidence: float = 0.0
    centroid: Optional[np.ndarray] = None      # [x, y, z] in target_frame
    centroid_optical: Optional[np.ndarray] = None  # [x, y, z] in optical frame
    distance: float = 0.0                      # 到目标坐标系原点的距离
    stats: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None


class SimpleConfig:
    """简单的配置类，用于初始化服务客户端"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


class ScenePerceptionCore:
    """3D 场景感知核心算法"""

    # 算法参数
    DEPTH_MIN = 0.3   # m
    DEPTH_MAX = 10.0  # m
    MASK_ERODE_KERNEL = 5
    IQR_FACTOR = 1.5
    LIDAR_DEPTH_TOLERANCE = 0.3  # m
    MIN_MASK_AREA = 300          # pixels
    MIN_DEPTH_POINTS = 10
    MIN_LIDAR_POINTS = 3

    def __init__(self,
                 transformer: CoordinateTransformer,
                 intrinsics: dict,
                 target_frame: str = 'base_link',
                 depth_optimizer=None):
        """
        Args:
            transformer: 坐标变换器
            intrinsics: 相机内参 {fx, fy, cx, cy, width, height}
            target_frame: 目标坐标系 ('base_link' 或 'arm_base_link')
            depth_optimizer: CDM 深度优化器实例 (可选，DepthOptimizerOnline)
        """
        self.transformer = transformer
        self.intrinsics = intrinsics
        self.target_frame = target_frame
        self.depth_optimizer = depth_optimizer

        # 验证目标坐标系
        if target_frame not in ('base_link', 'arm_base_link'):
            raise ValueError(f"Unsupported target frame: {target_frame}")

    def optimize_depth(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        使用 CDM 服务优化深度图

        Args:
            rgb: BGR 图像 (H, W, 3) uint8
            depth: 深度图 (H, W) float32, 单位: 米

        Returns:
            优化后的深度图 (H, W) float32, 单位: 米
        """
        if self.depth_optimizer is None:
            return depth

        # 转换为 mm uint16 (CDM 服务要求的格式)
        depth_mm = (depth * 1000).astype(np.uint16)

        result = self.depth_optimizer.forward(rgb, depth_mm)

        if result.get('success') and 'depth' in result:
            optimized = result['depth'].astype(np.float32) / 1000.0
            return optimized
        else:
            print(f"[WARN] Depth optimization failed: {result.get('error', 'unknown')}")
            return depth

    def camera_3d_percept(self, depth: np.ndarray, mask: np.ndarray,
                          skip_transform: bool = False) -> MeasurementResult:
        """
        深度相机 3D 感知 (基于 Mask)

        Args:
            depth: 深度图 (H, W) float32, 单位: 米
            mask: 二值 mask (H, W) uint8
            skip_transform: 是否跳过坐标变换（用于批量变换优化）

        Returns:
            MeasurementResult: 包含 3D 位置、距离、置信度
                - skip_transform=False: centroid 在目标坐标系
                - skip_transform=True: centroid 在 optical 坐标系
        """
        result = MeasurementResult()

        # 1. Mask 腐蚀 (去除边缘噪声)
        kernel = np.ones((self.MASK_ERODE_KERNEL, self.MASK_ERODE_KERNEL), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        if eroded_mask.sum() < self.MIN_DEPTH_POINTS:
            result.error_msg = f"Eroded mask too small: {eroded_mask.sum()}"
            return result

        # 2. 提取深度值
        depth_values = depth[eroded_mask > 0]

        # 3. 剔除无效值
        valid_mask = (depth_values > self.DEPTH_MIN) & (depth_values < self.DEPTH_MAX)
        depth_values = depth_values[valid_mask]

        if len(depth_values) < self.MIN_DEPTH_POINTS:
            result.error_msg = f"Too few valid depth points: {len(depth_values)}"
            return result

        # 4. IQR 异常值剔除
        q1, q3 = np.percentile(depth_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.IQR_FACTOR * iqr
        upper_bound = q3 + self.IQR_FACTOR * iqr
        depth_values = depth_values[(depth_values >= lower_bound) & (depth_values <= upper_bound)]

        if len(depth_values) < self.MIN_DEPTH_POINTS:
            result.error_msg = f"Too few points after IQR: {len(depth_values)}"
            return result

        # 5. 生成 3D 点云 (optical frame) - 向量化优化
        ys, xs = np.where(eroded_mask > 0)
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']

        # 提取所有 mask 位置的深度
        ds = depth[ys, xs]

        # 向量化过滤：深度范围 + IQR 边界
        valid = (ds >= lower_bound) & (ds <= upper_bound) & \
                (ds > self.DEPTH_MIN) & (ds < self.DEPTH_MAX)

        if valid.sum() < self.MIN_DEPTH_POINTS:
            result.error_msg = f"Too few 3D points: {valid.sum()}"
            return result

        # 向量化反投影
        xs_valid = xs[valid]
        ys_valid = ys[valid]
        ds_valid = ds[valid]

        X = (xs_valid - cx) * ds_valid / fx
        Y = (ys_valid - cy) * ds_valid / fy
        Z = ds_valid

        points = np.column_stack([X, Y, Z])

        # 6. 计算质心 (使用中值更鲁棒)
        centroid_optical = np.median(points, axis=0)

        # 7. 变换到目标坐标系 (可选，用于批量变换优化)
        if skip_transform:
            # 跳过变换，返回 optical 坐标系的结果
            centroid_target = centroid_optical
            distance = np.linalg.norm(centroid_optical)
        else:
            # 正常变换到目标坐标系
            centroid_target = self.transformer.optical_to_target(
                centroid_optical.reshape(1, 3), self.target_frame
            )[0]
            distance = np.linalg.norm(centroid_target)

        # 计算置信度
        depth_std = np.std(depth_values)
        confidence = self._compute_confidence(len(points), depth_std)

        result.valid = True
        result.confidence = confidence
        result.centroid = centroid_target
        result.centroid_optical = centroid_optical
        result.distance = distance
        result.stats = {
            'depth_median': float(np.median(depth_values)),
            'depth_std': float(depth_std),
            'num_points': len(points),
            'valid_ratio': len(depth_values) / max(mask.sum(), 1),
        }

        return result

    def batch_camera_3d_percept(self, depth: np.ndarray,
                                detections: List[Dict[str, Any]],
                                skip_transform: bool = False) -> List[MeasurementResult]:
        """
        批量 bbox-scoped 3D 测量（替代逐物体全图 np.where 扫描）

        Args:
            depth: 深度图 (H, W) float32, 单位: 米
            detections: 检测列表，每项含 'mask' (H,W uint8) 和 'bbox' [x1,y1,x2,y2]
            skip_transform: 是否跳过坐标变换

        Returns:
            list[MeasurementResult]，与 detections 一一对应
        """
        H, W = depth.shape[:2]
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        kernel = np.ones((self.MASK_ERODE_KERNEL, self.MASK_ERODE_KERNEL), np.uint8)

        results = []
        for det in detections:
            result = MeasurementResult()
            mask = det['mask']
            bbox = det.get('bbox', [0, 0, 0, 0])

            # bbox 安全裁剪到图像边界
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(W, int(bbox[2]))
            y2 = min(H, int(bbox[3]))

            # bbox 退化检查
            if x2 <= x1 or y2 <= y1:
                result.error_msg = f"Degenerate bbox: [{x1},{y1},{x2},{y2}]"
                results.append(result)
                continue

            # ROI 裁剪
            mask_roi = mask[y1:y2, x1:x2]
            depth_roi = depth[y1:y2, x1:x2]

            # 腐蚀（复用 kernel）
            eroded_roi = cv2.erode(mask_roi, kernel, iterations=1)

            if eroded_roi.sum() < self.MIN_DEPTH_POINTS:
                result.error_msg = f"Eroded mask too small: {eroded_roi.sum()}"
                results.append(result)
                continue

            # ROI 内 np.where（远小于全图扫描）
            ys_local, xs_local = np.where(eroded_roi > 0)

            # 提取深度值
            ds = depth_roi[ys_local, xs_local]

            # 剔除无效值
            valid = (ds > self.DEPTH_MIN) & (ds < self.DEPTH_MAX)
            if valid.sum() < self.MIN_DEPTH_POINTS:
                result.error_msg = f"Too few valid depth points: {valid.sum()}"
                results.append(result)
                continue

            depth_values = ds[valid]

            # IQR 异常值剔除
            q1, q3 = np.percentile(depth_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - self.IQR_FACTOR * iqr
            upper_bound = q3 + self.IQR_FACTOR * iqr
            depth_values = depth_values[(depth_values >= lower_bound) & (depth_values <= upper_bound)]

            if len(depth_values) < self.MIN_DEPTH_POINTS:
                result.error_msg = f"Too few points after IQR: {len(depth_values)}"
                results.append(result)
                continue

            # 向量化过滤：深度范围 + IQR
            valid2 = valid & (ds >= lower_bound) & (ds <= upper_bound)
            if valid2.sum() < self.MIN_DEPTH_POINTS:
                result.error_msg = f"Too few 3D points: {valid2.sum()}"
                results.append(result)
                continue

            # 偏移到全图坐标 + 反投影
            xs_img = xs_local[valid2].astype(np.float64) + x1
            ys_img = ys_local[valid2].astype(np.float64) + y1
            ds_valid = ds[valid2]

            X = (xs_img - cx) * ds_valid / fx
            Y = (ys_img - cy) * ds_valid / fy
            Z = ds_valid
            points = np.column_stack([X, Y, Z])

            # 中值质心
            centroid_optical = np.median(points, axis=0)

            # 坐标变换
            if skip_transform:
                centroid_target = centroid_optical
                distance = np.linalg.norm(centroid_optical)
            else:
                centroid_target = self.transformer.optical_to_target(
                    centroid_optical.reshape(1, 3), self.target_frame
                )[0]
                distance = np.linalg.norm(centroid_target)

            # 置信度
            depth_std = np.std(depth_values)
            confidence = self._compute_confidence(len(points), depth_std)

            result.valid = True
            result.confidence = confidence
            result.centroid = centroid_target
            result.centroid_optical = centroid_optical
            result.distance = distance
            result.stats = {
                'depth_median': float(np.median(depth_values)),
                'depth_std': float(depth_std),
                'num_points': len(points),
                'valid_ratio': len(depth_values) / max(mask_roi.sum(), 1),
            }
            results.append(result)

        return results

    def lidar_3d_percept(self, lidar_points: np.ndarray, bbox: List[float],
                         camera_depth: float = None) -> MeasurementResult:
        """
        LiDAR 3D 感知 (相机引导模式)

        使用相机检测的 bbox 和深度范围筛选 LiDAR 点云，
        计算目标物体的 3D 质心位置。

        Args:
            lidar_points: (N, 4) rslidar 坐标系点云 [x, y, z, intensity]
            bbox: [x1, y1, x2, y2] 检测框
            camera_depth: 相机测得的深度 (m), 用于深度范围筛选 (可选)

        Returns:
            MeasurementResult: 包含 3D 位置、距离、置信度
        """
        result = MeasurementResult()

        if lidar_points is None or len(lidar_points) == 0:
            result.error_msg = "No LiDAR points"
            return result

        # 1. 变换到 optical frame
        points_optical = self.transformer.rslidar_to_optical(lidar_points)

        # 2. 投影到图像平面
        uv = self.transformer.project_to_image(points_optical, self.intrinsics)

        # 3. bbox + 深度范围筛选
        x1, y1, x2, y2 = bbox
        Z = points_optical[:, 2]  # optical frame 中 Z 是深度

        # 深度范围
        if camera_depth is not None:
            depth_min = camera_depth - self.LIDAR_DEPTH_TOLERANCE
            depth_max = camera_depth + self.LIDAR_DEPTH_TOLERANCE
        else:
            depth_min = 0.3
            depth_max = 15.0

        in_depth = (Z >= depth_min) & (Z <= depth_max) & (Z > 0.1)

        # 逐步扩大 bbox 范围直到找到足够的点
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        expand_ratio = 0.0
        expand_step_ratio = 0.2
        img_w, img_h = self.intrinsics['width'], self.intrinsics['height']

        while True:
            expand_w = bbox_w * expand_ratio / 2
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
                break

            # 检查是否已扩展到图像边界
            if ex1 <= 0 and ey1 <= 0 and ex2 >= img_w and ey2 >= img_h:
                break

            expand_ratio += expand_step_ratio

        if len(filtered_indices) < self.MIN_LIDAR_POINTS:
            result.error_msg = f"Too few LiDAR points after filter: {len(filtered_indices)}"
            return result

        # 获取筛选后的点 (rslidar 坐标系)
        filtered_points = lidar_points[filtered_indices, :3]

        # 4. IQR 异常值剔除
        distances = np.linalg.norm(filtered_points, axis=1)
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        valid_mask = (distances >= q1 - self.IQR_FACTOR * iqr) & \
                     (distances <= q3 + self.IQR_FACTOR * iqr)
        filtered_points = filtered_points[valid_mask]

        if len(filtered_points) < self.MIN_LIDAR_POINTS:
            result.error_msg = f"Too few LiDAR points after IQR: {len(filtered_points)}"
            return result

        # 5. 计算质心 (rslidar 坐标系)
        centroid_rslidar = np.median(filtered_points, axis=0)

        # 6. 变换到目标坐标系
        centroid_target = self.transformer.rslidar_to_target(
            centroid_rslidar.reshape(1, 3), self.target_frame
        )[0]

        distance = np.linalg.norm(centroid_target)

        # 计算置信度
        depth_std = np.std(distances[valid_mask])
        confidence = self._compute_confidence(len(filtered_points), depth_std)

        result.valid = True
        result.confidence = confidence
        result.centroid = centroid_target
        result.centroid_optical = centroid_rslidar  # 实际是 rslidar 坐标系
        result.distance = distance
        result.stats = {
            'depth_median': float(np.median(distances[valid_mask])),
            'depth_std': float(depth_std),
            'num_points': len(filtered_points),
            'depth_range': [float(depth_min), float(depth_max)],
            'bbox_expand_ratio': expand_ratio,
        }

        return result

    def _compute_confidence(self, num_points: int, depth_std: float,
                            max_std: float = 0.3) -> float:
        """计算测量置信度"""
        point_score = min(num_points / 100.0, 1.0)
        std_score = max(0, 1.0 - depth_std / max_std)
        return 0.7 * point_score + 0.3 * std_score


def test_perception_core():
    """测试核心算法"""
    print("=" * 60)
    print("ScenePerceptionCore Test")
    print("=" * 60)

    # 初始化组件 (不加载外参)
    transformer = CoordinateTransformer()

    intrinsics = {
        'fx': 386.82, 'fy': 386.27,
        'cx': 320.18, 'cy': 242.99,
        'width': 640, 'height': 480,
    }

    # 添加单位变换用于测试
    transformer.add_transform('optical_to_base', np.eye(4))

    core = ScenePerceptionCore(
        transformer=transformer,
        intrinsics=intrinsics,
        target_frame='base_link',
    )

    # 创建测试数据
    depth = np.ones((480, 640), dtype=np.float32) * 1.5  # 1.5m 深度
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:280, 280:360] = 1  # 中心区域 mask

    # 测试相机 3D 感知
    result = core.camera_3d_percept(depth, mask)
    if result.valid:
        print(f"camera_3d_percept success:")
        print(f"  Position: {result.centroid}")
        print(f"  Distance: {result.distance:.3f}m")
        print(f"  Confidence: {result.confidence:.2f}")
    else:
        print(f"camera_3d_percept failed: {result.error_msg}")

    print("\nTest completed")


if __name__ == '__main__':
    test_perception_core()
