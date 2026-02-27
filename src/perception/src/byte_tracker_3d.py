#!/usr/bin/env python3
"""
ByteTracker3D - 基于ByteTrack思想的3D目标跟踪器

用于对双相机融合后的检测结果进行跨帧跟踪。

核心创新（来自ByteTrack）：
1. 两阶段匹配：利用低置信度检测恢复被遮挡的轨迹
2. 匈牙利算法：全局最优匹配（替代贪心）
3. Kalman滤波：预测轨迹位置，处理快速移动

适配你的场景：
- 高置信度 = source='fused'（双相机融合成功）
- 低置信度 = source='*_only'（单相机独立检测）

参考：
- ByteTrack: https://arxiv.org/abs/2110.06864
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from scipy.optimize import linear_sum_assignment

# 复用类别兼容性检查
from perception.dual_camera_matcher import CategoryCompatibility


# ============================================================================
# 配置
# ============================================================================

@dataclass
class TrackerConfig:
    """跟踪器配置"""
    # 距离阈值
    match_thresh: float = 0.15      # 第一阶段匹配阈值 (15cm)
    second_thresh: float = 0.25     # 第二阶段匹配阈值 (25cm, 更宽松)

    # 轨迹管理
    track_buffer: int = 15          # Lost轨迹保留帧数 (5Hz × 3秒)
    confirm_frames: int = 2         # 新轨迹确认帧数

    # Kalman滤波器参数
    process_noise_pos: float = 0.01     # 位置过程噪声
    process_noise_vel: float = 0.05     # 速度过程噪声
    measurement_noise: float = 0.05     # 观测噪声 (深度误差±5cm)


# ============================================================================
# 轨迹状态
# ============================================================================

class TrackState(Enum):
    """轨迹状态"""
    New = 0         # 新检测，待确认
    Tracked = 1     # 活跃跟踪中
    Lost = 2        # 暂时丢失
    Removed = 3     # 已删除


# ============================================================================
# Kalman滤波器（3D位置）
# ============================================================================

class KalmanFilter3D:
    """
    3D位置Kalman滤波器

    状态向量: [x, y, z, vx, vy, vz]
    观测向量: [x, y, z]

    运动模型: 匀速运动
    """

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg

        # 状态维度
        self.dim_x = 6  # [x, y, z, vx, vy, vz]
        self.dim_z = 3  # [x, y, z]

        # 状态转移矩阵 F (匀速运动，dt=1)
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 3] = 1  # x += vx
        self.F[1, 4] = 1  # y += vy
        self.F[2, 5] = 1  # z += vz

        # 观测矩阵 H
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # 过程噪声 Q
        self.Q = np.diag([
            cfg.process_noise_pos,  # x
            cfg.process_noise_pos,  # y
            cfg.process_noise_pos,  # z
            cfg.process_noise_vel,  # vx
            cfg.process_noise_vel,  # vy
            cfg.process_noise_vel,  # vz
        ]).astype(np.float32)

        # 观测噪声 R
        self.R = np.diag([
            cfg.measurement_noise,
            cfg.measurement_noise,
            cfg.measurement_noise,
        ]).astype(np.float32)

        # 状态和协方差
        self.x = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 0.1

    def init(self, position: np.ndarray):
        """用初始位置初始化"""
        self.x[:3] = position
        self.x[3:] = 0  # 初始速度为0
        self.P = np.eye(6, dtype=np.float32) * 0.1

    def predict(self) -> np.ndarray:
        """预测下一帧位置"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """用观测修正"""
        z = np.asarray(z, dtype=np.float32)

        # 残差
        y = z - self.H @ self.x

        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3].copy()

    @property
    def position(self) -> np.ndarray:
        """当前位置估计"""
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """当前速度估计"""
        return self.x[3:].copy()


# ============================================================================
# 单条轨迹
# ============================================================================

class STrack3D:
    """单条3D跟踪轨迹"""

    _count = 0  # 类变量，用于生成唯一ID

    def __init__(self, position: np.ndarray, category: str,
                 confidence: float, cfg: TrackerConfig,
                 detection: dict = None):
        # 分配唯一ID
        STrack3D._count += 1
        self.track_id = STrack3D._count

        # 基本信息
        self.category = category
        self.confidence = confidence

        # 状态
        self.state = TrackState.New
        self.frame_id = 0           # 最后更新的帧号
        self.start_frame = 0        # 开始帧号
        self.tracklet_len = 0       # 连续跟踪长度

        # 保存最后匹配的检测数据（包含 _original 等字段）
        self.last_detection = detection

        # Kalman滤波器
        self.kalman = KalmanFilter3D(cfg)
        self.kalman.init(np.asarray(position))

        # 配置
        self.cfg = cfg

    @property
    def position(self) -> np.ndarray:
        """当前位置（Kalman估计）"""
        return self.kalman.position

    def predict(self):
        """Kalman预测"""
        self.kalman.predict()

    def update(self, detection: dict, frame_id: int):
        """用检测结果更新轨迹"""
        # Kalman修正
        self.kalman.update(detection['position'])

        # 更新属性
        self.confidence = detection['confidence']
        self.frame_id = frame_id
        self.tracklet_len += 1

        # 保存最后匹配的检测数据（包含 _original 等字段）
        self.last_detection = detection

    def mark_lost(self):
        """标记为Lost"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """标记为Removed"""
        self.state = TrackState.Removed

    def activate(self, frame_id: int):
        """激活轨迹（New → Tracked）"""
        self.state = TrackState.Tracked
        self.start_frame = frame_id

    def reactivate(self, detection: dict, frame_id: int):
        """重新激活轨迹（Lost → Tracked）"""
        self.update(detection, frame_id)
        self.state = TrackState.Tracked

    @staticmethod
    def reset_id():
        """重置ID计数器"""
        STrack3D._count = 0


# ============================================================================
# ByteTracker3D 主类
# ============================================================================

class ByteTracker3D:
    """
    基于ByteTrack思想的3D目标跟踪器

    替换SimpleTracker，对融合后的检测结果进行跟踪。

    核心改进:
    1. 匈牙利算法（全局最优）替代贪心匹配
    2. Kalman滤波预测（处理快速移动）
    3. 两阶段匹配（恢复遮挡轨迹）- ByteTrack核心
    """

    def __init__(self, cfg: TrackerConfig = None, log=None):
        """
        Args:
            cfg: 跟踪器配置
            log: 日志对象
        """
        self.cfg = cfg or TrackerConfig()
        self.log = log

        # 轨迹池
        self.tracked_stracks: List[STrack3D] = []  # Tracked状态
        self.lost_stracks: List[STrack3D] = []     # Lost状态
        self.removed_stracks: List[STrack3D] = []  # Removed状态（可选保留）

        # 帧计数
        self.frame_id = 0

        # 重置ID计数器
        STrack3D.reset_id()

    def update(self, fused_objects: List[dict]) -> List[dict]:
        """
        核心更新函数 - ByteTrack两阶段匹配

        Args:
            fused_objects: DualCameraMatcher输出的融合物体列表
                [{'position': np.array, 'category': str,
                  'confidence': float, 'source': str, 'quality': str}, ...]

        Returns:
            tracked_objects: 带track_id的跟踪结果列表
                [{'track_id': int, 'category': str, 'position': np.array,
                  'confidence': float, 'source': str, 'quality': str}, ...]
        """
        self.frame_id += 1

        # ========== Step 0: 分离高/低置信度检测 ==========
        high_dets = []  # 双相机融合成功
        low_dets = []   # 单相机独立检测

        for obj in fused_objects:
            if obj.get('source') == 'fused':
                high_dets.append(obj)
            else:
                # chassis_only 或 top_only
                low_dets.append(obj)

        if self.log:
            self.log.debug(f"Frame {self.frame_id}: {len(high_dets)} high, {len(low_dets)} low detections")

        # ========== Step 1: Kalman预测所有轨迹 ==========
        # 合并活跃和丢失轨迹
        all_stracks = self.tracked_stracks + self.lost_stracks
        for track in all_stracks:
            track.predict()

        # ========== Step 2: 第一阶段匹配 (高置信度 vs 所有轨迹) ==========
        matches1, unmatched_track_indices, unmatched_det_indices = \
            self._match(all_stracks, high_dets, self.cfg.match_thresh)

        # 更新匹配的轨迹
        for t_idx, d_idx in matches1:
            track = all_stracks[t_idx]
            det = high_dets[d_idx]

            if track.state == TrackState.Lost:
                # Lost → Tracked (恢复)
                track.reactivate(det, self.frame_id)
            else:
                # 正常更新
                track.update(det, self.frame_id)
                if track.state == TrackState.New:
                    if track.tracklet_len >= self.cfg.confirm_frames:
                        track.activate(self.frame_id)
                else:
                    track.state = TrackState.Tracked

        # ========== Step 3: 第二阶段匹配 (低置信度 vs 未匹配轨迹) ==========
        # ByteTrack核心: 用单相机检测恢复/继续跟踪
        # 注意: 这里匹配所有未匹配轨迹，不仅是Lost轨迹
        remaining_tracks = [all_stracks[i] for i in unmatched_track_indices]

        matches2 = []
        if remaining_tracks and low_dets:
            matches2, unmatched_remaining_indices, _ = \
                self._match(remaining_tracks, low_dets, self.cfg.second_thresh)

            # 更新/恢复匹配的轨迹
            for t_idx, d_idx in matches2:
                track = remaining_tracks[t_idx]
                det = low_dets[d_idx]

                if track.state == TrackState.Lost:
                    # Lost → Tracked (恢复)
                    track.reactivate(det, self.frame_id)
                    if self.log:
                        self.log.debug(f"Recovered Lost track {track.track_id} with low-conf detection")
                else:
                    # 用低置信度检测继续跟踪
                    track.update(det, self.frame_id)
                    if track.state == TrackState.New:
                        if track.tracklet_len >= self.cfg.confirm_frames:
                            track.activate(self.frame_id)
                    else:
                        track.state = TrackState.Tracked
                    if self.log:
                        self.log.debug(f"Continued track {track.track_id} with low-conf detection")

        # ========== Step 4: 处理未匹配轨迹 ==========
        # 找出所有未匹配的轨迹
        matched_track_ids = set()
        for t_idx, _ in matches1:
            matched_track_ids.add(id(all_stracks[t_idx]))
        for t_idx, _ in matches2:
            matched_track_ids.add(id(remaining_tracks[t_idx]))

        for track in all_stracks:
            if id(track) in matched_track_ids:
                continue

            if track.state == TrackState.Tracked:
                # Tracked → Lost
                track.mark_lost()
            elif track.state == TrackState.Lost:
                # 检查是否超时
                if self.frame_id - track.frame_id > self.cfg.track_buffer:
                    track.mark_removed()
            elif track.state == TrackState.New:
                # 未确认的新轨迹也标记为Lost
                track.mark_lost()

        # ========== Step 5: 新轨迹 (未匹配的高置信度检测) ==========
        for d_idx in unmatched_det_indices:
            det = high_dets[d_idx]
            new_track = STrack3D(
                position=det['position'],
                category=det['category'],
                confidence=det['confidence'],
                cfg=self.cfg,
                detection=det,  # 保存原始检测数据
            )
            new_track.frame_id = self.frame_id
            new_track.start_frame = self.frame_id
            new_track.tracklet_len = 1
            self.tracked_stracks.append(new_track)

            if self.log:
                self.log.debug(f"New track {new_track.track_id}: {det['category']}")

        # ========== Step 6: 更新轨迹池 ==========
        self._update_pools()

        # ========== 构建输出 ==========
        return self._get_output(fused_objects)

    def _match(self, tracks: List[STrack3D], detections: List[dict],
               thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匈牙利算法匹配

        Args:
            tracks: 轨迹列表
            detections: 检测列表
            thresh: 距离阈值

        Returns:
            matches: 匹配对 [(track_idx, det_idx), ...]
            unmatched_tracks: 未匹配轨迹索引
            unmatched_dets: 未匹配检测索引
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # 构建代价矩阵
        cost_matrix = self._compute_cost(tracks, detections)

        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 过滤超出阈值的匹配
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    def _compute_cost(self, tracks: List[STrack3D],
                      detections: List[dict]) -> np.ndarray:
        """
        计算代价矩阵（3D距离 + 类别兼容性）

        类别不兼容的设为无穷大
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), 1e6, dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # 类别兼容性检查
                if not CategoryCompatibility.is_compatible(
                    track.category, det['category']
                ):
                    continue  # 不兼容，保持无穷大

                # 3D距离
                dist = np.linalg.norm(track.position - det['position'])
                cost[i, j] = dist

        return cost

    def _update_pools(self):
        """更新轨迹池"""
        # 分类轨迹
        new_tracked = []
        new_lost = []

        for track in self.tracked_stracks + self.lost_stracks:
            if track.state == TrackState.Tracked:
                new_tracked.append(track)
            elif track.state == TrackState.Lost:
                new_lost.append(track)
            elif track.state == TrackState.New:
                # 新轨迹放入tracked池
                new_tracked.append(track)
            elif track.state == TrackState.Removed:
                self.removed_stracks.append(track)

        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        # 限制removed池大小（可选）
        if len(self.removed_stracks) > 100:
            self.removed_stracks = self.removed_stracks[-50:]

    def _get_output(self, fused_objects: List[dict]) -> List[dict]:
        """
        构建输出结果

        策略：返回所有输入检测，为已确认轨迹添加 track_id
        - 已确认轨迹：添加 track_id + Kalman 滤波位置
        - 未确认检测：保持原样（track_id=None）

        这样保证融合结果不丢失任何检测，同时为稳定跟踪的物体提供持久 ID。
        """
        output = []

        # 1. 收集所有已确认轨迹，建立检测ID到轨迹的映射
        # 使用 id(detection) 作为 key，因为每个检测是独立的 dict 对象
        confirmed_tracks = {}
        for track in self.tracked_stracks:
            is_confirmed = (
                track.state == TrackState.Tracked or
                (track.state == TrackState.New and
                 track.tracklet_len >= self.cfg.confirm_frames)
            )
            if is_confirmed and track.last_detection:
                det_id = id(track.last_detection)
                confirmed_tracks[det_id] = track

        # 2. 遍历所有输入检测，构建输出
        for det in fused_objects:
            det_id = id(det)

            if det_id in confirmed_tracks:
                # 已确认轨迹：使用 track_id + Kalman 滤波位置
                track = confirmed_tracks[det_id]
                obj = {
                    'track_id': track.track_id,
                    'category': track.category,
                    'position': track.position,  # Kalman 滤波后的位置
                    'confidence': track.confidence,
                    'source': det.get('source', 'tracked'),
                    'quality': 'tracked',
                    'tracklet_len': track.tracklet_len,
                    '_original': det.get('_original'),
                    'bbox': det.get('bbox'),
                    'score': det.get('score'),
                }
            else:
                # 未确认检测：保持原样，track_id=None
                obj = {
                    'track_id': None,  # 无持久 ID
                    'category': det['category'],
                    'position': det['position'],  # 原始位置
                    'confidence': det['confidence'],
                    'source': det.get('source', 'unknown'),
                    'quality': det.get('quality', 'untracked'),
                    'tracklet_len': 0,
                    '_original': det.get('_original'),
                    'bbox': det.get('bbox'),
                    'score': det.get('score'),
                }

            output.append(obj)

        return output

    def reset(self):
        """重置跟踪器"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack3D.reset_id()

        if self.log:
            self.log.info("ByteTracker3D reset")


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ByteTracker3D 测试")
    print("=" * 70)

    # 创建跟踪器
    cfg = TrackerConfig(
        match_thresh=0.15,
        second_thresh=0.25,
        track_buffer=15,
        confirm_frames=2,
    )
    tracker = ByteTracker3D(cfg)

    # 模拟场景：3个物体，其中1个被遮挡
    print("\n--- 帧 1: 3个物体，都是fused ---")
    frame1 = [
        {'position': np.array([1.0, 0.0, 0.5]), 'category': 'bottle',
         'confidence': 0.9, 'source': 'fused', 'quality': 'EXCELLENT'},
        {'position': np.array([1.5, 0.2, 0.6]), 'category': 'cup',
         'confidence': 0.85, 'source': 'fused', 'quality': 'GOOD'},
        {'position': np.array([2.0, -0.1, 0.4]), 'category': 'box',
         'confidence': 0.88, 'source': 'fused', 'quality': 'GOOD'},
    ]
    result1 = tracker.update(frame1)
    print(f"跟踪结果: {len(result1)} 个轨迹")
    for obj in result1:
        print(f"  ID={obj['track_id']}, {obj['category']}, pos={obj['position']}")

    print("\n--- 帧 2: bottle被遮挡，只有chassis看到 ---")
    frame2 = [
        {'position': np.array([1.02, 0.01, 0.51]), 'category': 'bottle',
         'confidence': 0.7, 'source': 'chassis_only', 'quality': 'single_view'},  # 低置信度
        {'position': np.array([1.52, 0.21, 0.61]), 'category': 'cup',
         'confidence': 0.87, 'source': 'fused', 'quality': 'GOOD'},
        {'position': np.array([2.02, -0.09, 0.41]), 'category': 'box',
         'confidence': 0.89, 'source': 'fused', 'quality': 'GOOD'},
    ]
    result2 = tracker.update(frame2)
    print(f"跟踪结果: {len(result2)} 个轨迹")
    for obj in result2:
        print(f"  ID={obj['track_id']}, {obj['category']}, pos={obj['position']}")

    print("\n--- 帧 3: bottle完全遮挡（无检测） ---")
    frame3 = [
        {'position': np.array([1.54, 0.22, 0.62]), 'category': 'cup',
         'confidence': 0.86, 'source': 'fused', 'quality': 'GOOD'},
        {'position': np.array([2.04, -0.08, 0.42]), 'category': 'box',
         'confidence': 0.90, 'source': 'fused', 'quality': 'EXCELLENT'},
    ]
    result3 = tracker.update(frame3)
    print(f"跟踪结果: {len(result3)} 个轨迹")
    for obj in result3:
        print(f"  ID={obj['track_id']}, {obj['category']}, pos={obj['position']}")
    print(f"Lost轨迹: {len(tracker.lost_stracks)} 个")

    print("\n--- 帧 4: bottle重新出现（chassis_only），应该恢复 ---")
    frame4 = [
        {'position': np.array([1.06, 0.02, 0.52]), 'category': 'bottle',
         'confidence': 0.65, 'source': 'chassis_only', 'quality': 'single_view'},  # 低置信度恢复
        {'position': np.array([1.56, 0.23, 0.63]), 'category': 'cup',
         'confidence': 0.88, 'source': 'fused', 'quality': 'GOOD'},
        {'position': np.array([2.06, -0.07, 0.43]), 'category': 'box',
         'confidence': 0.91, 'source': 'fused', 'quality': 'EXCELLENT'},
    ]
    result4 = tracker.update(frame4)
    print(f"跟踪结果: {len(result4)} 个轨迹")
    for obj in result4:
        print(f"  ID={obj['track_id']}, {obj['category']}, pos={obj['position']}, len={obj['tracklet_len']}")

    print("\n" + "=" * 70)
    print("✅ ByteTracker3D 测试完成")
    print("   - 帧2: bottle用低置信度检测继续跟踪（第一阶段未匹配，第二阶段恢复）")
    print("   - 帧3: bottle完全丢失，进入Lost状态")
    print("   - 帧4: bottle用低置信度检测恢复（ByteTrack核心创新）")
    print("=" * 70)
