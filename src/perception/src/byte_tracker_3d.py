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

import time
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
    match_thresh: float = 0.20      # 第一阶段匹配阈值 (20cm, 适配2-3Hz低频)
    second_thresh: float = 0.30     # 第二阶段匹配阈值 (30cm, 更宽松)

    # 轨迹管理
    track_buffer_s: float = 3.0     # Lost轨迹保留时间（秒），与 update() 调用频率无关

    # 新轨迹确认：基于真实时间，不受 update() 调用频率影响
    # 物理含义：新检测必须持续观测 confirm_time_s 秒 + 至少 min_confirm_hits 次匹配才能成为稳定轨迹
    # 解决问题：遮挡边缘期 SAM3 产生的 ghost 检测在 0.4s 内就能被确认，
    #           导致 ghost 抢占正在恢复中的 Lost 老轨迹的 ID。
    confirm_time_s: float = 0.8     # 新轨迹确认所需最短观测时间（秒）
    min_confirm_hits: int = 2       # 新轨迹确认所需最少匹配次数（防单帧误检）

    # 年轻轨迹代价惩罚：在匈牙利匹配中给 tracklet_len 少的轨迹附加惩罚代价
    # 目的：防止 ghost 轨迹（tracklet_len=1）在位置恰好更近时抢走 Lost 老轨迹（tracklet_len=17）的检测
    # 公式：age_penalty = age_penalty_weight * (1 - min(tracklet_len/age_stable_frames, 1))
    # 效果示例（bottle 遮挡场景）：
    #   ghost (len=1):    stability=0.10, age_penalty = 0.15 * 0.90 = 0.135，总代价 0.003+0.135 = 0.138
    #   track_1 (len=17): stability=1.00, age_penalty = 0.15 * 0.00 = 0.000，总代价 0.045+0.000 = 0.045 ← 胜出
    #   注意: stability = min(tracklet_len/age_stable_frames, 1.0)，len≥age_stable_frames时上限截断为1.0
    age_penalty_weight: float = 0.15   # 年轻轨迹代价惩罚权重
    age_stable_frames: int = 10        # 达到此匹配次数后视为稳定（惩罚趋于 0）

    # Lost 轨迹速度衰减：每次 predict() 前将 Kalman 速度乘以 decay
    # 目的：遮挡边缘期深度腐蚀会给 Kalman 注入伪速度（如 z 漂移），
    #        Lost 期间 predict() 按伪速度外推 → 位置偏离真实位置 → 输给 ghost。
    # 衰减使 Lost 轨迹的预测位置收敛到 last-seen 位置附近，
    # 恢复后 Kalman 增益自动增大（P 在 Lost 期间增长），一次 update 即可修正。
    # 仅影响 Lost 状态，Tracked 状态完全不受影响。
    lost_velocity_decay: float = 0.7   # Lost 轨迹速度衰减因子（每次 predict 前乘以此值）

    # Re-ID: 从 Removed 池恢复旧 ID（物体消失后重新出现）
    # 注意: position 在 base_link 坐标系，机器人移动时 Re-ID 不可靠
    reid_thresh: float = 0.3        # Re-ID 距离阈值 (30cm)
    reid_buffer_s: float = 10.0     # Removed 轨迹保留时间（秒），与调用频率无关
    reid_cat_weight: float = 0.05   # Re-ID 类别软代价权重 (m)：同类=0, 兼容=+1.5cm, 不兼容=+5cm
                                    # 距离仍为硬门控，类别仅作排序偏好（防分类抖动导致 Re-ID 失败）

    # 代价函数权重（归一化代价，与 dual_camera_matcher 同构）
    # IoU可用时(Tracked+同相机): cost = 0.70*dist_norm + 0.15*(1-iou) + 0.15*cat_cost
    # IoU不可用时(Lost/跨相机): cost = 0.80*dist_norm + 0.20*cat_cost
    # cat_cost: 0.0=相同 / 0.3=兼容 / 1.0=不兼容（软门控，不再硬拦截）
    cost_max: float = 0.75          # 归一化代价门槛

    # 类别投票：连续 N 帧检测到不同类别后更新轨迹类别（仅 Tracked 状态）
    # 防止轨迹出生时类别标错后一直锁死（如 can 被标成 box 后永远是 box）
    confirm_cat_frames: int = 5     # 连续 N 帧不同类别才更新（防止单帧误检）

    # 调试：每次成功匹配时打印代价分解（配合 _cc_track_debug.py --cost 使用）
    cost_debug: bool = False

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
                 detection: dict = None, reuse_id: int = None):
        # ID 延迟分配：构造时不消耗 ID，确认时才分配
        # 噪声轨迹（观测时间不满 confirm_time_s）永远不会消耗 ID 号
        self.track_id = None
        self._reuse_id = reuse_id  # Re-ID 复用的旧 ID，在 activate() 时使用

        # 基本信息
        self.category = category
        self.confidence = confidence

        # 类别投票状态（连续不同类别匹配计数，达到阈值后更新类别）
        self._cat_vote_streak: int = 0   # 连续不同类别帧数
        self._cat_vote_pending: str = '' # 投票中的目标类别

        # 状态
        self.state = TrackState.New
        self.frame_id = 0           # 最后更新的帧号（用于调试和日志）
        self.start_frame = 0        # 开始帧号
        self.tracklet_len = 0       # 累计匹配帧数（只增不减，跨 Lost 也保留）
        self.first_seen_time: float = 0.0  # 轨迹第一次创建时的 wall-clock 时间（用于 confirm_time_s 判断）
        self.last_seen_time: float = 0.0   # 最后匹配时的 wall-clock 时间（time.monotonic，用于 TTL 判断）

        # 保存最后匹配的检测数据（包含 _original 等字段）
        self.last_detection = detection

        # 最后匹配帧的相机bbox（IoU计算用，按相机分开存储）
        # 仅在 Tracked 状态下更新，Lost 状态 bbox 不再可信
        self.last_chassis_bbox: Optional[List[float]] = None  # fused/chassis 检测的bbox
        self.last_top_bbox: Optional[List[float]] = None      # top 检测的bbox

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
        if self.state == TrackState.Lost:
            self.kalman.x[3:6] *= self.cfg.lost_velocity_decay
        self.kalman.predict()

    def update(self, detection: dict, frame_id: int):
        """用检测结果更新轨迹"""
        # Kalman修正
        self.kalman.update(detection['position'])

        # 更新属性
        self.confidence = detection['confidence']
        self.frame_id = frame_id
        self.tracklet_len += 1

        # 类别投票：连续 confirm_cat_frames 帧检测到不同类别后更新（仅 Tracked 状态）
        # 防止轨迹出生时标错类别后永远锁死，代价: cat_cost 惩罚累积 → 被新轨迹抢占
        if self.state == TrackState.Tracked:
            det_cat = detection.get('category', '')
            if det_cat and det_cat != self.category:
                if det_cat == self._cat_vote_pending:
                    self._cat_vote_streak += 1
                else:
                    # 新的不同类别，重新计票
                    self._cat_vote_pending = det_cat
                    self._cat_vote_streak = 1
                if self._cat_vote_streak >= self.cfg.confirm_cat_frames:
                    self.category = det_cat
                    self._cat_vote_streak = 0
                    self._cat_vote_pending = ''
            elif det_cat:
                # 类别一致，重置投票
                self._cat_vote_streak = 0
                self._cat_vote_pending = ''

        # 更新相机bbox（IoU 计算用，按相机来源分开存储）
        source = detection.get('source', '')
        bbox = detection.get('bbox')
        if bbox:
            if source in ('fused', 'chassis', 'chassis_only'):
                self.last_chassis_bbox = bbox
            elif source in ('top', 'top_only'):
                self.last_top_bbox = bbox

        # 保存最后匹配的检测数据（包含 _original 等字段）
        self.last_detection = detection

    def mark_lost(self):
        """标记为Lost"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """标记为Removed"""
        self.state = TrackState.Removed

    def activate(self, frame_id: int):
        """激活轨迹（New → Tracked），此时才分配 ID"""
        self.state = TrackState.Tracked
        self.start_frame = frame_id

        # 只在确认时分配 ID —— 噪声轨迹永远走不到这里
        if self.track_id is None:
            if self._reuse_id is not None:
                self.track_id = self._reuse_id
            else:
                STrack3D._count += 1
                self.track_id = STrack3D._count

    def _should_confirm(self, current_time: Optional[float]) -> bool:
        """
        判断此 New 轨迹是否已满足确认条件（基于真实时间 + 最少匹配次数）。

        条件：
          1. current_time - first_seen_time >= confirm_time_s （持续观测足够长）
          2. tracklet_len >= min_confirm_hits               （至少匹配过若干次）

        时间条件防止 ghost 在遮挡边缘期（0.4s 内）抢先确认；
        hit 条件防止单帧误检（e.g. double-call 第二次调用）立即确认。
        """
        if current_time is None or self.first_seen_time <= 0.0:
            return False
        elapsed = current_time - self.first_seen_time
        return (elapsed >= self.cfg.confirm_time_s and
                self.tracklet_len >= self.cfg.min_confirm_hits)

    def reactivate(self, detection: dict, frame_id: int,
                   current_time: Optional[float] = None):
        """重新激活轨迹（Lost → Tracked）"""
        self.update(detection, frame_id)
        self.state = TrackState.Tracked
        # 修复：New → Lost → reactivate 路径中 track_id 永远为 None 的 bug
        # 若已满足时间+次数确认条件，立即分配 ID
        if self.track_id is None and self._should_confirm(current_time):
            self.activate(frame_id)

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

        # 帧计数（调试/日志用，不再用于生命周期判断）
        self.frame_id = 0
        # 当前帧的 wall-clock 时间，由 update() 设置，供内部方法共用
        self._current_time: float = 0.0

        # 重置ID计数器
        STrack3D.reset_id()

    def update(self, fused_objects: List[dict],
               current_time: Optional[float] = None) -> List[dict]:
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
        self._current_time = current_time if current_time is not None else time.monotonic()

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
            self._match(all_stracks, high_dets, self.cfg.match_thresh, stage=1)

        # 更新匹配的轨迹
        for t_idx, d_idx in matches1:
            track = all_stracks[t_idx]
            det = high_dets[d_idx]
            old_cat = track.category  # 记录更新前类别（用于检测类别变化）

            if track.state == TrackState.Lost:
                # Lost → Tracked (恢复)
                track.reactivate(det, self.frame_id, self._current_time)
            else:
                # 正常更新
                track.update(det, self.frame_id)
                if track.state == TrackState.New:
                    if track._should_confirm(self._current_time):
                        track.activate(self.frame_id)
                else:
                    track.state = TrackState.Tracked

            # 延迟激活: 轨迹经历 New→Lost→reactivate 后 track_id 仍为 None
            # 此时 state=Tracked 但未分配 ID，需要在时间+次数达标后补发
            if track.track_id is None and track._should_confirm(self._current_time):
                track.activate(self.frame_id)

            # 更新 wall-clock 时间（用于基于时间的 TTL 判断）
            track.last_seen_time = self._current_time

            # 类别投票日志
            if self.log and track.category != old_cat:
                self.log.info(
                    f'[CAT_UPDATE] track_{track.track_id}: {old_cat} → {track.category} '
                    f'(after {self.cfg.confirm_cat_frames} consecutive votes)'
                )

        # ========== Step 3: 第二阶段匹配 (低置信度 vs 未匹配轨迹) ==========
        # ByteTrack核心: 用单相机检测恢复/继续跟踪
        # 注意: 这里匹配所有未匹配轨迹，不仅是Lost轨迹
        remaining_tracks = [all_stracks[i] for i in unmatched_track_indices]

        matches2 = []
        unmatched_low_det_indices = list(range(len(low_dets)))  # 默认：全部未匹配

        if remaining_tracks and low_dets:
            matches2, unmatched_remaining_indices, unmatched_low_det_indices = \
                self._match(remaining_tracks, low_dets, self.cfg.second_thresh, stage=2)

            # 更新/恢复匹配的轨迹
            for t_idx, d_idx in matches2:
                track = remaining_tracks[t_idx]
                det = low_dets[d_idx]
                old_cat = track.category  # 记录更新前类别

                if track.state == TrackState.Lost:
                    # Lost → Tracked (恢复)
                    track.reactivate(det, self.frame_id, self._current_time)
                    if self.log:
                        self.log.debug(f"Recovered Lost track {track.track_id} with low-conf detection")
                else:
                    # 用低置信度检测继续跟踪
                    track.update(det, self.frame_id)
                    if track.state == TrackState.New:
                        if track._should_confirm(self._current_time):
                            track.activate(self.frame_id)
                    else:
                        track.state = TrackState.Tracked
                    if self.log:
                        self.log.debug(f"Continued track {track.track_id} with low-conf detection")

                # 延迟激活（同 Step 2）
                if track.track_id is None and track._should_confirm(self._current_time):
                    track.activate(self.frame_id)

                # 更新 wall-clock 时间
                track.last_seen_time = self._current_time

                # 类别投票日志
                if self.log and track.category != old_cat:
                    self.log.info(
                        f'[CAT_UPDATE] track_{track.track_id}: {old_cat} → {track.category} '
                        f'(after {self.cfg.confirm_cat_frames} consecutive votes)'
                    )

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
                # 基于真实时间判断是否超时，不受 DOUBLE_CALL 等异常调用频率影响
                if self._current_time - track.last_seen_time > self.cfg.track_buffer_s:
                    track.mark_removed()
            elif track.state == TrackState.New:
                # 未确认的新轨迹也标记为Lost
                track.mark_lost()

        # ========== Step 5: 新轨迹 (未匹配的检测) ==========
        # 创建新轨迹前，先尝试 Re-ID（从 Removed 池复用旧 ID）
        all_unmatched = [(high_dets[i], 'fused') for i in unmatched_det_indices] + \
                        [(low_dets[i], 'single-cam') for i in unmatched_low_det_indices]

        for det, source_label in all_unmatched:
            reuse_id = self._try_reid(det)
            new_track = STrack3D(
                position=det['position'],
                category=det['category'],
                confidence=det['confidence'],
                cfg=self.cfg,
                detection=det,
                reuse_id=reuse_id,
            )
            new_track.frame_id = self.frame_id
            new_track.start_frame = self.frame_id
            new_track.tracklet_len = 1
            new_track.first_seen_time = self._current_time  # 记录出生时刻，用于 confirm_time_s 判断
            new_track.last_seen_time = self._current_time   # 用于 TTL（track_buffer_s）判断

            # Re-ID 轨迹立即激活：Re-ID 本身即代表"已确认"，无需再等 confirm_time_s
            # 修复 Re-ID 链断裂：激活后 track_id 立即分配，即使下一帧未被匹配
            # 也能以 track_id is not None 进入 removed_stracks，保持链的连续性
            if reuse_id is not None:
                new_track.activate(self.frame_id)

            self.tracked_stracks.append(new_track)

            if self.log:
                if reuse_id:
                    self.log.debug(
                        f"Re-ID track {new_track.track_id}: {det['category']} ({source_label})")
                else:
                    self.log.debug(
                        f"New track {new_track.track_id}: {det['category']} ({source_label})")

        # ========== Step 6: 更新轨迹池 ==========
        self._update_pools()

        # ========== 构建输出 ==========
        return self._get_output(fused_objects)

    def _match(self, tracks: List[STrack3D], detections: List[dict],
               thresh: float, stage: int = 1) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匈牙利算法匹配

        Args:
            tracks: 轨迹列表
            detections: 检测列表
            thresh: 距离阈值
            stage: 匹配阶段（1=高置信度，2=低置信度），用于代价日志

        Returns:
            matches: 匹配对 [(track_idx, det_idx), ...]
            unmatched_tracks: 未匹配轨迹索引
            unmatched_dets: 未匹配检测索引
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # 构建代价矩阵（归一化，dist 用 thresh 归一化，cat 为软权重）
        cost_matrix = self._compute_cost(tracks, detections, thresh)

        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 过滤超出归一化门槛的匹配
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.cfg.cost_max:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
                if self.log and self.cfg.cost_debug:
                    self._log_match_cost(tracks[r], detections[c], thresh,
                                         cost_matrix[r, c], stage)

        return matches, unmatched_tracks, unmatched_dets

    def _log_match_cost(self, track: 'STrack3D', det: dict,
                        thresh: float, cost_total: float, stage: int):
        """
        打印单次匹配的代价分解，供 _cc_track_debug.py --cost 解析。

        格式（key=value，便于脚本 split 解析）：
          [COST] s=1 mode=3T trk=track_3 tcat=box dcat=can src=fused
                 cost=0.320 d=0.183 i=0.010 c=0.127 dist=5.0 iou=0.96
        """
        W_DIST_3, W_IOU_3, W_CAT_3 = 0.70, 0.15, 0.15
        W_DIST_2, W_CAT_2 = 0.80, 0.20

        dist = float(np.linalg.norm(track.position - det['position']))
        dist_norm = min(dist / thresh, 1.0)
        cat_cost = CategoryCompatibility.compute_category_cost(
            track.category, det['category'])

        iou = 0.0
        use_iou = False
        if track.state == TrackState.Tracked:
            det_source = det.get('source', '')
            det_bbox = det.get('bbox')
            if det_source in ('fused', 'chassis', 'chassis_only'):
                track_bbox = track.last_chassis_bbox
            elif det_source in ('top', 'top_only'):
                track_bbox = track.last_top_bbox
            else:
                track_bbox = None
            if track_bbox is not None and det_bbox is not None:
                iou = self._compute_iou(track_bbox, det_bbox)
                use_iou = True

        if use_iou:
            d_term = W_DIST_3 * dist_norm
            i_term = W_IOU_3 * (1.0 - iou)
            c_term = W_CAT_3 * cat_cost
            mode = '3T'
        else:
            d_term = W_DIST_2 * dist_norm
            i_term = 0.0
            c_term = W_CAT_2 * cat_cost
            mode = 'FB'

        stability = min(track.tracklet_len / self.cfg.age_stable_frames, 1.0)
        age_term = self.cfg.age_penalty_weight * (1.0 - stability)

        tid = track.track_id if track.track_id else '?'
        self.log.info(
            f'[COST] s={stage} mode={mode} trk={tid} tcat={track.category} '
            f'dcat={det["category"]} src={det.get("source", "?")} '
            f'cost={cost_total:.3f} d={d_term:.3f} i={i_term:.3f} c={c_term:.3f} age={age_term:.3f} '
            f'hits={track.tracklet_len} dist={dist * 100:.1f} iou={iou:.2f}'
        )

    @staticmethod
    def _compute_iou(bbox_a: Optional[List[float]],
                     bbox_b: Optional[List[float]]) -> float:
        """
        计算两个bbox的IoU，格式 [x1, y1, x2, y2]。
        任意一个为空或面积为0则返回0。
        """
        if not bbox_a or not bbox_b:
            return 0.0
        x1a, y1a, x2a, y2a = bbox_a[:4]
        x1b, y1b, x2b, y2b = bbox_b[:4]
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        ix1 = max(x1a, x1b)
        iy1 = max(y1a, y1b)
        ix2 = min(x2a, x2b)
        iy2 = min(y2a, y2b)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _compute_cost(self, tracks: List[STrack3D],
                      detections: List[dict],
                      thresh: float) -> np.ndarray:
        """
        计算归一化代价矩阵（双分支）

        IoU可用时（Tracked状态 + 同相机bbox均有效）：
          cost = 0.70*dist_norm + 0.15*(1-iou) + 0.15*cat_cost
          优先级：距离 >> IoU = 类别（低频场景IoU不可靠，降权）

        IoU不可用时（Lost状态 / 跨相机 / bbox退化）：
          cost = 0.80*dist_norm + 0.20*cat_cost

        cat_cost: 0.0=相同 / 0.3=兼容 / 1.0=不兼容（软门控）
        最终由 cost_max=0.75 决定是否接受。
        """
        # IoU可用分支权重
        W_DIST_3 = 0.70
        W_IOU_3  = 0.15
        W_CAT_3  = 0.15
        # IoU不可用回退权重
        W_DIST_2 = 0.80
        W_CAT_2  = 0.20

        n_tracks = len(tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), 1e6, dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                dist = np.linalg.norm(track.position - det['position'])
                dist_norm = min(dist / thresh, 1.0)
                cat_cost = CategoryCompatibility.compute_category_cost(
                    track.category, det['category']
                )  # 0.0 / 0.3 / 1.0

                # IoU: 仅 Tracked 状态 + 同相机来源时可用
                use_iou = False
                iou = 0.0
                if track.state == TrackState.Tracked:
                    det_source = det.get('source', '')
                    det_bbox = det.get('bbox')
                    if det_source in ('fused', 'chassis', 'chassis_only'):
                        track_bbox = track.last_chassis_bbox
                    elif det_source in ('top', 'top_only'):
                        track_bbox = track.last_top_bbox
                    else:
                        track_bbox = None
                    if track_bbox is not None and det_bbox is not None:
                        iou = self._compute_iou(track_bbox, det_bbox)
                        use_iou = True

                # 年轻轨迹惩罚：tracklet_len 少的轨迹（ghost/噪声）附加代价，
                # 防止其在位置恰好更近时抢走 Lost 老轨迹的检测。
                # 公式：penalty = weight * (1 - min(len/stable_frames, 1))
                # tracklet_len=1:  penalty ≈ weight * 0.9  (最大惩罚)
                # tracklet_len=10: penalty = 0              (稳定轨迹，无惩罚)
                stability = min(track.tracklet_len / self.cfg.age_stable_frames, 1.0)
                age_penalty = self.cfg.age_penalty_weight * (1.0 - stability)

                if use_iou:
                    cost[i, j] = (W_DIST_3 * dist_norm
                                  + W_IOU_3  * (1.0 - iou)
                                  + W_CAT_3  * cat_cost
                                  + age_penalty)
                else:
                    cost[i, j] = W_DIST_2 * dist_norm + W_CAT_2 * cat_cost + age_penalty

        return cost

    def _update_pools(self):
        """更新轨迹池"""
        new_tracked = []
        new_lost = []

        for track in self.tracked_stracks + self.lost_stracks:
            if track.state == TrackState.Tracked:
                new_tracked.append(track)
            elif track.state == TrackState.Lost:
                new_lost.append(track)
            elif track.state == TrackState.New:
                new_tracked.append(track)
            elif track.state == TrackState.Removed:
                # 只保留已激活（track_id 已分配）的轨迹用于 Re-ID
                # - 普通轨迹：activate() 在 confirm_time_s 达标后触发，track_id 非 None
                # - Re-ID 轨迹：创建时立即 activate()，track_id 非 None（即使存活仅 1 帧）
                # - 噪声轨迹：从未 activate()，track_id = None，不保留
                if track.track_id is not None:
                    self.removed_stracks.append(track)

        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        # 清理过期的 Removed 轨迹（基于真实时间，与调用频率无关）
        self.removed_stracks = [
            t for t in self.removed_stracks
            if self._current_time - t.last_seen_time <= self.cfg.reid_buffer_s
        ]

    def _try_reid(self, detection: dict) -> Optional[int]:
        """
        尝试从 Removed 池中找到位置接近的旧轨迹，复用其 track_id。

        匹配策略：距离为硬门控（> reid_thresh 直接跳过），类别为软代价偏好。
        cost = dist + reid_cat_weight * cat_cost
        - 同类别：cost = dist（无惩罚）
        - 兼容类别：cost = dist + reid_cat_weight * 0.3
        - 不兼容：cost = dist + reid_cat_weight * 1.0
        分类器抖动（box→bottle）时距离足够近仍可 Re-ID，但同类候选优先。

        Returns:
            旧 track_id (int) 如果匹配成功，否则 None
        """
        if not self.removed_stracks:
            return None

        best_cost = float('inf')
        best_idx = -1

        det_pos = detection['position']
        det_cat = detection['category']

        for i, track in enumerate(self.removed_stracks):
            dist = float(np.linalg.norm(track.position - det_pos))
            if dist >= self.cfg.reid_thresh:
                continue  # 距离硬门控
            cat_cost = CategoryCompatibility.compute_category_cost(track.category, det_cat)
            cost = dist + self.cfg.reid_cat_weight * cat_cost
            if cost < best_cost:
                best_cost = cost
                best_idx = i

        if best_idx >= 0:
            old_track = self.removed_stracks.pop(best_idx)
            # 清除 pool 中所有同位置的竞争轨迹（同一物体被重复创建的不同 ID）
            # 根因：分类器抖动(box↔can)在 pool 里积累了 track_2(box)+track_5(can) 两个竞争者，
            # 之后检测类别决定复活哪个，导致振荡。清除同位置竞争者可彻底切断循环。
            self.removed_stracks = [
                t for t in self.removed_stracks
                if float(np.linalg.norm(t.position - det_pos)) >= self.cfg.reid_thresh
            ]
            return old_track.track_id

        return None

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
            if track.track_id is not None and track.last_detection:
                det_id = id(track.last_detection)
                confirmed_tracks[det_id] = track

        # 2. 遍历所有输入检测，构建输出
        for det in fused_objects:
            det_id = id(det)

            if det_id in confirmed_tracks:
                # 已确认轨迹：使用 track_id + Kalman 滤波位置
                track = confirmed_tracks[det_id]
                obj = {
                    'track_id': f"track_{track.track_id}",
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
        match_thresh=0.20,
        second_thresh=0.30,
        track_buffer_s=3.0,
        confirm_time_s=0.8,
        min_confirm_hits=2,
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
