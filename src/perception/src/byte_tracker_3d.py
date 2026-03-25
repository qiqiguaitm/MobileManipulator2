#!/usr/bin/env python3
"""
ByteTracker3D - 静态世界模型下的 3D 目标跟踪器

设计假设：所有物体在 map 系下静止不动。
核心改动（相比传统 ByteTrack）：
1. StaticPositionEstimator 替代 KalmanFilter3D：无速度状态，有上限的递推均值
2. Track ID 长期保持：物体出视野后 track 保留 120s，回来即复用原 ID
3. Re-ID 严格类别匹配：防止抓取机器人搞混不同类型物体

保留的 ByteTrack 机制：
- 两阶段匹配（高/低置信度）
- 匈牙利算法全局最优匹配
- 年轻轨迹惩罚 + 类别投票

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
    match_thresh: float = 0.15      # 第一阶段匹配阈值 (15cm, map系下静态物体噪声~3cm)
    second_thresh: float = 0.25     # 第二阶段匹配阈值 (25cm, 恢复Lost轨迹)

    # 轨迹管理（静态世界：物体不动，track 需长期保留以支持 Re-ID）
    track_buffer_s: float = 120.0   # Lost轨迹保留时间（秒），120s 够机器人转几圈回来

    # 新轨迹确认：基于真实时间，不受 update() 调用频率影响
    # 物理含义：新检测必须持续观测 confirm_time_s 秒 + 至少 min_confirm_hits 次匹配才能成为稳定轨迹
    # 解决问题：遮挡边缘期 SAM3 产生的 ghost 检测在 0.4s 内就能被确认，
    #           导致 ghost 抢占正在恢复中的 Lost 老轨迹的 ID。
    confirm_time_s: float = 0.8     # 新轨迹确认所需最短观测时间（秒）
    min_confirm_hits: int = 2       # 新轨迹确认所需最少匹配次数（防单帧误检）

    # 年轻轨迹代价惩罚：在匈牙利匹配中给 tracklet_len 少的轨迹附加惩罚代价
    # 目的：防止 ghost 轨迹（tracklet_len=1）在位置恰好更近时抢走 Lost 老轨迹（tracklet_len=17）的检测
    # 公式：age_penalty = age_penalty_weight * (1 - min(tracklet_len/age_stable_frames, 1))
    age_penalty_weight: float = 0.15   # 年轻轨迹代价惩罚权重
    age_stable_frames: int = 10        # 达到此匹配次数后视为稳定（惩罚趋于 0）

    # Re-ID: 从 Removed 池恢复旧 ID（物体消失后重新出现）
    # 静态世界：物体位置稳定，Re-ID 可靠；类别要求严格相同（抓取机器人安全考虑）
    reid_thresh: float = 0.3        # Re-ID 距离阈值 (30cm)
    reid_buffer_s: float = 120.0    # Removed 轨迹保留时间（秒），与 track_buffer_s 对齐

    # 代价函数权重（归一化代价，与 dual_camera_matcher 同构）
    # cost = 0.85*dist_norm + 0.15*cat_cost + age_penalty
    # cat_cost: 0.0=相同 / 0.3=兼容 / 1.0=不兼容（软门控，不再硬拦截）
    # map 系下距离为主要判据，类别辅助区分近距离同位置目标
    cost_max: float = 0.75          # 归一化代价门槛

    # 类别投票：连续 N 帧检测到不同类别后更新轨迹类别（仅 Tracked 状态）
    # 防止轨迹出生时类别标错后一直锁死（如 can 被标成 box 后永远是 box）
    confirm_cat_frames: int = 5     # 连续 N 帧不同类别才更新（防止单帧误检）

    # 调试：每次成功匹配时打印代价分解（配合 _cc_track_debug.py --cost 使用）
    cost_debug: bool = False

    # 静态位置估计器参数
    position_n_max: int = 30        # 递推均值观测上限（30次 ≈ 6s@5Hz → σ≈0.55cm）
    reactivate_n: int = 10          # Lost→Tracked 重激活时 N 重置到此值，加速收敛


# ============================================================================
# 轨迹状态
# ============================================================================

class TrackState(Enum):
    """轨迹状态"""
    New = 0         # 新检测，待确认
    Tracked = 1     # 活跃跟踪中
    Lost = 2        # 暂时丢失（物体仍在，只是不在视野内）
    Removed = 3     # 已删除（可用于 Re-ID）


# ============================================================================
# 静态位置估计器
# ============================================================================

class StaticPositionEstimator:
    """
    静态物体位置估计器 — 有上限的递推均值

    设计假设：物体在 map 系下不动，所有观测都是 true_position + noise。

    行为：
      n < N_max 时：标准 running mean，每个观测权重 1/n
      n >= N_max 时：EMA，α = 1/N_max，老观测指数衰减

    特性：
      - 无速度状态 → 不会从噪声中拟合出伪速度
      - predict() 返回当前位置 → 匹配目标极稳定
      - N_max 防止过度锁定 → 允许定位微漂时慢慢修正
    """

    def __init__(self, position: np.ndarray, n_max: int = 30):
        self._position = np.array(position, dtype=np.float64)
        self.n = 1.0
        self.n_max = n_max

    def predict(self) -> np.ndarray:
        """静态物体预测：位置不变"""
        return self._position.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """用新观测更新位置估计"""
        z = np.asarray(z, dtype=np.float64)
        self.n = min(self.n + 1.0, self.n_max)
        alpha = 1.0 / self.n
        self._position = (1.0 - alpha) * self._position + alpha * z
        return self._position.copy()

    @property
    def position(self) -> np.ndarray:
        """当前最佳位置估计"""
        return self._position.copy()


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

        # 静态位置估计器（替代 KalmanFilter3D）
        self.estimator = StaticPositionEstimator(
            np.asarray(position), n_max=cfg.position_n_max)

        # 配置
        self.cfg = cfg

    @property
    def position(self) -> np.ndarray:
        """当前位置（递推均值估计）"""
        return self.estimator.position

    def predict(self):
        """静态预测：位置不变，仅为匹配提供稳定的参考点"""
        self.estimator.predict()

    def update(self, detection: dict, frame_id: int):
        """用检测结果更新轨迹"""
        # 位置更新（递推均值）
        self.estimator.update(detection['position'])

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
        # 降低 N 以加速对新观测的响应（Lost 期间定位可能微漂）
        self.estimator.n = min(self.estimator.n, self.cfg.reactivate_n)
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
    静态世界 3D 目标跟踪器

    基于 ByteTrack 两阶段匹配 + 静态位置估计器。
    专为 map 系下静态物体设计：物体不动，track ID 长期保持。

    核心特性:
    1. 匈牙利算法（全局最优）替代贪心匹配
    2. 静态位置估计（递推均值，无速度状态）
    3. 两阶段匹配（恢复遮挡轨迹）- ByteTrack核心
    4. Track ID 永续：出视野 120s 内回来自动复用原 ID
    5. remove_object() API：抓取后显式移除物体
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

        # ========== Step 1: 预测所有轨迹（静态: 位置不变）==========
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
          [COST] s=1 trk=track_3 tcat=box dcat=can src=fused
                 cost=0.320 d=0.183 c=0.127 age=0.010 dist=5.0cm
        """
        W_DIST = 0.85
        W_CAT  = 0.15

        dist = float(np.linalg.norm(track.position - det['position']))
        dist_norm = min(dist / thresh, 1.0)
        cat_cost = CategoryCompatibility.compute_category_cost(
            track.category, det['category'])

        d_term = W_DIST * dist_norm
        c_term = W_CAT * cat_cost

        stability = min(track.tracklet_len / self.cfg.age_stable_frames, 1.0)
        age_term = self.cfg.age_penalty_weight * (1.0 - stability)

        tid = track.track_id if track.track_id else '?'
        self.log.info(
            f'[COST] s={stage} trk={tid} tcat={track.category} '
            f'dcat={det["category"]} src={det.get("source", "?")} '
            f'cost={cost_total:.3f} d={d_term:.3f} c={c_term:.3f} age={age_term:.3f} '
            f'hits={track.tracklet_len} dist={dist * 100:.1f}cm'
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
        计算归一化代价矩阵（距离 + 类别）

        cost = 0.85*dist_norm + 0.15*cat_cost + age_penalty

        cat_cost: 0.0=相同 / 0.3=兼容 / 1.0=不兼容（软门控）
        最终由 cost_max=0.75 决定是否接受。

        map 系下静态物体帧间距离 ~0.03m，dist_norm ≈ 0.15，区分度极好。
        """
        W_DIST = 0.85
        W_CAT  = 0.15

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

                # 年轻轨迹惩罚
                stability = min(track.tracklet_len / self.cfg.age_stable_frames, 1.0)
                age_penalty = self.cfg.age_penalty_weight * (1.0 - stability)

                cost[i, j] = W_DIST * dist_norm + W_CAT * cat_cost + age_penalty

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

        静态世界策略：
        - 距离硬门控：> reid_thresh 直接跳过
        - 类别严格相同：抓取机器人安全要求，不同类物体不复用 ID

        Returns:
            旧 track_id (int) 如果匹配成功，否则 None
        """
        if not self.removed_stracks:
            return None

        best_dist = self.cfg.reid_thresh
        best_idx = -1

        det_pos = detection['position']
        det_cat = detection['category']

        for i, track in enumerate(self.removed_stracks):
            # 严格类别匹配：不同类别直接跳过
            if track.category != det_cat:
                continue
            dist = float(np.linalg.norm(track.position - det_pos))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            old_track = self.removed_stracks.pop(best_idx)
            # 清除 pool 中所有同位置的竞争轨迹（同一物体被重复创建的不同 ID）
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
        - 已确认轨迹：添加 track_id + 递推均值位置（稳定）
        - 未确认检测：保持原样（track_id=None）

        这样保证融合结果不丢失任何检测，同时为稳定跟踪的物体提供持久 ID。
        """
        output = []

        # 1. 收集所有已确认轨迹，建立检测ID到轨迹的映射
        confirmed_tracks = {}
        for track in self.tracked_stracks:
            if track.track_id is not None and track.last_detection:
                det_id = id(track.last_detection)
                confirmed_tracks[det_id] = track

        # 2. 遍历所有输入检测，构建输出
        for det in fused_objects:
            det_id = id(det)

            if det_id in confirmed_tracks:
                # 已确认轨迹：使用 track_id + 递推均值位置
                track = confirmed_tracks[det_id]
                obj = {
                    'track_id': f"track_{track.track_id}",
                    'category': track.category,
                    'position': track.position,  # 递推均值估计位置
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

    def remove_object(self, track_id) -> bool:
        """
        显式移除一个物体（如抓取成功后调用）。

        从所有轨迹池中彻底删除，不进入 Re-ID 池。

        Args:
            track_id: "track_5" (str) 或 5 (int)

        Returns:
            True 如果找到并移除，False 如果未找到
        """
        if isinstance(track_id, str) and track_id.startswith('track_'):
            try:
                tid = int(track_id.split('_')[1])
            except (IndexError, ValueError):
                return False
        elif isinstance(track_id, int):
            tid = track_id
        else:
            return False

        for pool in (self.tracked_stracks, self.lost_stracks, self.removed_stracks):
            for i, track in enumerate(pool):
                if track.track_id == tid:
                    pool.pop(i)
                    if self.log:
                        self.log.info(f'[REMOVE] track_{tid} removed (category={track.category})')
                    return True
        return False

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
    print("ByteTracker3D 测试（静态世界模型）")
    print("=" * 70)

    # 创建跟踪器
    cfg = TrackerConfig(
        match_thresh=0.15,
        second_thresh=0.25,
        track_buffer_s=120.0,
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
         'confidence': 0.7, 'source': 'chassis_only', 'quality': 'single_view'},
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
         'confidence': 0.65, 'source': 'chassis_only', 'quality': 'single_view'},
        {'position': np.array([1.56, 0.23, 0.63]), 'category': 'cup',
         'confidence': 0.88, 'source': 'fused', 'quality': 'GOOD'},
        {'position': np.array([2.06, -0.07, 0.43]), 'category': 'box',
         'confidence': 0.91, 'source': 'fused', 'quality': 'EXCELLENT'},
    ]
    result4 = tracker.update(frame4)
    print(f"跟踪结果: {len(result4)} 个轨迹")
    for obj in result4:
        print(f"  ID={obj['track_id']}, {obj['category']}, pos={obj['position']}, len={obj['tracklet_len']}")

    # 测试 remove_object
    print("\n--- 测试 remove_object ---")
    removed = tracker.remove_object('track_1')
    print(f"remove_object('track_1'): {removed}")
    print(f"Tracked: {len(tracker.tracked_stracks)}, Lost: {len(tracker.lost_stracks)}")

    print("\n" + "=" * 70)
    print("ByteTracker3D 测试完成（静态世界模型）")
    print("   - 帧2: bottle用低置信度检测继续跟踪")
    print("   - 帧3: bottle完全丢失，进入Lost状态（保留120s）")
    print("   - 帧4: bottle用低置信度检测恢复")
    print("   - remove_object: 抓取后显式移除")
    print("=" * 70)
