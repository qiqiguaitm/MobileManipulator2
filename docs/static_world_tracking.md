# 静态世界 3D 目标跟踪系统

> ByteTracker3D — 为移动抓取机器人设计的 map 系静态物体跟踪器

## 1. 设计背景

### 1.1 问题

传统 ByteTrack 使用恒速模型 (CV) Kalman 滤波器，假设物体在帧间匀速运动。
但在 **map 坐标系下跟踪静态物体** 时，CV 模型有严重问题：

| 场景 | CV 模型行为 | 后果 |
|------|------------|------|
| 机器人静止 | 噪声 → 微小伪速度 → 位置微漂 | 轻微，可接受 |
| 机器人旋转 | TF 误差 ~4cm → 大伪速度 → 位置暴漂 | **致命：track 匹配失败 → ID 爆炸** |
| 机器人平移 | 深度噪声 ~2cm → 中等伪速度 | 中等，位置震荡 |

核心矛盾：**静态物体在 map 系下速度为零，但 CV 模型从噪声中拟合出非零速度**。

### 1.2 实测数据

使用 `_cc_kalman_observe.py` 采集，静止 ~10s → 旋转 ~10s → 停止 ~10s：

```
               静止期 std    旋转期 std    旋转期 range
Raw (无滤波)    ~0.4 cm      ~4.4 cm       ~20 cm
CV Kalman       ~0.3 cm      ~4.0 cm       ~18 cm   ← 几乎没帮助
CP Kalman       ~0.2 cm      ~0.8 cm       ~3 cm    ← 显著抑制
```

**结论**：CV Kalman 在旋转时基本无效，恒定位置 (CP) 模型才是正确选择。

## 2. 设计原则

```
核心假设：所有物体在 map 系下静止不动
```

由此推导出三个设计决策：

1. **无速度状态** → `StaticPositionEstimator`（递推均值，非 Kalman）
2. **Track ID 永续** → Lost 保留 120s，Re-ID 恢复旧 ID
3. **显式移除** → `remove_object()` API，抓取后清除

## 3. 核心组件

### 3.1 StaticPositionEstimator

替代 `KalmanFilter3D`，无速度状态，仅维护位置均值。

```python
class StaticPositionEstimator:
    def __init__(self, position, n_max=30):
        self._position = position   # 当前最佳估计
        self.n = 1.0               # 已累积观测数
        self.n_max = n_max         # 上限

    def predict(self):
        return self._position      # 静态：位置不变

    def update(self, z):
        self.n = min(self.n + 1, self.n_max)
        alpha = 1.0 / self.n
        self._position = (1 - alpha) * self._position + alpha * z
        return self._position
```

**行为特性**：

| 阶段 | n 值 | alpha | 行为 |
|------|------|-------|------|
| 初始 | 1→2 | 0.5 | 快速收敛 |
| 收敛中 | 2→30 | 1/n 递减 | 标准 running mean |
| 稳定态 | 30 (锁定) | 1/30 ≈ 0.033 | EMA，缓慢跟踪漂移 |

**为什么不用 Kalman**：

- Kalman 需要速度状态 → 静态物体速度恒为零 → Q_vel 是纯噪声
- Kalman 需要调 Q、R 矩阵 → 递推均值只需一个参数 N_max
- 数学上等价于 Q=0, R=σ²I 的 Kalman → 更简单直接

**N_max = 30 的选择依据**：

- 5Hz 感知 → 30 次 = 6 秒观测窗口
- σ_single ≈ 3cm → σ_mean = 3/√30 ≈ 0.55cm
- 足够平滑，又不会过度锁定（允许定位微漂修正）

### 3.2 Track 生命周期

```
          ┌──────── 确认 (confirm_time_s + min_confirm_hits) ────────┐
          │                                                          │
  New ────┼─→ Tracked ←──── reactivate ──── Lost ──── 超时 ──→ Removed
  (无ID)  │     (有ID)                       │                   │
          │                                  │     Re-ID         │
          └── 超时 → Lost → Removed          └───────────────────┘
```

**关键参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `confirm_time_s` | 0.8s | 新检测需持续观测 0.8s 才确认 |
| `min_confirm_hits` | 2 | 至少匹配 2 次（防单帧误检） |
| `track_buffer_s` | 120s | Lost 保留时间（够机器人转几圈回来） |
| `reid_buffer_s` | 120s | Removed 保留时间（与 track_buffer 对齐） |

**Confirm 机制的意义**：
- 遮挡边缘的 SAM3 ghost 检测通常在 0.4s 内消失
- 要求 0.8s + 2次 可过滤掉绝大多数 ghost
- 防止 ghost 抢占正在恢复的 Lost 老轨迹

### 3.3 Re-ID 策略

当物体离开视野后重新出现：

```
Lost (120s 内) ─── 直接匹配 ─── reactivate (保持原 ID)
       │
       └── 超时 ─→ Removed ─── Re-ID 匹配 ─── 新 track 复用旧 ID
```

**Re-ID 规则**（针对抓取机器人安全需求）：

1. **类别严格相同**：`track.category != det.category` → 直接跳过
   - 不使用"兼容"匹配（如 can 兼容 bottle）
   - 原因：抓取错误物体可能导致安全问题
2. **距离门控**：`dist < reid_thresh (30cm)`
3. **竞争清理**：Re-ID 成功后，清除同位置的其他 Removed 轨迹

### 3.4 Reactivate N-Reset

当 Lost 轨迹被重新激活时：

```python
def reactivate(self, detection, frame_id):
    self.estimator.n = min(self.estimator.n, reactivate_n)  # 10
    self.update(detection, frame_id)
```

**为什么需要 N-Reset**：
- Lost 期间机器人可能移动 → 定位微漂 → 旧估计位置有偏差
- 若保持 n=30，新观测只有 3.3% 权重 → 需要 ~30 帧才能修正
- Reset 到 n=10，新观测有 10% 权重 → ~10 帧即可收敛
- 不 reset 到 1：保留一部分历史信息，避免单帧噪声冲击

### 3.5 remove_object() API

```python
def remove_object(self, track_id: str | int) -> bool:
    """抓取成功后调用，从所有池中彻底删除，不进入 Re-ID 池"""
```

**使用场景**：
1. 机器人抓取物体 → 物体从世界中消失
2. 调用 `tracker.remove_object("track_5")`
3. 轨迹从 tracked/lost/removed 池中彻底删除
4. 该位置出现的新物体会获得新 ID（不会 Re-ID 到已被抓走的物体）

## 4. 匹配算法

保留 ByteTrack 两阶段匹配 + 匈牙利算法：

### 4.1 代价函数

```
cost = 0.85 × dist_norm + 0.15 × cat_cost + age_penalty
```

| 分量 | 权重 | 说明 |
|------|------|------|
| `dist_norm` | 0.85 | 欧氏距离 / thresh，截断到 [0,1] |
| `cat_cost` | 0.15 | 0.0=相同 / 0.3=兼容 / 1.0=不兼容 |
| `age_penalty` | ≤0.15 | 年轻轨迹惩罚，防 ghost 抢占老轨迹 |

### 4.2 两阶段匹配

```
第一阶段：高置信度检测 (fused) vs 所有轨迹 (Tracked + Lost)
         └── match_thresh = 15cm

第二阶段：低置信度检测 (single-cam) vs 第一阶段未匹配的轨迹
         └── second_thresh = 25cm
```

**为什么 15cm / 25cm**：
- 静态物体在 map 系下帧间噪声 ~3cm → 15cm = 5σ 余量
- 单相机深度噪声更大 ~8cm → 25cm = 3σ 余量

### 4.3 类别投票

防止轨迹出生时标错类别后锁死：

```
连续 5 帧检测到不同类别 → 更新轨迹类别
```

- 仅在 Tracked 状态生效
- 中断重新计票（防闪烁）

## 5. 配置参数总览

```python
@dataclass
class TrackerConfig:
    # 匹配阈值
    match_thresh: float = 0.15        # 第一阶段 (15cm)
    second_thresh: float = 0.25       # 第二阶段 (25cm)

    # 轨迹管理
    track_buffer_s: float = 120.0     # Lost 保留 120s
    confirm_time_s: float = 0.8       # 新轨迹确认时间
    min_confirm_hits: int = 2         # 新轨迹确认次数

    # 年轻轨迹惩罚
    age_penalty_weight: float = 0.15
    age_stable_frames: int = 10

    # Re-ID
    reid_thresh: float = 0.3          # Re-ID 距离 (30cm)
    reid_buffer_s: float = 120.0      # Removed 保留 120s

    # 代价函数
    cost_max: float = 0.75            # 归一化代价门槛

    # 类别投票
    confirm_cat_frames: int = 5

    # 静态位置估计器
    position_n_max: int = 30          # 递推均值上限
    reactivate_n: int = 10            # 重激活 N 重置值
```

## 6. 与原版 ByteTrack 的差异

| 维度 | 原版 ByteTrack (CV) | 静态世界版 |
|------|---------------------|-----------|
| 位置预测 | Kalman + 恒速模型 | 递推均值，位置不变 |
| 速度状态 | 有（vx, vy, vz） | **无** |
| track_buffer | ~30 帧 (~3s) | **120s** (基于真实时间) |
| Re-ID | 无 | **有**，严格类别匹配 |
| 显式移除 | 无 | **remove_object()** API |
| 重激活 | Kalman 继续 | **N-Reset** 加速收敛 |
| 生命周期判断 | 帧计数 | **真实时间** (time.monotonic) |

## 7. 抓取工作流

完整的 pick-and-place 流程：

```
1. 感知到桌上3个物体 → track_1(bottle), track_2(cup), track_3(box)
2. 决定抓取 track_1(bottle)
3. 机器人旋转移动到 track_1 位置
   - track_1 可能暂时离开视野 → Lost 状态
   - 120s 内回来 → 自动恢复为 track_1
4. 接近目标，精确对准
   - 递推均值位置 → σ ≈ 0.55cm → 足够精确
5. 抓取成功
   - 调用 tracker.remove_object("track_1")
   - track_1 从所有池中删除，不进入 Re-ID
6. 机器人移动到放置位置
   - 若 track_2, track_3 暂时离开视野 → Lost → 120s 内回来恢复
7. 放置完成，回到桌子附近
   - track_2, track_3 自动恢复原 ID → 继续抓取
```

## 8. 文件结构

```
src/perception/src/byte_tracker_3d.py    # 跟踪器实现
src/perception/msg/Object3D.msg          # 3D 物体消息定义
src/perception/src/multi_camera_perception_node.py  # 感知节点（调用跟踪器）
```

## 9. 已知限制与未来方向

### 已知限制

1. **旋转噪声未完全消除**：递推均值在旋转时仍接收噪声观测，只是不放大
   - 缓解：N_max=30 使旋转期新观测权重 ≤3.3%，影响有限
   - 根因：HDL localization TF 链在旋转时有 ~4cm 系统误差

2. **无观测质量区分**：所有观测等权重，不区分双相机融合 vs 单相机
   - V2 考虑：根据 odom ω 降低旋转期观测权重

3. **Re-ID 靠位置距离**：定位漂移 > 30cm 时 Re-ID 失效
   - 缓解：120s 超时足够覆盖大多数场景

### 未来方向

- **V2**: 接入 odom 获取 ω，旋转期降低观测权重或跳过更新
- **V2**: 观测质量权重（fused > single-cam）
- **V3**: 视觉特征辅助 Re-ID（不依赖位置距离）
