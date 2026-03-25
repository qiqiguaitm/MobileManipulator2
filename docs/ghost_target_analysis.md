# 感知跟踪幽灵目标问题分析与修复

> 文档日期：2026-03-25
> 涉及模块：dual_camera_matcher.py / byte_tracker_3d.py / multi_camera_perception_node.py

---

## 1. 问题现象

机器人在巡检/抓取过程中，同一个真实物体在跟踪输出中出现两个 track（幽灵目标），表现为：
- 同一位置附近出现两个不同 ID 的跟踪目标（距离 < 30cm）
- 物体 ID 频繁跳变（旧 ID 消失、新 ID 出现）
- 被抓取的物体类别与目标池记录不一致

---

## 2. 根因分析

### 2.1 核心问题：双相机对同一物体识别类别不同

Chassis 相机（前方平视）和 Top 相机（上方俯视）视角、光照不同，SAM3 分类器对同一物体经常给出不同标签——典型场景：Chassis 识别为 `can`，Top 识别为 `box`。

**旧代码处理方式（致命缺陷）**：

```python
# 旧版 byte_tracker_3d.py _compute_cost()
for i, track in enumerate(tracks):
    for j, det in enumerate(detections):
        # 类别不兼容 → 直接跳过，代价保持 1e6（无穷大）
        if not CategoryCompatibility.is_compatible(
            track.category, det['category']
        ):
            continue  # ← 硬拦截！类别不同就永远无法匹配

        dist = np.linalg.norm(track.position - det['position'])
        cost[i, j] = dist
```

同时，旧版的类别兼容性表也太窄：

```python
# 旧版 SIMILAR_CATEGORIES — can 和 box 不在 bottle 的兼容集里
SIMILAR_CATEGORIES = {
    'bottle': {'bottle', 'cup', 'glass', 'container'},     # ← 没有 can、box
    'cup':    {'cup', 'bottle', 'glass', 'mug'},            # ← 没有 can
    'box':    {'box', 'package', 'carton'},                  # ← 没有 can、bottle
}
```

**后果**：can 和 bottle 不兼容、can 和 box 不兼容。当两个相机对同一物体识别出这些类别时：

1. **匹配层**（DualCameraMatcher）：同一物体被判为两个独立检测，各自输出 `chassis_only` 和 `top_only`
2. **跟踪层**（ByteTracker3D）：类别不兼容 → 代价 1e6 → 匈牙利匹配永远不会把它们关联到同一 track
3. **结果**：同一物体产生两个 track ID —— 幽灵目标

### 2.2 加剧因素：新轨迹确认太快

旧版确认逻辑仅需 `confirm_frames=2`（约 0.4s@5Hz），ghost 检测很容易通过确认门槛成为正式 track。

```python
# 旧版 — 2帧即确认
confirm_frames: int = 2
```

### 2.3 辅助因素：双相机位置结算偏差

两个相机深度估计路径不同，对同一物体的 3D 位置估算在 Z 轴方向存在系统偏差（近距约 47mm）。旧版用完整 3D 欧氏距离做匹配，Z 轴偏差放大了两个检测的距离，进一步恶化匹配成功率。

---

## 3. 修复方案

### 3.1 扩展类别兼容性表：把 can、box 加入互兼容（dual_camera_matcher.py）

**改动**：将所有可抓取小物体（bottle/cup/can/box）互相加入兼容集。

```python
# 修复后 — 所有可抓取小物体互相兼容
SIMILAR_CATEGORIES = {
    'bottle': {'bottle', 'cup', 'glass', 'container', 'can', 'box'},  # ← 新增 can, box
    'cup':    {'cup', 'bottle', 'glass', 'mug', 'can'},               # ← 新增 can
    'can':    {'can', 'bottle', 'cup', 'box', 'container'},           # ← 全新条目
    'box':    {'box', 'package', 'carton', 'can', 'bottle'},          # ← 新增 can, bottle
}
```

这样 Chassis 识别 `can`、Top 识别 `box` 时，两者属于"兼容"类别，不再被硬拦截。

### 3.2 类别代价从硬拦截改为软惩罚（byte_tracker_3d.py）

**改动**：不兼容不再设为 1e6 跳过，改为归一化代价函数，类别分数按三级给出——**兼容类别不置零，保留 0.3 惩罚**。

```python
# 修复后 — 类别代价三级软门控
# cat_cost: 0.0 = 完全相同  /  0.3 = 兼容  /  1.0 = 不兼容
cost = 0.85 * dist_norm + 0.15 * cat_cost + age_penalty
```

**关键设计**：兼容类别的 `cat_cost=0.3`，**而非 0.0**。这意味着：
- 同类别匹配仍然被优先选择（代价更低）
- 兼容类别匹配被允许，但有额外惩罚（0.3 × 0.15 = 0.045 的代价增量）
- 完全不兼容仍然被高惩罚（1.0 × 0.15 = 0.15），虽然不是硬拦截但代价极高

**对比旧版**：

| 情况 | 旧版代价 | 新版代价 |
|------|---------|---------|
| 同类别 (can↔can) | dist | 0.85×dist_norm + 0.0 |
| 兼容类别 (can↔box) | **1e6（硬拦截）** | 0.85×dist_norm + **0.045** |
| 不兼容 (can↔person) | **1e6（硬拦截）** | 0.85×dist_norm + **0.15** |

### 3.3 增加多帧观测确认限制（byte_tracker_3d.py）

**改动**：从 `confirm_frames=2` 改为**时间 + 帧数双重门控**。

```python
# 修复后 — 必须持续观测 0.8s 且至少匹配 4 次
confirm_time_s: float = 0.8     # 新轨迹确认所需最短观测时间（秒）
min_confirm_hits: int = 4       # 新轨迹确认所需最少匹配次数（节点配置）

def _should_confirm(self, current_time):
    elapsed = current_time - self.first_seen_time
    return (elapsed >= self.cfg.confirm_time_s and
            self.tracklet_len >= self.cfg.min_confirm_hits)
```

**效果**：Ghost 检测在遮挡边缘期短暂出现（< 0.4s），无法通过 0.8s 的时间门控，永远不会成为正式 track。真实物体持续可见，轻松通过。

### 3.4 2D 距离代替 3D 距离（dual_camera_matcher.py）

**改动**：匹配代价函数中距离计算仅使用 XY 平面距离，忽略 Z 轴。

```python
# 旧版 — 3D 距离（Z 轴偏差 ~47mm 恶化匹配）
dist_3d = np.linalg.norm(det_chassis.position_3d - det_top.position_3d)

# 修复后 — 2D 距离（仅 XY，消除 Z 轴系统偏差）
dist_3d = np.linalg.norm(det_chassis.position_3d[:2] - det_top.position_3d[:2])
```

**效果**：消除 Z 轴系统偏差。实测两相机 XY 偏差仅 ~10mm，在 20cm 匹配阈值内可靠匹配。

---

## 4. 修复前后对比

### 具体场景：一个易拉罐被两个相机看到

| | 旧版 | 修复后 |
|---|---|---|
| Chassis 识别 | `can` | `can` |
| Top 识别 | `box` | `box` |
| 类别兼容? | **不兼容**（can 不在 box 的集合里） | **兼容**（can 在 box 的兼容集里） |
| 匹配层结果 | 两个独立检测输出 | **匹配成功**，输出一个 fused 检测 |
| 跟踪层代价 | **1e6（硬拦截，无法匹配）** | 0.85×dist_norm + 0.15×0.3 = **正常代价** |
| track 数量 | **2个（幽灵目标）** | **1个** |
| 确认速度 | 2帧（0.4s）即确认 | 需 0.8s + 4帧才确认 |

---

## 5. 辅助修复措施

以上三点是核心修复。以下是同期做的辅助改进：

| 改进 | 文件 | 说明 |
|------|------|------|
| StaticPositionEstimator | byte_tracker_3d.py | 替代 KalmanFilter3D，消除速度外推导致 Lost 轨迹漂移 |
| 年轻轨迹惩罚 (age_penalty) | byte_tracker_3d.py | 新轨迹在匈牙利匹配中附加额外代价，防止 ghost 抢占老轨迹 |
| 类别投票 (5帧) | byte_tracker_3d.py | 连续5帧不同类别才更新 track 类别，防止类别锁死 |
| ID 延迟分配 | byte_tracker_3d.py | 未确认的轨迹不分配 ID，避免 ghost 消耗 ID 号 |
| Track buffer 改时间 | byte_tracker_3d.py | 从帧数(15帧)改为时间(120s)，与 update() 调用频率解耦 |
| Re-ID 严格类别匹配 | byte_tracker_3d.py | 不同类别的物体不复用 ID（抓取安全要求） |
| TF 精确时间戳查询 | multi_camera_perception_node.py | 用 image_stamp 精确插值查 TF，消除双相机坐标偏移 |
| 融合位置选择 | dual_camera_matcher.py | 近距(<1.4m)用 Chassis、远距用 Top，替代简单平均 |

---

## 6. 关键代码位置

| 文件 | 关键函数/类 | 说明 |
|------|------------|------|
| `src/perception/src/dual_camera_matcher.py:82` | `SIMILAR_CATEGORIES` | 类别兼容性表（新增 can、box 互兼容） |
| `src/perception/src/dual_camera_matcher.py:109` | `compute_category_cost()` | 三级代价：0.0/0.3/1.0 |
| `src/perception/src/dual_camera_matcher.py:194` | `CostFunction.compute()` | 2D 距离计算（`[:2]`） |
| `src/perception/src/byte_tracker_3d.py:615` | `_compute_cost()` | 跟踪层代价函数：0.85×dist + 0.15×cat + age |
| `src/perception/src/byte_tracker_3d.py:256` | `_should_confirm()` | 多帧观测确认：0.8s + 4次 |

---

## 7. 架构示意

```
[Chassis Camera] → SAM3 → "can"  ──┐
                                     │  DualCameraMatcher
                                     │  (2D距离 + 类别兼容性: can↔box → cost=0.3, 非硬拦截)
[Top Camera]     → SAM3 → "box"  ──┘
                                     │
                              ┌──────┴──────┐
                              │             │
                           fused        single_only
                         (匹配成功)    (真正的独立检测)
                              │             │
                              ▼             ▼
                        ByteTracker3D.update()
                        ┌───────────────────────────────┐
                        │ 代价 = 0.85×dist + 0.15×cat   │
                        │       + age_penalty           │
                        │                               │
                        │ cat_cost: 0.0/0.3/1.0         │
                        │ (兼容类别 ≠ 0，保留0.3惩罚)    │
                        │                               │
                        │ 确认: ≥0.8s 且 ≥4次匹配       │
                        └───────────────────────────────┘
                                     │
                                     ▼
                          Tracked Objects (stable ID)
```
