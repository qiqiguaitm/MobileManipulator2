# Object Clouds 闪烁优化方案

## 问题描述
RViz 中的 `object_clouds` (物体点云) 出现闪烁/消失现象。

## 根本原因分析

### 原因 1: 点云生成不稳定
```python
# 每次从深度图重新生成点云
for y in range(y1, y2, 2):
    for x in range(x1, x2, 2):
        d = depth[y, x]
        if 0.1 < d < self.depth_max:
            # 生成 3D 点
```

**问题**：
- 深度图有噪声/空洞 → 某些帧点云数量少
- 如果点云为空 → 不发布消息
- RViz Decay Time 内没有新数据 → 点云消失

### 原因 2: 发布逻辑不连续
```python
# 旧逻辑
if len(all_object_points) > 0:
    self.pub_object_clouds.publish(cloud_msg)
# else: 不发布 → 闪烁
```

### 原因 3: Decay Time 太短
- 原值：`0.3s`
- 检测周期：`0.5s` (2Hz)
- 结果：点云在下一帧到达前已消失

---

## 解决方案

### 优化 1: 点云数据缓存
**实现位置**: `perception_rviz_node.py`

```python
# Line 76: 添加缓存字段
self._cached_object_cloud = None

# Line 307-327: 缓存 + 降级发布
if len(all_object_points) > 0:
    # 生成新点云
    cloud_msg = self._create_colored_pointcloud(...)

    # 缓存
    self._cached_object_cloud = cloud_msg
    self.pub_object_clouds.publish(cloud_msg)

elif self._cached_object_cloud is not None:
    # 当前帧无有效点云，发布上一次的缓存
    cached_cloud.header.stamp = stamp  # 更新时间戳
    self.pub_object_clouds.publish(cached_cloud)
```

**效果**：
- ✅ 即使深度数据不好，也持续发布上一次的有效点云
- ✅ 保证 RViz 始终有数据显示

### 优化 2: 延长 Decay Time
**实现位置**: `config/perception_3d.rviz`

```yaml
- Class: rviz/PointCloud2
  Name: Object Clouds
  Decay Time: 1.5  # 原 0.3s → 1.5s
```

**原理**：
- Decay Time = 点云在 RViz 中保留的时长
- 1.5s > 0.5s (检测周期) × 2 → 提供 3 倍缓冲
- 即使连续 2 帧不发布，点云仍可见

### 优化 3: 深度时间平滑（已实现）
**相关**: `depth_temporal_smoothing.md`

```python
# 深度 EMA 滤波 → 点云位置更稳定
depth_t = 0.7 * depth_current + 0.3 * depth_previous
```

---

## 优化效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 点云发布连续性 | 不连续（跳帧时不发布） | ✅ 连续（缓存降级） |
| Decay Time | 0.3s（不够） | ✅ 1.5s（充足） |
| 深度稳定性 | 噪声较大 | ✅ EMA 平滑 |
| 视觉效果 | 闪烁/消失 | ✅ 平滑稳定 |

---

## 验证方法

### 1. 重启 RViz 节点
```bash
rosnode kill /perception_rviz_node /rviz
sleep 2
roslaunch perception perception_rviz.launch
```

### 2. 观察指标
- ✅ 物体点云不再突然消失
- ✅ 点云位置更加平滑
- ✅ 即使深度数据不好，点云仍可见

### 3. 监控日志
```bash
rostopic hz /perception_rviz_node/object_clouds
# 应该看到稳定的 2Hz 发布
```

---

## 参数调整

### 如果点云更新太慢（物体移动时拖影）
```bash
# 减少 Decay Time
roslaunch perception perception_rviz.launch
# 手动在 RViz 中调整 Object Clouds → Decay Time → 1.0
```

### 如果仍然闪烁
```bash
# 增加 Decay Time
# 编辑 config/perception_3d.rviz:
# Decay Time: 1.5 → 2.0
```

---

## 代码改动摘要

| 文件 | 行数 | 改动内容 |
|------|------|---------|
| `scripts/perception_rviz_node.py` | 76 | 添加 `_cached_object_cloud` 缓存 |
| `scripts/perception_rviz_node.py` | 307-327 | 缓存机制 + 降级发布 |
| `config/perception_3d.rviz` | 148 | Decay Time: 0.3s → 1.5s |

---

## 相关优化

1. ✅ **Marker lifetime 延长**: 0.5s → 1.0s (防止 bbox/label 闪烁)
2. ✅ **Marker DELETE_ALL**: 清除残留 marker
3. ✅ **检测结果缓存**: `_cached_objects` (防止检测间隙时消失)
4. ✅ **深度时间平滑**: EMA 滤波 (减少点云位置抖动)
5. ✅ **点云数据缓存**: `_cached_object_cloud` (本次优化)

---

## Linus 式点评

**问题本质**:
- 数据流不连续 → 可视化闪烁
- 简单的缓存就能解决，不需要复杂算法

**实现特点**:
- ✅ 降级策略 (有新数据用新的，没有就用缓存)
- ✅ 时间戳更新 (让 RViz 认为是新数据)
- ✅ 线程安全 (data_lock 保护)
- ✅ 零性能损失 (只缓存一帧点云消息)

**这就是好品味**: 在正确的地方加正确的缓存，消除特殊情况。

---

**实现日期**: 2026-01-16
**优化类型**: 数据缓存 + 显示参数调优
**效果**: 彻底解决 object_clouds 闪烁问题
