# Perception 模块性能优化报告

## 优化摘要

通过向量化优化和预热机制，感知模块性能提升 **2倍**，实时检测帧率从 **0.4 Hz** 提升至 **0.8 Hz**。

---

## 优化前后对比

### 启用 LiDAR

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **3D测量** | 1223 ms | **322 ms** | **-74%** ⚡️ |
| **总耗时** | 2277 ms | **1217 ms** | **-47%** ⚡️ |
| **理论最大帧率** | 0.42 Hz | **0.82 Hz** | **+95%** 🚀 |
| **推荐配置** | 0.34 Hz | **0.67 Hz** | **+97%** 🚀 |

### 禁用 LiDAR

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **3D测量** | 797 ms | **119 ms** | **-85%** ⚡️ |
| **总耗时** | 1720 ms | **835 ms** | **-51%** ⚡️ |
| **理论最大帧率** | 0.55 Hz | **1.13 Hz** | **+105%** 🚀 |
| **推荐配置** | 0.44 Hz | **0.90 Hz** | **+105%** 🚀 |

---

## 优化详情

### 1. 向量化优化（主要优化）

**问题：** `scene_perception_core.py` 中的 3D 点云生成使用 Python for 循环，极其慢。

**优化前代码：**
```python
points = []
for y, x in zip(ys, xs):
    d = depth[y, x]
    if lower_bound <= d <= upper_bound and self.DEPTH_MIN < d < self.DEPTH_MAX:
        X = (x - cx) * d / fx
        Y = (y - cy) * d / fy
        Z = d
        points.append([X, Y, Z])
points = np.array(points)
```

**优化后代码：**
```python
# 向量化提取深度
ds = depth[ys, xs]

# 向量化过滤
valid = (ds >= lower_bound) & (ds <= upper_bound) & \
        (ds > self.DEPTH_MIN) & (ds < self.DEPTH_MAX)

# 向量化反投影
xs_valid = xs[valid]
ys_valid = ys[valid]
ds_valid = ds[valid]

X = (xs_valid - cx) * ds_valid / fx
Y = (ys_valid - cy) * ds_valid / fy
Z = ds_valid

points = np.column_stack([X, Y, Z])
```

**效果：** 3D 测量速度提升 **3-5 倍**（797ms → 143ms）

---

### 2. 预热机制

**问题：** DINO-X 和 CDM 首次调用存在冷启动开销（GPU 模型加载、编译）。

**解决方案：**

在配置文件中添加预热参数：
```yaml
# 检测服务 (DINO-X)
detector_warmup: 2              # DINO-X 预热次数，0=禁用

# 深度优化服务 (CDM)
depth_optimizer_warmup: 2       # CDM 预热次数，0=禁用
```

节点启动时自动进行预热调用，避免首次检测时的冷启动开销。

---

### 3. 日志优化

将检测过程中的日志从 `rospy.loginfo()` 改为 `rospy.logdebug()`，减少日志 I/O 开销。

---

## 优化后时间分布

### 启用 LiDAR（平均 1217ms）

| 步骤 | 平均耗时 | 占比 | 说明 |
|------|----------|------|------|
| 数据同步 | 0 ms | 0% | message_filters 三路同步 |
| **DINO-X检测** | **321 ms** | **26.4%** | ← 第二瓶颈 |
| **深度优化 (CDM)** | **428 ms** | **35.2%** | ← 最大瓶颈 |
| 3D测量 | 322 ms | 26.5% | 相机+LiDAR 测量（已优化） |
| 其他 | 146 ms | 12.0% | 坐标变换、消息构建等 |

### 禁用 LiDAR（平均 835ms）

| 步骤 | 平均耗时 | 占比 | 说明 |
|------|----------|------|------|
| 数据同步 | 0 ms | 0% | 仅 RGB+Depth 同步 |
| **DINO-X检测** | **310 ms** | **37.1%** | ← 第二瓶颈 |
| **深度优化 (CDM)** | **406 ms** | **48.6%** | ← 最大瓶颈 |
| 3D测量 (相机) | 119 ms | 14.3% | 已优化 |

---

## 当前瓶颈分析

### 第一瓶颈：CDM 深度优化（~400ms, 占 38-49%）

**特点：**
- 深度学习模型推理
- RGB-D 融合去噪
- 波动较大（std: 44-117ms）

**优化方向：**
- 降低输入分辨率
- 使用更快的去噪算法
- 或完全禁用（适用于深度质量已足够的场景）

### 第二瓶颈：DINO-X 检测（~300ms, 占 27-34%）

**特点：**
- 深度学习推理（Vision Transformer）
- 相对稳定（std: 42-59ms）

**优化方向：**
- 降低输入分辨率（当前 640x480）
- 减少检测目标类别
- 使用模型量化/蒸馏

---

## 进一步优化潜力

| 优化方案 | 预期提升 | 难度 | 说明 |
|---------|---------|------|------|
| **禁用 CDM** | +40-50% | 低 | 节省 ~400ms，适用于深度质量足够的场景 |
| **降低 DINO-X 分辨率** | +20-30% | 中 | 640x480 → 320x240，可能影响检测精度 |
| **LiDAR 处理优化** | +10-15% | 中 | 向量化 LiDAR 投影和筛选 |
| **GPU 推理优化** | +10-20% | 高 | TensorRT/ONNX 优化，需服务端支持 |

---

## 推荐配置

### 生产环境（平衡性能与精度）

```yaml
# 启用 LiDAR，启用 CDM
auto_detect_rate: 0.67         # ~1.5秒/帧 (留20%余量)
detector_warmup: 2
depth_optimizer_warmup: 2
enable_depth_optimizer: true
enable_lidar: true
```

### 高性能模式（牺牲深度质量）

```yaml
# 禁用 CDM
auto_detect_rate: 1.0          # ~1秒/帧
detector_warmup: 2
enable_depth_optimizer: false  # 禁用深度优化，节省 400ms
enable_lidar: true
```

### 纯视觉模式（最快速度）

```yaml
# 禁用 LiDAR 和 CDM
auto_detect_rate: 1.5          # ~0.67秒/帧
detector_warmup: 2
enable_depth_optimizer: false
enable_lidar: false
```

---

## 压测工具

### 基础压测（20次调用）

```bash
# 启用 LiDAR
cd /home/agilex/MobileManipulator
bash src/perception/scripts/run_perception_benchmark.sh

# 禁用 LiDAR
bash src/perception/scripts/benchmark_no_lidar.sh
```

### 时间分布分析（10次调用 + 详细统计）

```bash
# 启用 LiDAR
bash src/perception/scripts/benchmark_timing.sh with-lidar

# 禁用 LiDAR
bash src/perception/scripts/benchmark_timing.sh no-lidar
```

---

## 优化记录

| 日期 | 优化内容 | 提升效果 |
|------|---------|---------|
| 2026-01-16 | 向量化 3D 点云生成 | 3D测量 -73%，总耗时 -52% |
| 2026-01-16 | 添加 DINO-X/CDM 预热 | 消除冷启动开销 |
| 2026-01-16 | 日志级别优化 | 减少 I/O 开销 |
| 2026-01-16 | **并行化 DINO-X + CDM** | **总耗时 -47%，帧率 +101%** ⚡️ |

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/scene_perception_core.py:134-159` | 向量化优化核心代码 |
| `config/scene_perception_3d.yaml:19,24` | 预热配置参数 |
| `scripts/benchmark_timing.sh` | 性能压测脚本 |
| `docs/performance_optimization.md` | 本文档 |

---

## 禁用 CDM 深度优化的性能测试

### 测试配置
```yaml
enable_depth_optimizer: false   # 禁用 CDM 深度优化
enable_lidar: false             # 禁用 LiDAR
detector_warmup: 2              # 启用 DINO-X 预热
```

### 测试结果（禁用 CDM + 禁用 LiDAR）

| 指标 | 启用 CDM | 禁用 CDM | 提升 |
|------|----------|----------|------|
| **总耗时** | 835 ms | **390 ms** | **-53%** ⚡️ |
| **理论最大帧率** | 1.13 Hz | **2.56 Hz** | **+127%** 🚀 |
| **推荐配置** | 0.90 Hz | **2.05 Hz** | **+128%** 🚀 |

### 时间分布（平均 390ms）

| 步骤 | 平均耗时 | 占比 | 说明 |
|------|----------|------|------|
| 数据同步 | 0 ms | 0% | 仅 RGB+Depth 同步 |
| **DINO-X检测** | **284 ms** | **73%** | ← 主要时间消耗 |
| 深度优化 (CDM) | 0 ms | 0% | 已禁用 |
| 3D测量 (相机) | 85 ms | 22% | 已向量化优化 |
| 其他 | 21 ms | 5% | 坐标变换、消息构建等 |

### 性能分析

**禁用 CDM 的优势**：
- ✅ 节省 ~400ms CDM 深度优化时间
- ✅ 理论帧率提升至 2.56 Hz（实时性提升 2.3 倍）
- ✅ 推荐配置可达 2.05 Hz（约 0.5秒/帧）

**禁用 CDM 的劣势**：
- ⚠️ 深度图质量依赖于相机原始深度（无去噪优化）
- ⚠️ 噪声、孔洞等深度缺陷可能影响测量精度
- ⚠️ 在深度质量较差的场景下，测量可能不够鲁棒

### 使用场景推荐

**适合禁用 CDM 的场景**：
- 相机深度质量已经足够好（如近距离、良好光照）
- 对实时性要求高（需要 2Hz+ 检测频率）
- 物体检测任务为主，对深度精度要求不高

**应该启用 CDM 的场景**：
- 相机深度质量较差（远距离、低光照、反光表面）
- 对 3D 测量精度要求高（如精确抓取）
- 可以容忍较低帧率（0.8-1Hz）

### 配置建议

**高性能模式（禁用 CDM）**:
```yaml
auto_detect_rate: 2.0           # ~0.5秒/帧
detector_warmup: 2
enable_depth_optimizer: false   # 禁用 CDM，节省 400ms
enable_lidar: false
```

**超高性能模式（仅 DINO-X）**:
```yaml
auto_detect_rate: 2.5           # ~0.4秒/帧
detector_warmup: 2
enable_depth_optimizer: false
enable_lidar: false
min_score: 0.3                  # 稍微提高阈值减少误检
```


---

## 并行化优化（2026-01-16）

### 优化内容

将 DINO-X 检测和 CDM 深度优化从串行改为并行执行。这两个服务调用没有数据依赖，可以同时进行。

### 实现方式

使用 Python `ThreadPoolExecutor` 并行调用两个服务：

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    # 同时提交两个任务
    future_detection = executor.submit(_timed_detection)
    future_depth = executor.submit(_timed_depth_optimize)
    
    # 等待两个任务完成
    detections, detection_time = future_detection.result()
    depth, depth_optimize_time = future_depth.result()
```

### 性能提升（纯视觉模式：CDM 启用，无 LiDAR）

| 模式 | 平均耗时 | 理论帧率 | 推荐配置 | vs串行 |
|------|---------|---------|---------|--------|
| **串行** | 835 ms | 1.13 Hz | 0.90 Hz | 基准 |
| **并行** | 440 ms | 2.27 Hz | 1.82 Hz | **+101%** 🚀 |

**性能提升**: 47% (节省 395ms)

### 并行效率分析

**串行执行时间**:
- DINO-X: 310ms
- CDM: 406ms
- 总计: 716ms

**并行执行时间**:
- DINO-X: 358ms（并行执行）
- CDM: 318ms（并行执行）
- 实际墙上时间: 359ms ≈ max(358, 318)
- 理论节省: min(358, 318) = 318ms
- 实际节省: (358 + 318) - 359 = 317ms
- **并行效率**: 99.7% (接近理想)

### 时间分布对比

#### 串行模式（835ms）
```
数据同步:    0 ms
DINO-X:    310 ms  ┐
                   │ 串行执行
CDM:       406 ms  ┘ 716ms
3D测量:    119 ms
─────────────────
总计:      835 ms
```

#### 并行模式（440ms）
```
数据同步:    0 ms
DINO-X:    358 ms ┐
                  ├─ 并行执行: 359ms
CDM:       318 ms ┘  (节省 317ms!)
3D测量:     90 ms
─────────────────
总计:      447 ms
```

### 压测结果（10次调用）

- 平均耗时: 440ms (±41ms)
- 中位数: 440ms
- 最小: 371ms
- 最大: 519ms
- 标准差: 41ms
- 理论最大帧率: 2.27 Hz
- 推荐配置: 1.82 Hz (留20%余量)

### 完整优化历程对比

| 阶段 | 优化内容 | 耗时 | 帧率 | 累计提升 |
|------|---------|------|------|---------|
| 原始版本 | - | 1720ms | 0.55 Hz | 基准 |
| 向量化 + 预热 | 3D测量优化 + 服务预热 | 835ms | 1.13 Hz | +105% |
| **并行化** | **DINO-X + CDM 并行** | **440ms** | **2.27 Hz** | **+313%** |

### 推荐配置更新

#### 实时跟踪 + 深度优化（新推荐）
```yaml
auto_detect_rate: 1.8           # ~0.55秒/帧
enable_depth_optimizer: true    # 启用 CDM 深度优化
enable_lidar: false
detector_warmup: 2
depth_optimizer_warmup: 2
```

**优势**:
- 保留 CDM 深度优化，确保深度质量
- 帧率提升至 1.8 Hz，接近实时
- 适合大多数场景

#### 极限性能模式（禁用 CDM）
```yaml
auto_detect_rate: 2.5           # ~0.4秒/帧
enable_depth_optimizer: false   # 禁用 CDM
enable_lidar: false
detector_warmup: 2
```

**优势**:
- 最快速度：2.5 Hz
- 适合深度质量足够好的场景

### 相关文件

- `src/scene_perception_3d_node.py:340-368` - 并行调用实现
- `src/scene_perception_3d_node.py:309-317` - 并行时间统计显示

