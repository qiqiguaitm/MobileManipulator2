# Perception 3D 设计架构审查报告

> 审查日期: 2026-01-16
> 审查角色: 机器人架构专家 / 3D感知系统专家

---

## 一、架构总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 模块化 | ✅ 良好 | SyncedSensorSubscriber / PerceptionCore / PerceptionNode 职责分离清晰 |
| 可复用性 | ✅ 良好 | 复用现有 CoordinateTransformer、percept.py |
| 可配置性 | ✅ 良好 | YAML 配置化，参数可调 |
| 容错设计 | ⚠️ 部分 | CDM 降级有，但其他故障处理缺失 |
| 时序正确性 | ❌ 有问题 | 数据新鲜度、同步策略存在风险 |

---

## 二、关键架构问题

### 问题 1: CoordinateTransformer 未加载 base_link 相关变换 [严重]

**现状**:
```python
# coordinate_transformer.py:93-97 只加载了这些变换
extrinsics_files = {
    'rslidar_to_optical': '...',
    'arm_to_rslidar': '...',
    'arm_to_optical': '...',
}
```

**问题**:
- 设计要求默认输出 `base_link` 坐标系
- 但 CoordinateTransformer 没有加载 `optical_to_base` 变换
- 外参文件 `extrinsics_top_camera_optical_frame_to_base_link.yaml` **存在但未被使用**

**影响**: 配置 `target_frame: base_link` 时会运行失败

**修复方案**:
```python
extrinsics_files = {
    'rslidar_to_optical': '...',
    'arm_to_rslidar': '...',
    'arm_to_optical': '...',
    'optical_to_base': 'extrinsics_top_camera_optical_frame_to_base_link.yaml',  # 新增
    'rslidar_to_base': 'extrinsics_rslidar_to_base_link.yaml',  # 新增
}
```

---

### 问题 2: Service 调用与数据同步的时序问题 [严重]

**现状设计**:
```
SyncedSensorSubscriber 持续同步 → 缓存 _latest_data
                                        ↓
Service 调用时 ─────────────────► 读取 _latest_data
```

**问题**:
1. **数据可能过时**: Service 调用时获取的是上一次同步成功的数据，可能是几百毫秒前的
2. **无新鲜度检查**: 没有检查 `_latest_data.timestamp` 是否在合理范围内
3. **同步失败静默**: 如果 RGB/Depth/LiDAR 任一数据源丢帧，缓存数据不更新，但调用方不知道

**场景举例**:
```
t=0.0s: 同步成功，缓存数据 (timestamp=0.0)
t=0.1s: LiDAR 丢帧，同步失败
t=0.2s: LiDAR 丢帧，同步失败
t=0.3s: Service 调用 → 获取 t=0.0s 的过时数据！
```

**修复方案**:
```python
def get_synced_data(self, timeout=5.0, max_age=0.5) -> dict:
    """获取同步数据，带新鲜度检查"""
    start = time.time()
    while time.time() - start < timeout:
        with self._data_lock:
            if self._latest_data is not None:
                age = (rospy.Time.now() - self._latest_data['timestamp']).to_sec()
                if age < max_age:
                    return self._latest_data.copy()
        time.sleep(0.01)
    raise TimeoutError(f"无法获取新鲜数据 (max_age={max_age}s)")
```

---

### 问题 3: enable_lidar=false 时的同步策略 [中等]

**现状设计**:
- `SyncedSensorSubscriber` 始终同步 RGB + Depth + LiDAR 三路数据
- `ApproximateTimeSynchronizer` 需要三路数据都到达才触发回调

**问题**:
- 当 `enable_lidar=false` 或 LiDAR 不可用时，三路同步永远不会成功
- 导致 `_latest_data` 永远为 None

**修复方案**: 分层同步策略
```python
class SyncedSensorSubscriber:
    def __init__(self, ..., enable_lidar=True):
        self.enable_lidar = enable_lidar

    def connect(self):
        if self.enable_lidar:
            # 三路同步
            self._sync = ApproximateTimeSynchronizer(
                [color_sub, depth_sub, lidar_sub], ...)
        else:
            # 仅相机同步
            self._sync = ApproximateTimeSynchronizer(
                [color_sub, depth_sub], ...)
```

---

### 问题 4: 消息定义与配置矛盾 [低]

**Object3D.msg 注释**:
```
# 相机测量 (arm_base_link 坐标系)  ← 写死了 arm_base_link
geometry_msgs/Point position
```

**配置文件**:
```yaml
target_frame: base_link  # 默认是 base_link，不是 arm_base_link
```

**修复**: 修改 msg 注释为 "目标坐标系 (由 target_frame 配置决定)"

---

### 问题 5: Service 超时和阻塞 [中等]

**延迟估算**:
| 步骤 | 耗时 |
|------|------|
| 数据采集 | ~10ms |
| DINO-X 检测 (网络) | 200-500ms |
| CDM 深度优化 (网络) | 50-100ms |
| 3D 测量 | ~10ms |
| **总计** | **270-620ms** |

**问题**:
- ROS Service 是阻塞调用
- 如果调用方设置了短超时，可能频繁失败
- 如果多个客户端并发调用，请求会排队

**建议**:
1. 在文档中明确说明预期响应时间 (~500ms)
2. 考虑 Service 内部超时机制
3. 或改用 Action 接口（支持反馈和取消）

---

### 问题 6: 外部服务可用性检查缺失 [中等]

**依赖的外部服务**:
- DINO-X: http://192.168.112.14:10086
- CDM: http://192.168.112.14:8086

**问题**: 节点启动时不检查服务可用性，可能运行时才发现服务不可用

**建议**: 添加启动时健康检查
```python
def __init__(self):
    # 检查 DINO-X 服务
    if not self._check_service(self.detector_url):
        rospy.logwarn("DINO-X 服务不可用，检测功能将失败")

    # CDM 服务可选，失败时降级
    if self.enable_depth_optimizer and not self._check_service(self.cdm_url):
        rospy.logwarn("CDM 服务不可用，禁用深度优化")
        self.enable_depth_optimizer = False
```

---

## 三、数据流图问题

**当前设计图问题**:
```
/lidar/chassis/point_cloud ──────────────────────────────┼◄────────────────────────────┘
```

LiDAR 数据在图中显示为独立路径，但代码设计中 LiDAR 是 `SyncedSensorSubscriber` 同步订阅的一部分。**图和代码描述不一致**。

**修正后的数据流**:
```
/camera/top/color/image_raw ─────┐
/camera/top/aligned_depth... ────┼──► SyncedSensorSubscriber ──► synced_data
/lidar/chassis/point_cloud ──────┘     (三路同步)                    │
                                                                     │
                                              ┌──────────────────────┤
                                              │                      │
                                              ▼                      ▼
                                       DinoXDetector          CDM 深度优化
                                       (RGB only)             (RGB + Depth)
                                              │                      │
                                         bbox + mask          optimized_depth
                                              │                      │
                                              └──────────┬───────────┘
                                                         │
                                                         ▼
                                              PerceptionCore.measure_*()
                                              (depth, mask, lidar_points)
                                                         │
                                                         ▼
                                              CoordinateTransformer
                                              (optical → target_frame)
```

---

## 四、待确认事项

### 4.1 传感器参数确认

| 参数 | 设计值 | 需确认 |
|------|--------|--------|
| 相机帧率 | 15fps | 实际运行是否稳定？ |
| LiDAR 帧率 | ? | 未在文档中说明，需确认 |
| 同步容差 | 100ms | 是否过大？相机帧间隔 66ms |
| 图像分辨率 | 640x480 | 是否可能变化？ |

### 4.2 外部服务确认

| 服务 | URL | 需确认 |
|------|-----|--------|
| DINO-X | http://192.168.112.14:10086 | 是否始终可用？超时设置？ |
| CDM | http://192.168.112.14:8086 | 实际延迟是多少？ |

### 4.3 坐标系确认

| 问题 | 说明 |
|------|------|
| LiDAR 测量的坐标系变换链 | rslidar → optical → target_frame？还是 rslidar → target_frame 直接？ |
| measure_lidar_guided 的投影 | 使用 rslidar_to_optical + project_to_image？确认变换链正确性 |

### 4.4 错误处理确认

| 场景 | 当前处理 | 需确认 |
|------|----------|--------|
| 相机数据丢失 | ? | Service 应返回什么？ |
| LiDAR 数据为空 | ? | lidar_result 如何处理？ |
| DINO-X 检测失败 | ? | 返回空结果还是错误？ |
| 检测到 0 个物体 | ? | success=true + 空数组？ |

---

## 五、修改建议清单

### P0 (阻塞性问题，必须修复)

1. [ ] **CoordinateTransformer 加载 base_link 相关变换**
   - 添加 `optical_to_base`, `rslidar_to_base` 变换加载

2. [ ] **数据新鲜度检查**
   - `get_synced_data()` 增加 `max_age` 参数
   - 检查 timestamp 是否在合理范围内

### P1 (重要问题，强烈建议修复)

3. [ ] **分层同步策略**
   - `enable_lidar=false` 时只同步 RGB+Depth
   - 支持 LiDAR 可选

4. [ ] **启动时服务健康检查**
   - 检查 DINO-X 和 CDM 服务可用性
   - CDM 不可用时自动禁用

5. [ ] **修正消息定义注释**
   - Object3D.msg 中的坐标系说明

### P2 (建议改进)

6. [ ] **添加 Service 超时配置**
   - DINO-X 和 CDM 的请求超时可配置

7. [ ] **更新数据流图**
   - 使其与代码设计一致

8. [ ] **添加诊断接口**
   - `/perception_node/status` 话题或服务
   - 报告各组件状态

---

## 六、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 外部服务不可用 | 中 | 高 | 启动检查 + 运行时降级 |
| 数据同步失败 | 中 | 中 | 新鲜度检查 + 超时处理 |
| Service 响应超时 | 低 | 中 | 文档说明预期延迟 |
| 坐标变换错误 | 低 | 高 | 单元测试验证变换链 |

---

## 七、结论

设计总体合理，模块化清晰，复用了成熟组件。但存在几个需要修复的问题：

1. **CoordinateTransformer 不支持 base_link 变换** - 阻塞性问题
2. **数据新鲜度检查缺失** - 可能导致使用过时数据
3. **LiDAR 可选性处理不当** - enable_lidar=false 时会卡死

建议在实现前先修复 P0 级别问题，P1 问题在第一版实现中解决。
