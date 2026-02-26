# 默认配置更新记录

**日期**: 2026-01-16  
**版本**: v2.0

## 更新内容

### 默认配置变更

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `enable_lidar` | `true` | `false` | 纯视觉模式性能更好，大多数场景无需 LiDAR |
| `auto_detect_rate` | `1.8 Hz` | `2.0 Hz` | 优化后性能提升，可以支持更高帧率 |

### 性能提升总结

**优化历程**：
1. 原始版本: 1720ms (0.55 Hz)
2. 向量化 + 预热: 835ms (1.13 Hz) - 提升 105%
3. 并行化 DINO-X + CDM: 440ms (2.27 Hz) - 提升 313%
4. 并行化 3D 测量 + 批量变换: 424ms (2.36 Hz) - 提升 329%

**当前性能**（CDM + 纯视觉）：
- 平均耗时: 424ms
- 理论最大帧率: 2.36 Hz
- 配置帧率: 2.0 Hz (留 15% 安全余量)

## 修改的文件

### 1. config/scene_perception_3d.yaml

```yaml
# 变更
enable_lidar: false             # 默认禁用 LiDAR（纯视觉模式）
auto_detect_rate: 2.0           # 自动检测频率 (Hz)
```

### 2. launch/scene_perception_3d.launch

```xml
<!-- 变更 -->
<arg name="enable_lidar" default="false"/>  <!-- 默认禁用 LiDAR（纯视觉模式） -->
```

## 使用指南

### 默认启动（推荐）

```bash
roslaunch perception scene_perception_3d.launch
```

**配置**: CDM + 纯视觉 + 2Hz  
**性能**: 424ms 平均耗时  
**适用**: 大多数场景

### 高速模式

```bash
roslaunch perception scene_perception_3d.launch enable_depth_optimizer:=false
```

**配置**: 无CDM + 纯视觉 + 2Hz  
**性能**: 390ms 平均耗时  
**适用**: 实时跟踪，深度质量足够好的场景

### 高精度模式

```bash
roslaunch perception scene_perception_3d.launch enable_lidar:=true auto_detect_rate:=0.67
```

**配置**: CDM + LiDAR + 0.67Hz  
**性能**: 1217ms 平均耗时  
**适用**: 精确抓取，需要 LiDAR 辅助

## 向后兼容性

- ✅ 所有 launch 参数仍可覆盖默认值
- ✅ Service 调用接口不变
- ✅ Topic 输出格式不变
- ⚠️ 默认不再订阅 LiDAR topic（可通过参数启用）

## 验证步骤

```bash
# 1. 启动节点
roslaunch perception scene_perception_3d.launch

# 2. 验证配置
rosparam get /scene_perception_3d/auto_detect_rate  # 应输出: 2.0
rosparam get /scene_perception_3d/enable_lidar      # 应输出: false
rosparam get /scene_perception_3d/enable_depth_optimizer  # 应输出: true

# 3. 测试服务
rosservice call /scene_perception_3d/detect "prompt: 'bottle'
enable_lidar: false"

# 4. 检查自动检测帧率
rostopic hz /scene_perception_3d/objects_3d  # 应约为 2.0 Hz
```

## 性能监控

建议在实际场景中监控性能：

```bash
# 运行性能测试
cd /home/agilex/MobileManipulator
bash src/perception/scripts/benchmark_timing.sh no-lidar
```

**预期结果**：
- 平均耗时: 400-450ms
- 理论帧率: 2.2-2.5 Hz
- 3D测量: 40-50ms

## 相关文档

- `docs/performance_optimization.md` - 完整性能优化记录
- `docs/warmup_validation.md` - 预热机制验证
- `config/scene_perception_3d.yaml` - 完整配置文件
- `launch/scene_perception_3d.launch` - Launch 文件

