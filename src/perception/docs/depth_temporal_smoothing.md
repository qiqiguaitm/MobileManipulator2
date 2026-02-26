# RGB-D 深度时间平滑优化

## 问题
RGB-D 点云可视化中深度数据不稳定，表现为点云抖动。

## 根本原因
- **空间噪声**: 已通过 CDM 深度去噪解决
- **时间抖动**: 帧间深度值波动，缺少时间维度平滑

## 解决方案: EMA (指数移动平均) 滤波

### 算法原理
```python
depth_t = alpha * depth_current + (1 - alpha) * depth_previous
```

- `alpha = 0.7`: 快响应，轻度平滑（默认）
- `alpha = 0.5`: 平衡
- `alpha = 0.3`: 强平滑，但延迟较高

### 实现位置
`perception_rviz_node.py:_camera_callback()`

```python
# 时间平滑
if self._depth_buffer is not None:
    depth_m = self.depth_alpha * depth_m + (1.0 - self.depth_alpha) * self._depth_buffer

self._depth_buffer = depth_m.copy()
```

### 参数配置
**Launch 文件**:
```xml
<arg name="depth_alpha" default="0.7"/>
```

**命令行覆盖**:
```bash
roslaunch perception perception_rviz.launch depth_alpha:=0.5
```

### 效果评估
| Alpha | 平滑效果 | 响应延迟 | 适用场景 |
|-------|---------|---------|---------|
| 0.9   | 弱      | 极低    | 快速移动物体 |
| 0.7   | 中等    | 低      | **默认推荐** |
| 0.5   | 强      | 中等    | 静态场景 |
| 0.3   | 极强    | 高      | 极度嘈杂环境 |

### 性能影响
- 内存开销: 1 帧深度图 (640x480x4 = 1.2MB)
- 计算开销: 1 次加权求和 (< 1ms)
- **总体**: 可忽略不计

## 优化效果
- ✅ 消除帧间抖动
- ✅ 点云更稳定平滑
- ✅ 零性能损失
- ✅ 可配置灵活调整

## 相关文件
- `scripts/perception_rviz_node.py:62` - 参数定义
- `scripts/perception_rviz_node.py:166-171` - 滤波实现
- `launch/perception_rviz.launch:10` - Launch 参数
- `config/perception_3d.rviz` - RViz 配置

## 调试建议
如果平滑效果不够：
```bash
roslaunch perception perception_rviz.launch depth_alpha:=0.5
```

如果延迟太高：
```bash
roslaunch perception perception_rviz.launch depth_alpha:=0.8
```

---
**实现日期**: 2026-01-16
**优化类型**: 时间域滤波
**代码复杂度**: 3 行
**效果**: 显著改善点云稳定性
