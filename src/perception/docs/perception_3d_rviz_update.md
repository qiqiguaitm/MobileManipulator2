# perception_3d.rviz 配置更新

**更新日期**: 2026-01-16
**版本**: v2.0

## 更新内容

### ✅ 配置验证结果

| 项目 | 配置项 | 当前值 | 状态 |
|------|--------|--------|------|
| 2 | Camera Image Topic | `/camera/top/color/image_raw` | ✅ 正确 |
| 3 | Distance Labels Topic | `/perception_rviz_node/distance_labels` | ✅ 正确 |
| 4 | Object Markers Topic | `/perception_rviz_node/distance_labels` | ✅ 已更新 |
| 5 | TF 显示坐标系 | 见下表 | ✅ 正确 |

### TF 显示的坐标系

| 坐标系 | 显示状态 | 说明 |
|--------|---------|------|
| `base_link` | ✅ 显示 | 机器人底盘中心 |
| `arm_base_link` | ✅ 显示 | 机械臂基座 |
| `rslidar` | ✅ 显示 | LiDAR 坐标系 |
| `top_camera_link` | ✅ 显示 | 顶部相机 |
| `hand_camera_link` | ✅ 显示 | 手部相机 |
| `chassis_camera_link` | ✅ 显示 | 底盘相机 |
| `gripper_link` | ✅ 显示 | 夹爪坐标系 |

**总计**: 7 个坐标系（包含 3 个相机、1 个 LiDAR、底盘、机械臂、夹爪）

## 详细修改

### 修改 1: Object Markers Topic

**修改前**：
```yaml
- Class: rviz/MarkerArray
  Name: Object Markers
  Enabled: true
  Topic: /perception_rviz_node/object_markers
  Queue Size: 100
  Namespaces:
    object_bbox: true
```

**修改后**：
```yaml
- Class: rviz/MarkerArray
  Name: Object Markers
  Enabled: true
  Topic: /perception_rviz_node/distance_labels
  Queue Size: 100
  Namespaces:
    object_bbox: true
    distance_labels: true
```

**变更说明**：
- Topic 从 `/perception_rviz_node/object_markers` 改为 `/perception_rviz_node/distance_labels`
- 添加了 `distance_labels` namespace 支持

## 配置项说明

### Camera Image (相机图像)
- **Topic**: `/camera/top/color/image_raw`
- **功能**: 显示 top 相机的 RGB 图像
- **位置**: RViz 窗口右侧面板

### Distance Labels (距离标签)
- **Topic**: `/perception_rviz_node/distance_labels`
- **功能**: 在 3D 视图中显示物体距离的文字标签
- **Namespace**: `distance_labels`

### Object Markers (物体标记)
- **Topic**: `/perception_rviz_node/distance_labels`
- **功能**: 显示物体的边界框和距离标签
- **Namespaces**:
  - `object_bbox`: 物体边界框
  - `distance_labels`: 距离标签

> **注意**: Object Markers 和 Distance Labels 现在使用相同的 topic，两者会显示相同的内容。

### TF (坐标系变换)
- **显示的坐标系**:
  - 相机: `top_camera_link`, `hand_camera_link`, `chassis_camera_link`
  - LiDAR: `rslidar`
  - 机器人: `base_link`, `arm_base_link`, `gripper_link`
- **Marker Scale**: 0.3
- **显示**: 箭头 + 坐标轴 + 名称

## RViz 显示项列表

完整的 RViz 显示项：

1. **Grid** - 网格地面
2. **TF** - 坐标系变换（7 个坐标系）
3. **RobotModel** - 机器人模型
4. **Camera Image** - 相机图像 (top camera)
5. **RGB-D PointCloud** - RGB-D 点云
6. **LiDAR PointCloud** - LiDAR 点云
7. **Object Clouds** - 物体点云
8. **Object Markers** - 物体标记（边界框 + 距离标签）
9. **Distance Labels** - 距离标签

## 使用方法

### 启动 RViz

```bash
# 方式 1: 使用启动脚本
cd ~/MobileManipulator/scripts
./start_perception.sh --rviz --test

# 方式 2: 手动启动
roslaunch perception perception_rviz.launch rviz:=true

# 方式 3: 直接启动 RViz
rviz -d ~/MobileManipulator/src/perception/config/perception_3d.rviz
```

### 查看配置

在 RViz 中：
1. 左侧 **Displays** 面板显示所有显示项
2. 展开 **TF** 查看显示的坐标系
3. 展开 **Object Markers** 查看 namespaces
4. 展开 **Distance Labels** 查看 namespaces

### 调整显示

**隐藏/显示某个坐标系**：
1. 展开 **TF** → **Frames**
2. 取消勾选不需要的坐标系

**调整相机图像位置**：
1. 拖动 **Camera Image** 面板
2. 可停靠在任意位置

**调整点云大小**：
1. 选择点云显示项
2. 修改 **Size (m)** 参数

## 配置文件位置

- **RViz 配置**: `~/MobileManipulator/src/perception/config/perception_3d.rviz`
- **Launch 文件**: `~/MobileManipulator/src/perception/launch/perception_rviz.launch`

## 验证配置

运行以下命令验证配置：

```bash
cd ~/MobileManipulator/src/perception

# 验证 Camera Image topic
grep -A 2 "Name: Camera Image" config/perception_3d.rviz | grep "Topic:"
# 应输出: Topic: /camera/top/color/image_raw

# 验证 Distance Labels topic
grep -A 2 "Name: Distance Labels" config/perception_3d.rviz | grep "Topic:"
# 应输出: Topic: /perception_rviz_node/distance_labels

# 验证 Object Markers topic
grep -A 2 "Name: Object Markers" config/perception_3d.rviz | grep "Topic:"
# 应输出: Topic: /perception_rviz_node/distance_labels

# 验证 TF 显示的坐标系
grep -A 20 "Frames:" config/perception_3d.rviz | grep -B 1 "Value: true"
# 应显示 7 个坐标系
```

## 故障排查

### 问题：相机图像不显示

**原因**: Camera topic 没有数据
**解决**:
```bash
# 检查 topic 是否存在
rostopic list | grep /camera/top/color/image_raw

# 检查 topic 是否有数据
rostopic hz /camera/top/color/image_raw

# 重启相机驱动
roslaunch camera_driver camera_driver.launch top_enable:=true
```

### 问题：物体标记不显示

**原因**: 感知节点未运行或未检测到物体
**解决**:
```bash
# 检查节点是否运行
rosnode list | grep perception

# 检查 topic 是否有数据
rostopic echo /perception_rviz_node/distance_labels

# 运行检测测试
rosservice call /scene_perception_3d/detect "prompt: 'bottle', enable_lidar: false"
```

### 问题：TF 坐标系不显示

**原因**: TF 数据未发布
**解决**:
```bash
# 检查 TF 树
rosrun tf view_frames

# 检查特定坐标系
rosrun tf tf_echo base_link top_camera_link
```

## 性能优化

### 降低点云密度

如果 RViz 运行卡顿，可以降低点云的显示密度：

1. **RGB-D PointCloud**: `Size (m)` 从 0.008 改为 0.015
2. **LiDAR PointCloud**: `Size (m)` 从 0.015 改为 0.03
3. **Object Clouds**: `Size (m)` 从 0.015 改为 0.03

### 禁用不需要的显示项

取消勾选不需要的显示项：
- **LiDAR PointCloud** - 如果使用纯视觉模式
- **RGB-D PointCloud** - 如果只关注检测结果
- **RobotModel** - 如果不需要显示机器人模型

## 相关文档

- `docs/default_config_update.md` - 默认配置更新
- `scripts/QUICK_START.md` - 快速启动指南
- `launch/perception_rviz.launch` - RViz 启动文件

## 版本历史

**v2.0 (2026-01-16)**:
- ✅ 验证 Camera Image 使用 top camera topic
- ✅ 验证 Distance Labels 使用正确 topic
- ✅ 更新 Object Markers topic 为 `/perception_rviz_node/distance_labels`
- ✅ 验证 TF 显示 7 个坐标系（3 cameras + lidar + arm_base + base_link + gripper）
- ✅ 添加 Camera Image 到展开列表

**v1.0 (2026-01-15)**:
- 初始配置创建
