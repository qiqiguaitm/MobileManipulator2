# RViz 3D 可视化设计方案

## 目标
为 Scene Perception 3D 系统设计 RViz 可视化，实时显示感知结果。

## 需求分析

### 1. 基础可视化
- Robot Model - 机器人 URDF 模型
- RGB-D 点云 - 深度相机 3D 投影 (base_link 坐标系)
- LiDAR 点云 - 激光雷达原始点云

### 2. 检测结果可视化
- 物体 Mask 3D 投影 - 不同颜色区分不同物体
- 距离标签 - 显示 Camera 距离 (C:) 和 LiDAR 距离 (L:)

---

## 架构设计

```
scene_perception_3d_node
         │
         ├── /scene_perception_3d/objects_3d (Object3DArray)
         │
         ▼
perception_rviz_node (新增)
         │
         ├── /perception/rgb_pointcloud (PointCloud2) - RGB-D 点云
         ├── /perception/object_markers (MarkerArray) - 物体 3D 标记
         └── /perception/distance_labels (MarkerArray) - 距离文字标签
```

---

## 实现阶段

### Phase 1: RGB-D 点云发布 [pending]
- [ ] 订阅 RGB + Depth 图像
- [ ] 使用相机内参反投影到 3D
- [ ] 变换到 base_link 坐标系
- [ ] 发布 PointCloud2 (带 RGB 颜色)

### Phase 2: 物体 Mask 3D 投影 [pending]
- [ ] 订阅检测结果 (Object3DArray)
- [ ] 对每个物体的 mask 区域:
  - 提取对应的深度值
  - 生成带颜色的 3D 点云
  - 变换到 base_link
- [ ] 发布为 MarkerArray (POINTS 类型)

### Phase 3: 距离标签 [pending]
- [ ] 为每个物体创建 TEXT_VIEW_FACING marker
- [ ] 显示格式: "object_id\nC:1.23 L:1.25"
- [ ] 位置: 物体质心上方

### Phase 4: RViz 配置文件 [pending]
- [ ] 创建 perception_3d.rviz 配置
- [ ] 添加所有必要的 Display 项
- [ ] 配置合适的视角和颜色

### Phase 5: Launch 文件集成 [pending]
- [ ] 创建 perception_rviz.launch
- [ ] 可选参数控制各项显示

---

## 消息定义

### 发布的 Topics

| Topic | 类型 | 说明 |
|-------|------|------|
| `/perception/rgb_pointcloud` | PointCloud2 | RGB-D 彩色点云 (base_link) |
| `/perception/object_clouds` | PointCloud2 | 物体 mask 点云 (带颜色) |
| `/perception/object_markers` | MarkerArray | 物体边界框 |
| `/perception/distance_labels` | MarkerArray | 距离文字标签 |

### 订阅的 Topics

| Topic | 类型 | 说明 |
|-------|------|------|
| `/camera/top/color/image_raw` | Image | RGB 图像 |
| `/camera/top/aligned_depth_to_color/image_raw` | Image | 深度图 |
| `/camera/top/color/camera_info` | CameraInfo | 相机内参 |
| `/scene_perception_3d/objects_3d` | Object3DArray | 检测结果 |

---

## 颜色方案

物体使用不同颜色区分:
```python
COLORS = [
    (0.0, 1.0, 0.0, 0.8),  # 绿色
    (1.0, 0.0, 0.0, 0.8),  # 红色
    (0.0, 0.0, 1.0, 0.8),  # 蓝色
    (1.0, 1.0, 0.0, 0.8),  # 黄色
    (1.0, 0.0, 1.0, 0.8),  # 品红
    (0.0, 1.0, 1.0, 0.8),  # 青色
    ...
]
```

---

## 文件结构

```
src/perception/
├── scripts/
│   └── perception_rviz_node.py    # [新增] RViz 可视化节点
├── config/
│   └── perception_3d.rviz         # [新增] RViz 配置
└── launch/
    └── perception_rviz.launch     # [新增] 可视化启动文件
```

---

## 决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 点云坐标系 | base_link | 与机器人模型统一，便于规划 |
| Mask 可视化 | PointCloud2 | 比 Mesh 更高效，支持大量点 |
| 距离标签 | TEXT_VIEW_FACING | 始终面向相机，易读 |

---

## 进度

- [x] Phase 1: RGB-D 点云发布
- [x] Phase 2: 物体 Mask 3D 投影
- [x] Phase 3: 距离标签
- [x] Phase 4: RViz 配置文件
- [x] Phase 5: Launch 文件集成

## 验证完成

```
# 节点正在运行
rosnode list | grep perception_rviz
# 输出: /perception_rviz_node

# Topics 正常发布
rostopic list | grep perception_rviz
# 输出:
# /perception_rviz_node/distance_labels
# /perception_rviz_node/object_clouds
# /perception_rviz_node/object_markers
# /perception_rviz_node/rgb_pointcloud

# RGB 点云发布频率 ~5Hz
rostopic hz /perception_rviz_node/rgb_pointcloud
```
