# ROS1 vs ROS2 数据链路与时间同步对照

## 一、ROS1 完整链路（参照）

### 1. 点云 → Laserscan
```
rslidar/velodyne → /velodyne_points
    ↓
HDL globalmap_callback: /globalmap → registration->setInputTarget(globalmap)
HDL points_callback: 点云 → TF 转到 odom_child_frame_id → NDT 匹配 → /aligned_points
    ↓
pointcloud_to_laserscan: /aligned_points → /scan
```

**关键**: HDL 必须收到 /globalmap 才处理点云；globalmap_server 加载 PCD 并发布到 /globalmap。

### 2. TF 链 (ROS1)
```
map ←(HDL 发布)→ odom ←(robot_odom TF 源)→ base_link
odom->base 来源: lookupTransform(robot_odom_frame_id, odom_child_frame_id)
```

### 3. 时间
- map->odom、aligned_points、Odometry 均用**点云 stamp**
- Topic 与 TF 同源

---

## 二、ROS2 当前链路（问题）

### 1. 点云 → Laserscan（断裂）
```
rslidar_sdk → /rslidar_points → delayed_cloud_relay → /rslidar_points_delayed
    ↓
hdl_localization: 无 /globalmap 订阅！NDT 无 setInputTarget，has_aligned_points() 恒 false
    → /aligned_points 从未发布 ❌

方案 A (aligned): 需恢复 globalmap 输入，实现 has_aligned_points
方案 B (raw): use_raw_laserscan:=true → pointcloud_to_laserscan 直接订阅 /rslidar_points_delayed ✅
```

### 2. TF 链 (ROS2)
```
map ←(hdl_odom_to_tf)→ odom
odom->base: 来自 /odom/fused 或 TF (Fast-LIO)
base_link → lidar_link ←(static)→ rslidar
```

### 3. JointState 无效
- robot_state_publisher: `name.size() != position.size()` 导致拒绝
- 来源: joint_state_publisher 的 zeros 参数格式可能与 ROS2 不兼容
- 影响: 仅影响 arm 关节 TF，base_link→lidar_link 来自 URDF 固定关节，应仍发布

---

## 三、修复清单

| 问题 | 修复 |
|------|------|
| JointState 无效 | 用 static joint_state 脚本替代，或修正 zeros 格式 |
| aligned_points 空 | use_raw_laserscan:=true（默认），或实现 globalmap→HDL |
| 点云 frame | rslidar_sdk 用 frame_id=rslidar，需 lidar_link→rslidar 静态 TF |
