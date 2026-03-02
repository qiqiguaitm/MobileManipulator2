# LiDAR-Camera 外参与 ROS TF 说明

本文档说明标定结果的坐标系关系、转换逻辑以及如何在 ROS 中使用。

---

## 一、坐标系定义

### 1.1 相机坐标系

RealSense 相机有两个主要坐标系：

| 坐标系 | 完整名称 | X 轴 | Y 轴 | Z 轴 | 用途 |
|--------|----------|------|------|------|------|
| **camera_link** | `camera/{top\|chassis}_link` | 前 | 左 | 上 | ROS 机器人坐标系，SLAM/导航使用 |
| **camera_optical_frame** | `camera/{top\|chassis}_color_optical_frame` | 右 | 下 | 前 | 光学/图像坐标系，视觉算法使用 |

**静态 TF**（由 RealSense 驱动发布，`tf_echo link optical` 输出）：

| 相机 | Translation (m) | Quaternion (xyzw) |
|------|-----------------|-------------------|
| Top | `[-0.000283, -0.059184, -0.000029]` | `[-0.499962, 0.499032, -0.499347, 0.501655]` |
| Chassis | `[-0.000245, 0.014810, 0.000133]` | `[-0.498668, 0.505262, -0.495065, 0.500949]` |

### 1.2 LiDAR 坐标系

| 坐标系 | 完整名称 | X 轴 | Y 轴 | Z 轴 |
|--------|----------|------|------|------|
| **rslidar** | `rs16_lidar` | 前 | 左 | 上 |

---

## 二、标定输出与坐标系转换

### 2.1 两种外参文件

| 文件 | 目标坐标系 | 用途 |
|------|-----------|------|
| `interactive_extrinsics.yaml` | optical_frame | 视觉算法（图像投影） |
| `*_camera_link.yaml` | camera_link | SLAM/导航系统 |

### 2.2 转换原理

标定工具输出 `lidar → optical_frame`，SLAM 需要 `lidar → camera_link`。

**变换链**：
```
T_lidar_to_link = T_optical_to_link @ T_lidar_to_optical
```

**点变换形式**：
```python
# p_link = R @ p_lidar + t
R_lidar_to_link = R_optical_to_link @ R_lidar_to_optical
t_lidar_to_link = R_optical_to_link @ t_lidar_to_optical + t_optical_to_link
```

> **注意**：矩阵乘法顺序是 `R_o2l @ R_l2o`，不是 `R_l2o @ R_o2l`。

### 2.3 ROS TF 约定

`tf_echo A B` 输出用于点变换 `p_A = R(q) @ p_B + t`。

因此 `tf_echo camera/top_link camera/top_color_optical_frame` 的输出就是 `optical → link` 的点变换，可直接使用。

---

## 三、当前标定结果

### Top Camera (2025-12-29)

**rslidar → camera/top_link**：
```
Translation: [-0.409028, 0.045362, 0.482882]
Quaternion:  [0.017135, 0.033809, 0.002356, 0.999279] (xyzw)
Euler (deg): Roll=1.98°, Pitch=3.87°, Yaw=0.34°
```

### Chassis Camera (2025-12-29)

**rslidar → camera/chassis_link**：
```
Translation: [0.052560, 0.003509, 0.073273]
Quaternion:  [-0.010223, -0.067863, 0.015707, 0.997519] (xyzw)
Euler (deg): Roll=-1.30°, Pitch=-7.76°, Yaw=1.89°
```

---

## 四、在 ROS 中使用

### Launch 文件配置

```xml
<!-- Top Camera: rslidar -> camera/top_link -->
<node pkg="tf2_ros" type="static_transform_publisher" name="rslidar_to_top_camera_tf"
      args="-0.409028 0.045362 0.482882 0.017135 0.033809 0.002356 0.999279 rslidar camera/top_link"/>

<!-- Chassis Camera: rslidar -> camera/chassis_link -->
<node pkg="tf2_ros" type="static_transform_publisher" name="rslidar_to_chassis_camera_tf"
      args="0.052560 0.003509 0.073273 -0.010223 -0.067863 0.015707 0.997519 rslidar camera/chassis_link"/>
```

参数顺序：`x y z qx qy qz qw parent_frame child_frame`

---

## 五、TF 树结构

```
map/odom
    └── base_link
           └── rslidar (激光雷达)
                  ├── camera/top_link ──── camera/top_color_optical_frame
                  │   (标定输出)            (RealSense 驱动)
                  │
                  └── camera/chassis_link ── camera/chassis_color_optical_frame
                      (标定输出)              (RealSense 驱动)
```

---

## 六、验证方法

1. 启动 SLAM：`roslaunch slam rtabmap_lidar_rgbd.launch camera_type:=top`
2. 在 RViz 中检查点云投影是否与图像对齐
3. 正确时：点云轮廓与图像物体边缘吻合

---

## 七、常见问题

### Q1: 为什么需要两个外参文件？

- 标定工具使用相机内参 K（定义在 optical_frame）
- SLAM 需要 camera_link 坐标系
- 自动转换保证两者一致

### Q2: 投影整体偏移？

平移参数不准确，重新标定。

### Q3: 投影倾斜或扭曲？

旋转参数不准确，重新标定。
