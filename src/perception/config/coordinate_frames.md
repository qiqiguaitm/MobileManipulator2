# 坐标系说明文档

## 1. 坐标系布局

### 1.1 TF 树结构

```
                    map
                     │
                     ▼
                   odom
                     │
                     ▼
    ┌────────────base_link────────────┐
    │                │                │
    ▼                ▼                ▼
 rslidar        arm_base_link     (wheels)
    │                │
    ▼                ▼
camera/top_link   link1->...->link6
    │                      │
    ▼                      ▼
camera/top_        hand_camera_link
color_optical_frame      │
                         ▼
                   hand_camera_
                   optical_frame
```

### 1.2 物理布局

**俯视图 (X+ 为前方):**
```
                        X+ (前)
                         ↑
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         │    [LiDAR]    │               │
         │   (rslidar)   │               │
         │     27.3cm    │               │
    Y+ ←─┼───────────────●───────────────┼─→ Y-
  (左)   │           base_link           │  (右)
         │               │               │
         │        [arm_base]             │
         │          11.8cm               │
         │               │               │
         └───────────────┴───────────────┘
                        X- (后)
```

**侧视图 (X+ 为前方):**
```
         Z+ (上)
           ↑
           │    [top_camera] ← 57.8cm
           │         ○  ↙ 俯仰约 4°
           │         │
           │    [arm_base] ← 10cm
           │         ●────────────→ X+ (前)
           │     base_link
           │
    ───────┴──────────────────────────
           地面
```

## 2. 坐标系参数

### 2.1 base_link → top_camera_link

| 参数 | 值 | 说明 |
|------|-----|------|
| x | -0.136m | 相机在底盘后方 13.6cm |
| y | 0.048m | 相机在底盘左侧 4.8cm |
| z | 0.578m | 相机高度 57.8cm |
| roll | 1.73° | - |
| pitch | 3.94° | 向下俯视约 4° |
| yaw | 0.32° | - |

### 2.2 base_link → rslidar

| 参数 | 值 | 说明 |
|------|-----|------|
| x | 0.273m | LiDAR 在底盘前方 27.3cm |
| y | 0.0m | 居中 |
| z | 0.095m | LiDAR 高度 9.5cm |
| rotation | 几乎无旋转 | 与 base_link 基本对齐 |

### 2.3 base_link → arm_base_link

| 参数 | 值 | 说明 |
|------|-----|------|
| x | 0.118m | 机械臂在底盘前方 11.8cm |
| y | 0.0m | 居中 |
| z | 0.1m | 机械臂基座高度 10cm |
| rotation | 无旋转 | 与 base_link 对齐 |

### 2.4 arm_base_link → rslidar

| 参数 | 值 | 说明 |
|------|-----|------|
| x | 0.155m | LiDAR 在臂基座前方 15.5cm |
| y | 0.0m | 居中 |
| z | -0.005m | LiDAR 略低于臂基座 0.5cm |

### 2.5 arm_base_link → top_camera_link

| 参数 | 值 | 说明 |
|------|-----|------|
| x | -0.254m | 相机在臂基座后方 25.4cm |
| y | 0.048m | 相机在臂基座左侧 4.8cm |
| z | 0.478m | 相对臂基座高度 47.8cm |

### 2.6 top_camera_link 与 top_camera_optical_frame 详解

#### 2.6.1 准确含义

| 坐标系 | 物理位置 | 坐标轴约定 | 主要用途 |
|--------|----------|------------|----------|
| `top_camera_link` | **深度传感器**光学中心 | ROS REP-103 (X前 Y左 Z上) | TF 树连接、SLAM |
| `top_camera_optical_frame` | **彩色传感器**光学中心 | 光学约定 (X右 Y下 Z前) | 图像处理、像素投影 |

#### 2.6.2 坐标轴定义对比

```
top_camera_link (ROS REP-103):      top_camera_optical_frame (光学约定):

        Z (上)                              Y (下, v方向)
         ↑                                   ↓
         │                                   │
         │                                   │
         ●────→ X (前, 光轴)                 ●────→ X (右, u方向)
        ╱                                   ╱
       ↙                                   ↙
      Y (左)                              Z (前, 深度方向)
```

**变换关系**: optical_frame 相对于 camera_link 旋转了 -90° (绕 X 轴) 再 -90° (绕 Z 轴)

#### 2.6.3 物理偏移 (关键!)

**RealSense D455 相机内部结构:**
```
┌─────────────────────────────────────────┐
│                                         │
│   [IR Left]  [RGB]  [IR Right]         │
│      ●   ←5.9cm→  ●                    │
│   (深度基准)    (彩色)                  │
│                                         │
└─────────────────────────────────────────┘
      ↑                ↑
  camera_link     optical_frame
  (深度光心)       (彩色光心)
```

- `camera_link` 位于**深度传感器 (IR Left)** 的光学中心
- `optical_frame` 位于**彩色传感器 (RGB)** 的光学中心
- 两者之间存在约 **5.9cm 的物理偏移** (深度到彩色的基线距离)

#### 2.6.4 变换参数 (来自 RealSense ROS 驱动)

| 参数 | 值 | 说明 |
|------|-----|------|
| translation.x | -0.000283 m | 几乎为 0 |
| translation.y | **-0.059184 m** | **5.9cm 基线距离** |
| translation.z | -0.000029 m | 几乎为 0 |
| quaternion | [-0.500, 0.499, -0.499, 0.502] | 坐标轴旋转 |

**来源**: RealSense ROS 驱动自动发布的静态 TF
```bash
# 查询方法
rosrun tf tf_echo camera/top_link camera/top_color_optical_frame
```

#### 2.6.5 为什么需要两个坐标系?

| 场景 | 使用坐标系 | 原因 |
|------|------------|------|
| **TF 树 / SLAM** | camera_link | ROS 标准约定，与机器人其他部件一致 |
| **像素坐标计算** | optical_frame | 与图像坐标 (u, v) 对齐，Z 为深度 |
| **点云着色** | 需要两者转换 | 深度来自 IR，颜色来自 RGB |
| **抓取定位** | optical_frame | 检测结果是像素坐标 + 深度 |

#### 2.6.6 常见误区

❌ **错误理解**: camera_link 和 optical_frame 只是坐标轴方向不同，原点相同

✅ **正确理解**: 两者不仅坐标轴方向不同，**原点也不同** (相差约 5.9cm)

这个物理偏移在精确抓取等应用中**不可忽略**。

## 3. 外参文件列表

| 文件名 | 变换 | 用途 |
|--------|------|------|
| `extrinsics_base_link_to_rslidar.yaml` | base_link → rslidar | LiDAR 定位 |
| `extrinsics_arm_base_link_to_rslidar.yaml` | arm_base_link → rslidar | LiDAR-机械臂融合 |
| `extrinsics_rslidar_to_top_camera_optical_frame.yaml` | rslidar → top_camera_optical_frame | **LiDAR-相机标定** |
| `extrinsics_base_link_to_top_camera_link.yaml` | base_link → top_camera_link | 导航定位 |
| `extrinsics_arm_base_link_to_top_camera_link.yaml` | arm_base_link → top_camera_link | 机械臂规划 |
| `extrinsics_base_link_to_top_camera_optical_frame.yaml` | base_link → top_camera_optical_frame | 点云处理 |
| `extrinsics_arm_base_link_to_top_camera_optical_frame.yaml` | arm_base_link → top_camera_optical_frame | 抓取计算 |
| `extrinsics_flan_to_hand_camera.yaml` | link6(法兰) → hand_camera | 手眼标定 |

## 4. 参数来源

### 4.1 来自 launch 文件 (手动标定)

文件: `/home/agilex/MobileManipulator/src/slam/launch/rtabmap_slam.launch`

```xml
<!-- base_link -> rslidar -->
<node pkg="tf2_ros" type="static_transform_publisher" name="base_to_rslidar_tf"
      args="0.27254 0.00035292 0.095 -0.00218 0.00059 0.0 0.9999974 base_link rslidar"/>

<!-- rslidar -> camera/top_link -->
<node pkg="tf2_ros" type="static_transform_publisher" name="rslidar_to_top_camera_tf"
      args="-0.409028 0.045362 0.482882 0.017135 0.033809 0.002356 0.999279 rslidar camera/top_link"/>
```

### 4.2 来自 URDF (机械结构)

文件: `/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_description.urdf`

```xml
<!-- arm_base 相对于 base_link -->
<joint name="arm_joint" type="fixed">
  <origin rpy="0.0 0.0 0.0" xyz="0.118 0 0.1"/>
  <parent link="box_link"/>  <!-- box_link 与 base_link 重合 -->
  <child link="arm_base"/>
</joint>
```

### 4.3 来自 RealSense SDK (相机出厂标定)

深度传感器与彩色传感器之间的外参由相机固件提供。

## 5. RealSense SDK 获取外参

### 5.1 Python 代码示例

```python
import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_realsense_extrinsics():
    """从 RealSense 获取传感器间外参"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

    # 获取内参
    depth_intr = depth_profile.get_intrinsics()
    color_intr = color_profile.get_intrinsics()

    print(f"深度内参: fx={depth_intr.fx}, fy={depth_intr.fy}, cx={depth_intr.ppx}, cy={depth_intr.ppy}")
    print(f"彩色内参: fx={color_intr.fx}, fy={color_intr.fy}, cx={color_intr.ppx}, cy={color_intr.ppy}")

    # 获取外参 (深度 → 彩色)
    extr = depth_profile.get_extrinsics_to(color_profile)

    rot_matrix = np.array(extr.rotation).reshape(3, 3)
    translation = np.array(extr.translation)
    quaternion = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

    print(f"深度→彩色 平移: {translation * 1000} mm")
    print(f"深度→彩色 四元数: {quaternion}")

    pipeline.stop()

    return {
        'depth_intrinsics': depth_intr,
        'color_intrinsics': color_intr,
        'depth_to_color': {
            'translation': translation,
            'rotation': rot_matrix,
            'quaternion': quaternion,
        }
    }

if __name__ == '__main__':
    get_realsense_extrinsics()
```

### 5.2 保存外参到 YAML

```python
import pyrealsense2 as rs
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

def save_realsense_extrinsics(filepath):
    """从 RealSense 获取外参并保存到 YAML"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

    extr = depth_profile.get_extrinsics_to(color_profile)
    rot_matrix = np.array(extr.rotation).reshape(3, 3)
    quat = R.from_matrix(rot_matrix).as_quat()

    data = {
        'header': {
            'calibration_date': 'factory',
            'source': 'RealSense SDK',
            'frame_id': 'depth_frame',
            'child_frame_id': 'color_frame',
        },
        'transform': {
            'translation': {
                'x': float(extr.translation[0]),
                'y': float(extr.translation[1]),
                'z': float(extr.translation[2]),
            },
            'rotation': {
                'x': float(quat[0]),
                'y': float(quat[1]),
                'z': float(quat[2]),
                'w': float(quat[3]),
            }
        }
    }

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    pipeline.stop()
    print(f"已保存: {filepath}")

# 使用示例
# save_realsense_extrinsics('config/extrinsics_depth_to_color.yaml')
```

### 5.3 ROS TF 查询

```bash
# 查看实时 TF
rosrun tf tf_echo base_link camera/top_color_optical_frame

# 查看 TF 树
rosrun tf view_frames
evince frames.pdf
```

## 6. 使用场景

| 场景 | 使用的坐标系 | 说明 |
|------|-------------|------|
| 导航定位 | base_link | 底盘控制、路径规划 |
| 机械臂运动 | arm_base_link | IK 求解、轨迹规划 |
| 图像处理 | optical_frame | 像素坐标、深度投影 |
| 点云处理 | optical_frame | 3D 重建、物体检测 |
| 抓取计算 | arm_base_link + optical_frame | 目标定位 → 臂基座坐标 |

## 7. 坐标变换示例

### 7.1 图像点转机械臂基座坐标

```python
import numpy as np
import yaml

def load_extrinsics(yaml_path):
    """加载外参 YAML"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    t = data['transform']['translation']
    r = data['transform']['rotation']
    return np.array([t['x'], t['y'], t['z']]), np.array([r['x'], r['y'], r['z'], r['w']])

def pixel_to_arm_base(u, v, depth, intrinsics, extrinsics_yaml):
    """
    将图像像素坐标转换为机械臂基座坐标

    Args:
        u, v: 像素坐标
        depth: 深度值 (米)
        intrinsics: 相机内参 (fx, fy, cx, cy)
        extrinsics_yaml: arm_base_link_to_top_camera_optical_frame.yaml 路径

    Returns:
        机械臂基座坐标系下的 3D 点 (x, y, z)
    """
    from scipy.spatial.transform import Rotation as R

    fx, fy, cx, cy = intrinsics

    # 像素 → 光学坐标系
    x_optical = (u - cx) * depth / fx
    y_optical = (v - cy) * depth / fy
    z_optical = depth
    p_optical = np.array([x_optical, y_optical, z_optical])

    # 加载外参 (arm_base → optical)
    trans, quat = load_extrinsics(extrinsics_yaml)
    rot = R.from_quat(quat)

    # optical → arm_base (逆变换)
    p_arm = rot.inv().apply(p_optical - trans)

    return p_arm

# 使用示例
# intrinsics = (385.56, 385.01, 320.18, 242.99)  # 从 RealSense 获取
# point_arm = pixel_to_arm_base(320, 240, 0.5, intrinsics,
#                               'config/extrinsics_arm_base_link_to_top_camera_optical_frame.yaml')
```

---

**文档更新日期**: 2026-01-14
**数据来源**: rtabmap_slam.launch, mobile_manipulator2_description.urdf, RealSense SDK, RealSense ROS 驱动 TF
