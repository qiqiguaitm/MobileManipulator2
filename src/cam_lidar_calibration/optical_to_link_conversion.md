# Optical Frame 到 Camera Link 坐标系转换说明

本文档说明 `extrinsics_lidar_to_*_camera_link.yaml` 文件的生成流程。

---

## 一、文件关系

| 源文件 | 目标文件 | 说明 |
|--------|----------|------|
| `extrinsics_lidar_to_top_camera_optical_frame.yaml` | `extrinsics_lidar_to_top_camera_link.yaml` | Top 相机外参 |
| `extrinsics_lidar_to_chassis_camera_optical_frame.yaml` | `extrinsics_lidar_to_chassis_camera_link.yaml` | Chassis 相机外参 |

**为什么需要两个文件？**
- `optical_frame` 版本：标定工具直接输出，用于图像投影验证
- `camera_link` 版本：SLAM/导航系统使用，与 ROS TF 树对齐

---

## 二、转换脚本

### 2.1 核心函数

**文件**: `lib/transforms.py`

**函数**: `convert_optical_to_link_extrinsics(input_yaml, output_yaml, camera_type)`

```python
from lib import convert_optical_to_link_extrinsics

# 转换 Top 相机外参
convert_optical_to_link_extrinsics(
    input_yaml='config/extrinsics_lidar_to_top_camera_optical_frame.yaml',
    output_yaml='config/extrinsics_lidar_to_top_camera_link.yaml',
    camera_type='top'
)

# 转换 Chassis 相机外参
convert_optical_to_link_extrinsics(
    input_yaml='config/extrinsics_lidar_to_chassis_camera_optical_frame.yaml',
    output_yaml='config/extrinsics_lidar_to_chassis_camera_link.yaml',
    camera_type='chassis'
)
```

### 2.2 自动调用

`interactive_calibrate_v2.py` 在标定完成后自动调用此转换：

```python
# interactive_calibrate_v2.py:1265-1271
output_file_apply = os.path.join(output_dir, 'interactive_extrinsics_apply.yaml')
convert_optical_to_link_extrinsics(output_file, output_file_apply, camera_type)
```

---

## 三、数学原理

### 3.1 坐标系定义

| 坐标系 | X 轴 | Y 轴 | Z 轴 | 用途 |
|--------|------|------|------|------|
| `camera_link` | 前 | 左 | 上 | ROS 机器人坐标系 |
| `camera_optical_frame` | 右 | 下 | 前 | 光学/图像坐标系 |

### 3.2 变换链

```
T_lidar_to_camera_link = T_optical_to_link ∘ T_lidar_to_optical
```

**点变换形式**（p_link = R @ p_lidar + t）：
```python
R_final = R_optical_to_link @ R_lidar_to_optical
t_final = R_optical_to_link @ t_lidar_to_optical + t_optical_to_link
```

### 3.3 静态 TF 数据

`optical_frame → camera_link` 的变换来自 RealSense 驱动发布的静态 TF：

```bash
# 获取方法
rosrun tf tf_echo camera/top_link camera/top_color_optical_frame
```

**硬编码值** (`lib/transforms.py:141-152`)：

| 相机 | Translation (m) | Quaternion (xyzw) |
|------|-----------------|-------------------|
| Top | `[-0.000283, -0.059184, -0.000029]` | `[-0.499962, 0.499032, -0.499347, 0.501655]` |
| Chassis | `[-0.000245, 0.014810, 0.000133]` | `[-0.498668, 0.505262, -0.495065, 0.500949]` |

---

## 四、手动转换

如果需要手动转换（例如标定后复制文件），可以使用以下脚本：

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/agilex/MobileManipulator/src/cam_lidar_calibrate')

from lib import convert_optical_to_link_extrinsics

# Top 相机
convert_optical_to_link_extrinsics(
    '/home/agilex/MobileManipulator/src/robot_drivers/camera_driver/config/extrinsics_lidar_to_top_camera_optical_frame.yaml',
    '/home/agilex/MobileManipulator/src/robot_drivers/camera_driver/config/extrinsics_lidar_to_top_camera_link.yaml',
    'top'
)

# Chassis 相机
convert_optical_to_link_extrinsics(
    '/home/agilex/MobileManipulator/src/robot_drivers/camera_driver/config/extrinsics_lidar_to_chassis_camera_optical_frame.yaml',
    '/home/agilex/MobileManipulator/src/robot_drivers/camera_driver/config/extrinsics_lidar_to_chassis_camera_link.yaml',
    'chassis'
)
```

---

## 五、输出文件格式

转换后的文件包含 `note` 字段标明来源：

```yaml
child_frame_id: top_camera_link
frame_id: rs16_lidar
header:
  calibration_date: '2025-12-29'
  method: interactive_multi_frame_v2
  n_frames: 8
  note: Converted from optical_frame to camera_link using static TF  # <-- 标记
transform:
  rotation:
    w: 0.9992786319284481
    x: 0.01713463569007619
    y: 0.03380928394559147
    z: 0.002356342518321177
  translation:
    x: -0.4090279968149051
    y: 0.04536207960174699
    z: 0.48288159621062837
```

---

## 六、相关文件

| 文件 | 说明 |
|------|------|
| `lib/transforms.py` | 坐标变换核心函数 |
| `interactive_calibrate_v2.py` | 交互式标定工具（自动调用转换） |
| `lidar2cam_tf.md` | TF 使用说明 |
| `calibration_guide.md` | 标定操作指南 |
