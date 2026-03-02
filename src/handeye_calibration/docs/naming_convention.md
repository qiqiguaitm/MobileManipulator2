# 手眼标定命名约定说明

## 概述

本项目中存在 `gripper` 和 `flange` 两种命名方式，本文档对此进行说明。

## 术语定义

| 术语 | 含义 |
|------|------|
| **Flange (法兰)** | 机械臂末端的标准机械接口，是一个固定的刚体参考系，不随安装的工具变化 |
| **Gripper (夹爪)** | 安装在法兰上的末端执行器，可能相对法兰有额外的偏移 (TCP offset) |
| **TCP (Tool Center Point)** | 工具中心点，通常是夹爪的工作点 |

## 本项目的实际情况

### 代码中获取的位姿

本项目在获取机器人位姿时，统一使用：

```python
robot.arm.get_position(return_gripper_center=False)
robot.arm.set_position(..., use_gripper_center=False)
```

`return_gripper_center=False` 意味着返回的是**法兰(flange)坐标系**的位姿，而非夹爪中心。

### 命名约定

| 位置 | 命名 | 说明 |
|------|------|------|
| 代码变量 | `R_cam2gripper`, `R_gripper2base` | 沿用 OpenCV 惯例 |
| 输出文件 key | `camera_to_flan` | 反映实际语义 |
| 配置打印 | `cfg.T_cam2flan` | 反映实际语义 |

### 为什么存在差异

1. **OpenCV 惯例**：OpenCV `calibrateHandEye()` 函数官方文档使用 `R_gripper2base` 命名，学术论文和教程普遍采用此约定

2. **实际语义**：本项目标定的是相机到法兰的变换，输出时使用 `camera_to_flan` 更准确

## 重要说明

**本项目中的 `gripper` 变量实际上指的是法兰坐标系 (flange frame)**

在阅读和使用代码时请注意：

```
代码中的 T_cam2gripper  ≡  实际的 T_cam2flange
代码中的 T_gripper2base ≡  实际的 T_flange2base
```

## 坐标系变换链

### Eye-in-Hand (眼在手上)

```
Target → Camera → Flange → Base
         ↑         ↑
    T_target2cam  T_cam2flange (代码中叫 T_cam2gripper)
```

标定方程：
```
T_flange2base_i · T_cam2flange · T_target2cam_i = T_target2base (常量)
```

### Eye-to-Hand (眼在手外)

```
Target → Camera → Base
           ↑
      T_cam2base
```

标定方程：
```
T_flange2base_i · T_target2flange = T_cam2base · T_target2cam_i
```

## 使用标定结果

标定输出文件中的 `camera_to_flan` 字段即为相机到法兰的变换矩阵：

```yaml
camera_to_flan:
  translation:
    x: 0.xxx
    y: 0.xxx
    z: 0.xxx
  rotation_matrix: [...]
  quaternion:
    x: 0.xxx
    y: 0.xxx
    z: 0.xxx
    w: 0.xxx
```

如需将相机坐标系下的点转换到机器人基座坐标系：

```python
# P_base = T_flange2base @ T_cam2flange @ P_cam
P_flange = T_cam2flange @ P_cam
P_base = T_flange2base @ P_flange
```

## 参考

- OpenCV calibrateHandEye: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
- 手眼标定原理: Tsai, R.Y. and Lenz, R.K., "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration"


