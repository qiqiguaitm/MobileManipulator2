# 相机-激光雷达标定工具使用指南

本项目提供两种标定方法，直接输出 **T_lidar_to_camera_optical_frame**。

---

## 标定方法选择

### 方法 1：自动标定（推荐）

使用多帧数据自动优化求解外参，无需手动调参。

```bash
python auto_calibrate.py --data-dir ./calib_data_top
```

**优点**：
- 全自动，无需人工干预
- 使用多帧数据，精度更高
- 生成可视化结果

**适用场景**：
- 已采集多组（≥3组）标定数据
- 需要高精度标定

---

### 方法 2：交互式标定

用户手动选择标定板区域，多帧联合优化。

```bash
python interactive_calibrate.py --data-dir ./calib_data_top
```

**优点**：
- 用户可控制区域选择
- 适合复杂场景

**适用场景**：
- 自动检测失败时
- 需要精确控制标定区域

---

## 一、数据准备

### 1. 硬件准备

- 相机与激光雷达已固定安装，支架刚性连接
- 80cm × 60cm 硬质 KT 板，哑光棋盘格（格子 7.2cm）

### 2. 数据采集

使用 `collect_data.py` 采集标定数据：

```bash
python collect_data.py top
```

采集要求：
- 至少 3 组不同角度的数据
- 标定板尽量占据图像 1/3 以上
- 雷达扫描线 ≥ 5 条经过标定板

### 3. 数据提取

从 rosbag 提取图像和点云：

```bash
python extract_data.py calib_bags_top/data.bag top
```

提取后的目录结构：
```
calib_data_top/
├── image/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── pcd/
    ├── 0.pcd
    ├── 1.pcd
    └── ...
```

---

## 二、自动标定（推荐）

### 完整命令

```bash
python auto_calibrate.py \
  --data-dir ./calib_data_top \
  --pattern 7x6 \
  --square 0.072 \
  --visualize
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 数据目录 | 必填 |
| `--pattern` | 棋盘格内角点数 (宽×高) | 7x6 |
| `--square` | 格子边长 (米) | 0.072 |
| `--board-width` | 标定板物理宽度 (米) | 0.80 |
| `--board-height` | 标定板物理高度 (米) | 0.60 |
| `--visualize` | 生成可视化图像 | false |

### 点云过滤参数

```bash
python auto_calibrate.py \
  --data-dir ./calib_data_top \
  --x-min 0.5 --x-max 4.0 \
  --y-min -2.0 --y-max 2.0 \
  --z-min 0.2 --z-max 2.0  # 过滤地面点
```

### 输出结果

```
calib_data_top/
├── conf/
│   └── auto_extrinsics.yaml              # 标定结果
└── visualization/
    ├── projection_0.png                   # 可视化结果
    ├── projection_1.png
    └── ...
```

---

## 三、交互式标定

### 使用流程

```bash
python interactive_calibrate.py --data-dir ./calib_data_top
```

**操作步骤：**
1. 程序显示第一帧图像
2. 鼠标拖动选择标定板区域
3. 按 `Enter` 确认，按 `r` 重新选择
4. 重复直到所有帧都选择完毕
5. 自动优化并保存结果

### 使用初始外参

交互式标定会自动加载初始外参（如果存在）：

```bash
# 自动加载 config/init_extrinsics_lidar_to_top_camera.yaml
python interactive_calibrate.py --data-dir ./calib_data_top

# 手动指定初始外参
python interactive_calibrate.py \
  --data-dir ./calib_data_top \
  --init_extrinsic ./config/my_init_extrinsics.yaml
```

**加载优先级：**
1. `--init_extrinsic` 参数指定的文件
2. `config/init_extrinsics_lidar_to_<camera>_camera.yaml`（自动推断相机类型）
3. 硬编码的理想化估计（如果配置文件不存在）

### 输出结果

```
calib_data_top/
└── conf/
    └── interactive_extrinsics.yaml       # 标定结果
```

---

## 四、内参文件准备

两种标定方法都需要相机内参文件。

### 内参文件格式

```yaml
# calib_data_top/conf/camera_intrinsic.yaml
K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
D: [k1, k2, p1, p2, k3]
```

### 生成内参文件

从项目内参文件提取：

```python
import yaml
import re

# 读取项目内参
with open('../../robot_drivers/camera_driver/config/intrics_top_camera.yaml') as f:
    content = f.read()

# 提取目标分辨率的内参 (1280x720)
# 手动提取 fx, fy, cx, cy, k1, k2, p1, p2
K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
D = [k1, k2, p1, p2, 0]

# 保存
output = {
    'K': K,
    'D': D
}
with open('calib_data_top/conf/camera_intrinsic.yaml', 'w') as f:
    yaml.dump(output, f)
```

---

## 五、验证标定结果

### 在 RViz 中验证

1. 发布标定 TF：

```bash
# 从结果文件读取参数
rosrun tf2_ros static_transform_publisher \
  x y z qx qy qz qw \
  rs16_lidar top_camera_color_optical_frame
```

2. 在 RViz 中：
   - Fixed Frame: `rs16_lidar`
   - 添加 PointCloud2（雷达点云）
   - 添加 Image（相机图像）
   - 添加 TF（坐标系）

### 检查标准

- ✅ **正确**：点云投影与图像轮廓严丝合缝
- ❌ **平移错误**：投影整体偏移
- ❌ **旋转错误**：投影倾斜或扭曲

---

## 六、常见问题

### Q1: 自动标定检测不到标定板？

**原因**：
- 图像质量差（模糊、过曝）
- 标定板太小或角度太倾斜

**解决**：
1. 重新采集数据，确保标定板清晰
2. 使用交互式标定手动选择区域

### Q2: 点云投影偏差大？

**原因**：
- 内参不准确
- 标定数据质量差

**解决**：
1. 检查内参文件是否匹配当前分辨率
2. 重新采集数据，确保雷达扫描线充足

### Q3: 优化失败或误差过大？

**原因**：
- 有效帧数不足（< 3 帧）
- 标定板尺寸设置错误

**解决**：
1. 采集更多数据
2. 检查 `--board-width` 和 `--board-height` 参数

---

## 七、目录结构

```
cam_lidar_calibrate/
├── auto_calibrate.py           # 自动标定（推荐）
├── interactive_calibrate.py    # 交互式标定
├── collect_data.py             # 数据采集
├── extract_data.py             # 数据提取
├── lib/                        # 标定算法库
│   ├── feature_extractor.py    # 特征提取
│   ├── optimizer.py            # 优化求解
│   ├── pose_selector.py        # 姿态选择
│   └── visualizer.py           # 可视化
├── config/                     # 初始外参（可选）
├── calib_data_top/             # Top Camera 标定数据
│   ├── image/
│   ├── pcd/
│   ├── conf/
│   └── visualization/
└── calib_data_chassis/         # Chassis Camera 标定数据
```

---

## 八、快速开始

```bash
# 1. 采集数据
python collect_data.py top

# 2. 提取数据
python extract_data.py calib_bags_top/data.bag top

# 3. 自动标定（推荐）
python auto_calibrate.py --data-dir ./calib_data_top --visualize

# 4. 查看结果
cat calib_data_top/conf/auto_extrinsics.yaml
```

就这么简单！
