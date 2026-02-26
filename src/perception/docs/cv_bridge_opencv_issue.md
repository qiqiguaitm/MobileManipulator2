# cv_bridge OpenCV 版本兼容性问题

## 问题描述

在使用 `cv_bridge.imgmsg_to_cv2()` 进行颜色空间转换时（如 `rgb8` → `bgr8`），会触发 **段错误 (Segmentation Fault)**。

### 错误现象

```
Fatal Python error: Segmentation fault

Current thread (most recent call first):
  File ".../cv_bridge/core.py", line 186 in imgmsg_to_cv2
  ...
```

### 根本原因

`cv_bridge_boost.so` 链接了**多个不兼容版本**的 OpenCV 库：

```bash
$ ldd /opt/ros/noetic/lib/python3/dist-packages/cv_bridge/boost/cv_bridge_boost.so | grep opencv

libopencv_core.so.4.2      → /lib/aarch64-linux-gnu/          # 系统包
libopencv_imgcodecs.so.4.5 → /usr/local/lib/                  # 手动安装
libopencv_imgproc.so.4.5   → /usr/local/lib/                  # 手动安装
libopencv_core.so.4.5      → /usr/local/lib/                  # 手动安装
```

同时，Python 使用的 OpenCV 版本为 **4.12.0**：

```bash
$ python3 -c "import cv2; print(cv2.__version__)"
4.12.0
```

这种版本混乱导致 **ABI 不兼容**，当 `cv_bridge_boost.cvtColor2()` 被调用时会崩溃。

## 影响范围

所有使用 `imgmsg_to_cv2(msg, 'bgr8')` 或类似颜色转换的代码：

| 文件 | 调用 | 影响 |
|------|------|------|
| `cam_lidar_calibrate/extract_data.py` | `imgmsg_to_cv2(msg, "bgr8")` | ⚠️ |
| `cam_lidar_calibrate/lib/realtime_checker.py` | `imgmsg_to_cv2(msg, "bgr8")` | ⚠️ |
| `rtabmap_ros/.../netvlad_tf_ros.py` | `imgmsg_to_cv2(data, "rgb8")` | ⚠️ |

## 解决方案

### 方案 1: 使用 passthrough + cv2.cvtColor (推荐)

绕过 `cv_bridge_boost`，使用 OpenCV 原生函数进行颜色转换：

```python
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

# ❌ 会崩溃
# rgb = bridge.imgmsg_to_cv2(msg, 'bgr8')

# ✅ 安全方式
rgb_raw = bridge.imgmsg_to_cv2(msg, 'passthrough')
if msg.encoding == 'rgb8':
    rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
elif msg.encoding == 'bgr8':
    rgb = rgb_raw
else:
    rgb = rgb_raw  # 其他编码原样返回
```

### 方案 2: 使用原始编码

如果不需要颜色转换，直接使用消息的原始编码：

```python
# 安全：不触发颜色转换
img = bridge.imgmsg_to_cv2(msg, msg.encoding)
```

### 方案 3: 统一 OpenCV 版本 (长期)

1. 卸载冲突的 OpenCV 版本
2. 使用统一版本重新编译 cv_bridge

```bash
# 检查已安装的 OpenCV
dpkg -l | grep opencv
pip list | grep opencv

# 清理冲突版本
sudo apt remove libopencv*
pip uninstall opencv-python opencv-python-headless

# 安装统一版本
pip install opencv-python==4.5.5.64

# 重新编译 cv_bridge
cd ~/catkin_ws
catkin build cv_bridge --force-cmake
```

## 验证修复

```python
# 测试脚本
from cv_bridge import CvBridge
import cv2
import numpy as np

bridge = CvBridge()

# 模拟 ROS 消息
class FakeMsg:
    encoding = 'rgb8'
    data = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
    height = 480
    width = 640
    step = 640 * 3
    is_bigendian = 0

msg = FakeMsg()

# 安全方式
img = bridge.imgmsg_to_cv2(msg, 'passthrough')
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print(f"转换成功: {img_bgr.shape}")
```

## 相关文件

- `src/perception/src/synced_sensor_subscriber.py` - 已应用修复
- `/opt/ros/noetic/lib/python3/dist-packages/cv_bridge/` - ROS 官方 cv_bridge
- `/home/agilex/MobileManipulator/src/vision_opencv/cv_bridge/` - 项目自编译版本

## 参考

- [ROS cv_bridge 文档](http://wiki.ros.org/cv_bridge)
- [OpenCV ABI 兼容性说明](https://docs.opencv.org/4.x/d1/dfb/intro.html)
