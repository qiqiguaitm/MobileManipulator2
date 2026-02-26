# Perception2 手动测试指南

本指南提供分层测试方案，从基础验证到完整功能测试。

---

## 测试前准备

```bash
# 每个终端都需要执行
cd /data/workspace/MobileManipulator2
source /opt/ros/humble/setup.bash
source install/setup.bash
```

---

## 第一层：基础验证（无需硬件）

### 1.1 检查包安装

```bash
# 检查所有包是否正确安装
ros2 pkg list | grep perception

# 预期输出:
# perception_bringup
# perception_core
# perception_interfaces
# perception_nodes
```

### 1.2 检查节点可执行文件

```bash
# 列出所有可用节点
ros2 pkg executables perception_nodes

# 预期输出 (7个节点):
# perception_nodes multi_sensor_perception_node
# perception_nodes object_tracker_node
# perception_nodes perception_grasp_node
# perception_nodes perception_grasp_rviz_node
# perception_nodes perception_rviz_node
# perception_nodes perception_viz_node
# perception_nodes scene_perception_3d_node
```

### 1.3 检查消息和服务类型

```bash
# 检查消息定义
ros2 interface list | grep perception_interfaces

# 查看具体消息结构
ros2 interface show perception_interfaces/msg/Object3D
ros2 interface show perception_interfaces/srv/DetectObjects
```

### 1.4 检查 Launch 文件

```bash
# 列出可用的 launch 文件
ros2 pkg prefix perception_bringup
ls $(ros2 pkg prefix perception_bringup)/share/perception_bringup/launch/

# 预期输出 (6个):
# object_tracker.launch.py
# perception_3d_rviz.launch.py
# perception_grasp.launch.py
# perception_grasp_rviz.launch.py
# perception_tracker_rviz.launch.py
# scene_perception_3d.launch.py
```

---

## 第二层：节点启动测试（无需外部服务）

### 2.1 测试 scene_perception_3d_node 启动

**终端1 - 启动节点：**
```bash
ros2 run perception_nodes scene_perception_3d_node --ros-args \
  -p auto_detect_rate:=0.0 \
  -p enable_depth_optimizer:=false
```

**终端2 - 检查话题和服务：**
```bash
# 检查节点是否运行
ros2 node list | grep scene_perception

# 检查发布的话题
ros2 topic list | grep scene_perception

# 预期话题:
# /scene_perception_3d/objects_3d
# /scene_perception_3d/optimized_depth

# 检查服务
ros2 service list | grep scene_perception

# 预期服务:
# /scene_perception_3d/detect
```

**验证标准：** 节点启动无报错，话题和服务正确注册

### 2.2 测试 perception_grasp_node 启动

**终端1：**
```bash
ros2 run perception_nodes perception_grasp_node
```

**终端2：**
```bash
ros2 node list | grep grasp
ros2 topic list | grep grasp
ros2 service list | grep grasp

# 预期服务:
# /perception_grasp_node/detect
```

### 2.3 测试 object_tracker_node 启动

**终端1：**
```bash
ros2 run perception_nodes object_tracker_node --ros-args \
  -p track_rate:=1.0
```

**终端2：**
```bash
ros2 node list | grep tracker
ros2 topic list | grep tracker

# 预期话题:
# /object_tracker/tracked_objects
```

---

## 第三层：Launch 文件测试

### 3.1 测试 scene_perception_3d.launch.py

```bash
# 终端1: 启动
ros2 launch perception_bringup scene_perception_3d.launch.py \
  auto_detect_rate:=0.0 \
  enable_depth_optimizer:=false

# 终端2: 验证
ros2 node list
ros2 topic list
ros2 service list
```

### 3.2 测试带 Tracker 的 Launch

```bash
# 终端1: 启动场景感知 + 跟踪
ros2 launch perception_bringup scene_perception_3d.launch.py \
  enable_tracker:=true \
  auto_detect_rate:=0.0

# 终端2: 验证两个节点都启动
ros2 node list
# 预期: scene_perception_3d, object_tracker
```

---

## 第四层：模拟数据测试

### 4.1 发布模拟相机数据

**如果没有真实相机，可以发布模拟图像：**

```bash
# 创建测试脚本
cat > /tmp/publish_test_image.py << 'EOF'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.bridge = CvBridge()

        # 发布器
        self.rgb_pub = self.create_publisher(Image, '/camera/top/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/top/aligned_depth_to_color/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/top/color/camera_info', 10)

        # 定时器 (1Hz)
        self.timer = self.create_timer(1.0, self.publish_callback)
        self.get_logger().info('Test image publisher started')

    def publish_callback(self):
        # 创建测试图像 (640x480)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(500, 2000, (480, 640), dtype=np.uint16)  # 0.5m - 2m

        # 发布
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, 'bgr8')
        rgb_msg.header.stamp = self.get_clock().now().to_msg()
        rgb_msg.header.frame_id = 'camera_top_color_optical_frame'
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, '16UC1')
        depth_msg.header = rgb_msg.header
        self.depth_pub.publish(depth_msg)

        # Camera Info
        info = CameraInfo()
        info.header = rgb_msg.header
        info.width = 640
        info.height = 480
        info.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)

        self.get_logger().info('Published test images')

def main():
    rclpy.init()
    node = TestImagePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF

# 运行
python3 /tmp/publish_test_image.py
```

### 4.2 验证数据接收

```bash
# 检查图像话题
ros2 topic hz /camera/top/color/image_raw
ros2 topic hz /camera/top/aligned_depth_to_color/image_raw

# 预期: 约 1 Hz
```

---

## 第五层：服务调用测试（需要外部检测服务）

### 5.1 检查外部服务连接

```bash
# 检查检测服务是否可达 (DinoX)
curl -s http://192.168.112.14:11085/health || echo "DinoX 服务不可用"

# 检查跟踪服务 (SAM2)
curl -s http://192.168.112.14:11086/health || echo "SAM2 服务不可用"

# 检查抓取服务
curl -s http://192.168.112.14:12086/health || echo "Grasp 服务不可用"
```

### 5.2 调用检测服务

**如果外部服务可用且相机数据正常：**

```bash
# 终端1: 启动节点
ros2 launch perception_bringup scene_perception_3d.launch.py

# 终端2: 调用检测服务
ros2 service call /scene_perception_3d/detect perception_interfaces/srv/DetectObjects "{
  prompt: 'bottle, cup, keyboard',
  min_score: 0.3
}"
```

### 5.3 调用抓取检测服务

```bash
# 终端1: 启动抓取节点
ros2 launch perception_bringup perception_grasp.launch.py

# 终端2: 调用服务
ros2 service call /perception_grasp_node/detect perception_interfaces/srv/GraspDetect "{
  prompt: 'bottle',
  min_score: 0.3
}"
```

---

## 第六层：完整管道测试（需要真实硬件）

### 6.1 启动完整感知系统

```bash
# 终端1: 启动相机驱动 (RealSense)
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=top \
  enable_depth:=true \
  align_depth.enable:=true

# 终端2: 启动感知节点
ros2 launch perception_bringup scene_perception_3d.launch.py \
  camera_name:=top \
  auto_detect_rate:=1.0

# 终端3: 监控输出
ros2 topic echo /scene_perception_3d/objects_3d
```

### 6.2 启动带可视化的完整系统

```bash
# 终端1: 相机驱动
ros2 launch realsense2_camera rs_launch.py camera_name:=top

# 终端2: 感知 + RViz
ros2 launch perception_bringup perception_3d_rviz.launch.py

# 在 RViz 中检查:
# - PointCloud2 显示
# - MarkerArray 显示
# - 检测结果标注
```

---

## 快速测试脚本

将以下脚本保存为可执行文件，一键运行基础测试：

```bash
#!/bin/bash
# 保存为: test_perception2.sh

echo "=========================================="
echo "Perception2 快速测试"
echo "=========================================="

source /opt/ros/humble/setup.bash
source install/setup.bash

echo ""
echo "[1/5] 检查包安装..."
PKGS=$(ros2 pkg list | grep -c perception)
if [ "$PKGS" -eq 4 ]; then
    echo "✅ 4个包已安装"
else
    echo "❌ 包安装不完整 (found $PKGS)"
    exit 1
fi

echo ""
echo "[2/5] 检查节点..."
NODES=$(ros2 pkg executables perception_nodes | wc -l)
if [ "$NODES" -eq 7 ]; then
    echo "✅ 7个节点可用"
else
    echo "❌ 节点数量不对 (found $NODES)"
    exit 1
fi

echo ""
echo "[3/5] 检查消息类型..."
MSGS=$(ros2 interface list | grep -c perception_interfaces)
if [ "$MSGS" -ge 11 ]; then
    echo "✅ 消息/服务定义正常"
else
    echo "❌ 消息定义不完整"
    exit 1
fi

echo ""
echo "[4/5] 检查Launch文件..."
LAUNCHES=$(ls $(ros2 pkg prefix perception_bringup)/share/perception_bringup/launch/*.py 2>/dev/null | wc -l)
if [ "$LAUNCHES" -eq 6 ]; then
    echo "✅ 6个Launch文件可用"
else
    echo "❌ Launch文件不完整 (found $LAUNCHES)"
    exit 1
fi

echo ""
echo "[5/5] 检查配置文件..."
CONFIGS=$(ls $(ros2 pkg prefix perception_bringup)/share/perception_bringup/config/ 2>/dev/null | wc -l)
if [ "$CONFIGS" -ge 20 ]; then
    echo "✅ $CONFIGS 个配置文件已安装"
else
    echo "❌ 配置文件不完整 (found $CONFIGS)"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 所有基础检查通过!"
echo "=========================================="
echo ""
echo "下一步测试建议:"
echo "  1. 启动节点: ros2 launch perception_bringup scene_perception_3d.launch.py"
echo "  2. 检查话题: ros2 topic list | grep perception"
echo "  3. 检查服务: ros2 service list | grep perception"
```

---

## 测试结果记录表

| 测试项 | 预期结果 | 实际结果 | 通过 |
|--------|----------|----------|------|
| 包安装检查 | 4个包 | | □ |
| 节点检查 | 7个节点 | | □ |
| 消息定义 | 11个 | | □ |
| Launch文件 | 6个 | | □ |
| 配置文件 | 22个 | | □ |
| scene_perception_3d 启动 | 无报错 | | □ |
| perception_grasp 启动 | 无报错 | | □ |
| object_tracker 启动 | 无报错 | | □ |
| 检测服务调用 | 返回结果 | | □ |
| 抓取服务调用 | 返回结果 | | □ |

---

## 常见问题

### Q: NumPy 版本不兼容错误

如果看到以下错误：
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**解决方案：** 降级 NumPy 到 1.x 版本
```bash
pip install 'numpy<2'
```

这是因为 ROS2 Humble 的 cv_bridge 是用 NumPy 1.x 编译的。

### Q: 节点启动报错 "ModuleNotFoundError"
```bash
# 确保已 source 环境
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Q: 服务调用超时
检查外部检测服务是否运行：
```bash
curl http://192.168.112.14:11085/health
```

### Q: 没有相机数据
使用第四层的模拟数据发布脚本进行测试。
