# Approach Navigator

三阶段精确接近导航器，从 ROS1 `ThreeStageNavigator` 迁移而来。

## 代码结构

```
approach_navigator/
├── __init__.py         # 公共接口导出
├── types.py            # NavStage, ApproachResult 类型定义
├── config.py           # ApproachConfig 配置类
├── depth_sensor.py     # 深度传感器封装
├── navigator.py        # 核心三阶段导航器
├── utils.py            # 工具函数 (compute_approach_pose 等)
└── approach_navigator_node.py  # ROS2 节点
```

## 核心改动

| 项目 | ROS1 | ROS2 |
|------|------|------|
| Stage 1 | move_base Action | Nav2 BasicNavigator |
| Stage 2 | PD 对齐 | PD 对齐 (相同) |
| Stage 3 测距 | LiDAR 前方扇区 | Camera 深度图 ROI |
| 框架 | rospy | rclpy |

## 三阶段架构

```
Stage 1 (Navigate): Nav2 导航到 Approach Pose (距目标 45cm)
    └── 必须 > local_costmap inflation_radius (35cm)

Stage 2 (Align): PD 控制器原地旋转对齐
    └── 容差 5°

Stage 3 (Final Approach): Camera 深度测距 + 低速靠近
    └── 目标距离 8cm，紧急停止 3cm
```

## 使用方法

### 1. 启动导航器

```bash
ros2 launch approach_navigator approach_navigator.launch.py
```

### 2. 在代码中使用

```python
from approach_navigator import ApproachNavigator, compute_approach_pose
from geometry_msgs.msg import Point, PoseStamped

# 创建导航器
navigator = ApproachNavigator()

# 计算 approach pose
target = Point(x=1.0, y=0.0, z=0.0)
robot_pos = Point(x=0.0, y=0.0, z=0.0)
approach_pose = compute_approach_pose(target, robot_pos)

# 执行三阶段导航
result = navigator.approach_to_pose(approach_pose, target)

if result.success:
    print(f"到达目标，距离: {result.final_distance:.3f}m")
else:
    print(f"失败: {result.error_code}")
```

### 3. 测试深度测量

```bash
python3 scripts/_cc_test_approach_navigator.py --depth-only
```

## 配置参数

参见 `config/approach_navigator.yaml`

关键参数：
- `approach_distance`: Stage 1 目标距离 (默认 0.45m)
- `final_approach_distance`: Stage 3 最终距离 (默认 0.08m)
- `depth_topic`: 深度图话题
- `camera_forward_offset`: 相机前向偏移

## 依赖

- nav2_simple_commander
- tf2_ros
- cv_bridge
- sensor_msgs (Image)
