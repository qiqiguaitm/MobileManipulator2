# Phase 1: 建图模块迁移详细方案

## 1. SC-PGO 代码分析

### 1.1 核心文件结构

```
SC-PGO/
├── src/
│   ├── laserPosegraphOptimization.cpp  # 核心! 831行 (唯一需要迁移的节点)
│   ├── scanRegistration.cpp            # 不需要 (FAST_LIO已有)
│   ├── laserOdometry.cpp               # 不需要 (FAST_LIO已有)
│   ├── laserMapping.cpp                # 不需要 (FAST_LIO已有)
│   └── kittiHelper.cpp                 # 不需要 (测试工具)
│
├── include/
│   ├── aloam_velodyne/
│   │   ├── common.h                    # 需要 (点类型定义)
│   │   └── tic_toc.h                   # 需要 (计时工具)
│   │
│   └── scancontext/
│       ├── Scancontext.h               # 需要 (回环检测核心)
│       ├── Scancontext.cpp             # 需要 (回环检测实现)
│       ├── nanoflann.hpp               # 需要 (KD树库,header-only)
│       └── KDTreeVectorOfVectorsAdaptor.h  # 需要 (KD树适配器)
```

### 1.2 laserPosegraphOptimization.cpp 分析

#### 线程架构
```
┌─────────────────────────────────────────────────────────────────┐
│                    SC-PGO 多线程架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Main Thread                                                    │
│  └── ros::spin() (消息回调)                                     │
│                                                                 │
│  Thread 1: process_pg (位姿图构建)                              │
│  ├── 接收里程计和点云                                            │
│  ├── 关键帧选择                                                  │
│  ├── 添加里程计因子                                              │
│  └── 保存关键帧点云                                              │
│                                                                 │
│  Thread 2: process_lcd (回环检测)                               │
│  ├── 1Hz频率运行                                                │
│  └── 调用Scan Context检测回环                                   │
│                                                                 │
│  Thread 3: process_icp (ICP验证)                                │
│  └── 对候选回环进行ICP精配准                                     │
│                                                                 │
│  Thread 4: process_isam (优化)                                  │
│  ├── 1Hz频率运行                                                │
│  └── 调用ISAM2进行增量优化                                       │
│                                                                 │
│  Thread 5: process_viz_map (地图可视化)                         │
│  └── 0.1Hz发布全局地图                                          │
│                                                                 │
│  Thread 6: process_viz_path (路径可视化)                        │
│  └── 10Hz发布优化后路径                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 话题接口
```
订阅:
├── /velodyne_cloud_registered_local  → sensor_msgs/PointCloud2 (配准点云)
├── /aft_mapped_to_init               → nav_msgs/Odometry (里程计)
└── /gps/fix                          → sensor_msgs/NavSatFix (GPS,可选)

发布:
├── /aft_pgo_map      → sensor_msgs/PointCloud2 (优化后全局地图) ★重要
├── /aft_pgo_odom     → nav_msgs/Odometry (优化后里程计)
├── /aft_pgo_path     → nav_msgs/Path (优化后路径)
├── /loop_scan_local  → sensor_msgs/PointCloud2 (回环当前帧,调试)
└── /loop_submap_local→ sensor_msgs/PointCloud2 (回环子图,调试)

TF:
└── camera_init → aft_pgo (优化后位姿)
```

#### 参数列表
```yaml
# 从ROS参数服务器读取
save_directory: "/path/to/save/"        # 地图保存路径
keyframe_meter_gap: 2.0                 # 关键帧距离间隔(米)
keyframe_deg_gap: 10.0                  # 关键帧角度间隔(度)
sc_dist_thres: 0.2                      # Scan Context相似度阈值
sc_max_radius: 80.0                     # Scan Context最大半径
mapviz_filter_size: 0.4                 # 地图可视化降采样大小
```

### 1.3 依赖库分析

| 库 | 用途 | ROS2可用性 |
|---|------|------------|
| **GTSAM** | iSAM2位姿图优化 | apt: libgtsam-dev |
| **PCL** | 点云处理 | apt: ros-humble-pcl-ros |
| **Ceres** | ICP优化 | apt: libceres-dev (可选) |
| **OpenMP** | 并行加速 | 系统自带 |
| **Eigen** | 矩阵运算 | apt: libeigen3-dev |
| **nanoflann** | KD树 | header-only (已包含) |

---

## 2. ROS1→ROS2 API 转换对照

### 2.1 节点结构

```cpp
// ROS1
int main(int argc, char **argv) {
    ros::init(argc, argv, "laserPGO");
    ros::NodeHandle nh;
    // ...
    ros::spin();
}

// ROS2
class ScPgoNode : public rclcpp::Node {
public:
    ScPgoNode() : Node("sc_pgo") {
        // 初始化
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ScPgoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}
```

### 2.2 订阅/发布

```cpp
// ROS1
ros::Subscriber sub = nh.subscribe<MsgType>("/topic", 100, callback);
ros::Publisher pub = nh.advertise<MsgType>("/topic", 100);

// ROS2
auto sub = create_subscription<MsgType>("/topic", 100,
    std::bind(&ScPgoNode::callback, this, std::placeholders::_1));
auto pub = create_publisher<MsgType>("/topic", 100);
```

### 2.3 参数获取

```cpp
// ROS1
nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0);

// ROS2
declare_parameter("keyframe_meter_gap", 2.0);
keyframeMeterGap = get_parameter("keyframe_meter_gap").as_double();
```

### 2.4 TF广播

```cpp
// ROS1
static tf::TransformBroadcaster br;
tf::Transform transform;
br.sendTransform(tf::StampedTransform(transform, stamp, "parent", "child"));

// ROS2
tf2_ros::TransformBroadcaster br(this);
geometry_msgs::msg::TransformStamped t;
t.header.stamp = stamp;
t.header.frame_id = "parent";
t.child_frame_id = "child";
br.sendTransform(t);
```

### 2.5 时间处理

```cpp
// ROS1
ros::Time::now()
msg->header.stamp.toSec()
ros::Rate rate(10);

// ROS2
this->now()
rclcpp::Time(msg->header.stamp).seconds()
rclcpp::Rate rate(10);
```

### 2.6 四元数转换

```cpp
// ROS1
tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw)
tf::Matrix3x3(tf::Quaternion(x, y, z, w)).getRPY(roll, pitch, yaw);

// ROS2
tf2::Quaternion q;
q.setRPY(roll, pitch, yaw);
tf2::Matrix3x3(tf2::Quaternion(x, y, z, w)).getRPY(roll, pitch, yaw);
```

---

## 3. 迁移步骤

### Step 1: 创建ROS2包结构

```bash
cd /home/didi/workspace/MobileManipulator2/src
mkdir -p sc_pgo/{include/sc_pgo,src,config,launch}
```

目标结构:
```
sc_pgo/
├── CMakeLists.txt
├── package.xml
├── include/
│   └── sc_pgo/
│       ├── common.h              # 从SC-PGO复制
│       ├── tic_toc.h             # 从SC-PGO复制
│       ├── Scancontext.h         # 从SC-PGO复制
│       ├── nanoflann.hpp         # 从SC-PGO复制
│       └── KDTreeVectorOfVectorsAdaptor.h
├── src/
│   ├── sc_pgo_node.cpp           # 主节点 (从laserPosegraphOptimization转换)
│   └── Scancontext.cpp           # 从SC-PGO复制
├── config/
│   └── sc_pgo_params.yaml        # 参数配置
└── launch/
    └── sc_pgo_launch.py
```

### Step 2: 安装依赖

```bash
# GTSAM
sudo apt install libgtsam-dev libgtsam-unstable-dev

# PCL
sudo apt install ros-humble-pcl-ros ros-humble-pcl-conversions

# Ceres (可选,用于ICP)
sudo apt install libceres-dev
```

### Step 3: 创建package.xml

```xml
<?xml version="1.0"?>
<package format="3">
  <name>sc_pgo</name>
  <version>1.0.0</version>
  <description>Scan Context Pose Graph Optimization for ROS2</description>
  <maintainer email="dev@example.com">Developer</maintainer>
  <license>BSD</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>
  <depend>pcl_ros</depend>
  <depend>pcl_conversions</depend>

  <build_depend>libgtsam-dev</build_depend>
  <exec_depend>libgtsam</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Step 4: 创建CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(sc_pgo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

add_executable(sc_pgo_node
  src/sc_pgo_node.cpp
  src/Scancontext.cpp
)

target_compile_options(sc_pgo_node PRIVATE ${OpenMP_CXX_FLAGS})

ament_target_dependencies(sc_pgo_node
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

target_link_libraries(sc_pgo_node
  ${PCL_LIBRARIES}
  gtsam
  ${OpenMP_CXX_FLAGS}
)

install(TARGETS sc_pgo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
```

### Step 5: 转换核心代码

主要转换点:

| 代码位置 | ROS1 | ROS2 |
|----------|------|------|
| 第27行 | `#include <ros/ros.h>` | `#include <rclcpp/rclcpp.hpp>` |
| 第31-32行 | `tf/transform_*` | `tf2_ros/*` |
| 第128-130行 | `ros::Publisher` | `rclcpp::Publisher<>` |
| 第191-212行 | 回调函数 | 成员函数+bind |
| 第770-831行 | main函数 | Node类+main |

### Step 6: 创建参数文件

```yaml
# config/sc_pgo_params.yaml
sc_pgo:
  ros__parameters:
    # 保存路径
    save_directory: "/home/didi/workspace/MobileManipulator2/maps/sc_pgo/"

    # 关键帧选择
    keyframe_meter_gap: 0.5    # 室内更密集
    keyframe_deg_gap: 10.0

    # Scan Context参数
    sc_dist_thres: 0.15        # 更严格的阈值
    sc_max_radius: 20.0        # 室内半径

    # 可视化
    mapviz_filter_size: 0.05   # 5cm体素

    # 话题映射 (与FAST_LIO对接)
    cloud_topic: "/cloud_registered_body"
    odom_topic: "/Odometry"
```

### Step 7: 创建Launch文件

```python
# launch/sc_pgo_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('sc_pgo')
    config_file = os.path.join(pkg_share, 'config', 'sc_pgo_params.yaml')

    return LaunchDescription([
        Node(
            package='sc_pgo',
            executable='sc_pgo_node',
            name='sc_pgo',
            output='screen',
            parameters=[config_file],
            remappings=[
                # 与FAST_LIO对接
                ('/velodyne_cloud_registered_local', '/cloud_registered_body'),
                ('/aft_mapped_to_init', '/Odometry'),
            ]
        ),
    ])
```

---

## 4. 与FAST_LIO对接

### 4.1 话题映射

| SC-PGO原话题 | FAST_LIO话题 | 说明 |
|--------------|--------------|------|
| /velodyne_cloud_registered_local | /cloud_registered_body | body坐标系点云 |
| /aft_mapped_to_init | /Odometry | 里程计 |

### 4.2 坐标系对齐

FAST_LIO使用的坐标系:
- `camera_init`: 世界坐标系原点
- `body`: 机器人坐标系

SC-PGO需要的坐标系:
- `camera_init`: 与FAST_LIO一致 ✅
- 点云在body坐标系: 与FAST_LIO一致 ✅

---

## 5. Octomap配置

### 5.1 安装

```bash
sudo apt install ros-humble-octomap-server ros-humble-octomap-msgs
```

### 5.2 配置文件

```yaml
# config/octomap_params.yaml
octomap_server:
  ros__parameters:
    frame_id: "camera_init"
    resolution: 0.05               # 5cm分辨率
    sensor_model:
      max_range: 20.0
      hit: 0.7
      miss: 0.4
      min: 0.12
      max: 0.97
    occupancy_min_z: 0.1           # 过滤地面
    occupancy_max_z: 1.2           # 过滤天花板
    project_2d_map: true           # 投影为2D地图
    latch: true
```

### 5.3 Launch整合

```python
# 在mapping_launch.py中添加
Node(
    package='octomap_server',
    executable='octomap_server_node',
    name='octomap_server',
    parameters=[octomap_config],
    remappings=[
        ('cloud_in', '/aft_pgo_map'),  # 订阅SC-PGO输出
    ]
),
```

---

## 6. 测试验证

### 6.1 单元测试

```bash
# 1. 构建
cd /home/didi/workspace/MobileManipulator2
colcon build --packages-select sc_pgo

# 2. 启动FAST_LIO
ros2 launch slam fastlio_odom_launch.py

# 3. 启动SC-PGO
ros2 launch sc_pgo sc_pgo_launch.py

# 4. 验证话题
ros2 topic list | grep pgo
ros2 topic hz /aft_pgo_map
```

### 6.2 集成测试

```bash
# 完整建图流程
ros2 launch slam mapping_launch.py

# 移动机器人建图
# ...

# 验证输出
ls /home/didi/workspace/MobileManipulator2/maps/sc_pgo/
# 应该有: GlobalMap.pcd, optimized_poses.txt, Scans/
```

---

## 7. 时间估计

| 任务 | 预计时间 |
|------|----------|
| 创建包结构 | 0.5天 |
| 安装依赖 | 0.5天 |
| 代码转换 | 2天 |
| 调试修复 | 1天 |
| Octomap配置 | 0.5天 |
| 集成测试 | 0.5天 |
| **总计** | **5天** |

---

## 8. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| GTSAM编译问题 | 使用apt预编译版本 |
| 多线程同步问题 | 保持原有mutex结构 |
| TF时间戳问题 | 使用tf2的buffer机制 |
| 性能问题(ARM) | 降低可视化频率 |

---

## 下一步

确认后开始执行 **Step 1: 创建ROS2包结构**
