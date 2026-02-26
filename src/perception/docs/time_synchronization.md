# 多传感器时间同步方案

> **版本**: v1.0
> **日期**: 2026-01-30
> **类型**: 技术文档
> **相关**: [多传感器感知设计](./multi_sensor_perception_design.md)

---

## 目录

1. [核心问题](#1-核心问题)
2. [同步策略](#2-同步策略)
3. [参数配置](#3-参数配置)
4. [实现方案](#4-实现方案)
5. [诊断与监控](#5-诊断与监控)
6. [边缘情况处理](#6-边缘情况处理)
7. [最佳实践](#7-最佳实践)

---

## 1. 核心问题

### 1.1 问题描述

```text
系统配置：
├─ Chassis Camera (RealSense D435)
│  ├─ RGB: 30 Hz (33ms周期)
│  └─ Depth: 30 Hz (33ms周期)
│
├─ Top Camera (RealSense D435)
│  ├─ RGB: 30 Hz (33ms周期)
│  └─ Depth: 30 Hz (33ms周期)
│
└─ LiDAR (RSHELIOS 16P)
   └─ PointCloud: 10 Hz (100ms周期)

挑战：
1. 传感器频率不同（30Hz vs 10Hz）
2. 采集时刻不同步（硬件触发独立）
3. 通信延迟不确定（ROS消息传输）
4. 处理延迟不同（检测300ms vs 测量10ms）
```

### 1.2 同步目标

| 目标 | 指标 | 说明 |
|------|------|------|
| 时间精度 | < 100ms | 传感器数据时间差 |
| 同步频率 | 5 Hz | 感知系统输出频率 |
| 丢帧率 | < 5% | 同步失败导致的丢帧 |
| 延迟 | < 50ms | 同步机制本身的延迟 |

---

## 2. 同步策略

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     时间同步三层架构                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Layer 1    │      │   Layer 2    │      │   Layer 3    │
│  时间戳标注  │      │  消息过滤    │      │  同步触发    │
│              │      │              │      │              │
│ - 驱动层     │      │ - message_   │      │ - 定时器     │
│ - 采集时刻   │      │   filters    │      │ - 数据缓存   │
│ - ROS时间    │      │ - 容差匹配   │      │ - 主动触发   │
└──────────────┘      └──────────────┘      └──────────────┘
```

### 2.2 Layer 1: 时间戳标注

**目标**：确保每个消息有准确的时间戳

**实现**：传感器驱动层

```python
# RealSense驱动（示例）
def publish_frame(self, frame):
    """发布相机帧"""
    msg = Image()

    # ✅ 正确：使用帧的采集时刻
    msg.header.stamp = rospy.Time.from_sec(frame.get_timestamp())
    msg.header.frame_id = "camera_chassis_color_optical_frame"

    # ❌ 错误：使用发布时刻
    # msg.header.stamp = rospy.Time.now()  # 错！会包含处理延迟

    self.publisher.publish(msg)
```

**关键点**：
- 时间戳应是**数据采集时刻**，不是处理或发布时刻
- 使用ROS标准时间（rospy.Time）
- 所有传感器使用同一时钟源（系统时间）

---

### 2.3 Layer 2: 消息过滤（message_filters）

**目标**：匹配时间戳接近的消息组合

#### 方案A: 全局同步（简单场景）

```python
import message_filters
from sensor_msgs.msg import Image, PointCloud2

class GlobalSyncNode:
    """全局同步方案（5个传感器一起同步）"""

    def __init__(self):
        # 订阅所有传感器
        subs = [
            message_filters.Subscriber('/camera/chassis/color/image_raw', Image),
            message_filters.Subscriber('/camera/chassis/aligned_depth_to_color/image_raw', Image),
            message_filters.Subscriber('/camera/top/color/image_raw', Image),
            message_filters.Subscriber('/camera/top/aligned_depth_to_color/image_raw', Image),
            message_filters.Subscriber('/rslidar_points', PointCloud2),
        ]

        # ApproximateTimeSynchronizer
        # - 允许时间差（slop参数）
        # - 找"最接近"的消息组合
        sync = message_filters.ApproximateTimeSynchronizer(
            subs,
            queue_size=10,           # 每个topic缓存10条消息
            slop=0.1,                # 时间容差100ms
            allow_headerless=False   # 必须有header
        )
        sync.registerCallback(self.sync_callback)

        rospy.loginfo("Global sync initialized (slop=100ms)")

    def sync_callback(self, chassis_rgb, chassis_depth,
                      top_rgb, top_depth, lidar):
        """
        所有传感器同步回调

        保证：
        - 5个消息的时间戳在100ms内
        - 是最接近的组合
        """
        # 检查时间跨度
        timestamps = [
            chassis_rgb.header.stamp,
            chassis_depth.header.stamp,
            top_rgb.header.stamp,
            top_depth.header.stamp,
            lidar.header.stamp
        ]

        time_span = (max(timestamps) - min(timestamps)).to_sec()
        rospy.logdebug(f"Sync time span: {time_span*1000:.1f}ms")

        # 执行感知
        self.perception_pipeline(
            chassis_rgb, chassis_depth,
            top_rgb, top_depth, lidar
        )
```

**优点**：
- 实现简单（30行代码）
- ROS标准方案，稳定可靠

**缺点**：
- LiDAR频率低（10Hz），会拖累整体频率
- 5个传感器都必须有消息，任何一个失效就无法同步

---

#### 方案B: 分层同步（推荐）

```python
import threading
from collections import deque

class LayeredSyncNode:
    """
    分层同步方案

    Layer 1: 双相机内部同步（RGB+Depth）
    Layer 2: LiDAR独立缓存
    Layer 3: 定时触发感知
    """

    def __init__(self):
        # ===== Layer 1: Chassis相机同步 =====
        chassis_rgb_sub = message_filters.Subscriber(
            '/camera/chassis/color/image_raw', Image
        )
        chassis_depth_sub = message_filters.Subscriber(
            '/camera/chassis/aligned_depth_to_color/image_raw', Image
        )

        chassis_sync = message_filters.ApproximateTimeSynchronizer(
            [chassis_rgb_sub, chassis_depth_sub],
            queue_size=10,
            slop=0.05  # 50ms容差（同一相机，时间差小）
        )
        chassis_sync.registerCallback(self.chassis_callback)

        # ===== Layer 1: Top相机同步 =====
        top_rgb_sub = message_filters.Subscriber(
            '/camera/top/color/image_raw', Image
        )
        top_depth_sub = message_filters.Subscriber(
            '/camera/top/aligned_depth_to_color/image_raw', Image
        )

        top_sync = message_filters.ApproximateTimeSynchronizer(
            [top_rgb_sub, top_depth_sub],
            queue_size=10,
            slop=0.05
        )
        top_sync.registerCallback(self.top_callback)

        # ===== Layer 2: LiDAR独立订阅 =====
        rospy.Subscriber('/rslidar_points', PointCloud2,
                        self.lidar_callback, queue_size=5)

        # 数据缓存（线程安全）
        self.chassis_data = None
        self.top_data = None
        self.lidar_data = None
        self.data_lock = threading.Lock()

        # ===== Layer 3: 定时触发（5Hz）=====
        self.perception_timer = rospy.Timer(
            rospy.Duration(0.2),  # 200ms = 5Hz
            self.perception_callback
        )

        rospy.loginfo("Layered sync initialized (5Hz)")

    def chassis_callback(self, rgb, depth):
        """Chassis相机同步回调"""
        with self.data_lock:
            self.chassis_data = {
                'rgb': self.bridge.imgmsg_to_cv2(rgb, 'rgb8'),
                'depth': self.bridge.imgmsg_to_cv2(depth, 'passthrough'),
                'timestamp': rgb.header.stamp,
                'frame_id': rgb.header.frame_id
            }

    def top_callback(self, rgb, depth):
        """Top相机同步回调"""
        with self.data_lock:
            self.top_data = {
                'rgb': self.bridge.imgmsg_to_cv2(rgb, 'rgb8'),
                'depth': self.bridge.imgmsg_to_cv2(depth, 'passthrough'),
                'timestamp': rgb.header.stamp,
                'frame_id': rgb.header.frame_id
            }

    def lidar_callback(self, msg):
        """LiDAR回调（独立）"""
        with self.data_lock:
            self.lidar_data = {
                'msg': msg,  # 保留原始消息（后续转换）
                'timestamp': msg.header.stamp,
                'frame_id': msg.header.frame_id
            }

    def perception_callback(self, event):
        """
        定时感知触发（5Hz）

        使用最新的传感器数据
        """
        with self.data_lock:
            # 检查数据就绪
            if (self.chassis_data is None or
                self.top_data is None or
                self.lidar_data is None):
                rospy.logwarn_throttle(5.0, "Waiting for sensor data...")
                return

            # 检查数据新鲜度
            now = rospy.Time.now()
            chassis_age = (now - self.chassis_data['timestamp']).to_sec()
            top_age = (now - self.top_data['timestamp']).to_sec()
            lidar_age = (now - self.lidar_data['timestamp']).to_sec()

            # 数据过期检查（500ms阈值）
            if chassis_age > 0.5:
                rospy.logwarn(f"Chassis data stale: {chassis_age*1000:.0f}ms")
                return
            if top_age > 0.5:
                rospy.logwarn(f"Top data stale: {top_age*1000:.0f}ms")
                return
            if lidar_age > 0.5:
                rospy.logwarn(f"LiDAR data stale: {lidar_age*1000:.0f}ms")
                return

            # 复制数据（释放锁，避免阻塞回调）
            chassis = self.chassis_data.copy()
            top = self.top_data.copy()
            lidar = self.lidar_data.copy()

        # 日志：时间戳信息
        rospy.logdebug(
            f"Perception triggered: "
            f"chassis={chassis_age*1000:.0f}ms, "
            f"top={top_age*1000:.0f}ms, "
            f"lidar={lidar_age*1000:.0f}ms"
        )

        # 执行感知pipeline
        try:
            self.perception_pipeline(chassis, top, lidar)
        except Exception as e:
            rospy.logerr(f"Perception failed: {e}")
```

**优点**：
- ✅ 双相机独立同步（不受LiDAR影响）
- ✅ LiDAR低频不影响整体频率
- ✅ 定时触发，频率可控（5Hz）
- ✅ 容错性好（单传感器失效可降级）
- ✅ 数据新鲜度可控（500ms过期）

**缺点**：
- 代码稍复杂（~100行）
- 需要手动管理数据缓存

---

## 3. 参数配置

### 3.1 slop参数（时间容差）

**定义**：message_filters允许的最大时间差

**计算公式**：

```
slop ≥ max(传感器周期差, 系统抖动, 通信延迟)
```

**你的场景**：

```text
相机周期: 33ms (30Hz)
LiDAR周期: 100ms (10Hz)
系统抖动: ~20ms (ROS通信 + 驱动延迟)
通信延迟: ~10ms (LAN)

建议：
├─ 双相机同步: slop = 0.05s (50ms)
│  原因：同一类型传感器，时间差小
│
└─ 相机+LiDAR: slop = 0.1s (100ms)
   原因：频率差异大，需要更大容差
```

**权衡表**：

| slop | 时间精度 | 丢帧率 | 适用场景 |
|------|---------|--------|---------|
| 20ms | ⭐⭐⭐ | ❌ 高（>20%） | 硬件同步 |
| 50ms | ⭐⭐ | ✅ 低（<5%） | **双相机（推荐）** |
| 100ms | ⭐ | ✅ 极低（<2%） | **相机+LiDAR（推荐）** |
| 200ms | ❌ 差 | ✅ 几乎不丢 | 低频传感器 |

---

### 3.2 queue_size参数

**定义**：每个传感器缓存的消息数量

**计算公式**：

```
queue_size = 传感器频率(Hz) × slop(s) × 安全系数

安全系数：
- 稳定网络：2
- 不稳定网络：3-5
```

**你的场景**：

```text
相机频率: 30Hz
slop: 0.1s
安全系数: 2

queue_size = 30 × 0.1 × 2 = 6

建议：queue_size = 10（向上取整，保守）
```

**影响**：

| queue_size | 内存占用 | 丢帧风险 | 延迟 |
|-----------|---------|---------|------|
| 2 | 低 | 高 | 低 |
| 5 | 中 | 中 | 中 |
| **10** | **中** | **低** | **中（推荐）** |
| 20 | 高 | 极低 | 高 |

---

### 3.3 配置对比

| 方案 | slop | queue_size | 丢帧率 | 延迟 |
|------|------|-----------|--------|------|
| 全局同步 | 100ms | 10 | ~3% | ~50ms |
| **分层同步（推荐）** | **50ms/100ms** | **10** | **<2%** | **<30ms** |

---

## 4. 实现方案

### 4.1 完整代码（分层同步）

```python
#!/usr/bin/env python3
"""
多传感器分层时间同步节点
"""

import rospy
import threading
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from collections import deque
import numpy as np


class MultiSensorSyncNode:
    """多传感器分层同步节点"""

    def __init__(self):
        rospy.init_node('multi_sensor_sync_node')

        self.bridge = CvBridge()
        self.data_lock = threading.Lock()

        # 数据缓存
        self.chassis_data = None
        self.top_data = None
        self.lidar_data = None

        # 时间戳监控
        self.last_perception_time = None
        self.perception_intervals = deque(maxlen=100)

        # ===== Chassis相机同步 =====
        chassis_rgb_sub = message_filters.Subscriber(
            '/camera/chassis/color/image_raw', Image
        )
        chassis_depth_sub = message_filters.Subscriber(
            '/camera/chassis/aligned_depth_to_color/image_raw', Image
        )
        chassis_sync = message_filters.ApproximateTimeSynchronizer(
            [chassis_rgb_sub, chassis_depth_sub],
            queue_size=10,
            slop=0.05
        )
        chassis_sync.registerCallback(self.chassis_callback)

        # ===== Top相机同步 =====
        top_rgb_sub = message_filters.Subscriber(
            '/camera/top/color/image_raw', Image
        )
        top_depth_sub = message_filters.Subscriber(
            '/camera/top/aligned_depth_to_color/image_raw', Image
        )
        top_sync = message_filters.ApproximateTimeSynchronizer(
            [top_rgb_sub, top_depth_sub],
            queue_size=10,
            slop=0.05
        )
        top_sync.registerCallback(self.top_callback)

        # ===== LiDAR订阅 =====
        rospy.Subscriber('/rslidar_points', PointCloud2,
                        self.lidar_callback, queue_size=5)

        # ===== 定时感知触发 =====
        perception_rate = rospy.get_param('~perception_rate', 5.0)  # Hz
        self.perception_timer = rospy.Timer(
            rospy.Duration(1.0 / perception_rate),
            self.perception_callback
        )

        rospy.loginfo(f"Multi-sensor sync node started ({perception_rate}Hz)")

    def chassis_callback(self, rgb_msg, depth_msg):
        """Chassis相机同步回调"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')

            if depth_msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
                depth = depth.astype(np.float32) / 1000.0  # mm -> m
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

            with self.data_lock:
                self.chassis_data = {
                    'rgb': rgb,
                    'depth': depth,
                    'timestamp': rgb_msg.header.stamp,
                    'frame_id': rgb_msg.header.frame_id
                }
        except Exception as e:
            rospy.logerr(f"Chassis callback error: {e}")

    def top_callback(self, rgb_msg, depth_msg):
        """Top相机同步回调"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')

            if depth_msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

            with self.data_lock:
                self.top_data = {
                    'rgb': rgb,
                    'depth': depth,
                    'timestamp': rgb_msg.header.stamp,
                    'frame_id': rgb_msg.header.frame_id
                }
        except Exception as e:
            rospy.logerr(f"Top callback error: {e}")

    def lidar_callback(self, msg):
        """LiDAR回调"""
        with self.data_lock:
            self.lidar_data = {
                'msg': msg,
                'timestamp': msg.header.stamp,
                'frame_id': msg.header.frame_id
            }

    def perception_callback(self, event):
        """定时感知触发"""
        # 检查数据就绪
        with self.data_lock:
            if (self.chassis_data is None or
                self.top_data is None or
                self.lidar_data is None):
                rospy.logwarn_throttle(5.0, "Waiting for all sensors...")
                return

            # 检查数据新鲜度
            now = rospy.Time.now()
            chassis_age = (now - self.chassis_data['timestamp']).to_sec()
            top_age = (now - self.top_data['timestamp']).to_sec()
            lidar_age = (now - self.lidar_data['timestamp']).to_sec()

            max_age = 0.5  # 500ms
            if chassis_age > max_age or top_age > max_age or lidar_age > max_age:
                rospy.logwarn(
                    f"Stale data: chassis={chassis_age*1000:.0f}ms, "
                    f"top={top_age*1000:.0f}ms, lidar={lidar_age*1000:.0f}ms"
                )
                return

            # 复制数据
            chassis = self.chassis_data.copy()
            top = self.top_data.copy()
            lidar = self.lidar_data.copy()

        # 监控感知频率
        if self.last_perception_time is not None:
            interval = (now - self.last_perception_time).to_sec()
            self.perception_intervals.append(interval)

            if len(self.perception_intervals) >= 10:
                mean_interval = np.mean(self.perception_intervals)
                std_interval = np.std(self.perception_intervals)
                rospy.logdebug(
                    f"Perception interval: {mean_interval*1000:.0f}±{std_interval*1000:.0f}ms"
                )

        self.last_perception_time = now

        # 执行感知
        try:
            self.process(chassis, top, lidar)
        except Exception as e:
            rospy.logerr(f"Perception error: {e}", exc_info=True)

    def process(self, chassis, top, lidar):
        """感知处理（子类实现）"""
        rospy.loginfo(
            f"Processing: chassis_age={chassis['timestamp']}, "
            f"top_age={top['timestamp']}, lidar_age={lidar['timestamp']}"
        )


if __name__ == '__main__':
    try:
        node = MultiSensorSyncNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

---

## 5. 诊断与监控

### 5.1 时间戳监控

```python
class TimestampMonitor:
    """时间戳质量监控"""

    def __init__(self):
        self.sync_intervals = deque(maxlen=100)
        self.time_spans = deque(maxlen=100)
        self.last_sync_time = None

    def check(self, *msgs):
        """检查同步质量"""
        now = rospy.Time.now()
        timestamps = [msg.header.stamp for msg in msgs]

        # 1. 检查时间跨度
        max_ts = max(timestamps)
        min_ts = min(timestamps)
        time_span = (max_ts - min_ts).to_sec()
        self.time_spans.append(time_span)

        # 2. 检查同步间隔
        if self.last_sync_time is not None:
            interval = (now - self.last_sync_time).to_sec()
            self.sync_intervals.append(interval)

        self.last_sync_time = now

        # 3. 统计
        if len(self.time_spans) >= 10:
            mean_span = np.mean(self.time_spans)
            max_span = np.max(self.time_spans)

            if max_span > 0.15:  # 150ms
                rospy.logwarn(
                    f"Large time span: mean={mean_span*1000:.0f}ms, "
                    f"max={max_span*1000:.0f}ms"
                )

        if len(self.sync_intervals) >= 10:
            mean_interval = np.mean(self.sync_intervals)
            std_interval = np.std(self.sync_intervals)

            if std_interval > 0.05:  # 50ms抖动
                rospy.logwarn(
                    f"Unstable sync: {mean_interval*1000:.0f}±{std_interval*1000:.0f}ms"
                )

        return time_span < 0.15  # 通过检查
```

### 5.2 诊断命令

```bash
# 1. 检查消息频率
rostopic hz /camera/chassis/color/image_raw
rostopic hz /camera/top/color/image_raw
rostopic hz /rslidar_points

# 预期输出：
# average rate: 30.000 (chassis)
# average rate: 30.000 (top)
# average rate: 10.000 (lidar)

# 2. 检查时间戳
rostopic echo /camera/chassis/color/image_raw/header/stamp --noarr
# 输出：1706600000.123456

# 3. 检查延迟
rostopic delay /camera/chassis/color/image_raw
# 输出：average delay: 0.050  # 50ms

# 4. 可视化时间戳
rosrun rqt_plot rqt_plot /camera/chassis/color/image_raw/header/stamp/secs
```

### 5.3 ROS诊断工具

```python
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

class SyncDiagnostics:
    """同步诊断发布"""

    def __init__(self):
        self.pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=1)
        self.monitor = TimestampMonitor()

    def publish(self, *msgs):
        """发布诊断信息"""
        passed = self.monitor.check(*msgs)

        diag = DiagnosticArray()
        diag.header.stamp = rospy.Time.now()

        status = DiagnosticStatus()
        status.name = "Multi-Sensor Synchronization"
        status.hardware_id = "perception_system"

        if passed:
            status.level = DiagnosticStatus.OK
            status.message = "Sync OK"
        else:
            status.level = DiagnosticStatus.WARN
            status.message = "Sync quality degraded"

        # 添加详细信息
        if len(self.monitor.time_spans) > 0:
            status.values.append(KeyValue(
                "time_span_mean", f"{np.mean(self.monitor.time_spans)*1000:.1f}ms"
            ))
            status.values.append(KeyValue(
                "time_span_max", f"{np.max(self.monitor.time_spans)*1000:.1f}ms"
            ))

        if len(self.monitor.sync_intervals) > 0:
            status.values.append(KeyValue(
                "sync_rate", f"{1.0/np.mean(self.monitor.sync_intervals):.1f}Hz"
            ))

        diag.status.append(status)
        self.pub.publish(diag)
```

---

## 6. 边缘情况处理

### 6.1 传感器失效降级

```python
def perception_callback_with_fallback(self, event):
    """带降级的感知回调"""
    with self.data_lock:
        # 检查哪些传感器可用
        available = {
            'chassis': self.chassis_data is not None,
            'top': self.top_data is not None,
            'lidar': self.lidar_data is not None
        }

        num_available = sum(available.values())

        if num_available == 0:
            rospy.logwarn_throttle(5.0, "No sensor data available")
            return

        if num_available < 3:
            rospy.logwarn(
                f"Degraded mode: only {num_available}/3 sensors available"
            )

        # 降级处理
        if not available['chassis'] and not available['top']:
            rospy.logerr("No camera available, cannot proceed")
            return

        # 至少有一个相机，可以继续
        if not available['lidar']:
            rospy.logwarn("LiDAR not available, using camera-only mode")

        # 执行感知（根据可用传感器）
        self.process_with_available_sensors(available)
```

### 6.2 时钟漂移检测

```python
class ClockDriftDetector:
    """系统时钟漂移检测"""

    def __init__(self):
        self.drift_history = deque(maxlen=100)

    def check(self, sensor_time):
        """检查传感器时间 vs 系统时间"""
        system_time = rospy.Time.now()
        drift = (system_time - sensor_time).to_sec()

        self.drift_history.append(drift)

        if len(self.drift_history) >= 10:
            mean_drift = np.mean(self.drift_history)
            std_drift = np.std(self.drift_history)

            # 检查平均漂移
            if abs(mean_drift) > 0.5:  # 500ms
                rospy.logerr(
                    f"Clock drift detected: {mean_drift*1000:.0f}ms\n"
                    "Check system time synchronization (NTP)"
                )
                return False

            # 检查漂移抖动
            if std_drift > 0.1:  # 100ms
                rospy.logwarn(
                    f"Unstable clock: drift std={std_drift*1000:.0f}ms"
                )

        return True
```

### 6.3 LiDAR低频处理

```python
class LiDARCache:
    """LiDAR数据缓存（处理低频问题）"""

    def __init__(self, max_age=0.2):
        self.cached_data = None
        self.cached_timestamp = None
        self.max_age = max_age  # 200ms

    def update(self, lidar_msg):
        """更新缓存"""
        self.cached_data = lidar_msg
        self.cached_timestamp = lidar_msg.header.stamp

    def get(self, request_time=None):
        """获取LiDAR数据"""
        if self.cached_data is None:
            return None

        if request_time is None:
            request_time = rospy.Time.now()

        age = (request_time - self.cached_timestamp).to_sec()

        if age > self.max_age:
            rospy.logwarn(f"LiDAR data too old: {age*1000:.0f}ms")
            return None

        return self.cached_data

    def is_fresh(self, max_age=None):
        """检查数据是否新鲜"""
        if max_age is None:
            max_age = self.max_age

        if self.cached_timestamp is None:
            return False

        age = (rospy.Time.now() - self.cached_timestamp).to_sec()
        return age <= max_age
```

---

## 7. 最佳实践

### 7.1 推荐配置

```yaml
# config/time_sync.yaml

# 同步参数
synchronization:
  # 全局同步（不推荐，仅参考）
  global:
    enabled: false
    slop: 0.1  # 100ms
    queue_size: 10

  # 分层同步（推荐）
  layered:
    enabled: true

    # Chassis相机同步
    chassis:
      slop: 0.05  # 50ms
      queue_size: 10

    # Top相机同步
    top:
      slop: 0.05  # 50ms
      queue_size: 10

    # LiDAR缓存
    lidar:
      max_age: 0.2  # 200ms
      queue_size: 5

    # 定时触发
    perception_rate: 5.0  # Hz
    max_data_age: 0.5  # 500ms过期

# 监控参数
monitoring:
  enable_diagnostics: true
  enable_timestamp_check: true
  max_time_span: 0.15  # 150ms
  max_sync_jitter: 0.05  # 50ms
```

### 7.2 常见陷阱

```python
# ❌ 错误1: 使用TimeSynchronizer（精确同步）
sync = message_filters.TimeSynchronizer(...)
# 问题：传感器时间戳永远不会完全相同，丢帧率极高

# ✅ 正确：使用ApproximateTimeSynchronizer
sync = message_filters.ApproximateTimeSynchronizer(..., slop=0.1)

# ❌ 错误2: slop太小
sync = ApproximateTimeSynchronizer(..., slop=0.01)  # 10ms
# 问题：容差太小，容易丢帧

# ✅ 正确：根据传感器特性设置
sync = ApproximateTimeSynchronizer(..., slop=0.05)  # 50ms

# ❌ 错误3: queue_size太小
sync = ApproximateTimeSynchronizer(..., queue_size=2)
# 问题：缓存不足，消息被覆盖

# ✅ 正确：预留足够缓存
sync = ApproximateTimeSynchronizer(..., queue_size=10)

# ❌ 错误4: 不检查数据新鲜度
def callback(self, *msgs):
    self.process(*msgs)  # 直接处理
# 问题：可能使用过期数据

# ✅ 正确：检查时间戳
def callback(self, *msgs):
    now = rospy.Time.now()
    for msg in msgs:
        age = (now - msg.header.stamp).to_sec()
        if age > 0.5:  # 500ms
            rospy.logwarn("Stale data")
            return
    self.process(*msgs)

# ❌ 错误5: 阻塞回调
def callback(self, *msgs):
    result = self.heavy_processing(*msgs)  # 300ms处理
# 问题：阻塞message_filters，导致丢帧

# ✅ 正确：异步处理或使用定时器
def callback(self, *msgs):
    with self.lock:
        self.cached_data = msgs  # 只缓存
    # 在定时器中处理
```

### 7.3 性能优化

```python
# 优化1: 减少数据复制
def callback(self, rgb_msg, depth_msg):
    # ❌ 不要每次都转换
    rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')

    # ✅ 只在需要时转换
    with self.lock:
        self.rgb_msg = rgb_msg  # 保存消息

    # 在处理时转换
    def process(self):
        rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, 'rgb8')

# 优化2: 使用nogil操作
with self.lock:
    # 只锁关键区域
    data = self.cached_data.copy()
# 在锁外处理（释放GIL）

# 优化3: 监控频率
rospy.Timer(rospy.Duration(10.0), self.print_stats)  # 每10秒
```

---

## 总结

### 推荐方案（你的场景）

```python
方案：分层同步 + LiDAR缓存 + 定时触发

配置：
├─ Chassis相机同步: slop=50ms, queue=10
├─ Top相机同步: slop=50ms, queue=10
├─ LiDAR缓存: max_age=200ms
└─ 感知触发: 5Hz (200ms周期)

优点：
✅ 双相机独立同步（不受LiDAR影响）
✅ 定时触发，频率可控
✅ 容错性好（单传感器失效可降级）
✅ 数据新鲜度可控（500ms过期）

性能：
- 同步延迟: <30ms
- 丢帧率: <2%
- 内存占用: ~50MB
```

### 相关文档

- [多传感器感知设计](./multi_sensor_perception_design.md)
- [系统概要](./multi_sensor_perception_overview.md)

---

**文档结束**
