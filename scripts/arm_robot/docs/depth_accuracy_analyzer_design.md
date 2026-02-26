# 3D 深度性能分析工具 - 设计文档

## 1. 概述

### 1.1 目标

对比深度相机与 LiDAR 的测距精度，分析误差来源，为抓取定位提供精度参考。

### 1.2 核心特性

- **无 ROS 依赖**: LiDAR 通过 UDP 直接解析
- **双模式 LiDAR 测量**: 相机引导 + 独立聚类
- **基于 Mask 的深度提取**: 比 bbox 中心更准确
- **点云质心对比**: 相机和 LiDAR 测量方式对等

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      depth_accuracy_analyzer.py                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │  RealSenseCamera │    │ RobosenseLiDAR  │    │  DinoXDetectorOnline │ │
│  │  (camera.py)     │    │ (UDP 直接解析)  │    │  (percept.py)        │ │
│  └────────┬────────┘    └────────┬────────┘    └──────────┬──────────┘ │
│           │                      │                        │            │
│           ▼                      ▼                        ▼            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                     CoordinateTransformer                          ││
│  │  - load_extrinsics(yaml_path)                                      ││
│  │  - transform_points(points, T)                                     ││
│  │  - 支持: optical_frame, rslidar, arm_base_link                     ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                     DepthAccuracyAnalyzer                          ││
│  │  - analyze(prompt, num_samples)                                    ││
│  │  - measure_camera() → 基于 Mask 的点云质心                         ││
│  │  - measure_lidar_guided() → 相机引导模式                           ││
│  │  - measure_lidar_independent() → 独立聚类模式                      ││
│  │  - generate_report()                                               ││
│  └────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: 初始化                                                          │
│  - 连接相机 (top/chassis)                                               │
│  - 连接 LiDAR (UDP 6699)                                                │
│  - 加载外参文件                                                          │
│  - 初始化检测服务 (DinoXDetectorOnline)                                  │
│  - 初始化深度优化服务 (DepthOptimizerOnline, 可选)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 同步采集                                                         │
│  - 相机: get_image_bundle() → rgb, depth_raw                            │
│  - LiDAR: get_one_frame() → point_cloud (N, 4)                          │
│  - 时间戳记录                                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: 目标检测                                                         │
│  - DinoXDetectorOnline.forward(prompt, rgb)                             │
│  - 获取所有检测结果: [{bbox, mask, score, category}, ...]               │
│  - 如果检测到多个物体: 选择置信度 (score) 最高的目标                      │
│  - 输出: bbox, mask, score                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ Step 4A:        │      │ Step 4B:            │      │ Step 4C:            │
│ 深度相机测距    │      │ LiDAR 相机引导      │      │ LiDAR 独立模式      │
│ (基于 Mask)     │      │ (深度范围限制)      │      │ (DBSCAN 聚类)       │
└────────┬────────┘      └──────────┬──────────┘      └──────────┬──────────┘
         │                          │                            │
         └────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: 误差分析与报告                                                   │
│  - 三组测量结果对比                                                      │
│  - 生成详细报告                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 测量方法详解

### 4A. 深度相机测距 (基于 Mask)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4A: 深度相机测距                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  4A.1 深度修复 (可选)                                                    │
│    └─ DepthOptimizerOnline (CDM 深度去噪)                               │
│                                                                         │
│  4A.2 Mask 腐蚀 (去除边缘噪声)                                          │
│    ┌────────────────────────────────────────┐                          │
│    │  原始 mask        腐蚀后 mask           │                          │
│    │  ██████████       ░░░░░░░░░░           │                          │
│    │  ██████████  ───► ░████████░           │                          │
│    │  ██████████       ░████████░           │                          │
│    │  ██████████       ░░░░░░░░░░           │                          │
│    └────────────────────────────────────────┘                          │
│    - cv2.erode(mask, kernel=5x5)                                       │
│    - 物体边缘深度不可靠，腐蚀后只保留内部区域                             │
│                                                                         │
│  4A.3 深度提取与异常值剔除                                               │
│    - 提取腐蚀后 mask 内的所有深度值                                      │
│    - 剔除无效值: depth <= 0 或 depth > 10m                              │
│    - 剔除异常值: IQR 方法 (Q1-1.5*IQR, Q3+1.5*IQR)                      │
│                                                                         │
│  4A.4 生成 3D 点云                                                       │
│    for each (u, v) in eroded_mask:                                     │
│        if depth[v, u] is valid:                                        │
│            X = (u - cx) * depth / fx                                   │
│            Y = (v - cy) * depth / fy                                   │
│            Z = depth                                                   │
│            points.append([X, Y, Z])                                    │
│                                                                         │
│  4A.5 计算质心 (中值更鲁棒)                                              │
│    centroid = np.median(points, axis=0)                                │
│                                                                         │
│  4A.6 坐标变换 → arm_base_link                                          │
│    p_arm = T_arm_optical^(-1) @ p_optical                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

输出:
  - centroid_optical: 质心坐标 (optical_frame)
  - centroid_arm: 质心坐标 (arm_base_link)
  - stats: {depth_median, depth_std, valid_ratio, num_points}
```

### 4B. LiDAR 相机引导模式

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4B: LiDAR 相机引导模式                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原理: 使用相机测得的深度范围来限制 LiDAR 点云筛选                        │
│  优点: 快速、简单、不需要聚类                                            │
│  缺点: 依赖相机深度，不是完全独立的测量                                   │
│                                                                         │
│  4B.1 视锥投影                                                          │
│    - 将 LiDAR 点云变换到 optical_frame                                  │
│    - 投影到图像平面: u = fx*X/Z + cx, v = fy*Y/Z + cy                   │
│                                                                         │
│  4B.2 bbox + 深度范围筛选                                                │
│    ┌────────────────────────────────────────┐                          │
│    │  筛选条件:                              │                          │
│    │  - u ∈ [x1, x2] (bbox 水平范围)        │                          │
│    │  - v ∈ [y1, y2] (bbox 垂直范围)        │                          │
│    │  - Z ∈ [d_cam - 0.3, d_cam + 0.3]     │ ← 相机深度 ± 0.3m        │
│    └────────────────────────────────────────┘                          │
│    - d_cam 来自 Step 4A 的 depth_median                                │
│    - ±0.3m 容差考虑相机误差和物体厚度                                    │
│                                                                         │
│  4B.3 异常值剔除                                                         │
│    - IQR 方法剔除离群点                                                  │
│                                                                         │
│  4B.4 计算质心                                                          │
│    centroid = np.median(filtered_points, axis=0)                       │
│                                                                         │
│  4B.5 坐标变换 → arm_base_link                                          │
│    p_arm = T_arm_rslidar^(-1) @ p_rslidar                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

输出:
  - centroid_rslidar: 质心坐标 (rslidar frame)
  - centroid_arm: 质心坐标 (arm_base_link)
  - stats: {depth_median, depth_std, num_points}
```

### 4C. LiDAR 独立模式 (DBSCAN 聚类)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4C: LiDAR 独立模式                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原理: 不依赖相机深度，通过聚类自动分离前景目标和背景                      │
│  优点: 完全独立的测量，可真正对比两种传感器精度                           │
│  缺点: 聚类参数需要调优，计算量稍大                                       │
│                                                                         │
│  4C.1 视锥投影筛选 (仅 2D，不限制深度)                                   │
│    - 将 LiDAR 点云变换到 optical_frame                                  │
│    - 投影到图像平面                                                      │
│    - 仅筛选落在 bbox 内的点 (不使用相机深度)                             │
│                                                                         │
│  4C.2 基础深度范围过滤                                                   │
│    - 剔除 Z < 0.3m (太近，可能是噪声)                                   │
│    - 剔除 Z > 15m (太远，超出有效范围)                                  │
│    - 这是 LiDAR 物理限制，不是目标深度                                   │
│                                                                         │
│  4C.3 DBSCAN 聚类                                                       │
│    ┌────────────────────────────────────────┐                          │
│    │     俯视图 (bbox 内的 LiDAR 点)         │                          │
│    │                                        │                          │
│    │     ●●●  ← Cluster 2: 背景墙 (远)      │                          │
│    │                                        │                          │
│    │   ▲▲▲▲▲  ← Cluster 1: 目标物体 (近)   │                          │
│    │                                        │                          │
│    │  ○        ← Noise: 噪声点 (label=-1)  │                          │
│    │                                        │                          │
│    │     📷 相机                            │                          │
│    └────────────────────────────────────────┘                          │
│                                                                         │
│    clustering = DBSCAN(eps=0.1, min_samples=5).fit(points)             │
│    - eps=0.1m: 同一聚类内点的最大距离                                    │
│    - min_samples=5: 形成聚类的最小点数                                   │
│                                                                         │
│  4C.4 选择最近的聚类                                                     │
│    for each cluster:                                                   │
│        cluster_depth = median(Z values)                                │
│    target_cluster = argmin(cluster_depth)                              │
│                                                                         │
│    理由: bbox 投影是锥形，会包含目标后面的背景                           │
│          最近的聚类 = 前景目标                                           │
│                                                                         │
│  4C.5 计算质心                                                          │
│    centroid = np.median(target_cluster_points, axis=0)                 │
│                                                                         │
│  4C.6 坐标变换 → arm_base_link                                          │
│    p_arm = T_arm_rslidar^(-1) @ p_rslidar                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

输出:
  - centroid_rslidar: 质心坐标 (rslidar frame)
  - centroid_arm: 质心坐标 (arm_base_link)
  - stats: {depth_median, depth_std, num_points, num_clusters}
```

---

## 5. LiDAR UDP 解析模块

### 5.1 Robosense Helios-16P 基本参数

```
┌────────────────────────────────────────────────────────────────┐
│ Helios-16P 规格                                                │
├────────────────────────────────────────────────────────────────┤
│  线数:           16 线                                         │
│  帧率:           10 Hz (可配置 5/10/20 Hz)                     │
│  水平视场角:     360°                                          │
│  垂直视场角:     -15° ~ +15° (共 30°)                         │
│  每帧点数:       约 28,800 点 (单回波)                         │
│  测距范围:       0.2m ~ 150m                                   │
│  测距精度:       ±2cm (典型)                                   │
│  UDP 端口:       MSOP=6699, DIFOP=7788                        │
│  每秒 UDP 包:    约 750 包 (75 包/帧 × 10 帧/秒)              │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 rslidar 坐标系定义

```
rslidar 坐标系 (右手系):

          Z (上)
           ↑
           │
           │
           ●────→ X (前, 0°方位角方向)
          ╱
         ╱
        Y (左)

- X 轴: 指向 LiDAR 前方 (0° 方位角)
- Y 轴: 指向 LiDAR 左侧 (90° 方位角)
- Z 轴: 指向上方
- 方位角: 从 X 轴正方向顺时针旋转 (俯视)
```

### 5.3 MSOP 协议详解

```
MSOP Packet Structure (1248 bytes):
┌────────────────────────────────────────────────────────────────┐
│ Header (42 bytes)                                              │
│   [0-3]   Sync: 0x55 0xAA 0x05 0x0A (Helios 系列标识)         │
│   [4-5]   Protocol version: 主版本.次版本                      │
│   [6-11]  Timestamp (us): 6 字节微秒时间戳                     │
│   [12-13] Lidar type: 0x0606 = Helios-16P                     │
│   [14-41] Reserved / Temperature / Factory info                │
├────────────────────────────────────────────────────────────────┤
│ Data Blocks × 12 (1200 bytes, 100 bytes each)                  │
│   [0-1]   Flag: 0xFF 0xEE (块起始标识)                        │
│   [2-3]   Azimuth: uint16 LE, 单位 0.01°, 范围 0-35999        │
│   [4-99]  Channels × 32 (每通道 3 bytes):                      │
│           [0-1] Distance: uint16 LE, 单位 0.5cm               │
│           [2]   Reflectivity: uint8, 0-255                    │
│           注: 16P 只用前 16 个通道，后 16 个无效               │
├────────────────────────────────────────────────────────────────┤
│ Tail (6 bytes)                                                 │
│   Reserved / CRC                                               │
└────────────────────────────────────────────────────────────────┘

点云计算 (rslidar 坐标系):
  - 方位角 θ = azimuth * 0.01° (转弧度)
  - 垂直角 φ = vertical_angle_table[channel_id] (转弧度)
  - 距离 r = distance * 0.005m

  X = r * cos(φ) * cos(θ)   ← 注意: cos(θ) 不是 sin(θ)
  Y = r * cos(φ) * sin(θ)   ← 左手系转右手系
  Z = r * sin(φ)
```

### 5.4 Helios-16P 垂直角度表

```
Channel ID:  0      1      2      3      4      5      6      7
Angle (°): -15.0  -13.0  -11.0   -9.0   -7.0   -5.0   -3.0   -1.0

Channel ID:  8      9     10     11     12     13     14     15
Angle (°):   1.0    3.0    5.0    7.0    9.0   11.0   13.0   15.0

注: 角度从下到上递增，channel 0 指向最下方 (-15°)
```

### 5.5 帧同步策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 帧同步方案: 基于方位角回绕检测                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  方法: 检测方位角从大变小 (如 350° → 10°) 表示新一帧开始                 │
│                                                                         │
│  伪代码:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  prev_azimuth = 0                                               │   │
│  │  frame_points = []                                              │   │
│  │                                                                 │   │
│  │  while True:                                                    │   │
│  │      packet = recv_udp_packet()                                 │   │
│  │      for block in packet.data_blocks:                           │   │
│  │          curr_azimuth = block.azimuth                           │   │
│  │                                                                 │   │
│  │          # 检测帧边界 (方位角回绕)                               │   │
│  │          if curr_azimuth < prev_azimuth - 18000:  # 回绕阈值    │   │
│  │              yield frame_points  # 返回完整一帧                  │   │
│  │              frame_points = []                                  │   │
│  │                                                                 │   │
│  │          frame_points.extend(parse_block(block))                │   │
│  │          prev_azimuth = curr_azimuth                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  注意:                                                                  │
│  - 回绕阈值 18000 = 180°，避免抖动误判                                  │
│  - 每帧约 75 个 UDP 包，超时设为 150ms (1.5帧时间)                      │
│  - 首帧可能不完整，应丢弃                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.6 Python 类设计

```python
import socket
import struct
import time
import numpy as np


class RobosenseLiDAR:
    """Robosense Helios-16P LiDAR UDP 直接解析器"""

    # MSOP 协议常量
    MSOP_HEADER_SIZE = 42
    MSOP_BLOCK_SIZE = 100
    MSOP_BLOCKS_PER_PACKET = 12
    MSOP_PACKET_SIZE = 1248
    CHANNELS_PER_BLOCK = 16  # Helios-16P 只用 16 通道

    # 同步字节
    SYNC_BYTES = bytes([0x55, 0xAA, 0x05, 0x0A])
    BLOCK_FLAG = bytes([0xFF, 0xEE])

    def __init__(self, msop_port=6699, timeout=0.15):
        """
        Args:
            msop_port: MSOP 数据端口，默认 6699
            timeout: 帧超时时间 (秒)，默认 0.15s (1.5帧)
        """
        self.msop_port = msop_port
        self.timeout = timeout
        self.socket = None

        # Helios-16P 垂直角度表 (弧度)
        self.vertical_angles = np.deg2rad([
            -15, -13, -11, -9, -7, -5, -3, -1,
              1,   3,   5,  7,  9, 11, 13, 15
        ])

        # 距离分辨率: 0.5cm = 0.005m
        self.distance_resolution = 0.005

        # 帧同步状态
        self._prev_azimuth = 0
        self._frame_buffer = []
        self._first_frame_skipped = False

    def connect(self):
        """绑定 UDP 端口"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.msop_port))
        self.socket.settimeout(self.timeout)
        self._prev_azimuth = 0
        self._frame_buffer = []
        self._first_frame_skipped = False

    def disconnect(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.socket = None

    def get_one_frame(self) -> np.ndarray:
        """
        获取一帧完整点云

        Returns:
            points: (N, 4) ndarray - [x, y, z, intensity]
                    如果超时或失败返回空数组 shape=(0, 4)
        """
        start_time = time.time()

        while True:
            # 超时检查
            if time.time() - start_time > self.timeout * 2:
                print("[WARN] LiDAR frame timeout")
                return np.empty((0, 4))

            try:
                data, _ = self.socket.recvfrom(2048)
            except socket.timeout:
                continue

            if len(data) != self.MSOP_PACKET_SIZE:
                continue

            # 解析数据包
            block_results = self._parse_packet(data)

            for azimuth, points in block_results:
                # 检测帧边界 (方位角回绕: 如 35000 → 500)
                if self._prev_azimuth > 18000 and azimuth < self._prev_azimuth - 18000:
                    # 完成一帧
                    if self._first_frame_skipped and len(self._frame_buffer) > 0:
                        frame = np.vstack(self._frame_buffer)
                        self._frame_buffer = [points] if len(points) > 0 else []
                        self._prev_azimuth = azimuth
                        return frame
                    else:
                        # 丢弃首帧 (可能不完整)
                        self._first_frame_skipped = True
                        self._frame_buffer = [points] if len(points) > 0 else []

                else:
                    # 累积当前帧数据
                    if len(points) > 0:
                        self._frame_buffer.append(points)

                self._prev_azimuth = azimuth

    def _parse_packet(self, data: bytes) -> list:
        """
        解析单个 MSOP 数据包

        Returns:
            list of (azimuth, points) tuples
        """
        # 验证包头
        if data[:4] != self.SYNC_BYTES:
            return []

        # 解析 12 个数据块
        results = []
        for i in range(self.MSOP_BLOCKS_PER_PACKET):
            offset = self.MSOP_HEADER_SIZE + i * self.MSOP_BLOCK_SIZE
            block_data = data[offset:offset + self.MSOP_BLOCK_SIZE]

            # 验证块标识
            if block_data[:2] != self.BLOCK_FLAG:
                continue

            # 解析方位角 (little-endian uint16, 单位 0.01°)
            azimuth = struct.unpack('<H', block_data[2:4])[0]

            # 解析 16 个通道
            points = self._parse_channels(block_data[4:], azimuth)
            results.append((azimuth, points))

        return results

    def _parse_channels(self, data: bytes, azimuth: int) -> np.ndarray:
        """解析一个块的 16 个通道数据"""
        theta = np.deg2rad(azimuth * 0.01)
        points = []

        for ch in range(self.CHANNELS_PER_BLOCK):
            offset = ch * 3
            distance = struct.unpack('<H', data[offset:offset+2])[0]
            intensity = data[offset + 2]

            if distance == 0:  # 无效点
                continue

            r = distance * self.distance_resolution
            phi = self.vertical_angles[ch]

            # rslidar 坐标系
            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)

            points.append([x, y, z, intensity])

        return np.array(points) if points else np.empty((0, 4))
```

---

## 6. 外参文件

### 6.1 文件清单

| 文件 | 变换 | 来源 |
|------|------|------|
| `extrinsics_rslidar_to_top_camera_optical_frame.yaml` | rslidar → optical_frame | **LiDAR-相机标定** |
| `extrinsics_arm_base_link_to_top_camera_optical_frame.yaml` | arm_base → optical_frame | launch 计算 |
| `extrinsics_arm_base_link_to_rslidar.yaml` | arm_base → rslidar | launch 计算 |

### 6.2 坐标变换关系

```
                    arm_base_link
                    ╱           ╲
                   ╱             ╲
    T_arm_rslidar ╱               ╲ T_arm_optical
                 ╱                 ╲
                ▼                   ▼
            rslidar ─────────► top_camera_optical_frame
                    T_rslidar_optical
                    (标定结果)
```

### 6.3 变换矩阵计算

```python
# 加载外参
T_rslidar_optical = load_transform('extrinsics_rslidar_to_top_camera_optical_frame.yaml')
T_arm_optical = load_transform('extrinsics_arm_base_link_to_top_camera_optical_frame.yaml')
T_arm_rslidar = load_transform('extrinsics_arm_base_link_to_rslidar.yaml')

# LiDAR → optical (用于投影)
T_lidar_to_optical = T_rslidar_optical

# optical → arm_base (相机结果变换)
T_optical_to_arm = np.linalg.inv(T_arm_optical)

# rslidar → arm_base (LiDAR 结果变换)
T_rslidar_to_arm = np.linalg.inv(T_arm_rslidar)
```

---

## 7. 相机配置

### 7.1 设备列表

| 相机 | device_id | 用途 |
|------|-----------|------|
| top | 318122302992 | 俯视全局检测 |
| chassis | 337122071540 | 底盘前向视角 |

### 7.2 内参获取

```python
# 从 RealSense 相机自动获取内参
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(device_id)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# 获取深度流内参
depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_profile.get_intrinsics()

# 内参值
fx = intrinsics.fx  # 焦距 x
fy = intrinsics.fy  # 焦距 y
cx = intrinsics.ppx  # 主点 x (principal point)
cy = intrinsics.ppy  # 主点 y

# D455 典型值 (640x480):
# fx ≈ 383.5, fy ≈ 383.5, cx ≈ 318.0, cy ≈ 238.5
```

### 7.3 当前外参支持状态

| 相机 | LiDAR 外参 | arm_base 外参 | 状态 |
|------|-----------|--------------|------|
| top | ✅ 已标定 | ✅ 已计算 | **可用** |
| chassis | ❌ 未标定 | ❌ 未计算 | 暂不支持 |

**注意**: 当前版本仅支持 top 相机。如需支持 chassis 相机，需要先完成:
1. rslidar → chassis_camera_optical_frame 标定
2. arm_base_link → chassis_camera_optical_frame 外参计算

---

## 8. 使用方式

```bash
cd ~/MobileManipulator/arm_robot/src
export PATH="/usr/bin:$PATH"

# 基本用法
python3 depth_accuracy_analyzer.py

# 指定相机和目标
python3 depth_accuracy_analyzer.py --camera top --prompt "box"

# 多次采样
python3 depth_accuracy_analyzer.py --samples 5

# 禁用深度修复
python3 depth_accuracy_analyzer.py --no-depth-optimize

# 仅使用 LiDAR 独立模式
python3 depth_accuracy_analyzer.py --lidar-mode independent

# 保存结果到文件
python3 depth_accuracy_analyzer.py --output results.json
```

---

## 9. 输出报告

```
════════════════════════════════════════════════════════════════════════════════
                         3D 深度性能分析报告
════════════════════════════════════════════════════════════════════════════════
时间: 2026-01-15 10:30:45
相机: top (318122302992)
检测目标: box
检测数量: 3 个 → 选择置信度最高的目标
选中目标: #1 (置信度: 0.92)
检测框: [156, 89, 423, 287]

────────────────────────────────────────────────────────────────────────────────
【深度相机测量】(基于 Mask 点云质心)
────────────────────────────────────────────────────────────────────────────────
  深度统计:
    - 中值深度:        2.138 m
    - 深度标准差:      0.023 m
    - 有效点数:        1250 / 2000 (62.5%)

  质心坐标:
    - optical_frame:   [ 0.120, -0.082,  2.138] m
    - arm_base_link:   [-0.134,  0.058,  2.616] m

  到 arm_base 原点距离: 2.620 m

────────────────────────────────────────────────────────────────────────────────
【LiDAR 测量 - 相机引导模式】(相机深度 ± 0.3m)
────────────────────────────────────────────────────────────────────────────────
  筛选统计:
    - 筛选点数:        42 个
    - 深度范围:        [1.838, 2.438] m
    - 深度标准差:      0.018 m

  质心坐标:
    - rslidar:         [-0.285,  0.025,  2.135] m
    - arm_base_link:   [-0.130,  0.022,  2.630] m

  到 arm_base 原点距离: 2.633 m

────────────────────────────────────────────────────────────────────────────────
【LiDAR 测量 - 独立模式】(DBSCAN 聚类)
────────────────────────────────────────────────────────────────────────────────
  聚类统计:
    - bbox 内总点数:   156 个
    - 聚类数量:        3 个
    - 目标聚类点数:    38 个 (最近聚类)
    - 深度标准差:      0.015 m

  质心坐标:
    - rslidar:         [-0.282,  0.028,  2.140] m
    - arm_base_link:   [-0.127,  0.025,  2.635] m

  到 arm_base 原点距离: 2.638 m

────────────────────────────────────────────────────────────────────────────────
【误差分析】
────────────────────────────────────────────────────────────────────────────────

                        │ 相机 vs LiDAR相机引导 │ 相机 vs LiDAR独立 │ 相机引导 vs 独立
  ──────────────────────┼───────────────────┼───────────────────┼──────────────────
  距离误差              │      -13 mm       │      -18 mm       │       5 mm
  相对误差              │       0.50%       │       0.69%       │      0.19%
  ──────────────────────┼───────────────────┼───────────────────┼──────────────────
  arm_base 坐标差:      │                   │                   │
    Δx                  │        4 mm       │        7 mm       │       3 mm
    Δy                  │       36 mm       │       33 mm       │       3 mm
    Δz                  │       14 mm       │       19 mm       │       5 mm
    总距离              │       39 mm       │       39 mm       │       7 mm

────────────────────────────────────────────────────────────────────────────────
【结论】
────────────────────────────────────────────────────────────────────────────────
  - 深度相机与 LiDAR 距离误差: < 20mm (在 2m 距离)
  - 相对误差: < 1%
  - 主要误差来源: Y 轴方向 (约 35mm)
  - LiDAR 两种模式一致性良好 (差异 < 10mm)

════════════════════════════════════════════════════════════════════════════════
```

---

## 10. 文件结构

```
arm_robot/src/
├── depth_accuracy_analyzer.py       # 主程序
├── robosense_lidar.py               # LiDAR UDP 解析器
├── percept.py                       # 检测服务 (已有)
├── camera.py                        # 相机接口 (已有)
└── config/
    ├── extrinsics_rslidar_to_top_camera_optical_frame.yaml   # LiDAR→相机标定
    ├── extrinsics_arm_base_link_to_top_camera_optical_frame.yaml
    ├── extrinsics_arm_base_link_to_rslidar.yaml
    ├── coordinate_frames.md                                   # 坐标系文档
    └── depth_accuracy_analyzer_design.md                      # 本设计文档
```

---

## 11. 依赖

### 11.1 Python 包

```
numpy
scipy
opencv-python (cv2)
pyrealsense2
pyyaml
requests
scikit-learn (DBSCAN)
```

### 11.2 网络

- LiDAR UDP 端口 6699 可访问
- DinoXDetectorOnline 服务: http://192.168.112.14:10086
- DepthOptimizerOnline 服务: http://192.168.112.14:8086 (可选)

### 11.3 硬件

- Intel RealSense D455 相机
- Robosense Helios-16P LiDAR

---

## 12. 错误处理

### 12.1 错误类型与处理策略

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  阶段           │  错误类型                │  处理策略                          │
├────────────────────────────────────────────────────────────────────────────────┤
│  初始化         │  相机连接失败            │  抛出异常，终止程序                │
│                │  LiDAR 端口占用          │  提示关闭其他 LiDAR 程序后重试     │
│                │  外参文件不存在          │  抛出异常，提示标定步骤            │
│                │  检测服务不可达          │  警告，禁用对应功能                │
├────────────────────────────────────────────────────────────────────────────────┤
│  数据采集       │  相机帧超时              │  重试 3 次，失败则跳过本次采样     │
│                │  LiDAR 帧超时            │  警告，仅输出相机测量结果          │
│                │  深度图全黑              │  跳过本次采样，提示目标太远/太近   │
├────────────────────────────────────────────────────────────────────────────────┤
│  目标检测       │  未检测到目标            │  提示更换 prompt 或调整位置        │
│                │  检测服务超时            │  重试 1 次，失败则跳过             │
│                │  bbox 超出图像范围       │  裁剪到图像边界，输出警告          │
├────────────────────────────────────────────────────────────────────────────────┤
│  深度测量       │  mask 内无有效深度       │  跳过深度相机测量，标记为 NaN      │
│                │  深度点数太少 (<10)      │  警告，仍计算但标记为低置信度      │
│                │  深度标准差过大 (>0.5m)  │  警告，可能存在遮挡或多目标        │
├────────────────────────────────────────────────────────────────────────────────┤
│  LiDAR 测量    │  bbox 内无 LiDAR 点      │  跳过 LiDAR 测量，仅输出相机结果   │
│                │  DBSCAN 无有效聚类       │  降级使用相机引导模式              │
│                │  点数太少 (<5)           │  警告，仍计算但标记为低置信度      │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 返回值约定

```python
class MeasurementResult:
    """单次测量结果"""

    def __init__(self):
        self.valid = False           # 测量是否有效
        self.confidence = 0.0        # 置信度 0-1
        self.centroid = None         # 质心 [x, y, z] 或 None
        self.stats = {}              # 统计信息
        self.error_msg = None        # 如有错误，记录原因

# 有效性判断
def is_valid(result: MeasurementResult) -> bool:
    return (
        result.valid and
        result.centroid is not None and
        result.confidence >= 0.5  # 最低置信度阈值
    )

# 置信度计算
def compute_confidence(num_points, depth_std, max_std=0.3) -> float:
    """
    基于点数和标准差计算置信度
    - num_points: 点数 (>=100 为高置信度)
    - depth_std: 深度标准差 (<=0.05m 为高置信度)
    """
    point_score = min(num_points / 100.0, 1.0)
    std_score = max(0, 1.0 - depth_std / max_std)
    return 0.7 * point_score + 0.3 * std_score
```

### 12.3 程序容错模式

```python
class DepthAccuracyAnalyzer:

    def analyze(self, prompt: str, num_samples: int = 1) -> dict:
        """
        执行分析，即使部分测量失败也返回可用结果
        """
        results = {
            'camera': [],
            'lidar_guided': [],
            'lidar_independent': [],
            'errors': []
        }

        for i in range(num_samples):
            try:
                # Step 1: 采集
                rgb, depth = self._capture_camera()
                lidar_points = self._capture_lidar()

                # Step 2: 检测
                detection = self._detect(prompt, rgb)
                if detection is None:
                    results['errors'].append(f"Sample {i}: 未检测到目标")
                    continue

                # Step 3: 深度相机测量 (必须成功)
                cam_result = self._measure_camera(depth, detection['mask'])
                if not cam_result.valid:
                    results['errors'].append(f"Sample {i}: 深度测量失败")
                    continue
                results['camera'].append(cam_result)

                # Step 4: LiDAR 测量 (可选)
                if len(lidar_points) > 0:
                    # 相机引导模式
                    guided_result = self._measure_lidar_guided(
                        lidar_points, detection['bbox'], cam_result.stats['depth_median']
                    )
                    if guided_result.valid:
                        results['lidar_guided'].append(guided_result)

                    # 独立模式
                    independent_result = self._measure_lidar_independent(
                        lidar_points, detection['bbox']
                    )
                    if independent_result.valid:
                        results['lidar_independent'].append(independent_result)

            except Exception as e:
                results['errors'].append(f"Sample {i}: {str(e)}")

        return results
```

---

## 13. DBSCAN 参数自适应

### 13.1 问题背景

DBSCAN 的 `eps` 参数需要根据目标距离动态调整:
- 近距离 (1-2m): LiDAR 点密度高，eps 应较小 (0.05-0.1m)
- 远距离 (3-5m): 点密度低，eps 需要更大 (0.15-0.25m)

### 13.2 自适应策略

```python
def compute_adaptive_eps(estimated_depth: float, lidar_angular_resolution: float = 0.2) -> float:
    """
    根据估计深度计算自适应 eps

    原理: LiDAR 水平角度分辨率约 0.2°
          在距离 d 处，相邻点间距约为 d * tan(0.2°) ≈ d * 0.0035
          eps 应设为相邻点间距的 2-3 倍

    Args:
        estimated_depth: 估计深度 (m)，可来自相机引导的粗略值
        lidar_angular_resolution: LiDAR 角度分辨率 (度)

    Returns:
        eps: DBSCAN 邻域半径 (m)
    """
    # 计算相邻点间距
    point_spacing = estimated_depth * np.tan(np.deg2rad(lidar_angular_resolution))

    # eps = 3倍点间距，但限制在合理范围
    eps = np.clip(3 * point_spacing, 0.05, 0.3)

    return eps

# 使用示例
depth_estimate = 3.0  # 从 bbox 中心深度或相机测量粗略估计
eps = compute_adaptive_eps(depth_estimate)
# depth=3m → point_spacing≈0.01m → eps≈0.03m (clip to 0.05m)
```

### 13.3 无先验深度时的策略

```python
def compute_eps_from_points(points: np.ndarray) -> float:
    """
    当没有相机深度先验时，从点云本身估计 eps

    方法: 使用点云的中值深度作为估计值
    """
    if len(points) == 0:
        return 0.1  # 默认值

    depths = np.linalg.norm(points[:, :3], axis=1)
    median_depth = np.median(depths)

    return compute_adaptive_eps(median_depth)
```

---

## 14. 注意事项

1. **LiDAR 数据同步**: UDP 解析的点云与相机帧可能有时间差，建议多次采样取平均
2. **DBSCAN 参数**: 使用自适应 eps (参考第 13 节)，min_samples 保持 5
3. **Mask 腐蚀**: kernel=5 适用于 640x480 分辨率，高分辨率可适当增大
4. **深度修复**: CDM 服务需要 GPU，如果延迟敏感可禁用
5. **外参文件**: 当前只支持 top 相机，chassis 相机需要额外标定

---

## 15. 确认配置清单

### 15.1 硬件配置

| 项目 | 确认值 | 备注 |
|------|--------|------|
| LiDAR 型号 | Robosense Helios-16P | 16线 |
| LiDAR IP | 192.168.0.200 | 设备地址 |
| 主机 IP | 192.168.0.155 | eth0 接口 |
| MSOP 端口 | 6699 | 点云数据 |
| DIFOP 端口 | 7788 | 设备信息 |
| 垂直角度表 | 默认配置 | -15° ~ +15° |
| 相机型号 | Intel RealSense D455 | RGB-D |
| 相机分辨率 | 1280 × 720 | RGB 和深度 |
| top 相机 ID | 318122302992 | 俯视 |

### 15.2 服务端点

| 服务 | 地址 | 用途 |
|------|------|------|
| DinoXDetectorOnline | http://192.168.112.14:10086 | 目标检测 |
| DepthOptimizerOnline | http://192.168.112.14:8086 | 深度优化 (必须) |

### 15.3 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 深度有效范围 | 0.3m ~ 10m | 剔除无效深度 |
| Mask 腐蚀核 | 5×5 | 边缘去噪 |
| IQR 系数 | 1.5 | 异常值剔除 |
| 相机引导容差 | ±0.3m | LiDAR 深度筛选 |
| 最小 Mask 面积 | 500 像素 | 低于此值不可靠 |
| 测试距离范围 | 1-5m | 主要测试范围 |
| 默认采样次数 | 3 次 | 自动连续采样 |

### 15.4 数据格式约定

#### 四元数格式
```
顺序: (x, y, z, w) - scipy/ROS 标准格式
来源: cam_lidar_calibrate/lib/io_utils.py
用法: Rotation.from_quat([x, y, z, w])
```

#### 变换方向约定
```
文件名: extrinsics_A_to_B.yaml
含义: T_A_to_B * p_A = p_B (将 A 坐标系的点变换到 B 坐标系)
示例: extrinsics_rslidar_to_top_camera_optical_frame.yaml
      → T * p_rslidar = p_optical
```

#### 深度数据格式
```
camera.py 返回: float32, 单位 meters (已从 mm 转换)
DepthOptimizer 输入: uint16, 单位 mm
转换: depth_mm = (depth_meters * 1000).astype(np.uint16)
```

#### Mask 格式
```
检测服务返回: RLE (Run-Length Encoding) 格式
解码方法:
  from pycocotools import mask as coco_mask
  binary_mask = coco_mask.decode(rle_mask)  # → (H, W) uint8, 0/1
分辨率: 与检测输入一致 (1280×720)
```

### 15.5 已确认的代码实现

#### camera.py - 深度对齐
```python
# 已实现 RGB-Depth 对齐 (line 229-232)
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
aligned_depth_frame = aligned_frames.get_depth_frame()
```

#### percept.py - 检测服务
```python
# DinoXDetectorOnline.forward() 返回格式 (line 157-158)
{'objects': [
    {'bbox': [x1, y1, x2, y2], 'score': float, 'category': str, 'mask': rle_dict},
    ...
]}
```

#### percept.py - 深度优化服务
```python
# DepthOptimizerOnline.forward() 返回格式 (line 990-997)
{
    'success': bool,
    'depth': np.ndarray,  # (H, W) uint16, mm
    'vis_image': np.ndarray,  # (H, W, 3) BGR (可选)
}
```

### 15.6 输出配置

| 项目 | 配置 |
|------|------|
| 可视化 | 需要，保存到文件 |
| 可视化内容 | RGB + 深度伪彩色 + LiDAR 投影 (三合一) |
| 报告格式 | JSON + 文本 (两者都要) |
| 输出目录 | arm_robot/results/ |
| 采样方式 | 自动连续 (间隔 1.5 秒) |

### 15.7 外参支持状态

| 相机 | rslidar 外参 | arm_base 外参 | 状态 |
|------|-------------|--------------|------|
| top | ✅ 已标定 | ✅ 已计算 | **可用** |
| chassis | ❌ 未标定 | ❌ 未计算 | 后续补充 |

---

**文档版本**: v1.2
**创建日期**: 2026-01-15
**最后更新**: 2026-01-15
**作者**: Claude Code

---

## 更新记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-01-15 | 初始版本 |
| v1.1 | 2026-01-15 | 补充错误处理、DBSCAN自适应、相机内参获取、get_one_frame实现 |
| v1.2 | 2026-01-15 | 添加确认配置清单，包含硬件配置、数据格式约定、代码实现确认 |
