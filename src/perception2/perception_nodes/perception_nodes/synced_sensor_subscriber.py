#!/usr/bin/env python3
"""
同步传感器订阅器 (ROS2 版本)

同步订阅 RGB + Depth + LiDAR (可选) 三个数据源，
支持数据新鲜度检查和分层同步策略。

ROS1 → ROS2 关键变化:
- message_filters.Subscriber 需要 node 作为第一个参数
- 使用 QoS 配置替代默认订阅
- rospy.Time → rclpy.time.Time
- rospy.wait_for_message → 自定义实现
"""

import threading
import time
from typing import Optional

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from .utils import CvBridgeNumPy2 as CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import message_filters

from .utils import (
    wait_for_message,
    parse_pointcloud2_fast,
    stamp_to_sec,
    SENSOR_QOS,
    DEFAULT_QOS,
)


class SyncedSensorSubscriber:
    """
    同步订阅相机和LiDAR数据 (ROS2 版本)

    主要功能:
    - 同步 RGB + Depth 相机数据
    - 可选同步 LiDAR 点云
    - 数据新鲜度检查
    - 线程安全的数据访问

    Example:
        >>> node = rclpy.create_node('test_node')
        >>> subscriber = SyncedSensorSubscriber(
        ...     node=node,
        ...     color_topic='/camera/color/image_raw',
        ...     depth_topic='/camera/aligned_depth_to_color/image_raw',
        ...     info_topic='/camera/color/camera_info',
        ... )
        >>> if subscriber.connect(timeout=5.0):
        ...     data = subscriber.get_synced_data()
        ...     print(f"RGB shape: {data['rgb'].shape}")
    """

    def __init__(
        self,
        node: Node,
        color_topic: str = '/camera/top/color/image_raw',
        depth_topic: str = '/camera/top/aligned_depth_to_color/image_raw',
        info_topic: str = '/camera/top/color/camera_info',
        lidar_topic: str = '/lidar/chassis/point_cloud',
        enable_lidar: bool = True,
        sync_slop: float = 0.1,
        qos_profile: Optional[QoSProfile] = None,
    ):
        """
        初始化同步传感器订阅器

        Args:
            node: ROS2 节点实例 (必须)
            color_topic: RGB 图像 topic
            depth_topic: 深度图像 topic (aligned to color)
            info_topic: 相机内参 topic
            lidar_topic: LiDAR 点云 topic
            enable_lidar: 是否同步 LiDAR 数据
            sync_slop: 时间同步容差 (秒)
            qos_profile: 传感器 QoS 配置 (默认 SENSOR_QOS)
        """
        self._node = node
        self._logger = node.get_logger()

        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.info_topic = info_topic
        self.lidar_topic = lidar_topic
        self.enable_lidar = enable_lidar
        self.sync_slop = sync_slop
        self.qos_profile = qos_profile or SENSOR_QOS

        self._bridge = CvBridge()
        self._latest_data: Optional[dict] = None
        self._data_lock = threading.Lock()
        self._intrinsics: Optional[dict] = None
        self._connected = False

        # 订阅器引用 (保持活跃)
        self._color_sub = None
        self._depth_sub = None
        self._lidar_sub = None
        self._sync = None

    @property
    def node(self) -> Node:
        """ROS2 节点实例"""
        return self._node

    @property
    def intrinsics(self) -> Optional[dict]:
        """相机内参 {fx, fy, cx, cy, width, height}"""
        return self._intrinsics

    @property
    def connected(self) -> bool:
        """是否已连接"""
        return self._connected

    def connect(self, timeout: float = 10.0) -> bool:
        """
        启动同步订阅

        Args:
            timeout: 等待相机内参的超时时间 (秒)

        Returns:
            是否成功连接
        """
        if self._connected:
            self._logger.warn('[SyncedSensor] 已经连接，跳过')
            return True

        # 1. 获取相机内参 (一次性)
        self._logger.info(f'[SyncedSensor] 等待相机内参: {self.info_topic}')

        info_msg = wait_for_message(
            self._node,
            CameraInfo,
            self.info_topic,
            timeout=timeout,
            qos_profile=DEFAULT_QOS,  # RELIABLE + VOLATILE，兼容 RealSense
        )

        if info_msg is None:
            self._logger.error(f'[SyncedSensor] 获取相机内参超时 (timeout={timeout}s)')
            return False

        self._parse_camera_info(info_msg)
        self._logger.info(
            f"[SyncedSensor] 内参获取成功: {self._intrinsics['width']}x{self._intrinsics['height']}"
        )

        # 2. 使用 message_filters 同步数据源
        # ROS2 message_filters API: Subscriber(node, msg_type, topic, qos_profile)
        self._color_sub = message_filters.Subscriber(
            self._node, Image, self.color_topic, qos_profile=self.qos_profile
        )
        self._depth_sub = message_filters.Subscriber(
            self._node, Image, self.depth_topic, qos_profile=self.qos_profile
        )

        if self.enable_lidar:
            # 三路同步: RGB + Depth + LiDAR
            self._lidar_sub = message_filters.Subscriber(
                self._node, PointCloud2, self.lidar_topic, qos_profile=self.qos_profile
            )
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub, self._lidar_sub],
                queue_size=10,
                slop=self.sync_slop,
            )
            self._sync.registerCallback(self._sync_callback_with_lidar)
            self._logger.info(
                f'[SyncedSensor] 三路同步模式 (RGB+Depth+LiDAR), slop={self.sync_slop}s'
            )
        else:
            # 两路同步: RGB + Depth
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub],
                queue_size=10,
                slop=self.sync_slop,
            )
            self._sync.registerCallback(self._sync_callback_camera_only)
            self._logger.info(
                f'[SyncedSensor] 两路同步模式 (RGB+Depth), slop={self.sync_slop}s'
            )

        self._connected = True
        return True

    def disconnect(self):
        """断开订阅"""
        # ROS2 message_filters 没有 unregister，需要销毁订阅
        # 由于 message_filters.Subscriber 内部管理订阅，
        # 我们只需要清除引用让 Python GC 处理
        self._color_sub = None
        self._depth_sub = None
        self._lidar_sub = None
        self._sync = None
        self._connected = False
        self._logger.info('[SyncedSensor] 已断开连接')

    def _parse_camera_info(self, msg: CameraInfo):
        """解析相机内参"""
        self._intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'width': msg.width,
            'height': msg.height,
        }

    def _sync_callback_with_lidar(self, color_msg: Image, depth_msg: Image, lidar_msg: PointCloud2):
        """三路同步回调"""
        data = self._process_camera_data(color_msg, depth_msg)
        data['lidar_points'] = parse_pointcloud2_fast(lidar_msg)
        data['has_lidar'] = True

        with self._data_lock:
            self._latest_data = data

    def _sync_callback_camera_only(self, color_msg: Image, depth_msg: Image):
        """两路同步回调 (无 LiDAR)"""
        data = self._process_camera_data(color_msg, depth_msg)
        data['lidar_points'] = None
        data['has_lidar'] = False

        with self._data_lock:
            self._latest_data = data

    def _copy_data(self, data: dict) -> dict:
        """深拷贝数据字典，避免返回共享引用"""
        return {
            'rgb': data['rgb'].copy() if data['rgb'] is not None else None,
            'depth': data['depth'].copy() if data['depth'] is not None else None,
            'lidar_points': data['lidar_points'].copy() if data.get('lidar_points') is not None else None,
            'has_lidar': data.get('has_lidar', False),
            'timestamp': data['timestamp'],
            'frame_id': data['frame_id'],
        }

    def _process_camera_data(self, color_msg: Image, depth_msg: Image) -> dict:
        """处理相机数据"""
        # RGB: 使用 passthrough 避免 cv_bridge cvtColor2 崩溃，手动转换
        rgb_raw = self._bridge.imgmsg_to_cv2(color_msg, 'passthrough')
        if color_msg.encoding == 'rgb8':
            rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
        elif color_msg.encoding == 'bgr8':
            rgb = rgb_raw
        else:
            rgb = rgb_raw  # 其他编码原样返回

        # Depth: 16UC1 (mm) -> float32 (m)
        depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = depth_mm.astype(np.float32) / 1000.0

        return {
            'rgb': rgb,
            'depth': depth_m,
            'timestamp': color_msg.header.stamp,
            'frame_id': color_msg.header.frame_id,
        }

    def get_synced_data(self, timeout: float = 5.0, max_age: float = 0.5, strict: bool = True) -> dict:
        """
        获取同步数据，带新鲜度检查
        
        注意：此方法是线程安全的，但不要在 ROS2 spin 的回调中调用，
        因为内部会阻塞等待数据。

        Args:
            timeout: 等待超时时间 (秒)
            max_age: 数据最大年龄 (秒)，超过则认为数据过时
            strict: 是否严格检查新鲜度。如果为False，只要收到过数据就返回（无视年龄）

        Returns:
            dict: {
                'rgb': np.ndarray (H, W, 3) BGR uint8,
                'depth': np.ndarray (H, W) float32 米,
                'lidar_points': np.ndarray (N, 4) [x,y,z,intensity] 或 None,
                'has_lidar': bool,
                'timestamp': builtin_interfaces.msg.Time,
                'frame_id': str,
                'age': float 数据年龄(秒),
            }

        Raises:
            RuntimeError: 未连接
            TimeoutError: 无法在超时时间内获取新鲜数据
        """
        if not self._connected:
            raise RuntimeError('[SyncedSensor] 未连接，请先调用 connect()')

        start = time.time()
        first_data_time = None
        
        while time.time() - start < timeout:
            # 注意：不在此处调用 rclpy.spin_once()，避免多线程冲突
            # 数据更新由 message_filters 回调在 ROS2 spin 线程中完成
            
            with self._data_lock:
                if self._latest_data is not None:
                    # 计算数据年龄
                    now = self._node.get_clock().now()
                    data_time_sec = stamp_to_sec(self._latest_data['timestamp'])
                    now_sec = now.nanoseconds / 1e9  # 完整的纳秒数转秒
                    age = now_sec - data_time_sec
                    
                    # 记录第一次收到数据的时间
                    if first_data_time is None:
                        first_data_time = time.time()

                    # 检查新鲜度
                    if age < max_age:
                        # 数据新鲜，深拷贝后返回
                        result = self._copy_data(self._latest_data)
                        result['age'] = age
                        return result
                    elif not strict:
                        # 非严格模式：返回最新数据（即使有点旧）
                        result = self._copy_data(self._latest_data)
                        result['age'] = age
                        self._logger.debug(f'[SyncedSensor] 返回较旧数据: age={age:.3f}s')
                        return result
                    # 严格模式下，数据过时继续等待

            time.sleep(0.01)

        raise TimeoutError(
            f'[SyncedSensor] 无法获取新鲜数据 (timeout={timeout}s, max_age={max_age}s, strict={strict})'
        )

    def get_latest_data(self) -> Optional[dict]:
        """
        获取最新数据 (无新鲜度检查)

        Returns:
            dict 或 None
        """
        with self._data_lock:
            if self._latest_data is not None:
                # 深拷贝数据，避免数据竞争
                result = self._copy_data(self._latest_data)
                # 计算年龄
                now = self._node.get_clock().now()
                data_time_sec = stamp_to_sec(self._latest_data['timestamp'])
                now_sec = now.nanoseconds * 1e-9
                result['age'] = now_sec - data_time_sec
                return result
            return None

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        """
        等待至少一帧数据到达

        Args:
            timeout: 超时时间 (秒)

        Returns:
            是否成功获取数据
        """
        if not self._connected:
            return False

        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            with self._data_lock:
                if self._latest_data is not None:
                    return True
        return False


def main():
    """测试同步订阅器"""
    rclpy.init()

    node = rclpy.create_node('test_synced_subscriber')
    logger = node.get_logger()

    logger.info('=' * 60)
    logger.info('SyncedSensorSubscriber 测试 (ROS2)')
    logger.info('=' * 60)

    # 测试两种模式
    for enable_lidar in [False, True]:
        mode = '三路同步' if enable_lidar else '两路同步'
        logger.info(f'\n--- 测试 {mode} 模式 ---')

        subscriber = SyncedSensorSubscriber(
            node=node,
            color_topic='/camera/top/color/image_raw',
            depth_topic='/camera/top/aligned_depth_to_color/image_raw',
            info_topic='/camera/top/color/camera_info',
            lidar_topic='/lidar/chassis/point_cloud',
            enable_lidar=enable_lidar,
            sync_slop=0.1,
        )

        if not subscriber.connect(timeout=5.0):
            logger.warn(f'连接失败，跳过 {mode} 测试')
            continue

        logger.info(f'内参: {subscriber.intrinsics}')

        # 尝试获取数据
        try:
            data = subscriber.get_synced_data(timeout=3.0, max_age=1.0)
            logger.info(f"RGB shape: {data['rgb'].shape}")
            logger.info(f"Depth shape: {data['depth'].shape}")
            logger.info(f"Depth range: [{data['depth'].min():.3f}, {data['depth'].max():.3f}] m")
            logger.info(f"Has LiDAR: {data['has_lidar']}")
            if data['lidar_points'] is not None:
                logger.info(f"LiDAR points: {len(data['lidar_points'])}")
            logger.info(f"Data age: {data['age']:.3f}s")
        except TimeoutError as e:
            logger.warn(f'获取数据超时: {e}')
        except Exception as e:
            logger.error(f'错误: {e}')

        subscriber.disconnect()

    logger.info('\n测试完成')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
