#!/usr/bin/env python3
"""
同步传感器订阅器

同步订阅 RGB + Depth + LiDAR (可选) 三个数据源，
支持数据新鲜度检查和分层同步策略。
"""

import threading
import time
import numpy as np
import cv2
import rospy
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import message_filters


class SyncedSensorSubscriber:
    """同步订阅相机和LiDAR数据"""

    def __init__(self,
                 color_topic='/camera/top/color/image_raw',
                 depth_topic='/camera/top/aligned_depth_to_color/image_raw',
                 info_topic='/camera/top/color/camera_info',
                 lidar_topic='/lidar/chassis/point_cloud',
                 enable_lidar=True,
                 sync_slop=0.1):
        """
        Args:
            color_topic: RGB 图像 topic
            depth_topic: 深度图像 topic (aligned to color)
            info_topic: 相机内参 topic
            lidar_topic: LiDAR 点云 topic
            enable_lidar: 是否同步 LiDAR 数据
            sync_slop: 时间同步容差 (秒)
        """
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.info_topic = info_topic
        self.lidar_topic = lidar_topic
        self.enable_lidar = enable_lidar
        self.sync_slop = sync_slop

        self._bridge = CvBridge()
        self._latest_data = None
        self._data_lock = threading.Lock()
        self._intrinsics = None
        self._connected = False

        # 订阅器引用 (保持活跃)
        self._info_sub = None
        self._color_sub = None
        self._depth_sub = None
        self._lidar_sub = None
        self._sync = None

    @property
    def intrinsics(self) -> dict:
        """相机内参 {fx, fy, cx, cy, width, height}"""
        return self._intrinsics

    @property
    def connected(self) -> bool:
        """是否已连接"""
        return self._connected

    def connect(self, timeout=10.0) -> bool:
        """
        启动同步订阅

        Args:
            timeout: 等待相机内参的超时时间 (秒)

        Returns:
            是否成功连接
        """
        if self._connected:
            rospy.logwarn("[SyncedSensor] 已经连接，跳过")
            return True

        # 1. 订阅 CameraInfo (一次性获取内参)
        rospy.loginfo(f"[SyncedSensor] 等待相机内参: {self.info_topic}")
        try:
            info_msg = rospy.wait_for_message(self.info_topic, CameraInfo, timeout=timeout)
            self._parse_camera_info(info_msg)
            rospy.loginfo(f"[SyncedSensor] 内参获取成功: {self._intrinsics['width']}x{self._intrinsics['height']}")
        except rospy.ROSException as e:
            rospy.logerr(f"[SyncedSensor] 获取相机内参超时: {e}")
            return False

        # 2. 使用 message_filters 同步数据源
        self._color_sub = message_filters.Subscriber(self.color_topic, Image)
        self._depth_sub = message_filters.Subscriber(self.depth_topic, Image)

        if self.enable_lidar:
            # 三路同步: RGB + Depth + LiDAR
            self._lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub, self._lidar_sub],
                queue_size=10,
                slop=self.sync_slop
            )
            self._sync.registerCallback(self._sync_callback_with_lidar)
            rospy.loginfo(f"[SyncedSensor] 三路同步模式 (RGB+Depth+LiDAR), slop={self.sync_slop}s")
        else:
            # 两路同步: RGB + Depth
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub],
                queue_size=10,
                slop=self.sync_slop
            )
            self._sync.registerCallback(self._sync_callback_camera_only)
            rospy.loginfo(f"[SyncedSensor] 两路同步模式 (RGB+Depth), slop={self.sync_slop}s")

        self._connected = True
        return True

    def disconnect(self):
        """断开订阅"""
        if self._color_sub:
            self._color_sub.sub.unregister()
        if self._depth_sub:
            self._depth_sub.sub.unregister()
        if self._lidar_sub:
            self._lidar_sub.sub.unregister()
        self._connected = False
        rospy.loginfo("[SyncedSensor] 已断开连接")

    def _parse_camera_info(self, msg: CameraInfo):
        """解析相机内参"""
        self._intrinsics = {
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5],
            'width': msg.width,
            'height': msg.height,
        }

    def _sync_callback_with_lidar(self, color_msg, depth_msg, lidar_msg):
        """三路同步回调"""
        data = self._process_camera_data(color_msg, depth_msg)
        data['lidar_points'] = self._parse_pointcloud(lidar_msg)
        data['has_lidar'] = True

        with self._data_lock:
            self._latest_data = data

    def _sync_callback_camera_only(self, color_msg, depth_msg):
        """两路同步回调 (无 LiDAR)"""
        data = self._process_camera_data(color_msg, depth_msg)
        data['lidar_points'] = None
        data['has_lidar'] = False

        with self._data_lock:
            self._latest_data = data

    def _process_camera_data(self, color_msg, depth_msg) -> dict:
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
        }

    def _parse_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        """解析 PointCloud2 消息为 numpy 数组

        Returns:
            points: (N, 4) 数组 [x, y, z, intensity]
        """
        # 获取字段偏移
        field_names = [f.name for f in msg.fields]
        field_offsets = {f.name: f.offset for f in msg.fields}

        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            rospy.logwarn("[SyncedSensor] PointCloud2 缺少 xyz 字段")
            return np.empty((0, 4))

        # 解析点云
        points = []
        for i in range(msg.width * msg.height):
            offset = i * msg.point_step
            x = struct.unpack_from('f', msg.data, offset + field_offsets['x'])[0]
            y = struct.unpack_from('f', msg.data, offset + field_offsets['y'])[0]
            z = struct.unpack_from('f', msg.data, offset + field_offsets['z'])[0]

            # intensity 可选
            if 'intensity' in field_offsets:
                intensity = struct.unpack_from('f', msg.data, offset + field_offsets['intensity'])[0]
            else:
                intensity = 0.0

            # 过滤无效点 (NaN/Inf)
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z, intensity])

        return np.array(points, dtype=np.float32) if points else np.empty((0, 4))

    def get_synced_data(self, timeout=5.0, max_age=0.5) -> dict:
        """
        获取同步数据，带新鲜度检查

        Args:
            timeout: 等待超时时间 (秒)
            max_age: 数据最大年龄 (秒)，超过则认为数据过时

        Returns:
            dict: {
                'rgb': np.ndarray (H, W, 3) BGR uint8,
                'depth': np.ndarray (H, W) float32 米,
                'lidar_points': np.ndarray (N, 4) [x,y,z,intensity] 或 None,
                'has_lidar': bool,
                'timestamp': rospy.Time,
                'age': float 数据年龄(秒),
            }

        Raises:
            TimeoutError: 无法在超时时间内获取新鲜数据
        """
        if not self._connected:
            raise RuntimeError("[SyncedSensor] 未连接，请先调用 connect()")

        start = time.time()
        while time.time() - start < timeout:
            with self._data_lock:
                if self._latest_data is not None:
                    age = (rospy.Time.now() - self._latest_data['timestamp']).to_sec()
                    if age < max_age:
                        result = self._latest_data.copy()
                        result['age'] = age
                        return result
                    else:
                        rospy.logdebug(f"[SyncedSensor] 数据过时: age={age:.3f}s > max_age={max_age}s")
            time.sleep(0.01)

        raise TimeoutError(f"[SyncedSensor] 无法获取新鲜数据 (timeout={timeout}s, max_age={max_age}s)")

    def get_latest_data(self) -> dict:
        """
        获取最新数据 (无新鲜度检查)

        Returns:
            dict 或 None
        """
        with self._data_lock:
            if self._latest_data is not None:
                result = self._latest_data.copy()
                result['age'] = (rospy.Time.now() - self._latest_data['timestamp']).to_sec()
                return result
            return None


def test_synced_subscriber():
    """测试同步订阅器"""
    rospy.init_node('test_synced_subscriber', anonymous=True)

    print("=" * 60)
    print("SyncedSensorSubscriber 测试")
    print("=" * 60)

    # 测试两种模式
    for enable_lidar in [False, True]:
        mode = "三路同步" if enable_lidar else "两路同步"
        print(f"\n--- 测试 {mode} 模式 ---")

        subscriber = SyncedSensorSubscriber(
            color_topic='/camera/top/color/image_raw',
            depth_topic='/camera/top/aligned_depth_to_color/image_raw',
            info_topic='/camera/top/color/camera_info',
            lidar_topic='/lidar/chassis/point_cloud',
            enable_lidar=enable_lidar,
            sync_slop=0.1
        )

        if not subscriber.connect(timeout=5.0):
            print(f"  连接失败，跳过 {mode} 测试")
            continue

        print(f"  内参: {subscriber.intrinsics}")

        # 尝试获取数据
        try:
            data = subscriber.get_synced_data(timeout=3.0, max_age=1.0)
            print(f"  RGB shape: {data['rgb'].shape}")
            print(f"  Depth shape: {data['depth'].shape}")
            print(f"  Depth range: [{data['depth'].min():.3f}, {data['depth'].max():.3f}] m")
            print(f"  Has LiDAR: {data['has_lidar']}")
            if data['lidar_points'] is not None:
                print(f"  LiDAR points: {len(data['lidar_points'])}")
            print(f"  Data age: {data['age']:.3f}s")
        except TimeoutError as e:
            print(f"  获取数据超时: {e}")
        except Exception as e:
            print(f"  错误: {e}")

        subscriber.disconnect()

    print("\n测试完成")


if __name__ == '__main__':
    test_synced_subscriber()
