#!/usr/bin/env python3
"""
ROS-based LiDAR reader - 直接订阅ROS点云话题

使用方法与标定项目完全一致，确保坐标系匹配。
参考: cam_lidar_calibrate/extract_data.py
"""

import numpy as np
import time
import threading

try:
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[WARN] ROS not available, ros_lidar.py will not work")


class ROSLiDAR:
    """ROS点云订阅器 - 与标定时使用相同的数据源"""

    DEFAULT_TOPIC = '/lidar/chassis/point_cloud'

    def __init__(self, topic=None, timeout=1.0):
        """
        Args:
            topic: 点云话题，默认 /lidar/chassis/point_cloud
            timeout: 等待新帧的超时时间 (秒)
        """
        if not HAS_ROS:
            raise RuntimeError("ROS not available. Please source ROS environment.")

        self.topic = topic or self.DEFAULT_TOPIC
        self.timeout = timeout

        self._latest_cloud = None
        self._cloud_lock = threading.Lock()
        self._subscriber = None
        self._connected = False

    def connect(self):
        """初始化ROS节点并订阅话题"""
        if self._connected:
            return

        # 初始化ROS节点 (如果尚未初始化)
        try:
            rospy.init_node('lidar_reader', anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass  # 节点已存在

        # 订阅点云话题
        self._subscriber = rospy.Subscriber(
            self.topic,
            PointCloud2,
            self._cloud_callback,
            queue_size=1
        )

        self._connected = True
        print(f"[ROSLiDAR] 已订阅话题: {self.topic}")

        # 等待首帧
        start = time.time()
        while self._latest_cloud is None and time.time() - start < 3.0:
            time.sleep(0.1)

        if self._latest_cloud is None:
            print(f"[WARN] 未收到点云数据，请确保 rslidar_sdk 正在运行")
        else:
            print(f"[ROSLiDAR] 首帧已接收")

    def disconnect(self):
        """取消订阅"""
        if self._subscriber:
            self._subscriber.unregister()
            self._subscriber = None
        self._connected = False
        print("[ROSLiDAR] 已断开连接")

    def _cloud_callback(self, msg):
        """ROS回调函数"""
        with self._cloud_lock:
            self._latest_cloud = msg

    def get_one_frame(self) -> np.ndarray:
        """
        获取最新一帧点云

        Returns:
            points: (N, 4) ndarray - [x, y, z, intensity]
                    坐标系与ROS rslidar_sdk输出完全一致
        """
        if not self._connected:
            self.connect()

        # 清除旧数据，等待新帧
        with self._cloud_lock:
            self._latest_cloud = None

        start = time.time()
        while time.time() - start < self.timeout:
            with self._cloud_lock:
                if self._latest_cloud is not None:
                    cloud_msg = self._latest_cloud
                    break
            time.sleep(0.01)
        else:
            print("[WARN] LiDAR frame timeout")
            return np.empty((0, 4), dtype=np.float32)

        # 使用与标定项目相同的方法提取点云
        # 参考: cam_lidar_calibrate/extract_data.py line 72
        try:
            # 检查是否有intensity字段
            field_names = [f.name for f in cloud_msg.fields]
            if 'intensity' in field_names:
                points_gen = pc2.read_points(
                    cloud_msg,
                    skip_nans=True,
                    field_names=("x", "y", "z", "intensity")
                )
                points = np.array(list(points_gen), dtype=np.float32)
            else:
                points_gen = pc2.read_points(
                    cloud_msg,
                    skip_nans=True,
                    field_names=("x", "y", "z")
                )
                points_xyz = np.array(list(points_gen), dtype=np.float32)
                # 添加空的intensity列
                points = np.column_stack([
                    points_xyz,
                    np.zeros(len(points_xyz), dtype=np.float32)
                ])

            return points

        except Exception as e:
            print(f"[ERROR] 点云解析失败: {e}")
            return np.empty((0, 4), dtype=np.float32)


def test_ros_lidar():
    """测试ROS点云订阅"""
    print("=" * 60)
    print("ROSLiDAR 点云订阅测试")
    print("=" * 60)
    print(f"话题: {ROSLiDAR.DEFAULT_TOPIC}")
    print("坐标系: 与 ROS rslidar_sdk 输出完全一致")

    lidar = ROSLiDAR(timeout=1.0)

    try:
        lidar.connect()

        print("\n获取3帧点云数据...")
        for i in range(3):
            t0 = time.time()
            frame = lidar.get_one_frame()
            dt = time.time() - t0

            if len(frame) > 0:
                print(f"\n帧 {i+1} ({dt*1000:.0f}ms): {len(frame)} 点")
                print(f"  X: [{frame[:, 0].min():.2f}, {frame[:, 0].max():.2f}] median: {np.median(frame[:, 0]):.2f}")
                print(f"  Y: [{frame[:, 1].min():.2f}, {frame[:, 1].max():.2f}] median: {np.median(frame[:, 1]):.2f}")
                print(f"  Z: [{frame[:, 2].min():.2f}, {frame[:, 2].max():.2f}] median: {np.median(frame[:, 2]):.2f}")
            else:
                print(f"\n帧 {i+1}: 超时或无数据")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        lidar.disconnect()


if __name__ == '__main__':
    test_ros_lidar()
