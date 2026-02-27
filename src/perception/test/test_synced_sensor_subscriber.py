#!/usr/bin/env python3
"""
测试同步传感器订阅器

注意: 完整的集成测试需要 ROS2 环境和传感器数据
这里只测试不依赖 ROS2 运行时的部分
"""

import pytest
import numpy as np


class TestSyncedSensorSubscriberUnit:
    """SyncedSensorSubscriber 单元测试 (无 ROS2 运行时)"""

    def test_import(self):
        """测试导入"""
        from perception.synced_sensor_subscriber import SyncedSensorSubscriber
        assert SyncedSensorSubscriber is not None

    def test_init_requires_node(self):
        """测试初始化需要 node 参数"""
        from perception.synced_sensor_subscriber import SyncedSensorSubscriber

        # 应该抛出 TypeError，因为 node 是必需参数
        with pytest.raises(TypeError):
            SyncedSensorSubscriber()

    def test_default_topics(self):
        """测试默认 topic 值"""
        # 由于需要 node，这里只能测试类定义
        from perception.synced_sensor_subscriber import SyncedSensorSubscriber
        import inspect

        sig = inspect.signature(SyncedSensorSubscriber.__init__)
        params = sig.parameters

        assert params['color_topic'].default == '/camera/top/color/image_raw'
        assert params['depth_topic'].default == '/camera/top/aligned_depth_to_color/image_raw'
        assert params['info_topic'].default == '/camera/top/color/camera_info'
        assert params['lidar_topic'].default == '/lidar/chassis/point_cloud'
        assert params['enable_lidar'].default is True
        assert params['sync_slop'].default == 0.1


class TestSyncedSensorSubscriberIntegration:
    """SyncedSensorSubscriber 集成测试 (需要 ROS2 运行时)"""

    @pytest.fixture
    def ros2_node(self):
        """创建 ROS2 节点 fixture"""
        import rclpy
        rclpy.init()
        node = rclpy.create_node('test_synced_sensor')
        yield node
        node.destroy_node()
        rclpy.shutdown()

    @pytest.mark.skip(reason="需要完整 ROS2 环境和传感器")
    def test_connect_disconnect(self, ros2_node):
        """测试连接和断开"""
        from perception.synced_sensor_subscriber import SyncedSensorSubscriber

        subscriber = SyncedSensorSubscriber(
            node=ros2_node,
            enable_lidar=False,
        )

        # 初始状态
        assert subscriber.connected is False
        assert subscriber.intrinsics is None

        # 连接 (会超时，因为没有传感器)
        result = subscriber.connect(timeout=1.0)
        # 预期失败
        assert result is False

        # 断开
        subscriber.disconnect()
        assert subscriber.connected is False

    @pytest.mark.skip(reason="需要完整 ROS2 环境和传感器")
    def test_properties(self, ros2_node):
        """测试属性"""
        from perception.synced_sensor_subscriber import SyncedSensorSubscriber

        subscriber = SyncedSensorSubscriber(
            node=ros2_node,
            color_topic='/test/color',
            depth_topic='/test/depth',
            info_topic='/test/info',
            enable_lidar=False,
            sync_slop=0.2,
        )

        assert subscriber.node is ros2_node
        assert subscriber.color_topic == '/test/color'
        assert subscriber.depth_topic == '/test/depth'
        assert subscriber.info_topic == '/test/info'
        assert subscriber.enable_lidar is False
        assert subscriber.sync_slop == 0.2

    @pytest.mark.skip(reason="需要完整 ROS2 环境和传感器")
    def test_get_data_without_connect(self, ros2_node):
        """测试未连接时获取数据"""
        from perception.synced_sensor_subscriber import SyncedSensorSubscriber

        subscriber = SyncedSensorSubscriber(
            node=ros2_node,
            enable_lidar=False,
        )

        with pytest.raises(RuntimeError):
            subscriber.get_synced_data()


class TestCamerInfoParsing:
    """相机内参解析测试"""

    def test_parse_camera_info(self):
        """测试解析相机内参"""
        from sensor_msgs.msg import CameraInfo

        # 创建模拟的 CameraInfo
        info = CameraInfo()
        info.width = 640
        info.height = 480
        info.k = [
            500.0, 0.0, 320.0,
            0.0, 500.0, 240.0,
            0.0, 0.0, 1.0
        ]

        # 手动测试解析逻辑
        intrinsics = {
            'fx': info.k[0],
            'fy': info.k[4],
            'cx': info.k[2],
            'cy': info.k[5],
            'width': info.width,
            'height': info.height,
        }

        assert intrinsics['fx'] == 500.0
        assert intrinsics['fy'] == 500.0
        assert intrinsics['cx'] == 320.0
        assert intrinsics['cy'] == 240.0
        assert intrinsics['width'] == 640
        assert intrinsics['height'] == 480


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
