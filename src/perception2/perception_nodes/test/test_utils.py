#!/usr/bin/env python3
"""
测试 ROS2 工具函数
"""

import pytest
import numpy as np
import struct


class TestLogThrottle:
    """LogThrottle 单元测试"""

    def test_should_log_first_call(self):
        """第一次调用应该返回 True"""
        from perception_nodes.utils import LogThrottle

        throttle = LogThrottle(period=1.0)
        assert throttle.should_log() is True

    def test_should_log_within_period(self):
        """在周期内连续调用应该返回 False"""
        from perception_nodes.utils import LogThrottle
        import time

        throttle = LogThrottle(period=1.0)
        throttle.should_log()  # 第一次
        assert throttle.should_log() is False  # 立即第二次应该被节流

    def test_should_log_after_period(self):
        """超过周期后应该返回 True"""
        from perception_nodes.utils import LogThrottle
        import time

        throttle = LogThrottle(period=0.1)
        throttle.should_log()  # 第一次
        time.sleep(0.15)  # 等待超过周期
        assert throttle.should_log() is True

    def test_reset(self):
        """重置后应该立即允许日志"""
        from perception_nodes.utils import LogThrottle

        throttle = LogThrottle(period=100.0)  # 很长的周期
        throttle.should_log()  # 第一次
        assert throttle.should_log() is False  # 被节流
        throttle.reset()
        assert throttle.should_log() is True  # 重置后立即可用


class TestQoSProfiles:
    """QoS 配置测试"""

    def test_sensor_qos_exists(self):
        """SENSOR_QOS 应该存在且配置正确"""
        from perception_nodes.utils import SENSOR_QOS
        from rclpy.qos import QoSReliabilityPolicy

        assert SENSOR_QOS is not None
        assert SENSOR_QOS.reliability == QoSReliabilityPolicy.BEST_EFFORT

    def test_latched_qos_exists(self):
        """LATCHED_QOS 应该存在且配置正确"""
        from perception_nodes.utils import LATCHED_QOS
        from rclpy.qos import QoSDurabilityPolicy

        assert LATCHED_QOS is not None
        assert LATCHED_QOS.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL

    def test_default_qos_exists(self):
        """DEFAULT_QOS 应该存在且配置正确"""
        from perception_nodes.utils import DEFAULT_QOS
        from rclpy.qos import QoSReliabilityPolicy

        assert DEFAULT_QOS is not None
        assert DEFAULT_QOS.reliability == QoSReliabilityPolicy.RELIABLE


class TestPointCloud2Parsing:
    """PointCloud2 解析测试"""

    def _create_mock_pointcloud(self, points: list) -> 'PointCloud2':
        """创建模拟的 PointCloud2 消息"""
        from sensor_msgs.msg import PointCloud2, PointField

        msg = PointCloud2()
        msg.height = 1
        msg.width = len(points)
        msg.point_step = 16  # 4 floats * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        # 定义字段
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # 打包数据
        data = b''
        for p in points:
            data += struct.pack('ffff', p[0], p[1], p[2], p[3] if len(p) > 3 else 0.0)
        msg.data = list(data)

        return msg

    def test_parse_empty_pointcloud(self):
        """测试解析空点云"""
        from perception_nodes.utils import parse_pointcloud2_fast

        msg = self._create_mock_pointcloud([])
        result = parse_pointcloud2_fast(msg)

        assert result.shape == (0, 4)

    def test_parse_single_point(self):
        """测试解析单点"""
        from perception_nodes.utils import parse_pointcloud2_fast

        msg = self._create_mock_pointcloud([[1.0, 2.0, 3.0, 100.0]])
        result = parse_pointcloud2_fast(msg)

        assert result.shape == (1, 4)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0, 100.0])

    def test_parse_multiple_points(self):
        """测试解析多点"""
        from perception_nodes.utils import parse_pointcloud2_fast

        points = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
        ]
        msg = self._create_mock_pointcloud(points)
        result = parse_pointcloud2_fast(msg)

        assert result.shape == (3, 4)
        for i, p in enumerate(points):
            np.testing.assert_array_almost_equal(result[i], p)

    def test_parse_filters_nan(self):
        """测试过滤 NaN 点"""
        from perception_nodes.utils import parse_pointcloud2_fast

        points = [
            [1.0, 2.0, 3.0, 10.0],
            [float('nan'), 0.0, 0.0, 20.0],  # 应该被过滤
            [0.0, 1.0, 0.0, 30.0],
        ]
        msg = self._create_mock_pointcloud(points)
        result = parse_pointcloud2_fast(msg)

        assert result.shape == (2, 4)  # 只有2个有效点


class TestTimeUtilities:
    """时间工具测试"""

    def test_stamp_to_sec(self):
        """测试时间戳转秒"""
        from perception_nodes.utils import stamp_to_sec
        from builtin_interfaces.msg import Time

        stamp = Time()
        stamp.sec = 10
        stamp.nanosec = 500000000  # 0.5秒

        result = stamp_to_sec(stamp)
        assert abs(result - 10.5) < 1e-9

    def test_sec_to_stamp(self):
        """测试秒转时间戳"""
        from perception_nodes.utils import sec_to_stamp

        stamp = sec_to_stamp(10.5)
        assert stamp.sec == 10
        assert abs(stamp.nanosec - 500000000) < 1000  # 允许微小误差


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
