"""
ROS2 工具函数

提供 ROS1 到 ROS2 迁移所需的实用工具：
- wait_for_message: 等待单条消息 (ROS2 无内置实现)
- LogThrottle: 日志节流
- QoS 配置
- PointCloud2 解析
"""

import time
import struct
import threading
from typing import Any, Optional, Type, TypeVar

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSLivelinessPolicy,
)
from sensor_msgs.msg import PointCloud2

# Type variable for generic message type
MsgT = TypeVar('MsgT')


# =============================================================================
# QoS Profiles
# =============================================================================

# 传感器数据 QoS (兼容 RealSense 等相机驱动)
# 注意: RealSense 默认使用 RELIABLE + TRANSIENT_LOCAL
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# Latched QoS (替代 ROS1 的 latch=True)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# 默认 QoS (可靠传输)
DEFAULT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
)


# =============================================================================
# wait_for_message Implementation
# =============================================================================

def wait_for_message(
    node: Node,
    msg_type: Type[MsgT],
    topic: str,
    timeout: float = 10.0,
    qos_profile: Optional[QoSProfile] = None,
) -> Optional[MsgT]:
    """
    等待并返回单条消息 (同步阻塞版本)

    ROS2 没有内置的 wait_for_message，此函数提供兼容实现。

    Args:
        node: ROS2 节点实例
        msg_type: 消息类型
        topic: 话题名称
        timeout: 超时时间 (秒)
        qos_profile: QoS 配置 (默认使用 SENSOR_QOS)

    Returns:
        接收到的消息，超时返回 None

    Example:
        >>> info_msg = wait_for_message(self, CameraInfo, '/camera/info', timeout=5.0)
        >>> if info_msg is None:
        ...     self.get_logger().error('获取相机内参超时')
    """
    if qos_profile is None:
        qos_profile = SENSOR_QOS

    received_msg = None
    event = threading.Event()

    def callback(msg):
        nonlocal received_msg
        received_msg = msg
        event.set()

    # 创建临时订阅
    subscription = node.create_subscription(
        msg_type,
        topic,
        callback,
        qos_profile,
    )

    try:
        # 等待消息或超时
        start_time = time.time()
        while not event.is_set() and (time.time() - start_time) < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)

        return received_msg
    finally:
        # 清理订阅
        node.destroy_subscription(subscription)


async def wait_for_message_async(
    node: Node,
    msg_type: Type[MsgT],
    topic: str,
    timeout: float = 10.0,
    qos_profile: Optional[QoSProfile] = None,
) -> Optional[MsgT]:
    """
    等待并返回单条消息 (异步版本)

    Args:
        node: ROS2 节点实例
        msg_type: 消息类型
        topic: 话题名称
        timeout: 超时时间 (秒)
        qos_profile: QoS 配置

    Returns:
        接收到的消息，超时返回 None
    """
    import asyncio

    if qos_profile is None:
        qos_profile = SENSOR_QOS

    received_msg = None
    event = asyncio.Event()

    def callback(msg):
        nonlocal received_msg
        received_msg = msg
        event.set()

    subscription = node.create_subscription(
        msg_type,
        topic,
        callback,
        qos_profile,
    )

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return received_msg
    except asyncio.TimeoutError:
        return None
    finally:
        node.destroy_subscription(subscription)


# =============================================================================
# Log Throttle
# =============================================================================

class LogThrottle:
    """
    日志节流器

    防止高频事件导致日志刷屏。

    Example:
        >>> throttle = LogThrottle(period=5.0)
        >>> while True:
        ...     if throttle.should_log():
        ...         node.get_logger().info('定期日志')
    """

    def __init__(self, period: float = 1.0):
        """
        Args:
            period: 最小日志间隔 (秒)
        """
        self.period = period
        self._last_log_time = 0.0
        self._lock = threading.Lock()

    def should_log(self) -> bool:
        """检查是否应该记录日志"""
        with self._lock:
            now = time.time()
            if now - self._last_log_time >= self.period:
                self._last_log_time = now
                return True
            return False

    def reset(self):
        """重置计时器"""
        with self._lock:
            self._last_log_time = 0.0


class ThrottledLogger:
    """
    带节流功能的日志记录器封装

    Example:
        >>> logger = ThrottledLogger(node.get_logger(), period=2.0)
        >>> logger.info('这条日志每2秒最多输出一次')
    """

    def __init__(self, logger, period: float = 1.0):
        """
        Args:
            logger: ROS2 logger 实例
            period: 日志节流周期 (秒)
        """
        self._logger = logger
        self._throttles: dict[str, LogThrottle] = {}
        self._default_period = period

    def _get_throttle(self, key: str, period: Optional[float] = None) -> LogThrottle:
        """获取或创建指定 key 的节流器"""
        if key not in self._throttles:
            self._throttles[key] = LogThrottle(period or self._default_period)
        return self._throttles[key]

    def info(self, msg: str, throttle_key: Optional[str] = None, period: Optional[float] = None):
        """节流 INFO 日志"""
        key = throttle_key or msg
        if self._get_throttle(key, period).should_log():
            self._logger.info(msg)

    def warn(self, msg: str, throttle_key: Optional[str] = None, period: Optional[float] = None):
        """节流 WARN 日志"""
        key = throttle_key or msg
        if self._get_throttle(key, period).should_log():
            self._logger.warn(msg)

    def error(self, msg: str, throttle_key: Optional[str] = None, period: Optional[float] = None):
        """节流 ERROR 日志"""
        key = throttle_key or msg
        if self._get_throttle(key, period).should_log():
            self._logger.error(msg)

    def debug(self, msg: str, throttle_key: Optional[str] = None, period: Optional[float] = None):
        """节流 DEBUG 日志"""
        key = throttle_key or msg
        if self._get_throttle(key, period).should_log():
            self._logger.debug(msg)


# =============================================================================
# PointCloud2 Parsing
# =============================================================================

def parse_pointcloud2(msg: PointCloud2, fields: list[str] = None) -> np.ndarray:
    """
    解析 PointCloud2 消息为 numpy 数组

    Args:
        msg: PointCloud2 消息
        fields: 需要的字段列表，默认 ['x', 'y', 'z', 'intensity']

    Returns:
        points: (N, len(fields)) 数组

    Note:
        此实现为纯 Python，大规模点云建议使用 sensor_msgs_py 或 ros2_numpy
    """
    if fields is None:
        fields = ['x', 'y', 'z', 'intensity']

    # 构建字段映射
    field_info = {}
    for f in msg.fields:
        # datatype: 1=int8, 2=uint8, 3=int16, 4=uint16, 5=int32, 6=uint32, 7=float32, 8=float64
        dtype_map = {
            1: ('b', 1), 2: ('B', 1), 3: ('h', 2), 4: ('H', 2),
            5: ('i', 4), 6: ('I', 4), 7: ('f', 4), 8: ('d', 8),
        }
        if f.datatype in dtype_map:
            field_info[f.name] = {
                'offset': f.offset,
                'format': dtype_map[f.datatype][0],
                'size': dtype_map[f.datatype][1],
            }

    # 检查必要字段
    available_fields = [f for f in fields if f in field_info]
    if not available_fields:
        return np.empty((0, len(fields)), dtype=np.float32)

    # 解析点云
    num_points = msg.width * msg.height
    points = []

    data = bytes(msg.data)
    for i in range(num_points):
        offset = i * msg.point_step
        point = []

        for field_name in fields:
            if field_name in field_info:
                info = field_info[field_name]
                try:
                    value = struct.unpack_from(
                        info['format'],
                        data,
                        offset + info['offset']
                    )[0]
                except struct.error:
                    value = 0.0
            else:
                value = 0.0
            point.append(value)

        # 过滤无效点 (假设前三个是 x, y, z)
        if len(point) >= 3:
            if np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2]):
                points.append(point)

    return np.array(points, dtype=np.float32) if points else np.empty((0, len(fields)), dtype=np.float32)


def parse_pointcloud2_fast(msg: PointCloud2) -> np.ndarray:
    """
    快速解析 PointCloud2 (仅 xyz)

    使用 numpy 重新解释内存，比逐点解析快得多。
    假设点云格式为标准 XYZI (float32 x 4)

    Args:
        msg: PointCloud2 消息

    Returns:
        points: (N, 4) 数组 [x, y, z, intensity]
    """
    # 检查是否为标准格式
    field_names = [f.name for f in msg.fields]
    if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
        return np.empty((0, 4), dtype=np.float32)

    # 获取字段偏移
    x_offset = next((f.offset for f in msg.fields if f.name == 'x'), 0)
    y_offset = next((f.offset for f in msg.fields if f.name == 'y'), 4)
    z_offset = next((f.offset for f in msg.fields if f.name == 'z'), 8)
    i_offset = next((f.offset for f in msg.fields if f.name == 'intensity'), 12)

    # 转换为 numpy 数组
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    num_points = msg.width * msg.height

    if len(data) < num_points * msg.point_step:
        return np.empty((0, 4), dtype=np.float32)

    # 重塑为点云
    data = data.reshape(num_points, msg.point_step)

    # 提取各字段
    x = data[:, x_offset:x_offset+4].view(np.float32).flatten()
    y = data[:, y_offset:y_offset+4].view(np.float32).flatten()
    z = data[:, z_offset:z_offset+4].view(np.float32).flatten()

    if i_offset < msg.point_step - 3:
        intensity = data[:, i_offset:i_offset+4].view(np.float32).flatten()
    else:
        intensity = np.zeros(num_points, dtype=np.float32)

    # 合并并过滤无效点
    points = np.column_stack([x, y, z, intensity])
    valid_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])

    return points[valid_mask]


# =============================================================================
# Time Utilities
# =============================================================================

def stamp_to_sec(stamp) -> float:
    """将 ROS2 时间戳转换为秒"""
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp(sec: float, clock=None):
    """将秒转换为 ROS2 时间戳"""
    from builtin_interfaces.msg import Time
    stamp = Time()
    stamp.sec = int(sec)
    stamp.nanosec = int((sec - stamp.sec) * 1e9)
    return stamp


# =============================================================================
# CvBridge NumPy 2.x Compatible Implementation
# =============================================================================

import array
import cv2
from sensor_msgs.msg import Image


class CvBridgeNumPy2:
    """
    纯 Python 实现的 cv_bridge 替代品

    解决 NumPy 2.x 与 cv_bridge C++ 扩展的 ABI 不兼容问题。
    使用 array.array 进行快速数据转换，性能优于原生 cv_bridge。

    支持的编码格式:
    - 8-bit: rgb8, bgr8, rgba8, bgra8, mono8, 8UC1, 8UC3, 8UC4
    - 16-bit: mono16, 16UC1
    - 32-bit float: 32FC1

    Example:
        >>> bridge = CvBridgeNumPy2()
        >>> cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        >>> msg = bridge.cv2_to_imgmsg(cv_image, 'bgr8')
    """

    # 编码格式映射: encoding -> (numpy dtype, channels)
    ENCODING_MAP = {
        # 8-bit 格式
        'rgb8': (np.uint8, 3),
        'bgr8': (np.uint8, 3),
        'rgba8': (np.uint8, 4),
        'bgra8': (np.uint8, 4),
        'mono8': (np.uint8, 1),
        '8UC1': (np.uint8, 1),
        '8UC3': (np.uint8, 3),
        '8UC4': (np.uint8, 4),
        # 16-bit 格式
        'mono16': (np.uint16, 1),
        '16UC1': (np.uint16, 1),
        # 32-bit float 格式
        '32FC1': (np.float32, 1),
    }

    # 颜色转换映射: (src_encoding, dst_encoding) -> cv2 转换码
    COLOR_CONVERSIONS = {
        ('rgb8', 'bgr8'): cv2.COLOR_RGB2BGR,
        ('bgr8', 'rgb8'): cv2.COLOR_BGR2RGB,
        ('rgba8', 'bgra8'): cv2.COLOR_RGBA2BGRA,
        ('bgra8', 'rgba8'): cv2.COLOR_BGRA2RGBA,
        ('rgb8', 'mono8'): cv2.COLOR_RGB2GRAY,
        ('bgr8', 'mono8'): cv2.COLOR_BGR2GRAY,
        ('mono8', 'rgb8'): cv2.COLOR_GRAY2RGB,
        ('mono8', 'bgr8'): cv2.COLOR_GRAY2BGR,
        ('rgba8', 'rgb8'): cv2.COLOR_RGBA2RGB,
        ('bgra8', 'bgr8'): cv2.COLOR_BGRA2BGR,
        ('rgb8', 'rgba8'): cv2.COLOR_RGB2RGBA,
        ('bgr8', 'bgra8'): cv2.COLOR_BGR2BGRA,
    }

    def imgmsg_to_cv2(self, msg: Image, desired_encoding: str = 'passthrough') -> np.ndarray:
        """
        将 ROS Image 消息转换为 OpenCV 图像 (numpy 数组)

        Args:
            msg: ROS sensor_msgs/Image 消息
            desired_encoding: 目标编码格式，'passthrough' 表示保持原格式

        Returns:
            numpy 数组格式的图像

        Raises:
            ValueError: 不支持的编码格式
        """
        src_encoding = msg.encoding

        # 获取源编码信息
        if src_encoding not in self.ENCODING_MAP:
            raise ValueError(f"不支持的编码格式: {src_encoding}")

        dtype, channels = self.ENCODING_MAP[src_encoding]

        # 从消息数据构建 numpy 数组
        # 使用 bytes() 确保数据可以被 frombuffer 处理
        img = np.frombuffer(bytes(msg.data), dtype=dtype)

        # 重塑为图像形状
        if channels == 1:
            img = img.reshape(msg.height, msg.width)
        else:
            img = img.reshape(msg.height, msg.width, channels)

        # 确保数组是连续的（frombuffer 返回的是只读的，需要复制）
        img = np.ascontiguousarray(img)

        # 颜色空间转换
        if desired_encoding != 'passthrough' and desired_encoding != src_encoding:
            conversion_key = (src_encoding, desired_encoding)
            if conversion_key in self.COLOR_CONVERSIONS:
                img = cv2.cvtColor(img, self.COLOR_CONVERSIONS[conversion_key])
            elif desired_encoding in self.ENCODING_MAP:
                # 尝试通过中间格式转换
                pass  # 保持原样，复杂转换交给用户处理

        return img

    def cv2_to_imgmsg(self, img: np.ndarray, encoding: str = 'bgr8') -> Image:
        """
        将 OpenCV 图像 (numpy 数组) 转换为 ROS Image 消息

        Args:
            img: numpy 数组格式的图像
            encoding: 图像编码格式

        Returns:
            ROS sensor_msgs/Image 消息

        Raises:
            ValueError: 不支持的编码格式
        """
        if encoding not in self.ENCODING_MAP:
            raise ValueError(f"不支持的编码格式: {encoding}")

        # 确保数组是 C 连续的
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        # 创建消息
        msg = Image()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False

        # 计算 step (每行字节数)
        if img.ndim == 2:
            msg.step = img.shape[1] * img.dtype.itemsize
        else:
            msg.step = img.shape[1] * img.shape[2] * img.dtype.itemsize

        # 使用 array.array 进行快速数据转换
        # 这比直接赋值 bytes 快约 2-3 倍
        msg.data = array.array('B', img.tobytes())

        return msg

    def compressed_imgmsg_to_cv2(self, msg, desired_encoding: str = 'passthrough') -> np.ndarray:
        """
        将压缩的 ROS Image 消息转换为 OpenCV 图像

        Args:
            msg: ROS sensor_msgs/CompressedImage 消息
            desired_encoding: 目标编码格式

        Returns:
            numpy 数组格式的图像
        """
        # 解码压缩图像
        np_arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("无法解码压缩图像")

        # 颜色空间转换
        if desired_encoding == 'rgb8' and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def cv2_to_compressed_imgmsg(self, img: np.ndarray, dst_format: str = 'jpg') -> 'CompressedImage':
        """
        将 OpenCV 图像转换为压缩的 ROS Image 消息

        Args:
            img: numpy 数组格式的图像
            dst_format: 压缩格式 ('jpg', 'png')

        Returns:
            ROS sensor_msgs/CompressedImage 消息
        """
        from sensor_msgs.msg import CompressedImage

        msg = CompressedImage()
        msg.format = f"{'jpeg' if dst_format == 'jpg' else dst_format}"

        # 编码图像
        ext = f".{dst_format}"
        _, encoded = cv2.imencode(ext, img)
        msg.data = array.array('B', encoded.tobytes())

        return msg


# 创建全局实例，方便直接使用
_cv_bridge = None


def get_cv_bridge() -> CvBridgeNumPy2:
    """
    获取 CvBridge 实例

    Returns:
        CvBridgeNumPy2 实例
    """
    global _cv_bridge
    if _cv_bridge is None:
        _cv_bridge = CvBridgeNumPy2()
    return _cv_bridge
