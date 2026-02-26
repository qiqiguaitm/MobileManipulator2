#!/usr/bin/env python3
"""
Robosense Helios-16P LiDAR UDP 直接解析器

完全匹配 ROS rslidar_sdk 驱动的坐标系输出。
参考: rs_driver/decoder_RSHELIOS.hpp, chan_angles.hpp

坐标系: X前 Y左 Z上 (ROS标准)
- x =  distance * cos(vert) * cos(horiz) + RX * cos(horiz)
- y = -distance * cos(vert) * sin(horiz) - RX * sin(horiz)
- z =  distance * sin(vert) + RZ
"""

import socket
import struct
import time
import numpy as np


class RobosenseLiDAR:
    """Robosense Helios-16P LiDAR UDP 直接解析器 (匹配ROS驱动输出)"""

    # MSOP 协议常量
    MSOP_HEADER_SIZE = 42
    MSOP_BLOCK_SIZE = 100
    MSOP_BLOCKS_PER_PACKET = 12
    MSOP_PACKET_SIZE = 1248
    CHANNELS_PER_BLOCK = 32
    LASERS = 16

    # DIFOP 协议常量
    DIFOP_PACKET_SIZE = 1248
    DIFOP_SYNC_BYTES = bytes([0xA5, 0xFF, 0x00, 0x5A, 0x11, 0x11, 0x55, 0x55])

    # MSOP 同步字节
    MSOP_SYNC_BYTES = bytes([0x55, 0xAA, 0x05, 0x5A])
    BLOCK_FLAG = bytes([0xFF, 0xEE])

    # 距离分辨率: 0.0025m
    DISTANCE_RESOLUTION = 0.0025

    # 镜头中心偏移 (来自 decoder_RSHELIOS_16P.hpp)
    RX = 0.03498  # m
    RY = -0.015   # m (未使用)
    RZ = 0.0      # m

    def __init__(self, msop_port=6699, difop_port=7788, timeout=0.3):
        """
        Args:
            msop_port: MSOP 数据端口，默认 6699
            difop_port: DIFOP 设备信息端口，默认 7788
            timeout: 帧超时时间 (秒)
        """
        self.msop_port = msop_port
        self.difop_port = difop_port
        self.timeout = timeout
        self.msop_socket = None
        self.difop_socket = None

        # 角度校准数据 (单位: 0.01°)
        # 默认值 - 将从DIFOP包更新
        self.vert_angles = None  # int32, 单位 0.01°
        self.horiz_angles = None  # int32, 单位 0.01°
        self._angles_loaded = False

        # 默认垂直角度 (Helios-16P)
        self._default_vert_angles = np.array([
            -1500, -1300, -1100, -900, -700, -500, -300, -100,
              100,   300,   500,  700,  900, 1100, 1300, 1500
        ], dtype=np.int32)  # 单位 0.01°

        # 帧同步状态
        self._prev_azimuth = 0
        self._frame_buffer = []
        self._first_frame_skipped = False

    def connect(self):
        """绑定 UDP 端口"""
        # MSOP socket
        self.msop_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.msop_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.msop_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.msop_socket.bind(('0.0.0.0', self.msop_port))
        self.msop_socket.settimeout(self.timeout)

        # DIFOP socket (非阻塞，用于获取角度校准)
        self.difop_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.difop_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.difop_socket.bind(('0.0.0.0', self.difop_port))
        self.difop_socket.setblocking(False)

        self._prev_azimuth = 0
        self._frame_buffer = []
        self._first_frame_skipped = False
        self._angles_loaded = False

        print(f"[RobosenseLiDAR] 已绑定 MSOP:{self.msop_port} DIFOP:{self.difop_port}")

    def disconnect(self):
        """关闭连接"""
        if self.msop_socket:
            self.msop_socket.close()
            self.msop_socket = None
        if self.difop_socket:
            self.difop_socket.close()
            self.difop_socket = None
        print("[RobosenseLiDAR] 已断开连接")

    def _try_load_difop(self):
        """尝试从DIFOP包加载角度校准数据"""
        if self._angles_loaded:
            return

        try:
            data, _ = self.difop_socket.recvfrom(2048)
            if len(data) == self.DIFOP_PACKET_SIZE and data[:8] == self.DIFOP_SYNC_BYTES:
                self._parse_difop(data)
        except BlockingIOError:
            pass
        except Exception:
            pass

        # 如果未加载，使用默认值
        if not self._angles_loaded:
            self.vert_angles = self._default_vert_angles.copy()
            self.horiz_angles = np.zeros(self.LASERS, dtype=np.int32)

    def _parse_difop(self, data: bytes):
        """解析DIFOP包获取角度校准数据"""
        # DIFOP结构 (参考 decoder_RSHELIOS.hpp RSHELIOSDifopPkt):
        # offset 468: vert_angle_cali[32] (每个3字节: sign + value)
        # offset 564: horiz_angle_cali[32]

        vert_offset = 468
        horiz_offset = 564

        vert_angles = []
        horiz_angles = []

        for i in range(32):
            # 垂直角度
            v_sign = data[vert_offset + i * 3]
            v_value = struct.unpack('>H', data[vert_offset + i * 3 + 1:vert_offset + i * 3 + 3])[0]

            if v_sign == 0xFF:  # 无效标记
                continue

            v = v_value if v_sign == 0 else -v_value
            vert_angles.append(v)

            # 水平角度校正
            h_sign = data[horiz_offset + i * 3]
            h_value = struct.unpack('>H', data[horiz_offset + i * 3 + 1:horiz_offset + i * 3 + 3])[0]
            h = h_value if h_sign == 0 else -h_value
            horiz_angles.append(h)

        if len(vert_angles) >= self.LASERS:
            self.vert_angles = np.array(vert_angles[:self.LASERS], dtype=np.int32)
            self.horiz_angles = np.array(horiz_angles[:self.LASERS], dtype=np.int32)
            self._angles_loaded = True
            print(f"[RobosenseLiDAR] 已从DIFOP加载角度校准: vert=[{self.vert_angles[0]/100:.1f}°..{self.vert_angles[-1]/100:.1f}°]")

    def get_one_frame(self) -> np.ndarray:
        """
        获取一帧完整点云

        Returns:
            points: (N, 4) ndarray - [x, y, z, intensity]
                    rslidar 坐标系: X前, Y左, Z上 (与ROS rslidar_sdk一致)
        """
        # 尝试加载DIFOP角度校准
        self._try_load_difop()

        start_time = time.time()
        timeout_duration = self.timeout * 3

        while True:
            if time.time() - start_time > timeout_duration:
                print("[WARN] LiDAR frame timeout")
                return np.empty((0, 4))

            try:
                data, _ = self.msop_socket.recvfrom(2048)
            except socket.timeout:
                self._try_load_difop()  # 超时时尝试获取DIFOP
                continue

            if len(data) != self.MSOP_PACKET_SIZE:
                continue

            block_results = self._parse_packet(data)

            for azimuth, points in block_results:
                if self._prev_azimuth > 27000 and azimuth < 9000:
                    if self._first_frame_skipped and len(self._frame_buffer) > 0:
                        frame = np.vstack(self._frame_buffer)
                        self._frame_buffer = [points] if len(points) > 0 else []
                        self._prev_azimuth = azimuth
                        return frame
                    else:
                        self._first_frame_skipped = True
                        self._frame_buffer = [points] if len(points) > 0 else []
                else:
                    if len(points) > 0:
                        self._frame_buffer.append(points)

                self._prev_azimuth = azimuth

    def _parse_packet(self, data: bytes) -> list:
        """解析单个 MSOP 数据包"""
        if data[:4] != self.MSOP_SYNC_BYTES:
            return []

        results = []
        for i in range(self.MSOP_BLOCKS_PER_PACKET):
            offset = self.MSOP_HEADER_SIZE + i * self.MSOP_BLOCK_SIZE
            block_data = data[offset:offset + self.MSOP_BLOCK_SIZE]

            if block_data[:2] != self.BLOCK_FLAG:
                continue

            azimuth = struct.unpack('>H', block_data[2:4])[0]
            points = self._parse_channels(block_data[4:], azimuth)
            results.append((azimuth, points))

        return results

    def _parse_channels(self, data: bytes, azimuth: int) -> np.ndarray:
        """解析一个块的通道数据 - 完全匹配ROS驱动坐标计算"""
        points = []

        for ch in range(self.CHANNELS_PER_BLOCK):
            ch_offset = ch * 3
            distance_raw = struct.unpack('>H', data[ch_offset:ch_offset+2])[0]
            intensity = data[ch_offset + 2]

            if distance_raw == 0:
                continue

            distance = distance_raw * self.DISTANCE_RESOLUTION

            if distance < 0.1 or distance > 180.0:
                continue

            # 激光索引
            laser = ch % self.LASERS

            # 获取角度 (单位 0.01°)
            angle_vert = self.vert_angles[laser]  # 垂直角度
            angle_horiz = azimuth + self.horiz_angles[laser]  # 水平角度 + 校正

            # 转换为弧度
            vert_rad = np.deg2rad(angle_vert * 0.01)
            horiz_rad = np.deg2rad(angle_horiz * 0.01)

            # ROS rslidar_sdk 坐标计算 (decoder_RSHELIOS.hpp line 276-278)
            # x =  distance * cos(vert) * cos(horiz) + RX * cos(horiz)
            # y = -distance * cos(vert) * sin(horiz) - RX * sin(horiz)
            # z =  distance * sin(vert) + RZ
            cos_vert = np.cos(vert_rad)
            sin_vert = np.sin(vert_rad)
            cos_horiz = np.cos(horiz_rad)
            sin_horiz = np.sin(horiz_rad)

            x = distance * cos_vert * cos_horiz + self.RX * cos_horiz
            y = -distance * cos_vert * sin_horiz - self.RX * sin_horiz
            z = distance * sin_vert + self.RZ

            points.append([x, y, z, intensity])

        return np.array(points, dtype=np.float32) if points else np.empty((0, 4), dtype=np.float32)


def test_lidar():
    """测试 LiDAR 数据接收"""
    print("=" * 60)
    print("RobosenseLiDAR Helios-16P UDP 解析器 (ROS兼容)")
    print("=" * 60)
    print(f"坐标系: X前 Y左 Z上 (ROS标准)")
    print(f"镜头偏移: RX={RobosenseLiDAR.RX*1000:.1f}mm, RZ={RobosenseLiDAR.RZ*1000:.1f}mm")

    lidar = RobosenseLiDAR(msop_port=6699, difop_port=7788, timeout=0.3)

    try:
        lidar.connect()

        print("\n等待接收点云数据...")
        for i in range(3):
            t0 = time.time()
            frame = lidar.get_one_frame()
            dt = time.time() - t0

            if len(frame) > 0:
                print(f"\n帧 {i+1} (耗时 {dt*1000:.0f}ms):")
                print(f"  点数: {len(frame)}")
                print(f"  X 范围: [{frame[:, 0].min():.2f}, {frame[:, 0].max():.2f}] m (前后)")
                print(f"  Y 范围: [{frame[:, 1].min():.2f}, {frame[:, 1].max():.2f}] m (左右)")
                print(f"  Z 范围: [{frame[:, 2].min():.2f}, {frame[:, 2].max():.2f}] m (上下)")
            else:
                print(f"\n帧 {i+1}: 超时或无数据")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        lidar.disconnect()


if __name__ == '__main__':
    test_lidar()
