#!/usr/bin/env python3
"""
Globalmap publisher with dual output (ROS2 版本):
- /globalmap: Full point cloud for HDL localization (with intensity)
- /globalmap_visual: Colored obstacles only for RViz (no ground fog)

使用自适应地面分离 (与 ROS1 版本一致)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np
import struct
import os
from scipy import ndimage

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from hdl_global_localization.srv import SetGlobalMap


def load_pcd(pcd_path):
    """Load PCD file, return points and intensity."""
    with open(pcd_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('DATA'):
                break

        num_points = 0
        fields_info = []
        sizes_info = []
        for line in header_lines:
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('FIELDS'):
                fields_info = line.split()[1:]
            elif line.startswith('SIZE'):
                sizes_info = [int(x) for x in line.split()[1:]]

        point_step = sum(sizes_info) if sizes_info else len(fields_info) * 4
        has_intensity = 'intensity' in fields_info

        points = []
        intensities = []

        for i in range(num_points):
            data = f.read(point_step)
            if len(data) < point_step:
                break

            x, y, z = struct.unpack('fff', data[:12])
            points.append([x, y, z])

            if has_intensity and len(data) >= 16:
                intensity = struct.unpack('f', data[12:16])[0]
            else:
                intensity = 1.0
            intensities.append(intensity)

    return np.array(points), np.array(intensities)


def adaptive_ground_separation(points, grid_res=0.5, ground_margin=0.08, obstacle_min=0.10):
    """
    自适应地面分离 (与 ROS1 版本一致)
    返回: ground_mask (True=地面点)
    """
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    grid_w = int(np.ceil((x_max - x_min) / grid_res))
    grid_h = int(np.ceil((y_max - y_min) / grid_res))

    # 每个网格取最低10%点的均值作为地面高度
    ground_grid = np.full((grid_h, grid_w), np.nan)
    for gy in range(grid_h):
        for gx in range(grid_w):
            mask = (points[:, 0] >= x_min + gx * grid_res) & \
                   (points[:, 0] < x_min + (gx + 1) * grid_res) & \
                   (points[:, 1] >= y_min + gy * grid_res) & \
                   (points[:, 1] < y_min + (gy + 1) * grid_res)
            cell_pts = points[mask, 2]
            if len(cell_pts) > 10:
                sorted_z = np.sort(cell_pts)
                n_low = max(1, len(sorted_z) // 10)
                ground_grid[gy, gx] = sorted_z[:n_low].mean()

    ground_mean = np.nanmean(ground_grid)
    ground_smooth = ndimage.uniform_filter(
        np.nan_to_num(ground_grid, nan=ground_mean), size=3)

    # 计算每个点的相对高度
    ground_mask = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        gx = min(int((p[0] - x_min) / grid_res), grid_w - 1)
        gy = min(int((p[1] - y_min) / grid_res), grid_h - 1)
        local_ground = ground_smooth[gy, gx]
        relative_z = p[2] - local_ground

        # 相对高度 < ground_margin 视为地面
        if relative_z < ground_margin:
            ground_mask[i] = True

    return ground_mask


def create_full_cloud(points, intensities, header):
    """Create PointCloud2 with XYZI for HDL localization."""
    cloud_data = []
    for i in range(len(points)):
        x, y, z = points[i]
        intensity = intensities[i]
        cloud_data.append(struct.pack('ffff', x, y, z, intensity))

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.data = b''.join(cloud_data)
    msg.is_dense = True

    return msg


def create_visual_cloud(points, header):
    """Create colored PointCloud2 for visualization (height-colored, obstacles only)."""
    if len(points) == 0:
        return None

    # 使用实际点云的 Z 范围进行颜色映射
    z_min, z_max = points[:, 2].min(), min(points[:, 2].max(), 2.0)
    z_range = z_max - z_min if z_max > z_min else 1.0

    cloud_data = []
    for i in range(len(points)):
        x, y, z = points[i]

        # Height-based coloring (蓝 -> 青 -> 绿 -> 黄 -> 红)
        z_norm = np.clip((z - z_min) / z_range, 0, 1)
        if z_norm < 0.25:
            t = z_norm / 0.25
            r, g, b = 0, int(t * 255), 255
        elif z_norm < 0.5:
            t = (z_norm - 0.25) / 0.25
            r, g, b = 0, 255, int((1 - t) * 255)
        elif z_norm < 0.75:
            t = (z_norm - 0.5) / 0.25
            r, g, b = int(t * 255), 255, 0
        else:
            t = (z_norm - 0.75) / 0.25
            r, g, b = 255, int((1 - t) * 255), 0

        rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        cloud_data.append(struct.pack('ffff', x, y, z, rgb_packed))

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.data = b''.join(cloud_data)
    msg.is_dense = True

    return msg


class GlobalmapPublisher(Node):
    def __init__(self):
        super().__init__('globalmap_publisher')

        # 参数
        self.declare_parameter('globalmap_pcd', '')
        self.declare_parameter('use_adaptive', True)
        self.declare_parameter('frame_id', 'map')

        self.pcd_path = self.get_parameter('globalmap_pcd').value
        self.use_adaptive = self.get_parameter('use_adaptive').value
        self.frame_id = self.get_parameter('frame_id').value

        # Timer管理（避免重复调用service）
        self._retry_timer = None
        self._service_called = False

        if not self.pcd_path or not os.path.exists(self.pcd_path):
            self.get_logger().error(f"PCD file not found: {self.pcd_path}")
            return

        # QoS: Transient Local (latch)
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers
        self.pub_full = self.create_publisher(PointCloud2, '/globalmap', qos)
        self.pub_visual = self.create_publisher(PointCloud2, '/globalmap_visual', qos)

        # Service client for hdl_global_localization
        self._service_names = [
            '/hdl_global_localization/set_global_map',
            '/hdl_global_localization_node/set_global_map',
            '/set_global_map',
        ]
        self._svc_clients = [self.create_client(SetGlobalMap, s) for s in self._service_names]

        # 加载并处理地图
        self.load_and_publish()

    def load_and_publish(self):
        self.get_logger().info(f"Loading {self.pcd_path}")

        try:
            points, intensities = load_pcd(self.pcd_path)
            self.get_logger().info(f"Loaded {len(points)} points")
        except Exception as e:
            self.get_logger().error(f"Failed to load PCD: {e}")
            return

        # 地面分离
        if self.use_adaptive:
            self.get_logger().info("Using adaptive ground separation...")
            ground_mask = adaptive_ground_separation(points)
        else:
            ground_thresh = 0.25
            ground_mask = points[:, 2] < ground_thresh

        ground_pts = points[ground_mask]
        obstacle_pts = points[~ground_mask]
        obstacle_intensities = intensities[~ground_mask]

        self.get_logger().info(f"Ground: {len(ground_pts)}, Obstacles: {len(obstacle_pts)}")

        # 创建消息
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = self.get_clock().now().to_msg()

        full_cloud = create_full_cloud(points, intensities, header)
        visual_cloud = create_visual_cloud(obstacle_pts, header)

        # 发布
        self.pub_full.publish(full_cloud)
        self.get_logger().info("Published /globalmap (full, for HDL)")

        if visual_cloud:
            self.pub_visual.publish(visual_cloud)
            self.get_logger().info("Published /globalmap_visual (colored obstacles)")

        # 调用 SetGlobalMap 服务
        self.call_set_global_map(full_cloud)

        self.get_logger().info("Globalmap published. Node will keep running for latch.")

    def call_set_global_map(self, cloud_msg):
        """异步调用 SetGlobalMap 服务"""
        if self._service_called:
            return  # 防止重复调用

        for svc_name, client in zip(self._service_names, self._svc_clients):
            if client.service_is_ready():
                self.get_logger().info(f"Calling service: {svc_name}")
                req = SetGlobalMap.Request()
                req.global_map = cloud_msg
                future = client.call_async(req)
                future.add_done_callback(self.service_callback)
                self._service_called = True
                return

        # 如果服务不可用，创建一次性重试timer
        self.get_logger().info("Waiting for set_global_map service...")
        if self._retry_timer is None:
            self._retry_timer = self.create_timer(2.0, lambda: self.retry_service_call(cloud_msg))

    def retry_service_call(self, cloud_msg):
        if self._service_called:
            if self._retry_timer is not None:
                self._retry_timer.cancel()
                self._retry_timer = None
            return

        for svc_name, client in zip(self._service_names, self._svc_clients):
            if client.service_is_ready():
                self.get_logger().info(f"Service ready: {svc_name}")
                req = SetGlobalMap.Request()
                req.global_map = cloud_msg
                future = client.call_async(req)
                future.add_done_callback(self.service_callback)
                self._service_called = True
                # 销毁timer
                if self._retry_timer is not None:
                    self._retry_timer.cancel()
                    self._retry_timer = None
                return

    def service_callback(self, future):
        try:
            result = future.result()
            self.get_logger().info("SetGlobalMap service call succeeded")
        except Exception as e:
            self.get_logger().warn(f"SetGlobalMap service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalmapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
