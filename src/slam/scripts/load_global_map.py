#!/usr/bin/env python3
"""
全局地图加载节点 - ROS1 globalmap_server 等效
采用轮询 service_is_ready() 代替阻塞 wait_for_service，保证 executor 持续 spin，discovery 正常。

Usage:
  ros2 run slam load_global_map.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from hdl_global_localization.srv import SetGlobalMap
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import os
import struct
import time


class GlobalMapLoader(Node):
    def __init__(self):
        super().__init__('global_map_loader')
        self.declare_parameter('global_map_path', '/home/didi/workspace/MobileManipulator2/maps/sc_pgo/GlobalMap.pcd')
        self.declare_parameter('publish_globalmap_topic', '/globalmap')

        self.global_map_path = self.get_parameter('global_map_path').get_parameter_value().string_value
        self.globalmap_topic = self.get_parameter('publish_globalmap_topic').get_parameter_value().string_value

        self.load_done = False
        self.service_names = [
            '/hdl_global_localization/set_global_map',
            '/hdl_global_localization_node/set_global_map',
            '/set_global_map',
        ]
        self._set_global_map_clients = [self.create_client(SetGlobalMap, s) for s in self.service_names]
        self.last_log_time = 0.0

        if not os.path.exists(self.global_map_path):
            self.get_logger().error(f"地图文件不存在: {self.global_map_path}")
            return
        self.get_logger().info(f"全局地图加载启动，轮询等待服务...")
        # 1s 定时器：非阻塞轮询，executor 持续 spin
        self.timer = self.create_timer(1.0, self.try_load_and_publish)

    def try_load_and_publish(self):
        if self.load_done:
            return
        now = time.time()
        for i, (svc, client) in enumerate(zip(self.service_names, self._set_global_map_clients)):
            if client.service_is_ready():
                self.get_logger().info(f"服务可用: {svc}")
                self.timer.cancel()
                self.load_and_publish_map(client)
                self.load_done = True
                return
        if now - self.last_log_time > 10:
            self.get_logger().info("等待 set_global_map 服务...")
            self.last_log_time = now
    
    def read_pcd(self, file_path):
        """读取PCD文件并返回点云数据列表。支持 PCD 标准格式：KEY VALUE 或 KEY=VALUE"""
        points = []
        with open(file_path, 'rb') as f:
            header = {}
            while True:
                line = f.readline().decode('utf-8').strip()
                if not line or line.startswith('#'):
                    continue
                if line == 'DATA binary':
                    break
                # PCD 格式: "WIDTH 462677" 或 "KEY=value"
                if '=' in line:
                    key, value = line.split('=', 1)
                    header[key.strip()] = value.strip()
                else:
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        header[parts[0]] = parts[1].strip()
            
            width = int(header.get('WIDTH', '0'))
            height = int(header.get('HEIGHT', '0'))
            points_field = header.get('POINTS', '')
            point_count = width * height if (width and height) else int(points_field) if points_field else 0
            
            self.get_logger().info(f"PCD文件信息: {width}x{height} = {point_count} 点")
            
            # 读取点数据（假设是PointXYZ类型，每个点4字节*3=12字节）
            for i in range(point_count):
                data = f.read(12)
                if len(data) < 12:
                    break
                x, y, z = struct.unpack('fff', data)
                points.append((x, y, z))
        
        return points
    
    def load_and_publish_map(self, client):
        """加载PCD地图：先发布 /globalmap（hdl_localization 订阅），再异步设置给 hdl_global_localization 服务。"""
        try:
            self.get_logger().info(f"加载PCD: {self.global_map_path}")
            points = self.read_pcd(self.global_map_path)
            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
            point_cloud_msg = pc2.create_cloud_xyz32(header, points)
            self.get_logger().info(f"地图点云: {len(points)} 点")

            # 先发布 /globalmap：hdl_localization 订阅此 topic，必须立即可用
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
            pub = self.create_publisher(PointCloud2, self.globalmap_topic, qos)
            point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(point_cloud_msg)
            self.get_logger().info("globalmap 已发布（latch），hdl_localization 可订阅")

            # 再异步调用 SetGlobalMap 服务（hdl_global_localization 内部状态，不阻塞）
            req = SetGlobalMap.Request()
            req.global_map = point_cloud_msg
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
            if future.result() is not None:
                self.get_logger().info("hdl_global_localization 地图设置成功")
            else:
                self.get_logger().warn(f"hdl_global_localization 服务超时或失败（/globalmap 已发布，定位可用）: {future.exception()}")
        except Exception as e:
            self.get_logger().error(f"加载地图出错: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalMapLoader()
    # 节点需 spin 才能完成服务发现，定时器回调中执行加载与发布
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
