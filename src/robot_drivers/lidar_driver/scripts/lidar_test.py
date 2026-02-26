#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import Odometry
import math

class LidarTestNode(Node):
    def __init__(self):
        super().__init__('lidar_test_listener')

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.pointcloud_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/sensors/lidar/scan',
            self.laser_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info('LiDAR test listener started. Waiting for data...')

    def pointcloud_callback(self, msg):
        self.get_logger().info(f'Received point cloud with {msg.width * msg.height} points')

    def laser_callback(self, msg):
        valid_ranges = [r for r in msg.ranges if not (math.isinf(r) or math.isnan(r))]
        if valid_ranges:
            min_range = min(valid_ranges)
            max_range = max(valid_ranges)
            self.get_logger().info(
                f'Received laser scan with {len(msg.ranges)} points, '
                f'range: {min_range:.2f} - {max_range:.2f} meters'
            )
        else:
            self.get_logger().info(f'Received laser scan with {len(msg.ranges)} points, no valid ranges')

    def odom_callback(self, msg):
        self.get_logger().info(
            f'Received odometry: position '
            f'({msg.pose.pose.position.x:.2f}, '
            f'{msg.pose.pose.position.y:.2f}, '
            f'{msg.pose.pose.position.z:.2f})'
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
