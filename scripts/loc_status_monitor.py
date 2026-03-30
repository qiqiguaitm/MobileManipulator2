#!/usr/bin/env python3
"""Real-time monitor for localization status + inlier fraction."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from hdl_localization.msg import ScanMatchingStatus


class LocStatusMonitor(Node):
    def __init__(self):
        super().__init__('loc_status_monitor')
        self._ok = None
        self.create_subscription(Bool, '/localization_status', self._bool_cb, 5)
        self.create_subscription(ScanMatchingStatus, '/status', self._status_cb, 5)
        self.get_logger().info('Waiting for /localization_status + /status ...')

    def _bool_cb(self, msg):
        self._ok = msg.data

    def _status_cb(self, msg):
        tag = '--'
        if self._ok is not None:
            tag = '\033[92mOK\033[0m' if self._ok else '\033[91mBAD\033[0m'
        inlier = msg.inlier_fraction
        converged = msg.has_converged
        err = msg.matching_error
        self.get_logger().info(
            f'[{tag}] inlier={inlier:.3f}  converged={converged}  match_err={err:.4f}')


def main():
    rclpy.init()
    node = LocStatusMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
