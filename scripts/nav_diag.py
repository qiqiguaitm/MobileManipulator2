#!/usr/bin/env python3
"""Nav2 导航实时诊断记录器

记录内容:
- 机器人位姿 (map frame)
- cmd_vel 指令速度
- odom 实际速度
- Nav2 action 状态
- local_costmap 目标点附近代价
- controller_server 日志 (Failed to make progress 等)

用法: export PATH="/usr/bin:$PATH" && python3 scripts/_cc_nav_diag.py
"""

import math
import time
import json
import threading
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
import tf2_ros


class NavDiag(Node):
    def __init__(self):
        super().__init__('nav_diag')

        self._lock = threading.Lock()

        # TF
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf, self)

        # 状态
        self.cmd_vel = (0.0, 0.0)       # (linear.x, angular.z)
        self.odom_vel = (0.0, 0.0)      # (linear.x, angular.z)
        self.odom_pos = (0.0, 0.0)      # (x, y)
        self.goal_pose = None            # (x, y) in map
        self.rosout_lines = deque(maxlen=5)

        # cmd_vel
        self.create_subscription(Twist, '/cmd_vel', self._cb_cmd, 10)
        # odom
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        # goal
        self.create_subscription(
            PoseStamped, '/goal_pose', self._cb_goal,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # rosout (controller_server / bt_navigator)
        self.create_subscription(Log, '/rosout', self._cb_rosout, 50)

        # 日志文件
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = f'scripts/nav_diag_{ts}.jsonl'
        self.flog = open(self.log_path, 'w')
        self.get_logger().info(f'日志: {self.log_path}')

        # 定时采样 10Hz
        self.create_timer(0.1, self._sample)

    def _cb_cmd(self, msg: Twist):
        with self._lock:
            self.cmd_vel = (msg.linear.x, msg.angular.z)

    def _cb_odom(self, msg: Odometry):
        with self._lock:
            self.odom_vel = (
                msg.twist.twist.linear.x,
                msg.twist.twist.angular.z)
            self.odom_pos = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y)

    def _cb_goal(self, msg: PoseStamped):
        with self._lock:
            self.goal_pose = (
                msg.pose.position.x,
                msg.pose.position.y)
        self.get_logger().info(
            f'新目标: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f})')

    def _cb_rosout(self, msg: Log):
        if msg.name not in (
            'controller_server', 'bt_navigator',
            'behavior_server', 'approach_navigator',
            'cleaner_manager_node'):
            return
        with self._lock:
            self.rosout_lines.append(
                f'[{msg.name}] {msg.msg}')

    def _get_robot_pose(self):
        try:
            t = self.tf_buf.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            p = t.transform.translation
            r = t.transform.rotation
            yaw = math.atan2(
                2 * (r.w * r.z + r.x * r.y),
                1 - 2 * (r.y * r.y + r.z * r.z))
            return p.x, p.y, yaw
        except Exception:
            return None, None, None

    def _sample(self):
        rx, ry, ryaw = self._get_robot_pose()
        with self._lock:
            cmd_vx, cmd_wz = self.cmd_vel
            odom_vx, odom_wz = self.odom_vel
            odom_x, odom_y = self.odom_pos
            goal = self.goal_pose
            logs = list(self.rosout_lines)
            self.rosout_lines.clear()

        # 距目标距离
        dist_to_goal = None
        if rx is not None and goal is not None:
            dist_to_goal = math.sqrt(
                (rx - goal[0])**2 + (ry - goal[1])**2)

        rec = {
            't': round(time.time(), 3),
            'robot': {'x': round(rx, 4), 'y': round(ry, 4),
                      'yaw': round(math.degrees(ryaw), 1)
                      } if rx is not None else None,
            'goal': {'x': round(goal[0], 4),
                     'y': round(goal[1], 4)} if goal else None,
            'dist_to_goal': round(dist_to_goal, 4) if dist_to_goal else None,
            'cmd': {'vx': round(cmd_vx, 4), 'wz': round(cmd_wz, 4)},
            'odom_vel': {'vx': round(odom_vx, 4), 'wz': round(odom_wz, 4)},
            'odom_pos': {'x': round(odom_x, 4), 'y': round(odom_y, 4)},
        }
        if logs:
            rec['logs'] = logs

        self.flog.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # 控制台摘要 2Hz
        if not hasattr(self, '_print_cnt'):
            self._print_cnt = 0
        self._print_cnt += 1
        if self._print_cnt % 5 == 0:
            goal_str = (f'({goal[0]:.2f},{goal[1]:.2f}) d={dist_to_goal:.3f}m'
                        if dist_to_goal else 'N/A')
            pos_str = (f'({rx:.2f},{ry:.2f}) {math.degrees(ryaw):.0f}°'
                       if rx else 'N/A')
            print(
                f'[NAV] pos={pos_str} goal={goal_str} '
                f'cmd=({cmd_vx:.3f},{cmd_wz:.3f}) '
                f'odom_v=({odom_vx:.3f},{odom_wz:.3f})')

    def destroy_node(self):
        self.flog.close()
        self.get_logger().info(f'日志已保存: {self.log_path}')
        super().destroy_node()


def main():
    rclpy.init()
    node = NavDiag()
    print('=== Nav2 导航诊断 === Ctrl+C 停止')
    print(f'日志: {node.log_path}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
