#!/usr/bin/env python3
"""
Approach Navigator ROS2 节点

提供三阶段接近导航的 ROS2 节点封装
"""

import math
import rclpy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Point
from std_srvs.srv import Trigger

from approach_navigator import ApproachNavigator, compute_approach_pose


class ApproachNavigatorNode:
    """接近导航器 ROS2 节点封装

    提供测试服务接口
    """

    def __init__(self):
        """初始化节点"""
        # 创建导航器实例
        self.navigator = ApproachNavigator()

        # 测试服务: 向前接近 1m
        self.navigator.create_service(
            Trigger,
            '~/test_approach',
            self._test_approach_callback
        )

        self.navigator.get_logger().info("接近导航器节点已启动")

    def _test_approach_callback(self, request, response):
        """测试服务回调 - 向前接近 1m

        Args:
            request: Trigger 请求
            response: Trigger 响应

        Returns:
            Trigger 响应，包含执行结果
        """
        # 获取当前位姿
        robot_pos, robot_yaw = self.navigator._get_robot_pose()
        if robot_pos is None:
            response.success = False
            response.message = "无法获取机器人位姿"
            return response

        # 测试目标: 前方 1m
        target = Point()
        target.x = robot_pos.x + 1.0 * math.cos(robot_yaw)
        target.y = robot_pos.y + 1.0 * math.sin(robot_yaw)
        target.z = 0.0

        # 计算接近位姿
        approach_pose = compute_approach_pose(
            target,
            robot_pos,
            approach_distance=self.navigator.config.approach_distance,
            robot_front_offset=self.navigator.config.robot_front_offset
        )

        if approach_pose is None:
            response.success = True
            response.message = "已在目标位置"
            return response

        # 执行接近
        result = self.navigator.approach_to_pose(approach_pose, target)

        response.success = result.success
        response.message = f"{result.error_code or '成功'}, 距离: {result.final_distance:.3f}m"
        return response


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    node = ApproachNavigatorNode()

    # 使用多线程执行器以支持并发回调
    executor = MultiThreadedExecutor()
    executor.add_node(node.navigator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
