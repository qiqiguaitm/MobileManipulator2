#!/usr/bin/python3
"""
三阶段导航测试脚本

测试流程:
1. 订阅感知话题获取融合检测结果
2. 将物体位置从 base_link 转换到 map 坐标系
3. 显示检测到的物体列表
4. 用户选择目标物体
5. 调用 ApproachNavigator 执行三阶段导航

Usage:
    python3 scripts/test_approach_nav.py
    python3 scripts/test_approach_nav.py --auto
"""

import sys
import argparse
import math
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point, TransformStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from perception.msg import Object3DArray
from perception.msg import Object3D

from approach_navigator.navigator import ApproachNavigator
from approach_navigator.nav_types import NavStage


class TestApproachNav(Node):
    """三阶段导航测试节点"""

    def __init__(self, auto_select: bool = False):
        super().__init__('test_approach_nav')

        self.auto_select = auto_select

        # TF2 坐标变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 存储最新检测结果
        self.latest_objects: list = []
        self.objects_lock = __import__('threading').Lock()

        # 订阅感知融合话题
        self.perception_sub = self.create_subscription(
            Object3DArray,
            '/multi_camera_perception/fused/objects_3d',
            self._on_objects_received,
            qos_profile_sensor_data
        )

        self.get_logger().info("已订阅 /multi_camera_perception/fused/objects_3d")
        self.get_logger().info("等待感知数据...")

    def _on_objects_received(self, msg: Object3DArray):
        """接收感知话题数据"""
        with self.objects_lock:
            self.latest_objects = list(msg.objects)

    def get_objects(self, timeout: float = 5.0) -> list:
        """获取当前检测到的物体

        Args:
            timeout: 等待新数据超时时间(秒)

        Returns:
            list[Object3D]: 检测到的物体列表
        """
        import time
        from rclpy.executors import MultiThreadedExecutor
        import threading

        # 使用多线程 executor 来 spin 节点接收消息
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(self)

        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        start_time = time.time()

        # 等待至少有一些数据
        while rclpy.ok():
            with self.objects_lock:
                if self.latest_objects:
                    return self.latest_objects

            if time.time() - start_time > timeout:
                self.get_logger().warn("等待感知数据超时")
                return []

            time.sleep(0.1)

        return []

    def transform_to_map(self, position: Point, source_frame: str = "base_link") -> Point:
        """将位置从 source_frame 转换到 map 坐标系

        Args:
            position: 源坐标系中的位置
            source_frame: 源坐标系名称

        Returns:
            Point: map 坐标系中的位置，失败返回 None
        """
        try:
            # 获取变换
            transform = self.tf_buffer.lookup_transform(
                "map",
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # 转换点
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.point = position

            transformed = do_transform_point(point_stamped, transform)
            return transformed.point

        except Exception as e:
            self.get_logger().error(f"坐标变换失败: {e}")
            return None

    def display_objects(self, objects: list) -> list:
        """显示检测到的物体列表 (带 map 坐标)

        Args:
            objects: Object3D 列表

        Returns:
            list[tuple]: [(Object3D, Point_in_map), ...] 有效物体列表
        """
        valid_objects = []

        print("\n" + "=" * 70)
        print("检测到的物体 (map 坐标系)")
        print("=" * 70)
        print(f"{'序号':<4} {'类别':<15} {'置信度':<8} {'相机':<10} {'map 位置 (x, y, z)'}")
        print("-" * 70)

        for i, obj in enumerate(objects):
            # 转换到 map 坐标系
            map_pos = self.transform_to_map(obj.position, "base_link")

            if map_pos is None:
                print(f"{i:<4} {obj.category:<15} {obj.score:.2f}    {obj.source_camera:<10} [坐标变换失败]")
                continue

            valid_objects.append((obj, map_pos))

            # 计算到机器人的距离
            dist = math.sqrt(map_pos.x**2 + map_pos.y**2)

            print(f"{len(valid_objects)-1:<4} {obj.category:<15} {obj.score:.2f}    "
                  f"{obj.source_camera:<10} ({map_pos.x:.3f}, {map_pos.y:.3f}, {map_pos.z:.3f})")

        print("=" * 70)

        if not valid_objects:
            print("没有有效的物体可供导航")

        return valid_objects

    def select_target(self, valid_objects: list) -> tuple:
        """选择目标物体

        Args:
            valid_objects: [(Object3D, Point_in_map), ...]

        Returns:
            tuple: (Object3D, Point_in_map) 或 None
        """
        if not valid_objects:
            return None

        if self.auto_select:
            # 自动选择最近的物体
            # 需要获取机器人当前位置
            robot_pos = self._get_robot_position()
            if robot_pos is None:
                # 如果获取不到位置，选择第一个
                self.get_logger().info("自动选择第一个物体")
                return valid_objects[0]

            # 找最近的
            min_dist = float('inf')
            nearest = None
            for obj, map_pos in valid_objects:
                dist = math.sqrt(
                    (map_pos.x - robot_pos.x)**2 +
                    (map_pos.y - robot_pos.y)**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest = (obj, map_pos)

            self.get_logger().info(f"自动选择最近物体: {nearest[0].category}, 距离: {min_dist:.2f}m")
            return nearest

        # 交互式选择
        print(f"\n请输入要导航的物体序号 (0-{len(valid_objects)-1}), 或 'q' 退出:")

        while True:
            try:
                choice = input("> ").strip()
                if choice.lower() == 'q':
                    return None

                idx = int(choice)
                if 0 <= idx < len(valid_objects):
                    return valid_objects[idx]
                else:
                    print(f"无效序号，请输入 0-{len(valid_objects)-1}")
            except ValueError:
                print("请输入数字或 'q'")

    def _get_robot_position(self) -> Point:
        """获取机器人在 map 坐标系中的位置"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            pos = Point()
            pos.x = transform.transform.translation.x
            pos.y = transform.transform.translation.y
            pos.z = transform.transform.translation.z
            return pos
        except Exception as e:
            self.get_logger().warn(f"获取机器人位置失败: {e}")
            return None

    def run_navigation(self, target_obj: Object3D, map_position: Point, auto_select: bool = False, skip_stages: set = None):
        """执行三阶段导航

        Args:
            target_obj: 目标物体信息
            map_position: map 坐标系中的目标位置
            auto_select: 是否自动模式
            skip_stages: 要跳过的阶段集合，如 {1}, {1,2}, {2,3} 等
        """
        # 处理 skip_stages 默认值
        if skip_stages is None:
            skip_stages = set()

        if skip_stages:
            print(f"将跳过阶段: {sorted(skip_stages)}")
        import threading

        print("\n" + "=" * 70)
        print(f"开始导航到: {target_obj.category}")
        print(f"目标位置 (map): ({map_position.x:.3f}, {map_position.y:.3f}, {map_position.z:.3f})")
        print("=" * 70)

        # 创建导航器
        navigator = ApproachNavigator()

        # 使用多线程执行器在后台 spin 节点 (接收 TF 和深度图)
        executor = MultiThreadedExecutor()
        executor.add_node(navigator)

        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        # 等待 TF 可用
        print("等待 TF 变换...")
        import time
        for i in range(50):  # 最多等待 5 秒
            try:
                navigator.tf_buffer.lookup_transform(
                    'map', 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                print("TF 变换已就绪")
                break
            except Exception:
                time.sleep(0.1)
        else:
            print("警告: TF 等待超时，继续尝试...")

        # 状态回调
        stage3_confirmed = False

        def status_callback(stage: NavStage, msg: str):
            nonlocal stage3_confirmed
            stage_names = {
                NavStage.IDLE: "空闲",
                NavStage.NAVIGATING: "阶段1: Nav2 导航",
                NavStage.ALIGNING: "阶段2: 对齐",
                NavStage.FINAL_APPROACH: "阶段3: 精确接近",
                NavStage.ARRIVED: "已到达",
                NavStage.FAILED: "失败"
            }
            stage_name = stage_names.get(stage, stage.name)
            print(f"[{stage_name}] {msg}")

            # 阶段3开始时：显示距离，等用户确认后再执行
            if stage == NavStage.FINAL_APPROACH and not stage3_confirmed:
                stage3_confirmed = True
                import time
                time.sleep(0.5)

                # 获取并显示当前障碍物距离
                depth = None
                try:
                    depth = navigator.get_current_depth()
                    if depth is not None:
                        navigator.set_initial_depth(depth)
                except Exception as e:
                    print(f"\n获取深度失败: {e}")

                print("\n" + "=" * 70)
                if depth is not None:
                    print(f"  当前障碍物距离: {depth:.3f} m")
                else:
                    print("  当前障碍物距离: 无数据")

                print("继续执行阶段3...")
                print("=" * 70)

        # 执行导航
        result = navigator.approach_to_target(
            target_position=map_position,
            status_callback=status_callback,
            skip_stages=skip_stages
        )

        # 显示结果
        print("\n" + "=" * 70)
        if result.success:
            print(f"导航成功! 最终距离: {result.final_distance:.3f}m")
        else:
            print(f"导航失败: {result.error_code}")
            print(f"错误信息: {result.error_message}")
            print(f"失败阶段: {result.stage}")
        print("=" * 70)

        # 清理
        executor.shutdown()
        navigator.destroy_node()

        return result.success


def main():
    parser = argparse.ArgumentParser(description='三阶段导航测试')
    parser.add_argument('--auto', action='store_true',
                        help='自动选择最近的物体')
    args = parser.parse_args()

    rclpy.init()

    try:
        # 创建测试节点
        node = TestApproachNav(auto_select=args.auto)

        # 等待 TF 可用
        print("等待 TF 变换可用...")
        import time
        time.sleep(2.0)

        # 等待并获取物体数据
        print("等待感知数据...")
        objects = node.get_objects()
        if not objects:
            print("未检测到物体，退出")
            return

        # 显示并选择
        valid_objects = node.display_objects(objects)
        if not valid_objects:
            return

        target = node.select_target(valid_objects)
        if target is None:
            print("未选择目标，退出")
            return

        target_obj, map_pos = target
        print(f"\n选择目标: {target_obj.category}")
        print(f"map 位置: ({map_pos.x:.3f}, {map_pos.y:.3f}, {map_pos.z:.3f})")

        # 执行导航
        node.run_navigation(target_obj, map_pos, auto_select=args.auto)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
