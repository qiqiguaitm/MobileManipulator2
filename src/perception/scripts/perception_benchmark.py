#!/usr/bin/env python3
"""
感知节点性能压测脚本
测试最大检测帧率和平均耗时
"""

import rospy
import time
import numpy as np
from perception.srv import DetectObjects
from perception.msg import Object3DArray

class PerceptionBenchmark:
    """感知节点压测"""

    def __init__(self):
        rospy.init_node('perception_benchmark', anonymous=True)

        # 等待服务
        rospy.loginfo("[Benchmark] 等待 /scene_perception_3d/detect 服务...")
        rospy.wait_for_service('/scene_perception_3d/detect', timeout=10.0)

        self.detect_service = rospy.ServiceProxy('/scene_perception_3d/detect', DetectObjects)

        # 统计数据
        self.durations = []
        self.success_count = 0
        self.failure_count = 0

    def run_benchmark(self, prompt: str = 'bottle.cup.box',
                     enable_lidar: bool = True,
                     num_calls: int = 20):
        """运行压测

        Args:
            prompt: 检测提示词
            enable_lidar: 是否启用 LiDAR
            num_calls: 调用次数
        """
        rospy.loginfo(f"[Benchmark] 开始压测")
        rospy.loginfo(f"  Prompt: {prompt}")
        rospy.loginfo(f"  LiDAR: {'启用' if enable_lidar else '禁用'}")
        rospy.loginfo(f"  调用次数: {num_calls}")
        rospy.loginfo("")

        self.durations = []
        self.success_count = 0
        self.failure_count = 0

        for i in range(num_calls):
            rospy.loginfo(f"[Benchmark] 第 {i+1}/{num_calls} 次调用...")

            start_time = time.time()

            try:
                response = self.detect_service(
                    prompt=prompt,
                    enable_lidar=enable_lidar
                )

                duration = time.time() - start_time

                if response.success:
                    self.success_count += 1
                    self.durations.append(duration)

                    num_objects = len(response.result.objects)
                    rospy.loginfo(f"  ✓ 成功: {duration:.3f}s, 检测到 {num_objects} 个物体")
                else:
                    self.failure_count += 1
                    rospy.logwarn(f"  ✗ 失败: {response.error_message}")

            except rospy.ServiceException as e:
                self.failure_count += 1
                rospy.logerr(f"  ✗ 服务调用失败: {e}")

            # 短暂延迟避免过载
            rospy.sleep(0.1)

        self._print_report()

    def _print_report(self):
        """打印性能报告"""
        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        rospy.loginfo("压测报告")
        rospy.loginfo("=" * 60)

        rospy.loginfo(f"总调用次数: {self.success_count + self.failure_count}")
        rospy.loginfo(f"  成功: {self.success_count}")
        rospy.loginfo(f"  失败: {self.failure_count}")

        if self.durations:
            durations = np.array(self.durations)

            rospy.loginfo("")
            rospy.loginfo("耗时统计 (单位: 秒)")
            rospy.loginfo(f"  平均: {durations.mean():.3f}s")
            rospy.loginfo(f"  中位数: {np.median(durations):.3f}s")
            rospy.loginfo(f"  最小: {durations.min():.3f}s")
            rospy.loginfo(f"  最大: {durations.max():.3f}s")
            rospy.loginfo(f"  标准差: {durations.std():.3f}s")

            # 计算理论最大帧率
            avg_duration = durations.mean()
            max_fps = 1.0 / avg_duration if avg_duration > 0 else 0

            rospy.loginfo("")
            rospy.loginfo("理论最大帧率")
            rospy.loginfo(f"  基于平均耗时: {max_fps:.2f} Hz")

            # 推荐配置
            safe_fps = max_fps * 0.8  # 留 20% 余量
            rospy.loginfo("")
            rospy.loginfo("推荐配置 (留 20% 余量)")
            rospy.loginfo(f"  auto_detect_rate: {safe_fps:.2f}")

            # 百分位数
            rospy.loginfo("")
            rospy.loginfo("耗时百分位数 (单位: 秒)")
            for p in [50, 75, 90, 95, 99]:
                rospy.loginfo(f"  P{p}: {np.percentile(durations, p):.3f}s")

        rospy.loginfo("=" * 60)


def main():
    import sys

    try:
        benchmark = PerceptionBenchmark()

        # 解析命令行参数
        enable_lidar = True
        if len(sys.argv) > 1 and sys.argv[1] == '--no-lidar':
            enable_lidar = False

        # 运行压测
        benchmark.run_benchmark(
            prompt='bottle.cup.box',
            enable_lidar=enable_lidar,
            num_calls=20  # 调用 20 次
        )

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[Benchmark] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
