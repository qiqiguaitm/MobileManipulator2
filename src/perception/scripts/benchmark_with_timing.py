#!/usr/bin/env python3
"""
感知节点性能压测 - 带时间分布统计
"""

import rospy
import time
import re
import numpy as np
from perception.srv import DetectObjects
from rosgraph_msgs.msg import Log

class PerceptionBenchmarkWithTiming:
    """感知节点压测 - 带时间分布统计"""

    def __init__(self):
        rospy.init_node('perception_benchmark_timing', anonymous=True)

        # 等待服务
        rospy.loginfo("[Benchmark] 等待 /scene_perception_3d/detect 服务...")
        rospy.wait_for_service('/scene_perception_3d/detect', timeout=10.0)

        self.detect_service = rospy.ServiceProxy('/scene_perception_3d/detect', DetectObjects)

        # 统计数据
        self.durations = []
        self.timings_list = []
        self.success_count = 0
        self.failure_count = 0

        # 订阅 rosout 获取性能统计
        rospy.Subscriber('/rosout', Log, self._rosout_callback)
        self.pending_timings = {}

    def _rosout_callback(self, msg):
        """解析 rosout 中的性能统计"""
        if '性能统计' in msg.msg:
            self.current_timing_block = []
        elif hasattr(self, 'current_timing_block'):
            if msg.msg.startswith('  '):
                # 解析统计数据
                match = re.match(r'\s*-?\s*(.+?):\s+(\d+)\s+ms.*?(\d+\.\d+)%', msg.msg)
                if match:
                    name, ms, percent = match.groups()
                    self.pending_timings[name.strip()] = {
                        'ms': int(ms),
                        'percent': float(percent)
                    }
                elif '总耗时' in msg.msg:
                    match = re.search(r'(\d+)\s+ms', msg.msg)
                    if match:
                        self.pending_timings['总耗时'] = int(match.group(1))

    def run_benchmark(self, prompt: str = 'bottle.cup.box',
                     enable_lidar: bool = True,
                     num_calls: int = 10):
        """运行压测"""
        rospy.loginfo(f"[Benchmark] 开始压测")
        rospy.loginfo(f"  Prompt: {prompt}")
        rospy.loginfo(f"  LiDAR: {'启用' if enable_lidar else '禁用'}")
        rospy.loginfo(f"  调用次数: {num_calls}")
        rospy.loginfo("")

        self.durations = []
        self.timings_list = []
        self.success_count = 0
        self.failure_count = 0

        for i in range(num_calls):
            rospy.loginfo(f"[Benchmark] 第 {i+1}/{num_calls} 次调用...")

            self.pending_timings = {}
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

                    # 等待性能统计日志
                    rospy.sleep(0.5)
                    if self.pending_timings:
                        self.timings_list.append(self.pending_timings.copy())

                    num_objects = len(response.result.objects)
                    rospy.loginfo(f"  ✓ 成功: {duration:.3f}s, 检测到 {num_objects} 个物体")
                else:
                    self.failure_count += 1
                    rospy.logwarn(f"  ✗ 失败: {response.error_message}")

            except rospy.ServiceException as e:
                self.failure_count += 1
                rospy.logerr(f"  ✗ 服务调用失败: {e}")

            # 短暂延迟
            rospy.sleep(0.2)

        self._print_report(enable_lidar)

    def _print_report(self, enable_lidar: bool):
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
            safe_fps = max_fps * 0.8
            rospy.loginfo("")
            rospy.loginfo("推荐配置 (留 20% 余量)")
            rospy.loginfo(f"  auto_detect_rate: {safe_fps:.2f}")

        # 时间分布统计
        if self.timings_list:
            rospy.loginfo("")
            rospy.loginfo("=" * 60)
            rospy.loginfo("时间分布统计 (平均值)")
            rospy.loginfo("=" * 60)

            # 收集所有时间数据
            timing_keys = ['数据同步', 'DINO-X检测', '深度优化', '3D测量']
            if enable_lidar:
                timing_keys.extend(['相机', 'LiDAR'])

            for key in timing_keys:
                values = [t.get(key, {}).get('ms', 0) for t in self.timings_list if key in t]
                if values:
                    avg = np.mean(values)
                    std = np.std(values)
                    rospy.loginfo(f"  {key}: {avg:.0f} ± {std:.0f} ms")

            # 总耗时
            total_values = [t.get('总耗时', 0) for t in self.timings_list]
            if total_values:
                rospy.loginfo(f"\n  总耗时: {np.mean(total_values):.0f} ± {np.std(total_values):.0f} ms")

        rospy.loginfo("=" * 60)


def main():
    import sys

    try:
        benchmark = PerceptionBenchmarkWithTiming()

        # 解析命令行参数
        enable_lidar = True
        if len(sys.argv) > 1 and sys.argv[1] == '--no-lidar':
            enable_lidar = False

        # 运行压测
        benchmark.run_benchmark(
            prompt='bottle.cup.box',
            enable_lidar=enable_lidar,
            num_calls=10
        )

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[Benchmark] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
