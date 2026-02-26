#!/usr/bin/env python3
"""
测试预热效果 - 对比首次调用和后续调用的性能
"""

import rospy
import time
from perception.srv import DetectObjects

def test_warmup():
    rospy.init_node('test_warmup', anonymous=True)

    # 等待服务
    rospy.loginfo("等待 /scene_perception_3d/detect 服务...")
    rospy.wait_for_service('/scene_perception_3d/detect', timeout=10.0)

    detect_service = rospy.ServiceProxy('/scene_perception_3d/detect', DetectObjects)

    # 测试 3 次调用，观察第一次是否有冷启动
    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("测试预热效果 (禁用 LiDAR)")
    rospy.loginfo("="*60)

    times = []
    for i in range(3):
        rospy.loginfo(f"\n第 {i+1} 次调用...")

        start = time.time()
        response = detect_service(prompt='bottle.cup.box', enable_lidar=False)
        duration = time.time() - start

        times.append(duration)

        if response.success:
            num_obj = len(response.result.objects)
            rospy.loginfo(f"  耗时: {duration:.3f}s, 检测到 {num_obj} 个物体")
        else:
            rospy.logerr(f"  失败: {response.error_message}")

        # 短暂延迟
        if i < 2:
            rospy.sleep(0.5)

    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("性能对比")
    rospy.loginfo("="*60)
    rospy.loginfo(f"第 1 次调用: {times[0]:.3f}s")
    rospy.loginfo(f"第 2 次调用: {times[1]:.3f}s")
    rospy.loginfo(f"第 3 次调用: {times[2]:.3f}s")

    avg_后续 = (times[1] + times[2]) / 2
    冷启动开销 = times[0] - avg_后续

    rospy.loginfo(f"\n后续平均: {avg_后续:.3f}s")
    rospy.loginfo(f"冷启动开销: {冷启动开销*1000:.0f}ms ({冷启动开销/times[0]*100:.1f}%)")

    if 冷启动开销 < 0.05:
        rospy.loginfo("\n✓ 预热生效！首次调用无明显冷启动开销")
    else:
        rospy.logwarn(f"\n⚠ 首次调用存在冷启动开销: {冷启动开销*1000:.0f}ms")

if __name__ == '__main__':
    try:
        test_warmup()
    except Exception as e:
        rospy.logerr(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
