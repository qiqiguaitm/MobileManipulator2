#!/usr/bin/env python3
"""诊断: 对比两种 TF 查询方式的位姿差异.

对同一帧图像，同时用两种方式查 TF:
  A) img_time: lookup_transform(image_stamp)  — 修复后的方式
  B) Time(0):  lookup_transform(Time())       — 修复前的方式

关键指标:
  pos_diff   : 两种方式查到的位姿的位移差 (mm)
  yaw_diff   : 偏航角差 (deg)
  fallback   : img_time 查询失败次数

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/_cc_tf_image_gap.py
"""
import math
import time
import statistics
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
import tf2_ros
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def tf_to_xy_yaw(tf_stamped):
    """从 TransformStamped 提取 (x, y, z, yaw)"""
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation
    # yaw from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return t.x, t.y, t.z, yaw


class TfImageGapNode(Node):
    def __init__(self):
        super().__init__('tf_image_gap_diag')

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._loc_ready = False

        self.create_subscription(Bool, '/localization_status', self._loc_cb, 5)

        self._latest_img = {'top': None, 'chassis': None}

        for cam in ['top', 'chassis']:
            topic = f'/camera/{cam}/color/image_raw'
            self.create_subscription(
                Image, topic,
                lambda msg, c=cam: self._cache_image(c, msg),
                SENSOR_QOS,
            )

        self._diffs = {'top': [], 'chassis': []}
        self._fallback_counts = {'top': 0, 'chassis': 0}
        self._count = 0
        self._max_samples = 300

        self.create_timer(0.2, self._timer_cb)
        self.create_timer(5.0, self._print_stats)
        self.get_logger().info('等待定位就绪和相机图像...')

    def _loc_cb(self, msg):
        if not self._loc_ready and msg.data:
            self.get_logger().info('定位就绪!')
        self._loc_ready = msg.data

    def _cache_image(self, cam: str, msg: Image):
        self._latest_img[cam] = msg

    def _timer_cb(self):
        if not self._loc_ready:
            return

        for cam in ['top', 'chassis']:
            msg = self._latest_img[cam]
            if msg is None:
                continue

            t_now = time.time()
            t_img = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            img_age = (t_now - t_img) * 1000  # ms

            # 方法 B: Time(0) — 修复前
            try:
                tf_latest = self._tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time(),
                    timeout=Duration(seconds=0.05))
            except Exception:
                continue

            t_tf_latest = (tf_latest.header.stamp.sec +
                           tf_latest.header.stamp.nanosec * 1e-9)
            gap_old = (t_tf_latest - t_img) * 1000  # ms, 修复前的 img→TF gap

            # 方法 A: image_time — 修复后
            fallback = False
            try:
                query_time = rclpy.time.Time.from_msg(msg.header.stamp)
                tf_img = self._tf_buffer.lookup_transform(
                    'map', 'base_link', query_time,
                    timeout=Duration(seconds=0))
            except Exception:
                tf_img = tf_latest  # 回退
                fallback = True
                self._fallback_counts[cam] += 1

            # 计算两种位姿的差异
            x_a, y_a, z_a, yaw_a = tf_to_xy_yaw(tf_img)
            x_b, y_b, z_b, yaw_b = tf_to_xy_yaw(tf_latest)

            pos_diff = math.sqrt((x_a - x_b)**2 + (y_a - y_b)**2 + (z_a - z_b)**2) * 1000  # mm
            yaw_diff = math.degrees(abs(yaw_a - yaw_b))
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff

            self._diffs[cam].append({
                'pos_diff': pos_diff,
                'yaw_diff': yaw_diff,
                'gap_old': gap_old,
                'img_age': img_age,
                'fallback': fallback,
            })
            self._count += 1

            n = len(self._diffs[cam])
            if n <= 3 or n % 30 == 0:
                fb_str = ' [FALLBACK]' if fallback else ''
                self.get_logger().info(
                    '[{}] pos_diff={:.1f}mm yaw_diff={:.2f}deg | '
                    'old_gap={:+.0f}ms img_age={:.0f}ms{}'.format(
                        cam, pos_diff, yaw_diff, gap_old, img_age, fb_str))

        if self._count >= self._max_samples:
            self._print_final()
            raise SystemExit(0)

    def _print_stats(self):
        if not self._loc_ready:
            self.get_logger().info('等待定位就绪...')
            return
        for cam in ['top', 'chassis']:
            data = self._diffs[cam]
            if len(data) < 2:
                continue
            pos = [d['pos_diff'] for d in data]
            fb = self._fallback_counts[cam]
            self.get_logger().info(
                '[{}] n={} | pos_diff: P50={:.1f} mean={:.1f} max={:.1f}mm | '
                'fallback: {}/{}'.format(
                    cam, len(data),
                    statistics.median(pos), statistics.mean(pos), max(pos),
                    fb, len(data)))

    def _print_final(self):
        self.get_logger().info('=' * 70)
        self.get_logger().info('最终统计: img_time查TF vs Time(0)查TF 位姿差异')
        self.get_logger().info('  (差异越大 = 修复前误差越大)')
        self.get_logger().info('=' * 70)
        for cam in ['top', 'chassis']:
            data = self._diffs[cam]
            if len(data) < 2:
                continue
            fb = self._fallback_counts[cam]
            n_ok = len(data) - fb
            self.get_logger().info(
                '--- {} ({} samples, img_time成功={}, fallback={}) ---'.format(
                    cam, len(data), n_ok, fb))

            # 只统计非 fallback 的样本（fallback 时两种方法一样，diff=0）
            data_ok = [d for d in data if not d['fallback']]
            if len(data_ok) < 2:
                self.get_logger().info('  (img_time 成功样本不足，跳过)')
                continue

            for label, key, unit in [
                ('pos_diff', 'pos_diff', 'mm'),
                ('yaw_diff', 'yaw_diff', 'deg'),
                ('old_gap ', 'gap_old', 'ms'),
                ('img_age ', 'img_age', 'ms'),
            ]:
                vals = [d[key] for d in data_ok]
                vals_sorted = sorted(vals)
                p5 = vals_sorted[int(len(vals) * 0.05)]
                p95 = vals_sorted[int(len(vals) * 0.95)]
                self.get_logger().info(
                    '  {}: P50={:.1f} mean={:.1f} std={:.1f} '
                    'P5={:.1f} P95={:.1f} [{:.1f}, {:.1f}] {}'.format(
                        label,
                        statistics.median(vals), statistics.mean(vals),
                        statistics.stdev(vals),
                        p5, p95, min(vals), max(vals), unit))

        self.get_logger().info('=' * 70)
        self.get_logger().info(
            '说明:\n'
            '  pos_diff = 两种方式查到的位姿的3D距离差 (mm)\n'
            '  yaw_diff = 偏航角差 (deg)\n'
            '  old_gap  = 修复前 img->TF 时间差 (ms), Time(0)的TF时间戳 - 图像时间\n'
            '  img_age  = 图像年龄 (ms), 当前时间 - 图像时间\n'
            '  fallback的样本 pos_diff=0 (两种方法一致), 已从统计中排除\n'
            '  静止时 pos_diff≈0; 移动中 pos_diff 反映修复前的位姿误差大小')


def main():
    rclpy.init()
    node = TfImageGapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
