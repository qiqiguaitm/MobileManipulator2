#!/usr/bin/env python3
"""监控 TF buffer 状态：latest vs 图像时间戳查询对比。"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
import tf2_ros
import time
from sensor_msgs.msg import Image


class TFBufferMonitor(Node):
    def __init__(self):
        super().__init__('tf_buffer_monitor')
        self._buf = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buf, self)

        # 记录最新图像时间戳
        self._last_top_stamp = None
        self._last_chassis_stamp = None
        self.create_subscription(
            Image, '/camera/top/color/image_raw',
            lambda m: setattr(self, '_last_top_stamp', m.header.stamp), 1)
        self.create_subscription(
            Image, '/camera/chassis/color/image_raw',
            lambda m: setattr(self, '_last_chassis_stamp', m.header.stamp), 1)

        self.create_timer(1.0, self._check)
        self.get_logger().info('TF Buffer Monitor 启动，每秒检查一次')

    def _check(self):
        now_wall = time.time()
        lines = [f'\n=== TF Buffer Check  wall={now_wall:.3f} ===']

        # 1. 查 latest
        try:
            ts_latest = self._buf.lookup_transform(
                'map', 'base_link', Time(), timeout=Duration(seconds=0.1))
            t_latest = ts_latest.header.stamp.sec + ts_latest.header.stamp.nanosec * 1e-9
            age_latest = now_wall - t_latest
            pos = ts_latest.transform.translation
            lines.append(
                f'  LATEST:  t={t_latest:.3f}  age={age_latest:.3f}s  '
                f'pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')
        except Exception as e:
            lines.append(f'  LATEST:  FAILED — {e}')
            t_latest = None

        # 2. 用图像时间戳查
        for cam, stamp in [('top', self._last_top_stamp),
                           ('chassis', self._last_chassis_stamp)]:
            if stamp is None:
                lines.append(f'  {cam:8s} 图像时间戳: 无数据')
                continue
            img_t = stamp.sec + stamp.nanosec * 1e-9
            img_age = now_wall - img_t
            try:
                query_time = Time(seconds=stamp.sec, nanoseconds=stamp.nanosec)
                ts_img = self._buf.lookup_transform(
                    'map', 'base_link', query_time, timeout=Duration(seconds=0.1))
                t_result = ts_img.header.stamp.sec + ts_img.header.stamp.nanosec * 1e-9
                pos = ts_img.transform.translation

                # 与 latest 的位置差
                if t_latest is not None:
                    dx = pos.x - ts_latest.transform.translation.x
                    dy = pos.y - ts_latest.transform.translation.y
                    import math
                    delta = math.sqrt(dx*dx + dy*dy) * 1000  # mm
                    delta_str = f'  Δ={delta:.1f}mm'
                else:
                    delta_str = ''

                lines.append(
                    f'  {cam:8s} img_t={img_t:.3f} (age={img_age:.3f}s)  '
                    f'tf_t={t_result:.3f}  '
                    f'pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})'
                    f'{delta_str}')
            except Exception as e:
                lines.append(
                    f'  {cam:8s} img_t={img_t:.3f} (age={img_age:.3f}s)  '
                    f'FAILED — {e}')

        self.get_logger().info('\n'.join(lines))


def main():
    rclpy.init()
    node = TFBufferMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
