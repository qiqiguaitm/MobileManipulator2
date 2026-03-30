#!/usr/bin/env python3
"""综合监控: HDL 定位稳定性 + 静止目标 map 坐标漂移.

监控项:
  1. TF map→base_link 的帧间跳变 vs odom 轮式速度 (区分正常运动 / 定位跳变)
  2. 融合检测结果中目标的 map 坐标稳定性 (同 track_id 连续帧的坐标变化)

定位跳变判定:
  tf_vel  = TF帧间位移 / dt  (map→base_link 推算速度)
  odom_vel = /odom/fused twist  (轮式里程计速度, 独立于定位)
  vel_diff = |tf_vel - odom_vel| > 阈值 → 判定为定位跳变

数据保存:
  - JSONL 逐条记录: scripts/stability_log_<timestamp>.jsonl
  - 退出时打印最终统计 + 写入同目录 summary

Usage:
    export PATH="/usr/bin:$PATH"
    python3 scripts/_cc_stability_monitor.py
"""
import json
import math
import os
import time
import statistics
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
import tf2_ros
from rclpy.duration import Duration
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from perception.msg import Object3DArray

LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定位跳变阈值: TF推算速度与odom速度差 > 此值(m/s) 判定为跳变
VEL_DIFF_THRESHOLD = 0.15  # 150mm/s


class StabilityMonitor(Node):
    def __init__(self):
        super().__init__('stability_monitor')

        # ── 日志文件 ──
        ts_str = time.strftime('%Y%m%d_%H%M%S')
        self._log_path = os.path.join(SCRIPT_DIR, 'stability_log_{}.jsonl'.format(ts_str))
        self._log_file = open(self._log_path, 'w')
        self.get_logger().info('日志保存到: {}'.format(self._log_path))

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._loc_ready = False

        self.create_subscription(Bool, '/localization_status', self._loc_cb, 5)

        # ── 里程计速度 (独立参照) ──
        self._odom_linear = 0.0   # m/s, 最新 odom linear speed
        self._odom_angular = 0.0  # rad/s, 最新 odom angular speed
        self._odom_stamp = 0.0    # 最新 odom 时间戳
        self.create_subscription(
            Odometry, '/odom/fused', self._odom_cb, SENSOR_QOS)

        # ── 定位监控 ──
        self._prev_pose = None  # (x, y, z, yaw, t)
        self._map_to_base = None  # (tx, ty, tz, yaw) 缓存, 用于 map→base_link 反变换
        self._loc_records = []  # [{pos_jump, yaw_jump, tf_vel, odom_vel, vel_diff, dt, is_glitch}, ...]

        # ── IR-RGB 时间差监控 ──
        # 缓存每个相机的最新 IR/RGB 传感器时间戳 (header.stamp)
        self._ir_stamp = {}    # {cam: float} 最新 IR1 时间戳 (秒)
        self._rgb_stamp = {}   # {cam: float} 最新 RGB 时间戳 (秒)
        self._ir_rgb_gaps = {'top': [], 'chassis': []}  # 记录每次 RGB 到达时的 gap

        for cam in ['top', 'chassis']:
            self.create_subscription(
                Image, '/camera/{}/infra1/image_rect_raw'.format(cam),
                lambda msg, c=cam: self._ir_stamp_cb(c, msg),
                SENSOR_QOS)
            self.create_subscription(
                Image, '/camera/{}/color/image_raw'.format(cam),
                lambda msg, c=cam: self._rgb_stamp_cb(c, msg),
                SENSOR_QOS)

        # ── 目标坐标监控 ──
        self._obj_history = {}   # {key: [(x, y, z, t), ...]}
        self._obj_drifts = {}    # {key: [drift_mm, ...]}

        for topic in [
            '/multi_camera_perception/top/objects_3d',
            '/multi_camera_perception/chassis/objects_3d',
            '/multi_camera_perception/fused/objects_3d',
        ]:
            self.create_subscription(
                Object3DArray, topic,
                self._objects_cb, LATCHED_QOS)

        self.create_timer(0.1, self._tf_sample)
        self.create_timer(5.0, self._print_summary)

        self.get_logger().info('等待定位就绪...')

    def _log(self, record: dict):
        """写一条 JSONL 记录"""
        record['ts'] = time.time()
        self._log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._log_file.flush()

    def _loc_cb(self, msg):
        if not self._loc_ready and msg.data:
            self.get_logger().info('定位就绪!')
        self._loc_ready = msg.data

    def _ir_stamp_cb(self, cam: str, msg: Image):
        """记录最新 IR1 传感器时间戳 + 独立到达日志"""
        sensor_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._ir_stamp[cam] = sensor_ts
        # 独立记录 IR 到达, 用于精确测量 IR-RGB 延迟 δ
        self._log({
            'type': 'ir_arrive',
            'cam': cam,
            'sensor_ts': round(sensor_ts, 4),
        })

    def _rgb_stamp_cb(self, cam: str, msg: Image):
        """每次 RGB 到达时，计算 IR-RGB gap 并记录"""
        rgb_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._rgb_stamp[cam] = rgb_ts
        # 独立记录 RGB 到达
        self._log({
            'type': 'rgb_arrive',
            'cam': cam,
            'sensor_ts': round(rgb_ts, 4),
        })
        ir_ts = self._ir_stamp.get(cam)
        if ir_ts is None:
            return
        # gap > 0: IR 比 RGB 旧; gap < 0: IR 比 RGB 新
        gap_ms = (rgb_ts - ir_ts) * 1000
        self._ir_rgb_gaps[cam].append(gap_ms)

        # 估算像素偏移: fx * v * gap / depth
        # top: fx≈910, chassis: fx≈648; 简化取 800
        est_px_offset = 800 * self._odom_linear * abs(gap_ms) / 1000.0 / 1.5  # 假设 1.5m

        self._log({
            'type': 'ir_rgb_gap',
            'cam': cam,
            'gap_ms': round(gap_ms, 2),
            'ir_ts': round(ir_ts, 4),
            'rgb_ts': round(rgb_ts, 4),
            'odom_lin': round(self._odom_linear, 4),
            'odom_ang': round(self._odom_angular, 4),
            'est_px_offset': round(est_px_offset, 1),
        })

    def _odom_cb(self, msg: Odometry):
        """缓存最新轮式里程计速度"""
        self._odom_linear = math.sqrt(
            msg.twist.twist.linear.x ** 2 +
            msg.twist.twist.linear.y ** 2)
        self._odom_angular = abs(msg.twist.twist.angular.z)
        self._odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self._log({
            'type': 'odom',
            'linear': round(self._odom_linear, 4),
            'angular': round(self._odom_angular, 4),
        })

    def _tf_sample(self):
        if not self._loc_ready:
            return
        try:
            ts = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=0.02))
        except Exception:
            return

        t = ts.transform.translation
        q = ts.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        now = time.time()

        # 记录每个 TF 采样 (含当前 odom 速度)
        self._log({
            'type': 'tf',
            'x': round(t.x, 4), 'y': round(t.y, 4), 'z': round(t.z, 4),
            'yaw': round(math.degrees(yaw), 2),
            'odom_lin': round(self._odom_linear, 4),
            'odom_ang': round(self._odom_angular, 4),
        })

        if self._prev_pose is not None:
            px, py, pz, pyaw, pt = self._prev_pose
            dt = now - pt
            if 0.01 < dt < 0.5:
                # TF 推算速度
                dx_m = t.x - px
                dy_m = t.y - py
                dz_m = t.z - pz
                pos_jump_mm = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2) * 1000
                tf_vel = pos_jump_mm / 1000.0 / dt  # m/s

                yaw_jump = abs(math.degrees(yaw - pyaw))
                if yaw_jump > 180:
                    yaw_jump = 360 - yaw_jump
                tf_ang_vel = math.radians(yaw_jump) / dt  # rad/s

                # odom 速度 (独立参照)
                odom_vel = self._odom_linear
                odom_ang = self._odom_angular

                # 速度差异
                vel_diff = abs(tf_vel - odom_vel)
                ang_diff = abs(tf_ang_vel - odom_ang)

                # 判定: TF推算速度远超odom → 定位跳变
                is_glitch = vel_diff > VEL_DIFF_THRESHOLD

                rec = {
                    'pos_jump_mm': pos_jump_mm,
                    'yaw_jump_deg': yaw_jump,
                    'tf_vel': tf_vel,
                    'tf_ang_vel': tf_ang_vel,
                    'odom_vel': odom_vel,
                    'odom_ang': odom_ang,
                    'vel_diff': vel_diff,
                    'ang_diff': ang_diff,
                    'dt': dt,
                    'is_glitch': is_glitch,
                }
                self._loc_records.append(rec)

                self._log({
                    'type': 'loc_check',
                    'pos_jump_mm': round(pos_jump_mm, 1),
                    'yaw_jump_deg': round(yaw_jump, 3),
                    'tf_vel': round(tf_vel, 4),
                    'tf_ang_vel': round(tf_ang_vel, 4),
                    'odom_vel': round(odom_vel, 4),
                    'odom_ang': round(odom_ang, 4),
                    'vel_diff': round(vel_diff, 4),
                    'ang_diff': round(ang_diff, 4),
                    'is_glitch': is_glitch,
                })

                if is_glitch:
                    self.get_logger().warn(
                        '[定位跳变!] pos={:.0f}mm tf_vel={:.3f} odom_vel={:.3f} '
                        'diff={:.3f}m/s | yaw={:.1f}deg'.format(
                            pos_jump_mm, tf_vel, odom_vel, vel_diff, yaw_jump))

        self._prev_pose = (t.x, t.y, t.z, yaw, now)
        self._map_to_base = (t.x, t.y, t.z, yaw)

    def _objects_cb(self, msg: Object3DArray):
        if msg.header.frame_id != 'map':
            return

        now = time.time()
        for obj in msg.objects:
            oid = obj.object_id
            x, y, z = obj.position.x, obj.position.y, obj.position.z
            cat = obj.category
            src = obj.source_camera

            key = '{}|{}|{}'.format(cat, oid, src)

            # ── base_link 坐标 (map 反变换, 无 HDL 噪声传播) ──
            bl_x, bl_y, bl_z = x, y, z  # fallback: 用 map 坐标
            if self._map_to_base is not None:
                tx, ty, tz, yaw_r = self._map_to_base
                dx, dy = x - tx, y - ty
                cos_y = math.cos(-yaw_r)
                sin_y = math.sin(-yaw_r)
                bl_x = cos_y * dx - sin_y * dy
                bl_y = sin_y * dx + cos_y * dy
                bl_z = z - tz

            # 相机光学系位置 (无 TF 变换, 纯检测输出)
            po = obj.position_optical
            # 2D bbox → 中心 + 尺寸
            bb = obj.bbox  # [x1, y1, x2, y2]
            bbox_cx = (bb[0] + bb[2]) / 2.0
            bbox_cy = (bb[1] + bb[3]) / 2.0
            bbox_w = bb[2] - bb[0]
            bbox_h = bb[3] - bb[1]

            self._log({
                'type': 'obj',
                'key': key, 'cat': cat, 'src': src,
                # map 系坐标 (含 HDL TF)
                'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4),
                # base_link 系坐标 (map 反变换)
                'bl_x': round(bl_x, 4), 'bl_y': round(bl_y, 4), 'bl_z': round(bl_z, 4),
                'score': round(obj.score, 3),
                'conf': round(obj.confidence, 3),
                'dist': round(obj.distance, 3),
                # 相机光学系坐标 (纯检测, 无任何变换)
                'depth': round(obj.depth, 4),
                'opt_x': round(po.x, 4), 'opt_y': round(po.y, 4), 'opt_z': round(po.z, 4),
                # 2D bbox
                'bcx': round(bbox_cx, 1), 'bcy': round(bbox_cy, 1),
                'bw': round(bbox_w, 1), 'bh': round(bbox_h, 1),
                # 机器人运动状态
                'odom_lin': round(self._odom_linear, 4),
                'odom_ang': round(self._odom_angular, 4),
            })

            if key not in self._obj_history:
                self._obj_history[key] = []
                self._obj_drifts[key] = []

            history = self._obj_history[key]

            if history:
                px, py, pz, pt = history[-1]
                dt = now - pt
                if dt < 5.0:
                    drift = math.sqrt(
                        (x - px)**2 + (y - py)**2 + (z - pz)**2) * 1000
                    self._obj_drifts[key].append(drift)

                    self._log({
                        'type': 'drift',
                        'key': key,
                        'drift_mm': round(drift, 1),
                        'dt': round(dt, 3),
                        'odom_lin': round(self._odom_linear, 4),
                        'odom_ang': round(self._odom_angular, 4),
                    })

                    if drift > 30:
                        self.get_logger().warn(
                            '[{}] 坐标跳变 {:.0f}mm! '
                            '({:.3f},{:.3f},{:.3f}) -> ({:.3f},{:.3f},{:.3f}) '
                            'dt={:.2f}s'.format(
                                key, drift, px, py, pz, x, y, z, dt))

            history.append((x, y, z, now))
            if len(history) > 100:
                history.pop(0)

    def _print_summary(self):
        if not self._loc_ready:
            self.get_logger().info('等待定位就绪...')
            return

        self.get_logger().info('─' * 60)

        if len(self._loc_records) > 2:
            n = len(self._loc_records)
            glitches = [r for r in self._loc_records if r['is_glitch']]
            n_glitch = len(glitches)

            # 静止时 (odom_vel < 0.02 m/s) 的TF抖动
            static = [r for r in self._loc_records if r['odom_vel'] < 0.02]
            # 运动时
            moving = [r for r in self._loc_records if r['odom_vel'] >= 0.02]

            summary_parts = ['[定位] n={} glitch={}'.format(n, n_glitch)]

            if static:
                sp = [r['pos_jump_mm'] for r in static]
                summary_parts.append(
                    '静止({}) pos: P50={:.1f} max={:.1f}mm'.format(
                        len(static), statistics.median(sp), max(sp)))

            if moving:
                vd = [r['vel_diff'] * 1000 for r in moving]  # mm/s
                summary_parts.append(
                    '运动({}) vel_diff: P50={:.0f} P95={:.0f} max={:.0f}mm/s'.format(
                        len(moving),
                        statistics.median(vd),
                        sorted(vd)[int(len(vd) * 0.95)],
                        max(vd)))

            # odom 速度范围
            odom_vels = [r['odom_vel'] for r in self._loc_records]
            summary_parts.append(
                'odom_vel: [{:.3f}, {:.3f}]m/s'.format(
                    min(odom_vels), max(odom_vels)))

            self.get_logger().info(' | '.join(summary_parts))

            if glitches:
                worst = max(glitches, key=lambda r: r['vel_diff'])
                self.get_logger().warn(
                    '  最大跳变: pos={:.0f}mm tf_vel={:.3f} odom_vel={:.3f} '
                    'vel_diff={:.3f}m/s'.format(
                        worst['pos_jump_mm'], worst['tf_vel'],
                        worst['odom_vel'], worst['vel_diff']))
        else:
            self.get_logger().info('[定位] 数据不足')

        # ── IR-RGB gap 统计 ──
        for cam in ['top', 'chassis']:
            gaps = self._ir_rgb_gaps[cam]
            if len(gaps) >= 3:
                abs_gaps = [abs(g) for g in gaps]
                self.get_logger().info(
                    '[IR-RGB gap {}] n={} | P50={:.1f} P95={:.1f} max={:.1f}ms'.format(
                        cam, len(gaps),
                        statistics.median(abs_gaps),
                        sorted(abs_gaps)[int(len(abs_gaps) * 0.95)],
                        max(abs_gaps)))

        if not self._obj_drifts:
            self.get_logger().info('[目标] 无检测数据')
            return

        for key, drifts in sorted(self._obj_drifts.items()):
            if len(drifts) < 3:
                continue
            history = self._obj_history[key]
            if len(history) >= 2:
                x0, y0, z0, _ = history[0]
                xn, yn, zn, _ = history[-1]
                total_span = math.sqrt(
                    (xn-x0)**2 + (yn-y0)**2 + (zn-z0)**2) * 1000
            else:
                total_span = 0

            self.get_logger().info(
                '[目标] {} | n={} | 帧间漂移: P50={:.1f} P95={:.1f} max={:.1f}mm | '
                '总跨度={:.1f}mm'.format(
                    key, len(drifts),
                    statistics.median(drifts),
                    sorted(drifts)[int(len(drifts) * 0.95)],
                    max(drifts),
                    total_span))

    def _write_final_summary(self):
        """退出时写入最终统计"""
        summary_path = self._log_path.replace('.jsonl', '_summary.txt')
        lines = []
        lines.append('=' * 70)
        lines.append('稳定性监控最终统计')
        lines.append('日志文件: {}'.format(self._log_path))
        lines.append('=' * 70)

        if len(self._loc_records) > 2:
            n = len(self._loc_records)
            glitches = [r for r in self._loc_records if r['is_glitch']]
            static = [r for r in self._loc_records if r['odom_vel'] < 0.02]
            moving = [r for r in self._loc_records if r['odom_vel'] >= 0.02]

            lines.append('')
            lines.append('[定位稳定性] ({} samples, {} glitches)'.format(n, len(glitches)))

            odom_vels = [r['odom_vel'] for r in self._loc_records]
            lines.append('  odom速度范围: [{:.3f}, {:.3f}] m/s'.format(
                min(odom_vels), max(odom_vels)))

            if static:
                sp = [r['pos_jump_mm'] for r in static]
                sy = [r['yaw_jump_deg'] for r in static]
                lines.append('')
                lines.append('  [静止] ({} samples, odom_vel<0.02m/s)'.format(len(static)))
                lines.append('    pos 帧间跳变: P50={:.1f} P95={:.1f} max={:.1f} mm'.format(
                    statistics.median(sp),
                    sorted(sp)[int(len(sp) * 0.95)],
                    max(sp)))
                lines.append('    yaw 帧间跳变: P50={:.2f} P95={:.2f} max={:.2f} deg'.format(
                    statistics.median(sy),
                    sorted(sy)[int(len(sy) * 0.95)],
                    max(sy)))

            if moving:
                vd = [r['vel_diff'] * 1000 for r in moving]
                mp = [r['pos_jump_mm'] for r in moving]
                mv = [r['tf_vel'] for r in moving]
                ov = [r['odom_vel'] for r in moving]
                lines.append('')
                lines.append('  [运动] ({} samples, odom_vel>=0.02m/s)'.format(len(moving)))
                lines.append('    odom_vel: P50={:.3f} max={:.3f} m/s'.format(
                    statistics.median(ov), max(ov)))
                lines.append('    tf_vel:   P50={:.3f} max={:.3f} m/s'.format(
                    statistics.median(mv), max(mv)))
                lines.append('    vel_diff (tf-odom): P50={:.0f} P95={:.0f} max={:.0f} mm/s'.format(
                    statistics.median(vd),
                    sorted(vd)[int(len(vd) * 0.95)],
                    max(vd)))

                # 运动中的跳变
                moving_glitches = [r for r in moving if r['is_glitch']]
                lines.append('    运动中跳变: {}/{} ({:.1f}%)'.format(
                    len(moving_glitches), len(moving),
                    len(moving_glitches) / len(moving) * 100 if moving else 0))

            if glitches:
                lines.append('')
                lines.append('  [跳变详情] (vel_diff > {}mm/s)'.format(
                    int(VEL_DIFF_THRESHOLD * 1000)))
                # 按 vel_diff 降序列出前 20 个
                for r in sorted(glitches, key=lambda x: -x['vel_diff'])[:20]:
                    lines.append(
                        '    pos={:.0f}mm tf_vel={:.3f} odom_vel={:.3f} '
                        'vel_diff={:.3f}m/s yaw={:.1f}deg'.format(
                            r['pos_jump_mm'], r['tf_vel'],
                            r['odom_vel'], r['vel_diff'],
                            r['yaw_jump_deg']))

        lines.append('')
        lines.append('[目标 map 坐标漂移]')
        for key, drifts in sorted(self._obj_drifts.items()):
            if len(drifts) < 3:
                continue
            history = self._obj_history[key]
            if len(history) >= 2:
                x0, y0, z0, _ = history[0]
                xn, yn, zn, _ = history[-1]
                total_span = math.sqrt(
                    (xn-x0)**2 + (yn-y0)**2 + (zn-z0)**2) * 1000
            else:
                total_span = 0
            lines.append('  {} (n={})'.format(key, len(drifts)))
            lines.append('    帧间: P50={:.1f} P95={:.1f} max={:.1f} mm'.format(
                statistics.median(drifts),
                sorted(drifts)[int(len(drifts) * 0.95)],
                max(drifts)))
            lines.append('    总跨度: {:.1f} mm'.format(total_span))

        lines.append('')
        lines.append('=' * 70)

        text = '\n'.join(lines)
        with open(summary_path, 'w') as f:
            f.write(text)
        self.get_logger().info('最终统计已保存: {}'.format(summary_path))
        print(text)


def main():
    rclpy.init()
    node = StabilityMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._write_final_summary()
        node._log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
