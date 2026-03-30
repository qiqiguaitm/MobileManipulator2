#!/usr/bin/env python3
"""
Ghost Target 调试脚本 — 用图片时间戳插值 TF，分析幽灵目标成因。

核心思路：感知节点用 image_stamp 查 TF 插值做 base_link→map 变换。
如果两个相机的 image_stamp 差异 + 机器人运动 → 同一物体投影到不同 map 位置 → ghost。

订阅:
  - /multi_camera_perception/fused/objects_3d  (ByteTracker3D 输出, map frame)
  - /multi_camera_perception/top/objects_3d    (Top 单相机, map frame)
  - /multi_camera_perception/chassis/objects_3d (Chassis 单相机, map frame)
  - /camera/top/color/image_raw
  - /camera/chassis/color/image_raw
  - TF: map → base_link

输出:
  - scripts/ghost_debug_<ts>/log.jsonl    每帧: 图片时间戳 + TF@每个时间戳 + 检测位置
  - scripts/ghost_debug_<ts>/img_*.jpg    标注图（1Hz）
  - 终端: 近距离重复检测 + TF 差异

用法:
  export PATH="/usr/bin:$PATH" && python3 scripts/_cc_ghost_target_debug.py [--duration 60]
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time as RosTime
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from perception.msg import Object3DArray
from tf2_ros import Buffer, TransformListener


# ---------- config ----------
SAVE_IMAGE_INTERVAL = 1.0   # s
GHOST_DIST_THRESH = 0.30    # m, 同帧两检测 < 此距离 → 疑似 ghost


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_ros_time(sec):
    """float seconds → rclpy.time.Time"""
    s = int(sec)
    ns = int((sec - s) * 1e9)
    return RosTime(seconds=s, nanoseconds=ns)


def obj_to_dict(obj):
    return {
        'id': obj.object_id,
        'cat': obj.category,
        'score': round(obj.score, 3),
        'pos': [round(obj.position.x, 4), round(obj.position.y, 4), round(obj.position.z, 4)],
        'depth': round(obj.depth, 3),
        'dist': round(obj.distance, 3),
        'size': round(obj.physical_size, 4),
        'bbox': [round(b, 1) for b in obj.bbox],
        'src': obj.source_camera,
    }


def dist_2d(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


def pose_delta(p1, p2):
    """两个 pose dict 之间的 2D 距离和 yaw 差"""
    if not p1 or not p2:
        return None
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    dyaw = p1['yaw'] - p2['yaw']
    return {'dist_m': round(math.sqrt(dx*dx + dy*dy), 4),
            'dyaw_deg': round(dyaw, 2)}


class GhostDebugNode(Node):
    def __init__(self, output_dir: str, duration: float):
        super().__init__('ghost_target_debug')
        self.output_dir = output_dir
        self.duration = duration
        self.start_time = time.time()
        self.frame_count = 0
        self.ghost_events = []

        self.log_path = os.path.join(output_dir, 'log.jsonl')
        self.log_file = open(self.log_path, 'w')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Latest data + stamps
        self._img_top = None
        self._img_top_stamp = 0.0
        self._img_chassis = None
        self._img_chassis_stamp = 0.0
        self._last_img_save = 0.0

        self._det_top = []
        self._det_top_stamp = 0.0
        self._det_chassis = []
        self._det_chassis_stamp = 0.0

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            Object3DArray, '/multi_camera_perception/fused/objects_3d',
            self._cb_fused, 10)
        self.create_subscription(
            Object3DArray, '/multi_camera_perception/top/objects_3d',
            self._cb_top_det, 10)
        self.create_subscription(
            Object3DArray, '/multi_camera_perception/chassis/objects_3d',
            self._cb_chassis_det, 10)
        self.create_subscription(
            Image, '/camera/top/color/image_raw',
            self._cb_img_top, img_qos)
        self.create_subscription(
            Image, '/camera/chassis/color/image_raw',
            self._cb_img_chassis, img_qos)

        self.create_timer(1.0, self._check_done)

        self.get_logger().info(f'Ghost debug: output={output_dir}, duration={duration}s')

    # ----------------------------------------------------------------
    # TF query — 模拟感知节点的查询方式
    # ----------------------------------------------------------------
    def _query_tf(self, stamp_sec):
        """用指定时间戳插值查 TF，失败回退 latest。返回 (pose_dict, mode)"""
        pose = None
        mode = 'interp'

        # 1) 尝试插值查询
        if stamp_sec > 0:
            try:
                query_time = sec_to_ros_time(stamp_sec)
                tf = self.tf_buffer.lookup_transform(
                    'map', 'base_link', query_time,
                    timeout=Duration(seconds=0.0))
                pose = self._tf_to_dict(tf, stamp_sec)
            except Exception:
                mode = 'fallback_latest'

        # 2) 回退 latest
        if pose is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', 'base_link', RosTime(),
                    timeout=Duration(seconds=0.1))
                pose = self._tf_to_dict(tf, stamp_sec)
                if mode != 'fallback_latest':
                    mode = 'latest'
            except Exception:
                return None, 'fail'

        return pose, mode

    def _tf_to_dict(self, tf, query_stamp):
        t = tf.transform.translation
        q = tf.transform.rotation
        tf_stamp = stamp_to_sec(tf.header.stamp)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.degrees(math.atan2(siny, cosy))
        return {
            'x': round(t.x, 4), 'y': round(t.y, 4), 'z': round(t.z, 4),
            'yaw': round(yaw, 2),
            'tf_stamp': round(tf_stamp, 6),
            'query_stamp': round(query_stamp, 6),
            'tf_age_ms': round((query_stamp - tf_stamp) * 1000, 1) if query_stamp > 0 else 0,
        }

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------
    def _cb_img_top(self, msg):
        self._img_top = msg
        self._img_top_stamp = stamp_to_sec(msg.header.stamp)

    def _cb_img_chassis(self, msg):
        self._img_chassis = msg
        self._img_chassis_stamp = stamp_to_sec(msg.header.stamp)

    def _cb_top_det(self, msg):
        self._det_top = [obj_to_dict(o) for o in msg.objects]
        self._det_top_stamp = stamp_to_sec(msg.header.stamp)

    def _cb_chassis_det(self, msg):
        self._det_chassis = [obj_to_dict(o) for o in msg.objects]
        self._det_chassis_stamp = stamp_to_sec(msg.header.stamp)

    def _cb_fused(self, msg):
        now = time.time()
        self.frame_count += 1

        fused_objs = [obj_to_dict(o) for o in msg.objects]
        fused_stamp = stamp_to_sec(msg.header.stamp)

        # ---- 用各图片时间戳查 TF（模拟感知节点行为）----
        tf_top, tf_top_mode = self._query_tf(self._img_top_stamp)
        tf_chassis, tf_chassis_mode = self._query_tf(self._img_chassis_stamp)
        tf_latest, _ = self._query_tf(0)  # latest 作为参考

        # 两相机 TF 差异（核心 ghost 指标）
        tf_delta = pose_delta(tf_top, tf_chassis)

        # ---- Ghost 检测 ----
        ghosts = self._detect_ghosts(fused_objs)

        # ---- 时间戳 ----
        img_gap_ms = round((self._img_top_stamp - self._img_chassis_stamp) * 1000, 1) \
            if self._img_top_stamp > 0 and self._img_chassis_stamp > 0 else None

        stamps = {
            'fused': round(fused_stamp, 6),
            'img_top': round(self._img_top_stamp, 6),
            'img_chassis': round(self._img_chassis_stamp, 6),
            'img_gap_ms': img_gap_ms,
            'det_top': round(self._det_top_stamp, 6),
            'det_chassis': round(self._det_chassis_stamp, 6),
        }

        tf_info = {
            'tf@img_top': tf_top,
            'tf@img_top_mode': tf_top_mode,
            'tf@img_chassis': tf_chassis,
            'tf@img_chassis_mode': tf_chassis_mode,
            'tf@latest': tf_latest,
            'tf_delta_top_vs_chassis': tf_delta,
        }

        # ---- JSONL ----
        record = {
            't': round(now - self.start_time, 3),
            'frame': self.frame_count,
            'stamps': stamps,
            'tf': tf_info,
            'fused': fused_objs,
            'top': list(self._det_top),
            'chassis': list(self._det_chassis),
            'ghosts': ghosts,
        }
        self.log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.log_file.flush()

        # ---- 终端 ----
        fused_str = ' '.join(
            f"{o['id']}({o['cat']})@({o['pos'][0]:.2f},{o['pos'][1]:.2f})"
            for o in fused_objs)
        delta_str = (f"tf_delta={tf_delta['dist_m']*100:.1f}cm/{tf_delta['dyaw_deg']:.1f}°"
                     if tf_delta else "tf_delta=N/A")
        gap_str = f"img_gap={img_gap_ms:.0f}ms" if img_gap_ms is not None else ""
        self.get_logger().info(
            f'[{self.frame_count}] fused={len(fused_objs)} '
            f'top={len(self._det_top)} chassis={len(self._det_chassis)} '
            f'{gap_str} {delta_str} '
            f'tf_mode={tf_top_mode}/{tf_chassis_mode} | {fused_str}')

        if ghosts:
            for g in ghosts:
                self.ghost_events.append({**g, 'frame': self.frame_count, 'tf_delta': tf_delta})
                self.get_logger().warn(
                    f'  GHOST? {g["a"]["id"]} vs {g["b"]["id"]} '
                    f'map_dist={g["dist"]:.3f}m '
                    f'src={g["a"]["src"]}/{g["b"]["src"]} '
                    f'{delta_str}')

        # ---- 图片 ----
        if now - self._last_img_save >= SAVE_IMAGE_INTERVAL:
            self._save_images(fused_objs, tf_top, tf_chassis, tf_delta, img_gap_ms)
            self._last_img_save = now

    def _detect_ghosts(self, fused_objs):
        ghosts = []
        for i in range(len(fused_objs)):
            for j in range(i + 1, len(fused_objs)):
                a, b = fused_objs[i], fused_objs[j]
                d = dist_2d(a['pos'], b['pos'])
                if d < GHOST_DIST_THRESH:
                    ghosts.append({
                        'a': a, 'b': b,
                        'dist': round(d, 4),
                        'same_cat': a['cat'] == b['cat'],
                    })
        return ghosts

    def _save_images(self, fused_objs, tf_top, tf_chassis, tf_delta, img_gap_ms):
        try:
            import cv2
        except ImportError:
            return

        for name, img_msg, det_list in [
            ('top', self._img_top, self._det_top),
            ('chassis', self._img_chassis, self._det_chassis),
        ]:
            if img_msg is None:
                continue
            if img_msg.encoding not in ('rgb8', 'bgr8'):
                continue
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3)
            if img_msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.copy()

            # per-camera detections (green)
            for det in det_list:
                bbox = det['bbox']
                if len(bbox) >= 4 and bbox[2] > bbox[0]:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['id']} {det['cat']} s={det['score']:.2f} d={det['depth']:.2f}m"
                    cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # fused tracks (yellow)
            for fo in fused_objs:
                if name in fo.get('src', ''):
                    bbox = fo['bbox']
                    if len(bbox) >= 4 and bbox[2] > bbox[0]:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(img,
                            f"T:{fo['id']} map=({fo['pos'][0]:.2f},{fo['pos'][1]:.2f})",
                            (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # overlay: TF + timestamps
            tf_pose = tf_top if name == 'top' else tf_chassis
            img_stamp = self._img_top_stamp if name == 'top' else self._img_chassis_stamp
            y = 20
            if tf_pose:
                cv2.putText(img,
                    f"robot@img=({tf_pose['x']:.2f},{tf_pose['y']:.2f}) "
                    f"yaw={tf_pose['yaw']:.1f} tf_age={tf_pose['tf_age_ms']:.0f}ms",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y += 20
            if tf_delta:
                color = (0, 0, 255) if tf_delta['dist_m'] > 0.02 else (200, 200, 200)
                cv2.putText(img,
                    f"tf_delta(top-chassis)={tf_delta['dist_m']*100:.1f}cm "
                    f"dyaw={tf_delta['dyaw_deg']:.1f}deg "
                    f"img_gap={img_gap_ms:.0f}ms" if img_gap_ms is not None else "",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y += 18
            cv2.putText(img,
                f"img_stamp={img_stamp:.3f} frame={self.frame_count}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imwrite(
                os.path.join(self.output_dir, f"img_{name}_{self.frame_count:05d}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _check_done(self):
        if time.time() - self.start_time >= self.duration:
            self.get_logger().info(f'{self.duration}s reached, stopping...')
            self._print_summary()
            self.log_file.close()
            rclpy.shutdown()

    def _print_summary(self):
        print('\n' + '=' * 60)
        print('Ghost Target Debug Summary')
        print('=' * 60)
        print(f'Frames: {self.frame_count}')
        print(f'Ghost events: {len(self.ghost_events)}')
        print(f'Output: {self.output_dir}')

        if self.ghost_events:
            print('\n--- Ghost Events ---')
            pair_counts = defaultdict(list)
            for g in self.ghost_events:
                key = tuple(sorted([g['a']['id'], g['b']['id']]))
                pair_counts[key].append(g)
            for pair, events in sorted(pair_counts.items(), key=lambda x: -len(x[1])):
                e0 = events[0]
                tf_deltas = [e['tf_delta']['dist_m'] for e in events
                             if e.get('tf_delta')]
                avg_tf = np.mean(tf_deltas) if tf_deltas else 0
                print(f'  {pair[0]} vs {pair[1]}: '
                      f'{len(events)}frames '
                      f'cat={e0["a"]["cat"]}/{e0["b"]["cat"]} '
                      f'src={e0["a"]["src"]}/{e0["b"]["src"]} '
                      f'avg_map_dist={np.mean([e["dist"] for e in events]):.3f}m '
                      f'avg_tf_delta={avg_tf*100:.1f}cm')

        # TF delta 统计
        print('\n--- TF Delta (top vs chassis image timestamps) ---')
        try:
            deltas = []
            with open(self.log_path, 'r') as f:
                for line in f:
                    rec = json.loads(line)
                    td = rec.get('tf', {}).get('tf_delta_top_vs_chassis')
                    if td:
                        deltas.append(td['dist_m'])
            if deltas:
                arr = np.array(deltas)
                print(f'  n={len(arr)} '
                      f'P50={np.percentile(arr, 50)*100:.1f}cm '
                      f'P90={np.percentile(arr, 90)*100:.1f}cm '
                      f'P99={np.percentile(arr, 99)*100:.1f}cm '
                      f'max={np.max(arr)*100:.1f}cm')
                print(f'  >2cm: {np.sum(arr > 0.02)} frames '
                      f'>5cm: {np.sum(arr > 0.05)} frames '
                      f'>10cm: {np.sum(arr > 0.10)} frames')
        except Exception as e:
            print(f'  Error: {e}')

        # Track 位置稳定性
        print('\n--- Track Position Stability ---')
        try:
            track_pos = defaultdict(list)
            with open(self.log_path, 'r') as f:
                for line in f:
                    rec = json.loads(line)
                    for obj in rec.get('fused', []):
                        track_pos[obj['id']].append(obj['pos'][:2])
            for tid, pts_list in sorted(track_pos.items()):
                if len(pts_list) < 2:
                    continue
                pts = np.array(pts_list)
                mean = pts.mean(axis=0)
                dists = np.sqrt(((pts - mean)**2).sum(axis=1))
                print(f'  {tid}: n={len(pts)} '
                      f'mean=({mean[0]:.3f},{mean[1]:.3f}) '
                      f'std=({pts[:, 0].std():.4f},{pts[:, 1].std():.4f}) '
                      f'max_drift={dists.max():.4f}m')
        except Exception as e:
            print(f'  Error: {e}')

        print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Ghost target debug')
    parser.add_argument('--duration', type=float, default=60.0)
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'ghost_debug_{ts}')
    os.makedirs(output_dir, exist_ok=True)

    rclpy.init()
    node = GhostDebugNode(output_dir, args.duration)

    def shutdown_handler(sig, frame):
        node.get_logger().info('Interrupted')
        node._print_summary()
        node.log_file.close()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == '__main__':
    main()
