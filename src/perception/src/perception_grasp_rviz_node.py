#!/usr/bin/env python3
"""
Perception Grasp RViz Node - 抓取感知 RViz 可视化节点

订阅:
- perception_grasp_node/result (GraspObjectArray): 检测结果
- perception_grasp_node/depth (sensor_msgs/Image): CDM优化后深度图
- /camera/hand/color/image_raw (sensor_msgs/Image): RGB 图像

发布:
- ~vis/pointcloud (PointCloud2): RGBD 点云
- ~vis/panel (Image): 2x2 可视化面板
- ~markers (MarkerArray): 3D grasp markers
- ~grasp_pose (PoseStamped): 抓取位姿
"""

import os
import sys
import math
import threading

import numpy as np
import cv2
import rospy
import rospkg
import message_filters
from common.logger import get_logger
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point, Vector3, PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import tf.transformations as tft

# 导入自定义消息
from perception.msg import GraspObjectArray

# pycocotools for mask RLE decoding
try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


class PerceptionGraspRVizNode:
    """抓取感知 RViz 可视化节点"""

    # 物体颜色列表 (BGR)
    COLORS_BGR = [
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 红
        (255, 0, 0),    # 蓝
        (0, 255, 255),  # 黄
        (255, 0, 255),  # 品红
        (255, 255, 0),  # 青
        (0, 165, 255),  # 橙
        (255, 0, 128),  # 紫
    ]

    def __init__(self):
        rospy.init_node('perception_grasp_rviz_node')
        self.log = get_logger("GraspRViz")
        self.log.info("Initializing...")

        # 加载配置
        self.config = rospy.get_param('~perception_grasp_rviz', {})
        if not self.config:
            self.config = rospy.get_param('~', {})

        # === 参数 ===
        self._frame_id = self.config.get('frame_id', 'hand_camera_color_optical_frame')
        self._pointcloud_scale = self.config.get('pointcloud_scale', 0.5)
        self._marker_lifetime = self.config.get('marker_lifetime', 3.0)
        self._depth_min = self.config.get('depth_min', 0.001)
        self._depth_max = self.config.get('depth_max', 2.0)
        self.alpha = 0.1
        self.smooth_min = None
        self.smooth_max = None


        # 时间滤波参数
        self._temporal_filter_size = self.config.get('temporal_filter_size', 3)  # 帧数

        # === 数据缓存 ===
        self._intrinsics = None
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_result = None
        self._data_lock = threading.Lock()
        self._bridge = CvBridge()

        # 时间滤波缓冲区
        from collections import deque
        self._depth_buffer = deque(maxlen=self._temporal_filter_size)

        # === 获取相机内参 ===
        camera_info_topic = self.config.get('camera_info_topic', '/camera/hand/color/camera_info')
        self._get_camera_info(camera_info_topic)

        # === 订阅 RGB (单独订阅，用最新帧) ===
        rgb_topic = self.config.get('rgb_topic', '/camera/hand/color/image_raw')
        self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback, queue_size=1)
        self.log.info(f"订阅 RGB: {rgb_topic}")

        # === 订阅 result + depth（独立订阅以支持 latched 消息）===
        result_topic = self.config.get('result_topic', 'perception_grasp_node/result')
        depth_topic = self.config.get('depth_topic', 'perception_grasp_node/depth')

        # 使用独立订阅而不是 message_filters，因为 latched 消息的时间戳可能过时
        # result 和 depth 由同一节点同时发布，时间戳匹配，可以简单缓存
        self._result_sub = rospy.Subscriber(result_topic, GraspObjectArray, self._result_only_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_for_grasp_callback, queue_size=1)

        self.log.info(f"订阅 result: {result_topic}")
        self.log.info(f"订阅 depth: {depth_topic}")

        # === 发布器 ===
        pointcloud_topic = self.config.get('pointcloud_topic', '~vis/pointcloud')
        panel_topic = self.config.get('panel_topic', '~vis/panel')
        markers_topic = self.config.get('markers_topic', '~markers')
        pose_topic = self.config.get('grasp_pose_topic', '~grasp_pose')

        self._pointcloud_pub = rospy.Publisher(pointcloud_topic, PointCloud2, queue_size=1)
        self._panel_pub = rospy.Publisher(panel_topic, Image, queue_size=1)
        self._markers_pub = rospy.Publisher(markers_topic, MarkerArray, queue_size=1)
        self._pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)

        self.log.info(f"发布: {pointcloud_topic}, {panel_topic}, {markers_topic}")
        self.log.info("初始化完成")

    def _get_camera_info(self, topic):
        """获取相机内参"""
        self.log.info(f"等待相机内参: {topic}")
        try:
            info_msg = rospy.wait_for_message(topic, CameraInfo, timeout=10.0)
            self._intrinsics = {
                'fx': info_msg.K[0],
                'fy': info_msg.K[4],
                'cx': info_msg.K[2],
                'cy': info_msg.K[5],
                'width': info_msg.width,
                'height': info_msg.height,
            }
            self.log.info(f"内参: {info_msg.width}x{info_msg.height}")
        except rospy.ROSException as e:
            self.log.error(f"获取内参失败: {e}")
            raise

    def _rgb_callback(self, msg):
        """RGB 图像回调"""
        try:
            rgb_raw = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
            if msg.encoding == 'rgb8':
                rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                rgb = rgb_raw
            else:
                rgb = rgb_raw

            with self._data_lock:
                self._latest_rgb = rgb
        except Exception as e:
            self.log.warn(f"RGB 解码失败: {e}")

    def _result_only_callback(self, result_msg):
        """Result 独立回调"""
        try:
            with self._data_lock:
                self._latest_result = result_msg
                depth = self._latest_depth
                rgb = self._latest_rgb

            # 当收到新的 result，如果有 depth 和 rgb，立即生成可视化
            if depth is not None and rgb is not None:
                self._visualize(rgb, depth, result_msg)

        except Exception as e:
            self.log.warn(f"Result 回调失败: {e}")

    def _depth_for_grasp_callback(self, depth_msg):
        """Depth 独立回调（for grasp）"""
        try:
            # 解码深度图 (16UC1, mm)
            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            with self._data_lock:
                self._latest_depth = depth_m
                result = self._latest_result
                rgb = self._latest_rgb

            # 当收到新的 depth，如果有 result 和 rgb，立即生成可视化
            if result is not None and rgb is not None:
                self._visualize(rgb, depth_m, result)

        except Exception as e:
            self.log.warn(f"Depth 回调失败: {e}")

    def _sync_callback(self, result_msg, depth_msg):
        """result + depth 同步回调（已弃用，保留用于兼容）"""
        try:
            # 解码深度图 (16UC1, mm)
            depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            with self._data_lock:
                self._latest_depth = depth_m
                self._latest_result = result_msg
                rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None

            # 生成可视化
            if rgb is not None:
                self._visualize(rgb, depth_m, result_msg)

        except Exception as e:
            self.log.warn(f"同步回调失败: {e}")
            import traceback
            traceback.print_exc()

    def _visualize(self, rgb, depth, result):
        """生成并发布所有可视化"""
        # 使用 Time(0) 避免 TF 时间戳不匹配导致点云割裂
        # Time(0) 表示使用最新可用的 TF，而不是等待精确时间戳匹配
        stamp = rospy.Time(0)

        # 1. 发布点云
        try:
            pointcloud = self._create_pointcloud(rgb, depth, stamp)
            if pointcloud is not None:
                self._pointcloud_pub.publish(pointcloud)
        except Exception as e:
            self.log.warn(f"点云生成失败: {e}")

        # 2. 发布 2x2 Panel
        try:
            panel = self._create_panel(rgb, depth, result)
            if panel is not None:
                self._panel_pub.publish(self._bridge.cv2_to_imgmsg(panel, 'bgr8'))
        except Exception as e:
            self.log.warn(f"Panel 生成失败: {e}")

        # 3. 发布 3D Markers
        try:
            markers = self._create_markers(result, depth, stamp)
            self._markers_pub.publish(markers)
        except Exception as e:
            self.log.warn(f"Markers 生成失败: {e}")

        # 4. 发布 PoseStamped
        if result.chosen_index >= 0 and result.chosen_index < len(result.objects):
            try:
                pose = self._create_pose(result.objects[result.chosen_index], stamp)
                self._pose_pub.publish(pose)
            except Exception as e:
                self.log.warn(f"Pose 生成失败: {e}")

    def _create_panel(self, rgb, depth, result):
        """
        生成 2x2 可视化面板

        布局:
        ┌─────────────┬─────────────┐
        │  Detection  │    Mask     │
        ├─────────────┼─────────────┤
        │    Grasp    │    Depth    │
        └─────────────┴─────────────┘
        """
        img_h, img_w = rgb.shape[:2]
        panel_w, panel_h = img_w // 2, img_h // 2

        chosen_idx = result.chosen_index
        objects = result.objects

        # ============================================================
        # Panel 1: Detection (RGB + bbox + label/score)
        # ============================================================
        vis_detection = rgb.copy()
        for i, obj in enumerate(objects):
            color = self.COLORS_BGR[i % len(self.COLORS_BGR)]
            bbox = obj.bbox
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
                thickness = 3 if i == chosen_idx else 2
                cv2.rectangle(vis_detection, (x1, y1), (x2, y2), color, thickness)
                label = f"{obj.category} {obj.detection_score:.2f}"
                cv2.putText(vis_detection, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        vis_detection = cv2.resize(vis_detection, (panel_w, panel_h))

        # ============================================================
        # Panel 2: Mask (RGB + mask overlay)
        # ============================================================
        vis_mask = rgb.copy()
        for i, obj in enumerate(objects):
            if obj.mask_rle and HAS_COCO and obj.mask_size[0] > 0:
                try:
                    rle = {'counts': obj.mask_rle.encode('utf-8'), 'size': list(obj.mask_size)}
                    mask = coco_mask.decode(rle)
                    color = self.COLORS_BGR[i % len(self.COLORS_BGR)]
                    overlay = vis_mask.copy()
                    overlay[mask > 0] = color
                    alpha = 0.5 if i == chosen_idx else 0.3
                    cv2.addWeighted(overlay, alpha, vis_mask, 1 - alpha, 0, vis_mask)
                except Exception as e:
                    self.log.warn_throttle(5.0, f"Mask 解码失败: {e}")
        vis_mask = cv2.resize(vis_mask, (panel_w, panel_h))

        # ============================================================
        # Panel 3: Grasp (RGB + all grasp rects + chosen highlighted)
        # ============================================================
        vis_grasp = rgb.copy()
        for i, obj in enumerate(objects):
            is_chosen = (i == chosen_idx)

            if obj.grasp_score > 0 and obj.grasp_width > 0:
                # 有抓取点: 绘制抓取矩形
                cx, cy = obj.grasp_center
                w, h = obj.grasp_width, obj.grasp_width * 0.4
                angle_deg = math.degrees(obj.grasp_angle)
                rect = ((cx, cy), (w, h), angle_deg)
                box = cv2.boxPoints(rect).astype(np.int32)

                if is_chosen:
                    overlay = vis_grasp.copy()
                    cv2.fillPoly(overlay, [box], (0, 0, 255))  # 红色填充
                    cv2.addWeighted(overlay, 0.3, vis_grasp, 0.7, 0, vis_grasp)
                    cv2.drawContours(vis_grasp, [box], 0, (0, 0, 255), 5)  # 红色粗线 (5px)
                    cv2.circle(vis_grasp, (int(cx), int(cy)), 8, (0, 0, 255), -1)  # 红色粗圆点
                    # 显示：类别 D:检测分数 A:抓取分数 W:宽度mm
                    width_mm = obj.grasp_width3d * 1000
                    text = f"{obj.category} D:{obj.detection_score:.2f} A:{obj.grasp_score:.2f} W:{width_mm:.0f}mm"
                    # 透明背景 (减小一倍)
                    font_scale = 0.25
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    padding_x = 4
                    padding_y = 4
                    text_x = int(cx) + 8
                    text_y = int(cy)
                    overlay = vis_grasp.copy()
                    cv2.rectangle(overlay,
                                  (text_x - padding_x, text_y - text_h - padding_y),
                                  (text_x + text_w + padding_x, text_y + baseline + padding_y),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, vis_grasp, 0.5, 0, vis_grasp)  # 50% 透明
                    cv2.putText(vis_grasp, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                else:
                    # 非 chosen: 蓝色抓取框
                    cv2.drawContours(vis_grasp, [box], 0, (255, 128, 0), 2)
                    cv2.circle(vis_grasp, (int(cx), int(cy)), 4, (255, 128, 0), -1)
                    # 显示：类别 D:检测分数 A:抓取分数 W:宽度mm
                    width_mm = obj.grasp_width3d * 1000
                    text = f"{obj.category} D:{obj.detection_score:.2f} A:{obj.grasp_score:.2f} W:{width_mm:.0f}mm"
                    # 透明背景 (扩大背景框)
                    font_scale = 0.4
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    padding_x = 6
                    padding_y = 6
                    text_x = int(cx) + 8
                    text_y = int(cy)
                    overlay = vis_grasp.copy()
                    cv2.rectangle(overlay,
                                  (text_x - padding_x, text_y - text_h - padding_y),
                                  (text_x + text_w + padding_x, text_y + baseline + padding_y),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, vis_grasp, 0.6, 0, vis_grasp)  # 40% 透明
                    cv2.putText(vis_grasp, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 128, 0), font_thickness)
            else:
                # 无抓取点: 绘制 bbox 中心点
                if len(obj.bbox) >= 4:
                    x1, y1, x2, y2 = [int(x) for x in obj.bbox[:4]]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if is_chosen:
                        # chosen: 红色粗体
                        cv2.circle(vis_grasp, (cx, cy), 8, (0, 0, 255), -1)  # 粗圆点
                        # 显示：类别 D:检测分数 A:抓取分数 W:宽度mm
                        width_mm = obj.grasp_width3d * 1000
                        text = f"{obj.category} D:{obj.detection_score:.2f} A:{obj.grasp_score:.2f} W:{width_mm:.0f}mm"
                        # 透明背景 (减小一倍)
                        font_scale = 0.25
                        font_thickness = 1
                        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        padding_x = 4
                        padding_y = 4
                        text_x = cx + 8
                        text_y = cy
                        overlay = vis_grasp.copy()
                        cv2.rectangle(overlay,
                                      (text_x - padding_x, text_y - text_h - padding_y),
                                      (text_x + text_w + padding_x, text_y + baseline + padding_y),
                                      (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.5, vis_grasp, 0.5, 0, vis_grasp)  # 50% 透明
                        cv2.putText(vis_grasp, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                    else:
                        # 其他: 灰色正常
                        cv2.circle(vis_grasp, (cx, cy), 5, (128, 128, 128), -1)
                        # 显示：类别 D:检测分数 A:抓取分数 W:宽度mm
                        width_mm = obj.grasp_width3d * 1000
                        text = f"{obj.category} D:{obj.detection_score:.2f} A:{obj.grasp_score:.2f} W:{width_mm:.0f}mm"
                        # 透明背景 (扩大背景框)
                        font_scale = 0.4
                        font_thickness = 1
                        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        padding_x = 6
                        padding_y = 6
                        text_x = cx + 8
                        text_y = cy
                        overlay = vis_grasp.copy()
                        cv2.rectangle(overlay,
                                      (text_x - padding_x, text_y - text_h - padding_y),
                                      (text_x + text_w + padding_x, text_y + baseline + padding_y),
                                      (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.4, vis_grasp, 0.6, 0, vis_grasp)  # 40% 透明
                        cv2.putText(vis_grasp, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 128, 128), font_thickness)

        # 信息面板 (左上角)
        if chosen_idx >= 0 and chosen_idx < len(objects):
            chosen = objects[chosen_idx]
            info_lines = [
                f"Category: {chosen.category} ({chosen.grasp_score:.3f})",
                f"3D: [{chosen.position.x*1000:.1f}, {chosen.position.y*1000:.1f}, {chosen.position.z*1000:.1f}] mm",
                f"UV: [{chosen.grasp_center[0]:.0f}, {chosen.grasp_center[1]:.0f}]  Depth: {chosen.depth*1000:.0f}mm",
                f"Grasp: W={chosen.grasp_width3d*1000:.0f}mm  Angle={math.degrees(chosen.grasp_angle):.1f}deg",
            ]
            # 计算文字区域大小 (减小一倍)
            font_scale = 0.6  # 减小一倍字体
            line_height = 25  # 减小行高
            info_h = len(info_lines) * line_height + 10
            info_w = 330  # 减小宽度

            # 绘制背景矩形 (左上角)
            cv2.rectangle(vis_grasp, (10, 10), (info_w, info_h), (0, 0, 0), -1)
            cv2.rectangle(vis_grasp, (10, 10), (info_w, info_h), (0, 255, 0), 2)

            # 绘制文字 (左上角开始)
            for j, line in enumerate(info_lines):
                y = 15 + (j + 1) * line_height
                cv2.putText(vis_grasp, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        else:
            cv2.putText(vis_grasp, 'No valid grasp', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        vis_grasp = cv2.resize(vis_grasp, (panel_w, panel_h))

        # ============================================================
        # Panel 4: Depth (colormap + chosen marker)
        # ============================================================
        curr_min, curr_max = np.percentile(depth, [2, 98])
        if self.smooth_min is None:
            self.smooth_min = curr_min
            self.smooth_max = curr_max
        else:
            self.smooth_min = self.alpha * curr_min + (1 - self.alpha) * self.smooth_min
            self.smooth_max = self.alpha * curr_max + (1 - self.alpha) * self.smooth_max
        denom = max(self.smooth_max - self.smooth_min, 1e-5)
        depth_norm = (depth - self.smooth_min) / denom * 255
        depth_norm = np.clip(depth_norm, 0, 255).astype(np.uint8)
        #depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        vis_depth = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_TURBO)

        if chosen_idx >= 0 and chosen_idx < len(objects):
            chosen = objects[chosen_idx]
            cx, cy = int(chosen.grasp_center[0]), int(chosen.grasp_center[1])
            cv2.circle(vis_depth, (cx, cy), 12, (0, 255, 0), 3)
            cv2.drawMarker(vis_depth, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        vis_depth = cv2.resize(vis_depth, (panel_w, panel_h))

        # ============================================================
        # 组合 2x2 面板
        # ============================================================
        top_row = np.hstack([vis_detection, vis_mask])
        bottom_row = np.hstack([vis_grasp, vis_depth])
        panel = np.vstack([top_row, bottom_row])

        # 添加面板标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Detection", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Mask", (panel_w + 10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Grasp", (10, panel_h + 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Depth", (panel_w + 10, panel_h + 25), font, 0.7, (255, 255, 255), 2)

        return panel

    def _create_pointcloud(self, rgb, depth, stamp):
        """从 RGBD 生成 PointCloud2"""
        if self._intrinsics is None:
            return None

        # 降低分辨率
        scale = self._pointcloud_scale
        h, w = depth.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        rgb_small = cv2.resize(rgb, (new_w, new_h))
        depth_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 时间滤波: 中值滤波减少深度抖动
        self._depth_buffer.append(depth_small.copy())
        if len(self._depth_buffer) >= 2:
            depth_stack = np.stack(list(self._depth_buffer), axis=0)
            depth_small = np.median(depth_stack, axis=0).astype(np.float32)

        # 调整内参
        fx = self._intrinsics['fx'] * scale
        fy = self._intrinsics['fy'] * scale
        cx = self._intrinsics['cx'] * scale
        cy = self._intrinsics['cy'] * scale

        # 向量化投影
        u, v = np.meshgrid(np.arange(new_w), np.arange(new_h))
        u = u.flatten().astype(np.float32)
        v = v.flatten().astype(np.float32)
        z = depth_small.flatten()

        # 过滤无效深度
        valid = (z > self._depth_min) & (z < self._depth_max)
        u, v, z = u[valid], v[valid], z[valid]

        # 反投影
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # RGB 值 (BGR -> RGB packed)
        rgb_flat = rgb_small.reshape(-1, 3)[valid]
        r = rgb_flat[:, 2].astype(np.uint32)
        g = rgb_flat[:, 1].astype(np.uint32)
        b = rgb_flat[:, 0].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b

        # 构建 PointCloud2
        points = np.zeros(len(x), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
        ])
        points['x'] = x
        points['y'] = y
        points['z'] = z
        points['rgb'] = rgb_packed

        msg = PointCloud2()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = stamp
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = points.tobytes()

        return msg

    def _create_markers(self, result, depth, stamp):
        """生成 3D markers - 显示所有 grasp，chosen 高亮"""
        markers = MarkerArray()
        lifetime = rospy.Duration(self._marker_lifetime)

        # 清除旧 markers
        clear_marker = Marker()
        clear_marker.header.frame_id = self._frame_id
        clear_marker.header.stamp = stamp
        clear_marker.ns = "grasp"
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        chosen_idx = result.chosen_index
        marker_id = 1

        # 为每个物体生成 markers
        for i, obj in enumerate(result.objects):
            is_chosen = (i == chosen_idx)
            point3d = np.array([obj.position.x, obj.position.y, obj.position.z])

            # 跳过无效位置
            if np.linalg.norm(point3d) < 0.001:
                continue

            angle_rad = obj.grasp_angle
            width3d = obj.grasp_width3d

            # 颜色设置: chosen=绿色, 其他=灰色半透明
            if is_chosen:
                sphere_color = ColorRGBA(0, 1, 0, 1)
                arrow_color = ColorRGBA(1, 0.5, 0, 1)
                gripper_color = ColorRGBA(0, 1, 1, 0.9)
                bbox_color = ColorRGBA(1, 1, 0, 0.8)
                text_color = ColorRGBA(1, 1, 1, 0.9)  # 透明背景（降低 alpha）
                scale_factor = 1.0
            else:
                sphere_color = ColorRGBA(0.5, 0.5, 0.5, 0.5)
                arrow_color = ColorRGBA(0.6, 0.4, 0.2, 0.5)
                gripper_color = ColorRGBA(0.4, 0.6, 0.6, 0.4)
                bbox_color = ColorRGBA(0.6, 0.6, 0.3, 0.4)
                text_color = ColorRGBA(0.8, 0.8, 0.8, 0.5)  # 透明背景（降低 alpha）
                scale_factor = 0.7

            # Approach 方向 (-Z, 从物体指向相机)
            approach_dir = np.array([0, 0, -1])
            # 夹爪方向 (在 XY 平面内旋转)
            gripper_dir = np.array([math.cos(angle_rad), math.sin(angle_rad), 0])

            # === 1. Grasp Point (SPHERE) ===
            sphere = Marker()
            sphere.header.frame_id = self._frame_id
            sphere.header.stamp = stamp
            sphere.ns = "grasp"
            sphere.id = marker_id
            marker_id += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(*point3d)
            sphere.pose.orientation.w = 1.0
            s = 0.015 * scale_factor
            sphere.scale = Vector3(s, s, s)
            sphere.color = sphere_color
            sphere.lifetime = lifetime
            markers.markers.append(sphere)

            # === 2. Grasp Arrow (approach 方向) ===
            arrow = Marker()
            arrow.header.frame_id = self._frame_id
            arrow.header.stamp = stamp
            arrow.ns = "grasp"
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow_end = point3d + approach_dir * 0.08 * scale_factor
            arrow.points = [Point(*point3d), Point(*arrow_end)]
            arrow.scale = Vector3(0.008 * scale_factor, 0.015 * scale_factor, 0)
            arrow.color = arrow_color
            arrow.lifetime = lifetime
            markers.markers.append(arrow)

            # === 3. Gripper Lines (U 形) ===
            if width3d > 0:
                half_w = width3d / 2
                finger1 = point3d + gripper_dir * half_w
                finger2 = point3d - gripper_dir * half_w
                finger1_tip = finger1 + approach_dir * 0.04
                finger2_tip = finger2 + approach_dir * 0.04

                gripper = Marker()
                gripper.header.frame_id = self._frame_id
                gripper.header.stamp = stamp
                gripper.ns = "grasp"
                gripper.id = marker_id
                marker_id += 1
                gripper.type = Marker.LINE_LIST
                gripper.action = Marker.ADD
                gripper.points = [
                    Point(*finger1_tip), Point(*finger2_tip),
                    Point(*finger1_tip), Point(*finger1),
                    Point(*finger2_tip), Point(*finger2),
                ]
                gripper.scale.x = 0.005 * scale_factor
                gripper.color = gripper_color
                gripper.lifetime = lifetime
                markers.markers.append(gripper)

            # === 4. Bbox 3D (只为 chosen 显示) ===
            if is_chosen and obj.grasp_width3d > 0 and obj.depth > 0:
                edges = self._create_grasp_bbox_3d(obj)
                if edges:
                    bbox_marker = Marker()
                    bbox_marker.header.frame_id = self._frame_id
                    bbox_marker.header.stamp = stamp
                    bbox_marker.ns = "grasp"
                    bbox_marker.id = marker_id
                    marker_id += 1
                    bbox_marker.type = Marker.LINE_LIST
                    bbox_marker.action = Marker.ADD
                    for p1, p2 in edges:
                        bbox_marker.points.append(Point(*p1))
                        bbox_marker.points.append(Point(*p2))
                    bbox_marker.scale.x = 0.003
                    bbox_marker.color = bbox_color
                    bbox_marker.lifetime = lifetime
                    markers.markers.append(bbox_marker)

            # === 5. Text Label ===
            text = Marker()
            text.header.frame_id = self._frame_id
            text.header.stamp = stamp
            text.ns = "grasp"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(point3d[0], point3d[1] - 0.05, point3d[2])
            text.pose.orientation.w = 1.0
            text.scale.z = 0.025 * scale_factor
            # 显示类别、detection score、affordance score 和 grasp width
            width_mm = width3d * 1000  # 转换为毫米
            text.text = f"{obj.category}\nD:{obj.detection_score:.2f} A:{obj.grasp_score:.2f} W:{width_mm:.0f}mm"
            text.color = text_color
            text.lifetime = lifetime
            markers.markers.append(text)

        return markers

    def _create_grasp_bbox_3d(self, obj, thickness=0.03):
        """基于抓取位置和宽度创建 3D bbox

        使用 grasp_width3d 作为 bbox 宽度，避免 2D 投影导致的尺寸失真
        """
        if obj.depth <= 0 or obj.grasp_width3d <= 0:
            return []

        # 抓取中心 3D 坐标
        cx = obj.position.x
        cy = obj.position.y
        cz = obj.position.z

        # bbox 尺寸基于 grasp_width3d
        half_w = obj.grasp_width3d / 2  # 宽度方向 (X)
        half_h = obj.grasp_width3d / 2  # 高度方向 (Y)，假设正方形
        half_d = thickness / 2          # 深度方向 (Z)

        # 8 个角点 (以抓取中心为原点)
        corners = [
            [cx - half_w, cy - half_h, cz - half_d],  # 前左下
            [cx + half_w, cy - half_h, cz - half_d],  # 前右下
            [cx + half_w, cy + half_h, cz - half_d],  # 前右上
            [cx - half_w, cy + half_h, cz - half_d],  # 前左上
            [cx - half_w, cy - half_h, cz + half_d],  # 后左下
            [cx + half_w, cy - half_h, cz + half_d],  # 后右下
            [cx + half_w, cy + half_h, cz + half_d],  # 后右上
            [cx - half_w, cy + half_h, cz + half_d],  # 后左上
        ]

        # 12 条边
        edges = []
        # 前面 4 条
        for i in range(4):
            edges.append((corners[i], corners[(i + 1) % 4]))
        # 后面 4 条
        for i in range(4, 8):
            edges.append((corners[i], corners[4 + (i - 4 + 1) % 4]))
        # 连接前后 4 条
        for i in range(4):
            edges.append((corners[i], corners[i + 4]))

        return edges

    def _create_pose(self, obj, stamp):
        """创建 PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = self._frame_id
        pose.header.stamp = stamp
        pose.pose.position = obj.position

        # 四元数: 绕 Z 轴旋转
        q = tft.quaternion_from_euler(0, 0, obj.grasp_angle)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        return pose

    def spin(self):
        """主循环"""
        rospy.spin()


def main():
    try:
        node = PerceptionGraspRVizNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[GraspRViz] Exception exit: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
