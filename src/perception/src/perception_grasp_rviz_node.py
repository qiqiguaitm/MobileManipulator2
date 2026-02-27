#!/usr/bin/env python3
"""
Perception Grasp RViz Node - 抓取感知 RViz 可视化节点 (ROS2)

订阅:
- perception_grasp_node/result (GraspObjectArray): 检测结果
- perception_grasp_node/depth (sensor_msgs/Image): CDM优化后深度图
- /camera/hand/color/image_raw (sensor_msgs/Image): RGB 图像

发布:
- ~/vis/pointcloud (PointCloud2): RGBD 点云
- ~/vis/panel (Image): 2x2 可视化面板
- ~/markers (MarkerArray): 3D grasp markers
- ~/grasp_pose (PoseStamped): 抓取位姿
"""

import math
import threading
from collections import deque
from typing import Optional

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
from perception.utils import CvBridgeNumPy2 as CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point, Vector3, PoseStamped, Quaternion
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from perception.msg import GraspObjectArray

# pycocotools for mask RLE decoding
try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

# 传感器 QoS
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# Latched QoS
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1
)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """欧拉角转四元数"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class PerceptionGraspRVizNode(Node):
    """抓取感知 RViz 可视化节点"""

    COLORS_BGR = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 165, 255), (255, 0, 128),
    ]

    def __init__(self):
        super().__init__('perception_grasp_rviz_node')

        # === 加载参数 ===
        self._load_parameters()

        # === 数据缓存 ===
        self._intrinsics = None
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_result = None
        self._data_lock = threading.Lock()
        self._bridge = CvBridge()

        # 时间滤波缓冲区
        self._depth_buffer = deque(maxlen=self._temporal_filter_size)

        # 深度范围平滑
        self._smooth_min = None
        self._smooth_max = None
        self._alpha = 0.1

        # === 获取相机内参 ===
        self._get_camera_info()

        # === 订阅 RGB ===
        self._rgb_sub = self.create_subscription(
            Image,
            self._rgb_topic,
            self._rgb_callback,
            SENSOR_QOS
        )
        self.get_logger().info(f'订阅 RGB: {self._rgb_topic}')

        # === 订阅 result ===
        self._result_sub = self.create_subscription(
            GraspObjectArray,
            self._result_topic,
            self._result_callback,
            LATCHED_QOS
        )
        self.get_logger().info(f'订阅 result: {self._result_topic}')

        # === 订阅 depth ===
        self._depth_sub = self.create_subscription(
            Image,
            self._depth_topic,
            self._depth_callback,
            LATCHED_QOS
        )
        self.get_logger().info(f'订阅 depth: {self._depth_topic}')

        # === 发布器 ===
        self._pointcloud_pub = self.create_publisher(
            PointCloud2, '~/vis/pointcloud', 1
        )
        self._panel_pub = self.create_publisher(
            Image, '~/vis/panel', 1
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, '~/markers', 1
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, '~/grasp_pose', 1
        )

        self.get_logger().info('节点初始化完成')

    def _load_parameters(self):
        """声明和加载参数"""
        self.declare_parameter('frame_id', 'hand_camera_color_optical_frame')
        self.declare_parameter('pointcloud_scale', 0.5)
        self.declare_parameter('marker_lifetime', 3.0)
        self.declare_parameter('depth_min', 0.001)
        self.declare_parameter('depth_max', 2.0)
        self.declare_parameter('temporal_filter_size', 3)

        self.declare_parameter('rgb_topic', '/camera/hand/color/image_raw')
        self.declare_parameter('depth_topic', 'perception_grasp_node/depth')
        self.declare_parameter('result_topic', 'perception_grasp_node/result')
        self.declare_parameter('camera_info_topic', '/camera/hand/color/camera_info')

        self._frame_id = self.get_parameter('frame_id').value
        self._pointcloud_scale = self.get_parameter('pointcloud_scale').value
        self._marker_lifetime = self.get_parameter('marker_lifetime').value
        self._depth_min = self.get_parameter('depth_min').value
        self._depth_max = self.get_parameter('depth_max').value
        self._temporal_filter_size = self.get_parameter('temporal_filter_size').value

        self._rgb_topic = self.get_parameter('rgb_topic').value
        self._depth_topic = self.get_parameter('depth_topic').value
        self._result_topic = self.get_parameter('result_topic').value
        self._camera_info_topic = self.get_parameter('camera_info_topic').value

    def _get_camera_info(self):
        """获取相机内参"""
        self.get_logger().info(f'等待相机内参: {self._camera_info_topic}')
        try:
            # 使用一次性订阅获取
            from rclpy.qos import qos_profile_sensor_data
            msg = None
            timeout = 10.0
            start = self.get_clock().now()
            sub = self.create_subscription(
                CameraInfo, self._camera_info_topic,
                lambda m: setattr(self, '_temp_info', m),
                qos_profile_sensor_data
            )
            self._temp_info = None
            while self._temp_info is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                if (self.get_clock().now() - start).nanoseconds / 1e9 > timeout:
                    raise RuntimeError('等待相机内参超时')
            msg = self._temp_info
            self.destroy_subscription(sub)

            self._intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height,
            }
            self.get_logger().info(f'内参: {msg.width}x{msg.height}')
        except Exception as e:
            self.get_logger().error(f'获取内参失败: {e}')
            raise

    def _rgb_callback(self, msg: Image):
        """RGB 回调"""
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
            self.get_logger().warn(f'RGB 解码失败: {e}')

    def _result_callback(self, msg: GraspObjectArray):
        """Result 回调"""
        with self._data_lock:
            self._latest_result = msg
            depth = self._latest_depth
            rgb = self._latest_rgb

        if depth is not None and rgb is not None:
            self._visualize(rgb, depth, msg)

    def _depth_callback(self, msg: Image):
        """Depth 回调"""
        try:
            depth_mm = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
            depth_m = depth_mm.astype(np.float32) / 1000.0

            with self._data_lock:
                self._latest_depth = depth_m
                result = self._latest_result
                rgb = self._latest_rgb

            if result is not None and rgb is not None:
                self._visualize(rgb, depth_m, result)
        except Exception as e:
            self.get_logger().warn(f'Depth 解码失败: {e}')

    def _visualize(self, rgb, depth, result):
        """生成并发布所有可视化"""
        stamp = self.get_clock().now().to_msg()

        # 1. 点云
        try:
            pointcloud = self._create_pointcloud(rgb, depth, stamp)
            if pointcloud is not None:
                self._pointcloud_pub.publish(pointcloud)
        except Exception as e:
            self.get_logger().warn(f'点云生成失败: {e}')

        # 2. 2x2 Panel
        try:
            panel = self._create_panel(rgb, depth, result)
            if panel is not None:
                self._panel_pub.publish(self._bridge.cv2_to_imgmsg(panel, 'bgr8'))
        except Exception as e:
            self.get_logger().warn(f'Panel 生成失败: {e}')

        # 3. 3D Markers
        try:
            markers = self._create_markers(result, depth, stamp)
            self._markers_pub.publish(markers)
        except Exception as e:
            self.get_logger().warn(f'Markers 生成失败: {e}')

        # 4. PoseStamped
        if 0 <= result.chosen_index < len(result.objects):
            try:
                pose = self._create_pose(result.objects[result.chosen_index], stamp)
                self._pose_pub.publish(pose)
            except Exception as e:
                self.get_logger().warn(f'Pose 生成失败: {e}')

    def _create_panel(self, rgb, depth, result):
        """生成 2x2 可视化面板"""
        img_h, img_w = rgb.shape[:2]
        panel_w, panel_h = img_w // 2, img_h // 2

        chosen_idx = result.chosen_index
        objects = result.objects

        # Panel 1: Detection
        vis_detection = rgb.copy()
        for i, obj in enumerate(objects):
            color = self.COLORS_BGR[i % len(self.COLORS_BGR)]
            bbox = obj.bbox
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(x) for x in bbox[:4]]
                thickness = 3 if i == chosen_idx else 2
                cv2.rectangle(vis_detection, (x1, y1), (x2, y2), color, thickness)
                label = f'{obj.category} {obj.detection_score:.2f}'
                cv2.putText(vis_detection, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        vis_detection = cv2.resize(vis_detection, (panel_w, panel_h))

        # Panel 2: Mask
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
                except Exception:
                    pass
        vis_mask = cv2.resize(vis_mask, (panel_w, panel_h))

        # Panel 3: Grasp
        vis_grasp = rgb.copy()
        for i, obj in enumerate(objects):
            is_chosen = (i == chosen_idx)
            if obj.grasp_score > 0 and obj.grasp_width > 0:
                cx, cy = obj.grasp_center
                w, h = obj.grasp_width, obj.grasp_width * 0.4
                angle_deg = math.degrees(obj.grasp_angle)
                rect = ((cx, cy), (w, h), angle_deg)
                box = cv2.boxPoints(rect).astype(np.int32)

                if is_chosen:
                    cv2.drawContours(vis_grasp, [box], 0, (0, 0, 255), 5)
                    cv2.circle(vis_grasp, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                else:
                    cv2.drawContours(vis_grasp, [box], 0, (255, 128, 0), 2)
                    cv2.circle(vis_grasp, (int(cx), int(cy)), 4, (255, 128, 0), -1)

        if 0 <= chosen_idx < len(objects):
            chosen = objects[chosen_idx]
            info_lines = [
                f'Category: {chosen.category} ({chosen.grasp_score:.3f})',
                f'3D: [{chosen.position.x*1000:.1f}, {chosen.position.y*1000:.1f}, {chosen.position.z*1000:.1f}] mm',
                f'Width: {chosen.grasp_width3d*1000:.0f}mm Angle: {math.degrees(chosen.grasp_angle):.1f}deg',
            ]
            font_scale = 0.5
            line_height = 20
            for j, line in enumerate(info_lines):
                y = 20 + j * line_height
                cv2.putText(vis_grasp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        vis_grasp = cv2.resize(vis_grasp, (panel_w, panel_h))

        # Panel 4: Depth
        curr_min, curr_max = np.percentile(depth, [2, 98])
        if self._smooth_min is None:
            self._smooth_min = curr_min
            self._smooth_max = curr_max
        else:
            self._smooth_min = self._alpha * curr_min + (1 - self._alpha) * self._smooth_min
            self._smooth_max = self._alpha * curr_max + (1 - self._alpha) * self._smooth_max

        denom = max(self._smooth_max - self._smooth_min, 1e-5)
        depth_norm = (depth - self._smooth_min) / denom * 255
        depth_norm = np.clip(depth_norm, 0, 255).astype(np.uint8)
        vis_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        if 0 <= chosen_idx < len(objects):
            chosen = objects[chosen_idx]
            cx, cy = int(chosen.grasp_center[0]), int(chosen.grasp_center[1])
            cv2.circle(vis_depth, (cx, cy), 12, (0, 255, 0), 3)
        vis_depth = cv2.resize(vis_depth, (panel_w, panel_h))

        # 组合
        top_row = np.hstack([vis_detection, vis_mask])
        bottom_row = np.hstack([vis_grasp, vis_depth])
        panel = np.vstack([top_row, bottom_row])

        # 标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, 'Detection', (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, 'Mask', (panel_w + 10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, 'Grasp', (10, panel_h + 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, 'Depth', (panel_w + 10, panel_h + 25), font, 0.7, (255, 255, 255), 2)

        return panel

    def _create_pointcloud(self, rgb, depth, stamp):
        """从 RGBD 生成 PointCloud2"""
        if self._intrinsics is None:
            return None

        scale = self._pointcloud_scale
        h, w = depth.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        rgb_small = cv2.resize(rgb, (new_w, new_h))
        depth_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 时间滤波
        self._depth_buffer.append(depth_small.copy())
        if len(self._depth_buffer) >= 2:
            depth_stack = np.stack(list(self._depth_buffer), axis=0)
            depth_small = np.median(depth_stack, axis=0).astype(np.float32)

        fx = self._intrinsics['fx'] * scale
        fy = self._intrinsics['fy'] * scale
        cx = self._intrinsics['cx'] * scale
        cy = self._intrinsics['cy'] * scale

        u, v = np.meshgrid(np.arange(new_w), np.arange(new_h))
        u = u.flatten().astype(np.float32)
        v = v.flatten().astype(np.float32)
        z = depth_small.flatten()

        valid = (z > self._depth_min) & (z < self._depth_max)
        u, v, z = u[valid], v[valid], z[valid]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        rgb_flat = rgb_small.reshape(-1, 3)[valid]
        r = rgb_flat[:, 2].astype(np.uint32)
        g = rgb_flat[:, 1].astype(np.uint32)
        b = rgb_flat[:, 0].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b

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
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = points.tobytes()

        return msg

    def _create_markers(self, result, depth, stamp):
        """生成 3D markers"""
        markers = MarkerArray()
        lifetime = rclpy.duration.Duration(seconds=self._marker_lifetime).to_msg()

        # 清除旧 markers
        clear_marker = Marker()
        clear_marker.header.frame_id = self._frame_id
        clear_marker.header.stamp = stamp
        clear_marker.ns = 'grasp'
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        chosen_idx = result.chosen_index
        marker_id = 1

        for i, obj in enumerate(result.objects):
            is_chosen = (i == chosen_idx)
            point3d = np.array([obj.position.x, obj.position.y, obj.position.z])

            if np.linalg.norm(point3d) < 0.001:
                continue

            angle_rad = obj.grasp_angle
            width3d = obj.grasp_width3d

            # 颜色
            if is_chosen:
                sphere_color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                arrow_color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
                gripper_color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9)
                bbox_color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
                text_color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
                scale_factor = 1.0
            else:
                sphere_color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)
                arrow_color = ColorRGBA(r=0.6, g=0.4, b=0.2, a=0.5)
                gripper_color = ColorRGBA(r=0.4, g=0.6, b=0.6, a=0.4)
                bbox_color = ColorRGBA(r=0.6, g=0.6, b=0.3, a=0.4)
                text_color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.5)
                scale_factor = 0.7

            approach_dir = np.array([0, 0, -1])
            gripper_dir = np.array([math.cos(angle_rad), math.sin(angle_rad), 0])

            # Sphere
            sphere = Marker()
            sphere.header.frame_id = self._frame_id
            sphere.header.stamp = stamp
            sphere.ns = 'grasp'
            sphere.id = marker_id
            marker_id += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=point3d[0], y=point3d[1], z=point3d[2])
            sphere.pose.orientation.w = 1.0
            s = 0.015 * scale_factor
            sphere.scale = Vector3(x=s, y=s, z=s)
            sphere.color = sphere_color
            sphere.lifetime = lifetime
            markers.markers.append(sphere)

            # Arrow
            arrow = Marker()
            arrow.header.frame_id = self._frame_id
            arrow.header.stamp = stamp
            arrow.ns = 'grasp'
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow_end = point3d + approach_dir * 0.08 * scale_factor
            arrow.points = [
                Point(x=point3d[0], y=point3d[1], z=point3d[2]),
                Point(x=arrow_end[0], y=arrow_end[1], z=arrow_end[2])
            ]
            arrow.scale = Vector3(x=0.008 * scale_factor, y=0.015 * scale_factor, z=0.0)
            arrow.color = arrow_color
            arrow.lifetime = lifetime
            markers.markers.append(arrow)

            # Gripper
            if width3d > 0:
                half_w = width3d / 2
                finger1 = point3d + gripper_dir * half_w
                finger2 = point3d - gripper_dir * half_w
                finger1_tip = finger1 + approach_dir * 0.04
                finger2_tip = finger2 + approach_dir * 0.04

                gripper = Marker()
                gripper.header.frame_id = self._frame_id
                gripper.header.stamp = stamp
                gripper.ns = 'grasp'
                gripper.id = marker_id
                marker_id += 1
                gripper.type = Marker.LINE_LIST
                gripper.action = Marker.ADD
                gripper.points = [
                    Point(x=finger1_tip[0], y=finger1_tip[1], z=finger1_tip[2]),
                    Point(x=finger2_tip[0], y=finger2_tip[1], z=finger2_tip[2]),
                    Point(x=finger1_tip[0], y=finger1_tip[1], z=finger1_tip[2]),
                    Point(x=finger1[0], y=finger1[1], z=finger1[2]),
                    Point(x=finger2_tip[0], y=finger2_tip[1], z=finger2_tip[2]),
                    Point(x=finger2[0], y=finger2[1], z=finger2[2]),
                ]
                gripper.scale.x = 0.005 * scale_factor
                gripper.color = gripper_color
                gripper.lifetime = lifetime
                markers.markers.append(gripper)

            # Bbox 3D (只为 chosen 显示)
            if is_chosen and obj.grasp_width3d > 0 and obj.depth > 0:
                edges = self._create_grasp_bbox_3d(obj)
                if edges:
                    bbox_marker = Marker()
                    bbox_marker.header.frame_id = self._frame_id
                    bbox_marker.header.stamp = stamp
                    bbox_marker.ns = 'grasp'
                    bbox_marker.id = marker_id
                    marker_id += 1
                    bbox_marker.type = Marker.LINE_LIST
                    bbox_marker.action = Marker.ADD
                    for p1, p2 in edges:
                        bbox_marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
                        bbox_marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))
                    bbox_marker.scale.x = 0.003
                    bbox_marker.color = bbox_color
                    bbox_marker.lifetime = lifetime
                    markers.markers.append(bbox_marker)

            # Text
            text = Marker()
            text.header.frame_id = self._frame_id
            text.header.stamp = stamp
            text.ns = 'grasp'
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=point3d[0], y=point3d[1] - 0.05, z=point3d[2])
            text.pose.orientation.w = 1.0
            text.scale.z = 0.025 * scale_factor
            width_mm = width3d * 1000
            text.text = f'{obj.category}\nD:{obj.detection_score:.2f} A:{obj.grasp_score:.2f} W:{width_mm:.0f}mm'
            text.color = text_color
            text.lifetime = lifetime
            markers.markers.append(text)

        return markers

    def _create_grasp_bbox_3d(self, obj, thickness: float = 0.03):
        """基于抓取位置和宽度创建 3D bbox

        使用 grasp_width3d 作为 bbox 宽度，避免 2D 投影导致的尺寸失真

        Args:
            obj: GraspObject 消息
            thickness: bbox 深度方向厚度 (m)

        Returns:
            edges: 12 条边的列表，每条边是 (p1, p2) 元组
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

    def _create_pose(self, obj, stamp) -> PoseStamped:
        """创建 PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = self._frame_id
        pose.header.stamp = stamp
        pose.pose.position = obj.position
        pose.pose.orientation = quaternion_from_euler(0, 0, obj.grasp_angle)
        return pose


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionGraspRVizNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
