#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perception Visualization Node

RViz 可视化节点，实时显示检测和跟踪结果。

订阅:
    /scene_perception_3d/objects_3d (Object3DArray) - 检测结果
    /object_tracker/tracked_objects (TrackedObject3DArray) - 跟踪结果
    /camera/<name>/color/image_raw (Image) - RGB 图像

发布:
    ~detection_image (Image) - 2D 检测可视化
    ~tracking_image (Image) - 2D 跟踪可视化
    ~detection_markers (MarkerArray) - 3D 检测 Markers
    ~tracking_markers (MarkerArray) - 3D 跟踪 Markers
"""

import os
import sys

# 添加 perception/src 路径
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 添加 ROS devel Python 路径
_devel_python_path = os.path.realpath(os.path.join(
    os.path.dirname(_src_dir), '..', '..', 'devel', 'lib', 'python3', 'dist-packages'
))
if os.path.isdir(_devel_python_path) and _devel_python_path not in sys.path:
    sys.path.insert(0, _devel_python_path)

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

from perception.msg import Object3D, Object3DArray

# 尝试导入 TrackedObject3DArray (可能未构建)
try:
    from perception.msg import TrackedObject3D, TrackedObject3DArray
    HAS_TRACKER_MSG = True
except ImportError:
    HAS_TRACKER_MSG = False
    rospy.logwarn("[PerceptionViz] TrackedObject3DArray 未找到，跟踪可视化禁用")


class PerceptionVizNode:
    """感知结果可视化节点"""

    # 颜色调色板 (BGR for OpenCV, RGB for RViz)
    COLORS_BGR = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
    ]

    COLORS_RGB = [
        (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 1),
        (1, 0, 1), (1, 1, 0), (0, 1, 0.5), (0, 0.5, 1),
        (1, 0.5, 0), (0.5, 0, 1), (1, 0, 0.5), (0.5, 1, 0)
    ]

    def __init__(self):
        rospy.init_node('perception_viz_node')

        # 参数
        self.camera_name = rospy.get_param('~camera_name', 'top')
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.marker_size = rospy.get_param('~marker_size', 0.08)
        self.text_size = rospy.get_param('~text_size', 0.1)

        # CV Bridge
        self._bridge = CvBridge()

        # 缓存
        self._rgb_image = None
        self._rgb_stamp = None

        # 话题
        rgb_topic = f'/camera/{self.camera_name}/color/image_raw'
        detection_topic = rospy.get_param('~detection_topic', '/scene_perception_3d/objects_3d')
        tracking_topic = rospy.get_param('~tracking_topic', '/object_tracker/tracked_objects')

        # 订阅
        self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback, queue_size=1)
        self._det_sub = rospy.Subscriber(detection_topic, Object3DArray, self._detection_callback, queue_size=1)

        if HAS_TRACKER_MSG:
            self._track_sub = rospy.Subscriber(tracking_topic, TrackedObject3DArray, self._tracking_callback, queue_size=1)

        # 发布
        self._det_img_pub = rospy.Publisher('~detection_image', Image, queue_size=1)
        self._det_marker_pub = rospy.Publisher('~detection_markers', MarkerArray, queue_size=1)

        if HAS_TRACKER_MSG:
            self._track_img_pub = rospy.Publisher('~tracking_image', Image, queue_size=1)
            self._track_marker_pub = rospy.Publisher('~tracking_markers', MarkerArray, queue_size=1)

        rospy.loginfo(f"[PerceptionViz] 节点启动")
        rospy.loginfo(f"  相机: {self.camera_name}")
        rospy.loginfo(f"  检测话题: {detection_topic}")
        rospy.loginfo(f"  跟踪话题: {tracking_topic}")
        rospy.loginfo(f"  发布:")
        rospy.loginfo(f"    ~detection_image, ~detection_markers")
        if HAS_TRACKER_MSG:
            rospy.loginfo(f"    ~tracking_image, ~tracking_markers")

    def _rgb_callback(self, msg: Image):
        """RGB 图像回调"""
        try:
            self._rgb_image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._rgb_stamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[PerceptionViz] RGB 转换失败: {e}")

    def _detection_callback(self, msg: Object3DArray):
        """检测结果回调"""
        if self._rgb_image is None:
            return

        # 2D 可视化
        if self._det_img_pub.get_num_connections() > 0:
            img_vis = self._draw_detections_2d(self._rgb_image.copy(), msg.objects)
            try:
                img_msg = self._bridge.cv2_to_imgmsg(img_vis, 'bgr8')
                img_msg.header = msg.header
                self._det_img_pub.publish(img_msg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[PerceptionViz] 发布检测图像失败: {e}")

        # 3D Markers
        if self._det_marker_pub.get_num_connections() > 0:
            markers = self._create_detection_markers(msg)
            self._det_marker_pub.publish(markers)

    def _tracking_callback(self, msg):
        """跟踪结果回调"""
        if self._rgb_image is None:
            return

        # 2D 可视化
        if self._track_img_pub.get_num_connections() > 0:
            img_vis = self._draw_tracking_2d(self._rgb_image.copy(), msg.objects)
            try:
                img_msg = self._bridge.cv2_to_imgmsg(img_vis, 'bgr8')
                img_msg.header = msg.header
                self._track_img_pub.publish(img_msg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[PerceptionViz] 发布跟踪图像失败: {e}")

        # 3D Markers
        if self._track_marker_pub.get_num_connections() > 0:
            markers = self._create_tracking_markers(msg)
            self._track_marker_pub.publish(markers)

    def _draw_detections_2d(self, img: np.ndarray, objects: list) -> np.ndarray:
        """绘制 2D 检测结果"""
        for i, obj in enumerate(objects):
            color = self.COLORS_BGR[i % len(self.COLORS_BGR)]

            # 绘制 bbox
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 标签: 类别 + 分数 + 距离
            label1 = f"{obj.category} {obj.score:.2f}"
            label2 = f"{obj.distance:.2f}m"

            # 标签背景
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            max_tw = max(tw1, tw2)

            cv2.rectangle(img, (x1, y1 - th1 - th2 - 10), (x1 + max_tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(img, label1, (x1 + 3, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, label2, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 标题
        title = f"Detection: {len(objects)} objects"
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

    def _draw_tracking_2d(self, img: np.ndarray, objects: list) -> np.ndarray:
        """绘制 2D 跟踪结果"""
        for obj in objects:
            # 根据 track_id 选择颜色 (同一 ID 保持同一颜色)
            color = self.COLORS_BGR[obj.track_id % len(self.COLORS_BGR)]

            # 绘制 bbox
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # 更粗的线表示跟踪

            # 标签: ID + 类别 + 距离
            label1 = f"#{obj.track_id} {obj.category}"
            label2 = f"{obj.distance:.2f}m" if obj.distance > 0 else "N/A"

            # 标签背景
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_tw = max(tw1, tw2)

            cv2.rectangle(img, (x1, y1 - th1 - th2 - 12), (x1 + max_tw + 8, y1), color, -1)
            cv2.putText(img, label1, (x1 + 4, y1 - th2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, label2, (x1 + 4, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 标题
        title = f"Tracking: {len(objects)} objects"
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return img

    def _create_detection_markers(self, msg: Object3DArray) -> MarkerArray:
        """创建检测结果的 3D Markers"""
        marker_array = MarkerArray()

        # 清除旧 markers
        clear_marker = Marker()
        clear_marker.header = msg.header
        clear_marker.header.frame_id = self.target_frame
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for i, obj in enumerate(msg.objects):
            color_rgb = self.COLORS_RGB[i % len(self.COLORS_RGB)]

            # 球体 Marker
            sphere = Marker()
            sphere.header = msg.header
            sphere.header.frame_id = self.target_frame
            sphere.ns = "detection_spheres"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = obj.position
            sphere.pose.orientation.w = 1.0
            sphere.scale = Vector3(self.marker_size, self.marker_size, self.marker_size)
            sphere.color = ColorRGBA(color_rgb[0], color_rgb[1], color_rgb[2], 0.8)
            sphere.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(sphere)

            # 文字标签
            text = Marker()
            text.header = msg.header
            text.header.frame_id = self.target_frame
            text.ns = "detection_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(
                obj.position.x,
                obj.position.y,
                obj.position.z + self.marker_size + 0.02
            )
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_size
            text.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text.text = f"{obj.category}\n{obj.distance:.2f}m"
            text.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(text)

        return marker_array

    def _create_tracking_markers(self, msg) -> MarkerArray:
        """创建跟踪结果的 3D Markers"""
        marker_array = MarkerArray()

        # 清除旧 markers
        clear_marker = Marker()
        clear_marker.header = msg.header
        clear_marker.header.frame_id = self.target_frame
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for obj in msg.objects:
            color_rgb = self.COLORS_RGB[obj.track_id % len(self.COLORS_RGB)]

            # 立方体 Marker (区别于检测的球体)
            cube = Marker()
            cube.header = msg.header
            cube.header.frame_id = self.target_frame
            cube.ns = "tracking_cubes"
            cube.id = obj.track_id
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position = obj.position
            cube.pose.orientation.w = 1.0
            cube.scale = Vector3(self.marker_size, self.marker_size, self.marker_size)
            cube.color = ColorRGBA(color_rgb[0], color_rgb[1], color_rgb[2], 0.9)
            cube.lifetime = rospy.Duration(0.3)
            marker_array.markers.append(cube)

            # 文字标签 (包含 track_id)
            text = Marker()
            text.header = msg.header
            text.header.frame_id = self.target_frame
            text.ns = "tracking_labels"
            text.id = obj.track_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(
                obj.position.x,
                obj.position.y,
                obj.position.z + self.marker_size + 0.02
            )
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_size
            text.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)  # 黄色标签
            dist_str = f"{obj.distance:.2f}m" if obj.distance > 0 else ""
            text.text = f"#{obj.track_id} {obj.category}\n{dist_str}"
            text.lifetime = rospy.Duration(0.3)
            marker_array.markers.append(text)

        return marker_array

    def run(self):
        """运行节点"""
        rospy.spin()


def main():
    try:
        node = PerceptionVizNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[PerceptionViz] 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
