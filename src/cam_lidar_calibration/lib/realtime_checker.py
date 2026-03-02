#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数据质量检查器 - 用于采集时检测重投影误差
"""

import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import yaml


class RealtimeChecker:
    """实时检查棋盘格检测质量"""

    def __init__(self, intrinsic_file=None, K=None, D=None, pattern_size=(7, 6), square_size=0.072):
        """
        Args:
            intrinsic_file: 相机内参文件路径 (可选，与 K/D 二选一)
            K: 内参矩阵 3x3 (可选，与 intrinsic_file 二选一)
            D: 畸变系数 (可选，与 intrinsic_file 二选一)
            pattern_size: 棋盘格尺寸 (cols, rows)
            square_size: 格子边长(米)
        """
        self.bridge = CvBridge()
        self.pattern_size = pattern_size
        self.square_size = square_size

        # 加载内参 - 支持两种方式
        if K is not None and D is not None:
            # 方式1: 直接传入矩阵 (推荐，零配置)
            self.K = np.array(K)
            self.D = np.array(D)
        elif intrinsic_file is not None:
            # 方式2: 从文件加载 (向后兼容)
            with open(intrinsic_file, 'r') as f:
                data = yaml.safe_load(f)

            # 支持两种 YAML 格式
            if 'camera_matrix' in data:
                # 新格式: camera_intrinsics_<type>.yaml
                self.K = np.array(data['camera_matrix']['data']).reshape(3, 3)
                self.D = np.array(data['distortion_coefficients']['data'])
            else:
                # 旧格式: 直接 K 和 D 字段
                self.K = np.array(data['K']).reshape(3, 3)
                self.D = np.array(data['D'])
        else:
            raise ValueError("必须提供 intrinsic_file 或 (K, D)")

        # 生成3D角点
        cols, rows = pattern_size
        self.objp = np.zeros((rows * cols, 3), np.float32)
        for i in range(rows):
            for j in range(cols):
                self.objp[i * cols + j] = [
                    (j - (cols - 1) / 2.0) * square_size,
                    (i - (rows - 1) / 2.0) * square_size,
                    0
                ]

        self.latest_image = None
        self.latest_result = None

    def image_callback(self, msg):
        """ROS图像回调"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(f"[警告] 图像转换失败: {e}")

    def check_current_frame(self):
        """检查当前帧的质量"""
        if self.latest_image is None:
            return None

        image = self.latest_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)

        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if not ret:
            return {
                'success': False,
                'error': '未检测到棋盘格',
                'suggestion': '请调整标定板位置，确保完整可见',
                'vis': image
            }

        # 亚像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 求解位姿
        ret, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.D)
        if not ret:
            return {
                'success': False,
                'error': 'PnP求解失败',
                'suggestion': '标定板可能遮挡或角度过大',
                'vis': image
            }

        # 重投影
        projected, _ = cv2.projectPoints(self.objp, rvec, tvec, self.K, self.D)
        projected = projected.reshape(-1, 2)

        # 计算误差
        errors = np.linalg.norm(corners.reshape(-1, 2) - projected, axis=1)
        rms = np.sqrt(np.mean(errors**2))
        mean = np.mean(errors)
        max_err = np.max(errors)

        # 计算标定板姿态
        R, _ = cv2.Rodrigues(rvec)
        normal = R[:, 2]  # 标定板法向量

        # 计算倾斜角度
        # normal在相机坐标系中，Z轴指向前方
        # 标定板法向量与Z轴的夹角就是倾斜角
        angle_deg = np.degrees(np.arccos(abs(normal[2])))

        # 可视化
        vis = image.copy()

        # 绘制检测到的角点（绿色）
        cv2.drawChessboardCorners(vis, self.pattern_size, corners, True)

        # 绘制重投影点（红色×）
        for pt in projected:
            cv2.drawMarker(vis, tuple(pt.astype(int)),
                          (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

        # 绘制误差向量（黄色箭头，仅显示>1px的）
        for corner, proj, err in zip(corners.reshape(-1, 2), projected, errors):
            if err > 1.0:
                pt1 = tuple(corner.astype(int))
                pt2 = tuple(proj.astype(int))
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 255), 2, tipLength=0.3)

        # 绘制信息面板
        panel_h = 200
        panel = np.zeros((panel_h, vis.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        # 质量评估
        if rms > 3.0:
            quality = "POOR"
            quality_color = (0, 0, 255)  # 红色
            recommendation = "Adjust board - tilt too large or intrinsics off"
        elif rms > 1.5:
            quality = "OK"
            quality_color = (0, 165, 255)  # 橙色
            recommendation = "Acceptable - reduce tilt for better result"
        else:
            quality = "GOOD"
            quality_color = (0, 255, 0)  # 绿色
            recommendation = "Ready to record"

        # 绘制文本 (ASCII only - cv2.putText 不支持 Unicode)
        y = 30
        cv2.putText(panel, "Quality Check", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y += 40
        cv2.putText(panel, "Reproj: RMS=%.2fpx  Mean=%.2fpx  Max=%.2fpx" % (rms, mean, max_err),
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y += 35
        cv2.putText(panel, "Board tilt: %.1f deg" % angle_deg,
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y += 35
        cv2.putText(panel, quality,
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

        y += 35
        cv2.putText(panel, recommendation,
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 拼接面板和图像
        vis_with_panel = np.vstack([vis, panel])

        return {
            'success': True,
            'rms': rms,
            'mean': mean,
            'max': max_err,
            'angle_deg': angle_deg,
            'quality': quality,
            'quality_color': quality_color,
            'recommendation': recommendation,
            'vis': vis_with_panel,
            'acceptable': rms < 3.0  # 可接受的阈值
        }

    def start_subscriber(self, image_topic, node):
        """启动ROS订阅"""
        node.create_subscription(Image, image_topic, self.image_callback, 10)

    def show_live_preview(self, window_name="质量检测"):
        """显示实时预览窗口（不等待按键，由调用者控制）"""
        result = self.check_current_frame()
        if result and result['vis'] is not None:
            cv2.imshow(window_name, result['vis'])
        return result
