#!/usr/bin/env python3
"""
双相机 BaseLink 一致性验证工具（基于多相机感知节点）

功能：
1. 基础验证模式：比较两个相机对同一物体的3D检测差异
2. 实验模式：系统化验证不同距离和角度下的3D检测精度

使用方法：
  cd /home/didi/workspace/MobileManipulator2
  source install/setup.bash
  ros2 run perception dual_camera_baselink_validation

操作说明（基础模式）：
  - 鼠标左键：点击选择检测框（先左图Chassis，后右图Top）
  - 空格：确认匹配并记录误差
  - C：清除当前选择
  - S：保存数据到 CSV
  - E：进入/退出实验模式
  - Q/ESC：退出

操作说明（实验模式）：
  - 空格/S：保存当前测试点数据
  - N：下一个测试点
  - P：上一个测试点
  - C：清除当前测试点数据
  - E：退出实验模式
  - Q/ESC：结束实验并生成报告
"""

import os
import sys
import rclpy
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from threading import Lock

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from perception.msg import Object3D, Object3DArray
from perception.utils import CvBridgeNumPy2


# QoS 配置
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE
)

LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)


@dataclass
class Detection3D:
    """3D 检测结果"""
    object_id: str
    category: str
    score: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    position: np.ndarray  # base_link 下的 3D 位置 [x, y, z]
    distance: float
    camera_source: str  # 'chassis' 或 'top'

    @property
    def center_2d(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class MatchResult:
    """匹配结果"""
    timestamp: str
    category: str
    chassis_pos: np.ndarray
    top_pos: np.ndarray
    error_3d: float
    error_x: float
    error_y: float
    error_z: float


@dataclass
class ExperimentPoint:
    """实验测试点"""
    test_id: str
    distance: float
    angle: str  # 左/正/右
    gt_x: float
    gt_y: float
    gt_z: float
    # 测量结果
    chassis_pos: Optional[np.ndarray] = None
    top_pos: Optional[np.ndarray] = None
    chassis_error: Optional[float] = None
    top_error: Optional[float] = None
    consistency_error: Optional[float] = None
    timestamp: Optional[str] = None


class DualCameraValidator(Node):
    """双相机验证节点"""

    # 实验测试点定义
    TEST_POINTS = [
        # 0.6m 距离
        ("D0.6_L", 0.6, "左", 0.6, 0.3, 0.05),
        ("D0.6_C", 0.6, "正", 0.6, 0.0, 0.05),
        ("D0.6_R", 0.6, "右", 0.6, -0.3, 0.05),
        # 1.2m 距离
        ("D1.2_L", 1.2, "左", 1.2, 0.3, 0.05),
        ("D1.2_C", 1.2, "正", 1.2, 0.0, 0.05),
        ("D1.2_R", 1.2, "右", 1.2, -0.3, 0.05),
        # 1.6m 距离
        ("D1.6_L", 1.6, "左", 1.6, 0.3, 0.05),
        ("D1.6_C", 1.6, "正", 1.6, 0.0, 0.05),
        ("D1.6_R", 1.6, "右", 1.6, -0.3, 0.05),
    ]

    def __init__(self):
        super().__init__('dual_camera_validator')

        # 参数
        self.declare_parameter('output_dir', '~/dual_camera_validation')
        self.declare_parameter('perception_node_name', 'multi_camera_perception')

        self.output_dir = os.path.expanduser(self.get_parameter('output_dir').value)
        self.perception_node = self.get_parameter('perception_node_name').value
        os.makedirs(self.output_dir, exist_ok=True)

        # 数据锁
        self._lock = Lock()

        # 检测数据缓存
        self.chassis_detections: List[Detection3D] = []
        self.top_detections: List[Detection3D] = []

        # RGB 图像缓存
        self.chassis_rgb: Optional[np.ndarray] = None
        self.top_rgb: Optional[np.ndarray] = None

        # 选择状态（使用 object_id 而非索引，避免检测刷新时跳动）
        self.selected_chassis_id: Optional[str] = None
        self.selected_top_id: Optional[str] = None

        # 数据冻结标志（选中后冻结对应相机数据）
        self._chassis_frozen = False
        self._top_frozen = False

        # 记录
        self.records: List[MatchResult] = []

        # 实验模式
        self.experiment_mode = False
        self.experiment_points: List[ExperimentPoint] = []
        self.current_exp_idx = 0

        # CvBridge
        self.bridge = CvBridgeNumPy2()

        # 窗口名称
        self.window_name = 'Dual Camera Validation'
        self._window_initialized = False  # 窗口初始化标志

        # 设置订阅
        self._setup_subscribers()

        # 发布 markers
        self.pub_markers = self.create_publisher(MarkerArray, '/validation/markers', 10)

        # 定时器
        self.create_timer(0.05, self._main_loop)  # 20Hz

        self.get_logger().info('='*60)
        self.get_logger().info('双相机验证节点已启动')
        self.get_logger().info(f'感知节点: {self.perception_node}')
        self.get_logger().info(f'输出目录: {self.output_dir}')
        self.get_logger().info('='*60)
        self.get_logger().info('操作: 鼠标点选 → 空格记录 → S保存 → E实验模式')

    def _setup_subscribers(self):
        """设置订阅"""
        # 检测结果
        chassis_det_topic = f'/{self.perception_node}/chassis/objects_3d'
        top_det_topic = f'/{self.perception_node}/top/objects_3d'

        self.create_subscription(
            Object3DArray, chassis_det_topic,
            self._chassis_det_cb, LATCHED_QOS
        )
        self.create_subscription(
            Object3DArray, top_det_topic,
            self._top_det_cb, LATCHED_QOS
        )

        # RGB 图像
        self.create_subscription(
            Image, '/camera/chassis/color/image_raw',
            self._chassis_rgb_cb, SENSOR_QOS
        )
        self.create_subscription(
            Image, '/camera/top/color/image_raw',
            self._top_rgb_cb, SENSOR_QOS
        )

        self.get_logger().info(f'订阅检测: {chassis_det_topic}')
        self.get_logger().info(f'订阅检测: {top_det_topic}')

    def _chassis_det_cb(self, msg: Object3DArray):
        """Chassis 检测回调"""
        with self._lock:
            if self._chassis_frozen:
                return  # 已冻结，不更新
            self.chassis_detections = []
            for obj in msg.objects:
                det = Detection3D(
                    object_id=obj.object_id,
                    category=obj.category,
                    score=obj.score,
                    bbox=tuple(int(x) for x in obj.bbox),
                    position=np.array([obj.position.x, obj.position.y, obj.position.z]),
                    distance=obj.distance,
                    camera_source='chassis'
                )
                self.chassis_detections.append(det)

    def _top_det_cb(self, msg: Object3DArray):
        """Top 检测回调"""
        with self._lock:
            if self._top_frozen:
                return  # 已冻结，不更新
            self.top_detections = []
            for obj in msg.objects:
                det = Detection3D(
                    object_id=obj.object_id,
                    category=obj.category,
                    score=obj.score,
                    bbox=tuple(int(x) for x in obj.bbox),
                    position=np.array([obj.position.x, obj.position.y, obj.position.z]),
                    distance=obj.distance,
                    camera_source='top'
                )
                self.top_detections.append(det)

    def _chassis_rgb_cb(self, msg: Image):
        """Chassis RGB 回调"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            if msg.encoding == 'rgb8':
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            with self._lock:
                if not self._chassis_frozen:
                    self.chassis_rgb = rgb
        except Exception as e:
            self.get_logger().debug(f'Chassis RGB error: {e}')

    def _top_rgb_cb(self, msg: Image):
        """Top RGB 回调"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            if msg.encoding == 'rgb8':
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            with self._lock:
                if not self._top_frozen:
                    self.top_rgb = rgb
        except Exception as e:
            self.get_logger().debug(f'Top RGB error: {e}')

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        with self._lock:
            # 判断点击在左边还是右边
            if self.chassis_rgb is None:
                return

            w = self.chassis_rgb.shape[1]

            if x < w:
                # 点击左边 (Chassis)
                self._select_detection(x, y, 'chassis')
            else:
                # 点击右边 (Top)
                self._select_detection(x - w, y, 'top')

    def _select_detection(self, x: int, y: int, camera: str):
        """选择检测框"""
        detections = self.chassis_detections if camera == 'chassis' else self.top_detections

        best_det = None
        best_dist = float('inf')

        for det in detections:
            cx, cy = det.center_2d
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)

            # 检查是否在 bbox 内或距离中心足够近
            x1, y1, x2, y2 = det.bbox
            in_box = x1 <= x <= x2 and y1 <= y <= y2

            if (in_box or dist < 50) and dist < best_dist:
                best_dist = dist
                best_det = det

        if best_det is not None:
            if camera == 'chassis':
                self.selected_chassis_id = best_det.object_id
                self._chassis_frozen = True  # 冻结数据
                self.get_logger().info(f'选中 Chassis: {best_det.object_id} ({best_det.category}) [已冻结]')
            else:
                self.selected_top_id = best_det.object_id
                self._top_frozen = True  # 冻结数据
                self.get_logger().info(f'选中 Top: {best_det.object_id} ({best_det.category}) [已冻结]')

    def _create_visualization(self) -> np.ndarray:
        """创建可视化图像"""
        with self._lock:
            # 获取图像
            if self.chassis_rgb is not None:
                chassis_img = self.chassis_rgb.copy()
            else:
                chassis_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(chassis_img, "Waiting for Chassis camera...",
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            if self.top_rgb is not None:
                top_img = self.top_rgb.copy()
            else:
                top_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(top_img, "Waiting for Top camera...",
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            # 调整尺寸一致
            h = max(chassis_img.shape[0], top_img.shape[0])
            w = max(chassis_img.shape[1], top_img.shape[1])

            if chassis_img.shape[:2] != (h, w):
                chassis_img = cv2.resize(chassis_img, (w, h))
            if top_img.shape[:2] != (h, w):
                top_img = cv2.resize(top_img, (w, h))

            # 绘制检测框
            self._draw_detections(chassis_img, self.chassis_detections,
                                 self.selected_chassis_id, (0, 255, 100))
            self._draw_detections(top_img, self.top_detections,
                                 self.selected_top_id, (255, 200, 0))

            # 添加标签（显示冻结状态）
            chassis_label = f"CHASSIS ({len(self.chassis_detections)} objects)"
            if self._chassis_frozen:
                chassis_label += " [FROZEN]"
            cv2.putText(chassis_img, chassis_label,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 255) if self._chassis_frozen else (0, 255, 100), 2)

            top_label = f"TOP ({len(self.top_detections)} objects)"
            if self._top_frozen:
                top_label += " [FROZEN]"
            cv2.putText(top_img, top_label,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 255) if self._top_frozen else (255, 200, 0), 2)

            # 合并
            combined = np.hstack([chassis_img, top_img])

            # 添加信息面板
            combined = self._draw_info_panel(combined)

            return combined

    def _draw_detections(self, img: np.ndarray, detections: List[Detection3D],
                        selected_id: Optional[str], color: Tuple[int, int, int]):
        """绘制检测框"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # 边界检查
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            # 选中的用粗框（通过 object_id 判断）
            is_selected = (det.object_id == selected_id) if selected_id else False
            thickness = 3 if is_selected else 2
            box_color = (0, 255, 255) if is_selected else color

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

            # 标签
            label = f"{det.category} {det.score:.2f}"
            pos_label = f"({det.position[0]:.2f}, {det.position[1]:.2f}, {det.position[2]:.2f})"

            cv2.putText(img, label, (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            cv2.putText(img, pos_label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_info_panel(self, img: np.ndarray) -> np.ndarray:
        """绘制信息面板"""
        h, w = img.shape[:2]
        panel_h = 100 if self.experiment_mode else 60
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

        if self.experiment_mode:
            # 实验模式
            if self.current_exp_idx < len(self.experiment_points):
                pt = self.experiment_points[self.current_exp_idx]
                progress = f"{self.current_exp_idx + 1}/{len(self.experiment_points)}"

                cv2.putText(panel, f"[EXPERIMENT] {pt.test_id} | Distance: {pt.distance}m | Angle: {pt.angle} | Progress: {progress}",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                gt_text = f"Ground Truth: X={pt.gt_x:.2f}m, Y={pt.gt_y:.2f}m, Z={pt.gt_z:.2f}m"
                cv2.putText(panel, gt_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if pt.timestamp:
                    cv2.putText(panel, "[RECORDED] Press N for next", (10, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(panel, "[NOT RECORDED] Select objects and press SPACE", (10, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # 基础模式
            status = f"Records: {len(self.records)}"
            if self.selected_chassis_id is not None:
                status += f" | Chassis: {self.selected_chassis_id}"
            if self.selected_top_id is not None:
                status += f" | Top: {self.selected_top_id}"

            cv2.putText(panel, status, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(panel, "Keys: SPACE=Record | C=Clear | S=Save | E=Experiment | Q=Quit",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return np.vstack([panel, img])

    def _find_detection_by_id(self, detections: List[Detection3D], obj_id: str) -> Optional[Detection3D]:
        """通过 object_id 查找检测"""
        for det in detections:
            if det.object_id == obj_id:
                return det
        return None

    def _record_match(self):
        """记录当前匹配"""
        with self._lock:
            if self.selected_chassis_id is None or self.selected_top_id is None:
                self.get_logger().warn('请先在两个图像中选择检测框')
                return False

            det_c = self._find_detection_by_id(self.chassis_detections, self.selected_chassis_id)
            det_t = self._find_detection_by_id(self.top_detections, self.selected_top_id)

            if det_c is None:
                self.get_logger().warn(f'Chassis 选择已失效: {self.selected_chassis_id}')
                self.selected_chassis_id = None
                return False

            if det_t is None:
                self.get_logger().warn(f'Top 选择已失效: {self.selected_top_id}')
                self.selected_top_id = None
                return False

            # 计算误差
            error_vec = det_c.position - det_t.position
            error_3d = np.linalg.norm(error_vec)

            result = MatchResult(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                category=det_c.category,
                chassis_pos=det_c.position.copy(),
                top_pos=det_t.position.copy(),
                error_3d=error_3d,
                error_x=abs(error_vec[0]),
                error_y=abs(error_vec[1]),
                error_z=abs(error_vec[2])
            )

            self.records.append(result)

            self.get_logger().info(f'记录 #{len(self.records)}: {det_c.category}')
            self.get_logger().info(f'  Chassis: ({det_c.position[0]:.3f}, {det_c.position[1]:.3f}, {det_c.position[2]:.3f})')
            self.get_logger().info(f'  Top:     ({det_t.position[0]:.3f}, {det_t.position[1]:.3f}, {det_t.position[2]:.3f})')
            self.get_logger().info(f'  3D误差: {error_3d*100:.1f} cm')

            # 如果在实验模式，记录到当前测试点
            if self.experiment_mode and self.current_exp_idx < len(self.experiment_points):
                pt = self.experiment_points[self.current_exp_idx]
                pt.chassis_pos = det_c.position.copy()
                pt.top_pos = det_t.position.copy()
                pt.consistency_error = error_3d

                # 计算与真实位置的误差
                gt = np.array([pt.gt_x, pt.gt_y, pt.gt_z])
                pt.chassis_error = np.linalg.norm(det_c.position - gt)
                pt.top_error = np.linalg.norm(det_t.position - gt)
                pt.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                self.get_logger().info(f'  Chassis vs GT: {pt.chassis_error*100:.1f} cm')
                self.get_logger().info(f'  Top vs GT: {pt.top_error*100:.1f} cm')

            # 清除选择并解除冻结
            self.selected_chassis_id = None
            self.selected_top_id = None
            self._chassis_frozen = False
            self._top_frozen = False

            return True

    def _save_data(self):
        """保存数据"""
        if not self.records:
            self.get_logger().warn('没有数据可保存')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'{self.output_dir}/validation_{timestamp}.csv'

        with open(filepath, 'w') as f:
            f.write('timestamp,category,chassis_x,chassis_y,chassis_z,top_x,top_y,top_z,error_3d,error_x,error_y,error_z\n')
            for r in self.records:
                f.write(f'{r.timestamp},{r.category},'
                       f'{r.chassis_pos[0]:.4f},{r.chassis_pos[1]:.4f},{r.chassis_pos[2]:.4f},'
                       f'{r.top_pos[0]:.4f},{r.top_pos[1]:.4f},{r.top_pos[2]:.4f},'
                       f'{r.error_3d:.4f},{r.error_x:.4f},{r.error_y:.4f},{r.error_z:.4f}\n')

        self.get_logger().info(f'数据已保存: {filepath}')
        self._print_statistics()

    def _print_statistics(self):
        """打印统计信息"""
        if not self.records:
            return

        errors = [r.error_3d for r in self.records]

        print('\n' + '='*60)
        print('  双相机一致性验证统计')
        print('='*60)
        print(f'  样本数: {len(self.records)}')
        print(f'  3D误差:')
        print(f'    平均: {np.mean(errors)*100:.1f} cm')
        print(f'    中位: {np.median(errors)*100:.1f} cm')
        print(f'    最大: {np.max(errors)*100:.1f} cm')
        print(f'    最小: {np.min(errors)*100:.1f} cm')
        print(f'    标准差: {np.std(errors)*100:.1f} cm')
        print('='*60 + '\n')

    def _start_experiment(self):
        """开始实验"""
        self.experiment_mode = True
        self.experiment_points = []
        self.current_exp_idx = 0

        for test_id, dist, angle, gt_x, gt_y, gt_z in self.TEST_POINTS:
            self.experiment_points.append(ExperimentPoint(
                test_id=test_id,
                distance=dist,
                angle=angle,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_z=gt_z
            ))

        self.get_logger().info('='*60)
        self.get_logger().info('进入实验模式')
        self.get_logger().info(f'共 {len(self.experiment_points)} 个测试点')
        self.get_logger().info('操作: 空格=记录 | N=下一个 | P=上一个 | C=清除 | E=退出')
        self.get_logger().info('='*60)

    def _end_experiment(self):
        """结束实验"""
        self.experiment_mode = False

        # 统计完成数
        completed = sum(1 for pt in self.experiment_points if pt.timestamp is not None)

        if completed > 0:
            self._save_experiment_data()
            self._print_experiment_report()

        self.get_logger().info('退出实验模式')

    def _save_experiment_data(self):
        """保存实验数据"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'{self.output_dir}/experiment_{timestamp}.csv'

        with open(filepath, 'w') as f:
            f.write('test_id,distance,angle,gt_x,gt_y,gt_z,'
                   'chassis_x,chassis_y,chassis_z,chassis_error,'
                   'top_x,top_y,top_z,top_error,consistency_error,timestamp\n')

            for pt in self.experiment_points:
                c_x = f'{pt.chassis_pos[0]:.4f}' if pt.chassis_pos is not None else ''
                c_y = f'{pt.chassis_pos[1]:.4f}' if pt.chassis_pos is not None else ''
                c_z = f'{pt.chassis_pos[2]:.4f}' if pt.chassis_pos is not None else ''
                t_x = f'{pt.top_pos[0]:.4f}' if pt.top_pos is not None else ''
                t_y = f'{pt.top_pos[1]:.4f}' if pt.top_pos is not None else ''
                t_z = f'{pt.top_pos[2]:.4f}' if pt.top_pos is not None else ''
                c_err = f'{pt.chassis_error:.4f}' if pt.chassis_error is not None else ''
                t_err = f'{pt.top_error:.4f}' if pt.top_error is not None else ''
                cons = f'{pt.consistency_error:.4f}' if pt.consistency_error is not None else ''
                ts = pt.timestamp or ''

                f.write(f'{pt.test_id},{pt.distance},{pt.angle},'
                       f'{pt.gt_x},{pt.gt_y},{pt.gt_z},'
                       f'{c_x},{c_y},{c_z},{c_err},'
                       f'{t_x},{t_y},{t_z},{t_err},{cons},{ts}\n')

        self.get_logger().info(f'实验数据已保存: {filepath}')

    def _print_experiment_report(self):
        """打印实验报告"""
        completed = [pt for pt in self.experiment_points if pt.timestamp is not None]

        if not completed:
            return

        print('\n' + '='*70)
        print('  双相机 3D 精度实验报告')
        print('='*70)
        print(f'  完成测试点: {len(completed)}/{len(self.experiment_points)}')
        print()

        # Chassis 统计
        c_errors = [pt.chassis_error for pt in completed if pt.chassis_error is not None]
        if c_errors:
            print('  Chassis 相机精度:')
            print(f'    平均误差: {np.mean(c_errors)*100:.1f} cm')
            print(f'    最大误差: {np.max(c_errors)*100:.1f} cm')
            print()

        # Top 统计
        t_errors = [pt.top_error for pt in completed if pt.top_error is not None]
        if t_errors:
            print('  Top 相机精度:')
            print(f'    平均误差: {np.mean(t_errors)*100:.1f} cm')
            print(f'    最大误差: {np.max(t_errors)*100:.1f} cm')
            print()

        # 一致性统计
        cons_errors = [pt.consistency_error for pt in completed if pt.consistency_error is not None]
        if cons_errors:
            print('  双相机一致性:')
            print(f'    平均误差: {np.mean(cons_errors)*100:.1f} cm')
            print(f'    最大误差: {np.max(cons_errors)*100:.1f} cm')
            print()

        # 按距离统计
        print('  按距离统计:')
        for dist in [0.6, 1.2, 1.6]:
            pts = [pt for pt in completed if pt.distance == dist]
            if pts:
                c_errs = [pt.chassis_error*100 for pt in pts if pt.chassis_error]
                t_errs = [pt.top_error*100 for pt in pts if pt.top_error]
                print(f'    {dist}m: Chassis={np.mean(c_errs):.1f}cm, Top={np.mean(t_errs):.1f}cm')

        print('='*70 + '\n')

    def _publish_markers(self):
        """发布 RViz markers"""
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # 清除旧 markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        with self._lock:
            # Chassis 检测
            for i, det in enumerate(self.chassis_detections):
                m = Marker()
                m.header.frame_id = 'base_link'
                m.header.stamp = now
                m.ns = 'chassis'
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position = Point(x=det.position[0], y=det.position[1], z=det.position[2])
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.08

                is_selected = (det.object_id == self.selected_chassis_id) if self.selected_chassis_id else False
                if is_selected:
                    m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
                else:
                    m.color = ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.7)

                markers.markers.append(m)

            # Top 检测
            for i, det in enumerate(self.top_detections):
                m = Marker()
                m.header.frame_id = 'base_link'
                m.header.stamp = now
                m.ns = 'top'
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position = Point(x=det.position[0], y=det.position[1], z=det.position[2])
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.08

                is_selected = (det.object_id == self.selected_top_id) if self.selected_top_id else False
                if is_selected:
                    m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
                else:
                    m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.7)

                markers.markers.append(m)

        self.pub_markers.publish(markers)

    def _handle_key(self, key: int):
        """处理按键"""
        if key == ord('q') or key == 27:  # Q 或 ESC
            if self.experiment_mode:
                self._end_experiment()
            self._save_data()
            rclpy.shutdown()

        elif key == ord('e'):  # E - 切换实验模式
            if self.experiment_mode:
                self._end_experiment()
            else:
                self._start_experiment()

        elif key == ord(' '):  # 空格 - 记录
            self._record_match()

        elif key == ord('c'):  # C - 清除
            if self.experiment_mode and self.current_exp_idx < len(self.experiment_points):
                pt = self.experiment_points[self.current_exp_idx]
                pt.chassis_pos = None
                pt.top_pos = None
                pt.chassis_error = None
                pt.top_error = None
                pt.consistency_error = None
                pt.timestamp = None
                self.get_logger().info(f'清除测试点 {pt.test_id}')

            self.selected_chassis_id = None
            self.selected_top_id = None
            self._chassis_frozen = False
            self._top_frozen = False
            self.get_logger().info('清除选择，数据已解冻')

        elif key == ord('s'):  # S - 保存
            if self.experiment_mode:
                self._record_match()
            else:
                self._save_data()

        elif key == ord('n') and self.experiment_mode:  # N - 下一个
            if self.current_exp_idx < len(self.experiment_points) - 1:
                self.current_exp_idx += 1
                pt = self.experiment_points[self.current_exp_idx]
                self.get_logger().info(f'切换到: {pt.test_id}')
            else:
                self.get_logger().info('已是最后一个测试点')

        elif key == ord('p') and self.experiment_mode:  # P - 上一个
            if self.current_exp_idx > 0:
                self.current_exp_idx -= 1
                pt = self.experiment_points[self.current_exp_idx]
                self.get_logger().info(f'切换到: {pt.test_id}')

    def _main_loop(self):
        """主循环"""
        # 创建可视化
        display = self._create_visualization()

        # 显示
        cv2.imshow(self.window_name, display)

        # 只在首次创建窗口时设置鼠标回调
        if not self._window_initialized:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self._window_initialized = True

        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self._handle_key(key)

        # 发布 markers
        self._publish_markers()


def main():
    rclpy.init()
    node = DualCameraValidator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
