#!/usr/bin/env python3
"""
双相机棋盘格图像采集程序 (ROS2)

功能:
- 同步订阅 Top 和 Chassis 相机的彩色图像
- 键盘空格键触发采集
- 自动保存到各自文件夹
- 采集80组数据后自动停止

使用方法:
    ros2 run perception stereo_chessboard_capture
    或
    python3 scripts/stereo_chessboard_capture.py

按键说明:
    [SPACE] - 保存当前帧 (top + chassis 各一张)
    [q]     - 退出程序
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CameraInfo
import message_filters

# 添加 perception 包路径
sys.path.insert(0, '/home/didi/workspace/MobileManipulator2/install/perception/lib/python3.10/site-packages')
from perception.utils import CvBridgeNumPy2 as CvBridge
from perception.utils import SENSOR_QOS, stamp_to_sec


class StereoChessboardCapture(Node):
    """双相机棋盘格采集节点"""
    
    # 采集目标数量
    TARGET_COUNT = 80
    
    def __init__(self):
        super().__init__('stereo_chessboard_capture')
        
        # 参数声明
        self.declare_parameter('top_topic', '/camera/top/color/image_raw')
        self.declare_parameter('chassis_topic', '/camera/chassis/color/image_raw')
        self.declare_parameter('output_dir', '')  # 空则自动生成
        self.declare_parameter('sync_slop', 0.05)  # 50ms同步容差
        
        # 获取参数
        self.top_topic = self.get_parameter('top_topic').value
        self.chassis_topic = self.get_parameter('chassis_topic').value
        self.sync_slop = self.get_parameter('sync_slop').value
        
        # 创建输出目录
        output_dir = self.get_parameter('output_dir').value
        if not output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'/home/didi/workspace/MobileManipulator2/calibration_data/stereo_chessboard_{timestamp}'
        self.output_dir = output_dir
        
        self.top_dir = os.path.join(self.output_dir, 'top')
        self.chassis_dir = os.path.join(self.output_dir, 'chassis')
        os.makedirs(self.top_dir, exist_ok=True)
        os.makedirs(self.chassis_dir, exist_ok=True)
        
        # 初始化状态
        self.bridge = CvBridge()
        self.capture_count = 0
        self.latest_top_frame: Optional[np.ndarray] = None
        self.latest_chassis_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.running = True
        
        # 设置同步订阅
        self._setup_sync_subscribers()
        
        # 打印启动信息
        self.get_logger().info('=' * 60)
        self.get_logger().info('双相机棋盘格采集程序启动')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Top相机:    {self.top_topic}')
        self.get_logger().info(f'Chassis相机: {self.chassis_topic}')
        self.get_logger().info(f'输出目录:   {self.output_dir}')
        self.get_logger().info(f'同步容差:   {self.sync_slop * 1000:.0f}ms')
        self.get_logger().info('-' * 60)
        self.get_logger().info('按键说明:')
        self.get_logger().info('  [SPACE] - 保存一对图像')
        self.get_logger().info('  [q]     - 退出程序')
        self.get_logger().info('=' * 60)
    
    def _setup_sync_subscribers(self):
        """设置同步订阅器"""
        # 创建订阅器
        self.top_sub = message_filters.Subscriber(
            self, Image, self.top_topic, qos_profile=SENSOR_QOS
        )
        self.chassis_sub = message_filters.Subscriber(
            self, Image, self.chassis_topic, qos_profile=SENSOR_QOS
        )
        
        # 近似时间同步器
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.top_sub, self.chassis_sub],
            queue_size=10,
            slop=self.sync_slop
        )
        self.sync.registerCallback(self._sync_callback)
        
        self.get_logger().info('同步订阅器已启动，等待相机数据...')
    
    def _sync_callback(self, top_msg: Image, chassis_msg: Image):
        """同步回调：接收双相机数据"""
        try:
            # 转换图像
            top_frame = self.bridge.imgmsg_to_cv2(top_msg, 'bgr8')
            chassis_frame = self.bridge.imgmsg_to_cv2(chassis_msg, 'bgr8')
            
            # 计算时间差
            top_stamp = stamp_to_sec(top_msg.header.stamp)
            chassis_stamp = stamp_to_sec(chassis_msg.header.stamp)
            time_diff = abs(top_stamp - chassis_stamp) * 1000  # ms
            
            with self.frame_lock:
                self.latest_top_frame = top_frame.copy()
                self.latest_chassis_frame = chassis_frame.copy()
                self.last_sync_diff = time_diff
                
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
    
    def capture_pair(self) -> bool:
        """保存一对图像"""
        with self.frame_lock:
            if self.latest_top_frame is None or self.latest_chassis_frame is None:
                self.get_logger().warn('暂无同步数据，请等待相机连接...')
                return False
            
            if self.capture_count >= self.TARGET_COUNT:
                self.get_logger().info(f'已达到目标数量 {self.TARGET_COUNT}，停止采集')
                return False
            
            # 增加计数
            self.capture_count += 1
            idx = self.capture_count
            
            # 生成文件名 (001, 002, ...)
            filename = f'{idx:04d}.png'
            top_path = os.path.join(self.top_dir, filename)
            chassis_path = os.path.join(self.chassis_dir, filename)
            
            # 保存图像
            top_saved = cv2.imwrite(top_path, self.latest_top_frame)
            chassis_saved = cv2.imwrite(chassis_path, self.latest_chassis_frame)
            
            if top_saved and chassis_saved:
                self.get_logger().info(
                    f'[{idx}/{self.TARGET_COUNT}] 已保存: {filename} '
                    f'(sync_diff={self.last_sync_diff:.1f}ms)'
                )
                
                # 生成MATLAB脚本（首次保存时）
                if idx == 1:
                    self._generate_matlab_script()
                
                return True
            else:
                self.get_logger().error(f'保存失败: top={top_saved}, chassis={chassis_saved}')
                self.capture_count -= 1  # 回滚计数
                return False
    
    def _generate_matlab_script(self):
        """生成MATLAB加载脚本"""
        script_path = os.path.join(self.output_dir, 'load_for_matlab.m')
        
        script_content = f"""% MATLAB Stereo Camera Calibrator 加载脚本
% 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
%
% 使用方法:
%   1. 在MATLAB中运行此脚本
%   2. 或使用 Stereo Camera Calibrator App 手动加载图像
%

%% 获取图像文件列表
topDir = fullfile(pwd, 'top');
chassisDir = fullfile(pwd, 'chassis');

topImages = dir(fullfile(topDir, '*.png'));
chassisImages = dir(fullfile(chassisDir, '*.png'));

% 按文件名排序
[~, idx] = sort({{topImages.name}});
topImages = topImages(idx);
[~, idx] = sort({{chassisImages.name}});
chassisImages = chassisImages(idx);

% 构建完整路径
topImageFiles = fullfile(topDir, {{topImages.name}});
chassisImageFiles = fullfile(chassisDir, {{chassisImages.name}});

fprintf('找到 %d 对图像\\n', length(topImageFiles));

%% 打开 Stereo Camera Calibrator
% 棋盘格方格大小（根据实际情况修改，单位：米）
squareSize = 0.025;  % 25mm = 0.025m

stereoCameraCalibrator(topImageFiles, chassisImageFiles, ...
    'SquareSize', squareSize);

fprintf('Stereo Camera Calibrator 已启动\\n');
fprintf('提示: 如果检测不到棋盘格，请检查 SquareSize 参数是否正确\\n');
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.get_logger().info(f'MATLAB脚本已生成: {script_path}')
    
    def run_display_loop(self):
        """运行显示和键盘监听循环"""
        self.get_logger().info('启动显示窗口...')
        
        # 创建窗口
        cv2.namedWindow('Stereo Capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stereo Capture', 1280, 480)
        
        while self.running and rclpy.ok():
            # 获取当前帧（带超时保护）
            with self.frame_lock:
                top_frame = self.latest_top_frame.copy() if self.latest_top_frame is not None else None
                chassis_frame = self.latest_chassis_frame.copy() if self.latest_chassis_frame is not None else None
                sync_diff = getattr(self, 'last_sync_diff', 0)
            
            # 创建显示画面
            if top_frame is not None and chassis_frame is not None:
                # 确保尺寸一致
                h, w = 360, 640  # 显示用缩小尺寸
                top_display = cv2.resize(top_frame, (w, h))
                chassis_display = cv2.resize(chassis_frame, (w, h))
                
                # 添加信息文本
                info_text = f'Count: {self.capture_count}/{self.TARGET_COUNT} | Sync: {sync_diff:.1f}ms'
                cv2.putText(top_display, 'TOP', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(chassis_display, 'CHASSIS', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 合并显示
                combined = np.hstack([top_display, chassis_display])
                
                # 底部状态栏
                status_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
                color = (0, 255, 0) if self.capture_count < self.TARGET_COUNT else (0, 165, 255)
                cv2.putText(status_bar, info_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(status_bar, '[SPACE] Capture  [Q] Quit', (500, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                display = np.vstack([combined, status_bar])
            else:
                # 等待画面
                display = np.zeros((420, 1280, 3), dtype=np.uint8)
                cv2.putText(display, 'Waiting for camera data...', (400, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Stereo Capture', display)
            
            # 键盘处理 (10ms超时，允许ROS处理)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord(' '):  # 空格键
                self.capture_pair()
                
            elif key == ord('q'):  # Q键退出
                self.get_logger().info('用户退出')
                self.running = False
                break
            
            # 检查是否完成
            if self.capture_count >= self.TARGET_COUNT:
                self.get_logger().info('采集完成！')
                time.sleep(2)  # 显示完成状态2秒
                self.running = False
                break
        
        cv2.destroyAllWindows()
    
    def print_summary(self):
        """打印采集摘要"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('采集完成摘要')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'输出目录: {self.output_dir}')
        self.get_logger().info(f'采集数量: {self.capture_count} / {self.TARGET_COUNT}')
        self.get_logger().info(f'Top图像:   {self.top_dir}')
        self.get_logger().info(f'Chassis图像: {self.chassis_dir}')
        
        # 检查文件
        top_files = sorted([f for f in os.listdir(self.top_dir) if f.endswith('.png')])
        chassis_files = sorted([f for f in os.listdir(self.chassis_dir) if f.endswith('.png')])
        
        self.get_logger().info(f'实际文件: Top={len(top_files)}, Chassis={len(chassis_files)}')
        
        if len(top_files) == len(chassis_files) and len(top_files) > 0:
            self.get_logger().info('✓ 文件配对检查通过')
        else:
            self.get_logger().warn('✗ 文件数量不匹配！')
        
        matlab_script = os.path.join(self.output_dir, 'load_for_matlab.m')
        if os.path.exists(matlab_script):
            self.get_logger().info(f'✓ MATLAB脚本: {matlab_script}')
        
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    
    node = StereoChessboardCapture()
    
    # 在单独线程中运行ROS2 spin
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()
    
    try:
        # 运行显示和采集循环
        node.run_display_loop()
    except KeyboardInterrupt:
        node.get_logger().info('中断信号收到')
    finally:
        node.running = False
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
