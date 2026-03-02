#!/usr/bin/env python3
"""
双相机棋盘格采集程序 - 直接运行版本

用法:
    export ROS_DOMAIN_ID=42
    python3 scripts/capture_stereo.py

按键:
    [SPACE] - 保存一对图像
    [q]     - 退出
    [a]     - 自动模式(每2秒保存)
"""

import os
import sys
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# 添加perception模块路径
sys.path.insert(0, '/home/didi/workspace/MobileManipulator2/install/perception/lib/python3.10/site-packages')
from perception.utils import CvBridgeNumPy2 as CvBridge


class StereoCapture(Node):
    def __init__(self, auto_mode=False):
        super().__init__('stereo_capture')
        
        self.auto_mode = auto_mode
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/didi/workspace/MobileManipulator2/calibration_data/stereo_{timestamp}'
        self.top_dir = os.path.join(self.output_dir, 'top')
        self.chassis_dir = os.path.join(self.output_dir, 'chassis')
        os.makedirs(self.top_dir, exist_ok=True)
        os.makedirs(self.chassis_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        self.top_frame = None
        self.chassis_frame = None
        self.lock = threading.Lock()
        self.count = 0
        self.target = 200
        self.running = True
        
        # 订阅两个相机
        self.create_subscription(Image, '/camera/top/color/image_raw', self._top_cb, 10)
        self.create_subscription(Image, '/camera/chassis/color/image_raw', self._chassis_cb, 10)
        
        print('='*60)
        print('双相机棋盘格采集程序')
        print('='*60)
        print(f'输出目录: {self.output_dir}')
        print(f'目标数量: {self.target}对')
        if auto_mode:
            print('模式: 自动采集 (每2秒保存一对)')
        else:
            print('按键: [SPACE]保存  [a]自动模式  [q]退出')
        print('='*60)
    
    def _top_cb(self, msg):
        try:
            with self.lock:
                self.top_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            pass
    
    def _chassis_cb(self, msg):
        try:
            with self.lock:
                self.chassis_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            pass
    
    def save_pair(self):
        with self.lock:
            top = self.top_frame.copy() if self.top_frame is not None else None
            chassis = self.chassis_frame.copy() if self.chassis_frame is not None else None
        
        if top is None or chassis is None:
            status = []
            if top is None: status.append('top=等待中')
            if chassis is None: status.append('chassis=等待中')
            print(f'等待相机: {", ".join(status)}')
            return False
        
        if self.count >= self.target:
            print(f'已达到目标数量 {self.target}')
            self.running = False
            return False
        
        self.count += 1
        idx = self.count
        
        top_path = os.path.join(self.top_dir, f'{idx:04d}.png')
        chassis_path = os.path.join(self.chassis_dir, f'{idx:04d}.png')
        
        success = cv2.imwrite(top_path, top) and cv2.imwrite(chassis_path, chassis)
        
        if success:
            print(f'[{idx}/{self.target}] ✓ 已保存: {idx:04d}.png')
        else:
            print(f'[{idx}/{self.target}] ✗ 保存失败')
        
        return success
    
    def run(self):
        if self.auto_mode:
            self._run_auto()
        else:
            self._run_manual()
    
    def _run_auto(self):
        """自动采集模式"""
        print('自动采集模式启动，按Ctrl+C停止...')
        while self.running and rclpy.ok():
            self.save_pair()
            if self.count >= self.target:
                break
            time.sleep(2.0)
        self._print_summary()
    
    def _run_manual(self):
        """手动采集模式 (GUI)"""
        print('启动显示窗口...')
        cv2.namedWindow('Stereo Capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stereo Capture', 1280, 480)
        
        while self.running and rclpy.ok():
            with self.lock:
                top = self.top_frame.copy() if self.top_frame is not None else None
                chassis = self.chassis_frame.copy() if self.chassis_frame is not None else None
            
            # 创建显示画面
            if top is not None and chassis is not None:
                h, w = 360, 640
                top_disp = cv2.resize(top, (w, h))
                chassis_disp = cv2.resize(chassis, (w, h))
                
                cv2.putText(top_disp, 'TOP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(chassis_disp, 'CHASSIS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                combined = np.hstack([top_disp, chassis_disp])
                
                # 状态栏
                status = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
                color = (0, 255, 0) if self.count < self.target else (0, 165, 255)
                text = f'Count: {self.count}/{self.target}'
                cv2.putText(status, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(status, '[SPACE] Capture  [a] Auto  [Q] Quit', (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                display = np.vstack([combined, status])
            else:
                display = np.zeros((420, 1280, 3), dtype=np.uint8)
                status_text = []
                if top is None: status_text.append('Top相机')
                if chassis is None: status_text.append('Chassis相机')
                text = f'等待: {", ".join(status_text)}'
                cv2.putText(display, text, (400, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Stereo Capture', display)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord(' '):
                self.save_pair()
            elif key == ord('a'):
                print('切换到自动模式...')
                self.auto_mode = True
                cv2.destroyAllWindows()
                self._run_auto()
                return
            elif key == ord('q'):
                break
            
            if self.count >= self.target:
                print('采集完成！')
                time.sleep(2)
                break
        
        cv2.destroyAllWindows()
        self._print_summary()
    
    def _print_summary(self):
        """打印采集摘要"""
        print('='*60)
        print('采集完成')
        print('='*60)
        print(f'输出目录: {self.output_dir}')
        print(f'采集数量: {self.count}/{self.target}')
        print(f'Top图像:   {self.top_dir} ({len(os.listdir(self.top_dir))} files)')
        print(f'Chassis图像: {self.chassis_dir} ({len(os.listdir(self.chassis_dir))} files)')
        print('='*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='双相机棋盘格采集')
    parser.add_argument('--auto', '-a', action='store_true', help='自动采集模式')
    args = parser.parse_args()
    
    rclpy.init()
    node = StereoCapture(auto_mode=args.auto)
    
    # ROS spin线程
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()
    
    try:
        node.run()
    except KeyboardInterrupt:
        print('\n用户中断')
    finally:
        node._print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
