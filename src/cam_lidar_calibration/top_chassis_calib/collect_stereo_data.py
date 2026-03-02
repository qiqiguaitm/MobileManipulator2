#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# ================= 配置区域 =================
# 从你提供的文件中提取的序列号
SERIAL_TOP = '318122302992'
SERIAL_CHASSIS = '337122071540'

# 图像配置
WIDTH = 1280
HEIGHT = 720
FPS = 30

# 保存路径
SAVE_ROOT = "./stereo_data_matlab"
FOLDER_TOP = os.path.join(SAVE_ROOT, "top_camera")
FOLDER_CHASSIS = os.path.join(SAVE_ROOT, "chassis_camera")

# 采集目标
TARGET_COUNT = 60
# ===========================================

class RealsenseStereoCollector:
    def __init__(self):
        self.pipeline_top = None
        self.pipeline_chassis = None
        self.saved_count = 0
        self.is_paused = False
        self.frame_top = None
        self.frame_chassis = None

        # 创建目录
        if not os.path.exists(FOLDER_TOP): os.makedirs(FOLDER_TOP)
        if not os.path.exists(FOLDER_CHASSIS): os.makedirs(FOLDER_CHASSIS)

        print(f"[初始化] 正在连接相机...")
        print(f"  - Top Camera: {SERIAL_TOP}")
        print(f"  - Chassis Camera: {SERIAL_CHASSIS}")

        try:
            # --- 配置 Top Camera ---
            self.pipeline_top = rs.pipeline()
            config_top = rs.config()
            config_top.enable_device(SERIAL_TOP)
            config_top.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
            self.pipeline_top.start(config_top)
            print("[成功] Top Camera 已启动")

            # --- 配置 Chassis Camera ---
            self.pipeline_chassis = rs.pipeline()
            config_chassis = rs.config()
            config_chassis.enable_device(SERIAL_CHASSIS)
            config_chassis.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
            self.pipeline_chassis.start(config_chassis)
            print("[成功] Chassis Camera 已启动")

        except RuntimeError as e:
            print(f"\n[错误] 相机启动失败: {e}")
            print("请检查：\n1. USB线是否插好\n2. 是否有其他程序(如ROS)正在占用相机")
            self.cleanup()
            exit(1)

        # 预热相机 (自动曝光稳定)
        print("[预热] 等待自动曝光稳定...")
        for _ in range(30):
            self.pipeline_top.wait_for_frames()
            self.pipeline_chassis.wait_for_frames()

    def get_frames(self):
        """获取最新的一对帧"""
        try:
            # 获取 Top 帧
            frames_top = self.pipeline_top.wait_for_frames()
            color_frame_top = frames_top.get_color_frame()
            
            # 获取 Chassis 帧
            frames_chassis = self.pipeline_chassis.wait_for_frames()
            color_frame_chassis = frames_chassis.get_color_frame()

            if not color_frame_top or not color_frame_chassis:
                return None, None

            # 转为 numpy 数组
            img_top = np.asanyarray(color_frame_top.get_data())
            img_chassis = np.asanyarray(color_frame_chassis.get_data())

            return img_top, img_chassis

        except Exception as e:
            print(f"[警告] 获取图像失败: {e}")
            return None, None

    def draw_ui(self, img_top, img_chassis):
        """绘制预览界面"""
        # 缩放以适应屏幕显示 (不影响保存结果)
        display_h = 480
        scale = display_h / HEIGHT
        display_w = int(WIDTH * scale)

        view_top = cv2.resize(img_top, (display_w, display_h))
        view_chassis = cv2.resize(img_chassis, (display_w, display_h))

        # 添加标签
        cv2.putText(view_top, f"TOP ({SERIAL_TOP})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(view_chassis, f"CHASSIS ({SERIAL_CHASSIS})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 拼接
        combined = np.hstack((view_top, view_chassis))

        # 状态栏
        status_h = 60
        status_bar = np.zeros((status_h, combined.shape[1], 3), dtype=np.uint8)
        
        if self.is_paused:
            color = (0, 0, 200) # 红色背景表示冻结
            text = "CHECK MODE | [S] Save | [D] Discard/Retry"
        else:
            color = (50, 50, 50) # 灰色背景表示预览
            text = f"PREVIEW | Collected: {self.saved_count}/{TARGET_COUNT} | [SPACE] Capture | [Q] Quit"

        cv2.rectangle(status_bar, (0, 0), (combined.shape[1], status_h), color, -1)
        cv2.putText(status_bar, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return np.vstack((combined, status_bar))

    def save_images(self):
        """保存当前帧"""
        self.saved_count += 1
        filename = f"image_{self.saved_count:02d}.jpg"
        
        path_top = os.path.join(FOLDER_TOP, filename)
        path_chassis = os.path.join(FOLDER_CHASSIS, filename)

        cv2.imwrite(path_top, self.frame_top)
        cv2.imwrite(path_chassis, self.frame_chassis)
        print(f"[已保存] {filename}")

    def run(self):
        print("\n" + "="*50)
        print("  Realsense 双目采集工具启动成功")
        print("  操作说明:")
        print("  [空格] : 拍照 (冻结画面)")
        print("  [ S ]  : 保存当前照片")
        print("  [ D ]  : 丢弃/重拍")
        print("  [ Q ]  : 退出")
        print("="*50 + "\n")

        while True:
            # 如果没有暂停，持续更新画面
            if not self.is_paused:
                img_top, img_chassis = self.get_frames()
                if img_top is not None:
                    self.frame_top = img_top
                    self.frame_chassis = img_chassis

            if self.frame_top is None:
                continue

            # 绘制界面
            ui = self.draw_ui(self.frame_top, self.frame_chassis)
            cv2.imshow("Stereo Collection", ui)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break
            
            elif key == 32: # Space
                if not self.is_paused:
                    self.is_paused = True
                    print("\n[冻结] 请检查画面清晰度与同步情况...")

            elif key == ord('s'):
                if self.is_paused:
                    self.save_images()
                    self.is_paused = False
                    print("[恢复] 继续预览...")
            
            elif key == ord('d'):
                if self.is_paused:
                    print("[丢弃] 已取消")
                    self.is_paused = False
                    print("[恢复] 继续预览...")

        self.cleanup()

    def cleanup(self):
        print("[退出] 正在关闭相机...")
        if self.pipeline_top: self.pipeline_top.stop()
        if self.pipeline_chassis: self.pipeline_chassis.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = RealsenseStereoCollector()
    collector.run()
