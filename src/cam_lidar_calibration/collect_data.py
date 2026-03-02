#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标定数据自动化采集工具
支持 Top Camera 和 Chassis Camera 与 LiDAR 的联合标定数据采集
增加可视化预览和人工确认功能
"""

import os
import sys
import subprocess
import time
import datetime
import argparse
import signal
import threading
import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import font as tkfont
    HAS_TK = True
except ImportError:
    HAS_TK = False

# 质量检查依赖
try:
    import rclpy
    from sensor_msgs.msg import CameraInfo
    from lib.realtime_checker import RealtimeChecker
    HAS_QUALITY_CHECK = True
except ImportError:
    HAS_QUALITY_CHECK = False


# 相机配置
CAMERA_CONFIG = {
    'top': {
        'name': 'Top Camera',
        'image_topic': '/camera/top/color/image_raw',
        'camera_info_topic': '/camera/top/color/camera_info',
        'serial': '318122302992',
        'description': '顶部相机 (俯视视角)',
        'launch_arg': 'top_enable:=true enable_camera_tf:=false',
        'node_prefix': 'camera/top',
    },
    'chassis': {
        'name': 'Chassis Camera',
        'image_topic': '/camera/chassis/color/image_raw',
        'camera_info_topic': '/camera/chassis/color/camera_info',
        'serial': '337122071540',
        'description': '底盘相机 (前向视角)',
        'launch_arg': 'chassis_enable:=true enable_camera_tf:=false',
        'node_prefix': 'camera/chassis',
    }
}

# LiDAR 配置
LIDAR_TOPIC = '/rslidar_points'
LIDAR_NODES = ['rslidar_sdk_node', 'pointcloud_to_laserscan']

# 9 组标准采集姿态 (根据 guidance_to_collect_data.md)
CALIBRATION_POSES = [
    # === 近距离层 1.5m ===
    {
        'id': 1,
        'name': '1.5m_左边缘_前俯15°_右转30°',
        'distance': '1.5m',
        'position': '左边缘',
        'pitch': '前俯 15°',
        'yaw': '向右转 30°',
        'purpose': '近距离斜切线',
        'warning': '边缘位置留10%空白',
    },
    {
        'id': 2,
        'name': '1.5m_中心_后仰20°',
        'distance': '1.5m',
        'position': '中心',
        'pitch': '后仰 20°',
        'yaw': '正对相机',
        'purpose': '近距离 Pitch 约束',
    },
    {
        'id': 3,
        'name': '1.5m_右边缘_前俯15°_左转30°',
        'distance': '1.5m',
        'position': '右边缘',
        'pitch': '前俯 15°',
        'yaw': '向左转 30°',
        'purpose': '对称斜切线',
        'warning': '边缘位置留10%空白',
    },
    # === 中距离层 2.0m ===
    {
        'id': 4,
        'name': '2.0m_左侧_后仰15°_右转20°',
        'distance': '2.0m',
        'position': '左侧',
        'pitch': '后仰 15°',
        'yaw': '向右转 20°',
        'purpose': '复合约束',
    },
    {
        'id': 5,
        'name': '2.0m_中心_垂直',
        'distance': '2.0m',
        'position': '中心',
        'pitch': '垂直',
        'yaw': '正对相机',
        'purpose': '基准参考 (Baseline)',
        'warning': '微倾2-3°，确保≥5条扫描线',
    },
    {
        'id': 6,
        'name': '2.0m_右侧_后仰15°_左转20°',
        'distance': '2.0m',
        'position': '右侧',
        'pitch': '后仰 15°',
        'yaw': '向左转 20°',
        'purpose': '对称复合约束',
    },
    # === 远距离层 2.5m ===
    {
        'id': 7,
        'name': '2.5m_左侧_前俯15°_右转30°',
        'distance': '2.5m',
        'position': '左侧',
        'pitch': '前俯 15°',
        'yaw': '向右转 30°',
        'purpose': '远距离斜切线',
    },
    {
        'id': 8,
        'name': '2.5m_中心_后仰15°',
        'distance': '2.5m',
        'position': '中心',
        'pitch': '后仰 15°',
        'yaw': '正对相机',
        'purpose': '远距离稳定性检查',
    },
    {
        'id': 9,
        'name': '2.5m_右侧_前俯15°_左转30°',
        'distance': '2.5m',
        'position': '右侧',
        'pitch': '前俯 15°',
        'yaw': '向左转 30°',
        'purpose': '对称验证',
    },
]

# 全局变量存储后台进程
_background_processes = []
_preview_processes = []
_info_window = None
_node = None


def get_camera_intrinsics(camera_info_topic, timeout=10.0):
    """
    从 camera_info topic 读取相机内参

    Args:
        camera_info_topic: camera_info 话题名称
        timeout: 超时时间(秒)

    Returns:
        (K, D, width, height): 内参矩阵、畸变系数和图像尺寸，失败返回 (None, None, None, None)
    """
    if not HAS_QUALITY_CHECK:
        return None, None, None, None

    print("  [内参获取] 从话题读取: %s" % camera_info_topic)

    camera_info = [None]  # 使用列表来在闭包中修改

    def callback(msg):
        camera_info[0] = msg

    sub = _node.create_subscription(CameraInfo, camera_info_topic, callback, 10)

    start_time = time.time()
    while camera_info[0] is None and time.time() - start_time < timeout:
        rclpy.spin_once(_node, timeout_sec=0.1)

    _node.destroy_subscription(sub)

    if camera_info[0] is None:
        print("  [错误] 超时未收到内参数据")
        return None, None, None, None

    msg = camera_info[0]
    K = np.array(msg.k).reshape(3, 3)
    D = np.array(msg.d)
    width = msg.width
    height = msg.height

    print("  [成功] 已获取内参")
    print("    分辨率: %dx%d" % (width, height))
    print("    焦距: fx=%.1f, fy=%.1f" % (K[0,0], K[1,1]))
    print("    主点: cx=%.1f, cy=%.1f" % (K[0,2], K[1,2]))
    print("    畸变: %d 个系数" % len(D))

    return K, D, width, height


def save_camera_info(save_dir, camera_type, cam_cfg, K, D, pattern_size, square_size, image_width, image_height):
    """
    保存相机配置和内参到 YAML 文件

    Args:
        save_dir: 输出目录
        camera_type: 相机类型 (top/chassis)
        cam_cfg: 相机配置字典
        K: 内参矩阵 3x3
        D: 畸变系数
        pattern_size: 棋盘格尺寸 (cols, rows)
        square_size: 格子边长(米)
        image_width: 图像宽度
        image_height: 图像高度
    """
    import datetime

    # 构造文件路径
    yaml_file = os.path.join(save_dir, "camera_intrinsics_%s.yaml" % camera_type)

    # 准备数据
    data = {
        'camera_info': {
            'name': cam_cfg['name'],
            'type': camera_type,
            'description': cam_cfg['description'],
            'serial_number': cam_cfg['serial'],
            'image_topic': cam_cfg['image_topic'],
            'camera_info_topic': cam_cfg['camera_info_topic'],
        },
        'image_size': {
            'width': int(image_width),
            'height': int(image_height),
        },
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': K.flatten().tolist(),
        },
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(D),
            'data': D.tolist(),
        },
        'calibration_pattern': {
            'type': 'chessboard',
            'pattern_size': {
                'cols': pattern_size[0],
                'rows': pattern_size[1],
            },
            'square_size': float(square_size),
            'unit': 'meters',
        },
        'metadata': {
            'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'collect_data.py',
        }
    }

    # 写入 YAML 文件
    try:
        with open(yaml_file, 'w') as f:
            f.write("# Camera Intrinsics and Configuration\n")
            f.write("# Generated by collect_data.py\n")
            f.write("# Date: %s\n\n" % data['metadata']['saved_at'])

            # 手动格式化 YAML（简单实现）
            f.write("camera_info:\n")
            f.write("  name: %s\n" % data['camera_info']['name'])
            f.write("  type: %s\n" % data['camera_info']['type'])
            f.write("  description: %s\n" % data['camera_info']['description'])
            f.write("  serial_number: '%s'\n" % data['camera_info']['serial_number'])
            f.write("  image_topic: %s\n" % data['camera_info']['image_topic'])
            f.write("  camera_info_topic: %s\n\n" % data['camera_info']['camera_info_topic'])

            f.write("image_size:\n")
            f.write("  width: %d\n" % data['image_size']['width'])
            f.write("  height: %d\n\n" % data['image_size']['height'])

            f.write("camera_matrix:\n")
            f.write("  rows: 3\n")
            f.write("  cols: 3\n")
            f.write("  data: [")
            for i, val in enumerate(data['camera_matrix']['data']):
                if i > 0:
                    f.write(", ")
                f.write("%.6f" % val)
            f.write("]\n\n")

            f.write("distortion_coefficients:\n")
            f.write("  rows: 1\n")
            f.write("  cols: %d\n" % data['distortion_coefficients']['cols'])
            f.write("  data: [")
            for i, val in enumerate(data['distortion_coefficients']['data']):
                if i > 0:
                    f.write(", ")
                f.write("%.6f" % val)
            f.write("]\n\n")

            f.write("calibration_pattern:\n")
            f.write("  type: chessboard\n")
            f.write("  pattern_size:\n")
            f.write("    cols: %d\n" % data['calibration_pattern']['pattern_size']['cols'])
            f.write("    rows: %d\n" % data['calibration_pattern']['pattern_size']['rows'])
            f.write("  square_size: %.4f\n" % data['calibration_pattern']['square_size'])
            f.write("  unit: meters\n\n")

            f.write("metadata:\n")
            f.write("  saved_at: '%s'\n" % data['metadata']['saved_at'])
            f.write("  source: collect_data.py\n")

        print("  [OK] 相机内参已保存: %s" % yaml_file)
        return True
    except Exception as e:
        print("  [警告] 保存内参失败: %s" % str(e))
        return False


class InfoWindow:
    """提示信息窗口 - 显示在 RViz 旁边"""

    def __init__(self):
        self.root = None
        self.pose_label = None
        self.checklist_frame = None
        self.thread = None
        self.running = False

    def start(self):
        """在独立线程中启动窗口"""
        if not HAS_TK:
            print("  [警告] tkinter 不可用，提示窗口已禁用")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.5)  # 等待窗口创建

    def _run(self):
        """窗口主循环"""
        self.root = tk.Tk()
        self.root.title("标定采集助手")
        self.root.geometry("420x650+50+50")
        self.root.configure(bg='#2b2b2b')
        self.root.attributes('-topmost', True)

        # 标题
        title = tk.Label(
            self.root,
            text="标定数据采集",
            font=('Arial', 16, 'bold'),
            fg='#61afef',
            bg='#2b2b2b'
        )
        title.pack(pady=(15, 10))

        # 姿态信息框
        pose_frame = tk.LabelFrame(
            self.root,
            text=" 当前目标姿态 ",
            font=('Arial', 11, 'bold'),
            fg='#98c379',
            bg='#2b2b2b',
            padx=10,
            pady=10
        )
        pose_frame.pack(fill='x', padx=15, pady=10)

        self.pose_label = tk.Label(
            pose_frame,
            text="等待开始...",
            font=('Consolas', 10),
            fg='#abb2bf',
            bg='#2b2b2b',
            justify='left',
            anchor='w'
        )
        self.pose_label.pack(fill='x')

        # 检查清单框
        self.checklist_frame = tk.LabelFrame(
            self.root,
            text=" 数据质量检查清单 ",
            font=('Arial', 11, 'bold'),
            fg='#e5c07b',
            bg='#2b2b2b',
            padx=10,
            pady=10
        )
        self.checklist_frame.pack(fill='x', padx=15, pady=10)

        checklist_text = """【图像检查 - 左侧面板】
  □ 棋盘格四周白边完整可见
  □ 图像清晰，无运动模糊
  □ 光线均匀，无强反光

【点云检查 - 3D视图】
  □ 在RViz中找到标定板位置的点云
  □ 数一下有几条水平扫描线(需≥5条)
  □ 点云清晰，与背景有分离

提示: 用鼠标拖动旋转3D视角
     从侧面看更容易数扫描线"""

        checklist_label = tk.Label(
            self.checklist_frame,
            text=checklist_text,
            font=('Consolas', 10),
            fg='#abb2bf',
            bg='#2b2b2b',
            justify='left',
            anchor='w'
        )
        checklist_label.pack(fill='x')

        # 操作提示框
        ops_frame = tk.LabelFrame(
            self.root,
            text=" 操作指南 ",
            font=('Arial', 11, 'bold'),
            fg='#c678dd',
            bg='#2b2b2b',
            padx=10,
            pady=10
        )
        ops_frame.pack(fill='x', padx=15, pady=10)

        ops_text = """在终端中输入:
  y - 确认录制
  n - 重新调整标定板
  s - 跳过当前姿态
  q - 退出采集"""

        ops_label = tk.Label(
            ops_frame,
            text=ops_text,
            font=('Consolas', 10),
            fg='#abb2bf',
            bg='#2b2b2b',
            justify='left',
            anchor='w'
        )
        ops_label.pack(fill='x')

        # 状态栏
        self.status_label = tk.Label(
            self.root,
            text="状态: 准备就绪",
            font=('Arial', 10),
            fg='#56b6c2',
            bg='#2b2b2b'
        )
        self.status_label.pack(pady=(15, 10))

        # 运行主循环
        while self.running:
            try:
                self.root.update()
                time.sleep(0.05)
            except:
                break

    def update_pose(self, pose):
        """更新姿态信息"""
        if not self.root or not self.pose_label:
            return

        pose_text = """姿态 #{id}: {name}

  距离:     {distance}
  水平位置: {position}
  俯仰角:   {pitch}
  偏航角:   {yaw}

  目的: {purpose}""".format(**pose)

        # 添加警告提示（如果有）
        if 'warning' in pose:
            pose_text += "\n\n  ⚠️ 注意: %s" % pose['warning']

        try:
            self.pose_label.config(text=pose_text)
            self.status_label.config(text="状态: 请检查预览窗口中的数据质量")
        except:
            pass

    def update_status(self, status):
        """更新状态"""
        if not self.root or not self.status_label:
            return
        try:
            self.status_label.config(text="状态: %s" % status)
        except:
            pass

    def stop(self):
        """停止窗口"""
        self.running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号，清理后台进程"""
    global _info_window
    print("\n[中断] 收到退出信号，正在清理...")
    if _info_window:
        _info_window.stop()
    cleanup_preview()
    cleanup_background_processes()
    sys.exit(0)


def cleanup_background_processes():
    """清理所有后台启动的进程"""
    global _background_processes
    for proc in _background_processes:
        if proc.poll() is None:
            print("[清理] 终止后台进程 PID=%d" % proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    _background_processes = []


def cleanup_preview():
    """清理预览进程"""
    global _preview_processes
    for proc in _preview_processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
    _preview_processes = []


def get_running_nodes():
    """获取当前运行的 ROS 节点列表"""
    try:
        result = subprocess.check_output(
            ['ros2', 'node', 'list'], stderr=subprocess.DEVNULL, timeout=5
        )
        return result.decode('utf-8').strip().split('\n')
    except Exception:
        return []


def kill_node(node_name):
    """优雅地杀死指定的 ROS 节点"""
    try:
        subprocess.call(
            ['pkill', '-15', '-f', node_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
        )
        return True
    except Exception:
        return False


def cleanup_environment(camera_type):
    """清理环境：停止冲突的相机和 LiDAR 节点"""
    print("[环境清理] 检查并停止冲突节点...")

    running_nodes = get_running_nodes()
    if not running_nodes or running_nodes == ['']:
        print("  [INFO] 没有检测到运行中的节点")
        return True

    killed_count = 0
    cam_cfg = CAMERA_CONFIG[camera_type]
    node_prefix = cam_cfg['node_prefix']

    # 查找并停止相机节点
    for node in running_nodes:
        if node_prefix in node:
            print("  [KILL] 停止相机节点: %s" % node)
            kill_node(node)
            killed_count += 1

    # 查找并停止 LiDAR 节点
    for node in running_nodes:
        for lidar_node in LIDAR_NODES:
            if lidar_node in node:
                print("  [KILL] 停止 LiDAR 节点: %s" % node)
                kill_node(node)
                killed_count += 1

    if killed_count > 0:
        print("  [等待] 等待节点完全停止...")
        time.sleep(3)
        print("  [OK] 已停止 %d 个节点" % killed_count)
    else:
        print("  [OK] 没有需要停止的冲突节点")

    return True


def start_drivers(camera_type):
    """启动相机和 LiDAR 驱动"""
    global _background_processes
    cam_cfg = CAMERA_CONFIG[camera_type]

    print("\n[驱动启动] 启动传感器驱动...")

    # 启动 LiDAR 驱动
    print("  [启动] LiDAR 驱动 (rslidar_sdk)...")
    lidar_cmd = "ros2 launch rslidar_sdk start.py"
    lidar_proc = subprocess.Popen(
        lidar_cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _background_processes.append(lidar_proc)

    # 启动相机驱动
    print("  [启动] 相机驱动 (%s)..." % cam_cfg['name'])
    camera_cmd = "ros2 launch camera_driver camera_driver.launch.py %s" % cam_cfg['launch_arg']
    camera_proc = subprocess.Popen(
        camera_cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _background_processes.append(camera_proc)

    print("  [等待] 等待驱动初始化 (10秒)...")
    time.sleep(10)

    return True


def check_topic_available(topic, timeout=5):
    """检查 ROS topic 是否可用"""
    try:
        result = subprocess.check_output(
            ['ros2', 'topic', 'list'], stderr=subprocess.STDOUT, timeout=timeout
        )
        topics = result.decode('utf-8').strip().split('\n')
        return topic in topics
    except Exception:
        return False


def wait_for_topics(image_topic, lidar_topic, max_wait=30):
    """等待 topics 可用"""
    print("\n[话题检查] 等待传感器话题可用...")

    start_time = time.time()
    img_ok = False
    lidar_ok = False

    while time.time() - start_time < max_wait:
        if not img_ok:
            img_ok = check_topic_available(image_topic)
            if img_ok:
                print("  [OK] 图像话题可用: %s" % image_topic)

        if not lidar_ok:
            lidar_ok = check_topic_available(lidar_topic)
            if lidar_ok:
                print("  [OK] 点云话题可用: %s" % lidar_topic)

        if img_ok and lidar_ok:
            return True

        time.sleep(1)
        remaining = int(max_wait - (time.time() - start_time))
        print("\r  [等待] 剩余 %d 秒..." % remaining, end='')
        sys.stdout.flush()

    print("")
    if not img_ok:
        print("  [错误] 图像话题超时: %s" % image_topic)
    if not lidar_ok:
        print("  [错误] 点云话题超时: %s" % lidar_topic)

    return False


def start_combined_preview(image_topic):
    """启动组合预览（图像 + 点云 在同一个 RViz 窗口）"""
    global _preview_processes

    print("\n[预览] 启动可视化窗口 (RViz)...")

    # 创建 RViz 配置文件，同时显示图像和点云
    rviz_config = """Panels:
  - Class: rviz_default_plugins/Displays
    Name: Displays
  - Class: rviz_default_plugins/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Plane Cell Count: 20
      Reference Frame: <Fixed Frame>
      Enabled: true
    - Class: rviz_default_plugins/PointCloud2
      Name: LiDAR PointCloud
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /lidar/chassis/point_cloud
      Size (m): 0.05
      Size (Pixels): 3
      Style: Points
      Color Transformer: AxisColor
      Axis: Z
      Enabled: true
      Decay Time: 0
      Value: true
    - Class: rviz_default_plugins/Image
      Name: Camera Image
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: {image_topic}
      Enabled: true
      Value: true
  Global Options:
    Fixed Frame: rslidar
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 5
      Pitch: 0.5
      Yaw: 3.14
      Focal Point:
        X: 0
        Y: 0
        Z: 0
Window Geometry:
  Displays:
    collapsed: false
  Height: 800
  Width: 1200
  X: 100
  Y: 100
  Camera Image:
    collapsed: false
""".format(image_topic=image_topic)

    config_path = "/tmp/calib_preview.rviz"
    with open(config_path, 'w') as f:
        f.write(rviz_config)

    cmd = "ros2 run rviz2 rviz2 -d %s" % config_path
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _preview_processes.append(proc)

    time.sleep(3)  # 等待窗口打开
    print("  [OK] RViz 预览窗口已启动 (图像 + 点云)")
    print("  [提示] 请在 RViz 中调整视角，确保能同时看到图像和点云")

    return proc


def print_pose_info(pose):
    """打印姿态信息"""
    print("")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ 姿态 #%d: %s" % (pose['id'], pose['name']))
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │ 距离:     %s" % pose['distance'])
    print("  │ 水平位置: %s" % pose['position'])
    print("  │ 俯仰角:   %s" % pose['pitch'])
    print("  │ 偏航角:   %s" % pose['yaw'])
    print("  │ 目的:     %s" % pose['purpose'])
    print("  └─────────────────────────────────────────────────────┘")


def print_checklist():
    """打印检查清单"""
    print("")
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║              数 据 质 量 检 查 清 单                  ║")
    print("  ╠═══════════════════════════════════════════════════════╣")
    print("  ║  [图像检查]                                           ║")
    print("  ║    □ 棋盘格四周白边完整可见（不能切边）               ║")
    print("  ║    □ 图像清晰，无运动模糊                             ║")
    print("  ║    □ 光线均匀，无强反光                               ║")
    print("  ║                                                       ║")
    print("  ║  [点云检查]                                           ║")
    print("  ║    □ 标定板上有 5 根以上的雷达扫描线                  ║")
    print("  ║    □ 扫描线清晰，无杂点干扰                           ║")
    print("  ║    □ 标定板与背景有明显分离                           ║")
    print("  ╚═══════════════════════════════════════════════════════╝")


def quality_check_with_confirmation(checker, timeout=60):
    """
    显示实时质量检查并等待用户确认

    Args:
        checker: RealtimeChecker 实例
        timeout: 超时时间(秒)

    Returns:
        'record': 质量合格，继续录制
        'retry': 重新调整标定板
        'skip': 跳过当前姿态
        'quit': 退出采集
    """
    print("\n  [质量检查] 实时检测重投影误差...")
    print("  按键说明：")
    print("    q - 确认质量良好，继续录制")
    print("    r - 重新调整标定板位置")
    print("    s - 跳过当前姿态")
    print("    ESC - 退出采集")

    start_time = time.time()
    last_result = None

    while time.time() - start_time < timeout:
        result = checker.show_live_preview("实时质量检测 - 按q确认/r重调/s跳过")

        if result and result['success']:
            last_result = result

            # 实时打印质量（清除上一行）
            if result['rms'] > 3.0:
                status = "\r  [质量] RMS=%.2fpx (差) 倾斜=%.1f° - 建议重新调整" % (result['rms'], result['angle_deg'])
            elif result['rms'] > 1.5:
                status = "\r  [质量] RMS=%.2fpx (中等) 倾斜=%.1f° - 可接受" % (result['rms'], result['angle_deg'])
            else:
                status = "\r  [质量] RMS=%.2fpx (良好) 倾斜=%.1f° - 可录制" % (result['rms'], result['angle_deg'])

            print(status, end='')
            sys.stdout.flush()

        # 检查按键
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            if last_result and last_result['success']:
                if not last_result['acceptable']:
                    # 质量差时二次确认
                    print("\n\n  [警告] 当前质量较差（RMS=%.2fpx），确定要使用吗？" % last_result['rms'])
                    confirm = input("  继续录制？[y/n]: ")
                    if confirm.strip().lower() != 'y':
                        return 'retry'
                print("")  # 换行
                return 'record'
            else:
                print("\n  [错误] 未检测到棋盘格")
                return 'retry'

        elif key == ord('r') or key == ord('R'):
            cv2.destroyAllWindows()
            print("\n  [重新调整] 请调整标定板位置...")
            return 'retry'

        elif key == ord('s') or key == ord('S'):
            cv2.destroyAllWindows()
            print("\n  [跳过] 跳过当前姿态")
            return 'skip'

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            print("\n  [退出] 用户取消采集")
            return 'quit'

    cv2.destroyAllWindows()
    print("\n  [超时] 质量检查超时，请重新调整")
    return 'retry'


def user_confirm_preview():
    """用户确认预览数据质量"""
    print("")
    print("  请检查预览窗口中的数据质量...")
    print("")

    while True:
        response = input("  数据质量是否满足要求? [y=录制/n=重新调整/s=跳过/q=退出]: ")

        response = response.strip().lower()

        if response == 'y':
            return 'record'
        elif response == 'n':
            return 'retry'
        elif response == 's':
            return 'skip'
        elif response == 'q':
            return 'quit'
        else:
            print("  请输入 y/n/s/q")


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description='相机-LiDAR 联合标定数据采集工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
9 组标准采集姿态 (参考 guidance_to_collect_data.md):
  1. 1.5m 左边缘, 向右转30°     - 约束 X 轴和 Yaw
  2. 1.5m 中心, 后仰20°         - 约束 Z 轴和 Pitch
  3. 1.5m 右边缘, 向左转30°     - 约束 X 轴对称性
  4. 2.0m 左侧, 后仰20°+右转20° - 复合约束 (最关键)
  5. 2.0m 中心, 前俯15°         - 强制雷达产生斜切线
  6. 2.0m 右侧, 后仰20°+左转20° - 复合约束
  7. 2.5m 中心, 后仰15°         - 远距离稳定性
  8. 2.5m 左侧, 向右转30°       - 远距离畸变校正
  9. 1.8m 中心, 垂直            - 平移量基准

示例:
  python collect_data.py top              # 采集 Top Camera (自动启动驱动+预览+质量检查)
  python collect_data.py chassis          # 采集 Chassis Camera
  python collect_data.py top --no-preview # 跳过RViz预览（仍启用质量检查）
  python collect_data.py top --no-quality-check  # 禁用质量检查（回退到人工确认）
  python collect_data.py top --no-launch  # 跳过驱动启动 (驱动已手动启动)
  python collect_data.py top --pattern 8x6 --square 0.05  # 自定义棋盘格
        '''
    )
    parser.add_argument(
        'camera',
        choices=['top', 'chassis'],
        help='选择相机类型: top (顶部相机) 或 chassis (底盘相机)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=2,
        help='每个位置的录制时长(秒), 默认 2 秒'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出目录, 默认为 ./data/calib_bags_<camera>'
    )
    parser.add_argument(
        '--no-launch',
        action='store_true',
        help='跳过驱动启动 (假设驱动已在运行)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='跳过环境清理 (不停止现有节点)'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='跳过可视化预览 (直接录制)'
    )
    parser.add_argument(
        '--no-quality-check',
        action='store_true',
        help='禁用实时质量检查（回退到人工确认）'
    )
    parser.add_argument(
        '--quality-timeout',
        type=int,
        default=60,
        help='质量检查超时时间(秒), 默认 60 秒'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='7x6',
        help='棋盘格尺寸 (默认: 7x6)'
    )
    parser.add_argument(
        '--square',
        type=float,
        default=0.072,
        help='棋盘格格子边长(米) (默认: 0.072)'
    )

    args = parser.parse_args()

    # 获取相机配置
    cam_cfg = CAMERA_CONFIG[args.camera]
    image_topic = cam_cfg['image_topic']
    topics = "%s %s" % (image_topic, LIDAR_TOPIC)

    # 设置输出目录（默认带时间戳，避免覆盖历史数据）
    if args.output:
        save_dir = args.output
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = "./data/calib_bags_%s_%s" % (args.camera, timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 打印配置信息
    print("=" * 65)
    print("        标定数据自动化采集助手 (9组标准姿态)")
    print("=" * 65)
    print("")
    print("[相机配置]")
    print("  名称:     %s" % cam_cfg['name'])
    print("  描述:     %s" % cam_cfg['description'])
    print("  序列号:   %s" % cam_cfg['serial'])
    print("  图像话题: %s" % image_topic)
    print("")
    print("[LiDAR配置]")
    print("  点云话题: %s" % LIDAR_TOPIC)
    print("")
    print("[采集参数]")
    print("  录制时长: %d 秒/位置" % args.duration)
    print("  输出目录: %s" % save_dir)
    print("  自动启动: %s" % ("否" if args.no_launch else "是"))
    print("  可视化:   %s" % ("否" if args.no_preview else "是"))
    print("  质量检查: %s" % ("禁用" if args.no_quality_check else "启用"))
    print("  棋盘格:   %s (格子 %.3fm)" % (args.pattern, args.square))
    print("")

    # 环境清理
    if not args.no_cleanup and not args.no_launch:
        cleanup_environment(args.camera)

    # 启动驱动
    if not args.no_launch:
        if not start_drivers(args.camera):
            print("[错误] 驱动启动失败")
            cleanup_background_processes()
            sys.exit(1)

    # 等待话题可用
    if not wait_for_topics(image_topic, LIDAR_TOPIC, max_wait=30):
        print("\n[错误] 传感器话题不可用，请检查驱动状态")
        if not args.no_launch:
            cleanup_background_processes()
        sys.exit(1)

    # 启动可视化预览
    if not args.no_preview:
        start_combined_preview(image_topic)

        # 启动提示信息窗口
        global _info_window
        _info_window = InfoWindow()
        _info_window.start()

    # 初始化质量检查器
    checker = None
    if not args.no_quality_check:
        if not HAS_QUALITY_CHECK:
            print("\n[警告] 缺少质量检查依赖（rclpy/cv_bridge），已禁用质量检查")
            print("  请安装: sudo apt install ros-humble-cv-bridge")
        else:
            print("\n[质量检查] 初始化实时质量检查器...")

            # 初始化 ROS 节点（如果尚未初始化）
            global _node
            try:
                rclpy.init()
            except RuntimeError:
                pass  # rclpy 已初始化
            if _node is None:
                _node = rclpy.create_node('calib_data_collector')

            # 获取内参
            camera_info_topic = cam_cfg['camera_info_topic']
            K, D, img_width, img_height = get_camera_intrinsics(camera_info_topic, timeout=10.0)

            if K is None or D is None:
                print("  [警告] 无法获取相机内参，已禁用质量检查")
                print("  请检查相机是否正常: ros2 topic echo %s" % camera_info_topic)
                checker = None
            else:
                # 解析棋盘格参数
                pattern_size = tuple(map(int, args.pattern.split('x')))

                checker = RealtimeChecker(
                    K=K, D=D,
                    pattern_size=pattern_size,
                    square_size=args.square
                )

                # 启动订阅
                checker.start_subscriber(image_topic, _node)

                print("  [OK] 质量检查器已就绪")
                print("  棋盘格: %dx%d, 格子边长=%.3fm" % (pattern_size[0], pattern_size[1], args.square))

                # 等待数据
                for _ in range(20):
                    rclpy.spin_once(_node, timeout_sec=0.1)

                if checker.latest_image is None:
                    print("  [警告] 未接收到图像数据，已禁用质量检查")
                    checker = None
                else:
                    # 保存相机内参和配置信息
                    save_camera_info(
                        save_dir=save_dir,
                        camera_type=args.camera,
                        cam_cfg=cam_cfg,
                        K=K,
                        D=D,
                        pattern_size=pattern_size,
                        square_size=args.square,
                        image_width=img_width,
                        image_height=img_height
                    )

    print("")
    print("-" * 65)
    print("准备采集 %d 组标准姿态数据" % len(CALIBRATION_POSES))
    print("请参考 guidance_to_collect_data.md 进行标定板摆放")
    print("-" * 65)

    input("\n按 [Enter] 键开始采集流程...")

    recorded_count = 0
    skipped_poses = []

    try:
        for pose in CALIBRATION_POSES:
            print("\n" + "=" * 65)
            print("步骤 [%d/%d]: %s" % (pose['id'], len(CALIBRATION_POSES), pose['name']))
            print("=" * 65)

            # 更新提示窗口
            if _info_window:
                _info_window.update_pose(pose)

            retry = True
            while retry:
                input("\n摆放完成后按 [Enter] 检查预览...")

                # 数据质量确认
                if _info_window:
                    _info_window.update_status("等待确认...")

                # 选择确认方式（优先级：质量检查 > 人工确认 > 直接录制）
                if checker is not None:
                    # 方案A: 使用实时质量检查
                    action = quality_check_with_confirmation(checker, timeout=args.quality_timeout)
                elif not args.no_preview:
                    # 方案B: 回退到原始人工确认
                    action = user_confirm_preview()
                else:
                    # 方案C: 无预览模式，直接录制
                    action = 'record'

                # 处理用户决策
                if action == 'quit':
                    print("\n[退出] 用户取消采集")
                    raise KeyboardInterrupt

                if action == 'skip':
                    print("  [跳过] 姿态 #%d 已跳过" % pose['id'])
                    skipped_poses.append(pose['id'])
                    retry = False
                    continue

                if action == 'retry':
                    print("  [重试] 请重新调整标定板位置...")
                    if _info_window:
                        _info_window.update_status("请重新调整...")
                    continue

                # action == 'record'
                retry = False

                # 倒计时
                if _info_window:
                    _info_window.update_status("录制倒计时...")

                print("\n  [准备录制] 请保持静止，消除支架共振...")
                for count in range(5, 0, -1):
                    print("\r  [倒计时] %d 秒..." % count, end='')
                    sys.stdout.flush()
                    time.sleep(1)
                print("\r" + " " * 40 + "\r", end='')

                # 录制数据
                if _info_window:
                    _info_window.update_status("正在录制...")

                bag_name = os.path.join(save_dir, "%s_pose_%d" % (args.camera, pose['id']))
                print("  [录制中] %s" % bag_name)

                cmd = "timeout %d ros2 bag record -o %s %s" % (args.duration, bag_name, topics)
                subprocess.call(cmd, shell=True)

                print("  [完成] 姿态 #%d 录制完成" % pose['id'])
                recorded_count += 1

                if _info_window:
                    _info_window.update_status("录制完成 (%d/%d)" % (recorded_count, len(CALIBRATION_POSES)))

    except KeyboardInterrupt:
        print("\n[中断] 用户取消采集")
    finally:
        # 清理提示窗口
        if _info_window:
            _info_window.stop()
            _info_window = None

        # 清理预览和后台进程
        print("\n[清理] 关闭预览窗口...")
        cleanup_preview()

        if not args.no_launch:
            print("[清理] 停止后台驱动...")
            cleanup_background_processes()

    # 汇总
    print("\n" + "=" * 65)
    print("采集完毕！")
    print("=" * 65)
    print("")
    print("[统计]")
    print("  成功录制: %d / %d 组" % (recorded_count, len(CALIBRATION_POSES)))
    if skipped_poses:
        print("  跳过姿态: %s" % ', '.join(['#%d' % p for p in skipped_poses]))
    print("")
    print("[数据位置]")
    print("  %s" % save_dir)
    print("")

    if recorded_count >= 5:
        print("[下一步]")
        print("  python extract_data.py %s" % args.camera)
    else:
        print("[警告] 录制数量不足 5 组，建议补充更多数据以保证标定精度")

    print("=" * 65)


if __name__ == "__main__":
    main()
