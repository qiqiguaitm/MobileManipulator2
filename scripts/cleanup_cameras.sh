#!/bin/bash
# 清理所有相机相关进程

##    ./scripts/cleanup_cameras.sh

echo "正在清理相机进程..."

# 杀死 realsense 相机节点
pkill -9 -f "realsense2_camera_node" 2>/dev/null

# 杀死相机 TF 发布器
pkill -9 -f "camera_tf_publisher" 2>/dev/null

# 杀死多相机感知节点
pkill -9 -f "multi_camera_perception" 2>/dev/null

# 杀死相机驱动 launch
pkill -9 -f "camera_driver.launch.py" 2>/dev/null

# 等待 USB 复位
sleep 3

echo "清理完成，等待 USB 复位..."
sleep 2

echo "可以重新启动相机了"
