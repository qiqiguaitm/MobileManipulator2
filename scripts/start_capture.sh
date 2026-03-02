#!/bin/bash
# 双相机采集启动脚本

export ROS_DOMAIN_ID=42
cd /home/didi/workspace/MobileManipulator2
source install/setup.bash

echo "=== 检查相机驱动 ==="
if ! ros2 node list | grep -q "/camera"; then
    echo "相机驱动未启动，正在启动..."
    ros2 launch camera_driver camera_driver.launch.py top_enable:=true chassis_enable:=true &
    DRIVER_PID=$!
    echo "等待相机初始化..."
    sleep 10
else
    echo "相机驱动已运行"
fi

echo ""
echo "=== 启动采集程序 ==="
echo "按键说明: [SPACE]保存  [a]自动模式  [q]退出"
python3 scripts/capture_stereo.py
