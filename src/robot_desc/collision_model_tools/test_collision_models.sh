#!/bin/bash
# 测试双碰撞模型的加载和性能

echo "=================================="
echo "双碰撞模型测试脚本"
echo "=================================="

# 清理ROS环境
pkill -9 roscore
pkill -9 rosmaster
sleep 2

# 启动新的roscore
roscore &
ROSCORE_PID=$!
sleep 3

echo ""
echo "1. 测试 PERFORMANCE 模式（默认）"
echo "----------------------------------"
# 测试performance模式
timeout 10s roslaunch mobile_manipulator2_description load_description.launch collision_mode:=performance &
LAUNCH_PID=$!
sleep 5

# 检查robot_description参数
echo "检查加载的URDF..."
if rosparam get /robot_description | grep -q "collision/.*_collision.STL"; then
    echo "✓ Performance模式正确加载了简化的碰撞模型"
else
    echo "✗ Performance模式未能正确加载简化模型"
fi

kill $LAUNCH_PID 2>/dev/null
sleep 2

echo ""
echo "2. 测试 SAFETY 模式"
echo "-------------------"
# 测试safety模式
timeout 10s roslaunch mobile_manipulator2_description load_description.launch collision_mode:=safety &
LAUNCH_PID=$!
sleep 5

# 检查robot_description参数
echo "检查加载的URDF..."
if rosparam get /robot_description | grep -q "visual/.*\.STL"; then
    echo "✓ Safety模式正确加载了原始精确模型"
else
    echo "✗ Safety模式未能正确加载原始模型"
fi

kill $LAUNCH_PID 2>/dev/null
kill $ROSCORE_PID 2>/dev/null

echo ""
echo "3. 文件大小对比"
echo "----------------"
echo "原始模型总大小："
du -ch /home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/meshes/visual/*.STL | tail -1

echo ""
echo "简化模型总大小："
du -ch /home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/meshes/collision/*.STL | tail -1

echo ""
echo "测试完成！"
echo "=================================="