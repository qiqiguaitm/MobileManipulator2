#!/bin/bash
# 验证碰撞模型是否正确加载

echo "=================================="
echo "碰撞模型验证脚本"
echo "=================================="

# 清理环境
pkill -9 roscore
pkill -9 rosmaster
sleep 2

# 启动roscore
roscore &
ROSCORE_PID=$!
sleep 3

echo -e "\n1. 加载robot_description参数"
echo "----------------------------"
rosparam load /home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_performance.urdf robot_description

echo -e "\n2. 检查碰撞模型路径"
echo "-------------------"
echo "Mobile Manipulator2碰撞模型："
rosparam get /robot_description | grep -o "mobile_manipulator2_description/meshes/collision/[^\"]*" | sort | uniq

echo -e "\n3. 检查错误的外部引用"
echo "----------------------"
echo "检查是否还有piper_description引用："
if rosparam get /robot_description | grep -q "piper_description/meshes"; then
    echo "❌ 发现外部引用："
    rosparam get /robot_description | grep -o "piper_description/meshes/[^\"]*" | sort | uniq
else
    echo "✅ 没有外部引用，所有模型都是本地的"
fi

echo -e "\n4. 统计模型数量"
echo "----------------"
echo "碰撞模型总数："
rosparam get /robot_description | grep -o "collision.*\.STL" | wc -l

echo -e "\n5. 验证关键模型"
echo "----------------"
for model in base_link box_Link lifting_Link lidar_Link link1 link2 link3 hand_cam; do
    if rosparam get /robot_description | grep -q "${model}_collision.STL"; then
        echo "✅ ${model}_collision.STL"
    else
        echo "❌ ${model}_collision.STL 缺失"
    fi
done

kill $ROSCORE_PID 2>/dev/null

echo -e "\n验证完成！"
echo "=================================="