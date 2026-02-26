#!/usr/bin/env python3
"""
更新URDF文件使用本地模型
- safety版本：使用visual目录下的所有模型
- performance版本：使用collision目录下的碰撞模型
"""

import re

def update_safety_urdf():
    """更新safety版本，让所有Piper模型使用本地visual目录"""
    urdf_path = '/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2.urdf'
    
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # 替换Piper模型路径到本地
    replacements = [
        # 视觉模型 - DAE保持在piper目录（因为包含纹理）
        # 碰撞模型 - 改为本地STL
        (r'package://piper_description/meshes/base_link\.STL', 
         'package://mobile_manipulator2_description/meshes/visual/base_link.STL'),
        (r'package://piper_description/meshes/link([0-8])\.STL',
         r'package://mobile_manipulator2_description/meshes/visual/link\1.STL'),
        (r'package://piper_description/meshes/gripper_base\.STL',
         'package://mobile_manipulator2_description/meshes/visual/gripper_base.STL'),
    ]
    
    for old, new in replacements:
        content = re.sub(old, new, content)
    
    with open(urdf_path, 'w') as f:
        f.write(content)
    
    print("Safety URDF已更新")

def update_performance_urdf():
    """更新performance版本，使用本地碰撞模型"""
    urdf_path = '/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_performance.urdf'
    
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # 1. 更新Piper碰撞模型路径
    lines = content.split('\n')
    updated_lines = []
    in_collision = False
    
    for line in lines:
        if '<collision>' in line:
            in_collision = True
        elif '</collision>' in line:
            in_collision = False
            
        # 处理碰撞模型路径
        if in_collision and 'mesh filename=' in line and 'piper_description' in line:
            # 替换为本地碰撞模型
            if 'base_link' in line:
                line = line.replace('piper_description/meshes/collision/base_link_collision.STL',
                                  'mobile_manipulator2_description/meshes/collision/base_link_collision.STL')
            elif 'link' in line:
                for i in range(1, 9):
                    line = line.replace(f'piper_description/meshes/collision/link{i}_collision.STL',
                                      f'mobile_manipulator2_description/meshes/collision/link{i}_collision.STL')
            elif 'gripper_base' in line:
                line = line.replace('piper_description/meshes/collision/gripper_base_collision.STL',
                                  'mobile_manipulator2_description/meshes/collision/gripper_base_collision.STL')
        
        # 处理hand_cam碰撞模型
        if in_collision and 'hand_cam' in line and '<box size=' in line:
            # 替换为STL模型
            line = '        <mesh filename="package://mobile_manipulator2_description/meshes/collision/hand_cam_collision.STL"/>'
            
        updated_lines.append(line)
    
    with open(urdf_path, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print("Performance URDF已更新")

def main():
    print("更新URDF文件使用本地模型...")
    update_safety_urdf()
    update_performance_urdf()
    print("\n完成！所有模型现在都在mobile_manipulator2_description目录下")

if __name__ == '__main__':
    main()