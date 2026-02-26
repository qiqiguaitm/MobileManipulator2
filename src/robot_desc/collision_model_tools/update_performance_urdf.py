#!/usr/bin/env python3
"""
更新performance URDF，仅替换碰撞模型为简化版本
视觉模型保持不变
"""

import re

def update_collision_meshes(urdf_content):
    """仅更新collision部分的mesh路径"""
    
    # 定义需要替换的碰撞模型
    collision_replacements = {
        'base_link.STL': 'base_link_collision.STL',
        'box_Link.STL': 'box_Link_collision.STL', 
        'lifting_Link.STL': 'lifting_Link_collision.STL',
        'lidar_Link.STL': 'lidar_Link_collision.STL',
        'top_camera_Link.STL': 'top_camera_Link_collision.STL',
        'under_camera_Link.STL': 'under_camera_Link_collision.STL',
    }
    
    # 处理每个碰撞块
    lines = urdf_content.split('\n')
    in_collision = False
    updated_lines = []
    
    for line in lines:
        if '<collision>' in line:
            in_collision = True
        elif '</collision>' in line:
            in_collision = False
            
        if in_collision and 'mesh filename=' in line:
            # 检查是否需要替换
            for old_name, new_name in collision_replacements.items():
                if old_name in line:
                    # 替换路径：visual -> collision, 文件名也要改
                    line = line.replace('meshes/visual/', 'meshes/collision/')
                    line = line.replace(old_name, new_name)
                    break
                    
        updated_lines.append(line)
    
    return '\n'.join(updated_lines)

def main():
    # 读取performance URDF
    with open('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_performance.urdf', 'r') as f:
        content = f.read()
    
    # 更新碰撞模型
    updated_content = update_collision_meshes(content)
    
    # 写回文件
    with open('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_performance.urdf', 'w') as f:
        f.write(updated_content)
    
    print("Performance URDF已更新，仅碰撞模型使用简化版本")

if __name__ == '__main__':
    main()