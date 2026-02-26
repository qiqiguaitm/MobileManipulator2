#!/usr/bin/env python3
"""
更新performance URDF中Piper部分的碰撞模型
"""

def update_piper_collisions(urdf_path):
    # Piper碰撞模型替换规则
    piper_replacements = {
        'meshes/base_link.STL': 'meshes/collision/base_link_collision.STL',
        'meshes/link1.STL': 'meshes/collision/link1_collision.STL',
        'meshes/link2.STL': 'meshes/collision/link2_collision.STL',
        'meshes/link3.STL': 'meshes/collision/link3_collision.STL',
        'meshes/link4.STL': 'meshes/collision/link4_collision.STL',
        'meshes/link5.STL': 'meshes/collision/link5_collision.STL',
        'meshes/link7.STL': 'meshes/collision/link7_collision.STL',
        'meshes/link8.STL': 'meshes/collision/link8_collision.STL',
    }
    
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    in_collision = False
    updated_lines = []
    
    for line in lines:
        if '<collision>' in line:
            in_collision = True
        elif '</collision>' in line:
            in_collision = False
            
        if in_collision and 'piper_description' in line and 'mesh filename=' in line:
            # 检查并替换Piper的碰撞模型
            for old_path, new_path in piper_replacements.items():
                if old_path in line:
                    line = line.replace(old_path, new_path)
                    break
                    
        updated_lines.append(line)
    
    with open(urdf_path, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print("Piper碰撞模型已更新")

if __name__ == '__main__':
    update_piper_collisions('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/urdf/mobile_manipulator2_performance.urdf')