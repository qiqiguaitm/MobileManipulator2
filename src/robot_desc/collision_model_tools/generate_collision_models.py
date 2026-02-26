#!/usr/bin/env python3
"""
Generate simplified collision models for mobile_manipulator2
Linus式原则：简单直接，没有花哨的抽象
"""

import os
import trimesh
import numpy as np
from pathlib import Path

# 配置：文件名 -> (简化策略, 安全裕度mm)
# Linus: "Do one thing well. Don't duplicate safety margins."
# 移除了padding，统一在MoveIt的collision_padding.yaml中处理
SIMPLIFICATION_CONFIG = {
    'chassis_base_link.STL': ('convex_hull', 0),  # 底盘 - 只简化，不膨胀
    'box_Link.STL': ('convex_hull', 0),
    'lifting_Link.STL': ('composite', 0),  # 特殊处理：结构+显示器
    'lidar_Link.STL': ('cylinder', 0),
    'top_camera_Link.STL': ('box', 0),
    'under_camera_Link.STL': ('box', 0),
}

def simplify_to_box(mesh, padding_mm):
    """简化为包围盒（不再添加padding）"""
    bounds = mesh.bounds
    # padding参数保留用于向后兼容，但不使用
    
    # 创建精确尺寸的box
    box_extents = bounds[1] - bounds[0]
    box_center = (bounds[1] + bounds[0]) / 2
    
    return trimesh.creation.box(extents=box_extents, transform=trimesh.transformations.translation_matrix(box_center))

def simplify_to_cylinder(mesh, padding_mm):
    """简化为圆柱体（用于激光雷达，不添加padding）"""
    bounds = mesh.bounds
    # padding参数保留用于向后兼容，但不使用
    
    # 计算精确高度和半径
    height = bounds[1][2] - bounds[0][2]
    radius = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]) / 2
    center_z = (bounds[1][2] + bounds[0][2]) / 2
    
    return trimesh.creation.cylinder(radius=radius, height=height, 
                                   transform=trimesh.transformations.translation_matrix([0, 0, center_z]))

def simplify_to_convex_hull(mesh, padding_mm):
    """简化为凸包（不添加padding）"""
    # 获取凸包
    hull = mesh.convex_hull
    
    # 不再应用padding - 统一在MoveIt中处理
    # Linus: "One place for one thing"
    
    return hull

def simplify_lifting_link(mesh, padding_mm):
    """处理lifting_Link：直接使用凸包，不单独建模显示器"""
    # 显示器的安全性通过collision_padding.yaml中的lifting_Link padding值保证(35mm)
    return simplify_to_convex_hull(mesh, padding_mm)

def main():
    visual_dir = Path('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/meshes/visual')
    collision_dir = Path('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/meshes/collision')
    
    collision_dir.mkdir(exist_ok=True)
    
    for filename, (strategy, padding) in SIMPLIFICATION_CONFIG.items():
        visual_path = visual_dir / filename
        collision_path = collision_dir / filename.replace('.STL', '_collision.STL')
        
        if not visual_path.exists():
            print(f"警告: {visual_path} 不存在")
            continue
            
        print(f"处理 {filename}: {strategy} 策略, {padding}mm 裕度")
        
        # 加载原始模型
        mesh = trimesh.load(visual_path)
        
        # 应用简化策略
        if strategy == 'box':
            simplified = simplify_to_box(mesh, padding)
        elif strategy == 'cylinder':
            simplified = simplify_to_cylinder(mesh, padding)
        elif strategy == 'convex_hull':
            simplified = simplify_to_convex_hull(mesh, padding)
        elif strategy == 'composite':
            simplified = simplify_lifting_link(mesh, padding)
        else:
            print(f"未知策略: {strategy}")
            continue
        
        # 保存简化模型
        simplified.export(collision_path)
        
        # 报告文件大小
        original_size = os.path.getsize(visual_path) / 1024 / 1024  # MB
        new_size = os.path.getsize(collision_path) / 1024  # KB
        print(f"  原始: {original_size:.1f}MB -> 简化: {new_size:.1f}KB")

if __name__ == '__main__':
    main()