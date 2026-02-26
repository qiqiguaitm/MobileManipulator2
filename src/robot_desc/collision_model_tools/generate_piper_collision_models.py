#!/usr/bin/env python3
"""
为Piper机械臂生成简化碰撞模型
保持与设计方案一致的安全裕度
"""

import os
import trimesh
import numpy as np
from pathlib import Path

# Piper链接简化配置
# Linus: "Do one thing well. Don't duplicate safety margins."
# 移除了padding，统一在MoveIt的collision_padding.yaml中处理
PIPER_SIMPLIFICATION_CONFIG = {
    # link_name: (简化策略, 安全裕度mm, 目标大小)
    'base_link.STL': ('convex_hull', 0, 10),      # 固定基座 - 只简化，不膨胀
    'link1.STL': ('convex_hull', 0, 30),          # 旋转基座
    'link2.STL': ('convex_hull', 0, 50),          # 大臂
    'link3.STL': ('convex_hull', 0, 50),          # 小臂
    'link4.STL': ('convex_hull', 0, 30),          # 腕部旋转
    'link5.STL': ('convex_hull', 0, 30),          # 腕部俯仰
    'link6.STL': ('convex_hull', 0, 30),          # 末端（含相机）
    'link7.STL': ('convex_hull', 0, 10),          # 夹爪左
    'link8.STL': ('convex_hull', 0, 10),          # 夹爪右
    'gripper_base.STL': ('convex_hull', 0, 20),   # 夹爪基座
}

def simplify_to_convex_hull(mesh, padding_mm, target_size_kb):
    """简化为凸包，并尝试进一步简化到目标大小"""
    # 获取凸包
    hull = mesh.convex_hull
    
    # 不再应用padding - 统一在MoveIt中处理
    # Linus: "One place for one thing"
    
    # 如果凸包还是太大，进一步简化
    if len(hull.vertices) > 1000:
        # 使用体素化再重建来减少顶点
        voxel_size = max(mesh.extents) / 50  # 50x50x50体素网格
        voxelized = hull.voxelized(voxel_size)
        hull = voxelized.as_boxes().sum()
        
    return hull

def create_hand_cam_collision():
    """为hand_cam创建简单的box碰撞模型"""
    # hand_cam是一个简单的box，尺寸来自URDF
    # <box size="0.02 0.05 0.02"/>
    # 安全裕度 +10mm
    padding = 0.01  # 10mm
    box = trimesh.creation.box(extents=[
        0.02 + 2*padding,
        0.05 + 2*padding, 
        0.02 + 2*padding
    ])
    return box

def main():
    visual_dir = Path('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/meshes/visual')
    collision_dir = Path('/home/agilex/MobileManipulator/src/robot_desc/mobile_manipulator2_description/meshes/collision')
    
    # 生成Piper碰撞模型
    print("=== 生成Piper机械臂碰撞模型 ===")
    for filename, (strategy, padding, target_kb) in PIPER_SIMPLIFICATION_CONFIG.items():
        visual_path = visual_dir / filename
        
        # 对于arm_base，使用base_link.STL
        if filename == 'arm_base.STL':
            visual_path = visual_dir / 'base_link.STL'
            
        collision_path = collision_dir / filename.replace('.STL', '_collision.STL')
        
        if not visual_path.exists():
            print(f"警告: {visual_path} 不存在")
            continue
            
        print(f"\n处理 {filename}")
        print(f"  策略: {strategy}, 裕度: {padding}mm, 目标: <{target_kb}KB")
        
        # 加载原始模型
        mesh = trimesh.load(visual_path)
        
        # 简化
        if strategy == 'convex_hull':
            simplified = simplify_to_convex_hull(mesh, padding, target_kb)
        else:
            print(f"未知策略: {strategy}")
            continue
        
        # 保存
        simplified.export(collision_path)
        
        # 报告文件大小
        original_size = os.path.getsize(visual_path) / 1024  # KB
        new_size = os.path.getsize(collision_path) / 1024  # KB
        print(f"  原始: {original_size:.1f}KB -> 简化: {new_size:.1f}KB (压缩比: {original_size/new_size:.1f}x)")
        
        # 验证是否达到目标
        if new_size > target_kb:
            print(f"  ⚠️  警告: 未达到目标大小 {target_kb}KB")
    
    # 生成hand_cam碰撞模型
    print("\n\n=== 生成hand_cam碰撞模型 ===")
    hand_cam_collision = create_hand_cam_collision()
    hand_cam_path = collision_dir / 'hand_cam_collision.STL'
    hand_cam_collision.export(hand_cam_path)
    size_kb = os.path.getsize(hand_cam_path) / 1024
    print(f"hand_cam碰撞模型: {size_kb:.1f}KB")

if __name__ == '__main__':
    main()