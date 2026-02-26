#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
正确的参数提取脚本
注意: YAML文件给出的是 flan→camera 的变换，需要计算逆变换得到 camera→flan
"""

import yaml
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

def extract_yaml_params_corrected(yaml_file="config/extrinsics_flan_to_hand_camera.yaml"):
    """
    从YAML正确提取T_cam2flan和R_cam2flan
    
    注意: YAML给出 T_flan2cam, R_flan2cam
          需要计算逆变换得到 T_cam2flan, R_cam2flan
    """
    print("正确提取YAML参数:")
    print("=" * 30)
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"坐标系: {data['header']['frame_id']} → {data['child_frame_id']}")
    
    # YAML中的变换: flan → camera
    trans = data['transform']['translation']
    T_flan2cam = np.array([trans['x'], trans['y'], trans['z']])
    
    rot = data['transform']['rotation'] 
    quat = [rot['x'], rot['y'], rot['z'], rot['w']]
    R_flan2cam = R.from_quat(quat).as_matrix()
    
    print(f"YAML中的 T_flan2cam: {T_flan2cam}")
    print(f"YAML中的四元数: {quat}")
    
    # 计算逆变换: camera → flan
    R_cam2flan = R_flan2cam.T  # 旋转矩阵的逆 = 转置
    T_cam2flan = -R_cam2flan @ T_flan2cam  # 平移的逆变换
    
    # 计算逆变换的四元数
    q_cam2flan = R.from_matrix(R_cam2flan).as_quat()  # [x, y, z, w]
    
    print(f"计算的 T_cam2flan:\n {T_cam2flan}")
    print(f"计算的 R_cam2flan:\n {R_cam2flan}")
    print(f"计算的 q_cam2flan:\n {q_cam2flan}")
    
    # 验证逆变换
    print("\n逆变换验证:")
    identity_check = R_flan2cam @ R_cam2flan
    is_identity = np.allclose(identity_check, np.eye(3))
    print(f"R_flan2cam @ R_cam2flan ≈ I: {is_identity}")
    
    # 验证四元数转换
    R_from_quat = R.from_quat(q_cam2flan).as_matrix()
    quat_correct = np.allclose(R_cam2flan, R_from_quat)
    print(f"q_cam2flan → R_cam2flan 一致性: {quat_correct}")
    
    return T_cam2flan.tolist(), R_cam2flan.tolist(), q_cam2flan.tolist()

def extract_urdf_params(urdf_file="config/mobile_manipulator2_description.urdf"):
    """从URDF提取T_gripper2flan (夹爪到法兰的偏移)"""
    print("\n提取URDF参数:")
    print("=" * 20)
    
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    joint7_xyz = joint8_xyz = None
    
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name == 'joint7':
            origin = joint.find('origin')
            if origin is not None:
                joint7_xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                print(f"joint7: {joint7_xyz}")
        elif name == 'joint8':
            origin = joint.find('origin')
            if origin is not None:
                joint8_xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                print(f"joint8: {joint8_xyz}")
    
    if joint7_xyz and joint8_xyz:
        T_gripper2flan = [(joint7_xyz[i] + joint8_xyz[i]) / 2 for i in range(3)]
        print(f"夹爪中心 T_gripper2flan: {T_gripper2flan}")
    else:
        T_gripper2flan = [0.0, 0.0, 0.13503]
        print(f"使用默认 T_gripper2flan: {T_gripper2flan}")

    return T_gripper2flan


def main():
    print("提取 T_cam2flan 和 q_cam2flan")
    print("=" * 40)
    
    try:
        # 提取YAML参数 (逆变换)
        T_cam2flan, R_cam2flan, q_cam2flan = extract_yaml_params_corrected()
        
        # 提取URDF参数
        T_gripper2flan = extract_urdf_params()

        print("\n提取结果:")
        print("=" * 20)
        print(f"T_cam2flan = {T_cam2flan}")
        print(f"q_cam2flan = {q_cam2flan}")
        print(f"T_gripper2flan = {T_gripper2flan}")

        print("\n用于 piper_robot.py:")
        print("=" * 20)
        print(f"cfg.T_cam2flan = {T_cam2flan}")
        print(f"cfg.q_cam2flan = {q_cam2flan}  # [x,y,z,w]")
        print(f"cfg.R_cam2flan = R.from_quat(cfg.q_cam2flan).as_matrix().tolist()")
        print(f"cfg.T_gripper2flan = {T_gripper2flan}")
        
    except Exception as e:
        print(f"提取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
