#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import yaml
import tf2_ros
import os
import numpy as np
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R


class CameraTFPublisher:
    """
    相机TF发布器 - 从配置文件读取外参并发布静态TF变换
    """
    
    def __init__(self, config_path=None):
        rospy.init_node('camera_tf_publisher', anonymous=True)
        
        if config_path is None:
            config_path = rospy.get_param('~config_path', 
                '/home/agilex/MobileManipulator/src/robot_drivers/camera_driver/config')
        
        self.config_path = config_path
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # 加载并发布所有相机TF
        self._load_and_publish_transforms()
        
        rospy.loginfo("Camera TF Publisher initialized")
        
    def _load_transform_from_yaml(self, yaml_file, expected_parent=None, expected_child=None):
        """从YAML文件加载变换"""
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                
            transform_data = data['transform']
            
            # 提取变换信息
            translation = transform_data['translation']
            rotation = transform_data['rotation']
            
            # frame信息可能在header下或直接在根级
            if 'header' in data and 'frame_id' in data['header']:
                parent_frame = data['header']['frame_id']
                child_frame = data['header'].get('child_frame_id') or data.get('child_frame_id')
            else:
                parent_frame = data.get('frame_id') or data.get('parent_frame_id')
                child_frame = data.get('child_frame_id')
            
            # 可选：验证frame名称
            if expected_parent and parent_frame != expected_parent:
                rospy.logwarn(f"Expected parent {expected_parent}, got {parent_frame}")
            if expected_child and child_frame != expected_child:
                rospy.logwarn(f"Expected child {expected_child}, got {child_frame}")
                
            return parent_frame, child_frame, translation, rotation
            
        except Exception as e:
            rospy.logerr(f"Failed to load transform from {yaml_file}: {e}")
            return None
            
    def _create_transform_stamped(self, parent_frame, child_frame, translation, rotation):
        """创建TransformStamped消息"""
        t = TransformStamped()
        t.header.stamp = rospy.Time(0)  # 使用固定时间戳避免TF_OLD_DATA警告
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        # 平移
        t.transform.translation.x = translation['x']
        t.transform.translation.y = translation['y']
        t.transform.translation.z = translation['z']
        
        # 旋转 (四元数)
        t.transform.rotation.x = rotation['x']
        t.transform.rotation.y = rotation['y']
        t.transform.rotation.z = rotation['z']
        t.transform.rotation.w = rotation['w']
        
        return t
        
    def _standardize_frame_name(self, original_name):
        """标准化frame名称 - 避免与URDF冲突的标定frame"""
        frame_mapping = {
            'rs16_lidar': 'calibration/lidar',      # 标定的雷达frame，使用calibration namespace
            'rslidar': 'calibration/lidar',         # 新的标定雷达frame名称
            'chassis_camera': 'calibration/chassis_camera_physical',  # 标定的底盘相机
            'top_camera': 'calibration/top_camera_physical', 
            'hand_camera': 'calibration/hand_camera_physical',
            'gripper': 'gripper_link',        # 夹爪frame保持原名（机器人结构）
            'chassis_base': 'base_link',      # 底盘基座对应base_link
            'base': 'base_link'               # 简化的基座frame名称
        }
        
        return frame_mapping.get(original_name, original_name)
        
    def _load_and_publish_transforms(self):
        """加载配置文件并发布所有TF变换"""
        transforms_to_publish = []
        
        # 1. 基座到标定激光雷达frame
        lidar_config = os.path.join(self.config_path, 'extrinsics_chassis_base_to_rslidar.yaml')
        rospy.loginfo(f"Looking for lidar config: {lidar_config}")
        rospy.loginfo(f"File exists: {os.path.exists(lidar_config)}")
        if os.path.exists(lidar_config):
            result = self._load_transform_from_yaml(lidar_config)
            rospy.loginfo(f"Load result: {result}")
            if result:
                parent, child, trans, rot = result
                parent = self._standardize_frame_name(parent)
                child = self._standardize_frame_name(child)
                
                # 配置文件是 chassis_base -> rs16_lidar，直接发布标定结果
                rospy.loginfo(f"Lidar config: original parent={result[0]}, child={result[1]}")
                rospy.loginfo(f"Lidar config: mapped parent={parent}, child={child}")
                if parent == 'base_link' and child == 'calibration/lidar':
                    t = self._create_transform_stamped(parent, child, trans, rot)
                    transforms_to_publish.append(t)
                    rospy.loginfo(f"Loaded calibrated: {parent} -> {child}")
                else:
                    rospy.logwarn(f"Lidar config condition failed: parent={parent}, child={child}")
                    
        # 2. 标定激光雷达到底盘相机物理位置
        chassis_config = os.path.join(self.config_path, 'extrinsics_lidar_to_chassis_camera.yaml')
        if os.path.exists(chassis_config):
            result = self._load_transform_from_yaml(chassis_config)
            if result:
                parent, child, trans, rot = result
                parent = self._standardize_frame_name(parent)
                
                # 配置文件是 rs16_lidar -> chassis_camera（物理位置）
                if parent == 'calibration/lidar':
                    # 外参：激光雷达到相机物理位置
                    #inv_trans, inv_rot = self._invert_transform(trans, rot)
                    inv_trans, inv_rot = trans,rot
                    t = self._create_transform_stamped(parent, 'calibration/chassis_camera_physical', inv_trans, inv_rot)
                    transforms_to_publish.append(t)
                    rospy.loginfo(f"Loaded calibrated: {parent} -> calibration/chassis_camera_physical")
                    
                    # 内参相关：物理位置到光学中心的固定变换
                    optical_t = self._create_camera_to_optical_transform_4calib('calibration/chassis_camera_physical', 
                                                                         'calibration/chassis_camera_optical')
                    transforms_to_publish.append(optical_t)
                    rospy.loginfo(f"Added optical: calibration/chassis_camera_physical -> calibration/chassis_camera_optical")
                    
        # 3. 标定激光雷达到顶部相机物理位置
        top_config = os.path.join(self.config_path, 'extrinsics_lidar_to_top_camera.yaml')
        if os.path.exists(top_config):
            result = self._load_transform_from_yaml(top_config)
            if result:
                parent, child, trans, rot = result
                parent = self._standardize_frame_name(parent)
                
                if parent == 'calibration/lidar':
                    # 外参：激光雷达到相机物理位置
                    #inv_trans, inv_rot = self._invert_transform(trans, rot)
                    inv_trans, inv_rot = trans,rot
                    t = self._create_transform_stamped(parent, 'calibration/top_camera_physical', inv_trans, inv_rot)
                    transforms_to_publish.append(t)
                    rospy.loginfo(f"Loaded calibrated: {parent} -> calibration/top_camera_physical")
                    
                    # 内参相关：物理位置到光学中心的固定变换
                    optical_t = self._create_camera_to_optical_transform_4calib('calibration/top_camera_physical', 
                                                                         'calibration/top_camera_optical')
                    transforms_to_publish.append(optical_t)
                    rospy.loginfo(f"Added optical: calibration/top_camera_physical -> calibration/top_camera_optical")
                    
        # 4. 连接URDF相机link到RealSense相机link的静态变换
        # 这些变换连接机器人结构中的相机mount点到RealSense的base frame
        
        # 顶部相机连接
        top_connect_t = TransformStamped()
        top_connect_t.header.stamp = rospy.Time(0)
        top_connect_t.header.frame_id = 'top_camera_link'
        top_connect_t.child_frame_id = 'camera/top_link'
        top_connect_t.transform.translation.x = 0.0
        top_connect_t.transform.translation.y = 0.0
        top_connect_t.transform.translation.z = 0.0
        top_connect_t.transform.rotation.x = 0.0
        top_connect_t.transform.rotation.y = 0.0
        top_connect_t.transform.rotation.z = 0.0
        top_connect_t.transform.rotation.w = 1.0
        transforms_to_publish.append(top_connect_t)
        rospy.loginfo("Added connection: top_camera_link -> camera/top_link")
        
        # 底盘相机连接
        chassis_connect_t = TransformStamped()
        chassis_connect_t.header.stamp = rospy.Time(0)
        chassis_connect_t.header.frame_id = 'chassis_camera_link'
        chassis_connect_t.child_frame_id = 'camera/chassis_link'
        chassis_connect_t.transform.translation.x = 0.0
        chassis_connect_t.transform.translation.y = 0.0
        chassis_connect_t.transform.translation.z = 0.0
        chassis_connect_t.transform.rotation.x = 0.0
        chassis_connect_t.transform.rotation.y = 0.0
        chassis_connect_t.transform.rotation.z = 0.0
        chassis_connect_t.transform.rotation.w = 1.0
        transforms_to_publish.append(chassis_connect_t)
        rospy.loginfo("Added connection: chassis_camera_link -> camera/chassis_link")
        
        # 手部相机连接
        hand_connect_t = TransformStamped()
        hand_connect_t.header.stamp = rospy.Time(0)
        hand_connect_t.header.frame_id = 'hand_camera_link'
        hand_connect_t.child_frame_id = 'camera/hand_link'
        hand_connect_t.transform.translation.x = 0.0
        hand_connect_t.transform.translation.y = 0.0
        hand_connect_t.transform.translation.z = 0.0
        hand_connect_t.transform.rotation.x = 0
        hand_connect_t.transform.rotation.y = 0
        hand_connect_t.transform.rotation.z = 0
        hand_connect_t.transform.rotation.w = 1.0
        transforms_to_publish.append(hand_connect_t)
        rospy.loginfo("Added connection: hand_camera_link -> camera/hand_link")
                    
        # 发布所有变换
        if transforms_to_publish:
            self.tf_broadcaster.sendTransform(transforms_to_publish)
            rospy.loginfo(f"Published {len(transforms_to_publish)} camera TF transforms")
        else:
            rospy.logwarn("No valid TF transforms found to publish")
            
    def _create_camera_to_optical_transform_4rs(self, physical_frame, optical_frame):
        """创建从相机物理位置到光学中心的标准变换
        
        RealSense相机的光学坐标系定义：
        - Z轴：向前（进入场景）
        - X轴：向右
        - Y轴：向下
        
        相对于相机物理坐标系需要旋转：
        - 绕X轴旋转-90度（使Z轴向前）
        - 绕Z轴旋转-90度（调整XY方向）
        """
        t = TransformStamped()
        t.header.stamp = rospy.Time(0)
        t.header.frame_id = physical_frame
        t.child_frame_id = optical_frame
        
        # 无平移（光学中心与物理中心重合）
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # 旋转：从物理坐标系到光学坐标系
        # 标准的相机光学变换通常是 [-90°, 0°, -90°] (roll, pitch, yaw)
        r = R.from_euler('xyz', [-np.pi/2, 0, -np.pi/2])
        quat = r.as_quat()
        
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        return t
    
    def _create_camera_to_optical_transform_4calib(self, physical_frame, optical_frame):
        """创建从相机物理位置到光学中心的标准变换
        
        RealSense相机的光学坐标系定义：
        - Z轴：向前（进入场景）
        - X轴：向右
        - Y轴：向下
        """
        t = TransformStamped()
        t.header.stamp = rospy.Time(0)
        t.header.frame_id = physical_frame
        t.child_frame_id = optical_frame
        
        # 无平移（光学中心与物理中心重合）
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # 旋转：从物理坐标系到光学坐标系
        # 标准的相机光学变换通常是 [-90°, 0°, -90°] (roll, pitch, yaw)
        r = R.from_euler('xyz', [0, 0, 0])
        quat = r.as_quat()
        
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        return t
    
    
    def _invert_transform(self, translation, rotation):
        """计算变换的逆"""
        # 创建旋转对象
        r = R.from_quat([rotation['x'], rotation['y'], rotation['z'], rotation['w']])
        
        # 逆旋转
        r_inv = r.inv()
        quat_inv = r_inv.as_quat()
        
        # 逆平移：t_inv = -R^-1 * t
        trans_vec = [translation['x'], translation['y'], translation['z']]
        trans_inv_vec = -r_inv.apply(trans_vec)
        
        inv_translation = {
            'x': trans_inv_vec[0],
            'y': trans_inv_vec[1], 
            'z': trans_inv_vec[2]
        }
        
        inv_rotation = {
            'x': quat_inv[0],
            'y': quat_inv[1],
            'z': quat_inv[2],
            'w': quat_inv[3]
        }
        
        return inv_translation, inv_rotation


if __name__ == '__main__':
    try:
        publisher = CameraTFPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
