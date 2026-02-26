#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
PiperRobot Simple - Simplified XArmRobot Compatible Wrapper

Simplified version based on V2 examples, focusing on essential robot control
functionality with minimal complexity while maintaining XArm compatibility.
"""

import copy
import time
import numpy as np
from mmengine.config import Config
from scipy.spatial.transform import Rotation as R
from piper_api import PiperAPI
from common import debug_print

class PiperRobot:
    """
    Simplified Piper Robot Controller - XArmRobot compatible
    
    Provides essential robot control functionality based on V2 examples,
    with simplified coordinate transformations and streamlined operations.
    """
    
    def __init__(self, cfg):
        """
        Initialize simplified Piper robot controller
        
        Args:
            cfg: Configuration object with:
                - can_name: CAN device name
                - init_pos: Initial position dict {x,y,z,roll,pitch,yaw} (optional)
                - T_cam2flan: Camera to flange translation (m)
                - R_cam2flan: Camera to flange rotation matrix
                - T_gripper2flan: Gripper to flange offset (m)
                - gripper_max_mm: Gripper max range in mm
        """
        self.cfg = cfg
        self.init_pos = cfg.get('init_pos', None)
        
        # Coordinate transformation matrices (simplified)
        self.T_cam2flan = np.array(cfg.T_cam2flan)
        self.R_cam2flan = np.array(cfg.R_cam2flan)
        self.T_gripper2flan = np.array(cfg.T_gripper2flan)
        
        # End effector transform (camera to end)
        if cfg.get('gripper_as_end', True):
            self.T_cam2end = self.T_cam2flan - self.T_gripper2flan
        else:
            self.T_cam2end = self.T_cam2flan
        self.R_cam2end = self.R_cam2flan
        
        # Initialize simplified Piper API
        can_name = cfg.get('can_name', 'can0')
        gripper_max_mm = cfg.get('gripper_max_mm', 90.0)
        # Extract gripper offset from T_gripper2flan Z component (convert m to mm)
        gripper_offset_mm = self.T_gripper2flan[2] * 1000 if self.T_gripper2flan is not None else 135.03
        self.arm = PiperAPI(can_name, gripper_max_mm, gripper_offset_mm=gripper_offset_mm)
        
        print(f"PiperRobot Simple initialized: {can_name}")
    
    def connect(self):
        """
        Connect and initialize robot - simplified V2 style
        
        Following V2 pattern:
        1. Connect arm
        2. Basic setup
        3. Move to initial position
        4. Open gripper
        """
        try:
            # Connect arm
            self.arm.connect()
            time.sleep(0.5)
            
            # Basic setup (simplified)
            self.arm.clean_error(go_zero=True,force=True)
            self.arm.clean_gripper_error()
            self.arm.motion_enable(True,force=True,go_zero=True)
            #self.arm.set_mode(0)
            #self.arm.set_state(0)
            
            # Move to initial position (if specified)
            if self.init_pos is not None:
                self.arm.set_position(**self.init_pos, wait=True)
                state, new_pos = self.arm.get_position()
                print(f"✓ Moved to initial position: {self.init_pos}, actual position: {new_pos}")
            else:
                print("ℹ No initial position specified, staying at current position")
            
            # Open gripper to max
            self.arm.set_gripper_position(self.arm.gripper_max_units)
            
            print("✓ PiperRobot Simple connected")
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise e
    
    def disconnect(self):
        """Disconnect robot"""
        if self.arm:
            self.arm.disconnect()
    
    def point2base(self, point3d, src_frame='cam', end_pos=None):
        """
        Transform point to base coordinates - simplified
        
        Args:
            point3d: 3D point [x,y,z] in meters
            src_frame: Source frame ('cam', 'end', 'base')
            end_pos: Current end effector pose [x,y,z,roll,pitch,yaw] in meters and degrees
            
        Returns:
            ndarray: Point in base coordinates (in meters)
        """
        
        
        # Step 1: Transform to end frame
        if src_frame == 'cam':
            point = np.dot(self.R_cam2end, point3d) + self.T_cam2end
        elif src_frame == 'end':
            # Already in end frame
            point = point3d
        elif src_frame == 'base':
            # Already in base frame
            return np.array(point3d)
        else:
            raise ValueError(f"Unknown source frame: {src_frame}")
        # Step 2: End frame to base frame
        if end_pos is None:
            _, end_pos = self.arm.get_position(return_gripper_center=True)
            end_pos[0:3] = np.array(end_pos[0:3]) * 0.001 # mm to m
        
        trans = np.array(end_pos[0:3])   
        
        # Convert Euler angles to rotation matrix
        rot = R.from_euler('xyz', end_pos[3:6], degrees=True)
        rot_matrix = rot.as_matrix()
        
        # Transform point (both point and trans are in meters)
        point_in_base = np.dot(rot_matrix, point) + trans
        
        
        return point_in_base
    
    def offset_from_end(self, point3d, src_frame='cam', end_pos=None):
        """
        Calculate offset from end effector - simplified
        
        Args:
            point3d: Target point
            src_frame: Source coordinate frame
            end_pos: Current end effector pose (meters)
            
        Returns:
            dict: Point coordinates and offset vector
        """
        point_in_base = self.point2base(point3d, src_frame, end_pos)
        
        if end_pos is None:
            _, end_pos = self.arm.get_position(return_gripper_center=True)
            end_pos_m = np.array(end_pos[0:3]) * 0.001 # mm to m
        else:
            end_pos_m = np.array(end_pos[0:3])
            
        offset = point_in_base - end_pos_m
        
        #offset[0] -= 0.020  # Adjust for gripper offset (30mm)
        #offset[1] += 0.020  # Adjust for gripper offset (30mm)
        offset[2] -= 0.005  # Adjust for gripper offset (30mm)
        
        
        eef_in_base = end_pos_m
        return dict(point_in_base=point_in_base, offset=offset,eef_in_base=eef_in_base)
    
    def pick(self, offset, angle=None, min_z=-150, gripper_width=None):
        """
        Execute pick operation - simplified V2 style
        
        Streamlined pick sequence:
        1. Move to target position
        2. Adjust gripper and orientation
        3. Execute pick and place
        
        Args:
            offset: Target offset [x,y,z] in meters
            angle: Grasp angle in degrees
            min_z: Minimum z height in mm
            gripper_width: Gripper width in meters
        """
        try:
            # Setup
            #self.arm.clean_error()
            #self.arm.motion_enable(True)
            #self.arm.set_mode(0)
            #self.arm.set_state(0)
            
            # Get current position
            _, old_pos = self.arm.get_position(return_gripper_center=True)
            new_pos = copy.deepcopy(old_pos)
            
            # Apply offset (m to mm conversion)
            print(f"DEBUG: offset(m)={offset}, old_pos={[round(x,1) for x in old_pos[:3]]}")
            for i in range(3):
                new_pos[i] += offset[i] * 1000  # m to mm
            print(f"DEBUG: 应用offset后 new_pos={[round(x,1) for x in new_pos[:3]]}")
            
            # Adjust grasp angle
            if angle is not None:
                #if angle > 90:
                #    angle -= 180
                if angle < 0:
                    angle  = angle + 180
                if angle > 180:
                    angle = angle - 180
                new_pos[5] -= angle  # Adjust yaw
            
            # Apply minimum z constraint
            if new_pos[2] < min_z:
                print(f"DEBUG: Z值{new_pos[2]:.1f}mm 低于min_z={min_z}mm，调整到{min_z}mm")
                new_pos[2] = min_z
            
            print(f'Pick: {[round(x,1) for x in old_pos[:3]]} → {[round(x,1) for x in new_pos[:3]]}, angle: {angle}')
            
            # Calculate gripper positions
            gripper_open = self.arm.gripper_max_units  # Default fully open
            gripper_close = self.arm.gripper_min_units  # Default fully closed
            
            if gripper_width is not None:
                # Convert gripper width (m) to units (0.001mm)
                width_units = int(gripper_width * 1000 * 1000)  # m → 0.001mm
                safety_margin = 60000  # 20mm safety margin
                close_margin = 60000   # 10mm close margin
                
                gripper_open = min(width_units + safety_margin, self.arm.gripper_max_units)
                gripper_close = max(width_units - close_margin, self.arm.gripper_min_units)
                
                print(f'Gripper: open={gripper_open/1000:.1f}mm, close={gripper_close/1000:.1f}mm')
            
            # Simplified pick sequence
            # 1. Move above target
            print(f"DEBUG: 发送移动命令到 X={new_pos[0]:.1f}, Y={new_pos[1]:.1f}")
           
            self.arm.set_position(x=new_pos[0], y=new_pos[1], wait=True, speed=30, use_gripper_center=True)
            if angle is not None:
                self.arm.set_position(yaw=new_pos[5], wait=True, use_gripper_center=True)
            
            _, pos1 = self.arm.get_position(return_gripper_center=True)
            _, gripper1 = self.arm.get_gripper_position()
            print(f"步骤1 - 移动到目标上方:")
            print(f"  预期位置: [{new_pos[0]:.1f}, {new_pos[1]:.1f}, 保持当前Z]")
            print(f"  实际位置: {[round(x,1) for x in pos1[:3]]}")
            print(f"  夹爪状态: 保持当前={gripper1/1000:.1f}mm")
            
            # 2. Open gripper  
            self.arm.set_gripper_position(gripper_open, wait=True)
            _, pos2 = self.arm.get_position(return_gripper_center=True)
            _, gripper2 = self.arm.get_gripper_position()
            print(f"步骤2 - 张开夹爪:")
            print(f"  预期夹爪: {gripper_open/1000:.1f}mm")
            print(f"  实际夹爪: {gripper2/1000:.1f}mm")
            print(f"  位置保持: {[round(x,1) for x in pos2[:3]]}")
            
            # 3. Move to grasp position
            print(f"DEBUG: 发送下降命令到 Z={new_pos[2]:.1f}")
           
            self.arm.set_position(x=new_pos[0], y=new_pos[1], z=new_pos[2], wait=True, speed=20, use_gripper_center=True)
            _, pos3 = self.arm.get_position(return_gripper_center=True)
            _, gripper3 = self.arm.get_gripper_position()
            print(f"步骤3 - 下降到抓取位置:")
            print(f"  预期位置: {[round(x,1) for x in new_pos[:3]]}")
            print(f"  实际位置: {[round(x,1) for x in pos3[:3]]}")
            print(f"  位置偏差: X={pos3[0]-new_pos[0]:.1f}, Y={pos3[1]-new_pos[1]:.1f}, Z={pos3[2]-new_pos[2]:.1f}mm")
            print(f"  夹爪保持: {gripper3/1000:.1f}mm")
            
            # 4. Close gripper
            self.arm.set_gripper_position(gripper_close, wait=True)
            _, pos4 = self.arm.get_position(return_gripper_center=True)
            _, gripper4 = self.arm.get_gripper_position()
            print(f"步骤4 - 闭合夹爪抓取:")
            print(f"  预期夹爪: {gripper_close/1000:.1f}mm")
            print(f"  实际夹爪: {gripper4/1000:.1f}mm")
            print(f"  位置保持: {[round(x,1) for x in pos4[:3]]}")
            
            # 5. Lift up
            lift_target_z = new_pos[2] + 200
            print(f"DEBUG: 发送抬升命令到 Z={lift_target_z:.1f}")
            
            self.arm.set_position(z=lift_target_z, wait=True, speed=30, use_gripper_center=True)  # Lift 200mm
            _, pos5 = self.arm.get_position(return_gripper_center=True)
            _, gripper5 = self.arm.get_gripper_position()
            print(f"步骤5 - 抬升200mm:")
            print(f"  预期位置: [保持X,Y, Z={lift_target_z:.1f}]")
            print(f"  实际位置: {[round(x,1) for x in pos5[:3]]}")
            print(f"  Z轴偏差: {pos5[2] - lift_target_z:.1f}mm")
            print(f"  夹爪保持: {gripper5/1000:.1f}mm (应为{gripper_close/1000:.1f}mm)")
            
            
            # 6. Simple place logic (move to side) - 原方案保留
            '''
            if new_pos[1] > 0:
                place_y = new_pos[1] - 250
            else:
                place_y = new_pos[1] + 250
            
            self.arm.set_position(y=place_y, wait=True, use_gripper_center=True)
            self.arm.set_position(z=new_pos[2], wait=True, use_gripper_center=True)
            _, pos6 = self.arm.get_position(return_gripper_center=True)
            _, gripper6 = self.arm.get_gripper_position()
            print(f"步骤6 - 移动到放置位置:")
            print(f"  预期位置: [保持X, Y={place_y:.1f}, Z={new_pos[2]:.1f}]")
            print(f"  实际位置: {[round(x,1) for x in pos6[:3]]}")
            print(f"  夹爪保持: {gripper6/1000:.1f}mm (应为{gripper_close/1000:.1f}mm)")
            '''
            
            # 6. Move to predefined place position with orientation - 新方案
            # 固定的放置位置是end_effector位置（单位：毫米）
            place_end_x = 16  # mm (0.016522 m)
            place_end_y = -200  # mm (-0.158215 m) 
            place_end_z = 200  # mm (0.207421 m)
            # 四元数姿态 (x, y, z, w)
            quat_x = 0.6626460983817538
            quat_y = 0.7280640627916741
            quat_z = -0.10109481693978242
            quat_w = 0.14353643007485029
            # 四元数转换为欧拉角
            rot = R.from_quat([quat_x, quat_y, quat_z, quat_w])
            euler_angles = rot.as_euler('xyz', degrees=True)  # 返回 [roll, pitch, yaw] 度数
            place_roll = euler_angles[0]
            place_pitch = euler_angles[1]
            place_yaw = euler_angles[2]
            
            # 转换为gripper center位置（考虑姿态的偏移变换）
            # T_gripper2flan[2] 是夹爪到法兰的Z轴偏移（米）
            gripper_offset_z = self.T_gripper2flan[2] * 1000  # 米转毫米，约100mm
            
            # 根据目标姿态旋转gripper偏移向量
            # 偏移向量在end effector坐标系中是 [0, 0, gripper_offset_z]
            # gripper在end effector前方，沿工具坐标系+Z方向
            offset_vector_local = np.array([0, 0, gripper_offset_z])
            
            # 使用目标姿态的旋转矩阵变换偏移向量到基座坐标系
            rot_matrix = rot.as_matrix()
            offset_vector_base = rot_matrix @ offset_vector_local
            
            # Gripper center位置 = End effector位置 + 旋转后的gripper偏移
            place_x = place_end_x + offset_vector_base[0]
            place_y = place_end_y + offset_vector_base[1]
            place_z = place_end_z + offset_vector_base[2]
            
            print(f"步骤6 - 移动到固定放置位置:")
            print(f"  End Effector目标: X={place_end_x:.1f}, Y={place_end_y:.1f}, Z={place_end_z:.1f} mm")
            print(f"  Gripper Center目标: X={place_x:.1f}, Y={place_y:.1f}, Z={place_z:.1f} mm")
            print(f"  目标姿态: Roll={place_roll:.1f}°, Pitch={place_pitch:.1f}°, Yaw={place_yaw:.1f}°")
            print(f"  Gripper偏移: {gripper_offset_z:.1f} mm")
            
            # 先移动到放置位置上方（避免碰撞）
            self.arm.set_position(z=place_z+100, wait=True, use_gripper_center=True)
            # 移动到放置位置的XY坐标和姿态
            self.arm.set_position(x=place_x, y=place_y, 
                                 roll=place_roll, pitch=place_pitch, yaw=place_yaw,
                                 wait=True, use_gripper_center=True)
            # 下降到放置高度
            self.arm.set_position(z=place_z, wait=True, use_gripper_center=True)
            _, pos6 = self.arm.get_position(return_gripper_center=True)
            _, gripper6 = self.arm.get_gripper_position()
            print(f"  实际位置: {[round(x,1) for x in pos6[:3]]}")
            print(f"  位置偏差: X={pos6[0]-place_x:.1f}, Y={pos6[1]-place_y:.1f}, Z={pos6[2]-place_z:.1f}mm")
            print(f"  夹爪保持: {gripper6/1000:.1f}mm (应为{gripper_close/1000:.1f}mm)")
            
            
            
            
            
            # 7. Release
            self.arm.set_gripper_position(self.arm.gripper_max_units, wait=True)
            _, pos7 = self.arm.get_position(return_gripper_center=True)
            _, gripper7 = self.arm.get_gripper_position()
            print(f"步骤7 - 释放物体:")
            print(f"  预期夹爪: {self.arm.gripper_max_units/1000:.1f}mm")
            print(f"  实际夹爪: {gripper7/1000:.1f}mm")
            print(f"  位置保持: {[round(x,1) for x in pos7[:3]]}")
            
            # 8. Return to start
            self.arm.set_position(z=pos7[2]+100,wait=True, use_gripper_center=True)
            self.arm.set_position(x=old_pos[0], y=old_pos[1], z=old_pos[2],yaw=old_pos[5], wait=True, use_gripper_center=True)
            _, pos8 = self.arm.get_position(return_gripper_center=True)
            _, gripper8 = self.arm.get_gripper_position()
            print(f"步骤8 - 返回初始位置:")
            print(f"  预期位置: {[round(x,1) for x in old_pos[:3]]}")
            print(f"  实际位置: {[round(x,1) for x in pos8[:3]]}")
            print(f"  位置偏差: X={pos8[0]-old_pos[0]:.1f}, Y={pos8[1]-old_pos[1]:.1f}, Z={pos8[2]-old_pos[2]:.1f}mm")
            print(f"  夹爪保持: {gripper8/1000:.1f}mm (已张开)")
            
            print("✓ Pick completed")
            
        except Exception as e:
            print(f"✗ Pick failed: {e}")
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def __del__(self):
        try:
            self.disconnect()
        except:
            pass





def create_config():
    """Create simplified Piper configuration"""
    cfg = Config()
    
    # Basic connection
    cfg.can_name = 'can0'
    
    # Initial position
    # cfg.init_pos = dict(x=345, y=0, z=380, roll=180, pitch=30, yaw=180)  #flange
    cfg.init_pos =dict(x=400, y=0, z=150, roll=180, pitch=30, yaw=180) # gripper center
    
    # Coordinate transforms (from existing config)
    cfg.T_cam2flan = [0.04010927904656854, 0.08283986339504643, 0.0025333968091171308]
    cfg.q_cam2flan = [0.14206540330899609, -0.1423312973808512, 0.6810236905433535, 0.7041064947060541]
    cfg.R_cam2flan = R.from_quat(cfg.q_cam2flan).as_matrix().tolist()
    #cfg.T_gripper2flan = [0.0, 0.0, 0.13503]
    cfg.T_gripper2flan = [0.0, 0.0, 0.1000]
    cfg.gripper_as_end = True
    
    # Gripper config
    cfg.gripper_max_mm = 90.0  # 90mm max range
    
    return cfg


if __name__ == '__main__':
    print("PiperRobot Simple Test")
    print("=" * 25)
    
    try:
        # Create config and robot
        cfg = create_config()
        
        with PiperRobot(cfg) as robot:
            robot.connect()
            
            # Basic position test
            _, pos = robot.arm.get_position()
            print(f"Current position: {pos[:3]} mm")
            
            # Coordinate transform test
            test_point = [0.1, 0.05, 0.3]  # Point in camera frame
            point_base = robot.point2base(test_point, src_frame='cam')
            print(f"Camera point {test_point} → Base point {point_base}")
            
            # Offset calculation test
            offset_data = robot.offset_from_end(test_point, src_frame='cam')
            print(f"Offset from end: {offset_data['offset']}")
            
            
            # Small movement test
            test_offset = [0.01, 0.01, -0.02]  # 1cm x, 1cm y, -2cm z
            print(f"Test offset: {test_offset}")
            # robot.pick(offset=test_offset, gripper_width=0.03)  # Uncomment to test
            
            print("✓ Test completed")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
