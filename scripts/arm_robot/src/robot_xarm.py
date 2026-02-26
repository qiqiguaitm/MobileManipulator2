import copy
import time

import numpy as np
from mmengine.config import Config
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R

class XArmRobot:
    """xArm机械臂控制类
    
    负责机械臂的连接、控制和坐标变换。实现了从相机坐标系到机器人基座坐标系
    的完整变换链，以及抓取动作的规划和执行。
    
    坐标系变换链：
    相机坐标系 -> 法兰坐标系 -> 夹爪坐标系 -> 基座坐标系
    """
    def __init__(self, cfg):
        """初始化机械臂控制器
        
        Args:
            cfg: 配置对象，包含：
                - ip: 机械臂IP地址
                - init_pos: 初始位置姿态
                - T_cam2flan: 相机到法兰的平移向量
                - R_cam2flan: 相机到法兰的旋转矩阵
                - T_gripper2flan: 夹爪到法兰的偏移
        """
        self.cfg = cfg
        self.init_pos = cfg.init_pos  # 初始位姿字典 {x,y,z,roll,pitch,yaw}
        self.T_cam2flan = np.array(cfg.T_cam2flan)  # 相机到法兰平移(m)
        self.R_cam2flan = np.array(cfg.R_cam2flan)  # 相机到法兰旋转
        self.T_gripper2flan = np.array(cfg.T_gripper2flan)  # 夹爪到法兰偏移(m)
        self.R_cam2end = self.R_cam2flan  # 末端执行器旋转
        
        # 根据配置确定末端执行器是法兰还是夹爪
        if cfg.get('gripper_as_end', True):
            self.T_cam2end = self.T_cam2flan - self.T_gripper2flan
        else:
            self.T_cam2end = self.T_cam2flan
        self.arm = None  # xArm API对象

    def connect(self):
        """连接并初始化机械臂
        
        执行步骤：
        1. 建立与机械臂的网络连接
        2. 清除错误状态
        3. 使能运动和夹爪
        4. 移动到初始位置
        5. 设置工具负载和偏移参数
        """
        # 创建机械臂连接
        arm = self.arm = XArmAPI(self.cfg.ip)
        time.sleep(0.5)
        
        # 清除错误并使能
        arm.clean_error()  # 清除控制器错误
        arm.clean_gripper_error()  # 清除夹爪错误
        arm.motion_enable(enable=True)  # 使能电机
        arm.set_gripper_enable(enable=True)  # 使能夹爪

        # 设置运动模式和状态
        arm.set_mode(0)  # 0: 位置控制模式
        arm.set_state(0)  # 0: 就绪状态
        
        # 移动到初始位置
        arm.set_position(**self.init_pos, wait=True)

        # 设置工具参数
        arm.set_tcp_load(0.1, [0, 0, 48])  # 负载质量0.1kg，重心偏移
        arm.set_tcp_offset([0, 0, 172, 0, 0, 0])  # TCP偏移172mm
        arm.set_gripper_position(850)  # 夹爪张开到最大

    def point2base(self, point3d, src_frame='end', end_pos=None):
        """将点从指定坐标系转换到基座坐标系
        
        支持的坐标系变换：
        - 相机坐标系 -> 末端坐标系 -> 基座坐标系
        - 末端坐标系 -> 基座坐标系
        - 基座坐标系 -> 基座坐标系（恒等变换）
        
        Args:
            point3d: 3D点坐标 [x,y,z]，单位米
            src_frame: 源坐标系 ('cam'|'end'|'base')
            end_pos: 末端执行器当前位姿 [x,y,z,roll,pitch,yaw]
            
        Returns:
            ndarray: 基座坐标系下的3D点坐标
        """
        # 第一步：转换到末端坐标系
        if src_frame == 'cam':
            # 相机坐标系到末端坐标系
            point = np.dot(self.R_cam2end, point3d) + self.T_cam2end
        elif src_frame == 'end':
            # 已在末端坐标系
            point = point3d
        elif src_frame == 'base':
            # 已在基座坐标系，直接返回
            return point3d
        else:
            raise NotImplemented
            
        # 第二步：末端坐标系到基座坐标系
        trans = end_pos[0:3]  # 末端位置
        # 欧拉角转旋转矩阵（外旋）
        rot = R.from_euler('xyz', end_pos[3:6], degrees=True)
        mat = rot.as_matrix()
        point_in_base = np.dot(mat, point) + trans
        return point_in_base

    def offset_from_end(self, point3d, src_frame='cam', end_pos=None):
        point_in_base = self.point2base(point3d=point3d, src_frame=src_frame, end_pos=end_pos)
        offset = point_in_base - end_pos[0:3]
        return dict(point_in_base=point_in_base, offset=offset)
    
    def get_position(self):
        return self.arm.get_position()

    def pick(self, offset, angle=None, min_z=0, gripper_width=None):
        """执行完整的抓取动作序列
        
        抓取流程：
        1. 移动到目标上方
        2. 调整抓取角度
        3. 张开夹爪到合适宽度
        4. 下降到抓取位置
        5. 闭合夹爪抓取
        6. 提升到安全高度
        7. 移动到放置位置
        8. 释放物体
        
        Args:
            offset: 目标偏移量 [x,y,z]，单位米
            angle: 抓取角度（度）
            min_z: 最小z高度限制（mm）
            gripper_width: 夹爪宽度（米）
        """
        arm = self.arm
        
        # 清除错误状态并使能
        arm.clean_error()
        arm.clean_gripper_error()
        arm.motion_enable(enable=True)
        arm.set_gripper_enable(enable=True)

        arm.set_mode(0)  # 位置控制模式
        arm.set_state(0)  # 就绪状态

        state, old_pos = self.arm.get_position()
        new_pos = list(copy.deepcopy(old_pos))
        if angle is not None:
            print(f'{new_pos[5]=} {angle=}')
            if angle > 90:
                angle -= 180
            new_pos[5] -= angle

        for i in range(len(offset)):
            new_pos[i] += offset[i] * 1000  # m to mm
        if new_pos[2] < min_z:
            new_pos[2] = min_z
        print(f'old_pos={[round(x, 1) for x in old_pos[:3]]} new_pos={[round(x, 1) for x in new_pos[:3]]} angle={angle:.1f}')

        arm.set_position(new_pos[0], new_pos[1])  # move to the top position upon the object
        arm.set_position(yaw=new_pos[5])  #
        gripper_width2 = 850
        if gripper_width is not None:
            print(f'{gripper_width=}')
            arm.set_gripper_position(gripper_width * 100 * 100 + 200, wait=True)  # in mm
            gripper_width2 = gripper_width * 100 * 100 - 400
            if gripper_width2 < 100:
                gripper_width2 = 100
        arm.set_position(*new_pos)

        # breakpoint()
        arm.set_gripper_position(gripper_width2, wait=True)
        # arm.set_tcp_load(0.82, [0, 0, 48])
        # arm.set_tcp_offset([0,0,172,0,0,0])
        if new_pos[1] > 0:
            off_y = -400
        else:
            off_y = 400
        # breakpoint()
        arm.set_position(new_pos[0], new_pos[1], new_pos[2] + 200)
        arm.set_position(new_pos[0], new_pos[1] + off_y, new_pos[2] + 200)
        arm.set_position(new_pos[0], new_pos[1] + off_y, new_pos[2])
        arm.set_gripper_position(850, wait=True)
        arm.set_tcp_load(0.1, [0, 0, 48])
        arm.set_position(new_pos[0], new_pos[1] + off_y, old_pos[2], wait=True)
        arm.set_position(yaw=old_pos[5], wait=True)
        arm.set_position(*old_pos, wait=True)
        time.sleep(0.2)


if __name__ == '__main__':
    # Example usage
    cfg = Config()
    cfg.ip = '192.168.1.236'
    cfg.init_pos = dict(x=270, y=0, z=307, roll=-180, pitch=0, yaw=0)
    cfg.T_cam2flan = [0.06644488210075013, -0.034881058367930894, 0.023248884204089854]
    q = [
        0.000849727426002991,
        -0.002441518543559673,
        0.7125339065369926,
        0.7016329161218389,
    ]
    cfg.R_cam2flan = R.from_quat(q).as_matrix().tolist()
    cfg.T_gripper2flan = [0, 0, 0.172]
    arm = XArmRobot(cfg)
    arm.connect()

    arm.arm.set_mode(0)
    arm.arm.set_state(0)
    arm.arm.set_position(yaw=90, wait=True)
    if 0:
        _, flan_pos = arm.arm.get_position()
        flan_pos = np.array(flan_pos)
        flan_pos[0:3] /= 1000.0 #mm to m
        p0 = arm.flan2base(point3d=[0,0,0], flan_pos=flan_pos)
        p1 = arm.flan2base(point3d=[1, 0, 0], flan_pos=flan_pos)
        p2 = arm.flan2base(point3d=[0, 1, 0], flan_pos=flan_pos)
        p3 = arm.flan2base(point3d=[0, 0, 1], flan_pos=flan_pos)
        p4 = arm.flan2base(point3d=[1, 1, 0], flan_pos=flan_pos)

        p5 = arm.gripper_in_base(flan_pos=flan_pos)
        p6 = arm.flan2base(point3d=arm.T_gripper2flan, flan_pos=flan_pos)



    _, end_pos = arm.arm.get_position()
    end_pos = np.array(end_pos)
    end_pos[0:3] /= 1000.0 # mm to m

    p0 = arm.point2base(point3d=[0,0,0], src_frame='cam', end_pos=end_pos)
    p1 = arm.point2base(point3d=[0,0,0], src_frame='end', end_pos=end_pos)
    p2 = arm.point2base(point3d=[0,0,0], src_frame='base', end_pos=end_pos)



    p3 = arm.point2base(point3d=[1,0,0], src_frame='cam', end_pos=end_pos) 
    p4 = arm.point2base(point3d=[0,1,0], src_frame='cam', end_pos=end_pos)

    p5 = arm.point2base(point3d=[1,0,0], src_frame='end', end_pos=end_pos) 
    p6 = arm.point2base(point3d=[0,1,0], src_frame='end', end_pos=end_pos)

    q0 = arm.offset_from_end(point3d=[0,0,0], src_frame='cam', end_pos=end_pos)
    q1 = arm.offset_from_end(point3d=[0,0,0], src_frame='end', end_pos=end_pos)
    q2 = arm.offset_from_end(point3d=[0,0,0], src_frame='base', end_pos=end_pos)



    q3 = arm.offset_from_end(point3d=[1,0,0], src_frame='cam', end_pos=end_pos) 
    q4 = arm.offset_from_end(point3d=[0,1,0], src_frame='cam', end_pos=end_pos)
    q5 = arm.offset_from_end(point3d=[0,0,1], src_frame='cam', end_pos=end_pos)
    
    q6 = arm.offset_from_end(point3d=[1,0,0], src_frame='end', end_pos=end_pos) 
    q7 = arm.offset_from_end(point3d=[0,1,0], src_frame='end', end_pos=end_pos)
    q8 = arm.offset_from_end(point3d=[0,0,1], src_frame='end', end_pos=end_pos)
      
    breakpoint()