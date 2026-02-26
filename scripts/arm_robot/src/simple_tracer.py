#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracer底盘控制演示程序
实现简单的运动控制序列，包括直行、旋转和组合运动
"""

import rospy
import math
import time
from geometry_msgs.msg import Twist
from tracer_msgs.msg import TracerStatus, TracerLightCmd

class SimpleTracer:
    """Tracer底盘控制类"""
    
    def __init__(self):
        """初始化底盘控制器"""
        # 初始化ROS节点（如果尚未初始化）
        if not rospy.core.is_initialized():
            rospy.init_node('tracer_demo_node', anonymous=True)
        
        # 创建速度命令发布者
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 创建灯光控制发布者（用于切换控制模式）
        self.light_pub = rospy.Publisher('/tracer_light_control', TracerLightCmd, queue_size=10)
        
        # 创建状态订阅者
        self.status_sub = rospy.Subscriber('/tracer_status', TracerStatus, self.status_callback)
        
        # 控制参数
        self.linear_speed = 0.2  # 线速度 m/s (安全速度)
        self.angular_speed = 0.3  # 角速度 rad/s
        self.control_rate = 10  # 控制频率 Hz
        self.rate = rospy.Rate(self.control_rate)
        
        # 状态变量
        self.current_status = None
        self.position_x = 0.0
        self.position_y = 0.0
        self.heading = 0.0  # 航向角（弧度）
        
        # 状态监控变量
        self.status_update_time = None
        self.status_timeout = 1.0  # 状态超时时间（秒）
        
        # 等待连接建立
        rospy.sleep(0.5)
        rospy.loginfo("SimpleTracer初始化完成")
        
    def status_callback(self, msg):
        """状态回调函数"""
        self.current_status = msg
        self.status_update_time = rospy.Time.now()
        
    def stop(self):
        """停止运动"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        rospy.sleep(0.1)
    
    def get_status(self):
        """
        获取底盘当前状态
        
        Returns:
            dict: 包含底盘状态信息的字典，如果无状态则返回None
        """
        if self.current_status is None:
            return None
            
        # 检查状态是否超时
        if self.status_update_time is not None:
            time_diff = (rospy.Time.now() - self.status_update_time).to_sec()
            if time_diff > self.status_timeout:
                rospy.logwarn(f"状态更新超时: {time_diff:.2f}秒未更新")
                
        status_dict = {
            'linear_velocity': self.current_status.linear_velocity,
            'angular_velocity': self.current_status.angular_velocity,
            'battery_voltage': self.current_status.battery_voltage,
            'base_state': self.current_status.base_state,
            'control_mode': self.current_status.control_mode,
            'fault_code': self.current_status.fault_code,
            'left_odometer': self.current_status.left_odomter,
            'right_odometer': self.current_status.right_odomter,
            'light_enabled': self.current_status.light_control_enabled,
            'motor_states': []
        }
        
        # 添加电机状态
        for i, motor in enumerate(self.current_status.motor_states):
            motor_info = {
                'id': i,
                'rpm': motor.rpm
            }
            status_dict['motor_states'].append(motor_info)
            
        return status_dict
    
    def print_status(self):
        """打印底盘状态信息"""
        status = self.get_status()
        if status is None:
            rospy.logwarn("未接收到底盘状态")
            return
            
        rospy.loginfo("===== 底盘状态 =====")
        rospy.loginfo(f"线速度: {status['linear_velocity']:.3f} m/s")
        rospy.loginfo(f"角速度: {status['angular_velocity']:.3f} rad/s")
        rospy.loginfo(f"电池电压: {status['battery_voltage']:.2f} V")
        rospy.loginfo(f"控制模式: {status['control_mode']}")
        rospy.loginfo(f"底盘状态: {status['base_state']}")
        rospy.loginfo(f"故障代码: {status['fault_code']}")
        rospy.loginfo(f"左轮里程: {status['left_odometer']:.3f} m")
        rospy.loginfo(f"右轮里程: {status['right_odometer']:.3f} m")
        
        # 打印电机状态
        for motor in status['motor_states']:
            rospy.loginfo(f"电机{motor['id']}: {motor['rpm']:.1f} RPM")
            
    def check_battery(self, min_voltage=22.0):
        """
        检查电池电压
        
        Args:
            min_voltage: 最低电压阈值（默认22V）
            
        Returns:
            bool: 电压是否正常
        """
        status = self.get_status()
        if status is None:
            rospy.logwarn("无法获取电池状态")
            return False
            
        voltage = status['battery_voltage']
        if voltage < min_voltage:
            rospy.logwarn(f"电池电压过低: {voltage:.2f}V (最低: {min_voltage}V)")
            return False
            
        rospy.loginfo(f"电池电压正常: {voltage:.2f}V")
        return True
    
    def check_fault(self):
        """
        检查故障状态
        
        Returns:
            bool: 是否无故障
        """
        status = self.get_status()
        if status is None:
            rospy.logwarn("无法获取故障状态")
            return False
            
        if status['fault_code'] != 0:
            rospy.logerr(f"检测到故障! 故障代码: 0x{status['fault_code']:04X}")
            return False
            
        return True
    
    def is_connected(self):
        """
        检查与底盘的连接状态
        
        Returns:
            bool: 是否已连接
        """
        if self.current_status is None:
            return False
            
        if self.status_update_time is None:
            return False
            
        time_diff = (rospy.Time.now() - self.status_update_time).to_sec()
        return time_diff < self.status_timeout
    
    def enable_command_mode(self):
        """
        启用指令控制模式
        通过发送灯光控制命令来激活底盘的指令模式
        """
        rospy.loginfo("尝试启用指令控制模式...")
        
        # 发送灯光控制命令以激活指令模式
        light_cmd = TracerLightCmd()
        light_cmd.enable_cmd_light_control = True
        light_cmd.front_mode = TracerLightCmd.LIGHT_CONST_OFF
        light_cmd.rear_mode = TracerLightCmd.LIGHT_CONST_OFF
        light_cmd.front_custom_value = 0
        light_cmd.rear_custom_value = 0
        
        # 发布几次确保命令被接收
        for _ in range(3):
            self.light_pub.publish(light_cmd)
            rospy.sleep(0.1)
        
        # 发送一个微小的速度命令来激活控制
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(3):
            self.cmd_pub.publish(twist)
            rospy.sleep(0.1)
            
        rospy.loginfo("指令控制模式启用命令已发送")
        rospy.sleep(0.5)  # 等待模式切换
    
    def check_and_fix_fault(self):
        """
        检查并尝试修复故障
        
        Returns:
            bool: 是否成功修复或无故障
        """
        status = self.get_status()
        if status is None:
            rospy.logwarn("无法获取状态")
            return False
            
        fault_code = status['fault_code']
        control_mode = status['control_mode']
        
        rospy.loginfo(f"当前控制模式: {control_mode}, 故障代码: 0x{fault_code:04X}")
        
        # 故障代码0x0004通常表示需要切换到指令模式
        if fault_code == 0x0004 or control_mode != 1:
            rospy.loginfo("检测到需要切换控制模式")
            self.enable_command_mode()
            
            # 重新检查状态
            rospy.sleep(1)
            status = self.get_status()
            if status:
                fault_code = status['fault_code']
                control_mode = status['control_mode']
                rospy.loginfo(f"模式切换后 - 控制模式: {control_mode}, 故障代码: 0x{fault_code:04X}")
                
        # 如果故障代码为0，表示无故障
        if fault_code == 0:
            return True
        elif fault_code == 0x0004:
            rospy.logwarn("故障0x0004: 可能需要手动切换到指令模式（检查遥控器）")
            return False
        else:
            rospy.logwarn(f"未知故障代码: 0x{fault_code:04X}")
            return False
        
    def move_forward(self, distance, speed=None):
        """
        向前直行指定距离
        
        Args:
            distance: 移动距离（米），正值向前，负值向后
            speed: 线速度（米/秒），默认使用预设速度
        """
        if speed is None:
            speed = self.linear_speed
            
        # 计算运动时间
        duration = abs(distance / speed)
        
        # 设置速度命令
        twist = Twist()
        twist.linear.x = speed if distance > 0 else -speed
        twist.angular.z = 0.0
        
        # 发送速度命令
        start_time = rospy.Time.now()
        rospy.loginfo(f"直行 {distance:.2f}m，速度 {speed:.2f}m/s")
        
        while (rospy.Time.now() - start_time).to_sec() < duration:
            if rospy.is_shutdown():
                break
            self.cmd_pub.publish(twist)
            self.rate.sleep()
            
        # 停止
        self.stop()
        
        # 更新位置估计
        self.position_x += distance * math.cos(self.heading)
        self.position_y += distance * math.sin(self.heading)
        
        rospy.loginfo(f"直行完成，当前位置: ({self.position_x:.2f}, {self.position_y:.2f})")
        
    def rotate(self, angle_deg, angular_speed=None):
        """
        旋转指定角度
        
        Args:
            angle_deg: 旋转角度（度），正值逆时针，负值顺时针
            angular_speed: 角速度（弧度/秒），默认使用预设速度
        """
        if angular_speed is None:
            angular_speed = self.angular_speed
            
        # 转换为弧度
        angle_rad = math.radians(angle_deg)
        
        # 计算旋转时间
        duration = abs(angle_rad / angular_speed)
        
        # 设置速度命令
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed if angle_rad > 0 else -angular_speed
        
        # 发送速度命令
        start_time = rospy.Time.now()
        rospy.loginfo(f"旋转 {angle_deg:.1f}度，角速度 {angular_speed:.2f}rad/s")
        
        while (rospy.Time.now() - start_time).to_sec() < duration:
            if rospy.is_shutdown():
                break
            self.cmd_pub.publish(twist)
            self.rate.sleep()
            
        # 停止
        self.stop()
        
        # 更新航向
        self.heading += angle_rad
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))  # 归一化到[-π, π]
        
        rospy.loginfo(f"旋转完成，当前航向: {math.degrees(self.heading):.1f}度")
        
    def move_step1(self):
        """
        步骤1: 向前直行30cm
        """
        rospy.loginfo("=== 执行步骤1: 向前30cm ===")
        
        # 向前直行30cm
        self.move_forward(0.4)
        
    def move_step2(self):
        """
        步骤2: 逆时针旋转90度，直行100cm，顺时针旋转60度
        """
        rospy.loginfo("=== 执行步骤2: 左转90度->前进100cm->右转60度 ===")
        
        # 逆时针旋转90度
        self.rotate(90)
        rospy.sleep(0.5)
        
        # 直行100cm
        self.move_forward(1.1)
        rospy.sleep(0.5)
        
        # 顺时针旋转60度
        self.rotate(-60)
        
    def move_step3(self):
        """
        步骤3: 后退40cm，顺时针旋转45度，直行120cm，顺时针旋转30度
        """
        rospy.loginfo("=== 执行步骤3: 后退40cm->右转45度->前进120cm->右转30度 ===")
        
        # 后退40cm
        self.move_forward(-0.4)
        rospy.sleep(0.5)
        
        # 顺时针旋转45度
        self.rotate(-45)
        rospy.sleep(0.5)
        
        # 直行120cm
        self.move_forward(1.3)
        rospy.sleep(0.5)
        
        # 顺时针旋转30度
        self.rotate(-30)
        
    def move_step4(self):
        """
        步骤4: 直接回到起点
        """
        rospy.loginfo("=== 执行步骤4: 直接回到起点 ===")
        
        rospy.loginfo(f"当前位置: ({self.position_x:.2f}, {self.position_y:.2f})")
        rospy.loginfo(f"当前航向: {math.degrees(self.heading):.1f}度")
        
        # 计算回到原点的距离和角度
        distance_to_origin = math.sqrt(self.position_x**2 + self.position_y**2)
        angle_to_origin = math.atan2(-self.position_y, -self.position_x)
        
        # 计算需要旋转的角度
        rotation_needed = angle_to_origin - self.heading
        # 归一化角度到[-π, π]
        rotation_needed = math.atan2(math.sin(rotation_needed), math.cos(rotation_needed))
        rotation_needed_deg = math.degrees(rotation_needed)
        
        rospy.loginfo(f"距离原点: {distance_to_origin:.2f}m")
        rospy.loginfo(f"需要旋转: {rotation_needed_deg:.1f}度")
        
        # 转向原点
        if abs(rotation_needed_deg) > 1.0:  # 只有角度差大于1度时才旋转
            self.rotate(rotation_needed_deg)
            rospy.sleep(0.5)
        
        # 直行到原点
        if distance_to_origin > 0.05:  # 只有距离大于5cm时才移动
            self.move_forward(distance_to_origin)
            rospy.sleep(0.5)
        
        # 转回初始朝向（面向x轴正方向）
        final_rotation = -math.degrees(self.heading)
        if abs(final_rotation) > 1.0:
            self.rotate(final_rotation)
        
        # 重置位置和朝向
        self.position_x = 0.0
        self.position_y = 0.0
        self.heading = 0.0
        
        rospy.loginfo(f"返回后位置: ({self.position_x:.2f}, {self.position_y:.2f})")
        rospy.loginfo(f"返回后航向: {math.degrees(self.heading):.1f}度")
        rospy.loginfo("直接回到起点完成！")
        
    def execute_all_steps(self):
        """执行所有步骤"""
        try:
            # 检查初始状态
            rospy.loginfo("检查底盘状态...")
            rospy.sleep(0.5)  # 等待状态更新
            
            if not self.is_connected():
                rospy.logerr("底盘未连接！请检查连接")
                return
                
            if not self.check_battery():
                rospy.logerr("电池电压异常！请充电")
                return
                
            if not self.check_and_fix_fault():
                rospy.logerr("底盘存在故障且无法自动修复！")
                rospy.loginfo("请检查:")
                rospy.loginfo("1. 遥控器是否在指令模式")
                rospy.loginfo("2. 急停按钮是否按下")
                rospy.loginfo("3. 底盘电源是否正常")
                return
                
            # 打印初始状态
            self.print_status()
            rospy.sleep(1.0)
            
            # 执行步骤1
            self.move_step1()
            rospy.loginfo("步骤1完成，等待3秒...")
            rospy.sleep(3.0)
            
            # 执行步骤2
            self.move_step2()
            rospy.loginfo("步骤2完成，等待3秒...")
            rospy.sleep(3.0)
            
            # 执行步骤3
            self.move_step3()
            rospy.loginfo("步骤3完成，等待3秒...")
            rospy.sleep(3.0)
            
            # 执行步骤4
            #self.move_step4()
            
            # 打印最终状态
            rospy.loginfo("=== 所有步骤执行完成 ===")
            self.print_status()
            
        except rospy.ROSInterruptException:
            rospy.logwarn("程序被中断")
            self.stop()
        except Exception as e:
            rospy.logerr(f"执行过程中出错: {e}")
            self.stop()
            

def main():
    """主函数 - 实车测试"""
    try:
        # 创建Tracer控制器
        tracer = SimpleTracer()
        
        # 等待用户确认
        rospy.loginfo("========================================")
        rospy.loginfo("Tracer底盘控制演示程序")
        rospy.loginfo("确保:")
        rospy.loginfo("1. 底盘已通过can1接口连接")
        rospy.loginfo("2. roslaunch tracer_bringup tracer_robot_base.launch 已运行")
        rospy.loginfo("3. 周围有足够的空间（至少2m x 2m）")
        rospy.loginfo("========================================")
        rospy.loginfo("功能说明:")
        rospy.loginfo("- get_status(): 获取底盘状态字典")
        rospy.loginfo("- print_status(): 打印详细状态信息")
        rospy.loginfo("- check_battery(): 检查电池电压")
        rospy.loginfo("- check_fault(): 检查故障状态")
        rospy.loginfo("- is_connected(): 检查连接状态")
        rospy.loginfo("========================================")
        rospy.loginfo("按Ctrl+C可随时停止程序")
        rospy.loginfo("3秒后开始执行...")
        rospy.sleep(3.0)
        
        # 执行所有步骤
        tracer.execute_all_steps()
        
        # 确保停止
        tracer.stop()
        rospy.loginfo("程序结束")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"程序出错: {e}")
        

if __name__ == '__main__':
    main()
