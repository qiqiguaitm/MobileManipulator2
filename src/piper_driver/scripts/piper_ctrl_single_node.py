#!/usr/bin/env python3
# -*-coding:utf8-*-
# ROS2 node for controlling single Piper arm with gripper
# Converted from ROS1 version

from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PiperStatusMsg, PosCmd
from piper_msgs.srv import Enable, Gripper, GoZero

import time
import threading
import math
import numpy as np

# For quaternion conversion (ROS2 uses scipy instead of tf.transformations)
from scipy.spatial.transform import Rotation as R

# Piper SDK
from piper_sdk import C_PiperInterface


def quaternion_from_euler(roll, pitch, yaw):
    """Convert euler angles to quaternion (x, y, z, w)"""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    q = r.as_quat()  # Returns [x, y, z, w]
    return q


class PiperCtrlSingleNode(Node):
    """ROS2 node for Piper arm control"""

    def __init__(self):
        super().__init__('piper_ctrl_single_node')

        # Declare and get parameters
        self.declare_parameter('can_port', 'can0')
        self.declare_parameter('auto_enable', False)
        self.declare_parameter('gripper_exist', True)
        self.declare_parameter('rviz_ctrl_flag', False)
        self.declare_parameter('gripper_val_mutiple', 1.0)

        self.can_port = self.get_parameter('can_port').value
        self.auto_enable = self.get_parameter('auto_enable').value
        self.gripper_exist = self.get_parameter('gripper_exist').value
        self.rviz_ctrl_flag = self.get_parameter('rviz_ctrl_flag').value
        self.gripper_val_mutiple = self.get_parameter('gripper_val_mutiple').value

        self.get_logger().info(f'can_port: {self.can_port}')
        self.get_logger().info(f'auto_enable: {self.auto_enable}')
        self.get_logger().info(f'gripper_exist: {self.gripper_exist}')
        self.get_logger().info(f'gripper_val_mutiple: {self.gripper_val_mutiple}')

        if not self.can_port:
            self.get_logger().error('can_port parameter not set')
            raise RuntimeError('can_port parameter required')

        # Validate gripper_val_mutiple
        if not isinstance(self.gripper_val_mutiple, (int, float)) or self.gripper_val_mutiple <= 0:
            self.get_logger().warn('Invalid gripper_val_mutiple, using default 1.0')
            self.gripper_val_mutiple = 1.0

        # QoS profile for real-time control
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states_single', qos)
        self.arm_status_pub = self.create_publisher(PiperStatusMsg, 'arm_status', qos)
        self.end_pose_euler_pub = self.create_publisher(PosCmd, 'end_pose_euler', qos)
        self.end_pose_pub = self.create_publisher(PoseStamped, 'end_pose', qos)

        # Services
        self.enable_service = self.create_service(Enable, 'enable_srv', self.handle_enable_service)
        self.gripper_service = self.create_service(Gripper, 'gripper_srv', self.handle_gripper_service)
        self.stop_service = self.create_service(Trigger, 'stop_srv', self.handle_stop_service)
        self.reset_service = self.create_service(Trigger, 'reset_srv', self.handle_reset_service)
        self.go_zero_service = self.create_service(GoZero, 'go_zero_srv', self.handle_go_zero_service)

        # Subscriptions
        self.pos_sub = self.create_subscription(PosCmd, 'pos_cmd', self.pos_callback, qos)
        self.joint_sub = self.create_subscription(JointState, 'joint_ctrl_single', self.joint_callback, qos)
        self.enable_sub = self.create_subscription(Bool, 'enable_flag', self.enable_callback, qos)

        # Internal state
        self._enable_flag = False
        self.joint_states = JointState()
        self.joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        self.joint_states.position = [0.0] * 8
        self.joint_states.velocity = [0.0] * 6
        self.joint_states.effort = [0.0] * 8

        # Initialize Piper SDK
        self.get_logger().info(f'Connecting to Piper on {self.can_port}...')
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0xad)
        self.get_logger().info('Piper connected')

        # Auto-enable if configured
        if self.auto_enable:
            self._auto_enable_arm()

        # Create timer for publishing at 200Hz
        self.create_timer(0.005, self.publish_callback)

    def _auto_enable_arm(self):
        """Auto-enable arm with timeout"""
        self.get_logger().info('Auto-enabling arm...')
        timeout = 5.0
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.get_logger().error('Auto-enable timeout')
                return

            enable_flag = all([
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status,
            ])

            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)

            if enable_flag:
                self._enable_flag = True
                self.get_logger().info('Arm enabled successfully')
                return

            time.sleep(1.0)

    def get_enable_flag(self):
        return self._enable_flag

    def publish_callback(self):
        """Timer callback for publishing arm state"""
        self.publish_arm_state()
        self.publish_arm_end_pose()
        self.publish_arm_joint_and_gripper()

    def publish_arm_state(self):
        """Publish arm status message"""
        arm_status = PiperStatusMsg()
        status = self.piper.GetArmStatus().arm_status

        arm_status.ctrl_mode = status.ctrl_mode
        arm_status.arm_status = status.arm_status
        arm_status.mode_feedback = status.mode_feed
        arm_status.teach_status = status.teach_status
        arm_status.motion_status = status.motion_status
        arm_status.trajectory_num = status.trajectory_num
        arm_status.err_code = status.err_code

        err_status = status.err_status
        arm_status.joint_1_angle_limit = err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = err_status.communication_status_joint_1
        arm_status.communication_status_joint_2 = err_status.communication_status_joint_2
        arm_status.communication_status_joint_3 = err_status.communication_status_joint_3
        arm_status.communication_status_joint_4 = err_status.communication_status_joint_4
        arm_status.communication_status_joint_5 = err_status.communication_status_joint_5
        arm_status.communication_status_joint_6 = err_status.communication_status_joint_6

        self.arm_status_pub.publish(arm_status)

    def publish_arm_joint_and_gripper(self):
        """Publish joint states and gripper position"""
        joint_msgs = self.piper.GetArmJointMsgs().joint_state
        gripper_msgs = self.piper.GetArmGripperMsgs().gripper_state
        high_spd_msgs = self.piper.GetArmHighSpdInfoMsgs()

        # Convert joint angles from SDK units to radians
        # SDK: degrees * 1000, need to convert to radians
        deg_to_rad = 0.017444  # pi/180 * 1000 factor

        joint_0 = (joint_msgs.joint_1 / 1000) * deg_to_rad
        joint_1 = (joint_msgs.joint_2 / 1000) * deg_to_rad
        joint_2 = (joint_msgs.joint_3 / 1000) * deg_to_rad
        joint_3 = (joint_msgs.joint_4 / 1000) * deg_to_rad
        joint_4 = (joint_msgs.joint_5 / 1000) * deg_to_rad
        joint_5 = (joint_msgs.joint_6 / 1000) * deg_to_rad

        # Gripper position (meters), joint7 and joint8 move symmetrically
        gripper_pos = gripper_msgs.grippers_angle / 1000000
        joint_7 = gripper_pos
        joint_8 = -gripper_pos

        # Velocities
        vel_0 = high_spd_msgs.motor_1.motor_speed / 1000
        vel_1 = high_spd_msgs.motor_2.motor_speed / 1000
        vel_2 = high_spd_msgs.motor_3.motor_speed / 1000
        vel_3 = high_spd_msgs.motor_4.motor_speed / 1000
        vel_4 = high_spd_msgs.motor_5.motor_speed / 1000
        vel_5 = high_spd_msgs.motor_6.motor_speed / 1000

        effort_7 = gripper_msgs.grippers_effort / 1000

        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        self.joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_7, joint_8]
        self.joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
        self.joint_states.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_7, effort_7]

        self.joint_pub.publish(self.joint_states)

    def publish_arm_end_pose(self):
        """Publish end effector pose"""
        end_pose_data = self.piper.GetArmEndPoseMsgs().end_pose

        # Position in meters
        x = end_pose_data.X_axis / 1000000
        y = end_pose_data.Y_axis / 1000000
        z = end_pose_data.Z_axis / 1000000

        # Orientation in radians
        roll = math.radians(end_pose_data.RX_axis / 1000)
        pitch = math.radians(end_pose_data.RY_axis / 1000)
        yaw = math.radians(end_pose_data.RZ_axis / 1000)

        # Publish PoseStamped with quaternion
        endpos = PoseStamped()
        endpos.header.stamp = self.get_clock().now().to_msg()
        endpos.header.frame_id = 'base_link'
        endpos.pose.position.x = x
        endpos.pose.position.y = y
        endpos.pose.position.z = z

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        self.end_pose_pub.publish(endpos)

        # Publish PosCmd with euler angles
        end_pose_euler = PosCmd()
        end_pose_euler.x = x
        end_pose_euler.y = y
        end_pose_euler.z = z
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        if self.gripper_exist:
            end_pose_euler.gripper = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        else:
            end_pose_euler.gripper = -1.0
        end_pose_euler.mode1 = 0
        end_pose_euler.mode2 = 0
        self.end_pose_euler_pub.publish(end_pose_euler)

    def pos_callback(self, pos_data: PosCmd):
        """Position command callback"""
        factor = 180 / math.pi
        x = round(pos_data.x * 1000) * 1000
        y = round(pos_data.y * 1000) * 1000
        z = round(pos_data.z * 1000) * 1000
        rx = round(pos_data.roll * 1000 * factor)
        ry = round(pos_data.pitch * 1000 * factor)
        rz = round(pos_data.yaw * 1000 * factor)

        self.get_logger().info(f'Received PosCmd: x={x}, y={y}, z={z}, rx={rx}, ry={ry}, rz={rz}')

        if self.get_enable_flag():
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 50, 0xad)
            self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)

            gripper = round(pos_data.gripper * 1000 * 1000)
            gripper = max(0, min(gripper, 80000))

            if self.gripper_exist:
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 50, 0xad)

    def joint_callback(self, joint_data: JointState):
        """Joint control callback"""
        factor = 1000 * 180 / np.pi

        joint_0 = round(joint_data.position[0] * factor)
        joint_1 = round(joint_data.position[1] * factor)
        joint_2 = round(joint_data.position[2] * factor)
        joint_3 = round(joint_data.position[3] * factor)
        joint_4 = round(joint_data.position[4] * factor)
        joint_5 = round(joint_data.position[5] * factor)

        joint_6 = None
        if len(joint_data.position) >= 7:
            joint_6 = round(joint_data.position[6] * 1000 * 1000)
            joint_6 = int(joint_6 * self.gripper_val_mutiple)
            joint_6 = max(0, min(joint_6, 80000))

        if self.get_enable_flag():
            # Set motor speed
            if joint_data.velocity and not all(v == 0 for v in joint_data.velocity):
                if len(joint_data.velocity) == 7:
                    vel_all = max(0, min(round(joint_data.velocity[6]), 100))
                    self.piper.MotionCtrl_2(0x01, 0x01, vel_all, 0xad)
                else:
                    self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xad)
            else:
                self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xad)

            # Send joint positions
            self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

            # Gripper control
            if self.gripper_exist and joint_6 is not None:
                if abs(joint_6) < 200:
                    joint_6 = 0
                if len(joint_data.effort) >= 7:
                    gripper_effort = max(0.5, min(joint_data.effort[6], 3.0))
                    gripper_effort = round(gripper_effort * 1000)
                    self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)
                else:
                    self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    def enable_callback(self, enable_flag: Bool):
        """Enable/disable arm callback"""
        self.get_logger().info(f'Received enable flag: {enable_flag.data}')
        if enable_flag.data:
            self._enable_flag = True
            self.piper.EnableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self._enable_flag = False
            self.piper.DisableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x00, 0)

    def handle_enable_service(self, request, response):
        """Enable service handler"""
        self.get_logger().info(f'Enable request: {request.enable_request}')
        timeout = 5.0
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.get_logger().warn('Enable timeout')
                response.enable_response = False
                return response

            enable_list = [
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status,
            ]

            if request.enable_request:
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            else:
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)

            self._enable_flag = enable_flag

            if enable_flag == request.enable_request:
                response.enable_response = True
                return response

            time.sleep(0.5)

    def handle_gripper_service(self, request, response):
        """Gripper service handler"""
        response.code = 15999
        response.status = False

        if not self.gripper_exist:
            self.get_logger().warn('Gripper does not exist')
            response.code = 15903
            return response

        self.get_logger().info(f'Gripper request: angle={request.gripper_angle}, effort={request.gripper_effort}')

        gripper_angle = round(max(0, min(request.gripper_angle, 0.07)) * 1e6)
        gripper_effort = round(max(0.5, min(request.gripper_effort, 2)) * 1e3)

        gripper_code = request.gripper_code
        if gripper_code not in [0x00, 0x01, 0x02, 0x03]:
            self.get_logger().warn('Invalid gripper_code, using default 1')
            gripper_code = 1
            response.code = 15901

        set_zero = request.set_zero
        if set_zero not in [0x00, 0xAE]:
            self.get_logger().warn('Invalid set_zero, using default 0')
            set_zero = 0
            response.code = 15902

        self.piper.GripperCtrl(abs(gripper_angle), gripper_effort, gripper_code, set_zero)
        response.code = 15900
        response.status = True
        return response

    def handle_stop_service(self, request, response):
        """Stop service handler"""
        self.get_logger().info('Stop request received')
        self.piper.MotionCtrl_1(0x01, 0, 0)
        response.success = True
        response.message = 'Stop piper success'
        return response

    def handle_reset_service(self, request, response):
        """Reset service handler"""
        self.get_logger().info('Reset request received')
        self.piper.MotionCtrl_1(0x02, 0, 0)
        response.success = True
        response.message = 'Reset piper success'
        return response

    def handle_go_zero_service(self, request, response):
        """Go to zero position service handler"""
        self.get_logger().info('Go zero request received')
        if request.is_mit_mode:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        response.status = True
        response.code = 151001
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PiperCtrlSingleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
