#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Master/Slave dual-arm control node for Piper robotic arm
ROS2 version

This node manages master/slave arm communication:
- mode=0: Read and publish both master and slave arm messages
- mode=1: Control slave arm, subscribe to master arm commands
"""

from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import time
import threading

from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperStatusMsg, PosCmd

# Scipy for quaternion conversion (replaces tf.transformations)
from scipy.spatial.transform import Rotation as R


class PiperMSNode(Node):
    """Master/Slave Piper arm ROS2 node"""

    def __init__(self):
        super().__init__('piper_start_ms_node')

        # Declare parameters
        self.declare_parameter('can_port', 'can0')
        self.declare_parameter('mode', 0)
        self.declare_parameter('auto_enable', False)

        # Get parameters
        self.can_port = self.get_parameter('can_port').get_parameter_value().string_value
        self.mode = self.get_parameter('mode').get_parameter_value().integer_value
        auto_enable_param = self.get_parameter('auto_enable').get_parameter_value().bool_value

        # Auto enable only works in mode 1
        self.auto_enable = auto_enable_param and (self.mode == 1)

        self.get_logger().info(f'can_port: {self.can_port}')
        self.get_logger().info(f'mode: {self.mode}')
        self.get_logger().info(f'auto_enable: {self.auto_enable}')

        # Callback groups
        self.pub_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_cb_group = ReentrantCallbackGroup()

        # Publishers
        self.joint_std_pub_puppet = self.create_publisher(
            JointState, '/puppet/joint_states', 10)

        # Mode 0: publish master arm messages
        if self.mode == 0:
            self.joint_std_pub_master = self.create_publisher(
                JointState, '/master/joint_states', 10)

        self.arm_status_pub = self.create_publisher(
            PiperStatusMsg, '/puppet/arm_status', 10)
        self.end_pose_pub = self.create_publisher(
            Pose, '/puppet/end_pose', 10)

        # Enable flag
        self._enable_flag = False
        self._gripper_exist = True  # Assume gripper exists by default

        # Joint states for slave arm
        self.joint_state_slave = JointState()
        self.joint_state_slave.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_slave.position = [0.0] * 7
        self.joint_state_slave.velocity = [0.0] * 7
        self.joint_state_slave.effort = [0.0] * 7

        # Joint states for master arm
        self.joint_state_master = JointState()
        self.joint_state_master.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_master.position = [0.0] * 7
        self.joint_state_master.velocity = [0.0] * 7
        self.joint_state_master.effort = [0.0] * 7

        # Connect to Piper arm
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        self.get_logger().info(f'Connected to Piper on {self.can_port}')

        # Mode 1: subscribe to control messages
        if self.mode == 1:
            self.pos_sub = self.create_subscription(
                PosCmd, 'pos_cmd', self.pos_callback, 10,
                callback_group=self.sub_cb_group)
            self.joint_sub = self.create_subscription(
                JointState, '/master/joint_states', self.joint_callback, 10,
                callback_group=self.sub_cb_group)
            self.enable_sub = self.create_subscription(
                Bool, '/enable_flag', self.enable_callback, 10,
                callback_group=self.sub_cb_group)

        # Create timer for publishing (200 Hz)
        self.pub_timer = self.create_timer(
            0.005, self.publish_callback, callback_group=self.pub_cb_group)

        # Auto enable handling
        self._auto_enable_done = False
        if self.auto_enable:
            self._start_auto_enable()

    def _start_auto_enable(self):
        """Start auto enable process in background thread"""
        self._auto_enable_thread = threading.Thread(target=self._auto_enable_process)
        self._auto_enable_thread.daemon = True
        self._auto_enable_thread.start()

    def _auto_enable_process(self):
        """Auto enable process with timeout"""
        timeout = 5.0
        start_time = time.time()
        enable_flag = False

        while not enable_flag:
            elapsed_time = time.time() - start_time
            self.get_logger().info('--------------------')

            # Check all motor enable status
            low_spd_info = self.piper.GetArmLowSpdInfoMsgs()
            enable_flag = (
                low_spd_info.motor_1.foc_status.driver_enable_status and
                low_spd_info.motor_2.foc_status.driver_enable_status and
                low_spd_info.motor_3.foc_status.driver_enable_status and
                low_spd_info.motor_4.foc_status.driver_enable_status and
                low_spd_info.motor_5.foc_status.driver_enable_status and
                low_spd_info.motor_6.foc_status.driver_enable_status
            )

            self.get_logger().info(f'Enable status: {enable_flag}')

            if enable_flag:
                self._enable_flag = True
                self._auto_enable_done = True
                self.get_logger().info('Auto enable successful')
                return

            # Try to enable
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)

            self.get_logger().info('--------------------')

            # Check timeout
            if elapsed_time > timeout:
                self.get_logger().error('Auto enable timeout!')
                self._auto_enable_done = True
                return

            time.sleep(1)

    def get_enable_flag(self):
        return self._enable_flag

    def publish_callback(self):
        """Timer callback for publishing arm messages"""
        # Publish slave arm joint and gripper
        self.publish_slave_arm_joint_and_gripper()

        # Publish slave arm status
        self.publish_slave_arm_state()

        # Publish slave arm end pose
        self.publish_slave_arm_end_pose()

        # Mode 0: also publish master arm messages
        if self.mode == 0:
            self.publish_master_arm_joint_and_gripper()

    def publish_slave_arm_state(self):
        """Publish slave arm status"""
        arm_status = PiperStatusMsg()
        status = self.piper.GetArmStatus()

        arm_status.ctrl_mode = status.arm_status.ctrl_mode
        arm_status.arm_status = status.arm_status.arm_status
        arm_status.mode_feedback = status.arm_status.mode_feed
        arm_status.teach_status = status.arm_status.teach_status
        arm_status.motion_status = status.arm_status.motion_status
        arm_status.trajectory_num = status.arm_status.trajectory_num
        arm_status.err_code = status.arm_status.err_code

        # Error status
        err_status = status.arm_status.err_status
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

    def publish_slave_arm_end_pose(self):
        """Publish slave arm end pose"""
        endpos = Pose()
        # End pose publishing (commented out in ROS1 as well)
        # endpos.position.x = self.piper.ArmEndPose.end_pose.X_axis/1000000
        # ... etc
        self.end_pose_pub.publish(endpos)

    def publish_slave_arm_joint_and_gripper(self):
        """Publish slave arm joint states and gripper"""
        now = self.get_clock().now()
        self.joint_state_slave.header.stamp = now.to_msg()

        joint_msgs = self.piper.GetArmJointMsgs()
        high_spd_msgs = self.piper.GetArmHighSpdInfoMsgs()
        gripper_msgs = self.piper.GetArmGripperMsgs()

        # Joint angles (convert to radians)
        factor = 0.017444  # approximately pi/180
        joint_0 = (joint_msgs.joint_state.joint_1 / 1000) * factor
        joint_1 = (joint_msgs.joint_state.joint_2 / 1000) * factor
        joint_2 = (joint_msgs.joint_state.joint_3 / 1000) * factor
        joint_3 = (joint_msgs.joint_state.joint_4 / 1000) * factor
        joint_4 = (joint_msgs.joint_state.joint_5 / 1000) * factor
        joint_5 = (joint_msgs.joint_state.joint_6 / 1000) * factor
        joint_6 = gripper_msgs.gripper_state.grippers_angle / 1000000

        # Velocities
        vel_0 = high_spd_msgs.motor_1.motor_speed / 1000
        vel_1 = high_spd_msgs.motor_2.motor_speed / 1000
        vel_2 = high_spd_msgs.motor_3.motor_speed / 1000
        vel_3 = high_spd_msgs.motor_4.motor_speed / 1000
        vel_4 = high_spd_msgs.motor_5.motor_speed / 1000
        vel_5 = high_spd_msgs.motor_6.motor_speed / 1000

        # Gripper effort
        effort_6 = gripper_msgs.gripper_state.grippers_effort / 1000

        self.joint_state_slave.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
        self.joint_state_slave.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
        self.joint_state_slave.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort_6]

        self.joint_std_pub_puppet.publish(self.joint_state_slave)

    def publish_master_arm_joint_and_gripper(self):
        """Publish master arm joint states and gripper (mode 0 only)"""
        now = self.get_clock().now()
        self.joint_state_master.header.stamp = now.to_msg()

        joint_ctrl = self.piper.GetArmJointCtrl()
        gripper_ctrl = self.piper.GetArmGripperCtrl()

        # Joint angles (convert to radians)
        factor = 0.017444
        joint_0 = (joint_ctrl.joint_ctrl.joint_1 / 1000) * factor
        joint_1 = (joint_ctrl.joint_ctrl.joint_2 / 1000) * factor
        joint_2 = (joint_ctrl.joint_ctrl.joint_3 / 1000) * factor
        joint_3 = (joint_ctrl.joint_ctrl.joint_4 / 1000) * factor
        joint_4 = (joint_ctrl.joint_ctrl.joint_5 / 1000) * factor
        joint_5 = (joint_ctrl.joint_ctrl.joint_6 / 1000) * factor
        joint_6 = gripper_ctrl.gripper_ctrl.grippers_angle / 1000000

        self.joint_state_master.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]

        self.joint_std_pub_master.publish(self.joint_state_master)

    def pos_callback(self, pos_data: PosCmd):
        """End pose control callback (mode 1)"""
        self.get_logger().info('Received PosCmd:')
        self.get_logger().info(f'x: {pos_data.x}, y: {pos_data.y}, z: {pos_data.z}')
        self.get_logger().info(f'roll: {pos_data.roll}, pitch: {pos_data.pitch}, yaw: {pos_data.yaw}')
        self.get_logger().info(f'gripper: {pos_data.gripper}')
        self.get_logger().info(f'mode1: {pos_data.mode1}, mode2: {pos_data.mode2}')

        x = round(pos_data.x * 1000)
        y = round(pos_data.y * 1000)
        z = round(pos_data.z * 1000)
        rx = round(pos_data.roll * 1000)
        ry = round(pos_data.pitch * 1000)
        rz = round(pos_data.yaw * 1000)

        if self.get_enable_flag():
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.piper.MotionCtrl_2(0x01, 0x02, 50, 0xad)
            self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)

            gripper = round(pos_data.gripper * 1000 * 1000)
            if gripper > 80000:
                gripper = 80000
            if gripper < 0:
                gripper = 0

            if self._gripper_exist:
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)

            self.piper.MotionCtrl_2(0x01, 0x00, 50, 0xad)

    def joint_callback(self, joint_data: JointState):
        """Joint control callback (mode 1)"""
        factor = 57324.840764  # 1000 * 180 / 3.14

        self.get_logger().debug(f'Received joint command on {self.can_port}')

        joint_0 = round(joint_data.position[0] * factor)
        joint_1 = round(joint_data.position[1] * factor)
        joint_2 = round(joint_data.position[2] * factor)
        joint_3 = round(joint_data.position[3] * factor)
        joint_4 = round(joint_data.position[4] * factor)
        joint_5 = round(joint_data.position[5] * factor)
        joint_6 = round(joint_data.position[6] * 1000 * 1000)

        if joint_6 > 80000:
            joint_6 = 80000
        if joint_6 < 0:
            joint_6 = 0

        if self.get_enable_flag():
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xad)
            self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xad)

    def enable_callback(self, enable_flag: Bool):
        """Enable/disable arm callback"""
        self.get_logger().info(f'Received enable flag: {enable_flag.data}')

        if enable_flag.data:
            self._enable_flag = True
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self._enable_flag = False
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x00, 0)


def main(args=None):
    rclpy.init(args=args)

    node = PiperMSNode()

    # Use multi-threaded executor for callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
