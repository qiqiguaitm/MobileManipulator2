#!/usr/bin/env python3
# -*-coding:utf8-*-
# IK solver for Piper arm using Pinocchio
# ROS2 version - converted from ROS1

import os
import sys

# Add paths for Pinocchio with CasADi support
sys.path.insert(0, '/opt/ros/humble/lib/python3.10/site-packages')
sys.path.insert(0, '/usr/local/lib/python3.10/site-packages')
sys.path.insert(0, '/usr/local/lib/python3.8/site-packages')

import casadi
import numpy as np
import pinocchio as pin
import time

from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

# Optional imports for visualization
try:
    import meshcat.geometry as mg
    from pinocchio.visualize import MeshcatVisualizer
    MESHCAT_AVAILABLE = True
except ImportError:
    mg = None
    MeshcatVisualizer = None
    MESHCAT_AVAILABLE = False


def quaternion_from_euler(roll, pitch, yaw):
    """Convert euler angles to quaternion (x, y, z, w)"""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()


def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to euler angles"""
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz')


class Arm_IK:
    """IK solver for Piper arm using Pinocchio and CasADi"""

    # Joint offset: URDF_joint = SDK_joint + OFFSET
    JOINT_OFFSET_DEG = [0, 10, -40, 0, 0, 0]
    JOINT_OFFSET_RAD = [d * np.pi / 180.0 for d in JOINT_OFFSET_DEG]

    def __init__(self, headless=True):
        """
        Initialize IK solver.

        Args:
            headless: If True, disable Meshcat visualization (default for ROS2)
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.headless = headless

        # URDF paths - try multiple locations
        urdf_paths = [
            '/data/workspace/MobileManipulator2/src/robot_desc/piper_description/urdf/piper_description.urdf',
            '/home/agilex/MobileManipulator2/src/robot_desc/piper_description/urdf/piper_description.urdf',
            '/home/agilex/MobileManipulator/src/robot_desc/piper_description/urdf/piper_description.urdf',
        ]

        urdf_path = None
        for path in urdf_paths:
            if os.path.exists(path):
                urdf_path = path
                break

        if urdf_path is None:
            raise FileNotFoundError(f"URDF not found in: {urdf_paths}")

        self.urdf_path = urdf_path
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        # Lock gripper joints
        self.mixed_jointsToLockIDs = ["joint7", "joint8"]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # Add gripper_center frame
        self.GRIPPER_OFFSET_M = 0.13503
        self.reduced_robot.model.addFrame(
            pin.Frame('gripper_center',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          np.eye(3),
                          np.array([0.0, 0.0, self.GRIPPER_OFFSET_M]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        # Collision model
        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        # Initialize with URDF offset
        self.init_data = np.array(self.JOINT_OFFSET_RAD)
        self.history_data = np.array(self.JOINT_OFFSET_RAD)

        # Meshcat visualizer (only if not headless)
        self.vis = None
        if not self.headless and MESHCAT_AVAILABLE:
            try:
                self.vis = MeshcatVisualizer(
                    self.reduced_robot.model,
                    self.reduced_robot.collision_model,
                    self.reduced_robot.visual_model
                )
                self.vis.initViewer(open=True)
                self.vis.loadViewerModel("pinocchio")
                self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
                self.vis.display(pin.neutral(self.reduced_robot.model))
            except Exception as e:
                print(f"[Arm_IK] Meshcat init failed: {e}")
                self.vis = None

        # CasADi model
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Error function
        self.gripper_id = self.reduced_robot.model.getFrameId("gripper_center")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)

        err = self.error(self.var_q, self.param_tf)
        pos_weight = 100
        ori_weight = 10
        self.totalcost = pos_weight * casadi.sumsqr(err[:3]) + ori_weight * casadi.sumsqr(err[3:])
        self.regularization = casadi.sumsqr(self.var_q)

        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    def _sdk_to_urdf(self, sdk_joints):
        """Convert SDK joint angles to URDF joint angles"""
        return np.array(sdk_joints) + np.array(self.JOINT_OFFSET_RAD)

    def _urdf_to_sdk(self, urdf_joints):
        """Convert URDF joint angles to SDK joint angles"""
        return np.array(urdf_joints) - np.array(self.JOINT_OFFSET_RAD)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        """
        Solve inverse kinematics.

        Args:
            target_pose: 4x4 homogeneous transformation matrix
            gripper: Gripper opening
            motorstate: Initial joint angles in SDK convention (radians)
            motorV: Joint velocities

        Returns:
            (sol_q, tau_ff, success): Solution in SDK convention
        """
        gripper = np.array([gripper / 2.0, -gripper / 2.0])

        if motorstate is not None:
            self.init_data = self._sdk_to_urdf(motorstate)
        elif self.init_data is None or len(self.init_data) == 0:
            self.init_data = np.array(self.JOINT_OFFSET_RAD)

        self.opti.set_initial(self.var_q, self.init_data)

        if self.vis is not None:
            self.vis.viewer['ee_target'].set_transform(target_pose)

        self.opti.set_value(self.param_tf, target_pose)

        try:
            sol = self.opti.solve_limited()
            sol_q_urdf = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q_urdf))
                self.init_data = sol_q_urdf
                if max_diff > 30.0 / 180.0 * np.pi:
                    self.init_data = np.array(self.JOINT_OFFSET_RAD)
            else:
                self.init_data = sol_q_urdf
            self.history_data = sol_q_urdf

            if self.vis is not None:
                self.vis.display(sol_q_urdf)

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q_urdf - self.init_data) * 0.0

            tau_ff = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data,
                sol_q_urdf, v, np.zeros(self.reduced_robot.model.nv)
            )

            is_collision = self.check_self_collision(sol_q_urdf, gripper)
            sol_q_sdk = self._urdf_to_sdk(sol_q_urdf)

            return sol_q_sdk, tau_ff, not is_collision

        except Exception as e:
            print(f"IK solver error: {e}")
            sol_q_sdk = self._urdf_to_sdk(self.init_data) if self.init_data is not None else np.zeros(6)
            return sol_q_sdk, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        """Check for self-collision"""
        try:
            pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
            pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
            collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
            return collision
        except AttributeError:
            return False
        except Exception as e:
            print(f"[Arm_IK] Collision check error: {e}")
            return False

    def get_ik_solution(self, x, y, z, roll, pitch, yaw, piper_control=None):
        """
        Compute IK solution.

        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            piper_control: Optional PIPER instance for execution

        Returns:
            (sol_q, success): Joint angles and success flag
        """
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous, 0)

        if get_result:
            if piper_control is not None:
                piper_control.joint_control_piper(
                    sol_q[0], sol_q[1], sol_q[2],
                    sol_q[3], sol_q[4], sol_q[5], 0
                )
        else:
            print("IK solution has collision!")

        return sol_q, get_result


# ROS2 Node wrapper (optional - for standalone usage)
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from piper_msgs.msg import PosCmd

    class PiperIKNode(Node):
        def __init__(self):
            super().__init__('piper_ik_node')
            self.arm_ik = Arm_IK(headless=True)
            self.subscription = self.create_subscription(
                PosCmd, 'pin_pos_cmd', self.pos_cmd_callback, 10
            )
            self.get_logger().info('Piper IK node started')

        def pos_cmd_callback(self, msg):
            sol_q, success = self.arm_ik.get_ik_solution(
                msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw
            )
            if success:
                self.get_logger().info(f'IK solution: {sol_q}')
            else:
                self.get_logger().warn('IK failed')

    rclpy.init(args=args)
    node = PiperIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
