#!/usr/bin/env python3
"""
Piper Grasp Node - ROS2 High-level grasp control for Piper arm

Features:
- V1 compatible interfaces (piper_msgs)
- V2 enhanced interfaces (piper_msgs services/actions)
- Grasp capabilities (observe/pick/place)

All length units at interface: mm
All angle units at interface: degrees

Converted from ROS1 to ROS2
"""

import os
import sys
import time
import math
import threading
import yaml
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, String, UInt8
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

# Add current script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from piper_api_v2 import PiperAPI
from grasp_blacklist import DualLayerBlacklist

# Import piper_msgs
from piper_msgs.msg import PiperStatusMsg, PosCmd, PiperStatus
from piper_msgs.srv import (
    Enable, Gripper, GoZero,
    EnableEnhanced, SetPosition, SetGripper,
    GetStatus, GetPosition, GoReady,
    Observe, InWorkingArea, GraspDetect
)
from piper_msgs.action import PiperPick, PiperPlace

# Unit conversion constants
M_TO_MM = 1000.0
MM_TO_M = 0.001


class PiperGraspNode(Node):
    """Piper ROS2 grasp control node"""

    def __init__(self):
        super().__init__('piper_grasp_node')
        self.get_logger().info("Initializing Piper Grasp Node...")

        # Load configuration
        self.config = self._load_config()

        # State
        self.arm = None
        self.transformer = None
        self.ik_solver = None
        self.error_state = "NORMAL"
        self.error_description = ""

        # Observe cache
        self.last_observe_result = {
            'valid': False,
            'timestamp': 0,
            'category': '',
            'point3d_camera': [],
            'point3d_base': [],
            'offset': [],
            'angle_camera': 0,
            'angle_base': 0,
            'gripper_width': 0,
            'end_pose_at_observe': []
        }

        # Threading locks
        self._arm_lock = threading.Lock()
        self._observe_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._action_busy = False
        self._publish_rate = self.config.get('publish_rate', 200)

        # Enable state cache
        self._enable_cache = False
        self._enable_cache_time = 0
        self._enable_cache_interval = 1.0

        # Cancel tracking for actions
        self._pick_cancelled = False
        self._place_cancelled = False

        # Callback group for concurrent service handling
        self.callback_group = ReentrantCallbackGroup()

        # Initialize components
        self._init_arm()
        self._init_transformer()
        self._init_ik_solver()
        self._init_ros_interfaces()

        # Startup recovery
        if self.config.get('recovery', {}).get('enabled', True):
            self._perform_startup_recovery()

        self.get_logger().info("Piper Grasp Node initialized")

    def _load_config(self) -> dict:
        """Load configuration from parameter or file"""
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').value

        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('piper_ros', config)

        default_paths = [
            '/data/workspace/MobileManipulator2/src/piper_grasp/config/piper_grasp_node.yaml',
            os.path.join(os.path.dirname(__file__), '../config/piper_grasp_node.yaml'),
        ]

        for path in default_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('piper_ros', config)

        self.get_logger().warn("No config file found, using defaults")
        return {}

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected if self.arm else False

    @property
    def is_enabled(self) -> bool:
        if not self.arm or not self.arm.is_connected:
            self._enable_cache = False
            return False

        now = time.time()
        if now - self._enable_cache_time < self._enable_cache_interval:
            return self._enable_cache

        try:
            enable_list = self.arm.piper.GetArmEnableStatus()
            self._enable_cache = all(enable_list) if enable_list else False
            self._enable_cache_time = now
        except Exception:
            self._enable_cache = False

        return self._enable_cache

    def _init_arm(self):
        """Initialize PiperAPI"""
        can_port = self.config.get('can_port', 'can0')
        gripper_cfg = self.config.get('gripper', {})

        self.get_logger().info(f"Connecting to {can_port}...")

        self.arm = PiperAPI(
            can_name=can_port,
            gripper_max_mm=gripper_cfg.get('max_mm', 90.0),
            gripper_offset_mm=gripper_cfg.get('offset_mm', 135.03)
        )

        if self.config.get('auto_connect', True):
            self.arm.connect()

            if self.config.get('auto_enable', True):
                self.arm.motion_enable(True)

    def _init_transformer(self):
        """Initialize hand-eye calibration"""
        calibration = self.config.get('calibration', {})
        if not calibration:
            self.transformer = None
            return

        try:
            T_cam2flan = np.array(calibration.get('T_cam2flan', [0, 0, 0]))
            q_cam2flan = calibration.get('q_cam2flan', [0, 0, 0, 1])
            R_cam2flan = R.from_quat(q_cam2flan).as_matrix()
            T_gripper2flan = np.array(calibration.get('T_gripper2flan', [0, 0, 0.135]))

            self._R_cam2end = R_cam2flan
            self._T_cam2end = T_cam2flan - T_gripper2flan
            self.transformer = True

            self.get_logger().info("Hand-eye calibration initialized")

        except Exception as e:
            self.get_logger().error(f"Calibration init failed: {e}")
            self.transformer = None

    def _init_ik_solver(self):
        """Initialize IK solver for trajectory planning"""
        ik_cfg = self.config.get('ik_planning', {})
        self.ik_solver = None

        if not ik_cfg.get('enabled', False):
            self.get_logger().info("IK planning disabled in config")
            return

        try:
            # Import IK solver from piper_driver
            piper_driver_path = '/data/workspace/MobileManipulator2/src/piper_driver/scripts'
            if piper_driver_path not in sys.path:
                sys.path.insert(0, piper_driver_path)
            from piper_pinocchio import Arm_IK

            headless = ik_cfg.get('headless', True)
            self.ik_solver = Arm_IK(headless=headless)
            self.get_logger().info(f"IK solver initialized (headless={headless})")

        except ImportError as e:
            self.get_logger().warn(f"IK solver import failed: {e}")

        except Exception as e:
            self.get_logger().error(f"IK solver init failed: {e}")

    def _init_ros_interfaces(self):
        """Initialize ROS2 interfaces"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.pub_joint_states = self.create_publisher(JointState, '/joint_states', qos)
        self.pub_end_pose_euler = self.create_publisher(PosCmd, '/end_pose_euler', qos)
        self.pub_arm_status = self.create_publisher(PiperStatusMsg, '/arm_status', qos)
        self.pub_status = self.create_publisher(PiperStatus, '/piper/status', qos)
        self.pub_gripper_center = self.create_publisher(PoseStamped, '/piper/gripper_center', qos)
        self.pub_error_state = self.create_publisher(String, '/piper/error_state', qos)
        self.pub_connection_state = self.create_publisher(UInt8, '/piper/connection_state', qos)

        # V1 Compatible Services
        self.srv_enable_v1 = self.create_service(
            Enable, '/enable_srv', self._handle_enable_v1,
            callback_group=self.callback_group
        )
        self.srv_gripper_v1 = self.create_service(
            Gripper, '/gripper_srv', self._handle_gripper_v1,
            callback_group=self.callback_group
        )
        self.srv_stop_v1 = self.create_service(
            Trigger, '/stop_srv', self._handle_stop_v1,
            callback_group=self.callback_group
        )
        self.srv_reset_v1 = self.create_service(
            Trigger, '/reset_srv', self._handle_reset_v1,
            callback_group=self.callback_group
        )
        self.srv_go_zero_v1 = self.create_service(
            GoZero, '/go_zero_srv', self._handle_go_zero_v1,
            callback_group=self.callback_group
        )

        # V2 Services
        self.srv_enable = self.create_service(
            EnableEnhanced, '/piper/enable', self._handle_enable,
            callback_group=self.callback_group
        )
        self.srv_set_position = self.create_service(
            SetPosition, '/piper/set_position', self._handle_set_position,
            callback_group=self.callback_group
        )
        self.srv_set_gripper = self.create_service(
            SetGripper, '/piper/set_gripper', self._handle_set_gripper,
            callback_group=self.callback_group
        )
        self.srv_get_status = self.create_service(
            GetStatus, '/piper/get_status', self._handle_get_status,
            callback_group=self.callback_group
        )
        self.srv_get_position = self.create_service(
            GetPosition, '/piper/get_position', self._handle_get_position,
            callback_group=self.callback_group
        )
        self.srv_go_ready = self.create_service(
            GoReady, '/piper/go_ready', self._handle_go_ready,
            callback_group=self.callback_group
        )
        self.srv_observe = self.create_service(
            Observe, '/piper/observe', self._handle_observe,
            callback_group=self.callback_group
        )
        self.srv_in_working_area = self.create_service(
            InWorkingArea, '/piper/in_working_area', self._handle_in_working_area,
            callback_group=self.callback_group
        )

        # Actions
        self.action_pick = ActionServer(
            self, PiperPick, '/piper/pick',
            execute_callback=self._execute_pick,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group
        )
        self.action_place = ActionServer(
            self, PiperPlace, '/piper/place',
            execute_callback=self._execute_place,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group
        )

        # Blacklist system
        blacklist_cfg = self.config.get('grasp', {}).get('blacklist', {})
        if blacklist_cfg.get('enabled', False):
            self.blacklist = DualLayerBlacklist(blacklist_cfg)
            self.get_logger().info("Blacklist system enabled")
        else:
            self.blacklist = None

        # Perception client
        self.perception_client = None
        perception_service = self.config.get('grasp', {}).get(
            'perception_service', '/perception_grasp_node/detect'
        )
        self.perception_client = self.create_client(
            GraspDetect, perception_service,
            callback_group=self.callback_group
        )
        self.get_logger().info(f"Perception client created: {perception_service}")

        # Create publish timer
        period = 1.0 / self._publish_rate
        self.publish_timer = self.create_timer(period, self._publish_state)

    def _perform_startup_recovery(self):
        """Perform startup recovery sequence"""
        recovery_cfg = self.config.get('recovery', {}).get('on_startup', {})
        self.get_logger().info("Performing startup recovery...")

        try:
            if recovery_cfg.get('check_error', True):
                self.arm.clean_error(go_zero=True, force=True)
                time.sleep(0.5)

            if not self.is_enabled:
                self.arm.motion_enable(True)

            if recovery_cfg.get('close_gripper', True):
                self._set_gripper_mm(0, speed=500, wait=True)

            if recovery_cfg.get('go_ready', False):
                ready_cfg = self.config.get('positions', {}).get('ready', {})
                self.arm.go_ready(ready_cfg)

            self.get_logger().info("Startup recovery complete")

        except Exception as e:
            self.get_logger().error(f"Startup recovery failed: {e}")

    # ========== Action Callbacks ==========

    def _goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info("Cancel request received")
        return CancelResponse.ACCEPT

    # ========== Gripper Helpers ==========

    def _set_gripper_mm(self, pos_mm, speed=500, wait=True):
        """Set gripper position in mm"""
        self.arm.set_gripper_position(pos_mm, speed=speed, wait=wait)

    def _get_gripper_mm(self):
        """Get gripper position in mm"""
        _, pos_mm = self.arm.get_gripper_position()
        return pos_mm

    # ========== Working Area Check ==========

    def in_working_area(self, offset=None, yaw=None, point3d_cam_mm=None,
                        end_pose=None, point_in_base_mm=None):
        """Check if a grasp is within the working area limits."""
        provided = sum([offset is not None, point3d_cam_mm is not None,
                        point_in_base_mm is not None])
        if provided != 1:
            return False

        if point3d_cam_mm is not None:
            if end_pose is None:
                _, end_pose = self.arm.get_position(return_gripper_center=True)
            result = self._compute_grasp_offset(point3d_cam_mm, end_pose)
            offset = result['offset']

        if point_in_base_mm is not None:
            if end_pose is None:
                _, end_pose = self.arm.get_position(return_gripper_center=True)
            offset = [point_in_base_mm[i] - end_pose[i] for i in range(3)]

        wa_cfg = self.config.get('working_area', {})
        if not wa_cfg:
            return True

        x, y, z = offset
        if (x <= wa_cfg.get('offset_xmin', -float('inf')) or
            x >= wa_cfg.get('offset_xmax', float('inf')) or
            y <= wa_cfg.get('offset_ymin', -float('inf')) or
            y >= wa_cfg.get('offset_ymax', float('inf')) or
            z <= wa_cfg.get('offset_zmin', -float('inf')) or
            z >= wa_cfg.get('offset_zmax', float('inf'))):
            return False

        if yaw is not None:
            if yaw < 0:
                yaw += 360
            if (yaw < wa_cfg.get('yaw_min', 0) or
                yaw > wa_cfg.get('yaw_max', 360)):
                return False

        return True

    # ========== Blacklist Check ==========

    def _check_blacklist(self, category: str, point3d_base_mm: List[float],
                         object_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if object is blacklisted"""
        if not self.blacklist:
            return (False, None)

        is_black, reason = self.blacklist.is_blacklisted(
            center=point3d_base_mm,
            object_id=object_id,
            category=category
        )
        return (is_black, reason)

    # ========== Verification Helpers ==========

    def _verify_position(self, target, axes='xyz', tolerance=5.0):
        """Verify if current position reached target within tolerance."""
        _, current = self.arm.get_position(return_gripper_center=True)

        axis_map = {
            'x': (0, 'mm'), 'y': (1, 'mm'), 'z': (2, 'mm'),
            'roll': (3, 'deg'), 'pitch': (4, 'deg'), 'yaw': (5, 'deg'),
        }

        errors = {}
        all_ok = True

        for axis in axes.split(','):
            axis = axis.strip().lower()
            if axis in axis_map and axis in target:
                idx, unit = axis_map[axis]
                actual = current[idx]
                expected = target[axis]
                error = abs(actual - expected)
                errors[axis] = {'actual': actual, 'expected': expected, 'error': error, 'unit': unit}
                if error > tolerance:
                    all_ok = False

        if all_ok:
            msg_parts = [f"{k}={v['actual']:.1f}{v['unit']}(err={v['error']:.1f})" for k, v in errors.items()]
            msg = f"Reached: {', '.join(msg_parts)}"
        else:
            msg_parts = [f"{k}={v['actual']:.1f}{v['unit']}(target={v['expected']:.1f}, err={v['error']:.1f})" for k, v in errors.items()]
            msg = f"Position error: {', '.join(msg_parts)}"

        return all_ok, errors, msg

    def _verify_gripper(self, target_mm, tolerance=5.0):
        """Verify if gripper reached target position."""
        actual_mm = self._get_gripper_mm()
        error = abs(actual_mm - target_mm)

        if error <= tolerance:
            msg = f"Gripper: {actual_mm:.1f}mm (target={target_mm:.1f}mm, err={error:.1f}mm)"
            return True, actual_mm, error, msg
        else:
            msg = f"Gripper error: {actual_mm:.1f}mm (target={target_mm:.1f}mm, err={error:.1f}mm)"
            return False, actual_mm, error, msg

    def _verify_grasp_success(self, min_width_mm=5):
        """Verify grasp success by checking gripper width"""
        actual_width = self._get_gripper_mm()

        if actual_width < min_width_mm:
            return (False, actual_width, "Grasp failed: gripper fully closed")

        return (True, actual_width, f"Grasp success: width {actual_width:.1f}mm")

    def _handle_grasp_failure(self, reason):
        """Handle grasp failure: close, lift, return to ready"""
        self.get_logger().warn(f"Handling grasp failure: {reason}")

        try:
            self._set_gripper_mm(0, wait=True)

            _, current = self.arm.get_position(return_gripper_center=True)
            safe_z = current[2] + 50
            self.arm.set_position(z=safe_z, wait=True, speed=30, use_gripper_center=True)

            ready_cfg = self.config.get('positions', {}).get('ready', {})
            self.arm.set_position(
                x=ready_cfg.get('x', 400),
                y=ready_cfg.get('y', 0),
                z=ready_cfg.get('z', 150),
                roll=ready_cfg.get('roll', 180),
                pitch=ready_cfg.get('pitch', 30),
                yaw=ready_cfg.get('yaw', 180),
                wait=True, speed=30,
                use_gripper_center=True
            )
        except Exception as e:
            self.get_logger().error(f"Failure handling error: {e}")

    # ========== Calculation Helpers ==========

    def _calculate_target_yaw(self, current_yaw, angle_camera):
        """Calculate target yaw from camera angle"""
        if not (0 <= angle_camera <= 180):
            angle_camera = max(0, min(180, angle_camera))

        angle = angle_camera
        if angle < 0:
            angle = angle + 180
        if angle > 180:
            angle = angle - 180

        target_yaw = current_yaw - angle
        if target_yaw < 90:
            target_yaw += 180
        target_yaw = max(120, target_yaw)
        target_yaw = min(240, target_yaw)
        return target_yaw

    def _calculate_gripper_width(self, width3d_mm):
        """Calculate gripper open width with margin"""
        grasp_cfg = self.config.get('grasp', {})
        base_margin = grasp_cfg.get('gripper_margin_base', 40)
        ratio_margin = width3d_mm * grasp_cfg.get('gripper_margin_ratio', 0.5)

        margin = max(base_margin, ratio_margin)
        gripper_open = width3d_mm + margin

        gripper_max = self.config.get('gripper', {}).get('max_mm', 70)
        return min(gripper_open, gripper_max)

    def _get_connection_state(self):
        """Get current connection state code"""
        if not self.is_connected:
            return 0  # DISCONNECTED
        elif not self.is_enabled:
            return 1  # CONNECTED
        else:
            return 2  # ENABLED

    # ========== V1 Compatible Handlers ==========

    def _handle_enable_v1(self, request, response):
        """V1 enable service handler"""
        try:
            with self._arm_lock:
                self.arm.motion_enable(request.enable_request)
            response.enable_response = True
        except Exception as e:
            self.get_logger().error(f"Enable failed: {e}")
            response.enable_response = False
        return response

    def _handle_gripper_v1(self, request, response):
        """V1 gripper service handler"""
        try:
            with self._arm_lock:
                pos_mm = request.gripper_angle * M_TO_MM
                self.arm.set_gripper_position(pos_mm, speed=500, wait=True)
            response.gripper_state = True
        except Exception as e:
            self.get_logger().error(f"Gripper failed: {e}")
            response.gripper_state = False
        return response

    def _handle_stop_v1(self, request, response):
        """V1 stop service handler"""
        try:
            with self._arm_lock:
                self.arm.emergency_stop()
            self.error_state = "WARNING"
            self.error_description = "Emergency stop triggered"
            response.success = True
            response.message = "Emergency stop executed"
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _handle_reset_v1(self, request, response):
        """V1 reset service handler"""
        try:
            with self._arm_lock:
                self.arm.clean_error()
            self.error_state = "NORMAL"
            self.error_description = ""
            response.success = True
            response.message = "Reset complete"
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _handle_go_zero_v1(self, request, response):
        """V1 go zero service handler"""
        try:
            with self._arm_lock:
                self.arm.go_zero()
            response.status = True
        except Exception:
            response.status = False
        return response

    # ========== V2 Service Handlers ==========

    def _handle_enable(self, request, response):
        """V2 enhanced enable handler"""
        try:
            action = request.action

            with self._arm_lock:
                if action == request.ACTION_ENABLE:
                    self.arm.motion_enable(True)
                    self.error_state = "NORMAL"
                    response.message = "Enabled"

                elif action == request.ACTION_DISABLE:
                    self.arm.motion_enable(False)
                    response.message = "Disabled"

                elif action == request.ACTION_FORCE_ENABLE:
                    self.arm.clean_error()
                    time.sleep(0.5)
                    self.arm.motion_enable(True)
                    self.error_state = "NORMAL"
                    response.message = "Force enabled"

                elif action == request.ACTION_CLEAN_ERROR:
                    self.arm.clean_error()
                    self.error_state = "NORMAL"
                    self.error_description = ""
                    response.message = "Error cleared"

                elif action == request.ACTION_RECONNECT:
                    self.arm.disconnect()
                    time.sleep(1.0)
                    self.arm.connect()
                    response.message = "Reconnected"

                elif action == request.ACTION_SHUTDOWN:
                    self.arm.go_zero()
                    self.arm.motion_enable(False)
                    response.message = "Shutdown complete"

            response.success = True
            response.connection_state = self._get_connection_state()
            response.error_severity = self.error_state
            response.error_description = self.error_description

        except Exception as e:
            response.success = False
            response.message = str(e)
            response.connection_state = 0  # CONN_ERROR

        return response

    def _handle_set_position(self, request, response):
        """Set position service handler"""
        try:
            with self._arm_lock:
                result = self.arm.set_position(
                    x=request.x if request.x != 0 else None,
                    y=request.y if request.y != 0 else None,
                    z=request.z if request.z != 0 else None,
                    roll=request.roll if request.roll != 0 else None,
                    pitch=request.pitch if request.pitch != 0 else None,
                    yaw=request.yaw if request.yaw != 0 else None,
                    speed=request.speed if request.speed > 0 else 30,
                    wait=request.wait,
                    use_gripper_center=request.use_gripper_center
                )

            _, pos = self.arm.get_position(return_gripper_center=request.use_gripper_center)
            response.success = result
            response.final_position = [float(p) for p in pos]

        except Exception as e:
            response.success = False
            response.error_message = str(e)

        return response

    def _handle_set_gripper(self, request, response):
        """Set gripper service handler"""
        try:
            with self._arm_lock:
                self._set_gripper_mm(request.position, speed=request.speed, wait=request.wait)

            _, pos_mm = self.arm.get_gripper_position()
            response.success = True
            response.final_position = pos_mm

        except Exception as e:
            response.success = False
            response.error_message = str(e)

        return response

    def _handle_get_status(self, request, response):
        """Get status service handler"""
        try:
            response.connected = self.is_connected
            response.enabled = self.is_enabled
            response.error_state = self.error_state
            response.error_description = self.error_description

            _, pos = self.arm.get_position(return_gripper_center=True)
            response.position = [float(p) for p in pos]

            joints_deg = self.arm.get_joint_degrees()
            response.joints = [float(j) for j in joints_deg]

            _, gripper_mm = self.arm.get_gripper_position()
            response.gripper_position = gripper_mm

        except Exception as e:
            response.connected = False
            response.error_state = "ERROR"
            response.error_description = str(e)

        return response

    def _handle_get_position(self, request, response):
        """Get position service handler"""
        try:
            _, pos = self.arm.get_position(return_gripper_center=request.use_gripper_center)
            response.position = [float(p) for p in pos]
        except Exception as e:
            response.position = []
        return response

    def _handle_go_ready(self, request, response):
        """Go ready service handler"""
        try:
            with self._arm_lock:
                ready_cfg = self.config.get('positions', {}).get('ready', {})

                if request.open_gripper:
                    gripper_max = self.config.get('gripper', {}).get('max_mm', 70)
                    self._set_gripper_mm(gripper_max, wait=True)

                speed = request.speed if request.speed > 0 else 30
                self.arm.go_ready(ready_cfg, speed=speed)

            _, pos = self.arm.get_position(return_gripper_center=True)
            response.success = True
            response.final_position = [float(p) for p in pos]

        except Exception as e:
            response.success = False
            response.error_message = str(e)

        return response

    # ========== Coordinate Transform ==========

    def _compute_grasp_offset(self, point3d_cam_mm, end_pose):
        """Compute grasp offset from camera observation to base frame."""
        point_cam_m = np.array(point3d_cam_mm) * MM_TO_M
        point_end_m = np.dot(self._R_cam2end, point_cam_m) + self._T_cam2end
        end_pos_m = np.array(end_pose[0:3]) * MM_TO_M
        rot = R.from_euler('xyz', end_pose[3:6], degrees=True)
        R_end2base = rot.as_matrix()
        point_in_base_m = np.dot(R_end2base, point_end_m) + end_pos_m
        offset_m = point_in_base_m - end_pos_m

        point_in_base_mm = (point_in_base_m * M_TO_MM).tolist()
        offset_mm = (offset_m * M_TO_MM).tolist()

        return {
            'point_in_base': point_in_base_mm,
            'offset': offset_mm
        }

    def _handle_observe(self, request, response):
        """Handle observe request"""
        if not self.perception_client:
            response.success = False
            response.error_message = "Perception service not available"
            return response

        grasp_cfg = self.config.get('grasp', {})

        try:
            # Move to ready position if requested
            if request.go_ready_first:
                ready_cfg = self.config.get('positions', {}).get('ready', {})
                speed = request.speed if request.speed > 0 else 30
                self.arm.go_ready(ready_cfg, speed=speed)

            # Get current end pose
            _, end_pose = self.arm.get_position(return_gripper_center=True)

            # Call perception service
            if not self.perception_client.wait_for_service(timeout_sec=5.0):
                response.success = False
                response.error_message = "Perception service timeout"
                return response

            detect_req = GraspDetect.Request()
            detect_req.prompt = request.prompt if request.prompt else grasp_cfg.get('default_prompt', '')
            detect_req.enable_cdm = True

            future = self.perception_client.call_async(detect_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

            if not future.done():
                response.success = False
                response.error_message = "Perception call timeout"
                return response

            detect_resp = future.result()

            if not detect_resp.success:
                response.success = False
                response.error_message = detect_resp.error_message
                return response

            # Get 3D point in camera frame (mm)
            point3d_cam = list(detect_resp.point3d)
            point3d_cam_mm = [p * M_TO_MM for p in point3d_cam]

            # Compute offset and transform to base frame
            transform_result = self._compute_grasp_offset(point3d_cam_mm, end_pose)
            point3d_base = transform_result['point_in_base']
            offset = transform_result['offset']

            # Check blacklist
            category = detect_resp.category
            is_black, reason = self._check_blacklist(category, point3d_base)
            if is_black:
                response.success = False
                response.error_message = f"Blacklisted: {reason}"
                return response

            # Check working area
            if not self.in_working_area(offset=offset, yaw=detect_resp.angle):
                response.success = False
                response.error_message = "Target not in working area"
                return response

            # Calculate base yaw
            angle_base = self._calculate_target_yaw(end_pose[5], detect_resp.angle)

            # Update observe cache
            with self._observe_lock:
                self.last_observe_result = {
                    'valid': True,
                    'timestamp': time.time(),
                    'category': category,
                    'point3d_camera': point3d_cam_mm,
                    'point3d_base': point3d_base,
                    'offset': offset,
                    'angle_camera': detect_resp.angle,
                    'angle_base': angle_base,
                    'gripper_width': detect_resp.width3d * M_TO_MM,
                    'end_pose_at_observe': list(end_pose)
                }

            # Fill response
            response.success = True
            response.category = category
            response.point3d_camera = point3d_cam_mm
            response.point3d_base = point3d_base
            response.offset = offset
            response.angle_camera = detect_resp.angle
            response.angle_base = angle_base
            response.gripper_width = detect_resp.width3d * M_TO_MM
            response.score = detect_resp.score
            response.detection_time_ms = detect_resp.detection_time_ms

        except Exception as e:
            response.success = False
            response.error_message = str(e)

        return response

    def _handle_in_working_area(self, request, response):
        """Handle in_working_area service request."""
        offset = list(request.offset) if request.offset else None
        yaw = request.yaw if not (math.isnan(request.yaw) or math.isinf(request.yaw)) else None
        point3d_cam_mm = list(request.point3d_cam) if request.point3d_cam else None
        end_pose = list(request.end_pose) if request.end_pose else None
        point_in_base_mm = list(request.point_in_base) if request.point_in_base else None

        response.in_working_area = self.in_working_area(
            offset=offset,
            yaw=yaw,
            point3d_cam_mm=point3d_cam_mm,
            end_pose=end_pose,
            point_in_base_mm=point_in_base_mm
        )
        return response

    # ========== Cancel Handlers ==========

    def _handle_pick_cancel(self, result, start_time, category=""):
        """Handle pick cancellation with safe lift"""
        self.get_logger().warn("Pick cancelled, stopping safely...")
        try:
            _, current = self.arm.get_position(return_gripper_center=True)
            lift_height = self.config.get('grasp', {}).get('preempt_lift_mm', 50)
            self.arm.set_position(z=current[2] + lift_height, wait=True, speed=30,
                                  use_gripper_center=True)
        except Exception as e:
            self.get_logger().error(f"Cancel lift failed: {e}")

        result.success = False
        result.error_message = "Cancelled by user"
        result.category = category
        result.execution_time_ms = (time.time() - start_time) * 1000

    def _handle_place_cancel(self, result, start_time):
        """Handle place cancellation with safe lift"""
        self.get_logger().warn("Place cancelled, stopping safely...")
        try:
            _, current = self.arm.get_position(return_gripper_center=True)
            lift_height = self.config.get('grasp', {}).get('preempt_lift_mm', 50)
            self.arm.set_position(z=current[2] + lift_height, wait=True, speed=30,
                                  use_gripper_center=True)
        except Exception as e:
            self.get_logger().error(f"Cancel lift failed: {e}")

        result.success = False
        result.error_message = "Cancelled by user"
        result.execution_time_ms = (time.time() - start_time) * 1000

    # ========== Pick Action ==========

    async def _execute_pick(self, goal_handle):
        """Execute pick action"""
        result = PiperPick.Result()
        feedback = PiperPick.Feedback()

        grasp_cfg = self.config.get('grasp', {})
        start_time = time.time()
        category = ""

        def send_feedback(state, step_name, progress, message=""):
            feedback.state = state
            feedback.step_name = step_name
            feedback.progress = progress
            feedback.message = message
            goal_handle.publish_feedback(feedback)

        # Check for concurrent action
        with self._action_lock:
            if self._action_busy:
                result.success = False
                result.error_message = "Another action in progress"
                goal_handle.abort()
                return result
            self._action_busy = True
            self._pick_cancelled = False

        try:
            # === CHECKING ===
            self.get_logger().info("=== PICK: CHECKING ===")
            send_feedback(1, "Checking", 0.0, "Validating")

            goal = goal_handle.request

            # Check cancel
            if goal_handle.is_cancel_requested:
                self._handle_pick_cancel(result, start_time, category)
                goal_handle.canceled()
                return result

            # Check observe cache
            if goal.use_last_observe:
                with self._observe_lock:
                    observe_cache = dict(self.last_observe_result)

                if not observe_cache['valid']:
                    raise RuntimeError("No valid observe result")

                cache_timeout = grasp_cfg.get('observe_cache_timeout', 30.0)
                age = time.time() - observe_cache['timestamp']
                if age > cache_timeout:
                    raise RuntimeError(f"Observe result expired ({age:.1f}s > {cache_timeout}s)")

                target_base = list(observe_cache['point3d_base'])
                angle = observe_cache['angle_camera']
                gripper_width = observe_cache['gripper_width']
                category = observe_cache['category']
            else:
                if len(goal.target_offset) < 3:
                    raise RuntimeError("target_offset must have 3 elements")
                _, current_pos = self.arm.get_position(return_gripper_center=True)
                target_base = [
                    current_pos[0] + goal.target_offset[0],
                    current_pos[1] + goal.target_offset[1],
                    current_pos[2] + goal.target_offset[2]
                ]
                angle = goal.target_angle
                gripper_width = goal.gripper_width
                category = "manual"

            # Check ready position
            _, current_pos = self.arm.get_position(return_gripper_center=True)
            ready_cfg = self.config.get('positions', {}).get('ready', {})
            tolerance = grasp_cfg.get('ready_position_tolerance', 30)
            ready_pos = [ready_cfg.get('x', 400), ready_cfg.get('y', 0), ready_cfg.get('z', 150)]
            dist = math.sqrt(sum((current_pos[i] - ready_pos[i])**2 for i in range(3)))
            if dist > tolerance:
                raise RuntimeError(f"Not at ready position (distance={dist:.1f}mm)")

            # Calculate target
            target_x, target_y, target_z = target_base[0], target_base[1], target_base[2]

            # Apply grasp depth
            category_depths = grasp_cfg.get('category_depths', {})
            default_depth = grasp_cfg.get('grasp_depth_mm', 20)
            grasp_depth = category_depths.get(category, default_depth)
            target_z -= grasp_depth

            # Apply min_z limit
            min_z = grasp_cfg.get('min_z', -150) if math.isnan(goal.min_z) else goal.min_z
            if target_z < min_z:
                target_z = min_z

            # Calculate yaw and gripper width
            target_yaw = self._calculate_target_yaw(current_pos[5], angle)
            if gripper_width > 0:
                gripper_open = self._calculate_gripper_width(gripper_width)
            else:
                gripper_open = self.config.get('gripper', {}).get('max_mm', 70)

            speed = goal.speed if goal.speed > 0 else grasp_cfg.get('default_speed', 30)
            lift_height = goal.lift_height if goal.lift_height > 0 else grasp_cfg.get('lift_height', 200)

            self.get_logger().info(f"Pick target: [{target_x:.1f}, {target_y:.1f}, {target_z:.1f}]mm")

            # === APPROACHING ===
            approach_pitch = grasp_cfg.get('approach_pitch', 10)
            self.get_logger().info("=== PICK: APPROACHING ===")
            send_feedback(2, "Approaching", 0.2, "Moving XY")
            tmp = self.arm.set_position(x=target_x, y=target_y, z=target_z + 100,
                                        pitch=approach_pitch, wait=True, speed=speed,
                                        use_gripper_center=True)
            if not tmp:
                raise RuntimeError("Failed to move to approach position")

            if goal_handle.is_cancel_requested:
                self._handle_pick_cancel(result, start_time, category)
                goal_handle.canceled()
                return result

            # Adjust yaw
            tmp = self.arm.set_position(yaw=target_yaw, wait=True, speed=speed, use_gripper_center=True)

            # === OPENING ===
            self.get_logger().info("=== PICK: OPENING ===")
            send_feedback(3, "Opening gripper", 0.3)
            self._set_gripper_mm(gripper_open, speed=500, wait=True)

            # === DESCENDING ===
            ik_cfg = self.config.get('ik_planning', {})
            use_ik = ik_cfg.get('enabled', False) and self.ik_solver is not None
            use_segmented = grasp_cfg.get('use_segmented_descent', True)
            descent_speed = grasp_cfg.get('approach_speed', 20)

            self.get_logger().info("=== PICK: DESCENDING ===")
            send_feedback(4, "Descending", 0.4, f"Z={target_z:.1f}mm")

            if use_ik and hasattr(self.arm, 'move_z_with_ik'):
                success, msg = self.arm.move_z_with_ik(
                    target_z, ik_solver=self.ik_solver,
                    speed=ik_cfg.get('descent_speed', 15),
                    num_waypoints=ik_cfg.get('num_waypoints', 10),
                    use_gripper_center=True
                )
                tmp = success
                if not success and ik_cfg.get('fallback_to_segmented', True):
                    segment_size = grasp_cfg.get('descent_segment_size', 60)
                    tmp = self.arm.move_z_segmented(target_z, max_step=segment_size,
                                                    speed=descent_speed, use_gripper_center=True)
            elif use_segmented and hasattr(self.arm, 'move_z_segmented'):
                segment_size = grasp_cfg.get('descent_segment_size', 60)
                tmp = self.arm.move_z_segmented(target_z, max_step=segment_size,
                                                speed=descent_speed, use_gripper_center=True)
            else:
                tmp = self.arm.set_position(z=target_z, wait=True, speed=descent_speed,
                                            use_gripper_center=True)

            if not tmp:
                raise RuntimeError("Failed to descend to target")

            # === CLOSING ===
            self.get_logger().info("=== PICK: CLOSING ===")
            send_feedback(5, "Closing gripper", 0.6)
            self._set_gripper_mm(0, speed=500, wait=True)

            # === VERIFYING ===
            self.get_logger().info("=== PICK: VERIFYING ===")
            send_feedback(6, "Verifying grasp", 0.7)
            grasp_success, actual_width, verify_msg = self._verify_grasp_success()

            if not grasp_success:
                self.get_logger().warn(f"Grasp verification failed: {verify_msg}")
                self._handle_grasp_failure(verify_msg)
                result.success = False
                result.error_message = verify_msg
                result.category = category
                result.execution_time_ms = (time.time() - start_time) * 1000
                goal_handle.abort()
                return result

            # === LIFTING ===
            _, current = self.arm.get_position(return_gripper_center=True)
            lift_z = current[2] + lift_height
            max_z = grasp_cfg.get('max_z', 300)
            if lift_z > max_z:
                lift_z = max_z

            self.get_logger().info("=== PICK: LIFTING ===")
            send_feedback(7, "Lifting", 0.8)

            if use_ik and hasattr(self.arm, 'move_z_with_ik'):
                success, _ = self.arm.move_z_with_ik(lift_z, ik_solver=self.ik_solver,
                                                    speed=ik_cfg.get('lift_speed', 20),
                                                    use_gripper_center=True)
                tmp = success
                if not success:
                    tmp = self.arm.set_position(z=lift_z, wait=True, speed=speed, use_gripper_center=True)
            else:
                tmp = self.arm.set_position(z=lift_z, wait=True, speed=speed, use_gripper_center=True)

            if not tmp:
                raise RuntimeError("Failed to lift after pick")

            # === RETURNING ===
            self.get_logger().info("=== PICK: RETURNING ===")
            ready_yaw = ready_cfg.get('yaw', 180)
            send_feedback(8, "Returning to ready", 0.9)
            tmp = self.arm.set_position(
                x=ready_pos[0], y=ready_pos[1], z=ready_pos[2],
                roll=ready_cfg.get('roll', 180),
                pitch=ready_cfg.get('pitch', 30),
                yaw=ready_yaw,
                wait=True, speed=speed,
                use_gripper_center=True
            )
            if not tmp:
                raise RuntimeError("Failed to return to ready")

            # === DONE ===
            self.get_logger().info("=== PICK: DONE ===")
            send_feedback(9, "Done", 1.0)

            _, final_pos = self.arm.get_position(return_gripper_center=True)
            result.success = True
            result.category = category
            result.grasp_position = [target_x, target_y, target_z]
            result.execution_time_ms = (time.time() - start_time) * 1000

            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f"Pick failed: {e}")
            send_feedback(10, "Error", 0.0, str(e))

            # Smart recovery
            try:
                if hasattr(self.arm, '_get_error_type'):
                    has_error, error_type = self.arm._get_error_type()
                    if has_error:
                        if error_type == "TARGET_LIMIT":
                            self.arm.clean_error(go_zero=False, allow_disable=False)
                            time.sleep(0.5)
                            try:
                                self.arm.go_ready(speed=20)
                            except Exception:
                                pass
                        elif "JOINT_LIMIT" not in error_type:
                            self.arm.clean_error(go_zero=False, allow_disable=False)
            except Exception:
                pass

            result.success = False
            result.error_message = str(e)
            result.execution_time_ms = (time.time() - start_time) * 1000
            goal_handle.abort()

        finally:
            with self._action_lock:
                self._action_busy = False

        return result

    # ========== Place Action ==========

    async def _execute_place(self, goal_handle):
        """Execute place action"""
        result = PiperPlace.Result()
        feedback = PiperPlace.Feedback()

        grasp_cfg = self.config.get('grasp', {})
        start_time = time.time()

        def send_feedback(state, step_name, progress, message=""):
            feedback.state = state
            feedback.step_name = step_name
            feedback.progress = progress
            feedback.message = message
            goal_handle.publish_feedback(feedback)

        with self._action_lock:
            if self._action_busy:
                result.success = False
                result.error_message = "Another action in progress"
                goal_handle.abort()
                return result
            self._action_busy = True
            self._place_cancelled = False

        try:
            goal = goal_handle.request

            # === CHECKING ===
            self.get_logger().info("=== PLACE: CHECKING ===")

            # Get place position
            if goal.use_default_place:
                place_cfg = self.config.get('positions', {}).get('place', {})
                flange_x = place_cfg.get('x', 16)
                flange_y = place_cfg.get('y', -200)
                flange_z = place_cfg.get('z', 200)
                quat = place_cfg.get('quat', [0.6626, 0.7281, -0.1011, 0.1435])
            else:
                if len(goal.place_position) < 6:
                    raise RuntimeError("place_position must have 6 elements")
                flange_x, flange_y, flange_z = goal.place_position[0:3]
                angles_deg = goal.place_position[3:6]
                rot_custom = R.from_euler('xyz', angles_deg, degrees=True)
                quat = rot_custom.as_quat().tolist()

            speed = goal.speed if goal.speed > 0 else grasp_cfg.get('default_speed', 30)

            # Convert flange to gripper center
            rot = R.from_quat(quat)
            rot_matrix = rot.as_matrix()
            euler_angles = rot.as_euler('xyz', degrees=True)
            place_roll, place_pitch, place_yaw = euler_angles

            gripper_offset_z = self.config.get('gripper', {}).get('offset_mm', 135.03)
            offset_local = np.array([0, 0, gripper_offset_z])
            offset_base = rot_matrix @ offset_local

            place_x = flange_x + offset_base[0]
            place_y = flange_y + offset_base[1]
            place_z = flange_z + offset_base[2]

            # === SAFE LIFT ===
            safe_z = max(place_z, grasp_cfg.get('safe_z', 200))
            self.get_logger().info("=== PLACE: SAFE LIFT ===")
            send_feedback(1, "Lifting to safe height", 0.1)
            self.arm.set_position(z=safe_z, wait=True, speed=speed, use_gripper_center=True)

            if goal_handle.is_cancel_requested:
                self._handle_place_cancel(result, start_time)
                goal_handle.canceled()
                return result

            # === MOVE XY + ORIENTATION ===
            self.get_logger().info("=== PLACE: MOVING XY ===")
            send_feedback(2, "Moving to place XY", 0.2)
            self.arm.set_position(
                x=place_x, y=place_y,
                roll=place_roll, pitch=place_pitch, yaw=place_yaw,
                wait=True, speed=speed, use_gripper_center=True
            )

            # === DESCEND ===
            self.get_logger().info("=== PLACE: DESCENDING ===")
            send_feedback(3, "Descending to place", 0.3)
            use_movel = grasp_cfg.get('use_movel_descent', True)
            self.arm.set_position(z=place_z, wait=True, speed=speed,
                                  use_gripper_center=True, linear=use_movel)

            # === OPENING ===
            self.get_logger().info("=== PLACE: OPENING ===")
            gripper_max = self.config.get('gripper', {}).get('max_mm', 70)
            send_feedback(4, "Opening gripper", 0.5)
            self._set_gripper_mm(gripper_max, speed=500, wait=True)

            # === LIFTING ===
            self.get_logger().info("=== PLACE: LIFTING ===")
            _, current = self.arm.get_position(return_gripper_center=True)
            lift_z = current[2] + 100
            send_feedback(5, "Lifting", 0.7)
            self.arm.set_position(z=lift_z, wait=True, speed=speed, use_gripper_center=True)

            # === RETURNING ===
            self.get_logger().info("=== PLACE: RETURNING ===")
            send_feedback(6, "Returning to ready", 0.8)

            # Close gripper
            self._set_gripper_mm(0, speed=500, wait=True)

            if goal.return_to_ready:
                ready_cfg = self.config.get('positions', {}).get('ready', {})
                self.arm.set_position(
                    x=ready_cfg.get('x', 250),
                    y=ready_cfg.get('y', 37),
                    z=ready_cfg.get('z', 386),
                    roll=ready_cfg.get('roll', -171),
                    pitch=ready_cfg.get('pitch', 28),
                    yaw=ready_cfg.get('yaw', -166),
                    wait=True, speed=speed,
                    use_gripper_center=True
                )

            # === DONE ===
            self.get_logger().info("=== PLACE: DONE ===")
            send_feedback(7, "Done", 1.0)

            _, final_pos = self.arm.get_position(return_gripper_center=True)
            result.success = True
            result.final_position = [float(p) for p in final_pos]
            result.execution_time_ms = (time.time() - start_time) * 1000

            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f"Place failed: {e}")
            send_feedback(10, "Error", 0.0, str(e))

            # Smart recovery
            try:
                if hasattr(self.arm, '_get_error_type'):
                    has_error, error_type = self.arm._get_error_type()
                    if has_error:
                        if error_type == "TARGET_LIMIT":
                            self.arm.clean_error(go_zero=False, allow_disable=False)
                            time.sleep(0.5)
                            try:
                                self.arm.go_ready(speed=20)
                            except Exception:
                                pass
                        elif "JOINT_LIMIT" not in error_type:
                            self.arm.clean_error(go_zero=False, allow_disable=False)
            except Exception:
                pass

            result.success = False
            result.error_message = str(e)
            result.execution_time_ms = (time.time() - start_time) * 1000
            goal_handle.abort()

        finally:
            with self._action_lock:
                self._action_busy = False

        return result

    # ========== Publishing ==========

    def _publish_state(self):
        """Publish current state"""
        if not self.arm or not self.arm.is_connected:
            return

        try:
            stamp = self.get_clock().now().to_msg()

            # Get arm state
            _, pos = self.arm.get_position(return_gripper_center=False)
            _, gc_pos = self.arm.get_position(return_gripper_center=True)
            gripper_mm = self._get_gripper_mm()

            # Joint angles
            DEG_TO_RAD = 0.001 * math.pi / 180.0
            joint_msg = self.arm.piper.GetArmJointMsgs()
            joints_rad = [
                joint_msg.joint_state.joint_1 * DEG_TO_RAD,
                joint_msg.joint_state.joint_2 * DEG_TO_RAD,
                joint_msg.joint_state.joint_3 * DEG_TO_RAD,
                joint_msg.joint_state.joint_4 * DEG_TO_RAD,
                joint_msg.joint_state.joint_5 * DEG_TO_RAD,
                joint_msg.joint_state.joint_6 * DEG_TO_RAD,
            ]

            gripper_m = gripper_mm / 1000.0
            joint_7 = gripper_m
            joint_8 = -gripper_m

            # Joint states
            js = JointState()
            js.header.stamp = stamp
            js.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
            js.position = joints_rad + [joint_7, joint_8]
            self.pub_joint_states.publish(js)

            # End pose euler
            pose_euler = PosCmd()
            pose_euler.x = pos[0]
            pose_euler.y = pos[1]
            pose_euler.z = pos[2]
            pose_euler.roll = pos[3]
            pose_euler.pitch = pos[4]
            pose_euler.yaw = pos[5]
            self.pub_end_pose_euler.publish(pose_euler)

            # Gripper center pose
            gc_pose = PoseStamped()
            gc_pose.header.stamp = stamp
            gc_pose.header.frame_id = "arm_base"
            gc_pose.pose.position.x = gc_pos[0] / 1000.0
            gc_pose.pose.position.y = gc_pos[1] / 1000.0
            gc_pose.pose.position.z = gc_pos[2] / 1000.0
            quat = R.from_euler('xyz', [gc_pos[3], gc_pos[4], gc_pos[5]], degrees=True).as_quat()
            gc_pose.pose.orientation.x = quat[0]
            gc_pose.pose.orientation.y = quat[1]
            gc_pose.pose.orientation.z = quat[2]
            gc_pose.pose.orientation.w = quat[3]
            self.pub_gripper_center.publish(gc_pose)

            # Connection state
            conn_state = UInt8()
            conn_state.data = self._get_connection_state()
            self.pub_connection_state.publish(conn_state)

            # Error state
            err_state = String()
            err_state.data = self.error_state
            self.pub_error_state.publish(err_state)

            # Status
            status = PiperStatus()
            status.connected = self.is_connected
            status.enabled = self.is_enabled
            status.error_state = self.error_state
            status.error_description = self.error_description
            status.position = [float(p) for p in pos]
            status.joints = [j * 180.0 / math.pi for j in joints_rad[:6]]
            status.gripper_position = gripper_mm
            self.pub_status.publish(status)

            # V1 arm status
            try:
                arm_status_msg = self.arm.piper.GetArmStatus()
                low_spd_msg = self.arm.piper.GetArmLowSpdInfoMsgs()

                piper_status = PiperStatusMsg()
                piper_status.ctrl_mode = arm_status_msg.arm_status.ctrl_mode
                piper_status.motion_status = arm_status_msg.arm_status.arm_status
                piper_status.err_code = arm_status_msg.arm_status.err_code

                piper_status.joint_1_angle_limit = arm_status_msg.arm_status.err_status.joint_1_angle_limit
                piper_status.joint_2_angle_limit = arm_status_msg.arm_status.err_status.joint_2_angle_limit
                piper_status.joint_3_angle_limit = arm_status_msg.arm_status.err_status.joint_3_angle_limit
                piper_status.joint_4_angle_limit = arm_status_msg.arm_status.err_status.joint_4_angle_limit
                piper_status.joint_5_angle_limit = arm_status_msg.arm_status.err_status.joint_5_angle_limit
                piper_status.joint_6_angle_limit = arm_status_msg.arm_status.err_status.joint_6_angle_limit

                piper_status.communication_status_joint_1 = low_spd_msg.motor_1.foc_status.driver_enable_status
                piper_status.communication_status_joint_2 = low_spd_msg.motor_2.foc_status.driver_enable_status
                piper_status.communication_status_joint_3 = low_spd_msg.motor_3.foc_status.driver_enable_status
                piper_status.communication_status_joint_4 = low_spd_msg.motor_4.foc_status.driver_enable_status
                piper_status.communication_status_joint_5 = low_spd_msg.motor_5.foc_status.driver_enable_status
                piper_status.communication_status_joint_6 = low_spd_msg.motor_6.foc_status.driver_enable_status

                self.pub_arm_status.publish(piper_status)
            except Exception:
                pass

        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")

    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("Shutting down...")
        try:
            if self.config.get('safety', {}).get('disconnect_go_zero', True):
                self.arm.go_zero()
            safe_disconnect = self.config.get('safety', {}).get('disconnect_damped_stop', True)
            self.arm.disconnect(safe=safe_disconnect)
        except Exception as e:
            self.get_logger().error(f"Shutdown error: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = PiperGraspNode()
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
