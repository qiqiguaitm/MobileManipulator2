#!/usr/bin/env python3
"""
grasp_demo_node.py - Piper single grasp demo client
ROS2 version

Usage:
  ros2 run piper_grasp grasp_demo_node.py                    # single grasp
  ros2 run piper_grasp grasp_demo_node.py --prompt "bottle"  # specify target
  ros2 run piper_grasp grasp_demo_node.py --no-place         # skip place

All units: mm for length, degrees for angles
"""

import argparse
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from piper_msgs.srv import Observe, GoReady, GetStatus
from piper_msgs.action import PiperPick, PiperPlace


class DemoClient(Node):
    """Single grasp demo client"""

    def __init__(self, prompt='bottle.cup.pen', enable_cdm=True, speed=30, place_after=True):
        super().__init__('demo_client_node')

        # Declare ROS parameters (for launch file compatibility)
        self.declare_parameter('prompt', prompt)
        self.declare_parameter('enable_cdm', enable_cdm)
        self.declare_parameter('speed', speed)
        self.declare_parameter('place_after', place_after)

        # Get parameters (ROS params override constructor args)
        self.prompt = self.get_parameter('prompt').value
        self.enable_cdm = self.get_parameter('enable_cdm').value
        self.speed = self.get_parameter('speed').value
        self.place_after = self.get_parameter('place_after').value

        # Service clients
        self.get_logger().info("Waiting for services...")
        self.status_client = self.create_client(GetStatus, '/piper/get_status')
        self.observe_client = self.create_client(Observe, '/piper/observe')
        self.go_ready_client = self.create_client(GoReady, '/piper/go_ready')

        # Wait for services
        if not self.status_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("GetStatus service not available")
        if not self.observe_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Observe service not available")
        if not self.go_ready_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("GoReady service not available")

        # Action clients
        self.pick_client = ActionClient(self, PiperPick, '/piper/pick')
        self.place_client = ActionClient(self, PiperPlace, '/piper/place')

        self.get_logger().info("Waiting for /piper/pick ...")
        if not self.pick_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Pick action server not available")

        self.get_logger().info("Waiting for /piper/place ...")
        if not self.place_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Place action server not available")

        self.get_logger().info("Demo client initialized")

    def run(self, prompt=None, place_after=None):
        """
        Execute single grasp

        Args:
            prompt: target object (e.g. "bottle.cup")
            place_after: whether to place after pick

        Returns:
            bool: success
        """
        prompt = prompt or self.prompt
        place_after = place_after if place_after is not None else self.place_after

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Starting grasp: {prompt}")
        self.get_logger().info("=" * 50)

        # Step 0: Check arm status
        self.get_logger().info("\n[0/4] Checking arm status")
        try:
            request = GetStatus.Request()
            future = self.status_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            status = future.result()

            if not status.success:
                self.get_logger().error("  Failed to get status")
                return False
            if not status.connected:
                self.get_logger().error("  Arm not connected")
                return False
            if not status.enabled:
                self.get_logger().error("  Arm not enabled")
                return False
            if status.error_state != "NORMAL":
                self.get_logger().error(f"  Arm in error state: {status.error_state}")
                return False
            self.get_logger().info(f"  Position: [{status.gripper_center[0]:.1f}, {status.gripper_center[1]:.1f}, {status.gripper_center[2]:.1f}] mm")
            self.get_logger().info("  Status OK")
        except Exception as e:
            self.get_logger().error(f"  Service exception: {e}")
            return False

        # Step 1: Go to ready position
        self.get_logger().info("\n[1/4] Go to ready position")
        try:
            request = GoReady.Request()
            request.speed = self.speed
            request.open_gripper = False
            future = self.go_ready_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            resp = future.result()

            if not resp.success:
                self.get_logger().error(f"  Failed: {resp.message}")
                return False
            self.get_logger().info("  Done")
        except Exception as e:
            self.get_logger().error(f"  Service exception: {e}")
            return False

        # Small delay
        import time
        time.sleep(0.5)

        # Step 2: Observe (detect target)
        self.get_logger().info(f"\n[2/4] Detecting target: {prompt}")
        try:
            request = Observe.Request()
            request.prompt = prompt
            request.enable_cdm = self.enable_cdm
            future = self.observe_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            resp = future.result()

            if not resp.success:
                self.get_logger().error(f"  Failed: {resp.error_message}")
                return False

            self.get_logger().info(f"  Detected: {resp.category}")
            self.get_logger().info(f"  Base coord: [{resp.point3d_base[0]:.1f}, "
                                   f"{resp.point3d_base[1]:.1f}, {resp.point3d_base[2]:.1f}] mm")
            self.get_logger().info(f"  Gripper width: {resp.gripper_width:.1f} mm")
            self.get_logger().info(f"  Score: {resp.score:.3f}")

            detected_category = resp.category

        except Exception as e:
            self.get_logger().error(f"  Service exception: {e}")
            return False

        # Step 3: Pick
        self.get_logger().info("\n[3/4] Executing pick")
        goal = PiperPick.Goal()
        goal.use_last_observe = True
        goal.speed = self.speed
        goal.return_to_ready = not place_after

        send_goal_future = self.pick_client.send_goal_async(
            goal, feedback_callback=self._pick_feedback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("  Pick goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        result = result_future.result().result

        if not result.success:
            self.get_logger().error(f"  Failed: {result.error_message}")
            return False

        self.get_logger().info(f"  Done: {result.category}")

        # Step 4: Place (optional)
        if place_after:
            self.get_logger().info("\n[4/4] Placing object")
            place_goal = PiperPlace.Goal()
            place_goal.use_default_place = True
            place_goal.speed = self.speed
            place_goal.return_to_ready = True

            send_goal_future = self.place_client.send_goal_async(
                place_goal, feedback_callback=self._place_feedback)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()

            if not goal_handle.accepted:
                self.get_logger().error("  Place goal rejected")
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
            place_result = result_future.result().result

            if not place_result.success:
                self.get_logger().error(f"  Failed: {place_result.error_message}")
                return False

            self.get_logger().info("  Done")
        else:
            self.get_logger().info("\n[4/4] Skipping place")

        self.get_logger().info("\n" + "=" * 50)
        self.get_logger().info(f"Grasp successful: {detected_category}")
        self.get_logger().info("=" * 50)
        return True

    def _pick_feedback(self, feedback_msg):
        """Handle pick action feedback"""
        feedback = feedback_msg.feedback
        progress = feedback.progress * 100
        self.get_logger().info(f"  {feedback.step_name} ({progress:.0f}%)")

    def _place_feedback(self, feedback_msg):
        """Handle place action feedback"""
        feedback = feedback_msg.feedback
        progress = feedback.progress * 100
        self.get_logger().info(f"  {feedback.step_name} ({progress:.0f}%)")


def main(args=None):
    parser = argparse.ArgumentParser(description='Piper single grasp demo')
    parser.add_argument('--prompt', type=str, default='bottle.cup.pen',
                        help='Target object')
    parser.add_argument('--no-place', action='store_true',
                        help='Skip place after pick')
    parser.add_argument('--speed', type=int, default=30,
                        help='Motion speed 1-100%%')
    parsed_args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)

    try:
        client = DemoClient(
            prompt=parsed_args.prompt,
            speed=parsed_args.speed,
            place_after=not parsed_args.no_place
        )
        success = client.run()
        client.destroy_node()
        rclpy.shutdown()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        rclpy.shutdown()
        exit(1)


if __name__ == '__main__':
    main()
