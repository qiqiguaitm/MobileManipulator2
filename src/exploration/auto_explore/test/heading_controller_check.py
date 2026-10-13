"""Exercise installed Nav2 with fake odometry, isolated from robot hardware.

Run explicitly with ROS_DOMAIN_ID=166. No chassis node is launched. The only
command subscriber integrates a mathematical pose; CAN must remain down.
"""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path as RosPath
from nav2_msgs.action import FollowPath
from lifecycle_msgs.srv import ChangeState
from tf2_ros import TransformBroadcaster
from launch import LaunchContext
import yaml


def main():
    assert os.environ.get('ROS_DOMAIN_ID') == '166', 'Isolated ROS domain required'
    can = json.loads(subprocess.check_output(['ip', '-json', 'link', 'show', 'can2']))[0]
    assert 'UP' not in can['flags'], 'Run only with chassis CAN down'
    out = Path('/tmp/heading_controller_check_' + str(os.getpid()))
    out.mkdir()
    launch_file = Path(__file__).resolve().parents[1] / 'launch/explore.launch.py'
    spec = importlib.util.spec_from_file_location('exploration_launch_test', launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update(output=str(out), execute='false', bounds='[-10,-10,10,10]')
    module.setup(context)  # Generate the production config; do not execute its nodes.
    config = yaml.safe_load((out / 'nav2.yaml').read_text())
    config.pop('planner_server', None)
    config.pop('bt_navigator', None)
    config.pop('global_costmap', None)
    local = config['local_costmap']['local_costmap']['ros__parameters']
    local.update(global_frame='odom', plugins=['static_layer', 'inflation_layer'],
                 width=10, height=10, update_frequency=10., publish_frequency=5.)
    local['static_layer'].update(map_topic='/test/map', map_subscribe_transient_local=True)
    # Use the production inflation and padded footprint. No scan driver needed.
    rclpy.init()
    n = Node('heading_controller_check')
    tf = TransformBroadcaster(n)
    odom = n.create_publisher(Odometry, '/odom', 10)
    maps = n.create_publisher(OccupancyGrid, '/test/map',
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    state = dict(x=0., y=0., yaw=0., v=0., w=0., stamp=0., integrate=False)
    commands = []
    def command(msg):
        state.update(v=msg.linear.x, w=msg.angular.z, stamp=time.monotonic())
        commands.append((time.monotonic(), msg.linear.x, msg.angular.z))
    n.create_subscription(Twist, '/test/controller_cmd', command, 10)
    last = time.monotonic()
    def tick():
        nonlocal last
        now = time.monotonic(); dt = min(now-last, .1); last = now
        v, w = (state['v'], state['w']) if now-state['stamp'] < .4 else (0., 0.)
        if state['integrate']:
            state['x'] += v*math.cos(state['yaw'])*dt
            state['y'] += v*math.sin(state['yaw'])*dt
            state['yaw'] += w*dt
        stamp = n.get_clock().now().to_msg()
        t = TransformStamped(); t.header.stamp = stamp
        t.header.frame_id = 'odom'; t.child_frame_id = 'base_link'
        t.transform.translation.x = state['x']; t.transform.translation.y = state['y']
        t.transform.rotation.z = math.sin(state['yaw']/2)
        t.transform.rotation.w = math.cos(state['yaw']/2); tf.sendTransform(t)
        m = Odometry(); m.header = t.header; m.child_frame_id = 'base_link'
        m.pose.pose.position.x = state['x']; m.pose.pose.position.y = state['y']
        m.pose.pose.orientation = t.transform.rotation
        # Match the real FAST-LIO interface: measured pose, default-zero twist.
        odom.publish(m)
    n.create_timer(.02, tick)
    def wait(pred, timeout, label):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=.02)
            if pred(): return
        raise AssertionError(label)
    def spin(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end: rclpy.spin_once(n, timeout_sec=.02)
    results = []
    try:
        for case, heading, blocked, integrate in [
            ('rear_turn_then_track', 2.8, False, True),
            ('front_tracks_without_turn', 0., False, True),
            ('blocked_footprint_stops', 2.8, True, False),
        ]:
            state.update(x=0., y=0., yaw=0., v=0., w=0., stamp=0., integrate=integrate)
            commands.clear()
            cfg = copy.deepcopy(config)
            params = out / (case+'.yaml'); params.write_text(yaml.safe_dump(cfg))
            log = (out/(case+'.log')).open('w')
            p = subprocess.Popen(['ros2','run','nav2_controller','controller_server',
                '--ros-args','--params-file',str(params),'-r','cmd_vel:=/test/controller_cmd'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            client = ActionClient(n, FollowPath, '/follow_path')
            service = n.create_client(ChangeState, '/controller_server/change_state')
            try:
                wait(service.service_is_ready, 12, 'controller lifecycle unavailable')
                grid = OccupancyGrid(); grid.header.frame_id = 'odom'
                grid.info.resolution = .05; grid.info.width = grid.info.height = 200
                grid.info.origin.position.x = grid.info.origin.position.y = -5.
                grid.info.origin.orientation.w = 1.
                grid.data = [100 if blocked else 0] * 40000
                for transition in (1, 3):
                    maps.publish(grid)
                    req=ChangeState.Request(); req.transition.id=transition
                    f=service.call_async(req)
                    wait(f.done, 12, 'lifecycle transition timed out')
                    assert f.result().success, 'lifecycle failed'
                    spin(.5)
                wait(client.server_is_ready, 8, 'FollowPath not ready')
                goal=FollowPath.Goal(); goal.controller_id='FollowPath'; goal.goal_checker_id='goal_checker'
                path=RosPath(); path.header.frame_id='odom'; path.header.stamp=n.get_clock().now().to_msg()
                for i in range(61):
                    pose=PoseStamped(); pose.header=path.header
                    pose.pose.position.x=i*.05*math.cos(heading)
                    pose.pose.position.y=i*.05*math.sin(heading)
                    pose.pose.orientation.z=math.sin(heading/2); pose.pose.orientation.w=math.cos(heading/2)
                    path.poses.append(pose)
                goal.path=path; f=client.send_goal_async(goal)
                wait(f.done, 5, 'goal acceptance'); handle=f.result(); assert handle.accepted
                result=handle.get_result_async()
                if blocked:
                    spin(3.)
                    assert not any(abs(v)>.001 or abs(w)>.001 for _,v,w in commands), commands[-5:]
                else:
                    wait(lambda:math.hypot(state['x'],state['y'])>.20 or result.done(), 30, 'no forward progress')
                    assert math.hypot(state['x'],state['y'])>.20, 'controller aborted before forward progress'
                    if heading:
                        assert any(abs(v)<.001 and w>.05 for _,v,w in commands), 'no initial turn'
                        assert abs(math.atan2(math.sin(heading-state['yaw']),math.cos(heading-state['yaw'])))<.5
                    else:
                        assert max(abs(w) for _,_,w in commands)<.05, 'unnecessary turn'
                assert all(-1e-6<=v<=.100001 and abs(w)<=.180001 for _,v,w in commands)
                if not result.done():
                    f=handle.cancel_goal_async();wait(f.done,5,'cancel')
                record=dict(case=case,passed=True,pose=[state[k] for k in ('x','y','yaw')],commands=len(commands))
                results.append(record);print(json.dumps(record),flush=True)
            finally:
                if p.poll() is None: os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=6)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                log.close(); client.destroy(); n.destroy_client(service); spin(.3)
        (out/'result.json').write_text(json.dumps(results,indent=2))
        print('PASS '+str(out),flush=True)
    finally:
        n.destroy_node();rclpy.shutdown()


if __name__ == '__main__': main()
