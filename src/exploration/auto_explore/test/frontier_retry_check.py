"""Isolated action/marker integration check. No robot drivers or velocity publishers."""
import json, os, subprocess, time, threading, signal
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rosidl_runtime_py.set_message import set_message_fields
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster

assert os.environ['ROS_DOMAIN_ID']=='166'
rclpy.init();n=Node('goal_marker_check');qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
seen=[];recorded=[];goals=[];attempts=[];frontiers=[];finish=threading.Event();processes=[];logs=[]
def execute(handle):
    goals.append(handle.request.pose)
    if handle.request.pose.pose.position.x < 0:
        time.sleep(.25);handle.abort();return NavigateToPose.Result()
    while not finish.wait(.05):
        if handle.is_cancel_requested:
            handle.canceled();return NavigateToPose.Result()
    handle.succeed();return NavigateToPose.Result()
def accept(request):
    attempts.append((time.monotonic(),request.pose))
    return GoalResponse.REJECT if len(attempts)==1 else GoalResponse.ACCEPT
server=ActionServer(n,NavigateToPose,'navigate_to_pose',execute_callback=execute,goal_callback=accept,cancel_callback=lambda h:CancelResponse.ACCEPT,callback_group=ReentrantCallbackGroup())
mp=n.create_publisher(OccupancyGrid,'/map',qos);cp=n.create_publisher(OccupancyGrid,'/costmap',qos)
resume=n.create_publisher(Bool,'/explore/resume',10)
n.create_subscription(MarkerArray,'/explore/current_goal',seen.append,qos)
n.create_subscription(MarkerArray,'/explore/frontiers',frontiers.append,10)
n.create_subscription(MarkerArray,'/explore/current_goal_recorded',recorded.append,qos)
tf=StaticTransformBroadcaster(n);t=TransformStamped();t.header.frame_id='map';t.child_frame_id='base_link';t.transform.rotation.w=1.;tf.sendTransform(t)
m=OccupancyGrid();set_message_fields(m,json.loads(Path('/tmp/frontier_capture/global_costmap.json').read_text()))
def tick():m.header.stamp=n.get_clock().now().to_msg();mp.publish(m);cp.publish(m)
n.create_timer(.2,tick);ex=MultiThreadedExecutor(num_threads=3);ex.add_node(n);thread=threading.Thread(target=ex.spin,daemon=True);thread.start()
out=Path('/tmp/goal_marker_check_'+str(os.getpid()));out.mkdir()
def launch(args,name):
    log=(out/(name+'.log')).open('w');logs.append(log)
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);return p
def wait(pred,label,timeout=15):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if pred():return
        time.sleep(.1)
    raise AssertionError(label)
def label(msg):return next(x.text for x in msg.markers if x.ns=='current_navigation_goal_label')
try:
    launch(['ros2','run','auto_explore','rotation_debug','--ros-args','-p','output:='+str(out)],'recorder')
    explore=launch(['ros2','run','explore_lite','explore','--ros-args','-p','costmap_topic:=/costmap','-p','visualize:=true','-p','planner_frequency:=0.5','-p','return_to_init:=false'],'explore')
    wait(lambda:goals and goals[-1].pose.position.x>0 and seen and '/ ACTIVE' in label(seen[-1]),'switch from failed rear targets to front target',timeout=20)
    assert len(attempts)>=3
    assert max(b[0]-a[0] for a,b in zip(attempts,attempts[1:]))<2.0
    assert any(m.type==2 and m.pose.position.x>0 for m in frontiers[-1].markers)
    accepted_count=len(attempts);time.sleep(3.0);assert len(attempts)==accepted_count, 'stable active target must not be resubmitted'
    print('attempts',[(round(t-attempts[0][0],2),round(p.pose.position.x,2),round(p.pose.position.y,2)) for t,p in attempts],flush=True)
    arrow=next(x for x in seen[-1].markers if x.ns=='current_navigation_goal_pointer')
    assert abs(arrow.points[0].x-goals[-1].pose.position.x)<1e-8
    assert abs(arrow.points[0].y-goals[-1].pose.position.y)<1e-8
    assert all(x.lifetime.sec==0 and x.lifetime.nanosec==0 for x in seen[-1].markers)
    late=[];n.create_subscription(MarkerArray,'/explore/current_goal',late.append,qos)
    wait(lambda:late,'late subscriber marker');before=len(seen)
    wait(lambda:len(seen)>before,'periodic marker refresh')
    resume.publish(Bool(data=False))
    wait(lambda:seen and 'LAST GOAL' in label(seen[-1]),'stopped goal status')
    wait(lambda:(out/'navigation_goal.json').exists() and 'LAST GOAL' in (out/'navigation_goal.json').read_text(),'saved final status')
    launch(['ros2','run','auto_explore','rotation_debug','--ros-args','-p','replay:='+str(out/'navigation_goal.json')],'replay')
    wait(lambda:recorded,'recorded goal marker')
    assert label(recorded[-1]).startswith('RECORDED\nLAST GOAL')
    assert all(x.header.frame_id=='explore_recorded_map' for x in recorded[-1].markers)
    events=[json.loads(s) for s in (out/'navigation_goals.jsonl').read_text().splitlines()]
    assert len(events)>=2
    print(json.dumps(dict(success=True,checks=['actual costmap front candidates','rejected goal switches','aborted rear goals switch to front','no boundary jitter preemption','exact action target','late subscriber','periodic refresh','infinite lifetime','stop state','goal event recording','read-only replay'],output=str(out)),indent=2),flush=True)
finally:
    finish.set()
    for p in processes:
        if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
    for p in processes:
        try:p.wait(timeout=4)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    ex.shutdown();thread.join(timeout=2);n.destroy_node();rclpy.shutdown()
    for log in logs:log.close()
