#!/usr/bin/env python3
"""Bounded, guarded Nav2 approach; defaults to a stationary preflight."""
import argparse,ctypes,json,math,os,re,signal,socket,struct,subprocess,time
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from hdl_localization.msg import ScanMatchingStatus
from nav2_msgs.action import NavigateToPose
from action_msgs.srv import CancelGoal
from scipy.spatial.transform import Rotation
ROOT='/home/didi/junbin.zhang/MobileManipulator2'
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
goal=json.load(open('/tmp/junbin-board-coarse-goal.json'))
cal=json.load(open('/tmp/junbin-board-preapproach.json'))['base_link<-rslidar']
rot=Rotation.from_quat(cal['q']).as_matrix();trans=np.array(cal['t'])
rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
node=rclpy.create_node('junbin_board_guard')
pub=node.create_publisher(Twist,'/junbin_safe_cmd_vel',10)
data={};seen={};scanpts=None;cmd=Twist();cmd_at=0.;active=False;reason=None;driver=None;handle=None
start=None;startxy=None;travel=0.;lastxy=None
sock=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW);sock.bind(('can2',));sock.setblocking(False)
can_at=0.;can_state=None;motion=None
def child_parent_death():
 if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError('Cannot set chassis parent-death signal')
def stop_signal(s,f): raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
def log(**k):print(json.dumps(k),flush=True)
def cb(k,m):
 global scanpts,travel,lastxy
 data[k]=m;seen[k]=time.monotonic()
 if k=='scan':
  r=np.array(m.ranges);ang=m.angle_min+np.arange(len(r))*m.angle_increment
  valid=np.isfinite(r)&(r>m.range_min)&(r<m.range_max)
  pts=np.stack([r[valid]*np.cos(ang[valid]),r[valid]*np.sin(ang[valid]),np.zeros(valid.sum())],axis=1)
  scanpts=pts@rot.T+trans
 if k=='pose' and active:
  p=m.pose.pose.position;xy=np.array([p.x,p.y])
  if lastxy is not None:travel+=float(np.linalg.norm(xy-lastxy))
  lastxy=xy
def command(m):
 global cmd,cmd_at
 cmd=m;cmd_at=time.monotonic()
subs=[node.create_subscription(typ,topic,lambda m,k=k:cb(k,m),qos_profile_sensor_data) for k,topic,typ in [
 ('pose','/pose',Odometry),('odom','/odom',Odometry),('status','/status',ScanMatchingStatus),('scan','/scan',LaserScan)]]
subs.append(node.create_subscription(Twist,'/cmd_vel',command,10))
client=ActionClient(node,NavigateToPose,'/navigate_to_pose')
cancel=node.create_client(CancelGoal,'/navigate_to_pose/_action/cancel_goal')
def read_can():
 global can_at,can_state,motion
 while True:
  try:b=sock.recv(16)
  except BlockingIOError:break
  ident,l,p=struct.unpack('=IB3x8s',b)
  if ident&0x7ff==0x211:can_state=list(p);can_at=time.monotonic()
  elif ident&0x7ff==0x221:motion=list(p)
def health():
 now=time.monotonic()
 for k in ['pose','odom','status','scan']:
  if now-seen.get(k,0)>.65:return k+' data stale'
 if now-can_at>.4:return 'CAN feedback stale'
 if can_state[0]!=0 or can_state[1]!=1 or can_state[4]!=0 or can_state[5]!=0:return 'CAN state abnormal '+str(can_state)
 if not data['status'].has_converged or data['status'].inlier_fraction<.9:return 'localization confidence dropped'
 if scanpts is None or len(scanpts)<30:return 'insufficient lidar points'
 # Rectangular forward stopping envelope; front of robot is x=0.434 m.
 corridor=scanpts[(scanpts[:,0]>.15)&(abs(scanpts[:,1])<.48)]
 if len(corridor) and float(corridor[:,0].min())<.9:return 'obstacle in forward stopping envelope'
 if active and travel>1.5:return 'travel budget exceeded'
 if active:
  p=data['pose'].pose.pose.position
  if np.linalg.norm(np.array([p.x,p.y])-startxy)>1.0:return 'displacement budget exceeded'
 return None
def tick():
 global reason
 read_can()
 out=Twist()
 if active and reason is None:
  reason=health()
  if reason is None and time.monotonic()-cmd_at<.35:
   vals=[cmd.linear.x,cmd.linear.y,cmd.angular.z]
   if not all(math.isfinite(v) for v in vals):reason='invalid velocity command'
   elif cmd.linear.x < -.005:reason='reverse requested; reverse disabled'
   elif abs(cmd.linear.y)>.001:reason='lateral velocity requested'
   elif abs(cmd.angular.z)>.01 and np.any(np.linalg.norm(scanpts[:,:2],axis=1)<.65):reason='obstacle in turning clearance'
   else:
    scale=min(1.,.05/max(abs(cmd.linear.x),1e-6),.15/max(abs(cmd.angular.z),1e-6))
    out.linear.x=max(0.,cmd.linear.x)*scale;out.angular.z=cmd.angular.z*scale
 pub.publish(out)
def pump(duration):
 end=time.monotonic()+duration
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.02);tick()
def await_future(f,seconds):
 end=time.monotonic()+seconds
 while not f.done() and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.02);tick()
 if not f.done():raise RuntimeError('ROS request timeout')
 return f.result()
try:
 pump(4)
 issue=health()
 if issue:raise RuntimeError('Preflight: '+issue)
 log(stage='preflight',inlier=data['status'].inlier_fraction,can=can_state,scan_points=len(scanpts),goal=goal['goal_map'],execute=args.execute)
 if not args.execute:raise SystemExit(0)
 if not cancel.wait_for_service(timeout_sec=2):raise RuntimeError('Cancel service unavailable')
 await_future(cancel.call_async(CancelGoal.Request()),3)
 if not client.wait_for_server(timeout_sec=3):raise RuntimeError('Nav2 unavailable')
 # Driver receives only guarded velocities. It cannot publish conflicting odom TF.
 driver=subprocess.Popen(['/home/didi/junbin.zhang/MobileManipulator2/install/tracer_base/lib/tracer_base/tracer_base_node','--ros-args','-r','/cmd_vel:=/junbin_safe_cmd_vel','-p','port_name:=can2','-p','publish_tf:=false','-p','odom_topic_name:=/junbin_guard_wheel_odom','-p','odom_frame:=wheel_odom'],stdout=open('/tmp/junbin-board-chassis.log','w'),stderr=subprocess.STDOUT,preexec_fn=child_parent_death)
 pump(7)
 if driver.poll() is not None:raise RuntimeError('Chassis driver exited')
 if pub.get_subscription_count()!=1:raise RuntimeError('Expected one guarded chassis subscriber')
 if len(node.get_publishers_info_by_topic('/junbin_safe_cmd_vel'))!=1:raise RuntimeError('Unexpected guarded command publisher')
 issue=health()
 if issue:raise RuntimeError('Before goal: '+issue)
 p=data['pose'].pose.pose.position;startxy=np.array([p.x,p.y]);lastxy=startxy.copy()
 g=NavigateToPose.Goal();g.pose.header.frame_id='map';g.pose.header.stamp=node.get_clock().now().to_msg()
 g.pose.pose.position.x=goal['goal_map'][0];g.pose.pose.position.y=goal['goal_map'][1]
 g.pose.pose.orientation.z=math.sin(goal['goal_yaw']/2);g.pose.pose.orientation.w=math.cos(goal['goal_yaw']/2)
 g.behavior_tree='/tmp/junbin-board-no-recovery.xml'
 start=time.monotonic();active=True
 handle=await_future(client.send_goal_async(g),5)
 if not handle.accepted:raise RuntimeError('Navigation goal rejected')
 result=handle.get_result_async();last_report=0
 while time.monotonic()-start<90 and not result.done() and reason is None:
  rclpy.spin_once(node,timeout_sec=.02);tick()
  if driver.poll() is not None:reason='Chassis driver exited'
  if time.monotonic()-last_report>3:
   p=data['pose'].pose.pose.position
   log(stage='moving',xy=[p.x,p.y],inlier=data['status'].inlier_fraction,travel=travel,reason=reason);last_report=time.monotonic()
 active=False;pub.publish(Twist())
 if reason:raise RuntimeError(reason)
 if not result.done():raise RuntimeError('Navigation deadline exceeded')
 status=result.result().status
 p=data['pose'].pose.pose.position
 out={'status':status,'position':[p.x,p.y],'goal':goal['goal_map'],'position_error':float(np.linalg.norm(np.array([p.x,p.y])-goal['goal_map'][:2])),'travel':travel}
 log(stage='result',**out);open('/tmp/junbin-board-move-result.json','w').write(json.dumps(out))
 if status!=4:raise RuntimeError('Navigation did not succeed')
except (Exception,KeyboardInterrupt) as e:
 log(stage='stopped',reason=str(e))
finally:
 active=False
 try:
  if handle is not None:handle.cancel_goal_async()
  pump(2)
 finally:
  if driver is not None:
   driver.send_signal(signal.SIGINT)
   try:driver.wait(timeout=3)
   except subprocess.TimeoutExpired:driver.terminate();driver.wait(timeout=3)
  pw=re.search(r'sudo 的密码是(\S+)',open(ROOT+'/CLAUDE.md').read()).group(1)
  p=subprocess.run(['sudo','-S','-p','','ip','link','set','can2','down'],input=pw+'\n',text=True,capture_output=True)
  log(stage='safe_shutdown',can_down=p.returncode==0)
  node.destroy_node();rclpy.shutdown();sock.close()
