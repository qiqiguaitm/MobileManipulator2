"""Isolation regression: a 450 ms mapper callback must not block 20 Hz TF.
Run on robot02 with ROS_DOMAIN_ID=166; no chassis or navigation nodes are started.
"""
import json,os,signal,subprocess,time,tempfile
import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile,DurabilityPolicy
assert os.environ.get('ROS_DOMAIN_ID')=='166'
rclpy.init();n=rclpy.create_node('blocked_mapper_fixture');stamps=[]
q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
pub=n.create_publisher(TransformStamped,'/explore/map_correction',q)
def receive(msg):
 for t in msg.transforms:
  if t.header.frame_id=='map' and t.child_frame_id=='odom':
   assert t.transform.translation.x==1.25
   stamps.append(t.header.stamp.sec+t.header.stamp.nanosec*1e-9)
n.create_subscription(TFMessage,'/tf',receive,200)
def source():
 msg=TransformStamped();msg.header.frame_id='map';msg.child_frame_id='odom'
 msg.header.stamp=n.get_clock().now().to_msg();msg.transform.rotation.w=1.;msg.transform.translation.x=1.25;pub.publish(msg)
 until=time.monotonic()+.45
 while time.monotonic()<until:pass  # Hold the source executor/GIL like a heavy mapper callback.
timer=n.create_timer(.5,source)
with tempfile.TemporaryFile() as log:
 child=subprocess.Popen(['ros2','run','auto_explore','map_odom_tf'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  end=time.monotonic()+5
  while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
  timer.cancel()
  end=time.monotonic()+1.8
  while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
  count=len(stamps);end=time.monotonic()+.5
  while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
  assert count==len(stamps),'TF must expire after source stops'
  gaps=[b-a for a,b in zip(stamps,stamps[1:])]
  assert len(gaps)>40 and max(gaps)<.2,(len(gaps),max(gaps) if gaps else None)
  print(json.dumps(dict(success=True,source_callback_ms=450,tf_samples=len(stamps),max_tf_gap_ms=max(gaps)*1000,stale_source_stopped=True)),flush=True)
 finally:
  os.killpg(child.pid,signal.SIGINT)
  try:child.wait(timeout=3)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  n.destroy_node();rclpy.try_shutdown()
