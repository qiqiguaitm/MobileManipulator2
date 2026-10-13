import math,time,sys
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry,Path,OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped,PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from rclpy.qos import QoSProfile,DurabilityPolicy
from tf2_ros import TransformBroadcaster
rclpy.init();n=Node('explore_test_fixture');o=n.create_publisher(Odometry,'/odom',10);s=n.create_publisher(LaserScan,'/explore/scan',10);tf=TransformBroadcaster(n)
lio=n.create_publisher(Odometry,'/fastlio/odom_measurement',10);cloud=n.create_publisher(PointCloud2,'/fastlio/cloud_body',10)
snapshot=n.create_publisher(Path,'/sc_pgo/pose_snapshot',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL));first=None
if '--relay-map' in sys.argv:
 map_pub=n.create_publisher(OccupancyGrid,'/map',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
 n.create_subscription(OccupancyGrid,'/explore/full_map',map_pub.publish,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
def tick():
 global first
 now=n.get_clock().now().to_msg()
 t=TransformStamped();t.header.stamp=now;t.header.frame_id='odom';t.child_frame_id='base_link';t.transform.rotation.w=1.;tf.sendTransform(t)
 m=Odometry();m.header.stamp=now;m.header.frame_id='odom';m.child_frame_id='base_link';m.pose.pose.orientation.w=1.;o.publish(m)
 z=LaserScan();z.header.stamp=now;z.header.frame_id='base_link';z.angle_min=-math.pi;z.angle_max=math.pi;z.angle_increment=math.pi/180;z.scan_time=.1;z.range_min=.2;z.range_max=8.
 z.ranges=[float(3/max(abs(math.sin(z.angle_min+i*z.angle_increment)),abs(math.cos(z.angle_min+i*z.angle_increment)))) for i in range(361)];s.publish(z)
 m.header.frame_id='camera_init';m.child_frame_id='body';lio.publish(m)
 pts=[]
 for i,d in enumerate(z.ranges):
  a=z.angle_min+i*z.angle_increment
  for height in ((1.4,) if '--relay-map' in sys.argv else (.1,.3,.5,.7,1.)):pts.append((d*math.cos(a),d*math.sin(a),height))
 cloud.publish(point_cloud2.create_cloud_xyz32(Header(stamp=now,frame_id='body'),pts))
 if first is None:first=now
 path=Path();path.header.frame_id='camera_init';path.header.stamp=now
 p=PoseStamped();p.header.frame_id='camera_init';p.header.stamp=first;p.pose.orientation.w=1.
 path.poses=[p,p];snapshot.publish(path)
n.create_timer(.1,tick)
try:rclpy.spin(n)
except KeyboardInterrupt:pass
finally:n.destroy_node();rclpy.try_shutdown()
