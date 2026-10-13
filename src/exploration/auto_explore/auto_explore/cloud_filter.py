"""Remove returns physically inside the base footprint before exploration mapping."""
import numpy as np
from .geometry import HALF_LENGTH, HALF_WIDTH, SELF_FILTER_PADDING
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer,TransformListener


def outside_body(points):
    p=np.asarray(points)
    return p[~((np.abs(p[:,0])<=HALF_LENGTH+SELF_FILTER_PADDING)&(np.abs(p[:,1])<=HALF_WIDTH+SELF_FILTER_PADDING))]


class Filter(Node):
    def __init__(self):
        super().__init__('explore_cloud_filter');self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.out=self.create_publisher(PointCloud2,'/explore/obstacle_cloud',qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'/rslidar_points',self.receive,qos_profile_sensor_data)
    def receive(self,m):
        try:t=self.buffer.lookup_transform('base_link',m.header.frame_id,Time.from_msg(m.header.stamp))
        except Exception:return
        p=point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)
        if isinstance(p,np.ndarray) and p.dtype.names:p=np.column_stack([p[k] for k in ('x','y','z')])
        else:p=np.asarray(list(p))
        p=np.asarray(p,dtype=np.float32).reshape(-1,3)
        q=t.transform.rotation;x,y,z,w=q.x,q.y,q.z,q.w
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        v=t.transform.translation;p=p@R.T+[v.x,v.y,v.z]
        p=outside_body(p[np.isfinite(p).all(axis=1)])
        self.out.publish(point_cloud2.create_cloud_xyz32(Header(stamp=m.header.stamp,frame_id='base_link'),p))

def main():
    rclpy.init();n=Filter()
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    finally:n.destroy_node();rclpy.try_shutdown()
