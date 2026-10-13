"""Read-only DWB rejection overlay, with timestamp-checked costmap reconstruction."""
import json, math, time, signal
from collections import deque
from pathlib import Path
import numpy as np
from .geometry import footprint, PLANNING_PADDING
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import Costmap
from dwb_msgs.msg import LocalPlanEvaluation
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields


def seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def outline(pose, body):
    c,s=math.cos(pose.theta),math.sin(pose.theta)
    return [(pose.x+c*x-s*y,pose.y+s*x+c*y) for x,y in body]


def line_cells(a,b):
    x,y=a;xx,yy=b;dx=abs(xx-x);dy=abs(yy-y)
    sx=1 if xx>=x else -1;sy=1 if yy>=y else -1
    # Same dominant-axis rasterization as nav2_util::LineIterator.
    if dx>=dy:
        error=dx//2
        for _ in range(dx+1):
            yield x,y
            x+=sx;error+=dy
            if error>=dx and dx:error-=dx;y+=sy
    else:
        error=dy//2
        for _ in range(dy+1):
            yield x,y
            y+=sy;error+=dx
            if error>=dy:error-=dy;x+=sx


def first_blockage(poses,body,grid,res,origin):
    """Reconstruct footprint-edge rejection; unknown=255, lethal=254."""
    h,w=grid.shape
    for pose in poses:
        vertices=outline(pose,body)
        cells=[(math.floor((x-origin[0])/res),math.floor((y-origin[1])/res)) for x,y in vertices]
        if any(x<0 or y<0 or x>=w or y>=h for x,y in cells):
            return pose,[],[],'outside_costmap'
        edge=set()
        for a,b in zip(cells,cells[1:]+cells[:1]):edge.update(line_cells(a,b))
        unknown=[(x,y) for x,y in edge if grid[y,x]==255]
        obstacle=[(x,y) for x,y in edge if grid[y,x]==254]
        if unknown or obstacle:return pose,unknown,obstacle,'cells'
    return None,[],[],'not_reproduced'


class RotationDebug(Node):
    def __init__(self):
        super().__init__('explore_rotation_debug')
        for k,v in [('output','/tmp/explore_rotation_debug'),('replay',''),
                    ('footprint',[v for xy in footprint(PLANNING_PADDING) for v in xy])]:self.declare_parameter(k,v)
        self.body=np.asarray(self.get_parameter('footprint').value).reshape(-1,2)
        self.out=Path(self.get_parameter('output').value);self.out.mkdir(parents=True,exist_ok=True)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.replay=self.get_parameter('replay').value
        topic='/explore/rotation_debug_recorded' if self.replay else '/explore/rotation_debug'
        self.publisher=self.create_publisher(MarkerArray,topic,qos)
        self.maps=deque(maxlen=12);self.evaluation=None;self.evaluations=deque(maxlen=32);self.global_map=None
        self.markers=None;self.last_eval=0.;self.replay=self.get_parameter('replay').value
        self.goal_markers=None;self.goal_serialized=None;self.goal_signature=None
        if self.replay:
            saved=json.loads(Path(self.replay).read_text());self.markers=MarkerArray();set_message_fields(self.markers,saved['markers'])
            self.goal_pub=self.create_publisher(MarkerArray,'/explore/current_goal_recorded',qos)
            if saved.get('goal_markers'):
                self.goal_markers=MarkerArray();set_message_fields(self.goal_markers,saved['goal_markers'])
                for marker in self.goal_markers.markers:
                    marker.header.frame_id='explore_recorded_map'
                    if marker.type==Marker.TEXT_VIEW_FACING:marker.text='RECORDED\n'+marker.text
            if saved.get('selected_total',-1)<0:self.markers.markers=[m for m in self.markers.markers if m.ns!='selected_trajectory']
            for marker in self.markers.markers:
                marker.header.frame_id='explore_recorded_map'
                if marker.type==Marker.TEXT_VIEW_FACING:marker.text='RECORDED DIAGNOSTIC\n'+marker.text
            self.replay_tf=StaticTransformBroadcaster(self)
            t=TransformStamped();t.header.frame_id='explore_recorded_root';t.child_frame_id='explore_recorded_map';t.transform.rotation.w=1.
            self.replay_tf.sendTransform(t)
            self.map_pub=self.create_publisher(OccupancyGrid,'/explore/debug_recorded_map',qos)
            if saved.get('map'):
                self.global_map=OccupancyGrid();set_message_fields(self.global_map,saved['map']);self.global_map.header.frame_id='explore_recorded_map'
        else:
            self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
            self.create_subscription(Costmap,'/local_costmap/costmap_raw',self.maps.append,qos)
            self.create_subscription(LocalPlanEvaluation,'/evaluation',self.on_eval,10)
            self.create_subscription(OccupancyGrid,'/map',self.on_map,qos)
            self.create_subscription(MarkerArray,'/explore/current_goal',self.on_goal,qos)
        self.create_timer(.5,self.refresh)

    def on_map(self,msg):self.global_map=msg
    def on_eval(self,msg):self.evaluation=msg;self.evaluations.append(msg);self.last_eval=time.monotonic()

    def on_goal(self,msg):
        serialized=message_to_ordereddict(msg)
        # The publisher refreshes header stamps every second; ignore those
        # stamps when deciding whether goal geometry/state actually changed.
        signature=json.dumps([dict(m,header={k:v for k,v in m['header'].items() if k!='stamp'})
                              for m in serialized['markers']])
        if signature==self.goal_signature:return
        self.goal_signature=signature
        self.goal_markers=msg;self.goal_serialized=serialized
        # Also retain targets rejected before DWB ever produces an evaluation.
        saved=dict(markers={'markers':[]},goal_markers=serialized,
                   map=message_to_ordereddict(self.global_map) if self.global_map else None)
        temp=self.out/'navigation_goal.json.tmp';temp.write_text(json.dumps(saved));temp.replace(self.out/'navigation_goal.json')
        with (self.out/'navigation_goals.jsonl').open('a') as f:
            f.write(json.dumps(dict(received_ns=self.get_clock().now().nanoseconds,goal_markers=serialized))+'\n')

    def refresh(self):
        if self.replay:
            if self.global_map:self.map_pub.publish(self.global_map)
            if self.goal_markers:self.goal_pub.publish(self.goal_markers)
            self.publisher.publish(self.markers);return
        if self.evaluation and self.maps and time.monotonic()-self.last_eval<1.5:
            try:
                # The newest evaluation can precede the next 20 Hz TF sample.
                # Render the newest timestamp already covered by TF, not an
                # extrapolated transform and not a blocking wait in this node.
                for evaluation in reversed(self.evaluations):
                    if self.buffer.can_transform('map',evaluation.header.frame_id,Time.from_msg(evaluation.header.stamp)):
                        self.render(evaluation);break
            except Exception as exc:self.get_logger().warning('Rotation debug: '+str(exc))
        elif self.markers:
            # Keep the last observation inspectable, clearly labelled as history.
            for m in self.markers.markers:
                if m.type==Marker.TEXT_VIEW_FACING and not m.text.startswith('LAST OBSERVATION'):
                    m.text='LAST OBSERVATION\n'+m.text
            self.publisher.publish(self.markers)

    def render(self,e):
        if not e.twists:return
        cm=min(self.maps,key=lambda m:abs(seconds(m.header)-seconds(e.header)))
        dt=abs(seconds(cm.header)-seconds(e.header))
        if dt>.35 or cm.header.frame_id!=e.header.frame_id:return
        if abs(cm.metadata.origin.orientation.z)>1e-6:return
        tf=self.buffer.lookup_transform('map',e.header.frame_id,Time.from_msg(e.header.stamp))
        q=tf.transform.rotation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        c,s=math.cos(yaw),math.sin(yaw);tx=tf.transform.translation.x;ty=tf.transform.translation.y
        def point(x,y,z=.08):return Point(x=float(tx+c*x-s*y),y=float(ty+s*x+c*y),z=z)
        markers=[]
        def marker(ns,typ,color):
            m=Marker();m.header.frame_id='map';m.ns=ns;m.id=0;m.type=typ;m.action=Marker.ADD;m.pose.orientation.w=1.
            m.color.r,m.color.g,m.color.b,m.color.a=color
            markers.append(m);return m
        def polygon(ns,pose,color):
            m=marker(ns,Marker.LINE_STRIP,color);m.scale.x=.025
            v=outline(pose,self.body);m.points=[point(x,y) for x,y in v+v[:1]]
        best=e.twists[e.best_index];pose=best.traj.poses[0]
        polygon('current_footprint',pose,(0.,.85,1.,1.))
        arrow=marker('robot_heading',Marker.ARROW,(0.,.85,1.,1.));arrow.scale.x=.035;arrow.scale.y=.10;arrow.scale.z=.12
        arrow.points=[point(pose.x,pose.y,.12),point(pose.x+.65*math.cos(pose.theta),pose.y+.65*math.sin(pose.theta),.12)]
        if best.total>=0:
            m=marker('selected_trajectory',Marker.LINE_STRIP,(0.,1.,.25,1.));m.scale.x=.03;m.points=[point(p.x,p.y,.14) for p in best.traj.poses]
        grid=np.asarray(cm.data).reshape(cm.metadata.size_y,cm.metadata.size_x);res=cm.metadata.resolution
        origin=(cm.metadata.origin.position.x,cm.metadata.origin.position.y)
        unknown=set();obstacles=set();reasons=[];counts=[]
        for sign,name in [(1,'left'),(-1,'right')]:
            rejected=[t for t in e.twists if t.total<0 and abs(t.traj.velocity.x)<1e-6 and sign*t.traj.velocity.theta>.03 and any(cs.name=='ObstacleFootprint' for cs in t.scores)]
            if not rejected:continue
            t=max(rejected,key=lambda t:abs(t.traj.velocity.theta))
            failed,u,o,reason=first_blockage(t.traj.poses,self.body,grid,res,origin)
            counts.append(f'{name}: {len(rejected)} rejected');unknown.update(u);obstacles.update(o)
            reasons.append(reason)
            if failed:polygon('rejected_'+name+'_footprint',failed,(1.,.2,.1,1.))
            # Corner traces make a pure rotation visible even with a stationary centre.
            trace=marker('rejected_'+name+'_sweep',Marker.LINE_LIST,(1.,.45,.1,.75));trace.scale.x=.012
            for a,b in zip(t.traj.poses,t.traj.poses[1:]):
                for p1,p2 in zip(outline(a,self.body),outline(b,self.body)):trace.points.extend([point(*p1,.10),point(*p2,.10)])
        for ns,cells,col in [('unknown_blockers',unknown,(1.,.85,0.,.95)),('obstacle_blockers',obstacles,(1.,0.,.2,.95))]:
            m=marker(ns,Marker.CUBE_LIST,col);m.scale.x=m.scale.y=float(res);m.scale.z=.06
            m.points=[point(origin[0]+(x+.5)*res,origin[1]+(y+.5)*res,.16) for x,y in sorted(cells)]
        m=marker('status',Marker.TEXT_VIEW_FACING,(1.,1.,1.,1.));m.pose.position=point(pose.x,pose.y+.85,.3);m.scale.z=.13
        m.text=f'DWB footprint rejection (grid dt {dt:.2f}s)\nUnknown {len(unknown)} | obstacle {len(obstacles)}\n'+', '.join(counts)
        if any(r!='cells' for r in reasons):m.text+='\n'+', '.join(sorted(set(reasons)-{'cells'}))
        if not counts:m.text=f'DWB: no rejected pure rotation\nv={best.traj.velocity.x:.3f}, w={best.traj.velocity.theta:.3f}'
        if best.total<0:m.text+='\nNO LEGAL TRAJECTORY'
        # Explicit deletion only for disappeared namespaces; no DELETEALL flicker.
        current={m.ns for m in markers}
        if self.markers:
            for old in self.markers.markers:
                if old.ns not in current and old.action!=Marker.DELETE:
                    d=Marker();d.header.frame_id='map';d.ns=old.ns;d.id=old.id;d.action=Marker.DELETE;markers.append(d)
        self.markers=MarkerArray(markers=markers);self.publisher.publish(self.markers)
        saved=dict(markers=message_to_ordereddict(self.markers),map=message_to_ordereddict(self.global_map) if self.global_map else None,
                   goal_markers=self.goal_serialized,
                   selected_total=float(best.total),evaluation_stamp=seconds(e.header),costmap_stamp=seconds(cm.header),costmap_dt=dt,
                   explanation='Blocker cells reconstructed from nearest published grid, not an exact internal controller snapshot.')
        temp=self.out/'rotation_debug.json.tmp';temp.write_text(json.dumps(saved));temp.replace(self.out/'rotation_debug.json')


def main():
    rclpy.init();node=RotationDebug()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        node.destroy_node();rclpy.try_shutdown()
