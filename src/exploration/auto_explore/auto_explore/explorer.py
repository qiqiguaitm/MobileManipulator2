"""Upstream explore_lite lifecycle with existing chassis safety gate and export."""
import argparse,json,signal,subprocess,time
from pathlib import Path
import numpy as np
import rclpy
from std_msgs.msg import Bool,String
from explore_lite_msgs.msg import ExploreStatus
from rcl_interfaces.srv import GetParameters
from rclpy.qos import QoSProfile,DurabilityPolicy
from ament_index_python.packages import get_package_prefix
from board_approach.safety import can_error
from board_approach.motion import parent_death_signal
from .runtime import ExplorationRuntime


class Explorer(ExplorationRuntime):
    def __init__(self,args):
        super().__init__(args)
        self.mapper={};self.mapper_stamp=0.;self.process=None;self.process_log=None;self.paused=False
        self.returning=False;self.return_started=0.;self.mapper_wait_since=None
        self.startup_blockers={};self.last_startup_report=None
        self.remaining_boundary_cells=None
        self.resume=self.create_publisher(Bool,'/explore/resume',10)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String,'/explore/map_status',self.map_status,qos)
        self.create_subscription(ExploreStatus,'/explore/status',self.explore_status,qos)
    def map_status(self,msg):
        try:self.mapper=json.loads(msg.data);self.mapper_stamp=time.monotonic()
        except Exception:self.error='invalid_mapper_status'
    def mapper_ready(self):return self.mapper.get('ready',False) and time.monotonic()-self.mapper_stamp<2.
    def explore_status(self,msg):
        if self.done:return
        self.event('upstream_status',status=msg.status)
        if msg.status=='returning_to_origin':self.returning=True;self.return_started=time.monotonic()
        if msg.status=='returned_to_origin':
            # Completion means explore_lite exhausted its available targets,
            # including unreachable/blacklisted ones, then returned to origin.
            # Unknown boundaries are diagnostic, not a second completion gate.
            if self.map is not None:
                from scipy import ndimage
                a=np.asarray(self.map.data).reshape(self.map.info.height,self.map.info.width)
                boundary=((a>=0)&(a<=25))&ndimage.binary_dilation(a<0)
                self.remaining_boundary_cells=int(np.count_nonzero(boundary))
            self.stop('complete','exploration_finished_and_returned')
    def gate(self):
        enabled=self.enabled
        if hasattr(self,'mapper') and not self.mapper_ready():self.enabled=False
        try:super().gate()
        finally:self.enabled=enabled and not self.done
    def cancel_goal_async(self):self.resume.publish(Bool(data=False))
    def coordinate(self):
        now=time.monotonic()
        if self.done:return
        nav=self.navigation_ready()
        if self.error:self.stop('fault',self.error);return
        if self.mapper.get('error'):self.stop('fault',self.mapper['error']);return
        if (self.args.max_seconds>0 and now-self.started>self.args.max_seconds) or (self.args.max_distance>0 and self.travel>self.args.max_distance):
            self.stop('limited','time_or_distance_limit');return
        if not self.enabled:
            if self.args.execute and not self.watchdog_ok:
                if self.watchdog_future is None and self.watchdog.service_is_ready():self.watchdog_future=self.watchdog.call_async(GetParameters.Request(names=['command_timeout']))
                if self.watchdog_future is not None and self.watchdog_future.done():
                    v=self.watchdog_future.result().values
                    if len(v)!=1 or abs(v[0].double_value-.4)>.001:self.stop('fault','driver_watchdog_unverified');return
                    self.watchdog_ok=True
            pose_error=None
            try:pose=self.pose3()
            except Exception as exc:pose=None;pose_error=str(exc)
            problem=self.input_problem()
            chassis_error=can_error(self.can_state,now-self.can_time) if self.args.execute else None
            blockers={}
            if problem is not None:blockers['inputs']=problem
            if pose is None:blockers['pose']=pose_error or 'map_to_base_link_unavailable'
            if not nav:
                blockers['navigation']={name:dict(active=s['active'],response_age=round(now-s['stamp'],3))
                                        for name,s in self.nav_states.items() if not s['active'] or now-s['stamp']>=3.}
            if not self.watchdog_ok:blockers['driver_watchdog']='waiting_for_verified_command_timeout'
            if chassis_error:blockers['chassis']=chassis_error
            if not self.mapper_ready():blockers['mapper']=dict(ready=self.mapper.get('ready',False),status_age=round(now-self.mapper_stamp,3))
            self.startup_blockers=blockers
            if blockers:
                timed_out=now-self.started>90
                if timed_out or self.last_startup_report is None or now-self.last_startup_report>=5.:
                    self.event('startup_wait',blockers=blockers,remaining_seconds=max(0.,round(90-(now-self.started),1)))
                    self.last_startup_report=now
                if timed_out:self.stop('fault','startup_inputs_not_ready: '+', '.join(blockers))
                return
            if self.args.execute:
                subs=self.get_subscriptions_info_by_topic('/explore/safe_cmd_vel')
                if len(subs)!=1 or subs[0].node_name!='tracer' or self.count_publishers('/explore/safe_cmd_vel')!=1:self.stop('fault','unexpected_chassis_command_owner');return
            self.home=pose;self.event('ready',pose=pose)
            if not self.args.execute:self.stop('preview','pgo_map_and_navigation_ready_no_motion');return
            self.enabled=True;self.status='exploring';self.last_progress=now
            self.process_log=(self.out/'explore_lite.log').open('w')
            executable=str(Path(get_package_prefix('explore_lite'))/'lib/explore_lite/explore')
            self.process=subprocess.Popen([executable,'--ros-args','--params-file',str(self.out/'explore_lite.yaml')],stdout=self.process_log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=parent_death_signal)
        if not nav:self.stop('fault','navigation_lifecycle_not_active');return
        if self.process and self.process.poll() is not None:self.stop('fault','explore_lite_exited');return
        ready=self.mapper_ready()
        input_hold=self.input_wait_since is not None or not self.fresh()
        if (not ready or input_hold) and not self.paused:
            self.resume.publish(Bool(data=False));self.paused=True
            self.event('exploration_pause',reason='input_hold' if input_hold else 'mapper_not_ready')
        if ready and not input_hold and self.paused:
            self.resume.publish(Bool(data=True));self.paused=False;self.event('exploration_resume')
        if ready:self.mapper_wait_since=None
        elif self.mapper_wait_since is None:self.mapper_wait_since=now
        if not ready:
            if now-self.mapper_wait_since>60:self.stop('fault','mapper_not_ready_for_60s')
            if now-self.mapper_stamp>5:self.stop('fault','mapper_status_stale')
            return
        if input_hold:return
        if self.map is not None:
            known=int(np.count_nonzero(np.asarray(self.map.data)>=0))
            if known>self.known+10:self.known=known;self.last_progress=now
        if not self.returning and now-self.last_progress>180:self.stop('partial','no_map_growth_for_180s')
        if self.returning and now-self.return_started>120:self.stop('partial','return_to_origin_timeout')
    def save(self):
        report=super().save();report.update(exploration_backend='explore_lite',require_return_route=False,
            map_frame='map (SC-PGO optimized anchors + aligned LIO tail)',cloud_frame='camera_init (SC-PGO; same global coordinates as map)',mapper=self.mapper,
            startup_blockers=self.startup_blockers,
            completion_criterion='no_available_explore_lite_targets_and_returned',
            remaining_boundary_cells=self.remaining_boundary_cells)
        (self.out/'result.json').write_text(json.dumps(report,indent=2));return report
    def shutdown_upstream(self):
        if self.process and self.process.poll() is None:
            self.resume.publish(Bool(data=False));self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.process_log:self.process_log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True);p.add_argument('--bounds',nargs=4,type=float,required=True)
    p.add_argument('--max-seconds',type=float,default=0.);p.add_argument('--max-distance',type=float,default=0.)
    p.add_argument('--startup-rear-clearance',type=float,default=.20);args=p.parse_args()
    rclpy.init();n=Explorer(args)
    def interrupt(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
        while rclpy.ok() and not n.done:rclpy.spin_once(n,timeout_sec=.1)
    except KeyboardInterrupt:n.stop('interrupted','operator_stop')
    except Exception as e:n.stop('fault',str(e))
    finally:
        # CAN shutdown can enter cleanup just before the supervisor sends SIGINT.
        # Do not let that second interrupt skip the report/map save.
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        n.enabled=False;n.shutdown_upstream();report=n.save()
        if n.can:n.can.close()
        n.destroy_node();rclpy.shutdown()
    print(json.dumps(report),flush=True)
    return 0 if report['status'] in ('complete','preview') else 2

if __name__=='__main__':raise SystemExit(main())
