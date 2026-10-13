"""Own only our processes/interface; refuse to take over an active robot stack."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,math
from pathlib import Path
from datetime import datetime
from ament_index_python.packages import get_package_prefix
from board_approach.motion import parent_death_signal
from .export_map import finalize_run


def can(*args):subprocess.run(['sudo','-n','/usr/sbin/ip','link','set','can2',*args],check=True,timeout=8)

def main():
    p=argparse.ArgumentParser(description='One-command autonomous mapping; defaults to preflight only.')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--preview',action='store_true')
    p.add_argument('--bounds',nargs=4,type=float,metavar=('XMIN','YMIN','XMAX','YMAX'))
    p.add_argument('--max-seconds',type=float,default=0,help='Optional elapsed-time limit; 0 disables it (default).')
    p.add_argument('--max-distance',type=float,default=0,help='Optional cumulative travel limit in metres; 0 disables it (default).')
    p.add_argument('--startup-rear-clearance',type=float,default=.20,help='Startup rear unknown-space clearance in metres; 0 disables, maximum 0.25.')
    p.add_argument('--output');a=p.parse_args()
    if not math.isfinite(a.startup_rear_clearance) or not 0<=a.startup_rear_clearance<=.25:p.error('Startup rear clearance must be between 0 and 0.25 m')
    if not all(math.isfinite(x) and x>=0 for x in (a.max_seconds,a.max_distance)):p.error('Limits must be finite and nonnegative; 0 disables a limit')
    if a.execute or a.preview:
        if not a.bounds or not all(map(math.isfinite,a.bounds)) or not(a.bounds[0]<0<a.bounds[2] and a.bounds[1]<0<a.bounds[3]):
            p.error('--bounds XMIN YMIN XMAX YMAX must surround the mapping start origin')
        if a.bounds[2]-a.bounds[0]>50 or a.bounds[3]-a.bounds[1]>50:p.error('First version supports bounds up to 50 x 50 m')
    root=Path(os.environ['MM2_PROJECT_ROOT']);packages=['auto_explore','slam','sc_pgo','fast_lio','explore_lite','nav2_planner','nav2_controller','nav2_bt_navigator','tracer_base']
    for package in packages:get_package_prefix(package)
    import pypcd4, cv2  # Fail preflight if final-map conversion dependencies are absent.
    lock=(root/'log').resolve();lock.mkdir(exist_ok=True,parents=True)
    with (lock/'auto_explore.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        signatures=('hdl_navigation_launch.py','scpgo_mapping_launch.py','board_approach.runner','fastlio_mapping','tracer_base_node','async_slam_toolbox_node')
        conflicts=[line.strip() for line in processes if any(s in line for s in signatures)]
        state=json.loads(subprocess.check_output(['/usr/sbin/ip','-json','link','show','can2']))[0]
        ready=not conflicts and 'UP' not in state['flags']
        print(json.dumps(dict(preflight_ready=ready,conflicts=conflicts,can_down='UP' not in state['flags']),ensure_ascii=False),flush=True)
        if not (a.execute or a.preview):return 0 if ready else 2
        if not ready:raise SystemExit('Stop the existing mapping/navigation/chassis stack first; no automatic takeover.')
        output=Path(a.output) if a.output else root/'maps/auto_explore'/datetime.now().strftime('%Y%m%d_%H%M%S')
        # SC-PGO uses its save path in shell commands; constrain it until upstream is repaired.
        import re
        if not re.fullmatch(r'[A-Za-z0-9_./-]+',str(output.resolve())):p.error('Output path supports ASCII letters, digits, / _ . - only')
        output.mkdir(parents=True,exist_ok=False);stack=worker=None;owned=False
        def interrupted(*_):raise KeyboardInterrupt
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
        def start(command,log):
            return subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=parent_death_signal)
        def stop(proc,seconds):
            if proc is None:return False
            if proc.poll() is not None:return proc.returncode == 0
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=seconds)
                return proc.returncode == 0
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                return False
        try:
            if a.execute:
                can('type','can','bitrate','500000');owned=True;can('up')
            with (output/'stack.log').open('w') as log,(output/'explorer.log').open('w') as elog:
                stack=start(['ros2','launch','auto_explore','explore.launch.py',f'output:={output}',f'execute:={str(a.execute).lower()}',f'bounds:={json.dumps(a.bounds)}'],log)
                command=[sys.executable,'-m','auto_explore.explorer','--output',str(output),'--bounds',*map(str,a.bounds),
                    '--max-seconds',str(a.max_seconds),'--max-distance',str(a.max_distance),'--startup-rear-clearance',str(a.startup_rear_clearance)]
                if a.execute:command.append('--execute')
                worker=start(command,elog)
                print('Report: '+str(output),flush=True)
                while worker.poll() is None:
                    if stack.poll() is not None:raise RuntimeError('Mapping/navigation launch exited')
                    time.sleep(.5)
                exit_code=worker.returncode
        except KeyboardInterrupt:
            exit_code=130
        finally:
            # Disable physical interface first on all exit paths, before potentially slow map saving.
            try:
                if owned:can('down')
            finally:
                stop(worker,20)
                graceful=stop(stack,45)
                if a.preview:
                    path=output/'result.json'
                    report_data=json.loads(path.read_text()) if path.exists() else dict(status='fault')
                    report_data['navigation_export']=dict(ready=False,reason='preview_only')
                    path.write_text(json.dumps(report_data,indent=2))
                else:
                    # ros2 launch can exit 0 even if a child was forcibly terminated.
                    text=(output/'stack.log').read_text(errors='replace') if (output/'stack.log').exists() else ''
                    graceful=graceful and any('sc_pgo_node-' in line and 'process has finished cleanly' in line for line in text.splitlines())
                    report_data=finalize_run(output,root,graceful)
            report=output/'result.json'
            if report.exists():print(report.read_text(),flush=True)
        return exit_code if a.preview or report_data['navigation_export']['ready'] else 2

if __name__=='__main__':raise SystemExit(main())
