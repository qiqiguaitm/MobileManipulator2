#!/usr/bin/env python3
"""Desktop controls for isolated, visualization-only ROS 2 bag playback."""
import argparse
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time

import yaml
from PyQt5 import QtCore, QtWidgets

HERE = Path(__file__).resolve().parent


def bag_metadata(selection):
    path = Path(selection).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if (path/'bag/metadata.yaml').exists():
        path = path/'bag'
    metadata = yaml.safe_load((path/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    types = {x['topic_metadata']['name']: x['topic_metadata']['type'] for x in metadata['topics_with_message_count']}
    # Strict allowlist: never replay velocity commands, goals or task actions.
    allowed = {
        '/map': 'nav_msgs/msg/OccupancyGrid', '/tf': 'tf2_msgs/msg/TFMessage',
        '/tf_static': 'tf2_msgs/msg/TFMessage', '/scan': 'sensor_msgs/msg/LaserScan',
        '/trajectory': 'geometry_msgs/msg/PoseStamped',
        '/camera/top/color/image_raw': 'sensor_msgs/msg/Image',
    }
    selected, remaps = [], []
    for name, kind in allowed.items():
        source = '/reference'+name if '/reference'+name in types else name
        if name == '/trajectory' and '/simulation/trajectory' in types:
            source = '/simulation/trajectory'
        if types.get(source) != kind:
            continue
        selected.append(source)
        target = name if name in ('/tf', '/tf_static') else '/reference'+name
        if source != target:
            remaps.append(source+':='+target)
    for source, kind, target in [
        ('/reference/path', 'nav_msgs/msg/Path', '/replay/reference_path_input'),
        ('/plan', 'nav_msgs/msg/Path', '/replay/planned_path'),
        ('/goal_pose', 'geometry_msgs/msg/PoseStamped', '/replay/display_goal'),
    ]:
        if types.get(source) == kind:
            selected.append(source)
            remaps.append(source+':='+target)
    if not selected:
        raise ValueError('Bag 中没有支持的地图、激光、TF、轨迹或 RGB topic')
    return path, metadata, selected, remaps


class Engine(QtCore.QThread):
    state = QtCore.pyqtSignal(str, bool)
    loaded = QtCore.pyqtSignal(str, float)
    position = QtCore.pyqtSignal(float)
    error = QtCore.pyqtSignal(str)

    def __init__(self, domain):
        super().__init__()
        self.commands = queue.Queue()
        self.children = {}
        self.path = None
        self.playing = False
        self.step_target = None
        self.current = 0.0
        self.domain = domain
        self.run_dir = Path.home()/'.ros/navigation_bag_viewer'/time.strftime('%Y%m%d_%H%M%S')
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def send(self, command, value=None):
        self.commands.put((command, value))

    def spawn(self, name, args):
        with (self.run_dir/(name+'.log')).open('w') as output:
            self.children[name] = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)

    def stop(self):
        self.playing = False
        self.step_target = None
        for p in self.children.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        deadline = time.monotonic()+4
        for p in self.children.values():
            try:
                p.wait(timeout=max(0.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        self.children.clear()

    def service(self, kind, name, request):
        client = self.node.create_client(kind, '/rosbag2_player/'+name)
        try:
            if not client.wait_for_service(timeout_sec=6):
                raise RuntimeError('回放服务未就绪：'+name)
            future = client.call_async(request)
            self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=6)
            if not future.done():
                raise RuntimeError('回放服务超时：'+name)
            result = future.result()
            if hasattr(result, 'success') and not result.success:
                raise RuntimeError('回放操作失败：'+name)
            return result
        finally:
            self.node.destroy_client(client)

    def open(self, selection):
        path, meta, topics, remaps = bag_metadata(selection)
        self.state.emit('正在打开…', False)
        self.stop()
        self.path = path
        self.start_ns = meta['starting_time']['nanoseconds_since_epoch']
        self.duration = meta['duration']['nanoseconds']/1e9
        self.current = 0.0
        qos = {topic: dict(history='keep_last', depth=100, reliability='reliable',
            durability='transient_local' if topic.endswith(('/map','/tf_static')) else 'volatile') for topic in topics}
        (self.run_dir/'qos.yaml').write_text(yaml.safe_dump(qos))
        self.spawn('trajectory', [sys.executable, str(HERE/'trajectory_view.py'), '--bag', str(path), '--ros-args', '-p', 'use_sim_time:=true'])
        self.spawn('rviz', ['rviz2', '-d', str(HERE/'replay.rviz'), '--ros-args', '-r', '__node:=bag_viewer_rviz', '-p', 'use_sim_time:=true'])
        args = ['ros2','bag','play',str(path),'--clock','--start-paused',
            '--disable-keyboard-controls','--read-ahead-queue-size','200',
            '--qos-profile-overrides-path',str(self.run_dir/'qos.yaml'),'--topics',*topics]
        if remaps:
            args += ['--remap',*remaps]
        self.spawn('player',args)
        from rosbag2_interfaces.srv import IsPaused
        if not self.service(IsPaused,'is_paused',IsPaused.Request()).paused:
            raise RuntimeError('播放器未进入暂停状态')
        self.loaded.emit(str(path),self.duration)
        self.position.emit(0.0)
        self.state.emit('已打开 · 0 秒暂停',True)
        (self.run_dir/'session.json').write_text(json.dumps(dict(bag=str(path),domain=self.domain,
            processes={k:p.pid for k,p in self.children.items()}),indent=2))

    def pause(self):
        from rosbag2_interfaces.srv import Pause
        self.service(Pause,'pause',Pause.Request())
        self.playing = False
        self.step_target = None
        self.state.emit('已暂停',True)

    def resume(self, step=False):
        from rosbag2_interfaces.srv import Resume
        if not self.path:
            return
        if 'player' not in self.children or self.children['player'].poll() is not None:
            self.open(self.path)
        self.step_target = min(self.current+1.0,self.duration) if step else None
        self.service(Resume,'resume',Resume.Request())
        self.playing = True
        self.state.emit('前进约 1 秒…' if step else '正在播放',True)

    def clock(self,msg):
        if self.path:
            self.current = max(0.0,min(self.duration,(msg.clock.sec*10**9+msg.clock.nanosec-self.start_ns)/1e9))
            self.position.emit(self.current)

    def run(self):
        import rclpy
        from rosgraph_msgs.msg import Clock
        from rclpy.signals import SignalHandlerOptions
        from rclpy.qos import qos_profile_sensor_data
        self.rclpy=rclpy
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node=rclpy.create_node('bag_viewer_control')
        self.node.create_subscription(Clock,'/clock',self.clock,qos_profile_sensor_data)
        try:
            while True:
                try:
                    command,value=self.commands.get_nowait()
                except queue.Empty:
                    command=None
                try:
                    if command=='quit':break
                    if command=='open':self.open(value)
                    elif command=='play':self.resume()
                    elif command=='pause':self.pause()
                    elif command=='step':self.resume(step=True)
                    elif command=='rewind' and self.path:self.open(self.path)
                    rclpy.spin_once(self.node,timeout_sec=0.02)
                    if self.playing and self.step_target is not None and self.current>=self.step_target:
                        self.pause()
                    player=self.children.get('player')
                    if self.playing and player and player.poll() is not None:
                        self.playing=False
                        self.step_target=None
                        self.state.emit('播放完成 · 点击播放可重播' if player.returncode==0 else '播放器退出，请查看日志',True)
                except Exception as e:
                    self.playing=False
                    self.step_target=None
                    # Stop owned processes on control failure; never leave playback running silently.
                    self.stop()
                    self.state.emit('操作失败，请重新打开 bag',False)
                    self.error.emit(str(e)+'\n日志：'+str(self.run_dir))
        finally:
            self.stop()
            self.node.destroy_node()
            rclpy.shutdown()


class Window(QtWidgets.QWidget):
    def __init__(self,domain,initial):
        super().__init__()
        self.setWindowTitle('导航 Bag 回放控制')
        self.resize(760,260)
        self.move(80,80)
        self.duration=0
        self.engine=Engine(domain)
        layout=QtWidgets.QVBoxLayout(self)
        self.path=QtWidgets.QLabel('选择场景文件夹、bag 文件夹或 metadata.yaml')
        self.path.setWordWrap(True)
        layout.addWidget(self.path)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        folder=QtWidgets.QPushButton('打开 Bag 文件夹');folder.clicked.connect(self.choose_folder);row.addWidget(folder)
        file=QtWidgets.QPushButton('打开 metadata.yaml');file.clicked.connect(self.choose_file);row.addWidget(file)
        controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        self.buttons=[]
        for text,command in [('▶ 播放','play'),('⏸ 暂停','pause'),('回到 0 秒','rewind'),('前进约 1 秒','step')]:
            b=QtWidgets.QPushButton(text);b.setEnabled(False)
            b.clicked.connect(lambda checked=False,c=command:self.engine.send(c))
            controls.addWidget(b);self.buttons.append(b)
        self.timer=QtWidgets.QLabel('0.00 / 0.00 秒');layout.addWidget(self.timer)
        self.status=QtWidgets.QLabel('等待打开');layout.addWidget(self.status)
        layout.addWidget(QtWidgets.QLabel(f'独立 ROS 域 {domain} · 仅回放显示数据，不发布控制指令\n关闭本窗口会停止本程序启动的 RViz 和回放。'))
        self.engine.loaded.connect(self.loaded)
        self.engine.position.connect(lambda t:self.timer.setText(f'{t:.2f} / {self.duration:.2f} 秒'))
        self.engine.state.connect(self.state)
        self.engine.error.connect(lambda text:QtWidgets.QMessageBox.warning(self,'回放错误',text))
        self.engine.finished.connect(self.close)
        self.engine.start()
        if initial:self.engine.send('open',initial)

    def loaded(self,path,duration):
        self.path.setText(path);self.duration=duration
        self.raise_()
        self.activateWindow()

    def state(self,text,enabled):
        self.status.setText(text)
        for b in self.buttons:b.setEnabled(enabled)

    def choose_folder(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'选择场景或 Bag 文件夹','/data/junbin.zhang/navigation_scenarios')
        if path:self.engine.send('open',path)

    def choose_file(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择 metadata.yaml','/data/junbin.zhang/navigation_scenarios','Bag metadata (metadata.yaml)')
        if path:self.engine.send('open',path)

    def closeEvent(self,event):
        if self.engine.isRunning():
            event.ignore();self.setEnabled(False);self.status.setText('正在关闭…');self.engine.send('quit')
        else:event.accept()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag',nargs='?')
    p.add_argument('--domain',type=int,default=147)
    a=p.parse_args()
    if not 0<=a.domain<=232 or a.domain==144:
        p.error('请选择 0–232 范围内的独立 ROS 域；禁止使用实机域 144')
    os.environ['ROS_DOMAIN_ID']=str(a.domain)
    os.environ['ROS_LOCALHOST_ONLY']='1'
    # Avoid another viewer sharing player services in this domain.
    import fcntl
    lock=open('/tmp/navigation_bag_viewer_'+str(os.getuid())+'_'+str(a.domain)+'.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('该域已有回放控制窗口')
    app=QtWidgets.QApplication([sys.argv[0]])
    window=Window(a.domain,a.bag);window.show()
    return app.exec_()

if __name__=='__main__':raise SystemExit(main())
