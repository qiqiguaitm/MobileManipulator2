"""Hardware-free explore_lite/Nav2 integration, never starts a chassis driver."""
import importlib.util
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction,ExecuteProcess,TimerAction
from launch_ros.actions import Node

def setup(c):
    c.launch_configurations.update(output='/tmp/explore_upstream_smoke',execute='false',bounds='[-5.0,-5.0,5.0,5.0]')
    p=str(Path(get_package_share_directory('auto_explore'))/'launch/explore.launch.py')
    spec=importlib.util.spec_from_file_location('stack',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.setup(c)[8:]+[
        ExecuteProcess(cmd=['/usr/bin/python3',str(Path(__file__).with_name('smoke_fixture.py')),'--relay-map']),
        TimerAction(period=8.,actions=[Node(package='explore_lite',executable='explore',parameters=['/tmp/explore_upstream_smoke/explore_lite.yaml'],output='screen')])]
def generate_launch_description():return LaunchDescription([OpaqueFunction(function=setup)])
