import importlib.util
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction,ExecuteProcess
from launch.substitutions import LaunchConfiguration

def setup(c):
 c.launch_configurations.update(output='/tmp/explore_smoke',execute='false',bounds='[-5.0,-5.0,5.0,5.0]')
 p=str(Path(get_package_share_directory('auto_explore'))/'launch/explore.launch.py')
 s=importlib.util.spec_from_file_location('explore_stack',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 nodes=m.setup(c)[8:]
 return nodes+[ExecuteProcess(cmd=['/usr/bin/python3',str(Path(__file__).with_name('smoke_fixture.py'))]),ExecuteProcess(cmd=['/usr/bin/python3','-m','auto_explore.explorer','--output','/tmp/explore_smoke/result','--bounds','-5','-5','5','5'])]
def generate_launch_description():return LaunchDescription([OpaqueFunction(function=setup)])
