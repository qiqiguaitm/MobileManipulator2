from setuptools import setup
from glob import glob
setup(name='auto_explore',version='0.1.0',packages=['auto_explore'],
 data_files=[('share/ament_index/resource_index/packages',['resource/auto_explore']),
 ('share/auto_explore',['package.xml','README.md']),('share/auto_explore/launch',glob('launch/*.py')),
 ('share/auto_explore/config',glob('config/*'))],
 install_requires=['setuptools'],entry_points={'console_scripts':['map_odom_tf=auto_explore.map_odom_tf:main','rotation_debug=auto_explore.rotation_debug:main','exploration_mapper=auto_explore.exploration_mapper:main','cloud_filter=auto_explore.cloud_filter:main','explore=auto_explore.explorer:main','start=auto_explore.supervisor:main']})
