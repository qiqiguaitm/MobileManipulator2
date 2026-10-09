from setuptools import setup
setup(name='board_approach', version='0.1.0', packages=['board_approach'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/board_approach']),
                  ('share/board_approach', ['package.xml', 'README.md', 'TESTING.md', 'VALIDATION.md']),
                  ('share/board_approach/config', ['config/no_recovery.xml'])],
      install_requires=['setuptools'], zip_safe=True,
      description='Guarded visual target and LiDAR surface approach for robot02',
      license='Apache-2.0', maintainer='MobileManipulator2 maintainers',
      maintainer_email='maintainers@example.com',
      entry_points={'console_scripts': ['approach_board = board_approach.runner:main']})
