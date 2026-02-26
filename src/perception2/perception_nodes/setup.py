from setuptools import setup, find_packages

package_name = 'perception_nodes'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch', []),
        # Config files
        ('share/' + package_name + '/config', []),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='developer@example.com',
    description='ROS2 perception nodes for MobileManipulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_perception_3d_node = perception_nodes.scene_perception_3d_node:main',
            'object_tracker_node = perception_nodes.object_tracker_node:main',
            'perception_viz_node = perception_nodes.perception_viz_node:main',
            'perception_grasp_node = perception_nodes.perception_grasp_node:main',
            'perception_grasp_rviz_node = perception_nodes.perception_grasp_rviz_node:main',
            'multi_sensor_perception_node = perception_nodes.multi_sensor_perception_node:main',
            'multi_camera_perception_node = perception_nodes.multi_camera_perception_node:main',
            'perception_rviz_node = perception_nodes.perception_rviz_node:main',
            'multi_camera_rviz_node = perception_nodes.multi_camera_rviz_node:main',
            'dual_camera_baselink_validation = perception_nodes.dual_camera_baselink_validation:main',
        ],
    },
    python_requires='>=3.8',
)
