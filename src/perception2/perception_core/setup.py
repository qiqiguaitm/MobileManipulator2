from setuptools import setup, find_packages

package_name = 'perception_core'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'scipy',
        'requests',
        'Pillow',
        'pycocotools',
        'PyYAML',
    ],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='developer@example.com',
    description='Perception core algorithms library (ROS-independent)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 可以添加命令行工具
        ],
    },
    python_requires='>=3.8',
)
