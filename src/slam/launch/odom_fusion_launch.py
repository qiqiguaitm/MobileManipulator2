"""
Odom Fusion Launch - FastLIO 校准 + wheel_odom 融合
需在 tracer_base 和 fastlio 启动后运行
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    odom_fusion_node = Node(
        package='slam',
        executable='odom_fusion_sync.py',
        name='odom_fusion_sync',
        output='screen',
        parameters=[{
            'fastlio_topic': '/fastlio/odom',
            'wheel_topic': '/wheel_odom',
            'output': '/odom/fused',
            'twist_alpha': 0.3,
            'wz_alpha': 0.7,
            'wz_deadzone': 0.03,
            'yaw_calibration_alpha': 0.9,
        }]
    )
    return LaunchDescription([odom_fusion_node])
