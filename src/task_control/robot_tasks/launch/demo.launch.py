"""Isolated protocol demo. Does not start real perception or any hardware."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    if os.environ.get("ROS_DOMAIN_ID") != "166":
        raise RuntimeError(
            "Demo requires ROS_DOMAIN_ID=166; never run mocks on robot domain"
        )
    return LaunchDescription(
        [
            DeclareLaunchArgument("navigation_delay", default_value="0.2"),
            Node(
                package="robot_tasks",
                executable="mock_backends",
                output="screen",
                parameters=[
                    {
                        "navigation_delay": ParameterValue(
                            LaunchConfiguration("navigation_delay"), value_type=float
                        )
                    }
                ],
            ),
            Node(
                package="robot_tasks",
                executable="gc",
                parameters=[{"allow_execution": True}],
                output="screen",
            ),
            Node(
                package="robot_tasks",
                executable="nc",
                parameters=[
                    {"allow_execution": True, "allow_mock_front_execution": True}
                ],
                output="screen",
            ),
        ]
    )
