"""GC/NC/perception only; the session entry point owns hardware processes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def controller_nodes(context):
    parameters = [LaunchConfiguration("config")]
    profile = LaunchConfiguration("profile").perform(context)
    if profile:
        parameters.append(profile)
    execution = ParameterValue(LaunchConfiguration("allow_execution"), value_type=bool)
    skip_grasp = ParameterValue(LaunchConfiguration("skip_grasp"), value_type=bool)
    perception_overrides = {"workspace": LaunchConfiguration("workspace")}
    extrinsics = LaunchConfiguration("camera_extrinsics_file").perform(context)
    if extrinsics:
        perception_overrides["camera_extrinsics_file"] = extrinsics
    return [
        Node(
            package="robot_tasks",
            executable="gc",
            output="screen",
            parameters=[
                *parameters,
                {"allow_execution": execution, "skip_grasp": skip_grasp},
            ],
        ),
        Node(
            package="robot_tasks",
            executable="nc",
            output="screen",
            parameters=[*parameters, {"allow_execution": execution}],
        ),
        Node(
            package="robot_tasks",
            executable="scene_perception",
            output="screen",
            parameters=[*parameters, perception_overrides],
            condition=IfCondition(LaunchConfiguration("perception")),
        ),
    ]


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory("robot_tasks"), "config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("allow_execution", default_value="false"),
            DeclareLaunchArgument("skip_grasp", default_value="false"),
            DeclareLaunchArgument("workspace", default_value=os.getcwd()),
            DeclareLaunchArgument("camera_extrinsics_file", default_value=""),
            DeclareLaunchArgument("perception", default_value="true"),
            DeclareLaunchArgument(
                "config", default_value=os.path.join(config_dir, "controllers.yaml")
            ),
            DeclareLaunchArgument(
                "profile",
                default_value=os.path.join(config_dir, "robot02.yaml"),
                description="Site overrides; empty disables the robot02 profile",
            ),
            OpaqueFunction(function=controller_nodes),
        ]
    )
