cmake_minimum_required(VERSION 3.8)
project(piper_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  meshes
  rviz
  urdf
  launch
  config
  mujoco_model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
