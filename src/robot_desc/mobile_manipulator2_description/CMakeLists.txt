cmake_minimum_required(VERSION 3.8)
project(mobile_manipulator2_description)

find_package(ament_cmake REQUIRED)

# 安装资源目录
install(DIRECTORY
  config
  launch
  meshes
  urdf
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
