cmake_minimum_required(VERSION 3.8)
project(perception_interfaces)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/Object3D.msg"
  "msg/Object3DArray.msg"
  "msg/TrackedObject3D.msg"
  "msg/TrackedObject3DArray.msg"
  "msg/GraspObject.msg"
  "msg/GraspObjectArray.msg"
  "msg/GraspResult.msg"
  "msg/PerceptionConfig.msg"
  "msg/PerceptionStatus.msg"
  # Services
  "srv/DetectObjects.srv"
  "srv/GraspDetect.srv"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)

ament_package()
