cmake_minimum_required(VERSION 3.10)
project(hdl_localization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 架构特定优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
  message(STATUS "Building hdl_localization for ARM architecture")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(hdl_global_localization REQUIRED)

# 自定义 msg
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ScanMatchingStatus.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

# pose_estimator 库
add_library(pose_estimator SHARED
  src/hdl_localization/pose_estimator.cpp
)
target_include_directories(pose_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(pose_estimator
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
)
target_link_libraries(pose_estimator
  ndt_omp::ndt_omp
  OpenMP::OpenMP_CXX
  ${PCL_LIBRARIES}
)

# hdl_localization_node 可执行文件
add_executable(hdl_localization_node
  src/hdl_localization_node.cpp
)
target_include_directories(hdl_localization_node PRIVATE
  include
)
ament_target_dependencies(hdl_localization_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_srvs
  std_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  tf2_sensor_msgs
  hdl_global_localization
)
rosidl_target_interfaces(hdl_localization_node ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(hdl_localization_node
  pose_estimator
  ndt_omp::ndt_omp
  OpenMP::OpenMP_CXX
  ${PCL_LIBRARIES}
)

# Install
install(TARGETS
  pose_estimator
  hdl_localization_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/hdl_localization
)
install(DIRECTORY
  include/
  launch/
  rviz/
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
