cmake_minimum_required(VERSION 3.8)
project(sc_pgo)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

# 第三方库
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

# 头文件目录
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# 编译主节点
add_executable(sc_pgo_node
  src/sc_pgo_node.cpp
  src/Scancontext.cpp
)

# OpenMP编译选项
target_compile_options(sc_pgo_node PRIVATE ${OpenMP_CXX_FLAGS})

# ROS2依赖
ament_target_dependencies(sc_pgo_node
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
)

# 链接库
target_link_libraries(sc_pgo_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  gtsam
  ${OpenMP_CXX_LIBRARIES}
)

# 安装
install(TARGETS sc_pgo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
