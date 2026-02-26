cmake_minimum_required(VERSION 3.0.2)
project(perception)

# C++14 for RViz panel
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt5 for RViz panel
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  message_generation
  rviz
)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# Python setup
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  Object3D.msg
  Object3DArray.msg
  TrackedObject3D.msg
  TrackedObject3DArray.msg
  GraspResult.msg
  GraspObject.msg
  GraspObjectArray.msg
  PerceptionConfig.msg
  PerceptionStatus.msg
)

add_service_files(
  FILES
  DetectObjects.srv
  GraspDetect.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES percept_3d_panel percept_grasp_panel
  CATKIN_DEPENDS
    rospy
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    message_runtime
    rviz
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
)

# Qt5 MOC for headers with Q_OBJECT
qt5_wrap_cpp(MOC_FILES_3D
  include/perception/percept_3d_panel.h
)

qt5_wrap_cpp(MOC_FILES_GRASP
  include/perception/percept_grasp_panel.h
)

# RViz Panel libraries
add_library(percept_3d_panel
  src/percept_3d_panel.cpp
  ${MOC_FILES_3D}
)

add_dependencies(percept_3d_panel
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(percept_3d_panel
  ${catkin_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
)

add_library(percept_grasp_panel
  src/percept_grasp_panel.cpp
  ${MOC_FILES_GRASP}
)

add_dependencies(percept_grasp_panel
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(percept_grasp_panel
  ${catkin_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
  src/scene_perception_3d_node.py
  src/scene_perception_core.py
  src/synced_sensor_subscriber.py
  src/coordinate_transformer.py
  src/percept.py
  src/ros_lidar.py
  src/perception_visualizer.py
  src/perception_grasp_node.py
  src/perception_grasp_rviz_node.py
  src/perception_rviz_node.py
  src/object_tracker_node.py
  src/perception_viz_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS percept_3d_panel percept_grasp_panel
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(FILES plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
