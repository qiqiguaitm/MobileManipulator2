cmake_minimum_required(VERSION 3.8)
project(piper_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/PosCmd.msg"
  "msg/PiperStatusMsg.msg"
  "msg/PiperStatus.msg"
  # Services (basic)
  "srv/Enable.srv"
  "srv/GoZero.srv"
  "srv/Gripper.srv"
  # Services (enhanced)
  "srv/EnableEnhanced.srv"
  "srv/GetPosition.srv"
  "srv/GetStatus.srv"
  "srv/GoReady.srv"
  "srv/InWorkingArea.srv"
  "srv/Observe.srv"
  "srv/SetGripper.srv"
  "srv/SetPosition.srv"
  # Note: GraspDetect is defined in the perception package, not here
  # Actions
  "action/PiperPick.action"
  "action/PiperPlace.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_package()
