cmake_minimum_required(VERSION 3.8)
project(cleaner_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== Dependencies ==========
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)

# ========== Messages / Services ==========
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TargetEntry.msg"
  "msg/CleanerManagerStatus.msg"
  "msg/Candidate.msg"
  "msg/ManualStatus.msg"
  "srv/ManualCommand.srv"
  DEPENDENCIES
    std_msgs
    builtin_interfaces
    geometry_msgs
)

# ========== Python ==========
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_py_dir "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(FILES
  cleaner_manager/__init__.py
  cleaner_manager/target_pool.py
  cleaner_manager/target_filter.py
  cleaner_manager/pick_state_machine.py
  DESTINATION "${_py_dir}/cleaner_manager"
)

install(PROGRAMS
  src/cleaner_manager_node.py
  src/manual_control_node.py
  src/object_selector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== C++ RViz2 Panels ==========
find_package(Qt5 COMPONENTS Core Widgets QUIET)
find_package(rviz_common QUIET)
find_package(rclcpp QUIET)
find_package(pluginlib QUIET)
find_package(tracer_msgs QUIET)
find_package(piper_msgs QUIET)
find_package(perception QUIET)
find_package(std_srvs QUIET)

if(Qt5_FOUND AND rviz_common_FOUND AND rclcpp_FOUND AND pluginlib_FOUND
   AND tracer_msgs_FOUND AND piper_msgs_FOUND AND perception_FOUND)

  set(CMAKE_AUTOMOC ON)

  add_library(cleaner_panels SHARED
    src/cleaner_panel.cpp
    src/perception_panel.cpp
    include/cleaner_manager/cleaner_panel.h
    include/cleaner_manager/perception_panel.h
  )

  target_include_directories(cleaner_panels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  # Link generated message types
  rosidl_get_typesupport_target(cpp_ts ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(cleaner_panels
    Qt5::Core
    Qt5::Widgets
    "${cpp_ts}"
  )

  ament_target_dependencies(cleaner_panels
    rclcpp
    pluginlib
    rviz_common
    tracer_msgs
    piper_msgs
    perception
    std_srvs
  )

  pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

  install(TARGETS cleaner_panels
    LIBRARY DESTINATION lib
  )

  install(DIRECTORY include/
    DESTINATION include
  )

  message(STATUS "[cleaner_manager] C++ panels enabled")
else()
  message(WARNING "[cleaner_manager] C++ panels SKIPPED (missing: "
    "Qt5=${Qt5_FOUND} rviz_common=${rviz_common_FOUND} rclcpp=${rclcpp_FOUND} "
    "pluginlib=${pluginlib_FOUND} tracer_msgs=${tracer_msgs_FOUND} "
    "piper_msgs=${piper_msgs_FOUND} perception=${perception_FOUND})")
endif()

# ========== Config / Launch ==========
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# ========== Export ==========
ament_export_dependencies(rosidl_default_runtime)

ament_package()
