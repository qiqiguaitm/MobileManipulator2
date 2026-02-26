cmake_minimum_required(VERSION 3.8)
project(piper_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Qt MOC automatically
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(piper_msgs REQUIRED)

# Build RViz2 Panel (only if rviz_common is found)
if(rviz_common_FOUND)
  set(PANEL_SOURCES
    src/piper_grasp_panel.cpp
  )

  set(PANEL_HEADERS
    include/piper_grasp/piper_grasp_panel.h
  )

  add_library(piper_grasp_panel SHARED
    ${PANEL_SOURCES}
    ${PANEL_HEADERS}
  )

  target_include_directories(piper_grasp_panel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(piper_grasp_panel
    Qt5::Core
    Qt5::Widgets
  )

  ament_target_dependencies(piper_grasp_panel
    rclcpp
    rclcpp_action
    std_srvs
    pluginlib
    rviz_common
    rviz_default_plugins
    piper_msgs
  )

  # Export plugin
  pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

  # Install panel library
  install(TARGETS piper_grasp_panel
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  # Install headers
  install(DIRECTORY include/
    DESTINATION include
  )
endif()

# Install Python scripts
install(PROGRAMS
  scripts/piper_grasp_node.py
  scripts/piper_api_v2.py
  scripts/collision_checker.py
  scripts/workspace_validation.py
  scripts/grasp_blacklist.py
  scripts/ik_trajectory_planner.py
  scripts/safe_trajectory_planner.py
  scripts/movel_path_checker.py
  scripts/coordinate_transformer.py
  scripts/grasp_demo_node.py
  scripts/test_piper_api_v2.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config, launch, and docs
install(DIRECTORY
  config
  launch
  docs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
