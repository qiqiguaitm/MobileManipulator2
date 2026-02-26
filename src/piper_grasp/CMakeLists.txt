cmake_minimum_required(VERSION 3.8)
project(piper_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

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
  DESTINATION lib/${PROJECT_NAME}
)

# Install config and launch
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
