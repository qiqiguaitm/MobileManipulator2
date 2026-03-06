cmake_minimum_required(VERSION 3.8)
project(cleaner_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========== Dependencies ==========
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# ========== Messages ==========
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotManagerStatus.msg"
  DEPENDENCIES
    std_msgs
    builtin_interfaces
)

# ========== Python ==========
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_py_dir "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(FILES
  cleaner_manager/__init__.py
  cleaner_manager/target_pool.py
  cleaner_manager/pick_state_machine.py
  DESTINATION "${_py_dir}/cleaner_manager"
)

install(PROGRAMS
  src/robot_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Config / Launch ==========
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# ========== Export ==========
ament_export_dependencies(rosidl_default_runtime)

ament_package()
