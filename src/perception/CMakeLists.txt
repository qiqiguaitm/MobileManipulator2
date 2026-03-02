cmake_minimum_required(VERSION 3.8)
project(perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========== Dependencies ==========
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ========== Messages / Services ==========
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Object3D.msg"
  "msg/Object3DArray.msg"
  "msg/TrackedObject3D.msg"
  "msg/TrackedObject3DArray.msg"
  "msg/GraspObject.msg"
  "msg/GraspObjectArray.msg"
  "msg/GraspResult.msg"
  "msg/PerceptionConfig.msg"
  "msg/PerceptionStatus.msg"
  "srv/DetectObjects.srv"
  "srv/GraspDetect.srv"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
)

# ========== Python ==========
# Install Python modules alongside rosidl-generated code (local/ prefix is on PYTHONPATH)
set(_py_dir "local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/dist-packages")

install(FILES
  src/percept.py
  src/coordinate_transformer.py
  src/scene_perception_core.py
  src/dual_camera_matcher.py
  src/intelligent_fusion.py
  src/camera_confidence_model.py
  src/byte_tracker_3d.py
  src/synced_sensor_subscriber.py
  src/utils.py
  src/percept_benchmark.py
  DESTINATION "${_py_dir}/perception"
)

# Install Python nodes as executables (strip .py suffix for ROS2 convention)
set(_node_scripts
  scene_perception_3d_node
  object_tracker_node
  perception_viz_node
  perception_grasp_node
  perception_grasp_rviz_node
  multi_camera_perception_node
  perception_rviz_node
  multi_camera_rviz_node
  dual_camera_baselink_validation
  percept_benchmark
  stereo_chessboard_capture
)
foreach(_script ${_node_scripts})
  install(PROGRAMS "src/${_script}.py"
    DESTINATION lib/${PROJECT_NAME}
    RENAME ${_script}
  )
endforeach()

# ========== Config / Launch / RViz ==========
install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

# ========== Export ==========
ament_export_dependencies(rosidl_default_runtime)

ament_package()
