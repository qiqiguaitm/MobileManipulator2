cmake_minimum_required( VERSION 3.5 )
project(CamToLidarTool)

# Workaround for VTK/jsoncpp cmake target issue
find_package(jsoncpp REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(/opt/homebrew/include/eigen3)

set( CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#ceres
find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS} )

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

find_package(glog REQUIRED)

# include_directories(${EIGEN_ROOT})
add_library(CalibrationCore SHARED src/draw_ui.cpp src/io_yaml.cpp src/projection.cpp)
target_link_libraries(CalibrationCore ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp::yaml-cpp glog::glog)

add_executable(camToLidar examples/camera_to_lidar.cpp)
target_link_libraries(camToLidar CalibrationCore ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES} yaml-cpp::yaml-cpp)
