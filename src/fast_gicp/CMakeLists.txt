cmake_minimum_required(VERSION 3.18)
project(fast_gicp LANGUAGES CXX)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)
option(BUILD_apps "Build application programs" ON)
option(BUILD_test "Build test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA 语言支持
if(BUILD_VGICP_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # 检测 GPU 架构
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Jetson AGX Orin = sm_87
    set(CMAKE_CUDA_ARCHITECTURES 87)
    message(STATUS "Jetson detected, using CUDA architecture sm_87")
  else()
    # 默认使用常见桌面 GPU 架构
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
  endif()
endif()

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

if(DEFINED ENV{ROS_VERSION})
  set(ROS_VERSION $ENV{ROS_VERSION})
endif()

if(NOT BUILD_PYTHON_BINDINGS)
  if(${ROS_VERSION})
    if(${ROS_VERSION} EQUAL 1)
      find_package(catkin)
    elseif (${ROS_VERSION} EQUAL 2)
      find_package(ament_cmake)
    endif()
  endif()
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_VGICP_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
endif()

###################################
## catkin specific configuration ##
###################################
if(catkin_FOUND)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES fast_gicp
  )
endif()

###########
## Build ##
###########

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

### APPS ###
if(BUILD_apps)
  add_executable(gicp_align src/align.cpp)
  add_dependencies(gicp_align fast_gicp)
  target_link_libraries(gicp_align
    ${PCL_LIBRARIES}
    fast_gicp
  )

  add_executable(gicp_kitti src/kitti.cpp)
  add_dependencies(gicp_kitti fast_gicp)
  target_link_libraries(gicp_kitti
    ${PCL_LIBRARIES}
    fast_gicp
  )
endif()

### Python bindings ###
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG)
  pybind11_add_module(pygicp
    src/python/main.cpp
  )
  target_include_directories(pygicp PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(pygicp PRIVATE
    fast_gicp
  )
  set_target_properties(pygicp PROPERTIES
    INSTALL_RPATH "$ORIGIN/"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

### CUDA ###
if(BUILD_VGICP_CUDA)
  add_definitions(-DUSE_VGICP_CUDA)

  # 使用现代 CMake CUDA 支持
  add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
  )

  # CUDA 编译选项
  target_compile_options(fast_vgicp_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored>
  )

  # CUDA 12.x 兼容性定义
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
    message(STATUS "CUDA 12.x detected, applying compatibility fixes")
    target_compile_definitions(fast_vgicp_cuda PRIVATE
      THRUST_IGNORE_CUB_VERSION_CHECK
      _LIBCUDACXX_DISABLE_EXEC_CHECK
    )
    # 禁用 libcu++ 与 std 的互操作
    target_compile_options(fast_vgicp_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-D_LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES>
    )
  endif()

  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    thirdparty/Eigen
    thirdparty/nvbio
    ${EIGEN3_INCLUDE_DIR}
  )

  # 链接 CUDA 库
  target_link_libraries(fast_vgicp_cuda
    CUDA::cudart
    CUDA::cublas
  )

  # add vgicp_cuda to libfast_gicp
  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
    src/fast_gicp/ndt/ndt_cuda.cpp
  )
  target_link_libraries(fast_gicp
    fast_vgicp_cuda
  )
  add_dependencies(fast_gicp fast_vgicp_cuda)

  if(catkin_FOUND)
    install(TARGETS fast_vgicp_cuda
      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
  elseif(ament_cmake_FOUND)
    install(TARGETS fast_vgicp_cuda
      LIBRARY DESTINATION lib)
  endif()
endif()

### TEST ###
if(BUILD_test)
  find_package(GTest REQUIRED)

  add_executable(gicp_test src/test/gicp_test.cpp)
  add_dependencies(gicp_test fast_gicp)
  target_link_libraries(gicp_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)
  gtest_add_tests(TARGET gicp_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data")
endif()

if(catkin_FOUND)
  ###################################
  ## catkin specific configuration ##
  ###################################
  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

  install(DIRECTORY include/
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp")
elseif (ament_cmake_FOUND)
  ##################################
  ## ament specific configuration ##
  ##################################
  ament_export_include_directories(include)
  ament_export_libraries(fast_gicp)
  ament_package()

  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib)

  install(
      DIRECTORY "include/"
      DESTINATION include
    )
endif()
