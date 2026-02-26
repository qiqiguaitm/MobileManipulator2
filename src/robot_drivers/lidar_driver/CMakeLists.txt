cmake_minimum_required(VERSION 3.8)
project(lidar_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# 安装 Python 脚本
install(PROGRAMS
  scripts/lidar_test.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
