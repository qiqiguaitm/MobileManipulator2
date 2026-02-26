cmake_minimum_required(VERSION 3.8)
project(slam)

find_package(ament_cmake REQUIRED)

# 安装launch文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# 安装rviz配置
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# 安装脚本文件
install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)

ament_package()
