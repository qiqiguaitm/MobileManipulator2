# GraspResult.msg - 抓取检测结果消息
# 用于实时模式 Topic 输出

std_msgs/Header header
bool success
string error_message

# 核心 3D 输出 (相机光学坐标系: Z向前, X右, Y下)
float64[3] point3d                 # [x, y, z] 米
float64 width3d                    # 夹爪宽度 米
float64 angle                      # 抓取角度 度
string category                    # 物体类别
float64 score                      # 置信度

# 2D 信息
float64[2] center_uv               # [u, v] 像素
float64 depth                      # 深度值 米

float64 detection_time_ms          # 检测耗时 ms
