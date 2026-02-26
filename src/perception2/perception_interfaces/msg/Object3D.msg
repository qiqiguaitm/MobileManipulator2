# 3D 物体检测结果
# 坐标系由 target_frame 配置决定 (默认 base_link)

string object_id                   # 物体ID (category_index, 如 "bottle_1")
string category                    # 类别名称
float64 score                      # 检测置信度
float64[4] bbox                    # 2D bbox [x1, y1, x2, y2]
string source_camera               # 来源相机名称 (如 "top", "front")

# 相机测量 (target_frame 坐标系)
geometry_msgs/Point position       # 3D 位置
float64 distance                   # 到原点距离 (m)
float64 confidence                 # 测量置信度

# 相机光学坐标系下的位置 (用于调试)
geometry_msgs/Point position_optical  # optical frame 下的位置 (x右, y下, z前/深度)
float64 depth                         # 中心点深度 (m), 即 position_optical.z

# LiDAR 测量 (可选, distance_lidar=0 表示无效或禁用)
float64 distance_lidar             # LiDAR 测距 (m)
geometry_msgs/Point position_lidar # LiDAR 质心
float64 confidence_lidar           # LiDAR 置信度
