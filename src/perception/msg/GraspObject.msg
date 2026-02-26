# GraspObject.msg - 单个可抓取物体
# 包含检测、分割、抓取的完整信息

string object_id                   # 物体ID (category_index, 如 "bottle_1")
string category                    # 类别名称
float64 detection_score            # 检测置信度 (DinoX)

# 2D 检测
float64[4] bbox                    # 2D bbox [x1, y1, x2, y2] 像素
string mask_rle                    # Mask RLE 编码 (COCO格式 counts 字符串)
uint32[2] mask_size                # Mask 尺寸 [height, width]

# 抓取信息 (GraspAnything)
float64 grasp_score                # 抓取置信度
float64[2] grasp_center            # 抓取中心 [u, v] 像素
float64 grasp_angle                # 抓取角度 弧度
float64 grasp_width                # 夹爪宽度 像素
float64 grasp_width3d              # 夹爪宽度 米

# 3D 位置 (相机光学坐标系)
geometry_msgs/Point position       # 3D 位置 (x, y, z) 米
float64 depth                      # 深度值 米
