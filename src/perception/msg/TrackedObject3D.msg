# 跟踪物体3D信息
# 由 ObjectTrackerNode 发布

Header header

# 身份
int32 track_id                     # 跟踪ID (唯一, 不变)
string category                    # 类别名称

# 2D信息 (像素坐标)
float64[4] bbox                    # [x1, y1, x2, y2]

# 3D信息 (目标坐标系, 默认 base_link)
geometry_msgs/Point position       # 3D位置
float64 distance                   # 到原点距离 (m)

# 置信度
float64 track_score                # 跟踪置信度
float64 position_confidence        # 3D位置置信度
