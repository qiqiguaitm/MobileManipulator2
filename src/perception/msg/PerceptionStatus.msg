# 感知状态消息
# 反馈当前配置和检测结果

# 当前配置
string prompt              # 当前检测提示词
float32 min_score          # 当前最小分数阈值
float32 iou_threshold      # 当前 IoU 阈值

# 检测结果统计
uint32 object_count        # 检测到的物体数量
string[] categories        # 检测到的类别列表
float32 last_detect_time   # 上次检测耗时 (秒)
