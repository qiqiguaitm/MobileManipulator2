# 感知配置消息
# 用于动态配置 DINO-X 检测参数

string prompt              # 检测提示词 (多类别用 . 分隔)
float32 min_score          # 最小分数阈值 (0.0-1.0, 默认 0.25)
float32 iou_threshold      # IoU 阈值 (0.0-1.0, 默认 0.5)
