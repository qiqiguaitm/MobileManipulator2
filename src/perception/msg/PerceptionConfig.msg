# 感知配置消息
# 用于动态配置 DINO-X 检测参数

# === 识别参数 (PerceptionPanel 管理) ===
string prompt              # 检测提示词 (多类别用 . 分隔)
float32 min_score          # 最小分数阈值 (0.0-1.0, 默认 0.25)
float32 iou_threshold      # IoU 阈值 (0.0-1.0, 默认 0.5)

# === 抓取过滤参数 (CleanerPanel 管理) ===
# update_grasp_filter=false 时忽略以下字段
bool update_grasp_filter
float32 grasp_z_min           # base_link 最低高度 (m), 默认 -0.3
float32 grasp_z_max           # base_link 最高高度 (m), 默认  0.5
float32 grasp_distance_max    # 最远距离 (m), 默认 6.0
float32 grasp_physical_min_size  # 物体尺寸下限 (m), 默认 0.02
float32 grasp_physical_max_size  # 物体尺寸上限 (m), 默认 0.25

# === 检测控制 (CleanerManager 暂停/恢复Timer检测) ===
int8 detection_control        # 0=不变(默认), 1=暂停Timer检测, 2=恢复Timer检测

# === Tracker 重置 (清空目标池时同步清空跟踪器) ===
bool reset_tracker            # true=重置ByteTracker3D, 清除所有track
