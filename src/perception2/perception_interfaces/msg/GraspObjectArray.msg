# GraspObjectArray.msg - 抓取检测结果数组
# 由 perception_grasp_node 发布

std_msgs/Header header
string frame_id                    # 坐标系 (hand_camera_color_optical_frame)
string prompt                      # 检测提示词

GraspObject[] objects              # 检测到的物体列表
int32 chosen_index                 # 被选中物体的索引 (-1表示无)

float64 detection_time_ms          # 检测耗时 ms
