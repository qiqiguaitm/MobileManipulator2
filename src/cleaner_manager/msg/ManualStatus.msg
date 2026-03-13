# Manual control mode status (ManualControlNode -> ManualPanel, 10Hz)

uint8 STATE_IDLE = 0
uint8 STATE_DETECTING = 1
uint8 STATE_WAITING_SELECT = 2
uint8 STATE_WAITING_CONFIRM = 3
uint8 STATE_NAVIGATING = 4
uint8 STATE_APPROACHING = 5
uint8 STATE_OBSERVING = 6
uint8 STATE_PICKING = 7
uint8 STATE_PLACING = 8
uint8 STATE_ERROR = 9

std_msgs/Header header
uint8 state
string state_name
string message

# Prompt
string current_prompt

# Candidates (valid in WAITING_SELECT/WAITING_CONFIRM)
cleaner_manager/Candidate[] candidates
int32 selected_index

# Current target (valid from WAITING_CONFIRM onward)
string target_category
string target_object_id
float64 target_distance
geometry_msgs/Point target_position
geometry_msgs/Point robot_position

# Grasp progress (valid in PICKING/PLACING)
string grasp_step
float32 grasp_progress

# Error
string error_message
