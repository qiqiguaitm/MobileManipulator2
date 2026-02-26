# PiperStatus.msg - /piper/status
# Comprehensive arm status for RViz Panel

# Connection state
bool connected
bool enabled
string error_state           # NORMAL/WARNING/CRITICAL
string error_description

# Position (flange frame, mm/degrees)
float32[] position           # [x, y, z, roll, pitch, yaw]

# Gripper center (mm/degrees)
float32[] gripper_center     # [x, y, z, roll, pitch, yaw]

# Joint angles (degrees)
float32[] joints             # [j1, j2, j3, j4, j5, j6]

# Gripper
float32 gripper_position     # mm (0=closed, 70=open)
