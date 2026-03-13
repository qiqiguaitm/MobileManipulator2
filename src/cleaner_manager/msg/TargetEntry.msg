# Single target pool entry for CleanerPanel display
string category
float64 pos_x
float64 pos_y
float64 score
uint8 status            # 0=ACTIVE 1=PICKED 2=FAILED
uint32 observations
uint32 attempts
float32 last_seen_age   # seconds since last seen
bool viable             # passes TTL + min_observations + attempts filters
