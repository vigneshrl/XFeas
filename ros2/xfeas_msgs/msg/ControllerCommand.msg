# One XFeas decision for the robot's own controller. Unset values are NaN (posture) or "" (mode).
std_msgs/Header header
string controller_id      # g1s (G1 + GEAR-SONIC), spotp (Spot + ReLIC), ...
float32 lin_vel_x         # m/s, body frame
float32 lin_vel_y         # m/s, body frame
float32 yaw_rate          # rad/s
float32 body_height       # m (role pelvis_height; Spot: base height above the floor)
float32 torso_pitch       # rad
float32 torso_roll        # rad
float32 speed             # GEAR-SONIC planner speed (-1 = mode default)
string mode               # GEAR-SONIC planner mode (walk, slowWalk, crouch, handCrawling, ...); "" = no mode command
float32 face              # facing offset (rad) of a sidestep: walk with the body turned by this angle
string xfeas_class        # the chosen move class (stock, slow, low1, latL, ...)
string skill              # walk / low / side (skill rules)
float32 score             # model score U of the chosen command
bool enabled              # false: the dead-man switch is off and the twist is zero
bool at_goal
