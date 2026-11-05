# XFeas on real robots (ROS 2 Humble)

Runs the XFeas decision layer (`g5sm`: skill rules decide **when**, the feasibility model picks **which** command) on a
real **Unitree G1 + GEAR-SONIC** or **Boston Dynamics Spot + ReLIC**. One generic node; everything robot-specific is in
`xfeas_ros/config/<robot>.yaml`.

> **Status: tested offline only.** `test/test_core_offline.py` passes for both robots (the observation is accepted by the
> unchanged model server, the camera convention puts the floor at z = 0, a command comes back). It has **not** run on a
> robot or under ROS yet. Every config value marked `CHECK ON THE ROBOT` must be checked first. Start with the robot on a
> tether and the dead-man switch in hand.

```
ros2/
├── xfeas_msgs/                 the ControllerCommand message
└── xfeas_ros/
    ├── xfeas_ros/core.py       sensor samples -> observation -> model server -> skill rules -> command (no ROS; testable)
    ├── xfeas_ros/node.py       the ROS 2 wrapper: subscribers, TF, 50 Hz loop, publishers, dead-man switch
    ├── config/                 g1_sonic.yaml, spot_relic.yaml
    ├── launch/                 g1_sonic.launch.py, spot_relic.launch.py
    └── test/test_core_offline.py
```

## How it works

```
 /joint_states, /odom ──────────┐
 head RGB-D + camera_info + TF ─┼─► xfeas_node ──► model server (PyTorch, separate process) ──► skill rules
 /xfeas/goal, /xfeas/enable ────┘        │                                                        │
                                         └──────────── /xfeas/command  (xfeas_msgs/ControllerCommand) ◄┘
                                                       /xfeas/cmd_vel  (geometry_msgs/Twist)
                                                       /xfeas/status   (JSON per decision)
                                                              │
                                              your controller bridge ──► GEAR-SONIC / ReLIC ──► robot
```

The node rebuilds, from the robot's sensors, the same observation the simulator gives the model:
* **Camera:** the real image is re-projected into the simulator's square head-camera model (`CameraResampler`).
* **Body state:** a 2 s / 50 Hz proprio history and a 10 Hz command / achieved history.
* **Memory:** a 4-frame memory with the same perception features (`feas` v0.4).

Every 0.25 s it asks the model server, applies the skill rules and publishes the command. Steering (towards the goal,
straight line from where the goal was received) is the same stock path follower as in simulation.

## Build

```bash
# 1. the XFeas repo and its PyTorch environment (main README: Setup), with data/ in place (specs and self-test probes)
export XFEAS_ROOT=/path/to/XFeas XFEAS_PY=/path/to/torch/env/bin/python

# 2. the two ROS packages
mkdir -p ~/xfeas_ws/src && ln -s $XFEAS_ROOT/ros2/xfeas_msgs $XFEAS_ROOT/ros2/xfeas_ros ~/xfeas_ws/src/
cd ~/xfeas_ws && source /opt/ros/humble/setup.bash
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install && source install/setup.bash

# 3. offline test (no robot needed)
python $XFEAS_ROOT/ros2/xfeas_ros/test/test_core_offline.py
```

## Run

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true       # head camera, depth aligned to colour
ros2 launch xfeas_ros g1_sonic.launch.py                                   # or spot_relic.launch.py [config:=my.yaml]

ros2 topic pub --once /xfeas/goal geometry_msgs/PoseStamped "{header: {frame_id: odom}, pose: {position: {x: 5.0, y: 0.0}}}"
ros2 topic pub -r 10 /xfeas/enable std_msgs/Bool "{data: true}"           # dead-man switch: stop publishing -> twist 0
ros2 topic echo /xfeas/status                                              # what XFeas chose and why
```

The node **stops the robot** (zero twist, the posture command stays) whenever the enable switch is off, any sensor is
older than 0.5 s, or the goal is reached (within 0.5 m). Forward speed is capped by `max_vx` (0.5 m/s by default).

## What you must provide on each robot

**1. A controller bridge** that subscribes to `/xfeas/command` and drives the controller. Message fields:

| Field | G1 + GEAR-SONIC | Spot + ReLIC |
|---|---|---|
| `lin_vel_x`, `lin_vel_y`, `yaw_rate` | planner velocity command | ReLIC base velocity command |
| `mode`, `speed` | planner mode (`walk`, `slowWalk`, `crouch`, `handCrawling`, `elbowCrawling`, `careful`, `stealth`) and speed | unused (`""`, NaN) |
| `body_height`, `torso_pitch`, `torso_roll` | body height of the modes that use it | ReLIC base pose command (height 0.25–0.65 m, pitch / roll ±0.35 rad) |
| `face` | sidestep: walk with the body turned by this angle (rad) | 0 |
| `enabled`, `at_goal` | false / true → hold still | same |

Unset values are NaN (numbers) or `""` (mode): keep the current setting. The bridge should clip to the controller's own
ranges (`data/gen_v4/specs/<controller>.json`).
* **G1:** GEAR-SONIC runs through the `gear_sonic_deploy` stack of GR00T-WholeBodyControl.
* **Spot:** ReLIC needs Spot's low-level joint-control API. Spot's own Boston Dynamics controller (Spot SDK) is a
  *different* controller for XFeas, so it would need its own spec and a self-test on the robot.

**2. The head camera where the model was trained to see from.**

| Robot | Camera pose (simulator) | On the real robot |
|---|---|---|
| G1 | D435i in the head: `torso_link` + (0.01, 0.01, 0.44) m, ~27° down (`xfeas/sim/camera_mount.py`, `g1_head_php`) | the G1's own head D435i, which is at this place |
| Spot | body-fixed at the nose: `body` + (0.415, 0, −0.02) m, pitched 0.45 rad down (`spot_body`) | **mount a D435i there**; real Spot has no RGB-D camera at that pose |

TF must connect the camera's optical frame to the robot (`frames.camera_optical`) and the robot to `frames.world`.

**3. Calibration of the body top.** The skill rules compare the body's highest point with the standing height measured by
the self-test (G1 + SONIC 1.32 m, Spot + ReLIC 0.97 m). With the robot standing, check the `top` the node logs and set
`body_top_offset` so it matches.

## Known gaps (read before the first run)

* **Field of view.** The model was trained on a square ~82° image. A D435i colour image (~69° × 42°) fills only **~27%**
  of it; the rest is blank. Expect worse perception than in simulation until the model is fine-tuned on real or
  matching-FOV images. The node logs the coverage at start.
* **Sim-to-real.** The model has only seen simulated images (night with a headlight is the main training condition).
* **Proprio details.** Joint position/velocity limits (`joint_limits`) and efforts feed the joint features; without them
  those features are zero, which differs from training. Fill `joint_limits` from the robot's URDF.
* **The self-test** (`controller_selftest.py`) only runs in Isaac Sim. The node uses the simulator self-test of SONIC /
  ReLIC (`data/probes_v4/<controller>/`). A real-robot self-test would measure the real controller instead.
* **Route.** The route is the straight line from where the goal was received; there is no map or global planner.
