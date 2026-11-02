"""
PHP (Perceptive Humanoid Parkour, amazon-far/php_parkour @c7c9ce3, Holosoma @4a0bf04) released D435i student, reproduced
from the release inference stack with numpy + onnxruntime only (no torch). Used by the MuJoCo reference
(php_mujoco_ref.py) and the Isaac bridge server (php_server.py). Nothing here is trained or tuned.

Sources (all under baselines/php_parkour/thirdparty/holosoma/src):
  policy     holosoma_inference/policies/depth_distillation.py (DepthDistillationPolicy) + base.py / locomotion.py
  obs        config_values/observation.py wbt_distillation_g1:
             [actions 29 | base_ang_vel 3 | dof_pos - default 29 | dof_vel 29 | projected_gravity (TORSO frame) 3]
             | one-hot direction 15 | depth latent 32  = 140; scales 1; history 1; student input time_step = 0
  action     q* = default_dof_angles + 1.0 * a  (action_scale metadata = 1); a is clipped to +-100; the raw a feeds back
  gains      config_values/robot.py _DISTILL_MOTOR_KP/KD (= ONNX kp/kd metadata for this checkpoint)
  startup    stiff hold: _DISTILL_STIFF_POS with _DISTILL_STIFF_KP/KD (100-tick smoothstep blend), then ']' -> policy
  depth      holosoma/simulator/plugins/depth_shm_plugin.py + config_values/plugin.py depth-shm-d435i:
             106x60 metric z-depth -> nan/inf -> far; crop [2:, 4:-4] -> clip [0.3, 3.0] -> torch bicubic antialias
             resize to 58x87 (baked to matrices, resize_98x58_to_87x58.npz, max err 0.0 vs torch 2.7) ->
             min(., far) -> (< 0.15 -> far) -> (d - 0.3) / 2.7 - 0.5;  render 10 Hz, latency 0 frames
  camera     holosoma/config_values/wbt/g1/sensor.py d435i_front_depth_camera: torso_link, pos (0.01, 0.01, 0.44),
             quat wxyz (0.60290764, 0.37585401, -0.362958, -0.60290764) (camera -Z forward, +Y up), vfov 58.5953
  command    one-hot CMD_CODES (speed mode LOW = 0): stand 0, forward 1, left_45 2, left_90 3, right_45 4, right_90 5,
             back 11; HIGH: forward 6, left_45 7, left_90 8, right_45 9, right_90 10. Continuous (vx, vy, wz) ->
             class by DepthDistillationPolicy._apply_velocity (sector quantiser, dead zone 0.05), reproduced verbatim.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHP_ROOT = xfeas.paths.BASELINES + "/php_parkour"
STUDENT_DIR = os.environ.get("PHP_STUDENT_DIR", f"{PHP_ROOT}/assets_release/student")
HOLO = f"{PHP_ROOT}/thirdparty/holosoma/src/holosoma/holosoma"
MJCF = f"{HOLO}/data/robots/g1/g1_29dof_halfspherehand.xml"

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEFAULT = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ]
)
KP = np.array(
    [
        40.179,
        99.098,
        40.179,
        99.098,
        28.501,
        28.501,
        40.179,
        99.098,
        40.179,
        99.098,
        28.501,
        28.501,
        40.179,
        28.501,
        28.501,
        14.251,
        14.251,
        14.251,
        14.251,
        14.251,
        16.778,
        16.778,
        14.251,
        14.251,
        14.251,
        14.251,
        14.251,
        16.778,
        16.778,
    ]
)
KD = np.array(
    [
        2.558,
        6.309,
        2.558,
        6.309,
        1.814,
        1.814,
        2.558,
        6.309,
        2.558,
        6.309,
        1.814,
        1.814,
        2.558,
        1.814,
        1.814,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        1.068,
        1.068,
        0.907,
        0.907,
        0.907,
        0.907,
        0.907,
        1.068,
        1.068,
    ]
)
STIFF_POS = np.array(
    [
        -0.16277,
        -0.02224,
        0.00908,
        0.29148,
        -0.14721,
        0.02468,
        -0.16277,
        0.02224,
        -0.00908,
        0.29148,
        -0.14721,
        -0.02468,
        0.0,
        0.0,
        0.0063,
        0.25452,
        0.28698,
        -0.08782,
        0.80775,
        0.00437,
        0.00578,
        -0.00094,
        0.25452,
        -0.28698,
        0.08782,
        0.80775,
        -0.00437,
        0.00578,
        0.00094,
    ]
)
STIFF_KP = np.array([350, 200, 200, 300, 300, 150, 350, 200, 200, 300, 300, 150, 200, 200, 200] + [40] * 14, float)
STIFF_KD = np.array([5, 5, 5, 10, 5, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5] + [3] * 14, float)
STIFF_BLEND_TICKS = 100
# MJCF actuatorfrcrange (= g1_29dof dof_effort_limit_list, the bridge's torque clip)
TAU = np.array(
    [
        88,
        139,
        88,
        139,
        50,
        50,
        88,
        139,
        88,
        139,
        50,
        50,
        88,
        50,
        50,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ],
    float,
)

CAM_POS = (0.01, 0.01, 0.44)
CAM_QUAT_WXYZ = (0.60290764, 0.37585401, -0.362958, -0.60290764)
CAM_W, CAM_H, CAM_VFOV = 106, 60, 58.5953
CAM_HFOV = 89.5
NEAR, FAR, EMPTY = 0.3, 3.0, 0.15
CROP_TOP, CROP_LEFT, CROP_RIGHT = 2, 4, 4
DEPTH_HZ = 10.0
POLICY_HZ = 50.0

CMD_CODES = {
    0: {"stand": 0, "forward": 1, "left_45": 2, "left_90": 3, "right_45": 4, "right_90": 5, "back": 11},
    1: {"stand": 0, "forward": 6, "left_45": 7, "left_90": 8, "right_45": 9, "right_90": 10, "back": 11},
}
VELOCITY_DEADZONE = 0.05

_RS = np.load(os.path.join(HERE, "resize_98x58_to_87x58.npz"))
WW, WH = _RS["Ww"].astype(np.float32), _RS["Wh"].astype(np.float32)


def quantise(vx, vy, wz, speed_mode=0):
    """DepthDistillationPolicy._apply_velocity, verbatim: continuous (vx, vy, wz) -> direction class name."""
    if math.hypot(vx, vy) < VELOCITY_DEADZONE and abs(wz) < VELOCITY_DEADZONE:
        return "stand"
    if abs(vx) < VELOCITY_DEADZONE and abs(vy) < VELOCITY_DEADZONE:
        return "left_90" if wz > 0 else "right_90"
    a = math.degrees(math.atan2(vy, vx))
    if -22.5 <= a <= 22.5:
        return "forward"
    if 22.5 < a <= 67.5:
        return "left_45"
    if 67.5 < a <= 112.5:
        return "left_90"
    if -67.5 <= a < -22.5:
        return "right_45"
    if -112.5 <= a < -67.5:
        return "right_90"
    return "back"


def preprocess_depth(depth_m):
    """(60, 106) metric z-depth (inf / nan = no hit) -> (58, 87) normalised frame, as DepthShmPlugin."""
    d = np.asarray(depth_m, np.float32)
    d = np.nan_to_num(d, nan=FAR, posinf=FAR, neginf=NEAR)
    d = d[CROP_TOP:, CROP_LEFT:-CROP_RIGHT]
    d = np.clip(d, NEAR, FAR)
    d = WH @ d @ WW.T
    d = np.minimum(d, FAR)
    d = np.where(d < EMPTY, FAR, d)
    return ((d - NEAR) / (FAR - NEAR) - 0.5).astype(np.float32)


def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a - b + c


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_axis(angle, axis):
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def torso_quat(pelvis_quat_wxyz, q):
    """_get_anchor_quat: chain waist yaw (z), roll (x), pitch (y) onto the pelvis quaternion."""
    t = np.asarray(pelvis_quat_wxyz, float)
    for i, ax in ((12, (0, 0, 1)), (13, (1, 0, 0)), (14, (0, 1, 0))):
        t = quat_mul(t, quat_axis(float(q[i]), ax))
    return t


class PhpPolicy:
    """One composite policy (depth backbone + student). tick() = one 50 Hz control step."""

    def __init__(self, student_dir=STUDENT_DIR, threads=1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.bb = ort.InferenceSession(f"{student_dir}/depth_backbone.onnx", so, providers=["CPUExecutionProvider"])
        self.st = ort.InferenceSession(f"{student_dir}/student.onnx", so, providers=["CPUExecutionProvider"])
        self.speed_mode = 0
        self.reset()

    def reset(self):
        self.last_action = np.zeros(29, np.float32)
        self.direction = "stand"
        self.frame = None  # latest preprocessed depth frame (58, 87)
        self.latent = None
        self.ticks = 0

    def set_direction(self, name):
        assert name in CMD_CODES[self.speed_mode], name
        self.direction = name

    def onehot(self):
        v = np.zeros(15, np.float32)
        v[CMD_CODES[self.speed_mode][self.direction]] = 1.0
        return v

    def set_depth(self, frame58x87):
        self.frame = np.asarray(frame58x87, np.float32)

    def obs(self, q, dq, pelvis_quat_wxyz, gyro_body):
        g = quat_rotate_inverse(torso_quat(pelvis_quat_wxyz, q), np.array([0.0, 0.0, -1.0]))
        actor = np.concatenate(
            [self.last_action, np.asarray(gyro_body, np.float32), (np.asarray(q) - DEFAULT), np.asarray(dq), g]
        ).astype(np.float32)
        self.latent = self.bb.run(None, {"depth_image": self.frame[None]})[0][0]
        return np.concatenate([actor, self.onehot(), self.latent]).astype(np.float32)

    def tick(self, q, dq, pelvis_quat_wxyz, gyro_body):
        """-> joint targets q* (MJCF order)."""
        assert self.frame is not None, "no depth frame yet"
        o = self.obs(q, dq, pelvis_quat_wxyz, gyro_body)
        a = self.st.run(["actions"], {"obs": o[None], "time_step": np.zeros((1, 1), np.float32)})[0][0]
        a = np.clip(a, -100, 100).astype(np.float32)
        self.last_action = a.copy()
        self.ticks += 1
        return DEFAULT + a.astype(float)


def stiff_target(q_start, count):
    """Stiff-hold blend (q, kp, kd) at tick `count` (DepthDistillationPolicy._get_manual_command)."""
    t = min(count / STIFF_BLEND_TICKS, 1.0)
    al = t * t * (3.0 - 2.0 * t)
    return (1 - al) * q_start + al * STIFF_POS, al * STIFF_KP, al * STIFF_KD
