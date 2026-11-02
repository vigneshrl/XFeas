"""
Simulator-agnostic GR00T Decoupled WBC controller (NVlabs/GR00T-WholeBodyControl @45c03fa,
decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py + resources/robots/g1/g1_gear_wbc.yaml), numpy only.

Network (from the ONNX graph, both Walk and Balance): HIM-style
  e = estimator(obs516)  (516-256-256-35, ELU)
  v = e[:3] (estimated base velocity), z = e[3:]/max(||e[3:]||, 1e-12)
  action = actor([obs516[-86:], v, z])  (121-512-256-256-15, ELU)
Observation (86 per frame, history 6, OLDEST first -> input[-86:] is the newest frame):
  [0:3] (vx, vy, wz) * (2, 2, 0.5) | [3] absolute pelvis-height command (default 0.74) | [4:7] torso roll, pitch, yaw cmd
  [7:10] pelvis angular velocity (body frame, MuJoCo free-joint qvel[3:6]) * 0.5 | [10:13] projected gravity (pelvis)
  [13:42] q - default (29 joints, MJCF order, default = 15 lower values + 14 zeros) | [42:71] dq * 0.05 | [71:86] last action
History starts as 6 zero frames (deque of zeros, run_mujoco_gear_wbc.py:47), appended every control tick.
Switch: ||(vx, vy, wz)|| <= 0.05 -> Balance net, else Walk net (run_mujoco_gear_wbc.py:300; the stack's
g1_gear_wbc_policy.py uses < 0.05).
Target: q* = action * 0.25 + default (15 lower-body joints: 6 L leg, 6 R leg, waist yaw/roll/pitch).
PD (every 200 Hz physics step, effort): tau = Kp (q* - q) - Kd dq; Kp/Kd from the yaml. Upper body (14 arm joints):
tau = 100 (0 - q) - 0.5 dq (run_mujoco_gear_wbc.py:270-279). Policy at 50 Hz (dt 0.005, decimation 4); the
first policy tick happens after the 4th physics step (counter % 4 == 0 after increment); before that the target is
the default pose.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os
from collections import deque
import numpy as np

REPO = os.environ.get("GR00T_REPO", xfeas.paths.BASELINES + "/GR00T-WholeBodyControl")
G1_DIR = os.path.join(REPO, "decoupled_wbc/sim2mujoco/resources/robots/g1")
POLICY_DIR = os.path.join(G1_DIR, "policy")
NPZ_DIR = os.path.join(os.path.dirname(REPO), "GR00T-WBC", "npz")
os.makedirs(NPZ_DIR, exist_ok=True)

JOINTS = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]
KPS = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], np.float32)
DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], np.float32)
ARM_KP, ARM_KD = 100.0, 0.5
# actuatorfrcrange of g1_gear_wbc.xml (MuJoCo motor with actuatorfrclimited auto -> clamped joint actuator force)
TAU_LIM = np.array(
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
    np.float32,
)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], np.float32)
SIM_DT, DECIMATION, HIST, NOBS = 0.005, 4, 6, 86
NOMINAL_HEIGHT = 0.74


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0))).astype(np.float32)


class HIMPolicy:
    def __init__(self, npz):
        W = np.load(npz)
        self.E = [(W[f"estimator.{i}.weight"], W[f"estimator.{i}.bias"]) for i in (0, 2, 4)]
        self.A = [(W[f"actor.{i}.weight"], W[f"actor.{i}.bias"]) for i in (0, 2, 4, 6)]

    def __call__(self, x):
        x = np.asarray(x, np.float32).reshape(-1)
        h = x
        for i, (w, b) in enumerate(self.E):
            h = w @ h + b
            if i < 2:
                h = elu(h)
        v, z = h[:3], h[3:]
        z = z / max(float(np.linalg.norm(z)), 1e-12)
        h = np.concatenate([x[-NOBS:], v, z]).astype(np.float32)
        for i, (w, b) in enumerate(self.A):
            h = w @ h + b
            if i < 3:
                h = elu(h)
        return h


def gravity_orientation(quat):
    """run_mujoco_gear_wbc.py get_gravity_orientation: quat_rotate_inverse(q, [0,0,-1]); q = (w, x, y, z)"""
    w, x, y, z = quat
    return np.array([2 * (-x * z + w * y), -2 * (y * z + w * x), -(w * w - x * x - y * y + z * z)])


class Gr00tController:
    def __init__(self, backend="numpy"):
        if backend == "ort":
            import onnxruntime as ort

            def mk(n):
                s = ort.InferenceSession(os.path.join(POLICY_DIR, f"GR00T-WholeBodyControl-{n}.onnx"))
                return lambda x: s.run(None, {"input": x.reshape(1, -1).astype(np.float32)})[0].squeeze()

            self.balance, self.walk = mk("Balance"), mk("Walk")
        else:
            self.balance = HIMPolicy(os.path.join(NPZ_DIR, "gr00t_balance.npz"))
            self.walk = HIMPolicy(os.path.join(NPZ_DIR, "gr00t_walk.npz"))
        self.loco_cmd = np.zeros(3, np.float32)
        self.height_cmd = NOMINAL_HEIGHT
        self.rpy_cmd = np.zeros(3, np.float32)
        self.reset()

    def reset(self):
        self.action = np.zeros(15, np.float32)
        self.target = DEFAULT.copy()
        self.hist = deque([np.zeros(NOBS, np.float32)] * HIST, maxlen=HIST)
        self.counter = 0
        self.active = "balance"

    def set_command(self, vx=0.0, vy=0.0, wz=0.0, height=NOMINAL_HEIGHT, roll=0.0, pitch=0.0, yaw=0.0):
        self.loco_cmd = np.array([vx, vy, wz], np.float32)
        self.height_cmd = float(height)
        self.rpy_cmd = np.array([roll, pitch, yaw], np.float32)

    def observation(self, q, dq, quat, omega):
        o = np.zeros(NOBS, np.float32)
        o[0:3] = self.loco_cmd * CMD_SCALE
        o[3] = self.height_cmd
        o[4:7] = self.rpy_cmd
        o[7:10] = np.asarray(omega) * ANG_VEL_SCALE
        o[10:13] = gravity_orientation(quat)
        d = np.zeros(29, np.float32)
        d[:15] = DEFAULT
        o[13:42] = (np.asarray(q) - d) * DOF_POS_SCALE
        o[42:71] = np.asarray(dq) * DOF_VEL_SCALE
        o[71:86] = self.action
        return o

    def policy_tick(self, q, dq, quat, omega):
        """q, dq: 29 joints in JOINTS order; quat (w,x,y,z) of the pelvis; omega: pelvis ang vel in the pelvis frame."""
        self.hist.append(self.observation(q, dq, quat, omega))
        obs = np.concatenate(list(self.hist)).astype(np.float32)
        if np.linalg.norm(self.loco_cmd) <= 0.05:
            self.action, self.active = np.asarray(self.balance(obs), np.float32), "balance"
        else:
            self.action, self.active = np.asarray(self.walk(obs), np.float32), "walk"
        self.target = self.action * ACTION_SCALE + DEFAULT
        return self.target

    def torque(self, q, dq):
        """29 joint torques (before the actuator force clamp) -- run_mujoco_gear_wbc.py pd_control."""
        q = np.asarray(q, np.float32)
        dq = np.asarray(dq, np.float32)
        tau = np.zeros(29, np.float32)
        tau[:15] = (self.target - q[:15]) * KPS - dq[:15] * KDS
        tau[15:] = (0.0 - q[15:]) * ARM_KP - dq[15:] * ARM_KD
        return tau

    def physics_step_hook(self, read_state):
        """Call ONCE per physics step AFTER mj_step-equivalent; mirrors `counter += 1; if counter % 4 == 0: policy`."""
        self.counter += 1
        if self.counter % DECIMATION == 0:
            self.policy_tick(*read_state())
