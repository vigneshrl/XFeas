"""Simulator-agnostic reference re-implementation of the GEAR-SONIC planner-driven
G1 controller (NVlabs/GR00T-WholeBodyControl, gear_sonic_deploy C++ stack).

Faithfully mirrors, in pure numpy + onnxruntime (CPU):
  * G1Deploy::Control()                     src/g1_deploy_onnx_ref.cpp:3834-4075
  * G1Deploy::Planner()  (replan logic)      src/g1_deploy_onnx_ref.cpp:3595-3814
  * G1Deploy::CurrentFrameAdvancement()      src/g1_deploy_onnx_ref.cpp:3184-3436
    (planner pickup, 8-frame cross-fade blend, idle re-adaptation)
  * LocalMotionPlannerBase (Initialize / UpdatePlanning / context / 30->50 Hz)
                                             include/localmotion_kplanner.hpp
  * LocalMotionPlannerTensorRT input defaults (allowed_pred_num_tokens etc.)
                                             include/localmotion_kplanner_tensorrt.hpp:66-113,376-411
  * Observation registry / encoder mode filter
                                             src/g1_deploy_onnx_ref.cpp:1708-1815, 2022-2147
  * StateLogger::GetLatest (oldest-first, zero padded history)
                                             src/state_logger.cpp:199-260, 474-504
  * CreatePolicyCommand (action -> joint target)  src/g1_deploy_onnx_ref.cpp:3137-3167
  * constants (gains / default pose / action scale / joint maps)
                                             include/policy_parameters.hpp

Joint orders:
  * "mj"  = MuJoCo / hardware / URDF order (left leg 0-5, right leg 6-11, waist 12-14,
            left arm 15-21, right arm 22-28). All public I/O of this class is in mj order.
  * "il"  = IsaacLab order (policy-internal).
    ISAACLAB_TO_MUJOCO[mj_i] = il index of mj joint i ; MUJOCO_TO_ISAACLAB[il_i] = mj index.

Known, documented deviations from the C++ runtime (no effect on the math):
  * No reference-motion phase before the planner is enabled: reset() runs the planner
    Initialize() and installs the first planner motion immediately (the C++ state right
    after the first pickup tick: current_frame=1, reinitialize_heading=True).
  * Planner can run synchronously (default; planner tick every 5th control tick, result picked
    up at the end of the same tick => ~0 latency, close to the TensorRT build) or in a
    background thread (planner_async=True; result blended in with the C++ late-arrival logic).
  * Inference is ONNX Runtime CPU fp32 instead of TensorRT.
"""

from __future__ import annotations
import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)

import math
import os
import threading
from dataclasses import dataclass, field

import numpy as np
# onnxruntime is imported lazily in SonicController so the constants are importable from Isaac kit python

# ----------------------------------------------------------------------------------------
# Constants: include/policy_parameters.hpp
# ----------------------------------------------------------------------------------------
_ARM_5020, _ARM_7520_14, _ARM_7520_22, _ARM_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
_NF = 10 * 2.0 * 3.1415926535
_DR = 2.0
_K = {
    k: a * _NF * _NF
    for k, a in dict(s5020=_ARM_5020, s7520_14=_ARM_7520_14, s7520_22=_ARM_7520_22, s4010=_ARM_4010).items()
}
_D = {
    k: 2.0 * _DR * a * _NF
    for k, a in dict(s5020=_ARM_5020, s7520_14=_ARM_7520_14, s7520_22=_ARM_7520_22, s4010=_ARM_4010).items()
}
_E = dict(s5020=25.0, s7520_14=88.0, s7520_22=139.0, s4010=5.0)
# motor type per mj joint (policy_parameters.hpp:109-139)
_MT = (
    ["s7520_22", "s7520_22", "s7520_14", "s7520_22", "s5020", "s5020"] * 2
    + ["s7520_14", "s5020", "s5020"]
    + ["s5020", "s5020", "s5020", "s5020", "s5020", "s4010", "s4010"] * 2
)
G1_ACTION_SCALE = np.array([0.25 * _E[t] / _K[t] for t in _MT])
# kps/kds: ankles (4,5,10,11) and waist roll/pitch (13,14) are 2x (policy_parameters.hpp:143-207)
_X2 = {4, 5, 10, 11, 13, 14}
KPS = np.array([(2.0 if i in _X2 else 1.0) * _K[t] for i, t in enumerate(_MT)], dtype=np.float32).astype(np.float64)
KDS = np.array([(2.0 if i in _X2 else 1.0) * _D[t] for i, t in enumerate(_MT)], dtype=np.float32).astype(np.float64)
DEFAULT_ANGLES = np.array(
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
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
LOWER_BODY_IL_IN_IL = np.array([0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18])
LOWER_BODY_MJ_IN_IL = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18])
WRIST_IL_IN_IL = np.array([23, 24, 25, 26, 27, 28])
MJ_JOINT_NAMES = [
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
# Dex3 default hand pose when no hand data (input_interface.hpp:341-348), kp/kd (dex3_hands.hpp:360)
DEX3_LEFT_DEFAULT = np.array([0, 0, 1.75, -1.57, -1.75, -1.57, -1.75])
DEX3_RIGHT_DEFAULT = np.array([0, 0, -1.75, 1.57, 1.75, 1.57, 1.75])
DEX3_KP, DEX3_KD = 1.5, 0.1

CONTROL_DT = 0.02  # g1_deploy_onnx_ref.cpp:2186
PLANNER_DT = 0.1  # g1_deploy_onnx_ref.cpp:2187
PLANNER_EVERY = 5  # 10 Hz planner thread vs 50 Hz control thread
MOTION_LOOK_AHEAD = 2  # localmotion_kplanner.hpp:215
DEFAULT_HEIGHT = 0.788740  # localmotion_kplanner.hpp:216
INITIAL_SEED = 1234  # localmotion_kplanner.hpp:217
BLEND_FRAMES = 8  # g1_deploy_onnx_ref.cpp:3248
CAP = 1500  # MotionSequence ReserveCapacity(1500,...)

# ----------------------------------------------------------------------------------------
# Locomotion modes (localmotion_kplanner.hpp:78-106); names per docs planner_onnx.md
# ----------------------------------------------------------------------------------------
MODES = {
    "idle": 0,
    "slow_walk": 1,
    "walk": 2,
    "run": 3,
    "squat": 4,
    "kneel_two_legs": 5,
    "kneel_one_leg": 6,
    "lying_face_down": 7,
    "hand_crawl": 8,
    "idle_boxing": 9,
    "walk_boxing": 10,
    "left_jab": 11,
    "right_jab": 12,
    "random_punches": 13,
    "elbow_crawl": 14,
    "left_hook": 15,
    "right_hook": 16,
    "happy": 17,
    "stealth": 18,
    "injured": 19,
    "careful": 20,
    "object_carrying": 21,
    "crouch_walk": 22,
    "happy_dance": 23,
    "zombie": 24,
    "point": 25,
    "scared": 26,
}
STATIC_MODES = {0, 4, 5, 6, 7, 9}  # is_static_motion_mode
STANDING_MODES = {0, 1, 2, 3, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}
SQUAT_MODES = {4, 5, 6, 7, 8, 14}  # is_squat_motion_mode
BOXING_PUNCH = {11, 12, 13, 15, 16}
NUM_VALID_MODES_V2 = 27


# ----------------------------------------------------------------------------------------
# Quaternion math, ported 1:1 from include/math_utils.hpp (wxyz)
# ----------------------------------------------------------------------------------------
def quat_rotate(q, v):
    w = q[0]
    qv = np.asarray(q[1:4], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v * (2.0 * w * w - 1.0) + np.cross(qv, v) * w * 2.0 + qv * np.dot(qv, v) * 2.0


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    return np.array(
        [
            qq - ww + (z1 - y1) * (y2 - z2),
            qq - xx + (x1 + w1) * (x2 + w2),
            qq - yy + (w1 - x1) * (y2 + z2),
            qq - zz + (z1 + y1) * (w2 - x2),
        ]
    )


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    s = math.sin(angle / 2.0)
    q = np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])
    return q / max(np.linalg.norm(q), 1e-9)


def calc_heading(q):
    d = quat_rotate(q, [1.0, 0.0, 0.0])
    return math.atan2(d[1], d[0])


def heading_quat(q):
    return quat_from_angle_axis(calc_heading(q), [0, 0, 1])


def heading_quat_inv(q):
    return quat_from_angle_axis(-calc_heading(q), [0, 0, 1])


def quat_unit(q):
    return q / max(np.linalg.norm(q), 1e-9)


def quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_unit(q0 + t * (q1 - q0))
    th = math.acos(abs(dot))
    st = math.sin(th)
    return (math.sin((1.0 - t) * th) / st) * q0 + (math.sin(t * th) / st) * q1


def quat_to_rotmat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def normalize3(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), eps)


# ----------------------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------------------
class MotionSequence:
    """Planner motion buffer (root body only). jpos/jvel in IsaacLab order, absolute rad."""

    def __init__(self, cap=CAP):
        self.jpos = np.zeros((cap, 29))
        self.jvel = np.zeros((cap, 29))
        self.pos = np.zeros((cap, 3))
        self.quat = np.zeros((cap, 4))
        self.quat[:, 0] = 1.0
        self.timesteps = 0

    def copy(self):
        m = MotionSequence(len(self.jpos))
        m.timesteps = self.timesteps
        n = self.timesteps
        m.jpos[:n] = self.jpos[:n]
        m.jvel[:n] = self.jvel[:n]
        m.pos[:n] = self.pos[:n]
        m.quat[:n] = self.quat[:n]
        return m


@dataclass
class MovementState:  # localmotion_kplanner.hpp:60-73
    locomotion_mode: int = 0
    movement_direction: tuple = (0.0, 0.0, 0.0)
    facing_direction: tuple = (1.0, 0.0, 0.0)
    movement_speed: float = -1.0
    height: float = -1.0


@dataclass
class _Entry:  # StateLogger::Entry subset
    base_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_q: np.ndarray = field(default_factory=lambda: np.zeros(29))
    body_dq: np.ndarray = field(default_factory=lambda: np.zeros(29))
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(29))


# Encoder observation registry dims (g1_deploy_onnx_ref.cpp:1708-1815) for every name used by
# the released observation configs (default / sonic_v1_1 / low_latency / sonic_release alias).
_OBS_DIMS = {
    "token_state": 64,
    "encoder_mode": 3,
    "encoder_mode_4": 4,
    "encoder_index": 4,
    "command_multi_future_nonflat": 580,
    "motion_anchor_ori_b_mf_nonflat": 60,
    "command_multi_future_lower_body": 240,
    "motion_anchor_ori_b": 6,
    "smpl_joints_multi_future_local_nonflat": 720,
    "smpl_root_ori_b_multi_future": 60,
    "joint_pos_multi_future_wrist_for_smpl": 60,
    "motion_joint_positions": 29,
    "motion_joint_velocities": 29,
    "motion_anchor_orientation": 6,
    "motion_root_z_position": 1,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position_10frame_step1": 10,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_anchor_orientation_10frame_step1": 60,
    "motion_anchor_orientation_heading": 6,
    "motion_anchor_orientation_heading_10frame_step5": 60,
    "motion_anchor_orientation_heading_10frame_step1": 60,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_joint_positions_10frame_step1": 290,
    "motion_joint_velocities_10frame_step1": 290,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "motion_joint_positions_lowerbody_10frame_step1": 120,
    "motion_joint_velocities_lowerbody_10frame_step1": 120,
    "motion_joint_positions_wrists_10frame_step1": 60,
    "motion_joint_positions_wrists_4frame_step1": 24,
    "smpl_joints_10frame_step1": 720,
    "smpl_joints_4frame_step1": 288,
    "smpl_anchor_orientation_10frame_step1": 60,
    "smpl_anchor_orientation_4frame_step1": 24,
    "smpl_anchor_orientation_heading_10frame_step1": 60,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}


def load_obs_config(path):
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    pol = [o["name"] for o in cfg["observations"] if o.get("enabled", True)]
    enc = [o["name"] for o in cfg["encoder"]["encoder_observations"] if o.get("enabled", True)]
    modes = {m["mode_id"]: list(m["required_observations"]) for m in cfg["encoder"]["encoder_modes"]}
    return pol, enc, modes, int(cfg["encoder"]["dimension"])


# ----------------------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------------------
class SonicController:
    """GEAR-SONIC planner-driven controller (default release variant).

    Typical use (50 Hz):
        c = SonicController(xfeas.paths.BASELINES + '/GEAR-SONIC')
        c.reset(q_mj, base_quat_wxyz)
        c.set_command('slow_walk', vx=0.5)
        q_target = c.control_tick(q_mj, dq_mj, base_quat_wxyz, gyro_body)
    then apply tau = KPS*(q_target - q) - KDS*dq (dq_target=0, tau_ff=0) at the physics rate,
    clipped to the motor effort limits.
    """

    def __init__(
        self,
        model_dir=xfeas.paths.BASELINES + "/GEAR-SONIC",
        variant="default",
        planner_path=None,
        planner_async=False,
        intra_op_threads=0,
        verbose=False,
    ):
        import onnxruntime as ort

        sub = {"default": "", "sonic_v1_1": "sonic_v1_1", "low_latency": "low_latency"}[variant]
        vdir = os.path.join(model_dir, sub)
        self.variant = variant
        self.pol_obs, self.enc_obs, self.enc_modes, self.token_dim = load_obs_config(
            os.path.join(vdir, "observation_config.yaml")
        )
        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(os.path.join(vdir, "model_encoder.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(os.path.join(vdir, "model_decoder.onnx"), so, providers=prov)
        self.planner = ort.InferenceSession(
            planner_path or os.path.join(model_dir, "planner_sonic.onnx"), so, providers=prov
        )
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_in = self.dec.get_inputs()[0].name
        self.enc_dim = self.enc.get_inputs()[0].shape[1]
        self.dec_dim = self.dec.get_inputs()[0].shape[1]
        # encoder offsets
        self.enc_offsets = {}
        off = 0
        for n in self.enc_obs:
            self.enc_offsets[n] = (off, _OBS_DIMS[n])
            off += _OBS_DIMS[n]
        assert off == self.enc_dim, f"encoder dim mismatch {off} vs {self.enc_dim}"
        self.pol_offsets = {}
        off = 0
        for n in self.pol_obs:
            d = self.token_dim if n == "token_state" else _OBS_DIMS[n]
            self.pol_offsets[n] = (off, d)
            off += d
        assert off == self.dec_dim, f"decoder dim mismatch {off} vs {self.dec_dim}"
        self.planner_async = planner_async
        self.verbose = verbose
        self.encoder_mode = 0  # initial_encoder_mode_ = 0 when encoder loaded (cpp:2402)
        # planner input buffers (localmotion_kplanner_tensorrt.hpp:66-113)
        self._allowed = np.zeros((1, 11), np.int64)
        self._allowed[0, 3:6] = 1
        self._seed = np.array([INITIAL_SEED], np.int64)
        self._lock = threading.Lock()
        self._thread = None
        self.timing = {"planner_ms": [], "policy_ms": []}
        self.reset_state()

    # ------------------------------------------------------------------ state
    def reset_state(self):
        self.hist = []  # StateLogger ring (oldest..newest)
        self.last_action = np.zeros(29)  # IsaacLab order raw action
        self.motion = MotionSequence()  # planner_motion_
        self.current_frame = 0
        self.play = True
        self.reinitialize_heading = True
        self.heading_init_base_quat = np.array([1.0, 0, 0, 0])
        self.delta_heading = 0.0
        self.init_ref_root_rot = np.array([1.0, 0, 0, 0])
        self.movement = MovementState()  # movement_state_buffer_ (as keyboard writes it)
        self.last_movement = MovementState(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.0, -1.0)  # cpp:230
        self.replan_counter = 0.0
        self.tick = 0
        self.gen = None  # (motion_50hz, gen_frame) awaiting pickup
        self.idle_stored = False
        self.idle_state = "IDLE"
        self.idle_targets = np.zeros(29)
        self.facing_angle = 0.0  # planner-frame facing (keyboard planner_facing_angle)
        self.planner_initialized = False
        self.last_q = None

    # ------------------------------------------------------------------ planner core
    def _planner_run(self, ctx, mode, vel, mv, fc, height):
        import time

        t = time.perf_counter()
        mode = int(mode) if 0 <= int(mode) < NUM_VALID_MODES_V2 else 0  # tensorrt.hpp:384-388
        feeds = {
            "context_mujoco_qpos": ctx.astype(np.float32).reshape(1, 4, 36),
            "target_vel": np.array([vel], np.float32),
            "mode": np.array([mode], np.int64),
            "movement_direction": np.array([mv], np.float32),
            "facing_direction": np.array([fc], np.float32),
            "random_seed": self._seed,
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": self._allowed,
            "height": np.array([height], np.float32),
        }
        qpos, n = self.planner.run(None, feeds)
        self.timing["planner_ms"].append((time.perf_counter() - t) * 1e3)
        return self._resample_50hz(qpos[0], int(np.asarray(n).reshape(-1)[0]))

    @staticmethod
    def _resample_50hz(q30, n30):
        """LocalMotionPlannerBase::ResampleGeneratedSequence50Hz (localmotion_kplanner.hpp:395-519)."""
        if n30 > 64 or np.isnan(q30[:n30]).any():
            raise RuntimeError("planner output invalid")
        q30 = q30.astype(np.float64)
        m = MotionSequence()
        T = int(math.floor(n30 / 30.0 * 50))
        m.timesteps = T
        for f in range(T):
            f30 = (f / 50.0) * 30
            f0 = int(math.floor(f30))
            f1 = min(f0 + 1, n30 - 1)
            w0 = 1.0 - (f30 - f0)
            w1 = 1.0 - w0
            m.pos[f] = w0 * q30[f0, :3] + w1 * q30[f1, :3]
            m.quat[f] = quat_slerp(q30[f0, 3:7], q30[f1, 3:7], f30 - f0)
            m.jpos[f] = w0 * q30[f0, 7 + MUJOCO_TO_ISAACLAB] + w1 * q30[f1, 7 + MUJOCO_TO_ISAACLAB]
        m.jvel[: T - 1] = (m.jpos[1:T] - m.jpos[: T - 1]) * 50.0
        m.jvel[T - 1] = m.jvel[T - 2]
        return m

    @staticmethod
    def _context_from_motion(ms, gen_frame):
        """LocalMotionPlannerBase::UpdateContextFromMotion (localmotion_kplanner.hpp:628-678)."""
        ctx = np.zeros((4, 36), np.float32)
        gt = gen_frame / 50.0
        for n in range(4):
            f50 = (gt + n / 30.0) * 50.0
            f0 = min(int(math.floor(f50)), ms.timesteps - 1)
            f1 = min(f0 + 1, ms.timesteps - 1)
            w0 = 1.0 - (f50 - f0)
            w1 = 1.0 - w0
            ctx[n, 3:7] = quat_slerp(ms.quat[f0], ms.quat[f1], f50 - f0)
            ctx[n, 0:3] = w0 * ms.pos[f0] + w1 * ms.pos[f1]
            ctx[n, 7 + MUJOCO_TO_ISAACLAB] = w0 * ms.jpos[f0] + w1 * ms.jpos[f1]
        return ctx

    def _planner_initialize(self, q_mj):
        """LocalMotionPlannerBase::Initialize + InitializeContext (localmotion_kplanner.hpp:332-393, 591-624)."""
        ctx = np.zeros((4, 36), np.float32)
        ctx[:, 2] = DEFAULT_HEIGHT
        ctx[:, 3] = 1.0
        ctx[:, 7:] = np.asarray(q_mj, dtype=np.float32)
        gen = self._planner_run(ctx, 0, -1.0, (0, 0, 0), (1, 0, 0), -1.0)
        self.gen = (gen, 0)
        self.planner_initialized = True

    def _planner_tick(self):
        """G1Deploy::Planner() replan decision (g1_deploy_onnx_ref.cpp:3650-3800)."""
        if self.motion.timesteps == 0:
            return
        if self.planner_async and self._thread is not None and self._thread.is_alive():
            return  # the single C++ planner thread is still busy: this 10 Hz tick does not happen
        ms, lm = self.movement, self.last_movement
        facing_changed = tuple(ms.facing_direction) != tuple(lm.facing_direction)
        height_changed = ms.height != lm.height
        mode_changed = ms.locomotion_mode != lm.locomotion_mode
        speed_changed = ms.movement_speed != lm.movement_speed
        dir_changed = tuple(ms.movement_direction) != tuple(lm.movement_direction)
        static = ms.locomotion_mode in STATIC_MODES
        mode = ms.locomotion_mode
        self.replan_counter += PLANNER_DT
        interval = 0.1 if mode == 3 else (0.2 if mode == 8 else 1.0)  # running / CRAWLING / boxing+default
        time_to_replan = False
        if self.replan_counter >= interval - 1e-9:
            self.replan_counter = 0.0
            time_to_replan = True
        need = (
            mode_changed
            or facing_changed
            or height_changed
            or ((not static) and (speed_changed or dir_changed or (time_to_replan and ms.movement_speed != 0)))
        )
        if not need:
            return
        self.last_movement = MovementState(
            ms.locomotion_mode, tuple(ms.movement_direction), tuple(ms.facing_direction), ms.movement_speed, ms.height
        )
        gen_frame = self.current_frame + MOTION_LOOK_AHEAD
        ctx = self._context_from_motion(self.motion, gen_frame)
        args = (
            ctx,
            ms.locomotion_mode,
            float(ms.movement_speed),
            tuple(ms.movement_direction),
            tuple(ms.facing_direction),
            float(ms.height),
        )
        if self.verbose:
            print(
                f"[planner] replan mode={ms.locomotion_mode} v={ms.movement_speed} mv={ms.movement_direction} "
                f"fc={np.round(ms.facing_direction, 3)} h={ms.height} gen_frame={gen_frame}"
            )
        if not self.planner_async:
            self.gen = (self._planner_run(*args), gen_frame)
        else:

            def work():
                g = self._planner_run(*args)
                with self._lock:
                    self.gen = (g, gen_frame)

            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ commands
    def set_planner_command(
        self, mode, movement_direction=(0, 0, 0), facing_direction=(1, 0, 0), speed=-1.0, height=-1.0
    ):
        """Raw planner command in the PLANNER frame (as ZMQManager writes it, zmq_manager.hpp:594-613)."""
        mode = MODES[mode] if isinstance(mode, str) else int(mode)
        if mode in SQUAT_MODES and height < 0.2:
            height = 0.2
        if mode in STATIC_MODES:
            speed = -1.0
        self.movement = MovementState(
            mode,
            tuple(float(x) for x in normalize3(movement_direction)),
            tuple(float(x) for x in normalize3(facing_direction)),
            float(speed),
            float(height),
        )

    def set_command(self, mode, vx=0.0, vy=0.0, wz=None, heading=None, height=-1.0, speed=None):
        """Convenience body-frame command.

        vx, vy : desired planar velocity in the current *facing* frame (m/s); the planner gets
                 movement_direction = R(facing) [vx,vy] and target_vel = |v| (or `speed`).
                 |v| == 0 -> movement 0 and speed -1 for static modes / 0 otherwise (keyboard semantics).
        wz     : yaw rate (rad/s); integrated into the planner facing angle once per control tick.
        heading: absolute facing in the planner frame (rad); overrides wz.
        height : pelvis height for squat/kneel/crawl modes (m), -1 = mode default.
        """
        m = MODES[mode] if isinstance(mode, str) else int(mode)
        self._cmd = dict(mode=m, vx=float(vx), vy=float(vy), wz=wz, heading=heading, height=float(height), speed=speed)

    def _apply_cmd(self):
        c = getattr(self, "_cmd", None)
        if c is None:
            return
        if c["heading"] is not None:
            self.facing_angle = float(c["heading"])
        elif c["wz"]:
            self.facing_angle += float(c["wz"]) * CONTROL_DT
        fa = self.facing_angle
        facing = (math.cos(fa), math.sin(fa), 0.0)
        v = math.hypot(c["vx"], c["vy"])
        if v > 1e-6:
            mvx = math.cos(fa) * c["vx"] - math.sin(fa) * c["vy"]
            mvy = math.sin(fa) * c["vx"] + math.cos(fa) * c["vy"]
            mv = (mvx / v, mvy / v, 0.0)
            spd = float(c["speed"]) if c["speed"] is not None else v
        else:
            mv = (0.0, 0.0, 0.0)
            spd = -1.0 if c["mode"] in STATIC_MODES else 0.0
        mode = c["mode"]
        h = c["height"]
        if mode in STANDING_MODES:
            h = -1.0
        self.set_planner_command(mode, mv, facing, spd, h)

    def planner_to_world_yaw(self):
        """Yaw offset such that world_yaw = planner_yaw + offset (apply_delta_heading)."""
        return calc_heading(self._apply_delta_heading())

    # ------------------------------------------------------------------ obs helpers
    def _apply_delta_heading(self):
        a = quat_mul(heading_quat(self.heading_init_base_quat), heading_quat_inv(self.init_ref_root_rot))
        if self.delta_heading != 0.0:
            a = quat_mul(quat_from_angle_axis(self.delta_heading, [0, 0, 1]), a)
        return a

    def _frames(self, nf, step):
        T = self.motion.timesteps
        return [min(self.current_frame + k * step, T - 1) if self.play else self.current_frame for k in range(nf)]

    def _anchor_ori(self, nf, step, mode):
        base_quat = self.hist[-1].base_quat
        apply = self._apply_delta_heading()
        out = np.zeros(nf * 6)
        left = heading_quat(base_quat) if mode == 1 else base_quat
        for k, f in enumerate(self._frames(nf, step)):
            new_ref = quat_mul(apply, self.motion.quat[f])
            R = quat_to_rotmat(quat_mul(quat_conj(left), new_ref))
            out[k * 6 : (k + 1) * 6] = R[:, :2].reshape(-1)
        return out

    def _jpos(self, nf, step, idx=None):
        fr = self._frames(nf, step)
        a = self.motion.jpos[fr]
        return (a if idx is None else a[:, idx]).reshape(-1)

    def _jvel(self, nf, step, idx=None):
        T = self.motion.timesteps
        fr = [min(self.current_frame + k * step, T - 1) for k in range(nf)]  # velocities always advance (cpp:842)
        a = self.motion.jvel[fr] if self.play else np.zeros((nf, 29))
        return (a if idx is None else a[:, idx]).reshape(-1)

    def _encoder_term(self, name):
        if name in ("encoder_mode_4", "encoder_index"):
            return np.array([float(self.encoder_mode), 0, 0, 0])
        if name == "encoder_mode":
            return np.array([float(self.encoder_mode), 0, 0])
        if name == "command_multi_future_nonflat":
            return np.concatenate([self._jpos(10, 5), self._jvel(10, 5)])
        if name == "command_multi_future_lower_body":
            return np.concatenate([self._jpos(10, 5, LOWER_BODY_MJ_IN_IL), self._jvel(10, 5, LOWER_BODY_MJ_IN_IL)])
        if name.startswith("motion_joint_positions") or name.startswith("motion_joint_velocities"):
            fn = self._jpos if "positions" in name else self._jvel
            nf, step = 1, 1
            if "10frame_step5" in name:
                nf, step = 10, 5
            elif "10frame_step1" in name:
                nf, step = 10, 1
            elif "4frame_step1" in name:
                nf, step = 4, 1
            idx = LOWER_BODY_MJ_IN_IL if "lowerbody" in name else (WRIST_IL_IN_IL if "wrists" in name else None)
            return fn(nf, step, idx)
        if name in ("motion_anchor_ori_b_mf_nonflat",):
            return self._anchor_ori(10, 5, 0)
        if name in ("motion_anchor_ori_b",):
            return self._anchor_ori(1, 1, 0)
        if name.startswith("motion_anchor_orientation"):
            mode = 1 if "heading" in name else 0
            nf, step = (10, 5) if "10frame_step5" in name else ((10, 1) if "10frame_step1" in name else (1, 1))
            return self._anchor_ori(nf, step, mode)
        raise NotImplementedError(f"encoder observation '{name}' (only g1 mode 0 is reproduced)")

    def _hist(self, n=10):
        h = self.hist[-n:]
        return [_Entry()] * (n - len(h)) + h  # zero entries padded on the oldest side

    # ------------------------------------------------------------------ main tick
    def reset(self, q_mj, base_quat_wxyz=None):
        """Enable + initialise planner from current joint positions (mj order, absolute rad)."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.reset_state()
        self._cmd = None
        self.set_planner_command(0, (0, 0, 0), (1, 0, 0), -1.0, -1.0)  # keyboard writes IDLE,-1,-1 after enable
        self._planner_initialize(np.asarray(q_mj, dtype=np.float64))
        # first pickup happens at the end of a control tick: emulate it now (state right after that tick)
        self._frame_advancement(np.asarray(q_mj, dtype=np.float64))

    def control_tick(self, q_mj, dq_mj, base_quat_wxyz, gyro_body):
        """One 50 Hz tick. Inputs: joint pos/vel (mj order, absolute), pelvis quat (wxyz, world),
        pelvis gyro (body frame). Returns q_target (mj order, rad). Gains: KPS/KDS."""
        import time

        q = np.asarray(q_mj, dtype=np.float64)
        dq = np.asarray(dq_mj, dtype=np.float64)
        # GatherRobotStateToLogger (cpp:2842-2978)
        e = _Entry(
            base_quat=np.asarray(base_quat_wxyz, dtype=np.float64).copy(),
            base_ang_vel=np.asarray(gyro_body, dtype=np.float64).copy(),
            body_q=q[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB],
            body_dq=dq[MUJOCO_TO_ISAACLAB].copy(),
            last_action=self.last_action.copy(),
        )
        self.hist.append(e)
        if len(self.hist) > 64:
            self.hist.pop(0)
        # input thread (100 Hz) -> movement_state_buffer_
        self._apply_cmd()
        # planner thread (10 Hz)
        if self.tick % PLANNER_EVERY == 0:
            self._planner_tick()
        self.tick += 1
        t = time.perf_counter()
        # UpdateHeadingState (cpp:559-585)
        if self.reinitialize_heading:
            self.heading_init_base_quat = e.base_quat.copy()
            self.delta_heading = 0.0
            self.reinitialize_heading = False
            self.init_ref_root_rot = self.motion.quat[self.current_frame].copy()
        if self.current_frame == 0:
            self.init_ref_root_rot = self.motion.quat[0].copy()
        # encoder obs with mode filter (cpp:2022-2147)
        req = set(self.enc_modes.get(self.encoder_mode, self.enc_obs))
        eobs = np.zeros(self.enc_dim)
        for n, (o, d) in self.enc_offsets.items():
            if n in req:
                eobs[o : o + d] = self._encoder_term(n)
        token = self.enc.run(None, {self.enc_in: eobs.astype(np.float32)[None]})[0][0].astype(np.float64)
        # policy obs
        h = self._hist(10)
        pobs = np.zeros(self.dec_dim)
        for n, (o, d) in self.pol_offsets.items():
            if n == "token_state":
                v = token
            elif n == "his_base_angular_velocity_10frame_step1":
                v = np.concatenate([x.base_ang_vel for x in h])
            elif n == "his_body_joint_positions_10frame_step1":
                v = np.concatenate([x.body_q for x in h])
            elif n == "his_body_joint_velocities_10frame_step1":
                v = np.concatenate([x.body_dq for x in h])
            elif n == "his_last_actions_10frame_step1":
                v = np.concatenate([x.last_action for x in h])
            elif n == "his_gravity_dir_10frame_step1":
                v = np.concatenate([quat_rotate(quat_conj(x.base_quat), [0, 0, -1.0]) for x in h])
            else:
                raise NotImplementedError(n)
            pobs[o : o + d] = v
        self.last_obs = pobs
        self.last_enc_obs = eobs
        self.last_token = token
        act = self.dec.run(None, {self.dec_in: pobs.astype(np.float32)[None]})[0][0].astype(np.float64)
        self.timing["policy_ms"].append((time.perf_counter() - t) * 1e3)
        # CreatePolicyCommand (cpp:3137-3167)
        q_target = DEFAULT_ANGLES + act[ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE
        q_target = q_target.astype(np.float32).astype(np.float64)
        self.last_action = act.copy()
        # CurrentFrameAdvancement (cpp:3184-3405)
        self._frame_advancement(q)
        return q_target

    def _frame_advancement(self, q_mj):
        with self._lock:
            gen = self.gen
            self.gen = None
        if gen is not None and gen[0].timesteps > 0:
            g, fgen = gen
            pm = self.motion
            success = False
            first = False
            if pm.timesteps == 0:
                n = g.timesteps
                pm.jpos[:n] = g.jpos[:n]
                pm.jvel[:n] = g.jvel[:n]
                pm.pos[:n] = g.pos[:n]
                pm.quat[:n] = g.quat[:n]
                pm.timesteps = n
                success = True
                first = True
            else:
                cf = self.current_frame
                new_len = fgen - cf + g.timesteps
                if new_len > 0:
                    bs = max(0, fgen - cf)
                    for f in range(new_len):  # in-place, exactly as the C++ loop
                        fo = min(max(f + cf, 0), pm.timesteps - 1)
                        fn = min(max(f + cf - fgen, 0), g.timesteps - 1)
                        wn = min(max((f - bs) / BLEND_FRAMES, 0.0), 1.0)
                        wo = 1.0 - wn
                        pm.jpos[f] = wo * pm.jpos[fo] + wn * g.jpos[fn]
                        pm.jvel[f] = wo * pm.jvel[fo] + wn * g.jvel[fn]
                        pm.pos[f] = wo * pm.pos[fo] + wn * g.pos[fn]
                        pm.quat[f] = quat_slerp(pm.quat[fo], g.quat[fn], wn)
                    pm.timesteps = new_len
                    success = True
            if success:
                self.current_frame = 0
                self.idle_stored = False
                if first:
                    self.reinitialize_heading = True
        # advance (50 Hz)
        if self.motion.timesteps > 0 and self.play:
            nf = self.current_frame + 1
            if nf >= self.motion.timesteps:
                nf = self.motion.timesteps - 1
                if self.movement.locomotion_mode == 0:  # idle re-adaptation (cpp:3340-3401)
                    pm = self.motion
                    if not self.idle_stored:
                        self.idle_targets[LOWER_BODY_IL_IN_IL] = pm.jpos[nf, LOWER_BODY_IL_IN_IL]
                        self.idle_stored = True
                        self.idle_state = "IDLE"
                    actual = q_mj[MUJOCO_TO_ISAACLAB[LOWER_BODY_IL_IN_IL]]
                    err = float(np.mean(np.abs(pm.jpos[nf, LOWER_BODY_IL_IN_IL] - actual)))
                    if self.idle_state == "IDLE":
                        if err > 0.10:
                            self.idle_state = "ADAPTING"
                        elif err < 0.045:
                            self.idle_state = "RECOVERING"
                    elif self.idle_state == "ADAPTING":
                        if err < 0.05:
                            self.idle_state = "IDLE"
                    elif self.idle_state == "RECOVERING":
                        if err > 0.10:
                            self.idle_state = "ADAPTING"
                    if self.idle_state == "ADAPTING":
                        pm.jpos[nf, LOWER_BODY_IL_IN_IL] = 0.98 * pm.jpos[nf, LOWER_BODY_IL_IN_IL] + 0.02 * actual
                    elif self.idle_state == "RECOVERING":
                        pm.jpos[nf, LOWER_BODY_IL_IN_IL] = (
                            0.98 * pm.jpos[nf, LOWER_BODY_IL_IN_IL] + 0.02 * self.idle_targets[LOWER_BODY_IL_IN_IL]
                        )
            self.current_frame = nf

    # ------------------------------------------------------------------ introspection
    def reference_frame(self):
        """Current reference (planner) root pos/quat (planner frame) and joint pos (mj order)."""
        f = self.current_frame
        return self.motion.pos[f].copy(), self.motion.quat[f].copy(), self.motion.jpos[f][ISAACLAB_TO_MUJOCO].copy()
