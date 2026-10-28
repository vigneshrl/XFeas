#!/usr/bin/env python
"""
Unitree G1 (23-DoF) driven by the RELEASED AMO policy (Li, Cheng, Huang et al., "AMO: Adaptive Motion Optimization
for Hyper-Dexterous Humanoid Whole-Body Control", RSS 2025; github.com/OpenTeleVision/AMO, Apache-2.0):
amo_jit.pt + adapter_jit.pt + adapter_norm_stats.pt. Commands: vx, vy, heading, BODY HEIGHT, TORSO YAW/PITCH/ROLL.

    --robot g1a --robot-factory g1_amo:make_g1a        (with controllers/ on PYTHONPATH)

The release is a MuJoCo deployment (play_amo.py). Everything is reproduced from it; the observation / action /
gait-clock / PD logic is the SAME code object (amo_mujoco_check.AMOController, a line-by-line copy of
play_amo.py's HumanoidEnv used for the MuJoCo reference run). Nothing tuned:
  physics 500 Hz (sim_dt 0.002), policy 50 Hz (decimation 10)
  PD in effort mode every physics step: tau = clip(Kp (q* - q) - Kd dq, +-torque_limits), Kp/Kd/limits from
     play_amo.py; arms held at the default pose (arm toggle off)
  passive joint terms of g1.xml: damping 2 (imported as PhysX drive damping, stiffness 0), armature (imported),
     frictionloss 0.2 N*m (emulated: -0.2 * clip(dq / 0.05, -1, 1); PhysX has no torque-valued joint friction)
  observation: ang vel of imu_in_pelvis (pelvis frame) * 0.25 | roll, pitch | sin, cos(yaw - target_yaw) (0 when
     |vx| < 0.1) | q - q0 (23) | dq * 0.05 (23) | last action (15 raw + 8 arm) | gait clock (2) | adapter output (15)
     + demo (17) + priv zeros (3) + 10-step history; extra 25-step history to the jit
  commands: height = pelvis height target (play_amo: 0.75 + offset), torso yaw/pitch/roll, vx, vy; target yaw is
     a HEADING (absolute) -- here wz is integrated into it and kept within 0.5 rad of the current yaw
  deployment quirks kept as released: for |vx| < 0.1 the policy is in 'stand' mode (dyaw forced to 0 and the gait
     clock frozen), so it can neither turn in place nor side-step without forward speed.
Robot model: AMO's own g1.xml (ground line removed) imported with Isaac 5.1's MJCF importer
(controllers/convert_g1_mjcf.py). Contacts are g1.xml's: 4 r=5 mm spheres per foot + the pelvis contour mesh.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

AMO = os.environ.get("AMO_DIR", xfeas_paths.BASELINES + "/AMO")
G1_USD = os.path.join(AMO, "usd/g1_amo.usd")
NOMINAL_H = 0.75
KEYFRAME = dict(zip([n + "_joint" for n in ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
                                            "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
                                            "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                                            "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
                                            "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                                            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                                            "right_elbow"]],
                    [-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0, 0, 0, 0, 0, 0.4, 0, 1.2, 0, -0.4, 0, 1.2]))
# setpoint clip ranges: README "I.D." (in-distribution) ranges; height from the MuJoCo reference run
RANGES = dict(vx=(-0.5, 0.5), vy=(-0.4, 0.4), wz=(-1.0, 1.0), body_height=(0.30, 0.80), torso_yaw=(-1.57, 1.57),
              pitch=(-0.52, 1.57), roll=(-0.7, 0.7))
DEFAULT_SP = dict(vx=0.0, vy=0.0, wz=0.0, body_height=NOMINAL_H, torso_yaw=0.0, pitch=0.0, roll=0.0)
FRICTIONLOSS = 0.2


def _spec():
    import robots as RB
    return RB.RobotSpec("g1a", "legged", (0.5, 0.5, 0.4, 0.0, 1.0), 0.80, 0.8, 500, usd=G1_USD,
                        foot_expr="pelvis/.*_ankle_roll_link", v_min=0.1, cam=(0.1, 0.55, -0.5), min_z=0.3,
                        extra=dict(controller="AMO (released)", setpoint_ranges=RANGES,
                                   # v11 (additive): the feasibility layer's search box for THIS robot (G1_OVERHANG.md:
                                   # AMO crouches under overhangs only at 0.30 m with torso pitch +1.0..+1.4)
                                   layer_box=dict(body_height=(0.30, 0.75), pitch=(-0.35, 1.5), roll=(-0.35, 0.35))))


class G1AMOAdapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        import robots as RB
        import amo_mujoco_check as AM
        self.RB, self.AM = RB, AM
        self.spec = make_g1a.SPEC
        self.kind, self.prim_path = kind, prim_path
        self.cmd = np.zeros(4); self.cmd_req = np.zeros(4); self.clipped = False; self.clipped_keys = []
        self.sp = dict(DEFAULT_SP)
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(G1_USD, prim_path)
        q = Rotation.from_euler("z", yaw).as_quat()
        # the MJCF importer puts the articulation root on <default prim>/pelvis/pelvis
        self._art = SingleArticulation(prim_path=prim_path + "/pelvis/pelvis", name=kind, position=np.asarray(position, float),
                                       orientation=np.array([q[3], q[0], q[1], q[2]]))
        self.robot = self._art
        self.ctrl = None
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self.C = AM.AMOController(device="cuda:0")         # the released jit hard-codes cuda:0
        self._floor_z = 0.0

    def initialize(self):
        self._art.initialize()
        names = list(self._art.dof_names)
        amo_names = [n + "_joint" for n in self.AM.AMOController.dof_names]
        self.idx = np.array([names.index(n) for n in amo_names])
        n = len(names)
        av = self._art._articulation_view
        av.set_gains(kps=np.zeros((1, n)), kds=np.full((1, n), 2.0))      # g1.xml passive damping 2 (implicit)
        cap = np.full(n, 1e6); cap[self.idx] = self.AM.AMOController.torque_limits
        av.set_max_efforts(cap.reshape(1, -1))
        self._effort_cap = cap
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        self._q0 = np.zeros(n); self._q0[self.idx] = [KEYFRAME[k] for k in amo_names]
        dpos = np.zeros(n); dpos[self.idx] = self.AM.AMOController.default_dof_pos
        self.ctrl = SimpleNamespace(default_pos=dpos)
        # pelvis -> imu_in_pelvis offset is a pure translation, so the site's orientation/gyro = the pelvis'
        print("[g1a] dofs", n, "mass", round(float(self._base_masses.sum()), 2), flush=True)
        self._reset_policy()
        self.ready = True

    def _reset_policy(self):
        self.C.reset(); self.C.commands[:] = 0
        self.sub = 0
        self.target_yaw = None
        self.last_tau = np.zeros(23)

    def _state(self):
        _, quat = self._art.get_world_pose()
        quat = np.asarray(quat, float)                    # w, x, y, z
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        gyro = R.T @ np.asarray(self._art.get_angular_velocity(), float)
        q = np.asarray(self._art.get_joint_positions(), float).reshape(-1)[self.idx]
        dq = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)[self.idx]
        return quat, gyro, q, dq

    def _policy_tick(self):
        quat, gyro, q, dq = self._state()
        yaw = self.AM.quatToEuler(quat)[2]
        if self.target_yaw is None:
            self.target_yaw = yaw
        self.target_yaw += self.sp["wz"] * self.C.control_dt
        d = (self.target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        self.target_yaw = yaw + float(np.clip(d, -0.5, 0.5))
        s = self.sp
        self.C.commands[:] = np.array([s["vx"], self.target_yaw, s["vy"], s["body_height"] - 0.75, s["torso_yaw"],
                                       s["pitch"], s["roll"], 0.0], np.float32)
        self.C.control_tick(q.astype(np.float32), dq.astype(np.float32), quat.astype(np.float32), gyro.astype(np.float32))

    def _apply_torques(self):
        from isaacsim.core.utils.types import ArticulationAction
        _, _, q, dq = self._state()
        tau = self.C.torque(q, dq) * self._variant["effort_scale"]
        tau = tau - FRICTIONLOSS * np.clip(dq / 0.05, -1.0, 1.0)
        self.last_tau = tau
        full = np.zeros(len(self._art.dof_names)); full[self.idx] = tau
        self._art.apply_action(ArticulationAction(joint_efforts=full))

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        if self.sub % self.C.sim_decimation == 0:
            self._policy_tick()
        self._apply_torques()
        self.sub += 1

    @property
    def tau_max(self):
        return self.AM.AMOController.torque_limits

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, cmd=None, **kw):
        """cmd: [vx, vy, vz, wz] and/or setpoint dict: vx, vy, wz, body_height (pelvis height, m), torso_yaw,
        pitch, roll (torso, rad). Clipped to RANGES (velocity also to max_cmd * cmd_scale), clips reported."""
        sp = {}
        if isinstance(cmd, dict):
            sp.update(cmd)
        elif cmd is not None:
            t = np.zeros(4); t[:len(cmd)] = np.asarray(cmd, float)[:4]
            sp.update(vx=t[0], vy=t[1], wz=t[3])
        sp.update(kw)
        L = self.limits()
        vlim = dict(vx=(-L["vx_rev"], L["vx_fwd"]), vy=(-L["vy"], L["vy"]), wz=(-L["wz"], L["wz"]))
        self.clipped_keys = []
        for k, v in sp.items():
            if k not in RANGES:
                raise KeyError(f"unknown setpoint {k!r}; have {sorted(RANGES)}")
            lo, hi = RANGES[k]
            if k in vlim:
                lo, hi = max(lo, vlim[k][0]), min(hi, vlim[k][1])
            c = float(np.clip(float(v), lo, hi))
            if abs(c - float(v)) > 1e-6:
                self.clipped_keys.append(k)
            self.sp[k] = c
        self.cmd_req = np.array([sp.get("vx", self.cmd_req[0]), sp.get("vy", self.cmd_req[1]), 0.0,
                                 sp.get("wz", self.cmd_req[3])], float)
        self.cmd = np.array([self.sp["vx"], self.sp["vy"], 0.0, self.sp["wz"]])
        self.clipped = bool(self.clipped_keys)
        return self.cmd

    def setpoint(self):
        return dict(self.sp)

    def stop(self):
        self.command(np.zeros(4))

    def reset_setpoints(self):
        self.sp = dict(DEFAULT_SP); self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        av = self._art._articulation_view
        if getattr(self, "_base_masses", None) is not None:
            m = self._base_masses.copy(); m[0] += float(payload_kg)
            av.set_body_masses(m.reshape(1, -1))
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + self.spec.spawn_dz]), np.array([q[3], q[0], q[1], q[2]]))
        self._art.set_linear_velocity(np.zeros(3)); self._art.set_angular_velocity(np.zeros(3))
        if getattr(self, "_q0", None) is not None:
            self._art.set_joint_positions(self._q0)
            self._art.set_joint_velocities(np.zeros(len(self._q0)))
            self._reset_policy()
        self.stop()

    def base_state(self):
        return self.RB.RobotAdapter.base_state(self)

    def fallen(self, bs=None):
        bs = bs or self.base_state()
        return bs is None or bs["tilt"] > self.spec.fall_tilt or bs["pos"][2] - self._floor_z < self.spec.min_z

    def _joint_limits(self):
        return self.RB.RobotAdapter._joint_limits(self)

    def proprio(self, bs=None):
        out = self.RB.RobotAdapter.proprio(self, bs)
        if out is not None:
            out[2]["setpoint"] = dict(self.sp); out[2]["clipped_keys"] = list(self.clipped_keys)
        return out

    def descriptor(self, stage=None):
        d = self.RB.RobotAdapter.descriptor(self, stage)
        d.update(controller="AMO (released)", setpoint_ranges={k: list(v) for k, v in RANGES.items()},
                 setpoints_supported=["vx", "vy (only while |vx|>=0.1)", "wz (only while |vx|>=0.1)", "body_height",
                                      "torso_yaw", "pitch", "roll"],
                 setpoints_unsupported=["gait"])
        return d

    def torso_rpy(self):
        """World roll/pitch/yaw of torso_link (what the torso commands act on)."""
        from pxr import UsdGeom
        import omni.usd
        st = omni.usd.get_context().get_stage()
        p = None
        for cand in (self.prim_path + "/torso_link", self.prim_path + "/pelvis/torso_link"):
            if st.GetPrimAtPath(cand).IsValid():
                p = st.GetPrimAtPath(cand); break
        if p is None:
            return None
        M = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
        q = M.ExtractRotationQuat()
        return self.AM.quatToEuler([q.GetReal(), *q.GetImaginary()])


def register_local_nav(kind):
    try:
        import local_nav
        local_nav.FOOT.setdefault(kind, (0.18, 0.22))
        local_nav.LASER_H.setdefault(kind, (0.15, 0.5, 0.9, 1.2))
    except Exception:
        pass


def make_g1a(kind, prim_path, position, yaw=0.0):
    register_local_nav(kind)
    return G1AMOAdapter(kind, prim_path, position, yaw)


try:
    make_g1a.SPEC = _spec()
except Exception:
    make_g1a.SPEC = None
