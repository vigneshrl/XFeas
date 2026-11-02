#!/usr/bin/env python
"""
Boston Dynamics Spot (with arm) driven by the RELEASED ReLIC locomotion policy (RAI Institute, "Versatile
Loco-Manipulation through Flexible Interlimb Coordination", CoRL 2025; github.com/rai-opensource/relic,
source/relic/relic/assets/spot/pretrained/policy.pt -- byte-identical ONNX also shipped as judo's
spot_locomotion.onnx). Command interface = velocity + TORSO ROLL, PITCH, HEIGHT (+ arm joint targets).

    --robot spotp --robot-factory spot_posture:make_spotp        (with controllers/ on PYTHONPATH)

Reproduced from relic: interlimb_env_cfg.py (obs/actions/commands/sim), assets/spot/spot.py + constants.py
(actuators, default pose), actuators/actuator_spot.py (knee), and judo's deployment wrapper
judo/controller/batched_spot_locomotion.py (obs layout, joint-order maps). Nothing tuned:
  sim dt 0.005 (200 Hz), decimation 4 (50 Hz policy)
  obs (84) = base lin vel (body, 3) | base ang vel (body, 3) | projected gravity (3) | velocity cmd vx,vy,wz (3)
             | arm joint targets (7) | leg joint targets (12, zeros = leg not commanded) | torso roll, pitch, height (3)
             | joint pos - default (19, Isaac Lab order) | joint vel (19) | last raw action (12)
     no scaling, no clipping (policy.pt contains its own observation normaliser)
  action (12 legs, Isaac Lab order fl/fr/hl/hr_hx, .._hy, .._kn): q_target = 0.2 * a + default; arm q_target =
     arm command (passthrough, MixedPDArmMultiLegJointPositionAction)
  actuators, computed every physics step in EFFORT mode (Isaac Lab explicit actuators):
     hips (hx, hy): DelayedPD Kp 60, Kd 1.5, |tau| <= 45
     knees: Remotized PD Kp 60, Kd 1.5, |tau| <= angle-dependent limit (JOINT_PARAMETER_LOOKUP_TABLE col 2),
            then torque-speed limit (POS/NEG_TORQUE_SPEED_LIMIT)
     arm: DelayedPD Kp (120,120,120,100,100,100,16), Kd (2,2,2,2,2,2,0.32), effort limits, armature 0.01/0.001
     delay: training samples 0..3 physics steps per episode; a FIXED delay of DELAY steps is used here
  default pose: legs hx +-0.12, hy 0.5, kn -1.0; arm (0, -0.9, 1.8, 0, -0.9, 0, -1.54)
  TRAINED ranges (ArmLegJointBasePoseCommandCfg / UniformVelocityCommandCfg): roll, pitch +-0.35 rad,
     height 0.25..0.65 m (base z in world, flat ground at z=0 -> height above the floor), vx, vy, wz +-1.0
Robot model: relic's spot_with_arm.urdf imported with Isaac 5.1's URDF importer using the Isaac Lab spawn cfg
(merge_fixed_joints False, self-collisions on, link density 1e-8, 4/0 solver iterations).
Not reproduced: observation noise and domain randomisation (off at play), heading-controlled wz (wz is commanded
directly; the policy only ever sees the resulting wz command).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

RELIC = os.environ.get("RELIC_DIR", xfeas.paths.BASELINES + "/relic")
POLICY = os.path.join(RELIC, "source/relic/relic/assets/spot/pretrained/policy.pt")
SPOT_USD = os.path.join(RELIC, "usd/spot_arm_relic.usd")
ISAAC19 = [
    "arm_sh0",
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "arm_sh1",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "arm_el0",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]
LEGS12 = ["fl_hx", "fr_hx", "hl_hx", "hr_hx", "fl_hy", "fr_hy", "hl_hy", "hr_hy", "fl_kn", "fr_kn", "hl_kn", "hr_kn"]
ARM7 = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]
DEFAULT = dict(
    arm_sh0=0.0,
    arm_sh1=-0.9,
    arm_el0=1.8,
    arm_el1=0.0,
    arm_wr0=-0.9,
    arm_wr1=0.0,
    arm_f1x=-1.54,
    fl_hx=0.12,
    fr_hx=-0.12,
    hl_hx=0.12,
    hr_hx=-0.12,
    fl_hy=0.5,
    fr_hy=0.5,
    hl_hy=0.5,
    hr_hy=0.5,
    fl_kn=-1.0,
    fr_kn=-1.0,
    hl_kn=-1.0,
    hr_kn=-1.0,
)
ARM_KP = dict(zip(ARM7, (120.0, 120.0, 120.0, 100.0, 100.0, 100.0, 16.0)))
ARM_KD = dict(zip(ARM7, (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.32)))
ARM_TAU = dict(zip(ARM7, (90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32)))
ARM_ARMATURE = dict(zip(ARM7, (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001)))
LEG_KP, LEG_KD, HIP_TAU = 60.0, 1.5, 45.0
ACTION_SCALE, DECIM, SIM_DT = 0.2, 4, 0.005
DELAY = int(os.environ.get("SPOTP_DELAY", "2"))  # physics steps; training: uniform 0..3 per episode
POS_TS = np.array([[-30.0, 96.9972], [0.0, 96.9972], [14.0, 0.0]])
NEG_TS = np.array([[-15.0, 0.0], [0.0, -96.9972], [30.0, -96.9972]])
NOMINAL_H = 0.52  # judo STANDING_HEIGHT_CMD
RANGES = dict(
    vx=(-1.0, 1.0), vy=(-1.0, 1.0), wz=(-1.0, 1.0), body_height=(0.25, 0.65), pitch=(-0.35, 0.35), roll=(-0.35, 0.35)
)
DEFAULT_SP = dict(vx=0.0, vy=0.0, wz=0.0, body_height=NOMINAL_H, pitch=0.0, roll=0.0)


def _knee_table():
    from importlib.util import spec_from_file_location, module_from_spec

    sp = spec_from_file_location(
        "relic_spot_constants", os.path.join(RELIC, "source/relic/relic/assets/spot/constants.py")
    )
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    t = np.array(m.JOINT_PARAMETER_LOOKUP_TABLE)
    return t[:, 0], t[:, 2]


def _spec():
    import robots as RB

    return RB.RobotSpec(
        "spotp",
        "legged",
        (1.0, 1.0, 1.0, 0.0, 1.0),
        0.62,
        0.8,
        200,
        usd=SPOT_USD,
        foot_expr=".*_foot",
        v_min=0.1,
        cam=(0.52, -0.02, -0.45),
        min_z=0.15,
        extra=dict(controller="ReLIC pretrained (released)", setpoint_ranges=RANGES),
    )


class SpotPostureAdapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        import torch
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        import robots as RB

        self.RB, self.torch = RB, torch
        self.spec = make_spotp.SPEC
        self.kind, self.prim_path = kind, prim_path
        self.cmd = np.zeros(4)
        self.cmd_req = np.zeros(4)
        self.clipped = False
        self.clipped_keys = []
        self.sp = dict(DEFAULT_SP)
        self.arm = {k: DEFAULT[k] for k in ARM7}
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(SPOT_USD, prim_path)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art = SingleArticulation(
            prim_path=prim_path,
            name=kind,
            position=np.asarray(position, float),
            orientation=np.array([q[3], q[0], q[1], q[2]]),
        )
        self.robot = self._art
        self.ctrl = None
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self._solver_iters(prim_path)
        self.policy = torch.jit.load(POLICY, map_location="cpu").eval()
        self.kn_q, self.kn_tau = _knee_table()
        self._floor_z = 0.0

    @staticmethod
    def _solver_iters(prim_path):
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema

        st = omni.usd.get_context().get_stage()
        for p in Usd.PrimRange(st.GetPrimAtPath(prim_path)):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                a = PhysxSchema.PhysxArticulationAPI.Apply(p)
                a.CreateSolverPositionIterationCountAttr().Set(4)
                a.CreateSolverVelocityIterationCountAttr().Set(0)
                a.CreateEnabledSelfCollisionsAttr().Set(True)

    # ------------------------------------------------------------------ lifecycle
    def initialize(self):
        self._art.initialize()
        names = list(self._art.dof_names)
        self.i19 = np.array([names.index(n) for n in ISAAC19])
        self.i12 = np.array([names.index(n) for n in LEGS12])
        self.iarm = np.array([names.index(n) for n in ARM7])
        n = len(names)
        av = self._art._articulation_view
        try:
            av.set_gains(kps=np.zeros((1, n)), kds=np.zeros((1, n)))
        except Exception:
            pass
        arm_ = np.zeros(n)
        for k in ARM7:
            arm_[names.index(k)] = ARM_ARMATURE[k]
        try:
            av.set_armatures(arm_.reshape(1, -1))
        except Exception as e:
            print("[spotp] armature not set:", e, flush=True)
        self.kp = np.zeros(n)
        self.kd = np.zeros(n)
        self.tau_max = np.full(n, np.inf)
        for i, nm in enumerate(names):
            if nm in ARM_KP:
                self.kp[i], self.kd[i], self.tau_max[i] = ARM_KP[nm], ARM_KD[nm], ARM_TAU[nm]
            elif nm.endswith(("_hx", "_hy")):
                self.kp[i], self.kd[i], self.tau_max[i] = LEG_KP, LEG_KD, HIP_TAU
            elif nm.endswith("_kn"):
                self.kp[i], self.kd[i], self.tau_max[i] = LEG_KP, LEG_KD, 115.0  # nominal; real limit per angle
        self.knee = np.array([nm.endswith("_kn") for nm in names])
        av.set_max_efforts(np.where(np.isfinite(self.tau_max), self.tau_max, 1e6).reshape(1, -1))
        self._effort_cap = self.tau_max.copy()
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        self.default_full = np.array([DEFAULT[nm] for nm in names])
        self._q0 = self.default_full.copy()
        self.ctrl = SimpleNamespace(default_pos=self._q0.copy())
        self.d19 = np.array([DEFAULT[k] for k in ISAAC19])
        self.d12 = np.array([DEFAULT[k] for k in LEGS12])
        print(
            "[spotp] dofs:",
            names,
            "\n[spotp] body masses:",
            np.round(self._base_masses, 3).tolist(),
            "total",
            round(float(self._base_masses.sum()), 2),
            flush=True,
        )
        self._reset_policy()
        self.ready = True

    def _reset_policy(self):
        self.last_action = np.zeros(12)
        self.target = self._q0.copy()
        self.tbuf = [self._q0.copy() for _ in range(DELAY + 1)]
        self.sub = 0

    # ------------------------------------------------------------------ released observation / policy
    def _cmd25(self):
        s = self.sp
        return np.concatenate(
            [
                [s["vx"], s["vy"], s["wz"]],
                [self.arm[k] for k in ARM7],
                np.zeros(12),
                [s["roll"], s["pitch"], s["body_height"]],
            ]
        )

    def _obs(self):
        _, quat = self._art.get_world_pose()
        qw, qx, qy, qz = [float(v) for v in quat]
        RT = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
        v_b = RT @ np.asarray(self._art.get_linear_velocity(), float)
        w_b = RT @ np.asarray(self._art.get_angular_velocity(), float)
        g_b = RT @ np.array([0.0, 0.0, -1.0])
        q = np.asarray(self._art.get_joint_positions(), float).reshape(-1)
        dq = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        return np.concatenate(
            [v_b, w_b, g_b, self._cmd25(), q[self.i19] - self.d19, dq[self.i19], self.last_action]
        ).astype(np.float32)

    def _policy_tick(self):
        with self.torch.no_grad():
            a = self.policy(self.torch.from_numpy(self._obs()[None])).numpy()[0].astype(float)
        self.last_action = a
        t = self.target.copy()
        t[self.i12] = a * ACTION_SCALE + self.d12
        t[self.iarm] = [self.arm[k] for k in ARM7]
        self.target = t

    def _apply_torques(self):
        from isaacsim.core.utils.types import ArticulationAction

        self.tbuf = self.tbuf[1:] + [self.target.copy()]
        tgt = self.tbuf[0]  # target from DELAY physics steps ago
        q = np.asarray(self._art.get_joint_positions(), float).reshape(-1)
        dq = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        tau = self.kp * (tgt - q) - self.kd * dq
        es = self._variant["effort_scale"]
        lim = np.where(self.knee, np.interp(q, self.kn_q, self.kn_tau), self.tau_max) * es
        tau = np.clip(tau, -lim, lim)
        k = self.knee  # knee torque-speed limit
        tau[k] = np.clip(
            tau[k], np.interp(dq[k], NEG_TS[:, 0], NEG_TS[:, 1]) * es, np.interp(dq[k], POS_TS[:, 0], POS_TS[:, 1]) * es
        )
        self.last_tau = tau
        self._art.apply_action(ArticulationAction(joint_efforts=tau))

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        if self.sub % DECIM == 0:
            self._policy_tick()
        self._apply_torques()
        self.sub += 1

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, cmd=None, **kw):
        """cmd: [vx, vy, vz, wz] and/or a setpoint dict with vx, vy, wz, body_height (base height above the floor,
        m), pitch, roll (rad), arm ({joint: rad}). Clipped to the trained RANGES (velocity also to max_cmd *
        cmd_scale); clips reported in self.clipped_keys."""
        sp = {}
        if isinstance(cmd, dict):
            sp.update(cmd)
        elif cmd is not None:
            t = np.zeros(4)
            t[: len(cmd)] = np.asarray(cmd, float)[:4]
            sp.update(vx=t[0], vy=t[1], wz=t[3])
        sp.update(kw)
        L = self.limits()
        vlim = dict(vx=(-L["vx_rev"], L["vx_fwd"]), vy=(-L["vy"], L["vy"]), wz=(-L["wz"], L["wz"]))
        self.clipped_keys = []
        for k, v in sp.items():
            if k == "arm":
                for jn, a in v.items():
                    self.arm[jn] = float(a)
                continue
            if k not in RANGES:
                raise KeyError(f"unknown setpoint {k!r}; have {sorted(RANGES) + ['arm']}")
            lo, hi = RANGES[k]
            if k in vlim:
                lo, hi = max(lo, vlim[k][0]), min(hi, vlim[k][1])
            c = float(np.clip(float(v), lo, hi))
            if abs(c - float(v)) > 1e-6:
                self.clipped_keys.append(k)
            self.sp[k] = c
        self.cmd_req = np.array(
            [sp.get("vx", self.cmd_req[0]), sp.get("vy", self.cmd_req[1]), 0.0, sp.get("wz", self.cmd_req[3])], float
        )
        self.cmd = np.array([self.sp["vx"], self.sp["vy"], 0.0, self.sp["wz"]])
        self.clipped = bool(self.clipped_keys)
        return self.cmd

    def setpoint(self):
        return dict(self.sp)

    def stop(self):
        self.command(np.zeros(4))

    def reset_setpoints(self):
        self.sp = dict(DEFAULT_SP)
        self.arm = {k: DEFAULT[k] for k in ARM7}
        self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        av = self._art._articulation_view
        if getattr(self, "_base_masses", None) is not None:
            m = self._base_masses.copy()
            m[0] += float(payload_kg)
            av.set_body_masses(m.reshape(1, -1))
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + self.spec.spawn_dz]), np.array([q[3], q[0], q[1], q[2]]))
        self._art.set_linear_velocity(np.zeros(3))
        self._art.set_angular_velocity(np.zeros(3))
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
            out[2]["setpoint"] = dict(self.sp)
            out[2]["clipped_keys"] = list(self.clipped_keys)
        return out

    def descriptor(self, stage=None):
        d = self.RB.RobotAdapter.descriptor(self, stage)
        d.update(
            controller="ReLIC Spot pretrained (released)",
            setpoint_ranges={k: list(v) for k, v in RANGES.items()},
            setpoints_supported=["vx", "vy", "wz", "body_height", "pitch", "roll", "arm"],
            setpoints_unsupported=["gait type", "gait frequency", "swing height"],
        )
        return d


def register_local_nav(kind):
    try:
        import local_nav

        local_nav.FOOT.setdefault(kind, local_nav.FOOT["spot"])
        local_nav.LASER_H.setdefault(kind, local_nav.LASER_H["spot"])
    except Exception:
        pass


def make_spotp(kind, prim_path, position, yaw=0.0):
    register_local_nav(kind)
    return SpotPostureAdapter(kind, prim_path, position, yaw)


try:
    make_spotp.SPEC = _spec()
except Exception:
    make_spotp.SPEC = None
