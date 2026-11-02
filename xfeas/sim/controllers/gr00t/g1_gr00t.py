#!/usr/bin/env python
"""
Unitree G1 (29-DoF) driven by the RELEASED GR00T Decoupled WBC (NVlabs/GR00T-WholeBodyControl @45c03fa,
decoupled_wbc; weights GR00T-WholeBodyControl-{Balance,Walk}.onnx, NVIDIA Open Model Licence, code Apache-2.0).
Commands: vx, vy, wz (body frame), ABSOLUTE pelvis height, torso roll / pitch / yaw.

    --robot g1g --robot-factory g1_gr00t:make_g1g          (scripts/gr00t on PYTHONPATH)

Everything is reproduced from the release's MuJoCo deployment (sim2mujoco/scripts/run_mujoco_gear_wbc.py +
g1_gear_wbc.yaml); the observation / history / Walk-Balance switch / PD code is gr00t_np.Gr00tController, the
SAME object the MuJoCo reference run (gr00t_mujoco_ref.py) uses. Numpy forward pass = ONNX within 2e-5.
  physics 200 Hz (dt 0.005), policy 50 Hz (decimation 4, first tick after the 4th physics step, as released)
  PD in effort mode every physics step, legs+waist Kp/Kd from the yaml, arms Kp 100 / Kd 0.5 to q = 0;
  torques clamped to the MJCF actuatorfrcrange (MuJoCo clamps the joint actuator force)
  passive joint terms of g1_gear_wbc.xml: damping 0.001 and armature 0.01 (imported), frictionloss 0.1 N*m
  (emulated: -0.1 * clip(dq / 0.05, -1, 1), as for AMO / URMA)
Robot model: the release's g1_gear_wbc.xml (floor removed) imported with Isaac 5.1's MJCF importer
(convert_gr00t_mjcf.py). Colliders are the MJCF's: convex hulls of every link mesh (pelvis contour, legs, torso + head,
arms, wrists, hands; shoulders are cylinders) + 4 r = 5 mm spheres per foot. Self-collision off (see notes).
Bridge setting (env GR00T_FOOT_COFF, default 0.02): contactOffset 2 cm / restOffset 0 on the foot spheres
(URMA lesson: small spheres tunnel through triangle-mesh floors); geometry unchanged.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

import gr00t_np as G

USD = os.environ.get("GR00T_USD", xfeas.paths.BASELINES + "/GR00T-WBC/usd/g1_gear_wbc.usd")
FRICTIONLOSS, FRICTION_V = 0.1, 0.05
# setpoint ranges: the training ranges are NOT released; these are the ranges the release's own MuJoCo run holds
# without falling (analysis/gr00t/gr00t_mujoco_ref.json): height 0.35-0.74 stand, torso pitch -0.2..0.8.
RANGES = dict(
    vx=(-0.8, 0.8),
    vy=(-0.5, 0.5),
    wz=(-1.0, 1.0),
    body_height=(0.30, 0.80),
    torso_yaw=(-1.0, 1.0),
    pitch=(-0.5, 1.0),
    roll=(-0.4, 0.4),
)
DEFAULT_SP = dict(vx=0.0, vy=0.0, wz=0.0, body_height=G.NOMINAL_HEIGHT, torso_yaw=0.0, pitch=0.0, roll=0.0)
KEY = os.environ.get("GR00T_KIND", "g1g")


def _spec():
    import robots as RB

    return RB.RobotSpec(
        KEY,
        "legged",
        (0.8, 0.8, 0.5, 0.0, 1.0),
        0.80,
        0.8,
        200,
        usd=USD,
        foot_expr="pelvis/.*_ankle_roll_link",
        v_min=0.1,
        cam=(0.1, 0.55, -0.5),
        min_z=0.2,
        extra=dict(
            controller="GR00T Decoupled WBC (released)",
            setpoint_ranges=RANGES,
            layer_box=dict(body_height=(0.35, 0.74), pitch=(-0.2, 0.8), roll=(-0.2, 0.2)),
        ),
    )


class G1Gr00tAdapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        import robots as RB

        self.RB = RB
        self.spec = make_g1g.SPEC
        self.kind, self.prim_path = kind, prim_path
        self.cmd = np.zeros(4)
        self.cmd_req = np.zeros(4)
        self.clipped = False
        self.clipped_keys = []
        self.sp = dict(DEFAULT_SP)
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(USD, prim_path)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art = SingleArticulation(
            prim_path=prim_path + "/pelvis/pelvis",
            name=kind,
            position=np.asarray(position, float),
            orientation=np.array([q[3], q[0], q[1], q[2]]),
        )
        self.robot = self._art
        self.ctrl = None
        st = omni.usd.get_context().get_stage()
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self.foot_coff = self._foot_contact_offset(st)
        self.C = G.Gr00tController(os.environ.get("GR00T_BACKEND", "numpy"))
        self.top_pts = np.load(os.path.join(G.NPZ_DIR, "torso_top_points.npz"))["pts"].astype(float)
        self._floor_z = 0.0
        from isaacsim.core.prims import RigidPrim

        self._torso_view = RigidPrim(
            prim_paths_expr=prim_path + "/pelvis/torso_link", name=kind + "_torso", reset_xform_properties=False
        )

    def _foot_contact_offset(self, st):
        coff = float(os.environ.get("GR00T_FOOT_COFF", "0.02"))
        if coff <= 0:
            return 0.0
        from pxr import Usd, UsdGeom, PhysxSchema

        n = 0
        for side in ("left", "right"):
            fp = f"{self.prim_path}/pelvis/{side}_ankle_roll_link"
            sph = [
                str(p.GetPath())
                for p in Usd.PrimRange(st.GetPrimAtPath(fp), Usd.TraverseInstanceProxies())
                if p.IsA(UsdGeom.Sphere)
            ]
            for sp in sph:  # colliders are instance proxies: un-instance their ancestors first
                q = st.GetPrimAtPath(sp)
                while q.GetPath().pathString != "/":
                    if q.IsInstanceable():
                        q.SetInstanceable(False)
                    q = q.GetParent()
            for sp in sph:
                pc = PhysxSchema.PhysxCollisionAPI.Apply(st.GetPrimAtPath(sp))
                pc.CreateContactOffsetAttr().Set(coff)
                pc.CreateRestOffsetAttr().Set(0.0)
                n += 1
        print(f"[g1g] foot spheres contactOffset {coff} restOffset 0 on {n} colliders", flush=True)
        return coff

    def initialize(self):
        self._art.initialize()
        names = list(self._art.dof_names)
        self.idx = np.array([names.index(n + "_joint") for n in G.JOINTS])
        n = len(names)
        av = self._art._articulation_view
        av.set_gains(kps=np.zeros((1, n)), kds=np.full((1, n), 0.001))  # MJCF passive damping 0.001 (implicit)
        cap = np.full(n, 1e6)
        cap[self.idx] = G.TAU_LIM
        av.set_max_efforts(cap.reshape(1, -1))
        self._effort_cap = cap
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        self._q0 = np.zeros(n)  # release start state: all joints 0 (mj_resetData)
        dpos = np.zeros(n)
        dpos[self.idx[:15]] = G.DEFAULT
        self.ctrl = SimpleNamespace(default_pos=dpos)
        self._torso_view.initialize()
        print(f"[g1g] dofs {n} mass {self._base_masses.sum():.3f} (MJCF 34.394)", flush=True)
        self._reset_policy()
        self.ready = True

    def _reset_policy(self):
        self.C.reset()
        self._push_cmd()
        self.sub = 0
        self.last_tau = np.zeros(29)

    def _push_cmd(self):
        s = self.sp
        self.C.set_command(
            vx=s["vx"],
            vy=s["vy"],
            wz=s["wz"],
            height=s["body_height"],
            roll=s["roll"],
            pitch=s["pitch"],
            yaw=s["torso_yaw"],
        )

    def _state(self):
        _, quat = self._art.get_world_pose()
        quat = np.asarray(quat, float)  # w, x, y, z
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        omega = R.T @ np.asarray(self._art.get_angular_velocity(), float)  # = MuJoCo free-joint qvel[3:6]
        q = np.asarray(self._art.get_joint_positions(), float).reshape(-1)[self.idx]
        dq = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)[self.idx]
        return q, dq, quat, omega

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        from isaacsim.core.utils.types import ArticulationAction

        q, dq, quat, omega = self._state()
        if self.sub > 0 and self.sub % G.DECIMATION == 0:  # release: counter += 1 after mj_step; policy if %4 == 0
            self.C.policy_tick(q, dq, quat, omega)
        tau = np.clip(self.C.torque(q, dq), -G.TAU_LIM, G.TAU_LIM) * self._variant["effort_scale"]
        self.last_tau = tau
        dq_all = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        full = -FRICTIONLOSS * np.clip(dq_all / FRICTION_V, -1.0, 1.0)
        full[self.idx] += tau
        self._art.apply_action(ArticulationAction(joint_efforts=full))
        self.sub += 1

    @property
    def tau_max(self):
        return G.TAU_LIM

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, cmd=None, **kw):
        """cmd: [vx, vy, vz, wz] and/or setpoint dict: vx, vy, wz, body_height (ABSOLUTE pelvis height, m), torso_yaw,
        pitch, roll (torso, rad). Clipped to RANGES (velocity also to max_cmd * cmd_scale); clips reported."""
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
            if k == "height":
                k = "body_height"
            if k not in RANGES:
                raise KeyError(f"unknown setpoint {k!r}; have {sorted(RANGES)}")
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
        self._push_cmd()
        return self.cmd

    def setpoint(self):
        return dict(self.sp)

    def stop(self):
        self.command(np.zeros(4))

    def reset_setpoints(self):
        self.sp = dict(DEFAULT_SP)
        self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        av = self._art._articulation_view
        if getattr(self, "_base_masses", None) is not None:
            m = self._base_masses.copy()
            m[0] += float(payload_kg)
            av.set_body_masses(m.reshape(1, -1))
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        """Release start state: pelvis 0.793 m above the floor, all joints 0, zero velocity, fresh policy history."""
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + 0.793]), np.array([q[3], q[0], q[1], q[2]]))
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
            controller="GR00T Decoupled WBC (released)",
            setpoint_ranges={k: list(v) for k, v in RANGES.items()},
            setpoints_supported=["vx", "vy", "wz", "body_height", "torso_yaw", "pitch", "roll"],
            setpoints_unsupported=["gait", "swing_height"],
        )
        return d

    def _torso_pose(self):
        P, Q = self._torso_view.get_world_poses()
        t = np.asarray(P, float).reshape(-1)[:3]
        q = np.asarray(Q, float).reshape(-1)[:4]  # w, x, y, z
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), t

    def torso_rpy(self):
        """World roll/pitch/yaw (intrinsic ZYX) of torso_link."""
        R, _ = self._torso_pose()
        pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        return np.array([math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])])

    def top_z(self):
        """max world z of the torso/head collision vertices (same point set as the MuJoCo reference)."""
        R, t = self._torso_pose()
        return float((self.top_pts @ R.T + t)[:, 2].max())


def register_local_nav(kind):
    try:
        import local_nav

        local_nav.FOOT.setdefault(kind, (0.18, 0.22))
        local_nav.LASER_H.setdefault(kind, (0.15, 0.5, 0.9, 1.2))
    except Exception:
        pass


def make_g1g(kind, prim_path, position, yaw=0.0):
    register_local_nav(kind)
    return G1Gr00tAdapter(kind, prim_path, position, yaw)


try:
    make_g1g.SPEC = _spec()
except Exception:
    make_g1g.SPEC = None
