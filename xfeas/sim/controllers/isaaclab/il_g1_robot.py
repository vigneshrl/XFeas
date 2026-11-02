#!/usr/bin/env python
"""
Isaac Lab's RELEASED rough-terrain G1 policy (Isaac-Velocity-Rough-G1-v0, rsl_rl, Nucleus
PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/checkpoint.pt, iter 2999, md5 4aa61f37...) driving Isaac Lab's
OWN G1 asset (G1_MINIMAL_CFG: Robots/Unitree/G1/g1_minimal.usd, 37 DoF) inside our Isaac Sim 5.1 harness, for
scripts/terrain/run_terrain.py (--robot g1l) and episode_runner / gym_course_episode.py
(--robot g1l --robot-factory il_g1_robot:make_il_g1, scripts/isaaclab_humanoid on PYTHONPATH).

Reproduced from Isaac Lab 2.3.2 (velocity_env_cfg.py + config/g1/rough_env_cfg.py + isaaclab_assets unitree.py), nothing tuned:
  physics dt 0.005 s (200 Hz), decimation 4 -> policy 50 Hz; joint position targets held for the 4 substeps
  action: q_target = default_joint_pos + 0.5 * a (JointPositionActionCfg scale 0.5, use_default_offset), no clipping
     (agent clip_actions None); PD = PhysX implicit drives (ImplicitActuatorCfg) with Isaac Lab's gains:
     legs+torso Kp 150/150/200/200/200 (hip yaw/roll/pitch, knee, torso) Kd 5, effort 300; ankles Kp 20 Kd 2 effort 20;
     arms+hands Kp 40 Kd 10 effort 300; armature 0.01 (hands 0.001)
  articulation: self collisions OFF, solver iterations 8 pos / 4 vel; rigid bodies: max depenetration velocity 1.0,
     linear / angular damping 0, max lin / ang velocity 1000 (spawn cfg of G1_CFG)
  observation (310, order of ObservationsCfg.PolicyCfg, play mode = no noise):
     base_lin_vel (root COM velocity in the root frame, 3), base_ang_vel (3), projected_gravity (3),
     velocity_commands (vx, vy, wz, 3), joint_pos - default (37), joint_vel (37), last raw action (37),
     height_scan (187) = clip(z_sensor - z_hit - 0.5, -1, 1), sensor = torso_link, yaw-aligned 17 x 11 grid
     (1.6 x 1.0 m, 0.1 m, x inner / y outer = GridPatternCfg ordering "xy"), rays straight down.
     Isaac Lab casts from torso + 20 m against ONE static mesh (/World/ground). Here: PhysX scene-query raycast_all
     against every collider except the robot, starting at torso + RAY_START (2.0 m, so an indoor ROOF is not hit
     instead of the floor; any hit above torso + 0.5 m already clips to -1, so on roof-free ground the values are
     identical to the +20 m cast); miss -> -1 (Isaac Lab: inf hit -> clip -1).
  joint order = PhysX articulation order (= Isaac Lab's; checked against data/isaaclab_humanoid/ilref_plane_v0.5.json)
  commands: trained ranges vx [0, 1] m/s, vy 0 (lin_vel_y = (0, 0) in G1RoughEnvCfg), wz [-1, 1] rad/s.
     -> max_cmd (vx_fwd 1.0, vx_rev 0.0, vy 0.0, vz 0, wz 1.0): sidestep (vy) is NOT an executable action of this policy.
  velocity-only interface: no body height / pitch / roll / gait setpoints (spec.extra has no setpoint_ranges).
Policy = actor MLP 310-512-256-128-37 ELU, deterministic mean (rsl_rl act_inference), weights from the checkpoint.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os, sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
CKPT = os.environ.get(
    "IL_G1_CKPT", os.path.join(ROOT, "data", "isaaclab_ckpt", "Isaac-Velocity-Rough-G1-v0", "checkpoint.pt")
)
_G1_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/"
)
# IL_G1_ASSET=full -> Isaac Lab's G1_CFG asset g1.usd (same joints / masses, full body colliders) for scenes with obstacles
# above the ground (gym course: fences, handrails); default = g1_minimal.usd (G1_MINIMAL_CFG, what the policy was trained on)
ASSET = os.environ.get("IL_G1_ASSET", "minimal")
USD = _G1_DIR + ("g1.usd" if ASSET == "full" else "g1_minimal.usd")
PHYS_HZ, CTRL_HZ = 200, 50
SUB = PHYS_HZ // CTRL_HZ
SCALE = 0.5
RAY_START = float(os.environ.get("IL_G1_RAY_START", "2.0"))
# G1_CFG.init_state.joint_pos (regex -> value), everything else 0
INIT = {
    r".*_hip_pitch_joint": -0.20,
    r".*_knee_joint": 0.42,
    r".*_ankle_pitch_joint": -0.23,
    r".*_elbow_pitch_joint": 0.87,
    r"left_shoulder_roll_joint": 0.16,
    r"left_shoulder_pitch_joint": 0.35,
    r"right_shoulder_roll_joint": -0.16,
    r"right_shoulder_pitch_joint": 0.35,
    r"left_one_joint": 1.0,
    r"right_one_joint": -1.0,
    r"left_two_joint": 0.52,
    r"right_two_joint": -0.52,
}
# actuator groups (regex, Kp, Kd, effort, armature)
ACT = [
    (r".*_hip_yaw_joint", 150.0, 5.0, 300.0, 0.01),
    (r".*_hip_roll_joint", 150.0, 5.0, 300.0, 0.01),
    (r".*_hip_pitch_joint", 200.0, 5.0, 300.0, 0.01),
    (r".*_knee_joint", 200.0, 5.0, 300.0, 0.01),
    (r"torso_joint", 200.0, 5.0, 300.0, 0.01),
    (r".*_ankle_pitch_joint", 20.0, 2.0, 20.0, 0.01),
    (r".*_ankle_roll_joint", 20.0, 2.0, 20.0, 0.01),
    (r".*_(shoulder|elbow)_.*", 40.0, 10.0, 300.0, 0.01),
    (r".*_(zero|one|two|three|four|five|six)_joint", 40.0, 10.0, 300.0, 0.001),
]
KIND_CFG = dict(
    spawn_dz=0.74, fall_tilt=0.8, min_z=0.3, cam=(0.1, 0.55, -0.5), foot=(0.18, 0.22), laser=(0.15, 0.5, 0.9, 1.2)
)


def _spec(kind="g1l"):
    import robots as RB

    c = KIND_CFG
    return RB.RobotSpec(
        kind,
        "legged",
        (1.0, 0.0, 0.0, 0.0, 1.0),
        c["spawn_dz"],
        c["fall_tilt"],
        PHYS_HZ,
        usd=USD,
        foot_expr=".*ankle_roll_link",
        v_min=0.1,
        cam=c["cam"],
        min_z=c["min_z"],
        extra=dict(controller="Isaac Lab Isaac-Velocity-Rough-G1-v0 (released rsl_rl checkpoint)", height_scan=True),
    )


class ILPolicy:
    def __init__(self, path):
        import torch

        d = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        self.W = [
            (d[f"actor.{i}.weight"].numpy().astype(np.float64), d[f"actor.{i}.bias"].numpy().astype(np.float64))
            for i in (0, 2, 4, 6)
        ]

    def __call__(self, x):
        h = np.asarray(x, np.float64)
        for k, (W, b) in enumerate(self.W):
            h = W @ h + b
            if k < 3:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))  # ELU(alpha 1)
        return h


def grid_xy():
    """GridPatternCfg(resolution 0.1, size [1.6, 1.0]), ordering 'xy': x inner, y outer (torch.meshgrid indexing 'xy')"""
    x = np.arange(-0.8, 0.8 + 1e-9, 0.1)
    y = np.arange(-0.5, 0.5 + 1e-9, 0.1)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    return np.stack([gx.ravel(), gy.ravel()], 1)


class ILG1Adapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation
        import robots as RB

        self.RB = RB
        self.kind, self.prim_path = kind, prim_path
        self.spec = _spec(kind)
        self.pol = ILPolicy(CKPT)
        self.cmd = np.zeros(4)
        self.cmd_req = np.zeros(4)
        self.clipped = False
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(USD, prim_path)
        st = omni.usd.get_context().get_stage()
        root = None
        for p in Usd.PrimRange(st.GetPrimAtPath(prim_path)):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                root = str(p.GetPath())
                break
        self.links = {}
        for p in Usd.PrimRange(st.GetPrimAtPath(prim_path)):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                self.links.setdefault(p.GetName(), str(p.GetPath()))
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)  # G1_CFG.spawn.rigid_props
                rb.CreateMaxDepenetrationVelocityAttr().Set(1.0)
                rb.CreateLinearDampingAttr().Set(0.0)
                rb.CreateAngularDampingAttr().Set(0.0)
                rb.CreateMaxLinearVelocityAttr().Set(1000.0)
                rb.CreateMaxAngularVelocityAttr().Set(1000.0)  # deg/s, as Isaac Lab writes it
                rb.CreateRetainAccelerationsAttr().Set(False)
                rb.CreateDisableGravityAttr().Set(False)
        rp = st.GetPrimAtPath(root)
        ar = PhysxSchema.PhysxArticulationAPI.Apply(rp)  # G1_CFG.spawn.articulation_props
        ar.CreateEnabledSelfCollisionsAttr().Set(False)
        ar.CreateSolverPositionIterationCountAttr().Set(8)
        ar.CreateSolverVelocityIterationCountAttr().Set(4)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art = SingleArticulation(
            prim_path=root,
            name=kind,
            position=np.asarray(position, float),
            orientation=np.array([q[3], q[0], q[1], q[2]]),
        )
        self.robot = self._art
        self.root_path = root
        self.ctrl = None
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self.torso_path = self.links.get("torso_link")
        self.grid = grid_xy()
        self._pxq = None
        self._floor_z = 0.0
        n_col = sum(
            1
            for p in Usd.PrimRange(st.GetPrimAtPath(prim_path), Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)
        )
        print(
            f"[il_g1] {kind}: asset {os.path.basename(USD)}, root {root}, {len(self.links)} links, {n_col} colliders, torso {self.torso_path}",
            flush=True,
        )

    # ------------------------------------------------------------------ lifecycle
    def initialize(self):
        import re
        from omni.physx import get_physx_scene_query_interface

        self._pxq = get_physx_scene_query_interface()
        self._art.initialize()
        names = list(self._art.dof_names)
        n = len(names)
        self.names = names
        av = self._art._articulation_view
        kp, kd, eff, arm = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        q0 = np.zeros(n)
        for i, nm in enumerate(names):
            for rx, v in INIT.items():
                if re.fullmatch(rx, nm):
                    q0[i] = v
            hit = [g for g in ACT if re.fullmatch(g[0], nm)]
            assert hit, f"no actuator group for {nm}"
            _, kp[i], kd[i], eff[i], arm[i] = hit[0]
        av.set_gains(kps=kp.reshape(1, -1), kds=kd.reshape(1, -1))
        av.set_max_efforts(eff.reshape(1, -1))
        try:
            av.set_armatures(arm.reshape(1, -1))
        except Exception as e:
            print("[il_g1] set_armatures:", e, flush=True)
        self._effort_cap = eff.copy()
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        self.q0 = q0
        self._q0 = q0
        self.ctrl = SimpleNamespace(default_pos=q0)
        self.body_names = list(av.body_names)
        self.i_torso = self.body_names.index("torso_link")
        self.gains = dict(kp=kp.tolist(), kd=kd.tolist(), effort=eff.tolist(), armature=arm.tolist())
        print(
            f"[il_g1] dofs {n}: {names}\n[il_g1] mass {self._base_masses.sum():.2f} kg, q0 {np.round(q0, 2).tolist()}",
            flush=True,
        )
        self._reset_policy()
        self._art.set_joint_positions(q0)
        self.ready = True

    def _reset_policy(self):
        self.sub = 0
        self.action = np.zeros(37)
        self.q_tgt = self.q0.copy() if getattr(self, "q0", None) is not None else None
        self.n_ticks = 0
        self.last_obs = None
        self.last_scan = None

    # ------------------------------------------------------------------ sensing
    def _torso_pose(self):
        T = np.asarray(self._art._articulation_view._physics_view.get_link_transforms()).reshape(
            -1, len(self.body_names), 7
        )[0]
        return T[self.i_torso, :3].astype(float), T[self.i_torso, 3:7].astype(float)  # pos, quat xyzw

    def height_scan(self):
        import carb

        p, qx = self._torso_pose()
        yaw = Rotation.from_quat(qx).as_euler("xyz")[2]  # ray_alignment 'yaw' (quat_apply_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        P = self.grid @ np.array([[c, s], [-s, c]]) + p[:2]
        z0 = p[2] + RAY_START
        L = RAY_START + 30.0
        out = np.empty(len(P))
        rr = self.prim_path
        for k, (x, y) in enumerate(P):
            best = [None]

            def rep(h):
                if str(h.collision).startswith(rr):
                    return True
                d = float(h.distance)
                if best[0] is None or d < best[0]:
                    best[0] = d
                return True

            self._pxq.raycast_all(carb.Float3(float(x), float(y), float(z0)), carb.Float3(0.0, 0.0, -1.0), L, rep)
            out[k] = (p[2] - (z0 - best[0]) - 0.5) if best[0] is not None else -np.inf
        return np.clip(out, -1.0, 1.0)

    def observe(self):
        _, quat = self._art.get_world_pose()
        quat = np.asarray(quat, float)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        v_b = R.T @ np.asarray(self._art.get_linear_velocity(), float)
        w_b = R.T @ np.asarray(self._art.get_angular_velocity(), float)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        q = np.asarray(self._art.get_joint_positions(), float).reshape(-1)
        dq = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        hs = self.height_scan()
        self.last_scan = hs
        return np.concatenate(
            [v_b, w_b, g_b, [self.cmd[0], self.cmd[1], self.cmd[3]], q - self.q0, dq, self.action, hs]
        )

    def _policy_tick(self):
        o = self.observe()
        a = self.pol(o)
        self.last_obs = o
        self.action = a
        self.q_tgt = self.q0 + SCALE * a
        self.n_ticks += 1

    def _apply(self):
        from isaacsim.core.utils.types import ArticulationAction

        self._art.apply_action(ArticulationAction(joint_positions=self.q_tgt))

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        if self.sub % SUB == 0:
            self._policy_tick()
            self._apply()
        self.sub += 1

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, twist):
        if isinstance(twist, dict):  # posture setpoints are NOT part of this policy's interface
            twist = [twist.get("vx", self.cmd[0]), twist.get("vy", self.cmd[1]), 0.0, twist.get("wz", self.cmd[3])]
        return self.RB.RobotAdapter.command(self, twist)

    def setpoint(self):
        return dict(vx=float(self.cmd[0]), vy=float(self.cmd[1]), wz=float(self.cmd[3]))

    def reset_setpoints(self):
        self.command(np.zeros(4))

    def stop(self):
        self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + self.spec.spawn_dz]), np.array([q[3], q[0], q[1], q[2]]))
        self._art.set_linear_velocity(np.zeros(3))
        self._art.set_angular_velocity(np.zeros(3))
        if getattr(self, "q0", None) is not None:
            self._art.set_joint_positions(self.q0)
            self._art.set_joint_velocities(np.zeros(len(self.q0)))
            self._reset_policy()
            self._apply()
        self.stop()

    def base_state(self):
        return self.RB.RobotAdapter.base_state(self)

    def fallen(self, bs=None):
        bs = bs or self.base_state()
        return bs is None or bs["tilt"] > self.spec.fall_tilt or bs["pos"][2] - self._floor_z < self.spec.min_z

    def _joint_limits(self):
        return self.RB.RobotAdapter._joint_limits(self)

    def proprio(self, bs=None):
        return self.RB.RobotAdapter.proprio(self, bs)

    def descriptor(self, stage=None):
        d = self.RB.RobotAdapter.descriptor(self, stage)
        d.update(
            controller="Isaac Lab rough G1 (released)",
            setpoints_supported=["vx", "wz"],
            setpoints_unsupported=["vy", "body_height", "pitch", "roll", "gait"],
        )
        return d


def _register_local_nav(kind):
    try:
        import local_nav

        local_nav.FOOT.setdefault(kind, KIND_CFG["foot"])
        local_nav.LASER_H.setdefault(kind, KIND_CFG["laser"])
    except Exception:
        pass


def make_il_g1(kind, prim_path, position, yaw=0.0):
    _register_local_nav(kind)
    return ILG1Adapter(kind, prim_path, position, yaw)


try:
    make_il_g1.SPEC = _spec("g1l")
except Exception:
    make_il_g1.SPEC = None
