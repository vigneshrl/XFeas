#!/usr/bin/env python
"""
Unitree G1 (29-DoF + Dex3 hands) driven by the RELEASED GEAR-SONIC stack (NVlabs/GR00T-WholeBodyControl gear_sonic_deploy,
HF nvidia/GEAR-SONIC default variant + planner_sonic.onnx): kinematic planner (modes) -> universal-token encoder ->
decoder -> joint targets.  The controller runs in a SEPARATE process (sonic_server.py in the gr00t_wbc venv, which has
onnxruntime; Isaac kit python does not) and is called in LOCKSTEP once per 50 Hz tick over a stdin/stdout pipe
(sonic_client.SonicClient). The planner runs in the server's background thread (as the C++ planner thread does);
env SONIC_PLANNER_ASYNC=0 makes it synchronous (deterministic, = MuJoCo reference).

    --robot g1s --robot-factory g1_sonic:make_g1s          (scripts/sonic on PYTHONPATH)

Reproduces the release sim2sim physics (gear_sonic/utils/mujoco_sim/base_sim.py + wbc yaml) as sonic_mujoco_ref.py does:
  physics 200 Hz, control 50 Hz (policy tick before the 4 physics sub-steps of each tick)
  PD in effort mode every physics step: tau = Kp (q* - q) - Kd dq with the deploy Kp/Kd (sonic_ref.KPS/KDS), Dex3 hands
  held at the release default hand pose with kp 1.5 / kd 0.1; torques clamped to motor_effort_limit_list (= MJCF
  actuatorfrcrange for the body)
  passive joint terms of g1_29dof_with_hand.xml: armature 0.01 (imported), damping 0.05 (implicit drive damping),
  frictionloss 0.2 / 0.1 (wrist pitch+yaw, fingers) N*m emulated as -fl * clip(dq / 0.05, -1, 1)
Robot model: the release MJCF imported with Isaac 5.1's MJCF importer (convert_sonic_mjcf.py): convex hulls of the link
meshes, box feet, cylinder shoulders; self-collision: see SONIC_SELF_COLLISION / report. Bridge setting SONIC_FOOT_COFF (default 0.02): contactOffset
2 cm / restOffset 0 on the two foot boxes (same harness convention as g1g).

Command mapping (command(dict)) -- release keyboard / ZMQ-manager semantics:
  mode     : string (idle, slowWalk, walk, run, squat, kneelTwoLeg, kneelOneLeg, handCrawl, elbowCrawl, crouchWalk,
             stealth, injured, careful, ... or sonic_ref.MODES names) or int planner mode. If absent: nominal
             "walk at v" mapping on |v| = hypot(vx, vy): 0 -> idle, (0, 0.8] -> slowWalk (speed clamped to >= 0.2,
             keyboard range 0.2-0.8), (0.8, 2.5] -> walk, > 2.5 -> run (clamped to 3.0).
  vx, vy   : body-frame velocity (m/s) w.r.t. the commanded facing: planner movement_direction = R(facing) [vx, vy]/|v|,
             target_vel = |v| (zero |v| -> movement 0, speed -1 in static modes else 0, as the keyboard writes).
             Clamped to the keyboard speed range of the mode (run 1.5-3.0, handCrawl 0.4-1.0, elbowCrawl 0.7-1.0).
  wz       : yaw rate (rad/s): integrated into the planner facing angle every 20 ms (keyboard Q/E is the same
             facing change in steps of pi/6); with |v| = 0 and no mode this is IDLE + facing change = in-place turn.
  speed    : optional explicit planner target_vel (m/s; -1 = the mode's default speed), overrides |v|.
  body_height (alias height): planner `height` for squat/kneel/lying/crawl modes, clamped to 0.2-0.8 (keyboard);
             ignored (-1) in standing modes.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from sonic_ref import (KPS, KDS, DEFAULT_ANGLES, DEX3_LEFT_DEFAULT, DEX3_RIGHT_DEFAULT, DEX3_KP, DEX3_KD, MODES,
                       MJ_JOINT_NAMES, STATIC_MODES, STANDING_MODES)
import sonic_plan as PL

USD = os.environ.get("SONIC_USD", xfeas_paths.BASELINES + "/GEAR-SONIC/usd/g1_sonic.usd")
TOP_NPZ = xfeas_paths.BASELINES + "/GEAR-SONIC/usd/torso_top_points.npz"
KEY = os.environ.get("SONIC_KIND", "g1s")
DECIM = 4
FRICTION_V = 0.05
HAND_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
# motor_effort_limit_list of wbc_configs/g1_29dof_sonic_model12.yaml, body part (= MJCF actuatorfrcrange)
TAU_BODY = np.array([88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50,
                     25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], float)
TAU_HAND = np.array([2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
ALIASES = {"slowwalk": "slow_walk", "kneeltwoleg": "kneel_two_legs", "kneeltwolegs": "kneel_two_legs",
           "kneelone": "kneel_one_leg", "kneeloneleg": "kneel_one_leg", "lyingfacedown": "lying_face_down",
           "handcrawl": "hand_crawl", "handcrawling": "hand_crawl", "elbowcrawl": "elbow_crawl",
           "elbowcrawling": "elbow_crawl", "crouchwalk": "crouch_walk", "crouch": "crouch_walk",
           "idleboxing": "idle_boxing", "walkboxing": "walk_boxing", "objectcarrying": "object_carrying",
           "happydance": "happy_dance"}
SPEED_RANGE = {MODES["slow_walk"]: (0.2, 0.8), MODES["run"]: (1.5, 3.0), MODES["hand_crawl"]: (0.4, 1.0),
               MODES["elbow_crawl"]: (0.7, 1.0)}
for m in (10, 11, 12, 13, 15, 16):
    SPEED_RANGE[m] = (0.7, 1.5)
LOW_MODES = {MODES["hand_crawl"], MODES["elbow_crawl"], MODES["lying_face_down"]}
RANGES = dict(vx=(-1.0, 3.0), vy=(-0.6, 0.6), wz=(-1.0, 1.0), body_height=(0.2, 0.8))
DEFAULT_SP = dict(vx=0.0, vy=0.0, wz=0.0, body_height=-1.0, mode=None, speed=None)


def mode_id(mode):
    if mode is None:
        return None
    if isinstance(mode, (int, np.integer)):
        return int(mode)
    k = str(mode).strip()
    k2 = ALIASES.get(k.lower().replace("_", ""), k.lower())
    if k2 in MODES:
        return MODES[k2]
    raise KeyError(f"unknown SONIC mode {mode!r}; have {sorted(MODES)}")


def _spec():
    import robots as RB
    return RB.RobotSpec(KEY, "legged", (3.0, 1.0, 0.6, 0.0, 1.0), PL.START_PELVIS_Z, 0.8, 200, usd=USD,
                        foot_expr="pelvis/.*_ankle_roll_link", v_min=0.2, cam=(0.1, 0.55, -0.5), min_z=0.2,
                        extra=dict(controller="GEAR-SONIC planner + universal-token tracker (released)",
                                   setpoint_ranges=RANGES, modes=sorted(MODES, key=MODES.get)))


class G1SonicAdapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        import robots as RB
        self.RB = RB
        self.spec = make_g1s.SPEC
        self.kind, self.prim_path = kind, prim_path
        self.cmd = np.zeros(4); self.cmd_req = np.zeros(4); self.clipped = False; self.clipped_keys = []
        self.sp = dict(DEFAULT_SP)
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(USD, prim_path)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art = SingleArticulation(prim_path=prim_path + "/pelvis/pelvis", name=kind, position=np.asarray(position, float),
                                       orientation=np.array([q[3], q[0], q[1], q[2]]))
        self.robot = self._art
        self.ctrl = None
        st = omni.usd.get_context().get_stage()
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self.foot_coff = self._foot_contact_offset(st)
        # MuJoCo collides every non parent-child body pair of the release MJCF (contype/conaffinity 1): in the reference
        # runs the Dex3 thumbs touch the hip_roll links during every segment and the hands touch each other in elbow
        # crawl. PhysX articulation self-collision (adjacent links filtered, like MuJoCo) matches that; SONIC_SELF_COLLISION=0
        # turns it off (the convert default).
        self.self_collision = os.environ.get("SONIC_SELF_COLLISION", "1") != "0"
        from pxr import PhysxSchema
        root = st.GetPrimAtPath(prim_path + "/pelvis/pelvis")
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr().Set(bool(self.self_collision))
        self.top_pts = np.load(TOP_NPZ)["pts"].astype(float)
        self._floor_z = 0.0
        self._torso_view = RigidPrim(prim_paths_expr=prim_path + "/pelvis/torso_link", name=kind + "_torso",
                                     reset_xform_properties=False)
        from sonic_client import SonicClient
        self.async_planner = os.environ.get("SONIC_PLANNER_ASYNC", "1") != "0"
        self.client = SonicClient(planner_async=self.async_planner, threads=int(os.environ.get("SONIC_THREADS", "4")))
        self.tick_ms = []
        self.hold = PL.HOLD_STEPS
        self.q_target = DEFAULT_ANGLES.copy()
        self.mode_now = 0
        self.last_info = {}

    def _foot_contact_offset(self, st):
        coff = float(os.environ.get("SONIC_FOOT_COFF", "0.02"))
        if coff <= 0:
            return 0.0
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        n = 0
        for side in ("left", "right"):
            fp = f"{self.prim_path}/pelvis/{side}_ankle_roll_link"
            cols = [str(p.GetPath()) for p in Usd.PrimRange(st.GetPrimAtPath(fp), Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.CollisionAPI)]
            for cp in cols:                      # colliders may be instance proxies: un-instance ancestors first
                q = st.GetPrimAtPath(cp)
                while q.GetPath().pathString != "/":
                    if q.IsInstanceable():
                        q.SetInstanceable(False)
                    q = q.GetParent()
            for cp in cols:
                pc = PhysxSchema.PhysxCollisionAPI.Apply(st.GetPrimAtPath(cp))
                pc.CreateContactOffsetAttr().Set(coff); pc.CreateRestOffsetAttr().Set(0.0)
                n += 1
        print(f"[g1s] foot boxes contactOffset {coff} restOffset 0 on {n} colliders", flush=True)
        return coff

    def initialize(self):
        self._art.initialize()
        names = list(self._art.dof_names)
        self.idx = np.array([names.index(n + "_joint") for n in MJ_JOINT_NAMES])
        self.lh = np.array([names.index(f"left_hand_{h}_joint") for h in HAND_NAMES])
        self.rh = np.array([names.index(f"right_hand_{h}_joint") for h in HAND_NAMES])
        n = len(names)
        av = self._art._articulation_view
        av.set_gains(kps=np.zeros((1, n)), kds=np.full((1, n), 0.05))       # MJCF passive damping 0.05 (implicit)
        cap = np.full(n, 1e6); cap[self.idx] = TAU_BODY; cap[self.lh] = TAU_HAND; cap[self.rh] = TAU_HAND
        av.set_max_efforts(cap.reshape(1, -1))
        self._effort_cap = cap
        self.fl = np.array([0.1 if ("wrist_pitch" in nm or "wrist_yaw" in nm or "hand" in nm) else 0.2 for nm in names])
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        self._q0 = np.zeros(n); self._q0[self.idx] = DEFAULT_ANGLES      # start: default pose, hands 0
        self.ctrl = SimpleNamespace(default_pos=self._q0.copy())
        self._torso_view.initialize()
        print(f"[g1s] dofs {n} mass {self._base_masses.sum():.3f} (MJCF 36.165) planner_async={self.async_planner} "
              f"self_collision={self.self_collision}", flush=True)
        self.hold = PL.HOLD_STEPS
        self.ready = True

    def _state(self):
        _, quat = self._art.get_world_pose()
        quat = np.asarray(quat, float)                    # w, x, y, z
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        omega = R.T @ np.asarray(self._art.get_angular_velocity(), float)       # = MuJoCo free-joint qvel[3:6]
        qa = np.asarray(self._art.get_joint_positions(), float).reshape(-1)
        dqa = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        return qa, dqa, quat, omega

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        from isaacsim.core.utils.types import ArticulationAction
        qa, dqa, quat, omega = self._state()
        q, dq = qa[self.idx], dqa[self.idx]
        if self.hold > 0:                       # WAIT_FOR_CONTROL: default-pose PD hold, then controller reset
            self.q_target = DEFAULT_ANGLES.copy()
            self.hold -= 1
        else:
            if self.sub == 0 and not self._started:
                self.client.reset(q, quat)
                self._started = True
                self._push_cmd()
            if self.sub % DECIM == 0:
                t = time.perf_counter()
                self.q_target, self.last_info = self.client.tick(q, dq, quat, omega)
                self.tick_ms.append((time.perf_counter() - t) * 1e3)
            self.sub += 1
        tau = np.clip(KPS * (self.q_target - q) - KDS * dq, -TAU_BODY, TAU_BODY) * self._variant["effort_scale"]
        th_l = np.clip(DEX3_KP * (DEX3_LEFT_DEFAULT - qa[self.lh]) - DEX3_KD * dqa[self.lh], -TAU_HAND, TAU_HAND)
        th_r = np.clip(DEX3_KP * (DEX3_RIGHT_DEFAULT - qa[self.rh]) - DEX3_KD * dqa[self.rh], -TAU_HAND, TAU_HAND)
        self.last_tau = tau
        full = -self.fl * np.clip(dqa / FRICTION_V, -1.0, 1.0)
        full[self.idx] += tau; full[self.lh] += th_l; full[self.rh] += th_r
        self._art.apply_action(ArticulationAction(joint_efforts=full))

    @property
    def tau_max(self):
        return TAU_BODY

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def _resolve(self):
        """setpoints -> (mode, vx, vy, wz, speed, height) for SonicController.set_command; records clips."""
        s = self.sp
        v = math.hypot(s["vx"], s["vy"])
        m = mode_id(s["mode"])
        if m is None:                                   # nominal "walk at v"
            m = 0 if v < 1e-6 else (MODES["slow_walk"] if v <= 0.8 else (MODES["walk"] if v <= 2.5 else MODES["run"]))
        speed = None
        if m in STATIC_MODES:
            vx = vy = 0.0
        else:
            vx, vy = s["vx"], s["vy"]
            if v > 1e-6:
                lo, hi = SPEED_RANGE.get(m, (0.0, 3.0))
                sp = min(max(v, lo), hi)
                if abs(sp - v) > 1e-6:
                    self.clipped_keys.append("speed")
                speed = sp
                if s.get("speed") is not None:          # explicit target_vel overrides |v| (direction still from vx, vy)
                    speed = s["speed"]
        h = -1.0 if m in STANDING_MODES else (s["body_height"] if s["body_height"] > 0 else 0.8)
        self.mode_now = m
        return dict(mode=int(m), vx=float(vx), vy=float(vy), wz=float(s["wz"]) if s["wz"] else None, heading=None,
                    height=float(h), speed=speed)

    def _push_cmd(self):
        if not getattr(self, "_started", False):
            return                                      # pushed at controller reset
        c = self._resolve()
        self.client.cmd(c["mode"], c["vx"], c["vy"], c["wz"], c["heading"], c["height"], c["speed"])
        self.applied = c

    def command(self, cmd=None, **kw):
        """cmd: [vx, vy, vz, wz] and/or dict with vx, vy, wz (body frame), mode (str|int), body_height|height (m)."""
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
            if k == "height":
                k = "body_height"
            if k == "mode":
                mode_id(v); self.sp["mode"] = v; continue
            if k == "speed":                            # optional explicit planner target_vel (-1 = mode default)
                self.sp["speed"] = None if v is None else float(v); continue
            if k not in RANGES:
                raise KeyError(f"unknown setpoint {k!r}; have {sorted(RANGES) + ['mode']}")
            if k == "body_height" and float(v) < 0:
                self.sp[k] = -1.0; continue
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
        self._push_cmd()
        self.clipped = bool(self.clipped_keys)
        return self.cmd

    def setpoint(self):
        return dict(self.sp, resolved=getattr(self, "applied", None))

    def stop(self):
        self.command(dict(vx=0.0, vy=0.0, wz=0.0))

    def reset_setpoints(self):
        self.sp = dict(DEFAULT_SP); self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        av = self._art._articulation_view
        if getattr(self, "_base_masses", None) is not None:
            m = self._base_masses.copy(); m[0] += float(payload_kg)
            av.set_body_masses(m.reshape(1, -1))
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        """Start state of sonic_mujoco_ref: default pose (hands 0), soles on the floor, zero velocity; then 0.1 s
        default-pose PD hold and a fresh controller (planner re-initialised from the measured joints, IDLE)."""
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + PL.START_PELVIS_Z]), np.array([q[3], q[0], q[1], q[2]]))
        self._art.set_linear_velocity(np.zeros(3)); self._art.set_angular_velocity(np.zeros(3))
        if getattr(self, "_q0", None) is not None:
            self._art.set_joint_positions(self._q0)
            self._art.set_joint_velocities(np.zeros(len(self._q0)))
        self.hold = PL.HOLD_STEPS; self.sub = 0; self._started = False
        self.q_target = DEFAULT_ANGLES.copy()

    sub = 0
    _started = False

    def base_state(self):
        return self.RB.RobotAdapter.base_state(self)

    def fallen(self, bs=None):
        """Mode-aware: crawling / lying modes are SUPPOSED to be low and pitched, so for them only an upside-down torso
        or the torso/head top below 0.12 m counts (sonic_plan crawl criterion); otherwise tilt > fall_tilt or pelvis
        below min_z."""
        bs = bs or self.base_state()
        if bs is None:
            return True
        if self.mode_now in LOW_MODES:
            R, _ = self._torso_pose()
            return R[2, 2] < -0.2 or self.top_z() - self._floor_z < 0.12
        return bs["tilt"] > self.spec.fall_tilt or bs["pos"][2] - self._floor_z < self.spec.min_z

    def _joint_limits(self):
        return self.RB.RobotAdapter._joint_limits(self)

    def proprio(self, bs=None):
        out = self.RB.RobotAdapter.proprio(self, bs)
        if out is not None:
            out[2]["setpoint"] = self.setpoint(); out[2]["clipped_keys"] = list(self.clipped_keys)
        return out

    def descriptor(self, stage=None):
        d = self.RB.RobotAdapter.descriptor(self, stage)
        d.update(controller="GEAR-SONIC (released, planner-driven)", setpoint_ranges={k: list(v) for k, v in RANGES.items()},
                 setpoints_supported=["vx", "vy", "wz", "mode", "body_height"],
                 modes=sorted(MODES, key=MODES.get), mode_speed_ranges={str(k): list(v) for k, v in SPEED_RANGE.items()},
                 setpoints_unsupported=["pitch", "roll", "torso_yaw", "gait", "swing_height"])
        return d

    def _torso_pose(self):
        P, Q = self._torso_view.get_world_poses()
        t = np.asarray(P, float).reshape(-1)[:3]; q = np.asarray(Q, float).reshape(-1)[:4]     # w, x, y, z
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), t

    def torso_rpy(self):
        """World roll/pitch/yaw (intrinsic ZYX) of torso_link."""
        R, _ = self._torso_pose()
        pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        return np.array([math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])])

    def top_z(self):
        """max world z of the torso_link geom vertices (torso, logo, head, waist support; the point set of the MuJoCo
        reference's head_top)."""
        R, t = self._torso_pose()
        return float((self.top_pts @ R.T + t)[:, 2].max())

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


def register_local_nav(kind):
    try:
        import local_nav
        local_nav.FOOT.setdefault(kind, (0.18, 0.22))
        local_nav.LASER_H.setdefault(kind, (0.15, 0.5, 0.9, 1.2))
    except Exception:
        pass


def make_g1s(kind, prim_path, position, yaw=0.0):
    register_local_nav(kind)
    return G1SonicAdapter(kind, prim_path, position, yaw)


try:
    make_g1s.SPEC = _spec()
except Exception:
    make_g1s.SPEC = None
