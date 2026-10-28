#!/usr/bin/env python
"""
Unitree G1 (29-DoF, half-sphere hands) driven by the RELEASED PHP D435i student (amazon-far/php_parkour @c7c9ce3,
student-assets-v1: depth_backbone.onnx + student.onnx), as run by its sim2sim (run_php_sim.sh / run_php_inference.sh,
Holosoma @4a0bf04). The policy runs in a SEPARATE process (php_server.py in the gr00t_wbc venv, which has onnxruntime;
Isaac kit python does not), called in LOCKSTEP once per 50 Hz tick (php_client.PhpClient). Nothing trained or tuned.

    --robot g1p --robot-factory g1_php:make_g1p          (scripts/php on PYTHONPATH)

Physics (harness): 200 Hz, control 50 Hz (= PHP training: holosoma simulator fps 200, control_decimation 4; the release
MuJoCo sim2sim runs 500 Hz). PD in effort mode every physics step: tau = Kp (q* - q) - Kd dq, Kp/Kd = holosoma
g1_29dof_wbt_distillation motor_kp/kd (= the ONNX kp/kd metadata), clamped to the MJCF actuatorfrcrange; MJCF passive
frictionloss 0.1 emulated as -0.1 * clip(dq / 0.05, -1, 1) (as the g1s bridge does).
Robot model: the release MJCF g1_29dof_halfspherehand.xml imported with Isaac 5.1's MJCF importer (convert_php_mjcf.py:
capsule / sphere colliders as in the MJCF, mass 33.591 kg = MuJoCo); self-collision off (holosoma
enable_self_collisions=False). PHP_FOOT_COFF (default 0.02): contactOffset on the foot capsules (harness convention of
g1g / g1s).
Start: teleport to the stiff-hold pose (_DISTILL_STIFF_POS) at full stiff gains (the state the gantry + 100-tick blend of
the release ends in); after PHP_HOLD_S (0.5 s) the policy starts (']') with direction 'stand'.

Depth (the policy's only exteroception): a USD camera posed every render from the torso_link world pose and the release
mount (torso_link, pos (0.01, 0.01, 0.44), quat wxyz (0.6029, 0.3759, -0.3630, -0.6029), camera -Z forward / +Y up as in
MuJoCo and USD), 106 x 60, hfov 89.5 deg (vfov 58.595), distance_to_image_plane (= the training raycast's z-depth),
rendered at 10 Hz (D435i rate, render_hz of the release depth plugin) and preprocessed exactly as DepthShmPlugin
(php_ref.preprocess_depth). The render happens in command() (outside the physics callback, as local_nav.DepthSensor).

Command mapping (command(cmd)): the policy takes a one-hot DIRECTION class (speed mode LOW):
  dict(dir=name)  -> stand | forward | left_45 | left_90 | right_45 | right_90 | back, used as is;
  [vx, vy, vz, wz] or dict(vx, vy, wz) -> class by the release's own continuous->discrete quantiser
                     (DepthDistillationPolicy._apply_velocity: sector of atan2(vy, vx), dead zone 0.05; pure wz -> 90 turn).
  Measured in the MuJoCo reference (php_mujoco_ref.py, flat): forward 0.87 m/s straight; left_45 = arc at ~45 deg/s,
  0.39 m/s; left_90 = ~90 deg/s turn nearly in place; 'back' (class 11) walks FORWARD (0.70 m/s).
  There is NO speed, lateral-velocity, body-height, pitch or gait input.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, struct, time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

import php_ref as P

USD = os.environ.get("PHP_USD", xfeas_paths.BASELINES + "/php_parkour/usd/g1_php.usd")
MESHES = xfeas_paths.BASELINES + "/php_parkour/usd/meshes"
KEY = os.environ.get("PHP_KIND", "g1p")
DECIM = 4
FRICTION_V, FRICTION_LOSS = 0.05, 0.1
HOLD_S = float(os.environ.get("PHP_HOLD_S", "0.5"))
START_PELVIS_Z = 0.786                      # stiff-hold pose standing on the floor (MuJoCo reference: 0.785-0.786)
RENDER_EVERY = int(round(200 / P.DEPTH_HZ))  # physics steps between depth renders (10 Hz at 200 Hz)


def _spec():
    import robots as RB
    return RB.RobotSpec(KEY, "legged", (0.87, 0.0, 0.0, 0.0, 1.6), START_PELVIS_Z, 0.8, 200, usd=USD,
                        foot_expr="pelvis/.*_ankle_roll_link", v_min=0.87, cam=(0.12, 0.62, -0.47), min_z=0.2,
                        extra=dict(controller="PHP depth student (released, D435i)", directions=list(P.CMD_CODES[0])))


def _stl_vertices(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
    return np.unique(a["v"].reshape(-1, 3).astype(float).round(5), axis=0)


def _torso_top_points():
    """torso_link frame vertices of the torso, logo and head geoms (MJCF: logo / head at pos (0.0039635, 0, -0.044))."""
    off = np.array([0.0039635, 0.0, -0.044])
    pts = [_stl_vertices(f"{MESHES}/torso_link_rev_1_0.STL"), _stl_vertices(f"{MESHES}/logo_link.STL") + off,
           _stl_vertices(f"{MESHES}/head_link.STL") + off]
    pts = np.concatenate(pts)
    return pts[pts[:, 2] > np.percentile(pts[:, 2], 90)]


class G1PhpAdapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        import robots as RB
        self.RB = RB
        self.spec = make_g1p.SPEC
        self.kind, self.prim_path = kind, prim_path
        self.cmd = np.zeros(4); self.cmd_req = np.zeros(4); self.clipped = False; self.clipped_keys = []
        self.direction = "stand"
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(USD, prim_path)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art = SingleArticulation(prim_path=prim_path + "/pelvis/pelvis", name=kind, position=np.asarray(position, float),
                                       orientation=np.array([q[3], q[0], q[1], q[2]]))
        self.robot = self._art
        self.ctrl = None
        st = omni.usd.get_context().get_stage()
        self.stage = st
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self.foot_coff = self._foot_contact_offset(st)
        from pxr import PhysxSchema
        root = st.GetPrimAtPath(prim_path + "/pelvis/pelvis")
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr().Set(False)
        self.self_collision = False
        self.top_pts = _torso_top_points()
        self._floor_z = 0.0
        self._torso_view = RigidPrim(prim_paths_expr=prim_path + "/pelvis/torso_link", name=kind + "_torso",
                                     reset_xform_properties=False)
        self._make_camera(st)
        # The training raycast (G1FlatRsD435iConfig.ray_cast_bodies) sees every robot link mesh EXCEPT torso_link
        # (torso, logo, head) -- the camera sits inside the head shell. Isaac renders everything, so the torso_link
        # visuals are hidden for the depth render only (render_depth), and the MJCF sites (imu / mid360 marker spheres,
        # not robot geometry) are hidden always. Robot collision geoms are already invisible (importer).
        from pxr import Usd, UsdGeom
        self._torso_vis = UsdGeom.Imageable(st.GetPrimAtPath(prim_path + "/pelvis/torso_link/visuals"))
        n_sites = 0
        for p in Usd.PrimRange(st.GetPrimAtPath(prim_path)):
            if p.GetName() == "sites":
                UsdGeom.Imageable(p).MakeInvisible(); n_sites += 1
        print(f"[g1p] {n_sites} site groups hidden; torso_link visuals hidden during depth renders", flush=True)
        from php_client import PhpClient
        self.client = PhpClient(threads=int(os.environ.get("PHP_THREADS", "2")))
        self.tick_ms, self.render_ms = [], []
        self.q_target = P.STIFF_POS.copy(); self.kp, self.kd = P.STIFF_KP, P.STIFF_KD
        self.mode_now = -1
        self.last_info = {}
        self.sub = 0; self.hold = int(HOLD_S * 200); self._started = False
        self._last_render = -10 ** 9; self._have_frame = False
        self.depth_log = []                     # (t, preprocessed frame) samples for QA, every 2 s
        self.dir_log = []

    # ------------------------------------------------------------------ depth camera
    def _make_camera(self, st):
        from pxr import UsdGeom, Gf
        import omni.replicator.core as rep
        self.Gf, self.rep = Gf, rep
        path = f"/World/PhpDepthCam_{self.kind}"
        cam = UsdGeom.Camera.Define(st, path)
        f = 10.0
        hap = 2 * f * math.tan(math.radians(P.CAM_HFOV / 2))
        cam.CreateFocalLengthAttr(f); cam.CreateHorizontalApertureAttr(hap)
        cam.CreateVerticalApertureAttr(hap * P.CAM_H / P.CAM_W)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        self.cam_xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product(path, (P.CAM_W, P.CAM_H))
        self.ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); self.ann.attach([rp])
        self.vfov_deg = math.degrees(2 * math.atan(math.tan(math.radians(P.CAM_HFOV / 2)) * P.CAM_H / P.CAM_W))
        w, x, y, z = P.CAM_QUAT_WXYZ
        self.R_mount = Rotation.from_quat([x, y, z, w]).as_matrix(); self.p_mount = np.array(P.CAM_POS)

    def camera_pose(self):
        R, t = self._torso_pose()
        return R @ self.R_mount, t + R @ self.p_mount

    def render_depth(self):
        """Pose the camera from the CURRENT torso pose, render, return (60, 106) metric z-depth (inf = no hit)."""
        Rc, pc = self.camera_pose()
        q = Rotation.from_matrix(Rc).as_quat()              # x, y, z, w
        M = self.Gf.Matrix4d(); M.SetTransform(self.Gf.Rotation(self.Gf.Quatd(float(q[3]), self.Gf.Vec3d(*map(float, q[:3])))),
                                               self.Gf.Vec3d(*map(float, pc)))
        self.cam_xf.Set(M)
        t = time.perf_counter()
        self._torso_vis.MakeInvisible()
        self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        d = np.asarray(self.ann.get_data(), np.float32)
        self._torso_vis.MakeVisible()
        self.render_ms.append((time.perf_counter() - t) * 1e3)
        if d.ndim != 2 or d.shape != (P.CAM_H, P.CAM_W):
            return None
        d = d.copy(); d[~np.isfinite(d) | (d <= 0)] = np.inf
        return d

    def _maybe_depth(self, force=False):
        if not self.ready:
            return
        if force or self.sub - self._last_render >= RENDER_EVERY:
            d = self.render_depth()
            if d is not None:
                self.client.depth(d); self._have_frame = True
                self._last_render = self.sub
                if self.sub % 400 < RENDER_EVERY:
                    self.depth_log.append((round(self.sub / 200.0, 2), P.preprocess_depth(d)))

    # ------------------------------------------------------------------ physics
    def _foot_contact_offset(self, st):
        coff = float(os.environ.get("PHP_FOOT_COFF", "0.02"))
        if coff <= 0:
            return 0.0
        from pxr import Usd, UsdPhysics, PhysxSchema
        n = 0
        for side in ("left", "right"):
            fp = f"{self.prim_path}/pelvis/{side}_ankle_roll_link"
            cols = [str(p.GetPath()) for p in Usd.PrimRange(st.GetPrimAtPath(fp), Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.CollisionAPI)]
            for cp in cols:
                q = st.GetPrimAtPath(cp)
                while q.GetPath().pathString != "/":
                    if q.IsInstanceable():
                        q.SetInstanceable(False)
                    q = q.GetParent()
            for cp in cols:
                pc = PhysxSchema.PhysxCollisionAPI.Apply(st.GetPrimAtPath(cp))
                pc.CreateContactOffsetAttr().Set(coff); pc.CreateRestOffsetAttr().Set(0.0)
                n += 1
        print(f"[g1p] foot capsules contactOffset {coff} restOffset 0 on {n} colliders", flush=True)
        return coff

    def initialize(self):
        self._art.initialize()
        names = list(self._art.dof_names)
        self.idx = np.array([names.index(n) for n in P.JOINT_NAMES])
        n = len(names)
        av = self._art._articulation_view
        av.set_gains(kps=np.zeros((1, n)), kds=np.zeros((1, n)))
        cap = np.full(n, 1e6); cap[self.idx] = P.TAU
        av.set_max_efforts(cap.reshape(1, -1))
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        self._q0 = np.zeros(n); self._q0[self.idx] = P.STIFF_POS
        self.ctrl = SimpleNamespace(default_pos=self._q0.copy())
        self._torso_view.initialize()
        print(f"[g1p] dofs {n} mass {self._base_masses.sum():.3f} (MJCF 33.591) depth {P.CAM_W}x{P.CAM_H} hfov {P.CAM_HFOV} "
              f"vfov {self.vfov_deg:.3f} @ {P.DEPTH_HZ} Hz", flush=True)
        self.ready = True

    def _state(self):
        _, quat = self._art.get_world_pose()
        quat = np.asarray(quat, float)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        omega = R.T @ np.asarray(self._art.get_angular_velocity(), float)        # = MuJoCo free-joint qvel[3:6]
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
        if self.hold > 0 or not self._have_frame:
            self.q_target, self.kp, self.kd = P.STIFF_POS, P.STIFF_KP, P.STIFF_KD
            self.hold -= 1
        else:
            if not self._started:
                self.client.direction(self.direction); self._started = True
                self._dsub = 0
            if self._dsub % DECIM == 0:
                t = time.perf_counter()
                self.q_target, self.last_info = self.client.tick(q, dq, quat, omega)
                self.tick_ms.append((time.perf_counter() - t) * 1e3)
                self.kp, self.kd = P.KP, P.KD
            self._dsub += 1
        self.sub += 1
        tau = np.clip(self.kp * (self.q_target - q) - self.kd * dq, -P.TAU, P.TAU) * self._variant["effort_scale"]
        self.last_tau = tau
        full = np.zeros(len(qa))
        full[self.idx] = tau - FRICTION_LOSS * np.clip(dq / FRICTION_V, -1.0, 1.0)
        self._art.apply_action(ArticulationAction(joint_efforts=full))

    @property
    def tau_max(self):
        return P.TAU

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, cmd=None, **kw):
        """dict(dir=name) | [vx, vy, vz, wz] | dict(vx, vy, wz) -> direction class (release quantiser); renders depth
        when due (10 Hz)."""
        name = None
        if isinstance(cmd, dict) and cmd.get("dir"):
            name = str(cmd["dir"])
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            t = np.zeros(4)
            if isinstance(cmd, dict):
                t[0], t[1], t[3] = cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("wz", 0.0)
            elif cmd is not None:
                c = np.asarray(cmd, float)[:4]; t[:len(c)] = c
            vx, vy, wz = float(t[0]), float(t[1]), float(t[3])
            name = P.quantise(vx, vy, wz, 0)
        self.cmd_req = np.array([vx, vy, 0.0, wz]); self.cmd = self.cmd_req.copy()
        self.clipped_keys = []; self.clipped = False
        if name != self.direction:
            self.dir_log.append((round(self.sub / 200.0, 2), name))
        self.direction = name
        if self._started:
            self.client.direction(name)
        self._maybe_depth(force=not self._have_frame)
        return self.cmd

    def setpoint(self):
        return dict(dir=self.direction)

    def stop(self):
        self.command(dict(dir="stand"))

    def reset_setpoints(self):
        self.command(dict(dir="stand"))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        av = self._art._articulation_view
        if getattr(self, "_base_masses", None) is not None:
            m = self._base_masses.copy(); m[0] += float(payload_kg)
            av.set_body_masses(m.reshape(1, -1))
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        """Stiff-hold pose, soles on the floor, zero velocity; HOLD_S of stiff hold, then a fresh policy (direction stand)."""
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + START_PELVIS_Z]), np.array([q[3], q[0], q[1], q[2]]))
        self._art.set_linear_velocity(np.zeros(3)); self._art.set_angular_velocity(np.zeros(3))
        if getattr(self, "_q0", None) is not None:
            self._art.set_joint_positions(self._q0)
            self._art.set_joint_velocities(np.zeros(len(self._q0)))
        self.hold = int(HOLD_S * 200); self._started = False; self._have_frame = False
        self.direction = "stand"; self.depth_log = []; self.dir_log = []
        self.client.reset()                     # fresh policy state (last action 0, no frame) before the first render
        self.q_target, self.kp, self.kd = P.STIFF_POS.copy(), P.STIFF_KP, P.STIFF_KD

    def base_state(self):
        return self.RB.RobotAdapter.base_state(self)

    def fallen(self, bs=None):
        bs = bs or self.base_state()
        if bs is None:
            return True
        return bs["tilt"] > self.spec.fall_tilt or bs["pos"][2] - self._floor_z < self.spec.min_z

    def _joint_limits(self):
        return self.RB.RobotAdapter._joint_limits(self)

    def proprio(self, bs=None):
        out = self.RB.RobotAdapter.proprio(self, bs)
        if out is not None:
            out[2]["setpoint"] = self.setpoint(); out[2]["clipped_keys"] = []
        return out

    def descriptor(self, stage=None):
        d = self.RB.RobotAdapter.descriptor(self, stage)
        d.update(controller="PHP depth student (released, D435i)", directions=list(P.CMD_CODES[0]),
                 setpoints_supported=["dir"], setpoints_unsupported=["vx", "vy", "wz", "pitch", "roll", "body_height", "gait"])
        return d

    def _torso_pose(self):
        Pp, Q = self._torso_view.get_world_poses()
        t = np.asarray(Pp, float).reshape(-1)[:3]; q = np.asarray(Q, float).reshape(-1)[:4]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), t

    def torso_rpy(self):
        R, _ = self._torso_pose()
        pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        return np.array([math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])])

    def top_z(self):
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


def make_g1p(kind, prim_path, position, yaw=0.0):
    register_local_nav(kind)
    return G1PhpAdapter(kind, prim_path, position, yaw)


try:
    make_g1p.SPEC = _spec()
except Exception:
    make_g1p.SPEC = None
