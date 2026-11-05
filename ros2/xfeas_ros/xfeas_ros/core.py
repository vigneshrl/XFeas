"""XFeas decision core for a real robot: sensor samples in, controller command out. No ROS imports (node.py is the ROS
wrapper), so the whole pipeline can be tested offline.

It rebuilds, from real sensor data, the observation the closed-loop layer builds in simulation
(xfeas/closed_loop/xfeas_method.py, _Gen5Layer._decide5) and asks the unchanged model server for a command:

    on_joints / on_base       50 Hz proprio ring (2 s) + 10 Hz command / achieved history h(t)
    on_frame                  RGB-D frame re-projected into the simulator's head-camera model, frame ring (4 s)
    decide(t)                 every 0.25 s: 4-frame memory + feas v0.4 -> request -> model server -> (skill rules) -> command
    twist(t)                  steering towards the goal + the speed of the command in force

Conventions (all as in simulation):
    world frame   z up (the robot's odometry frame); poses are (x, y, yaw)
    camera matrix M (4x4, row-vector form): rows 0-2 = camera x (right), y (up), z (backwards) axes in world, row 3 = camera
                  position (USD camera convention). camera_matrix_from_optical() converts a ROS optical-frame pose.
    images        square N x N, pinhole f = 12 / 20.955 * N (clearmap.intrinsics), depth in metres, inf = no return.
"""

import json
import math
import os
import struct
import subprocess
import io
from collections import deque

import numpy as np


def _import_xfeas(root):
    import sys

    if root and root not in sys.path:
        sys.path.insert(0, root)
    import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)

    return xfeas


def load_config(path):
    """config/<robot>.yaml -> dict: $VARS / ~ expanded, default_pos filled from the repo's controller adapter"""
    import yaml

    cfg = yaml.safe_load(open(path))
    cfg["xfeas_root"] = os.path.expandvars(os.path.expanduser(cfg.get("xfeas_root", "$XFEAS_ROOT")))
    for k in ("checkpoint", "python", "spec_path", "probe_path", "model_log"):
        if cfg.get(k):
            cfg[k] = os.path.expandvars(os.path.expanduser(cfg[k]))
    _import_xfeas(cfg["xfeas_root"])
    names = cfg["joint_names"]
    if not cfg.get("default_pos"):  # the controller's own default pose, from the repo's controller adapter
        if cfg["robot"] == "g1":
            import sonic_ref as SR

            d = {f"{k}_joint": float(v) for k, v in zip(SR.MJ_JOINT_NAMES, SR.DEFAULT_ANGLES)}
        else:
            import spot_posture as SP

            d = {f"{k}_joint": float(v) for k, v in SP.DEFAULT.items()}
        cfg["default_pos"] = [d.get(n, 0.0) for n in names]
    return cfg


# ---------------------------------------------------------------------- camera
SIM_F = 12.0 / 20.955  # focal length / image size of the simulator head camera (clearmap.intrinsics)


class CameraResampler:
    """Re-project a real pinhole camera image into the simulator camera model (square N x N, f = SIM_F * N, same optical
    centre and orientation). Pixels outside the real field of view become depth inf / RGB 0."""

    def __init__(self, K, width, height, N=224):
        import cv2

        self.cv2, self.N = cv2, N
        f = SIM_F * N
        u, v = np.meshgrid(np.arange(N, dtype=np.float64) + 0.5, np.arange(N, dtype=np.float64) + 0.5)
        x, y = (u - N / 2.0) / f, (v - N / 2.0) / f  # normalised ray of every simulator pixel
        K = np.asarray(K, np.float64).reshape(3, 3)
        self.mx = (K[0, 0] * x + K[0, 2] - 0.5).astype(np.float32)
        self.my = (K[1, 1] * y + K[1, 2] - 0.5).astype(np.float32)
        self.inside = (self.mx >= 0) & (self.mx <= width - 1) & (self.my >= 0) & (self.my <= height - 1)
        self.coverage = float(self.inside.mean())

    def rgb(self, img):
        out = self.cv2.remap(np.asarray(img, np.uint8)[..., :3], self.mx, self.my, self.cv2.INTER_LINEAR)
        out[~self.inside] = 0
        return out

    def depth(self, dep_m):
        d = np.asarray(dep_m, np.float32).copy()
        d[~np.isfinite(d) | (d <= 0.0)] = np.inf  # no return
        out = self.cv2.remap(d, self.mx, self.my, self.cv2.INTER_NEAREST)
        out[~self.inside] = np.inf
        return out


def camera_matrix_from_optical(R_world_opt, t_world_opt):
    """ROS optical frame (x right, y down, z forward) -> the simulator camera matrix M (see module docstring)"""
    R = np.asarray(R_world_opt, np.float64)
    M = np.eye(4)
    M[0, :3] = R[:, 0]  # camera x = right
    M[1, :3] = -R[:, 1]  # camera y = up
    M[2, :3] = -R[:, 2]  # camera z = backwards
    M[3, :3] = np.asarray(t_world_opt, np.float64)
    return M


def yaw_of_quat(q_xyzw):
    x, y, z, w = q_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_of_quat(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------- model server client
class ModelClient:
    """The unchanged model server (xfeas/closed_loop/model_server.py --serve) in its own process: 4-byte length + npz"""

    def __init__(self, root, python, ckpt, device_env="0", device="cuda:0", log=None):
        env = dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES=str(device_env), OMP_NUM_THREADS="2")
        srv = os.path.join(root, "xfeas", "closed_loop", "model_server.py")
        self.p = subprocess.Popen(
            [python, srv, "--ckpt", ckpt, "--device", device, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(log, "a") if log else None,
            env=env,
        )

    def ask(self, req, ctrl_id, spec_path, probe_path, robot):
        b = io.BytesIO()
        np.savez(b, ctrl_id=ctrl_id, spec_path=spec_path, probe_path=probe_path, robot=robot, **req)
        raw = b.getvalue()
        self.p.stdin.write(struct.pack("<I", len(raw)) + raw)
        self.p.stdin.flush()
        hdr = self.p.stdout.read(4)
        if len(hdr) < 4:
            raise RuntimeError("model server died (see its log)")
        n = struct.unpack("<I", hdr)[0]
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                raise RuntimeError("model server died mid-reply")
            buf += c
        ch = json.loads(str(np.load(io.BytesIO(buf), allow_pickle=False)["choice"]))
        if "error" in ch:
            raise RuntimeError(f"model server error: {ch['error']}")
        return ch

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# ---------------------------------------------------------------------- the core
class XFeasCore:
    PERIOD = 0.25  # decision interval (s), as in simulation
    PROP_HZ, PROP_S = 50.0, 2.0  # proprio ring
    HIST_HZ, HIST_N = 10.0, 60  # h(t)
    T_HIST_V2, T_FALLBACK, V_EST = (-1.5, -1.0, -0.5), (-2.0, -1.33, -0.67), 0.85  # history-frame picking rule

    def __init__(self, cfg):
        """cfg: dict from config/<robot>.yaml (see there for every key)"""
        self.cfg = cfg
        root = os.path.expanduser(cfg["xfeas_root"])
        for k, v in (cfg.get("env") or {}).items():  # arm switches read at import (skill_rules: XF_RULES_MODELSKILL, ...)
            os.environ[str(k)] = str(v)
        self.ctrl = cfg["controller_id"]
        self.robot = cfg["robot"]  # "g1" | "spot"
        data = os.path.join(root, "data")
        self.spec_path = cfg.get("spec_path") or os.path.join(data, "gen_v4", "specs", f"{self.ctrl}.json")
        self.probe_path = cfg.get("probe_path") or os.path.join(data, "probes_v4", self.ctrl, "office_v4.json")
        os.environ["XFEAS_GEN3_PROBE"] = self.probe_path  # skill_rules reads the probe ops from here
        _import_xfeas(root)
        import ctrl_spec as CS
        import clearmap as CLM
        import feas_mem as FM
        import path_follower as PF
        import virtual as VC

        self.CS, self.CLM, self.FM, self.PF, self.VC = CS, CLM, FM, PF, VC
        self.spec = CS.load(self.spec_path)
        self.probe = json.load(open(self.probe_path))
        self.v_nom = float(self.spec.get("nominal", {}).get("speed", 0.3))
        self.arm = cfg.get("arm", "rules")  # "rules" (g5sm / g5sr) | "gen5" (model only)
        self.rules = None
        if self.arm == "rules":
            import skill_rules as SKR

            self.rules = SKR.SkillRules(self.probe["fp"])
        self.names = list(cfg["joint_names"])  # model (simulator) joint names, in the order of the joint arrays
        self.q0 = np.asarray(cfg["default_pos"], np.float64)
        lim = cfg.get("joint_limits") or {}
        J = len(self.names)
        self.lo = np.asarray(lim.get("lower", [np.nan] * J), np.float64)
        self.hi = np.asarray(lim.get("upper", [np.nan] * J), np.float64)
        self.vmax = np.asarray(lim.get("velocity", [np.nan] * J), np.float64)
        self.tmax = np.asarray(lim.get("effort", [np.nan] * J), np.float64)
        self.client = None
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def start_model(self):
        c = self.cfg
        root = os.path.expanduser(c["xfeas_root"])
        self.client = ModelClient(root, c.get("python", "python3"), os.path.expanduser(c["checkpoint"]),
                                  c.get("cuda_device", "0"), c.get("device", "cuda:0"), c.get("model_log"))

    def reset(self, goal_xy=None):
        self.prop_base, self.prop_joints = deque(maxlen=int(self.PROP_HZ * self.PROP_S)), deque(maxlen=int(self.PROP_HZ * self.PROP_S))
        self.hist = deque(maxlen=self.HIST_N)
        self.ring, self.bases = [], {}
        self.base = None  # latest base state dict
        self.joints = None  # (t, q, dq, tau)
        self.top = None  # body top above the floor (m), from forward kinematics / TF
        self.floor_z = None  # world z of the support floor
        self.t_prop = self.t_hist = self.t_next = -1e9
        self.cmd_now, self.ch_now, self.t_sw, self.n_switch = None, None, None, 0
        self.v_cap, self.vy_cmd, self.face, self.v_over, self.hold_still = self.v_nom, 0.0, 0.0, None, False
        self.last_sent = np.zeros(4)  # [vx, vy, vz, wz] last twist (base token cmd)
        self.goal = None if goal_xy is None else np.asarray(goal_xy, float)
        self.start = self.t_start = None
        self.follower = self.geo = None
        if self.rules is not None:
            self.rules.reset()

    def set_goal(self, goal_xy):
        """new goal (world x, y): the route is the straight line from the current position (as the simulator tests)"""
        self.goal = np.asarray(goal_xy, float)
        self.start = None

    # ------------------------------------------------------------------ sensor input
    def on_base(self, t, pos, quat_xyzw, v_body, w_body):
        """odometry: position (world), orientation, body-frame linear / angular velocity"""
        R = rot_of_quat(quat_xyzw)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])  # gravity direction in the body frame
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        self.base = dict(t=t, pos=np.asarray(pos, float), yaw=yaw_of_quat(quat_xyzw), roll=roll, pitch=pitch, tilt=tilt,
                         v_b=np.asarray(v_body, float), w_b=np.asarray(w_body, float), g_b=g_b)
        if self.goal is not None and self.start is None:
            self.start, self.t_start = self.base["pos"][:2].copy(), t
            heading = math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])
            self.follower = self.PF.PathFollower([tuple(self.start), tuple(self.goal)])
            d = np.array([math.cos(heading), math.sin(heading)])
            self.geo = dict(P0=self.start.copy(), yaw=heading, d=d, n=np.array([-d[1], d[0]]))
        self._sample(t)

    def on_joints(self, t, q, dq, tau=None):
        """joint states in the order of cfg joint_names"""
        q = np.asarray(q, np.float64)
        self.joints = (t, q, np.asarray(dq, np.float64), np.zeros_like(q) if tau is None else np.asarray(tau, np.float64))
        self._sample(t)

    def on_body(self, top_above_floor=None, floor_z=None):
        """body top above the floor and the floor height (world z), e.g. from TF / forward kinematics"""
        if top_above_floor is not None:
            self.top = float(top_above_floor)
        if floor_z is not None:
            self.floor_z = float(floor_z)

    def on_frame(self, t, rgb, depth, M):
        """one head-camera frame already in the simulator camera model (CameraResampler) + its camera matrix M"""
        if self.base is None:
            return
        self.ring.append((t, np.asarray(rgb, np.uint8), np.asarray(depth, np.float32), np.asarray(M, np.float64)))
        self.bases[round(t, 4)] = (float(self.base["pos"][0]), float(self.base["pos"][1]), float(self.base["yaw"]))
        self.ring = [f for f in self.ring if t - f[0] <= 4.0]
        keep = {round(f[0], 4) for f in self.ring}
        self.bases = {k: v for k, v in self.bases.items() if k in keep}

    def _sample(self, t):
        """50 Hz proprio token + 10 Hz h(t) record (the simulator's ProprioBuffer / VRob._record)"""
        if self.base is None or self.joints is None:
            return
        b, (_, q, dq, tau) = self.base, self.joints
        if t - self.t_prop >= 1.0 / self.PROP_HZ - 1e-6:
            self.t_prop = t
            span = self.hi - self.lo
            ok = np.isfinite(span) & (span > 1e-6) & (span < 2 * math.pi + 1e-3)
            in_rng = np.where(ok, 2 * (q - self.lo) / np.where(ok, span, 1) - 1, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                vr = np.where(np.isfinite(self.vmax) & (self.vmax > 0), dq / self.vmax, 0.0)
                tr = np.where(np.isfinite(self.tmax) & (self.tmax > 0), tau / self.tmax, 0.0)
            wheel = np.array(["wheel" in n.lower() for n in self.names], float)
            self.prop_joints.append(np.stack([q - self.q0, dq, tau, in_rng, vr, tr, wheel], 1).astype(np.float32))
            self.prop_base.append(np.concatenate([b["v_b"], b["w_b"], b["g_b"], self.last_sent]).astype(np.float32))
        if t - self.t_hist >= 1.0 / self.HIST_HZ - 1e-6:
            self.t_hist = t
            ach = np.array([b["v_b"][0], b["v_b"][1], b["w_b"][2], b["pos"][2], np.nan if self.top is None else self.top
                            + (self.floor_z or 0.0), b["roll"], b["pitch"], b["yaw"], b["tilt"], b["pos"][0], b["pos"][1],
                            np.nan], np.float64)
            env = np.array([self.v_nom, 9.9, 9.9, 0.0], np.float32)  # == _send: set_env_intent(v_nom, 9.9, 9.9, 0)
            self.hist.append((t, self._role_vec(), env, ach, q.astype(np.float32), dq.astype(np.float32)))

    def _role_vec(self):
        CS = self.CS
        out = np.full(len(CS.HIST_ROLES), np.nan, np.float32)
        cmd = self.cmd_now if self.cmd_now is not None else CS.nominal_cmd(self.spec)
        named = {c["role"]: cmd.get(c["name"], cmd.get(c["role"])) for c in self.spec["commands"]}
        named.update({k: v for k, v in (self.cmd_now or {}).items() if k in CS.HIST_ROLES})
        # velocities as actually sent (the simulator records the follower's twist in these roles)
        named.update(lin_vel_x=self.last_sent[0], lin_vel_y=self.last_sent[1], yaw_rate=self.last_sent[3])
        for i, r in enumerate(CS.HIST_ROLES):
            v = named.get(r)
            if v is None:
                continue
            out[i] = CS.mode_index(self.spec, v) if r == "mode" else float(v)
        return out

    # ------------------------------------------------------------------ the 4-frame observation memory
    def _pick_hist(self, t, x, y):
        ring = self.ring[:-1]
        dist = [None if (b := self.bases.get(round(f[0], 4))) is None else math.hypot(x - b[0], y - b[1]) for f in ring]
        moved = max([d for d in dist if d is not None] or [0.0])
        sel = []
        for th, tf in zip(self.T_HIST_V2, self.T_FALLBACK):
            d_tgt, best = self.V_EST * self.v_nom * abs(th), None
            if moved >= d_tgt - 0.05:
                c = [(abs(d - d_tgt), i) for i, d in enumerate(dist) if d is not None and all(ring[i] is not s for s in sel)]
                if c and min(c)[0] <= 0.08:
                    best = ring[min(c)[1]]
            if best is None:
                c = [(abs(f[0] - (t + tf)), i) for i, f in enumerate(ring) if all(f is not s for s in sel)]
                if c and min(c)[0] <= 0.3:
                    best = ring[min(c)[1]]
            if best is not None:
                sel.append(best)
        sel.sort(key=lambda f: f[0])
        return sel + [self.ring[-1]]

    def _hist_obs(self, t, sel):
        import cv2

        CLM = self.CLM
        return dict(
            rgb_hist=np.stack([cv2.resize(f[1], (128, 128), interpolation=cv2.INTER_AREA) for f in sel]).astype(np.uint8),
            depth_hist=np.stack([CLM.min_pool_depth(np.clip(f[2], 0, 20.0), 128) for f in sel]).astype(np.float16),
            t_hist=np.array([f[0] - t for f in sel], np.float32),
            cam_hist=np.stack([f[3] for f in sel]).astype(np.float64),
            K128=CLM.intrinsics(128),
        )

    # ------------------------------------------------------------------ decision
    def ready(self):
        return (self.base is not None and len(self.prop_base) >= self.prop_base.maxlen and self.ring
                and self.floor_z is not None and self.goal is not None and self.client is not None)

    def decide(self, t):
        """every PERIOD s: -> the choice dict of the model server (+ 'rule' when the skill rules ran), or None"""
        if t < self.t_next or not self.ready():
            return None
        self.t_next = t + self.PERIOD
        b, (tf, rgb, dep, M) = self.base, self.ring[-1]
        x, y, yaw = float(b["pos"][0]), float(b["pos"][1]), float(b["yaw"])
        sel = self._pick_hist(t, x, y)
        ho = self._hist_obs(t, sel)
        dh = ho["depth_hist"].astype(np.float32)
        frames = [(dh[k], sel[k][3], self.bases.get(round(sel[k][0], 4))) for k in range(len(sel))]
        P = self.FM.mem_points(frames, (x, y, yaw), float(self.floor_z), self_x=0.0)
        fv, fd, _ = self.FM.feas_from_points(P, self.probe["fp"], self.probe["caps"], rules="v04")
        H = self.VC.pack_records(list(self.hist), self.geo, float(self.floor_z), t, self.HIST_N)
        pb, pj = np.stack(self.prop_base), np.stack(self.prop_joints)
        req = dict(
            rgb=rgb, depth=dep, prop_base=pb, prop_joints=pj, fmt="gen_v3", joint_names=json.dumps(self.names),
            hist_t=H["t"], hist_valid=H["valid"], hist_cmd=H["cmd"], hist_env=H["env"], hist_ach=H["ach"],
            hist_q=H["q"], hist_dq=H["dq"], ach_keys=json.dumps(list(self.VC.ACH_KEYS)), feas=np.asarray(fv, np.float32),
            cam_mount_M=np.asarray(M, np.float64), cam_mount_base=np.array([x, y, yaw], np.float64),
            cam_mount_camh=np.float64(M[3, 2] - self.floor_z),
            g5_cur_cmd=json.dumps(self.cmd_now) if self.cmd_now is not None else "",
            g5_t_since=np.float64(t - self.t_sw if self.t_sw is not None else t - self.t_start),
            g5_any_switch=np.bool_(self.t_sw is not None), g5_vx=np.float64(b["v_b"][0]),
        )
        if len(ho["t_hist"]) > 1:
            req.update(ho)
        if self.rules is not None:
            req["g5_all"] = np.bool_(True)
        ch = self.client.ask(req, self.ctrl, self.spec_path, self.probe_path, self.robot)
        if self.rules is not None:
            ch = self._rules_pick(t, P, ch)
        cmd = dict(ch["cmd"])
        prev = self.cmd_now if self.cmd_now is not None else dict(ch["table"][0]["cmd"])
        if json.dumps(prev, sort_keys=True) != json.dumps(cmd, sort_keys=True):
            self.t_sw, self.n_switch = t, self.n_switch + 1
        self.cmd_now, self.ch_now = cmd, ch
        self.v_cap = float(cmd.get("lin_vel_x", self.v_nom))
        self.vy_cmd = float(cmd.get("lin_vel_y", 0.0) or 0.0)
        self.face = self.PF.face_of(cmd)
        self.v_over = ch.get("v_over")
        return ch

    def _rules_pick(self, t, P, ch):
        """skill rules (== _Gen5Layer._rules_pick): hazard map while upright, then WHEN to switch + the staged stand-up"""
        s = self.follower.errors(self.base["pos"][:2], self.base["yaw"])["s"] if self.follower else 0.0
        top = self.top
        upright = top is not None and top >= self.rules.stand_top - 0.15 and not self.face
        seen = self.rules.observe(P, s, True) if upright else None
        row, v_over, why = self.rules.step(t, s, ch["table"], top, float(self.base["tilt"]))
        self.hold_still = row["cls"] in ("stoplow", "standup")
        return dict(ch, cmd=row["cmd"], cls=row["cls"], U=row["U"], p=row["p"], stock=row["cls"] == "stock", v_over=v_over,
                    model_cls=ch["cls"], rule=dict(skill=self.rules.skill, why=why, s=round(s, 2), seen=seen))

    # ------------------------------------------------------------------ output
    def command(self, t):
        """-> dict for the controller: twist (vx, vy, wz) + the posture / mode commands of the command in force"""
        cmd = self.cmd_now if self.cmd_now is not None else self.CS.nominal_cmd(self.spec)
        out = dict(lin_vel_x=0.0, lin_vel_y=0.0, yaw_rate=0.0, cls=(self.ch_now or {}).get("cls", "nominal"),
                   skill=((self.ch_now or {}).get("rule") or {}).get("skill", ""), U=(self.ch_now or {}).get("U"))
        for role in ("pelvis_height", "torso_pitch", "torso_roll", "mode", "speed"):
            if role in cmd:
                out[role] = cmd[role]
        out["face"] = self.face
        at_goal = self.goal is not None and self.base is not None and np.hypot(*(self.base["pos"][:2] - self.goal)) < 0.5
        if self.hold_still or self.base is None or self.follower is None or at_goal:
            self.last_sent = np.zeros(4)
            out["at_goal"] = bool(at_goal)
            return out
        err = self.follower.errors(self.base["pos"][:2], self.PF.travel_yaw(self.base["yaw"], cmd))
        steer = self.PF.follower_cmd(self.spec, None, err, wz_cap=float(self.cfg.get("wz_cap", 0.5)))
        c = self.CS.cmd(self.spec, "yaw_rate")
        wz = float(steer.get(c["name"], 0.0)) if c is not None else 0.0
        vx = self.v_cap if self.v_over is None else min(self.v_cap, float(self.v_over))
        out.update(lin_vel_x=float(vx), lin_vel_y=float(self.vy_cmd), yaw_rate=wz, at_goal=False)
        self.last_sent = np.array([vx, self.vy_cmd, 0.0, wz])
        return out
