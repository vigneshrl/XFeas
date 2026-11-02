"""Map-free local navigation used by every method: the robot's own odometry, the goal relative to it and its head
camera only.

Details:
Map-free local navigation for episode_runner.py: what every robot runs WITHOUT a prior map.

Inputs (all onboard): the robot's own pose (odometry), the goal relative to the robot, and its HEAD RGB-D
camera (the only sensor; the same camera the feasibility layer uses). No map, no route, no waypoints, no lidar.

  LocalNav    -- stock controller + local A* / DWA on obstacle points from the HEAD RGB-D camera (one sensor,
                 no lidar), with a short-term
                 obstacle memory and a stop/back-up/rotate recovery. This is the "stock" arm.
  XFeasLocal  -- the same LocalNav with our XFeas model deciding speed / marking areas infeasible
                 (xfeas_decider.XFeasDecider). The ONLY difference between the two arms is XFeas.

Laser / DWA / FOOT / LASER_H are copied from nav_baseline.py (it parses argv on import).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
from collections import deque
import numpy as np

FOOT = dict(
    spot=(0.55, 0.25), anymal=(0.52, 0.30), h1=(0.20, 0.28), carter=(0.36, 0.32), jackal=(0.26, 0.22), iris=(0.30, 0.30)
)
LASER_H = dict(
    spot=(0.15, 0.35, 0.6),
    anymal=(0.15, 0.4, 0.7),
    h1=(0.15, 0.6, 1.1, 1.65),
    carter=(0.15, 0.45, 0.8),
    jackal=(0.15, 0.3),
    iris=(0.9, 1.0, 1.1),
)
LASER_N, LASER_RANGE = 90, 5.0
BLOCK_S, BACKUP_S, ROTATE_S, MAX_RECOVER = 2.0, 1.2, 1.6, 4
MEM_S, LOOK = 8.0, 3.0  # obstacle memory (s); local goal look-ahead along the goal direction (m)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Laser:
    def __init__(self, pxq, carb, robot_root, kind):
        self.q, self.carb, self.root = pxq, carb, robot_root
        self.hx, self.hy = FOOT[kind]
        self.H = LASER_H[kind]
        a = np.linspace(0, 2 * math.pi, LASER_N, endpoint=False)
        with np.errstate(divide="ignore"):
            self.r0 = np.minimum(self.hx / np.abs(np.cos(a)), self.hy / np.abs(np.sin(a))) + 0.05
        self.a = a

    def scan(self, pos, yaw, floor_z):
        pts = []
        for h in self.H:
            z = floor_z + h
            for a, r0 in zip(self.a, self.r0):
                c, s_ = math.cos(yaw + a), math.sin(yaw + a)
                o = [pos[0] + r0 * c, pos[1] + r0 * s_, z]
                rem = LASER_RANGE - r0
                for _ in range(2):
                    hit = self.q.raycast_closest(self.carb.Float3(*o), self.carb.Float3(c, s_, 0.0), rem)
                    if not (hit and hit.get("hit")):
                        break
                    if str(hit.get("collision", "")).startswith(self.root):
                        d = float(hit["distance"]) + 0.03
                        o = [o[0] + d * c, o[1] + d * s_, z]
                        rem -= d
                        continue
                    pts.append((float(hit["position"][0]), float(hit["position"][1]), h))
                    break
        return np.array(pts, np.float32).reshape(-1, 3)


FOOT_SELF = (0.3, 0.3)


class DepthSensor:
    """The ONLY navigation sensor: the robot's head RGB-D camera (same mount as the feasibility layer).
    Depth pixels are back-projected to world points; points whose height above the local floor lies in the robot's
    obstacle band (lowest .. highest LASER_H level, +0.15 m) become obstacle points. Narrow field of view -> the
    planner's short-term point memory provides the rest. 160x120 depth, every 2nd pixel, 0.2-5 m."""

    W, H, FOCAL, HAP = 160, 120, 12.0, 20.955

    def __init__(self, stage, rep, kind, cam_mount):
        from pxr import UsdGeom, Gf

        self.Gf, self.rep, self.mount = Gf, rep, cam_mount
        cam = UsdGeom.Camera.Define(stage, "/World/NavDepthCam")
        cam.CreateFocalLengthAttr(self.FOCAL)
        cam.CreateHorizontalApertureAttr(self.HAP)
        cam.CreateVerticalApertureAttr(self.HAP * self.H / self.W)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
        self.xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product("/World/NavDepthCam", (self.W, self.H))
        self.ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.ann.attach([rp])
        hs = LASER_H[kind]
        self.band = (min(hs), max(hs) + 0.15)
        global FOOT_SELF
        FOOT_SELF = FOOT.get(kind, (0.3, 0.3))
        fx = self.FOCAL / self.HAP * self.W
        u, v = np.meshgrid(np.arange(0, self.W, 2), np.arange(0, self.H, 2))
        self.u, self.v, self.fx = u.ravel(), v.ravel(), fx

    def scan(self, pos, yaw, floor_z):
        Gf = self.Gf
        fwd, up, pitch = self.mount
        c, s = math.cos(yaw), math.sin(yaw)
        eye = np.asarray(pos, float) + np.array([fwd * c, fwd * s, up])
        tgt = eye + np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        c2w = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        self.xf.Set(c2w)
        self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        d = np.asarray(self.ann.get_data(), np.float32)
        if d.ndim != 2:
            return np.zeros((0, 3), np.float32)
        z = d[self.v, self.u]
        ok = np.isfinite(z) & (z > 0.5) & (z < LASER_RANGE)  # < 0.5 m: own arm/gripper in view
        z, uu, vv = z[ok], self.u[ok], self.v[ok]
        pc = np.stack([(uu - self.W / 2) / self.fx * z, -(vv - self.H / 2) / self.fx * z, -z, np.ones_like(z)], 1)
        M = np.array(c2w, float)  # row-vector convention: p_world = p_cam @ M
        pw = pc @ M
        h = pw[:, 2] - floor_z
        r_self = math.hypot(*FOOT_SELF) + 0.1  # drop the robot's own body (e.g. Spot's gripper)
        keep = (h >= self.band[0]) & (h <= self.band[1]) & (np.hypot(pw[:, 0] - pos[0], pw[:, 1] - pos[1]) > r_self)
        return np.c_[pw[keep, 0], pw[keep, 1], h[keep]].astype(np.float32)


class DWA:
    def __init__(self, kind, v_nom, v_min_fwd, wz_cap, T=1.5, dt=0.1, margin=0.08):
        hx, hy = FOOT[kind]
        n = max(1, int(round((hx - hy) / hy)) + 1) if hx > hy else 1
        self.cx = np.linspace(-(hx - hy), hx - hy, n) if n > 1 else np.zeros(1)
        self.rad = hy + margin
        self.hard = hy + 0.01
        lo = max(v_min_fwd, 0.25 * v_nom)
        slow = max(v_min_fwd, min(0.2, lo))
        self.vs = np.unique(np.r_[0.0, slow, np.linspace(lo, v_nom, 4)])
        self.ws = np.linspace(-wz_cap, wz_cap, 13)
        self.v_nom, self.T, self.dt = v_nom, T, dt

    def plan(self, x, y, yaw, carrot, pts):
        V, Wz = np.meshgrid(self.vs, self.ws, indexing="ij")
        V, Wz = V.ravel(), Wz.ravel()
        n = int(self.T / self.dt)
        t = np.arange(1, n + 1) * self.dt
        th = yaw + Wz[:, None] * t[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            rx = np.where(
                np.abs(Wz[:, None]) > 1e-4,
                V[:, None] / Wz[:, None] * (np.sin(th) - math.sin(yaw)),
                V[:, None] * t[None] * math.cos(yaw),
            )
            ry = np.where(
                np.abs(Wz[:, None]) > 1e-4,
                -V[:, None] / Wz[:, None] * (np.cos(th) - math.cos(yaw)),
                V[:, None] * t[None] * math.sin(yaw),
            )
        PX, PY = x + rx, y + ry
        CX = PX[..., None] + self.cx[None, None] * np.cos(th)[..., None]
        CY = PY[..., None] + self.cx[None, None] * np.sin(th)[..., None]
        also_now = np.stack([x + self.cx * math.cos(yaw), y + self.cx * math.sin(yaw)], -1)
        near = np.zeros((0, 2))
        if len(pts):
            P = pts[:, :2]
            near = P[np.hypot(P[:, 0] - x, P[:, 1] - y) < self.v_nom * self.T + 1.5]
        if len(near):
            d = np.sqrt((CX[..., None] - near[:, 0]) ** 2 + (CY[..., None] - near[:, 1]) ** 2).min(-1).min(-1)
            dmin = d.min(1)
            raw_now = float(np.sqrt(((also_now[:, None, :] - near[None]) ** 2).sum(-1)).min())
            thr = self.rad if raw_now > self.rad else max(self.hard, raw_now - 0.02)
            clear = dmin - self.rad
            ok = dmin > thr
            d_now = raw_now - self.rad
        else:
            clear = np.full(len(V), 9.0)
            d_now = 9.0
            ok = np.ones(len(V), bool)
        cx_, cy_ = carrot
        d0 = math.hypot(cx_ - x, cy_ - y)
        d1 = np.hypot(cx_ - PX[:, -1], cy_ - PY[:, -1])
        prog = (d0 - d1) / max(self.v_nom * self.T, 1e-3)
        head = np.cos(np.arctan2(cy_ - PY[:, -1], cx_ - PX[:, -1]) - th[:, -1])
        score = 1.0 * prog + 0.35 * head + 0.3 * np.minimum(clear, 1.0) + 0.25 * V / self.v_nom
        moving = (V > 0) | (np.abs(Wz) > 1e-3)
        adm = ok & moving
        fwd = bool((ok & (V > 0)).any())
        if not adm.any():
            return dict(v=0.0, w=0.0, admissible=False, fwd_ok=False, clear=float(np.max(clear)), d_now=float(d_now))
        k = int(np.argmax(np.where(adm, score, -1e9)))
        return dict(
            v=float(V[k]), w=float(Wz[k]), admissible=True, fwd_ok=fwd, clear=float(clear[k]), d_now=float(d_now)
        )


class LocalNav:
    """Stock controller + map-free reactive local navigation toward the relative goal."""

    name = "stock"
    label = "STOCK controller + map-free local nav"

    def setup(self, ctx):
        self.ctx = ctx
        kind, spec = ctx["kind"], ctx["spec"]
        self.kind, self.spec = kind, spec
        self.laser = DepthSensor(ctx["stage"], ctx["rep"], kind, spec.cam)  # head RGB-D only (no lidar)

    def reset(self, ep):
        sp = self.spec
        self.v = max(ep["speed_frac"] * sp.max_cmd[0], sp.v_min)
        self.wz_cap = ep["speed_frac"] * sp.max_cmd[4]
        self.dwa = DWA(self.kind, self.v, sp.v_min, self.wz_cap)
        self.floor_z = ep["floor_z0"]  # the robot knows the floor it starts on
        self.mem, self.avoid_mem = deque(), deque()
        self.mode, self.mode_t, self.blocked_t, self.n_recover = "drive", 0.0, 0.0, 0
        self.k, self.dw, self.rot_dir = 0, dict(v=0.0, w=0.0, fwd_ok=True, admissible=True), 1.0
        self.v_back = -(0.4 if self.kind == "spot" else 0.25)

    def avoid(self, pts_xy, t):
        """mark points infeasible (e.g. by XFeas): they join the obstacle memory for 20 s"""
        if len(pts_xy):
            self.avoid_mem.append((t, np.c_[np.asarray(pts_xy, np.float32), np.zeros(len(pts_xy), np.float32)]))

    def _points(self, t, x, y):
        while self.mem and self.mem[0][0] < t - MEM_S:
            self.mem.popleft()
        while self.avoid_mem and self.avoid_mem[0][0] < t - 20.0:
            self.avoid_mem.popleft()
        P = [p for _, p in self.mem] + [p for _, p in self.avoid_mem]
        P = np.concatenate(P) if P else np.zeros((0, 3), np.float32)
        return P[np.hypot(P[:, 0] - x, P[:, 1] - y) < LASER_RANGE] if len(P) else P

    def _astar_carrot(self, x, y, gx, gy, P):
        """Local A* on an 8x8 m, 0.1 m grid around the robot built ONLY from remembered sensor points
        (laser + areas marked infeasible), inflated by the footprint radius; unseen = free. Returns a
        carrot 1.2 m along the path toward the goal (projected onto the window edge), or None."""
        import heapq
        from scipy.ndimage import binary_dilation

        R, C = 4.0, 0.1
        n = int(2 * R / C)
        occ = np.zeros((n, n), bool)
        if len(P):
            ij = np.floor((P[:, :2] - [x - R, y - R]) / C).astype(int)
            ok = (ij >= 0).all(1) & (ij < n).all(1)
            occ[ij[ok, 0], ij[ok, 1]] = True
        k = int(np.ceil((self.dwa.rad + 0.02) / C))
        yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
        occ = binary_dilation(occ, structure=(xx**2 + yy**2) <= k * k)
        s0 = (n // 2, n // 2)
        occ[s0[0] - 1 : s0[0] + 2, s0[1] - 1 : s0[1] + 2] = False  # never trapped by own inflation
        dg = math.hypot(gx - x, gy - y)
        if dg > R - 0.3:  # goal outside: aim at the window edge
            f = (R - 0.3) / dg
            tx, ty = x + f * (gx - x), y + f * (gy - y)
        else:
            tx, ty = gx, gy
        g = (int((tx - x + R) / C), int((ty - y + R) / C))
        g = (min(max(g[0], 0), n - 1), min(max(g[1], 0), n - 1))
        if occ[g]:  # target cell blocked: nearest free cell to it
            fr = np.argwhere(~occ)
            if not len(fr):
                return None
            g = tuple(fr[np.argmin(((fr - g) ** 2).sum(1))])
        D = {s0: 0.0}
        prev = {}
        pq = [(0.0, s0)]
        nb = [
            (1, 0, 1),
            (-1, 0, 1),
            (0, 1, 1),
            (0, -1, 1),
            (1, 1, 1.414),
            (1, -1, 1.414),
            (-1, 1, 1.414),
            (-1, -1, 1.414),
        ]
        while pq:
            _, u = heapq.heappop(pq)
            if u == g:
                break
            for di, dj, w in nb:
                v_ = (u[0] + di, u[1] + dj)
                if 0 <= v_[0] < n and 0 <= v_[1] < n and not occ[v_]:
                    nd = D[u] + w
                    if nd < D.get(v_, 1e9):
                        D[v_] = nd
                        prev[v_] = u
                        heapq.heappush(pq, (nd + math.hypot(v_[0] - g[0], v_[1] - g[1]), v_))
        if g not in prev and g != s0:
            return None
        path = [g]
        while path[-1] != s0:
            path.append(prev[path[-1]])
        path = path[::-1]
        L = 0.0
        for a_, b_ in zip(path[:-1], path[1:]):
            L += math.hypot(b_[0] - a_[0], b_[1] - a_[1]) * C
            if L >= 1.2:
                return (x - R + (b_[0] + 0.5) * C, y - R + (b_[1] + 0.5) * C)
        return (x - R + (path[-1][0] + 0.5) * C, y - R + (path[-1][1] + 0.5) * C)

    def _carrot(self, x, y, gx, gy, P):
        """Gap selection (VFH-lite): the free direction closest to the goal direction. Directions
        every 10 deg; a direction is free if the straight segment of length min(LOOK, goal dist)
        keeps the footprint radius + 0.1 m from every remembered obstacle point."""
        dg = math.hypot(gx - x, gy - y)
        L = min(LOOK, dg)
        a_g = math.atan2(gy - y, gx - x)
        if L < 0.3:
            return (gx, gy)
        rad = self.dwa.rad + 0.1
        Q = P[:, :2] - np.array([x, y]) if len(P) else np.zeros((0, 2))
        best, best_c = None, 1e9
        for da in np.radians(np.arange(-120, 121, 10)):
            a = a_g + da
            u = np.array([math.cos(a), math.sin(a)])
            if len(Q):
                s_ = np.clip(Q @ u, 0, L)
                d = np.hypot(*(Q - s_[:, None] * u).T)
                if (d < rad).any():
                    continue
            c = abs(da) + 0.3 * abs(wrap(a - getattr(self, "a_prev", a_g)))
            if c < best_c:
                best, best_c = a, c
        if best is None:
            return (x + L * math.cos(a_g), y + L * math.sin(a_g))
        self.a_prev = best
        return (x + L * math.cos(best), y + L * math.sin(best))

    def act(self, obs):
        t, bs = obs["t"], obs["bs"]
        x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
        gx, gy = obs["goal_xy"]  # = pose + relative goal (GPS+compass)
        dt = obs["dt"]
        self.k += 1
        if self.k % 5 == 1:  # 10 Hz laser + DWA
            self.mem.append((t, self.laser.scan(bs["pos"], yaw, self.floor_z)))
            P = self._points(t, x, y)
            c_ = self._astar_carrot(x, y, gx, gy, P) if self.k % 10 == 1 or not hasattr(self, "c_last") else self.c_last
            self.c_last = c_ = c_ if c_ is not None else self._carrot(x, y, gx, gy, P)
            self.dw = self.dwa.plan(x, y, yaw, c_, P)
            self.P = P
        dw = self.dw
        if self.mode == "drive":
            self.blocked_t = 0.0 if dw.get("fwd_ok", dw["admissible"]) else self.blocked_t + dt
            if self.blocked_t >= BLOCK_S:
                self.n_recover += 1
                if self.n_recover > MAX_RECOVER:
                    return dict(twist=np.zeros(4), giveup=True, info="no admissible motion: gives up")
                self.mode, self.mode_t = "backup", t
        if self.mode == "drive":
            tw = np.array([dw["v"], 0.0, 0.0, dw["w"]])
        elif self.mode == "backup":
            tw = np.array([self.v_back, 0.0, 0.0, 0.0])
            P = getattr(self, "P", np.zeros((0, 3)))
            rel = P[:, :2] - np.array([x, y]) if len(P) else np.zeros((0, 2))
            bx = rel @ np.array([math.cos(yaw), math.sin(yaw)])
            by = rel @ np.array([-math.sin(yaw), math.cos(yaw)])
            rear = np.any((bx < 0) & (bx > -(FOOT[self.kind][0] + 0.5)) & (np.abs(by) < FOOT[self.kind][1] + 0.1))
            if rear or t - self.mode_t > BACKUP_S:
                left = np.sum((by > 0) & (np.abs(bx) < 2.0))
                right = np.sum((by < 0) & (np.abs(bx) < 2.0))
                self.rot_dir = 1.0 if left <= right else -1.0
                self.mode, self.mode_t = "rotate", t
        else:  # rotate toward the freer side
            tw = np.array([0.0, 0.0, 0.0, self.rot_dir * max(0.4, 0.7 * self.wz_cap)])
            if t - self.mode_t > ROTATE_S:
                self.mode, self.blocked_t = "drive", 0.0
        return dict(twist=tw, info=f"{self.mode}  v={tw[0]:.2f} w={tw[3]:.2f}")


class XFeasLocal(LocalNav):
    """LocalNav + XFeas (ours) deciding speed / declaring the area ahead infeasible."""

    name = "xfeas"
    label = "+ XFeas (ours): decides speed / avoids"

    def setup(self, ctx):
        super().setup(ctx)
        from xfeas_decider import XFeasDecider

        self.dec = XFeasDecider()
        self.dec.setup(
            stage=ctx["stage"], rep=ctx["rep"], world=ctx["world"], rob=ctx["rob"], spec=ctx["spec"], kind=ctx["kind"]
        )

    def reset(self, ep):
        super().reset(ep)
        self.dec.reset(dict(speed=self.v))
        self.last = ""

    def act(self, obs):
        base = LocalNav.act(self, obs)
        if base.get("giveup"):
            return base
        d = self.dec.decide(
            dict(t=obs["t"], bs=obs["bs"], robot=obs["rob"], recovery="drive" if self.mode == "drive" else self.mode),
            dict(twist=base["twist"]),
        )
        if d.get("avoid"):
            self.avoid(d.get("avoid_pts", []), obs["t"])
        if self.dec.log:
            e = self.dec.log[-1]
            self.last = (
                f"XFeas: speed cap {e['v']:.2f} m/s  P={e['p']:.2f}"
                if str(e.get("action", "")).startswith("cap")
                else f"XFeas: {e.get('action')}  P={(e.get('p') or 0):.2f}"
            )
        return dict(twist=np.asarray(d["twist"], float), info=self.last or base["info"], decisions=self.dec.log)


class XFeasLocalV2(LocalNav):
    """Ours, v5 decision layer: XFeas queried at 4 Hz on the head camera; the best-utility speed caps
    the stock command; when even the best option's P_success (LCB) is below tau -- calibrated OFFLINE
    on held-out object episodes (calibrate_avoid.py -> avoid_threshold.json), never on test scenes --
    the band ahead is marked infeasible and the robot commits to a 2.5 s detour around it (gap selection)."""

    name = "xfeas5"
    label = "+ XFeas v5 (ours): speed + hazard avoidance"
    PERIOD, DETOUR_S = 0.25, 2.5

    def setup(self, ctx):
        super().setup(ctx)
        import json, os
        from xfeas_decider import XFeasDecider

        self.dec = XFeasDecider()
        self.dec.setup(
            stage=ctx["stage"], rep=ctx["rep"], world=ctx["world"], rob=ctx["rob"], spec=ctx["spec"], kind=ctx["kind"]
        )
        self.tau = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "avoid_threshold.json")))[
            "tau"
        ]
        print(f"[xfeas5] tau={self.tau:.2f}", flush=True)

    def reset(self, ep):
        super().reset(ep)
        self.dec.reset(dict(speed=self.v))
        self.cap, self.t_next, self.detour_until, self.log, self.last = None, 2.0, -1.0, [], ""

    def _band(self, bs):
        c, s_ = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        half = 0.5 * max(self.dec.desc["bbox_xyz_m"][1], 0.4) + 0.3
        return [
            (float(bs["pos"][0] + d * c - l * s_), float(bs["pos"][1] + d * s_ + l * c))
            for d in np.arange(0.4, 2.51, 0.1)
            for l in np.arange(-half, half + 1e-6, 0.1)
        ]

    def act(self, obs):
        t, bs, rob, sp, dec = obs["t"], obs["bs"], obs["rob"], self.spec, self.dec
        dec.buf.tick(rob, bs)
        if dec.desc is None:
            dec.desc = rob.descriptor(self.ctx["stage"])
        if t >= self.t_next and dec.buf.full() and self.mode == "drive":
            self.t_next = t + self.PERIOD
            rgb, dep = dec._snapshot(bs)
            pb, pj = dec.buf.arrays()
            v_hi = max(min(self.v, sp.max_cmd[0]), sp.v_min + 0.01)
            ch = dec._ask(
                rgb,
                dep,
                pb,
                pj,
                dict(
                    alts=[dict(skill="go", v=[sp.v_min, v_hi], task=True)],
                    v_ref=max(sp.max_cmd[0], 0.3),
                    seed=int(t * 10),
                ),
            )
            p = float(ch.get("p_success") or 0.0)
            if (ch.get("skill") != "go" or p < self.tau) and t >= self.detour_until + 1.0:
                # one band per detour (+1 s cooldown): repeated bands at every new heading while the
                # robot turns away would wall it in
                self.avoid(self._band(bs), t)
                self.detour_until = t + self.DETOUR_S
                self.cap, act = sp.v_min, "avoid ahead"
            elif ch.get("skill") != "go" or p < self.tau:
                self.cap, act = sp.v_min, "hazard (detouring)"
            else:
                self.cap, act = float(ch["v"]), f"cap {float(ch['v']):.2f}"
            self.log.append(
                dict(
                    t=round(t, 2),
                    x=round(float(bs["pos"][0]), 2),
                    y=round(float(bs["pos"][1]), 2),
                    v=self.cap,
                    p=round(p, 3),
                    action=act,
                )
            )
            self.last = (
                f"XFeas: speed cap {self.cap:.2f} m/s  P={p:.2f}"
                if act.startswith("cap")
                else f"XFeas: hazard ahead (P={p:.2f}) -> detour"
            )
        base = LocalNav.act(self, obs)
        if base.get("giveup"):
            return dict(base, decisions=self.log)
        tw = np.asarray(base["twist"], float).copy()
        cap = self.cap if t >= self.detour_until else min(self.cap or sp.v_min, max(sp.v_min, 0.5 * self.v))
        if cap is not None and tw[0] > cap > 0:
            r = cap / tw[0]
            tw[0] = cap
            tw[3] *= r
        return dict(twist=tw, info=self.last or base["info"], decisions=self.log)


def robot_box(spec, nom):
    """gen phase 1: a robot's posture search box from its OWN spec (spec.extra['setpoint_ranges']), as offsets
    dh = body_height - nom, pitch, roll; dims the controller lacks are fixed at 0 (lo == hi)."""
    R = spec.extra.get("setpoint_ranges", {})
    if nom is None or "body_height" not in R:
        return dict(dh=[0.0, 0.0], pitch=[0.0, 0.0], roll=[0.0, 0.0])
    return dict(
        dh=[R["body_height"][0] - nom, R["body_height"][1] - nom],
        pitch=list(R.get("pitch", (0.0, 0.0))),
        roll=list(R.get("roll", (0.0, 0.0))),
    )


class XFeasPosture(LocalNav):
    """Ours, the action-feasibility layer proper: the controller is unchanged and the map-free planner still
    decides WHERE to go; at 4 Hz XFeas picks HOW the body moves -- forward speed and, where the controller
    accepts them, body height / pitch / roll setpoints -- maximising predicted success (CEM over the robot's
    setpoint box, stock posture always a candidate). No area is ever blocked."""

    name = "xfeaspos"
    label = "+ XFeas layer (ours): speed + posture setpoints"
    PERIOD = 0.25
    MARGIN = 0.05  # leave the stock posture only for a predicted gain >= 5 points (fixed a priori)

    def setup(self, ctx):
        super().setup(ctx)
        import sys
        from xfeas_decider import XFeasDecider

        self.dec = XFeasDecider()
        self.dec.setup(
            stage=ctx["stage"], rep=ctx["rep"], world=ctx["world"], rob=ctx["rob"], spec=ctx["spec"], kind=ctx["kind"]
        )
        rob = ctx["rob"]
        R = ctx["spec"].extra.get("setpoint_ranges", {})
        self.nom = getattr(sys.modules.get(type(rob).__module__), "NOMINAL_H", None)
        self.posture = self.nom is not None and "body_height" in R
        if self.posture:
            self.box = dict(
                dh=[R["body_height"][0] - self.nom, R["body_height"][1] - self.nom],
                pitch=[max(R["pitch"][0], -0.35), min(R["pitch"][1], 0.6)],
                roll=[max(R["roll"][0], -0.35), min(R["roll"][1], 0.35)],
            )
            # v11 (additive, opt-in XFEAS_ROBOT_BOX=1): the robot's OWN layer box from its spec (extra['layer_box'],
            # e.g. G1/AMO pitch up to 1.5 for the forward-lean crouch); robots without one keep the box above (Spot).
            import os as _os

            LB = ctx["spec"].extra.get("layer_box")
            if LB and _os.environ.get("XFEAS_ROBOT_BOX", "0") == "1":
                self.box = dict(
                    dh=[LB["body_height"][0] - self.nom, LB["body_height"][1] - self.nom],
                    pitch=list(LB["pitch"]),
                    roll=list(LB["roll"]),
                )
            # gen phase 1 (additive, opt-in XFEAS_ROBOT_BOX=spec): the FULL box the robot's own spec declares
            # (spec.extra['setpoint_ranges']: body height / pitch / roll). Any other value (default '0', or '1') =
            # unchanged. (Speed above the planner's: climb_methods.StraightXFeasHoldV, CLIMB_VMAX.)
            if _os.environ.get("XFEAS_ROBOT_BOX", "0") == "spec":
                self.box = robot_box(ctx["spec"], self.nom)
            # v11 (additive, opt-in): posture-departure cost / gain gate in the selector utility (select_pcost.py)
            for _k, _e in (("pcost", "XFEAS_PCOST"), ("gain_z", "XFEAS_GAINZ")):
                if float(_os.environ.get(_e, "0") or 0) > 0:
                    self.box[_k] = float(_os.environ[_e])
        else:
            self.box = dict(dh=[0.0, 0.0], pitch=[0.0, 0.0], roll=[0.0, 0.0])
        print(
            f"[xfeaspos] posture setpoints {'ON ' + str(self.box) if self.posture else 'OFF (velocity-only controller)'}",
            flush=True,
        )

    def reset(self, ep):
        super().reset(ep)
        self.dec.reset(dict(speed=self.v))
        self.cap, self.t_next, self.log, self.last = None, 2.0, [], ""
        self.sp_now = dict(dh=0.0, pitch=0.0, roll=0.0)
        # FIX (LAYER_DIAGNOSIS): posture setpoints persisted across trials (trials 2-3 started in the previous
        # trial's posture). Restore the controller's nominal posture at episode start.
        rob = self.ctx["rob"]
        if self.posture and hasattr(rob, "reset_setpoints"):
            rob.reset_setpoints()

    def act(self, obs):
        t, bs, rob, sp, dec = obs["t"], obs["bs"], obs["rob"], self.spec, self.dec
        dec.buf.tick(rob, bs)
        if dec.desc is None:
            dec.desc = rob.descriptor(self.ctx["stage"])
        if t >= self.t_next and dec.buf.full():
            self.t_next = t + self.PERIOD
            rgb, dep = dec._snapshot(bs)
            pb, pj = dec.buf.arrays()
            v_hi = max(min(self.v, sp.max_cmd[0]), sp.v_min + 0.01)
            box = dict(self.box, v=[max(sp.v_min, 0.1), v_hi], v_ref=max(sp.max_cmd[0], 0.3))
            ch = dec._ask(rgb, dep, pb, pj, dict(setpoint=box, seed=int(t * 10)))
            if float(ch["p_success"]) - float(ch["p_stock"]) < self.MARGIN:  # no clear gain: stock posture
                ch = dict(ch, dh=0.0, pitch=0.0, roll=0.0, v=ch["v_stock"], p_success=ch["p_stock"])
            self.cap = float(ch["v"])
            for k in ("dh", "pitch", "roll"):  # low-pass so posture does not jitter at 4 Hz
                self.sp_now[k] += 0.5 * (float(ch[k]) - self.sp_now[k])
            if self.posture:
                rob.command(
                    dict(body_height=self.nom + self.sp_now["dh"], pitch=self.sp_now["pitch"], roll=self.sp_now["roll"])
                )
            self.log.append(
                dict(
                    t=round(t, 2),
                    x=round(float(bs["pos"][0]), 2),
                    y=round(float(bs["pos"][1]), 2),
                    v=self.cap,
                    dh=round(self.sp_now["dh"], 3),
                    pitch=round(self.sp_now["pitch"], 3),
                    roll=round(self.sp_now["roll"], 3),
                    p=ch.get("p_success"),
                    p_stock=ch.get("p_stock"),
                )
            )
            self.last = (
                f"XFeas: v {self.cap:.2f}  dh {self.sp_now['dh']:+.2f}  pitch {self.sp_now['pitch']:+.2f}  "
                f"roll {self.sp_now['roll']:+.2f}  P={ch.get('p_success'):.2f} (stock {ch.get('p_stock'):.2f})"
            )
        base = LocalNav.act(self, obs)
        if base.get("giveup"):
            return dict(base, decisions=self.log)
        tw = np.asarray(base["twist"], float).copy()
        if self.cap is not None and tw[0] > self.cap > 0:
            r = self.cap / tw[0]
            tw[0] = self.cap
            tw[3] *= r
        return dict(twist=tw, info=self.last or base["info"], decisions=self.log)
