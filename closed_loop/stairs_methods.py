"""
Arms for the REAL-staircase scenes (stairs_scenes.py; episode_runner and stairs_probe.py). Both arms follow the SAME
scripted path through the stairwell (the scene's polyline: approach, flight, landing U-turn, flight, upper floor) with
a pure-pursuit heading controller; they differ ONLY in the controller setpoints. No planner, no camera, no model.

  StairsStock    stock controller, its default setpoints (Spot: body 0.52 m, pitch 0; G1: pelvis 0.75 m, torso 0),
                 nominal speed (0.6 x max: Spot 0.6 m/s, G1 0.3 m/s)
  StairsOracle   ORACLE (not a method): scripted setpoints from env STAIRS_ORACLE = "h,p,v[,mode[,arg]]"
                 mode fixed    (h, p, v) for the whole run
                 mode flights  (h, p, v) on the flights, stock setpoints (and v) on the flat landings/floors
                 mode angled   (h, p, v) with the body yawed by arg deg off the stair line, walking along the line with
                               vy (Spot only; G1/AMO tracks vy poorly)
G1/AMO: |vx| < 0.1 puts the released policy in stand mode (no stepping, no turning), so vx is never commanded below 0.1.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
import numpy as np
import local_nav as LN
import stairs_scenes as SS


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class StairsStock(LN.LocalNav):
    name = "stairs_stock"
    label = "STOCK controller, default setpoints"
    LOOK = 0.6

    def setup(self, ctx):                       # no sensors: the arm never looks at anything
        self.ctx = ctx; self.kind, self.spec = ctx["kind"], ctx["spec"]
        self.g1 = self.kind.startswith("g1")

    def reset(self, ep):
        LN.LocalNav.reset(self, ep)
        self.pp = SS.PathProgress(SS.STAIR_SCENES[ep["scene"]]["path"])
        self.vmin = 0.1 if self.g1 else 0.0
        self.sp_done = False
        self.stock_sp = dict(self.spec.extra.get("default_sp", {}))

    def setpoints(self, obs, s, k):
        """(setpoint dict or None, speed, yaw offset rad)"""
        if not self.sp_done:
            self.sp_done = True
            obs["rob"].reset_setpoints()
        return None, self.v, 0.0

    def act(self, obs):
        bs = obs["bs"]
        x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
        s, e = self.pp.update(x, y)
        P, k = self.pp.point(s + self.LOOK)
        sp, v, yoff = self.setpoints(obs, s, k)
        if sp:
            obs["rob"].command(sp)
        path_dir = math.atan2(P[1] - y, P[0] - x)
        err = wrap(path_dir + yoff - yaw)
        wz = float(np.clip(1.8 * err, -self.wz_cap, self.wz_cap))
        if abs(err) > 0.5:                       # turn first (the landing U-turn)
            vx, vy = self.vmin, 0.0
        else:
            vx, vy = v * math.cos(yoff), -v * math.sin(yoff)
            vx = max(vx, self.vmin)
        return dict(twist=np.array([vx, vy, 0.0, wz]), info=self.info_str(v, s))

    def info_str(self, v, s):
        return f"stock setpoints, v {v:.2f} m/s"


class StairsOracle(StairsStock):
    name = "stairs_oracle"
    label = "ORACLE: scripted setpoints"

    def reset(self, ep):
        super().reset(ep)
        f = os.environ.get("STAIRS_ORACLE", "0.60,-0.35,0.9").split(",")
        self.h, self.p, self.vv = float(f[0]), float(f[1]), float(f[2])
        self.mode = f[3] if len(f) > 3 else "fixed"
        self.arg = float(f[4]) if len(f) > 4 else 0.0
        self.last = None
        P = self.pp.P
        self.flight = [abs(P[i + 1, 2] - P[i, 2]) > 0.05 for i in range(len(P) - 1)]
        self.label = f"ORACLE: h {self.h:.2f} pitch {self.p:+.2f} v {self.vv:.2f} ({self.mode}{' %g deg' % self.arg if self.mode == 'angled' else ''})"

    def setpoints(self, obs, s, k):
        # segment the BASE is on (lookahead point segment k may already be the next flight)
        _, kb = self.pp.point(s + 0.3)
        on_flight = self.flight[kb]
        if self.mode == "flights" and not on_flight:
            want = ("stock",)
            sp = None; v = self.v; yoff = 0.0
        else:
            want = (self.h, self.p)
            sp = dict(body_height=self.h, pitch=self.p, roll=0.0); v = self.vv
            yoff = math.radians(self.arg) if (self.mode == "angled" and on_flight) else 0.0
        if want != self.last:
            self.last = want
            if want == ("stock",):
                obs["rob"].reset_setpoints()
            else:
                obs["rob"].command(sp)
        return None, v, yoff

    def info_str(self, v, s):
        return f"oracle h {self.h:.2f} p {self.p:+.2f} v {v:.2f} {self.mode}"
