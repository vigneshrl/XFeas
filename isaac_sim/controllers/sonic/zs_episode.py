#!/usr/bin/env python
"""episode_runner.py for the SONIC + gen2-b zero-shot test (analysis/SONIC_GEN2B_ZEROSHOT.md). Additive 2026-10-17;
episode_runner, rec_cam and the scenes are unchanged. Registers: the gym-course scene GC0 + stairs arms
(gym_course_episode), the posture oracle (g1ctl), and a recording-only oblique SIDE camera REC_CAM=zsside whose
placement comes from env ZS_CAM (json, lane frame):
   A [x, y] lane start, yaw, floor, side (+1 left of heading; auto-picked by rec_cam's PhysX check when auto=1),
   lat (m from lane centre), z (eye above floor), z_tgt (look-at height above floor), back (eye s offset vs the robot,
   negative = behind), s_lo / s_hi (clamp of the look-at s), focal (mm).
Never seen by any method (rec_cam renders it with its own fill light, OFF for the head camera)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math, os, sys
import numpy as np
THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "scenes"),
           os.path.join(ROOT, "scenes"), os.path.join(ROOT, "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import episode_runner as ER
import gym_course_episode  # noqa: F401  (registers GC0 + stairs arms into ER)
import rec_cam
ER.BUILTIN["posture_oracle"] = "g1ctl_methods:PostureOracle"


class ZsSide:
    def __init__(self):
        c = json.loads(os.environ["ZS_CAM"])
        self.A = np.asarray(c["A"], float); self.yaw = float(c["yaw"]); self.FL = float(c["floor"])
        self.d = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.n = np.array([-self.d[1], self.d[0]]) * float(c.get("side", 1))
        self.lat, self.z, self.zt = float(c["lat"]), float(c["z"]), float(c["z_tgt"])
        self.back = float(c.get("back", 0.0)); self.s_lo = float(c.get("s_lo", 0.0)); self.s_hi = float(c.get("s_hi", 99.0))
        self.focal = float(c.get("focal", 11.0)); self.auto_side = bool(c.get("auto", 0))
        self.fixed_s = c.get("fixed_s")
        self.ys = None

    def lane_origin(self, eye):
        s = float((np.asarray(eye[:2]) - self.A) @ self.d)
        p = self.A + s * self.d
        return np.array([p[0], p[1], eye[2]])

    def pose(self, bs):
        rs = float((np.asarray(bs["pos"][:2], float) - self.A) @ self.d)
        self.ys = rs if self.ys is None else self.ys + 0.12 * (rs - self.ys)
        st = min(max(self.ys, self.s_lo), self.s_hi) if self.fixed_s is None else float(self.fixed_s)
        p = self.A + (st + self.back) * self.d
        eye = np.array([p[0] + self.lat * self.n[0], p[1] + self.lat * self.n[1], self.FL + self.z])
        q = self.A + st * self.d
        tgt = np.array([q[0], q[1], self.FL + self.zt])
        return eye, tgt

    def samples(self):
        out = []
        for rs in np.arange(self.s_lo, min(self.s_hi, self.s_lo + 8.0) + 0.01, 0.5):
            self.ys = None
            p = self.A + rs * self.d
            out.append((rs,) + tuple(self.pose(dict(pos=np.array([p[0], p[1], 0.0])))))
        self.ys = None
        return out


rec_cam.CAMS["zsside"] = ZsSide

if __name__ == "__main__":
    ER.main()
