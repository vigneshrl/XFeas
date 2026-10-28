"""
The school's REAL staircases (scenes/assets/school.usd, mesh SM_school_ladder_element_1; nothing is built or modified).
Geometry measured by stair_survey.py with PhysX raycasts (data/stairs/survey.json): the colliders are the exact
triangle mesh of the visual stairs. Every flight: riser 0.17 m, tread (going) 0.21 m, run width 1.44 m, 13 risers
between the ground-floor landing (z 0.681) and the mid landing (2.89), 13 more to the upper floor (5.10); dog-leg
(U-turn) layout, stairwell 3.07 m wide; stair slope atan(0.17/0.21) = 39 deg.

  SP  PROBE lane (oracle choice): EAST staircase, lower flight only (0.681 -> 2.89, 13 risers), x -2.6 -> 1.6 at
      y 5.34 (start on the bottom landing; its back wall is at x -3.34). A different staircase of the same type as the test one. (The east stairwell's UPPER flight has a fallen
      window frame lying on it -- InstancedSM_school_window_2_3 + glass, colliders 0.06-0.47 m above the treads -- so
      the east staircase is not used as the test.)
  ST  TEST: WEST staircase, ground floor (z 0.681) -> first/upper floor (z 5.10): from the west ground corridor
      (x -27.0; a fallen radiator lies on the corridor floor at x -26.0..-26.5) through the stairwell's double door onto the bottom landing, lower flight heading -x at y 3.71,
      U-turn on the mid landing (x -34.0..-32.2, z 2.89), upper flight heading +x at y 5.34, goal on the upper floor.

episode_runner keys (additive, used only when present): path (3-D polyline the progress/stuck rule measures along),
goal_z (success also needs the base above this floor), t_max, fall_tilt (stairs: the 39 deg slope alone tilts a
level-walking body by 0.68 rad, so the generic 0.8 rad tilt rule would call a robot standing on the stairs fallen;
here: tilt > fall_tilt OR base below the path surface + spec.min_z).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np

RISER, TREAD = 0.17, 0.21

# path vertices: (x, y, walking-surface z at that vertex)
STAIR_SCENES = {
    "SP": dict(path=[(-2.6, 5.34, 0.681), (-1.71, 5.34, 0.681), (0.81, 5.34, 2.89), (1.6, 5.34, 2.89)],
               floor=0.681, goal_z=2.89, desc="REAL school stairs (east, lower flight = probe): 13 risers 0.17 m / tread 0.21 m",
               cam_side=-1.0),
    "ST": dict(path=[(-27.0, 4.5, 0.681), (-28.4, 4.5, 0.681), (-29.2, 3.71, 0.681), (-29.68, 3.71, 0.681), (-32.2, 3.71, 2.89), (-33.1, 3.95, 2.89),
                     (-33.1, 5.10, 2.89), (-32.42, 5.34, 2.89), (-29.89, 5.34, 5.10), (-28.9, 5.34, 5.10)],
               floor=0.681, goal_z=5.10, desc="REAL school stairs (west): ground floor -> upper floor, 26 risers 0.17 m, U-turn landing"),
}


class PathProgress:
    """monotone projection onto the 3-D polyline: the current segment may advance only to the next one, so the
    parallel flights (1.6 m apart in xy, 2.2 m apart in z) never get confused."""

    def __init__(self, path):
        self.P = np.asarray(path, float)
        self.seg = np.diff(self.P[:, :2], axis=0)
        self.len = np.hypot(*self.seg.T)
        self.cum = np.r_[0.0, np.cumsum(self.len)]
        self.total = float(self.cum[-1])
        self.k = 0

    def reset(self):
        self.k = 0

    def _proj(self, k, xy):
        a = self.P[k, :2]; d = self.seg[k] / max(self.len[k], 1e-9)
        u = float(np.clip((xy - a) @ d, 0.0, self.len[k]))
        return u, float(np.hypot(*(xy - (a + u * d))))

    def update(self, x, y):
        xy = np.array([x, y], float)
        while self.k + 1 < len(self.seg):
            u, e = self._proj(self.k, xy); u2, e2 = self._proj(self.k + 1, xy)
            if u >= self.len[self.k] - 0.05 or (u2 > 0.05 and e2 < e):
                self.k += 1
            else:
                break
        u, e = self._proj(self.k, xy)
        return float(self.cum[self.k] + u), e

    def remaining(self, x, y):
        s, _ = self.update(x, y)
        return self.total - s

    def surface_z(self, s):
        s = float(np.clip(s, 0.0, self.total))
        k = int(min(np.searchsorted(self.cum, s, side="right") - 1, len(self.seg) - 1))
        u = (s - self.cum[k]) / max(self.len[k], 1e-9)
        return float(self.P[k, 2] + u * (self.P[k + 1, 2] - self.P[k, 2]))

    def point(self, s):
        s = float(np.clip(s, 0.0, self.total))
        k = int(min(np.searchsorted(self.cum, s, side="right") - 1, len(self.seg) - 1))
        u = (s - self.cum[k]) / max(self.len[k], 1e-9)
        return self.P[k] + u * (self.P[k + 1] - self.P[k]), k


def runner_scenes():
    out = {}
    for sid, sc in STAIR_SCENES.items():
        P = np.asarray(sc["path"], float)
        L = float(np.sum(np.hypot(*np.diff(P[:, :2], axis=0).T)))
        mid = P[len(P) // 2]
        out[sid] = dict(route=None, hazard=(float(mid[0]), float(mid[1])), desc=sc["desc"], start=tuple(P[0, :2]),
                        goal=tuple(P[-1, :2]), floor=sc["floor"], path=sc["path"], goal_z=sc["goal_z"],
                        t_max=6.0 * L + 60.0, fall_tilt=1.1, build="stairs_scenes:build")
    return out


def build(stage, sid, sc):
    """nothing is built (real stairs); registers the recording camera 'stairside' (env REC_CAM=stairside)"""
    try:
        import rec_cam
        rec_cam.CAMS["stairside"] = lambda: StairSide(sid)
    except Exception as e:
        print("[stairs] rec cam not registered:", e, flush=True)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class StairSide:
    """recording camera beside the flight the robot is on, perpendicular to it, at the robot's height (+0.45 m),
    dollying along; on the mid landing it looks down from above the far end of the upper flight's start.
    ST lower flight (y 3.71): camera at y 5.85 under the upper flight (the void beside the lower flight);
    ST upper flight (y 5.34): camera at y 3.05 above the lower flight; SP: camera at y 3.05 (beside the lower flight,
    under the east upper flight)."""

    def __init__(self, sid):
        self.sid = sid; self.sc = STAIR_SCENES[sid]
        self.pp = PathProgress(self.sc["path"])
        self.focal = 9.0; self.e = None; self.t = None; self._pairs = {}

    def _raw(self, x, y, z):
        s, _ = self.pp.update(x, y)
        P, k = self.pp.point(s)
        if self.sid == "SP":
            eye = np.array([x - 0.35, 3.05, z + 0.35]); tgt = np.array([x + 0.25, 5.34, z - 0.25])
        else:
            if k <= 1:                                           # corridor + doorway: from the corridor, behind-left
                eye = np.array([-25.4, 6.4, 0.681 + 1.6]); tgt = np.array([x - 0.6, y, z - 0.2])
            elif k <= 3:                                         # bottom landing + lower flight (heading -x, y 3.71)
                eye = np.array([min(x + 0.35, -28.45), 5.85, z + 0.35]); tgt = np.array([x - 0.25, 3.71, z - 0.25])
            elif k <= 6:                                         # mid landing: look down from the upper flight's side
                eye = np.array([-31.2, 4.5, 2.89 + 2.35]); tgt = np.array([-33.0, 4.5, 2.89 + 0.3])
            else:                                               # upper flight (heading +x, y 5.34)
                eye = np.array([x - 0.35, 3.05, z + 0.35]); tgt = np.array([x + 0.25, 5.34, z - 0.25])
        return eye, tgt

    def pose(self, bs):
        x, y, z = map(float, bs["pos"])
        eye, tgt = self._raw(x, y, z)
        if self.e is None:
            self.e, self.t = eye, tgt
        else:
            self.e = self.e + 0.10 * (eye - self.e); self.t = self.t + 0.10 * (tgt - self.t)
        return self.e.copy(), self.t.copy()

    def lane_origin(self, eye):
        return self._pairs.get(tuple(np.round(eye, 4)), eye)

    def samples(self):
        """(s, eye, tgt) along the path for the install-time raycast check; lane_origin = robot body point"""
        out = []; pp = PathProgress(self.sc["path"])
        for s in np.arange(0.0, pp.total + 1e-6, 0.4):
            P, _ = pp.point(s)
            self.pp = PathProgress(self.sc["path"])
            for s2 in np.arange(0.0, s + 1e-6, 0.2):              # walk the monotone projection up to s
                Q, _ = pp.point(s2); self.pp.update(Q[0], Q[1])
            eye, tgt = self._raw(P[0], P[1], P[2] + 0.5)
            self._pairs[tuple(np.round(eye, 4))] = np.array([P[0], P[1], P[2] + 0.5])
            out.append((round(float(s), 2), eye, tgt))
        self.pp = PathProgress(self.sc["path"])
        return out
