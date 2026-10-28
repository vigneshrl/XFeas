"""
GYM OBSTACLE COURSE as an episode_runner scene (additive 2026-10-16, G1 + gen2-b course study).

  GC0   gym_course.make_spec(0), built IN MEMORY by gym_course.GymCourse (scenes/assets/school.usd is never written).
        Start (-23.926, 2.6) on the gym floor facing +x, goal (-10.44, 2.6) on the 0.846 m platform; success needs the
        base within 0.5 m of the goal AND above goal_z + 0.15 (i.e. on the platform).

The scene is also registered in stairs_scenes.STAIR_SCENES so the stairs path follower (stairs_methods.StairsStock,
stairs_gen2.StairsGen2) follows the course centreline unchanged. Recording camera REC_CAM=gymcourse: a high
rear three-quarter view (-y side), dollying along with the robot: eye 2.4 m above the walking surface, 1.1 m beside the
lane centre (0.43 m outside the fence line), 2.4 m behind; the line of sight to the feet clears the fence's top rail.

Sections along the lane (s = x - ramp foot x, from the build's layout, verify_seed0.json):
  lane < 0 <= ramp < 0.70 <= deck < 2.70 <= hump1 < 4.26 <= valley1 < 5.61 <= hump2 < 7.16 <= valley2 < 8.36
  <= deck2 < 10.36 = riser 1, risers every 0.30 m, platform from 11.26.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "scenes"), os.path.join(ROOT, "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SID = "GC0"
X_FOOT = -22.726                     # ramp foot (lane s = 0)
LANE_Y = 2.6
RISERS = [10.3618, 10.6618, 10.9618, 11.2618]
TOPS = [0.0090, 0.1788, 0.3485, 0.5183, 0.6880, 0.8578]      # floor, deck, step1..3, platform (walking z, + floor 0.009)
SECTIONS = [("lane", -9.0), ("ramp", 0.0), ("deck", 0.7009), ("hump 1", 2.7009), ("valley 1", 4.2564),
            ("hump 2", 5.6064), ("valley 2", 7.1618), ("deck 2", 8.3618)]
# 3-D walking-surface path (build.robot.path); the zero-length vertical segment at the first riser is split by 1 cm
PATH = [(-23.926, 2.6, 0.009), (-22.726, 2.6, 0.009), (-22.026, 2.6, 0.179), (-12.375, 2.6, 0.179),
        (-12.365, 2.6, 0.348), (-12.065, 2.6, 0.518), (-11.765, 2.6, 0.688), (-11.465, 2.6, 0.858), (-10.44, 2.6, 0.858)]
GOAL_Z = 0.846


def section_of(x, z_base=None, z_nom=0.75):
    """section label of a base position (x world; z_base pelvis world z for the stairs)"""
    s = x - X_FOOT
    if s >= RISERS[0] - 0.15 and z_base is not None:
        step = int(np.argmin([abs((z_base - z_nom) - t_) for t_ in TOPS]))   # walking level from the pelvis height
        if step >= 5:
            return "platform", 13
        if step >= 2:
            return f"stairs: on step {step - 1} (riser {step - 1} climbed)", 8 + step - 1
        return "stairs: at riser 1 (not climbed)", 8
    k = max(i for i, (_, s0) in enumerate(SECTIONS) if s >= s0)
    return SECTIONS[k][0], k


def surface_z(x):
    P = np.asarray(PATH, float)
    return float(np.interp(x, P[:, 0], P[:, 2]))


def runner_scenes():
    import stairs_scenes as SS
    SS.STAIR_SCENES[SID] = dict(path=PATH, floor=0.009, goal_z=GOAL_Z, desc="gym obstacle course (seed 0)")
    P = np.asarray(PATH, float)
    L = float(np.sum(np.hypot(*np.diff(P[:, :2], axis=0).T)))
    return {SID: dict(route=None, hazard=(-17.0, LANE_Y), desc="gym obstacle course: ramp, deck, humps, valleys, 4 stairs",
                      start=(-23.926, LANE_Y), goal=(-10.44, LANE_Y), floor=0.009, path=PATH, goal_z=GOAL_Z,
                      t_max=6.0 * L + 60.0, fall_tilt=1.0, build="gym_course_scene:build")}


def build(stage, sid, sc):
    import gym_course as GC, json
    off = GC.prepare_gym(stage)
    spec = GC.make_spec(0)
    B = GC.GymCourse(stage)
    info = B.build(spec)
    print(f"[gymcourse] built {info['n_parts']} parts, deactivated {len(off)} in memory; robot {json.dumps(info['robot'])}", flush=True)
    try:
        import rec_cam
        rec_cam.CAMS["gymcourse"] = CourseCam
    except Exception as e:
        print("[gymcourse] rec cam not registered:", e, flush=True)


class CourseCam:
    LAT, UP, BACK, AHEAD = -1.1, 2.4, 2.4, 1.0

    def __init__(self):
        self.ss = None; self.focal = 14.0

    def _raw(self, s):
        se = max(s - self.BACK, -1.40)                       # gym west wall at x -24.445
        zs = surface_z(X_FOOT + s)
        eye = np.array([X_FOOT + se, LANE_Y + self.LAT, zs + self.UP])
        tgt = np.array([X_FOOT + s + self.AHEAD, LANE_Y, surface_z(X_FOOT + s + self.AHEAD) + 0.5])
        return eye, tgt

    def pose(self, bs):
        s = float(bs["pos"][0]) - X_FOOT
        self.ss = s if self.ss is None else self.ss + 0.08 * (s - self.ss)
        return self._raw(self.ss)

    def lane_origin(self, eye):
        return np.array([eye[0], LANE_Y, eye[2]])

    def samples(self):
        out = []
        for s in np.arange(-1.2, 12.3, 0.5):
            e, t = self._raw(s)
            out.append((round(float(s), 2), e, t))
        return out
