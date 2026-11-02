"""
Gym obstacle course v2 ("v2_crouch_funnel", xfeas/scenes/gym_course_v2.py) as an episode_runner scene GC2 (gen3 model
agent, 2026-10-18; additive -- gym_course_scene.py / GC0 are unchanged). Path / goal / sections from
scene_buckets/gym_obstacle_course/v2_crouch_funnel/task_v2.json. Success as v1: base within 0.5 m of the goal and above
goal_z + 0.15 (on the platform). Recording camera REC_CAM=gymcourse2 (the v1 CourseCam dolly, v2 ramp foot).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
TASK = json.load(open(xfeas.paths.COURSES + "/gym_obstacle_course/v2_crouch_funnel/task_v2.json"))
SID = "GC2"
LANE_Y = float(TASK["start"][1])
X_FOOT = float(TASK["sections"][1]["x"][0])  # ramp foot
PATH = [tuple(map(float, p)) for p in TASK["path"]]
P_ = np.asarray(PATH, float)
for i in range(1, len(P_)):  # split zero-length vertical segments by 1 cm (stairs follower)
    if np.hypot(*(P_[i, :2] - P_[i - 1, :2])) < 1e-6:
        P_[i, 0] += 0.01
PATH = [tuple(p) for p in P_]
GOAL = TASK["goal"]
GOAL_Z = float(GOAL[2]) - 0.012


def section_of(x, z_base=None, z_nom=0.75):
    """(label, index) of the course section containing world x (task_v2 sections; stairs/platform by height)"""
    secs = TASK["sections"]
    k = 0
    for i, s in enumerate(secs):
        if x >= s["x"][0]:
            k = i
    return secs[k]["name"], k


def surface_z(x):
    return float(np.interp(x, P_[:, 0], P_[:, 2]))


def runner_scenes():
    import stairs_scenes as SS

    SS.STAIR_SCENES[SID] = dict(
        path=PATH, floor=0.009, goal_z=GOAL_Z, desc="gym obstacle course v2 crouch+funnel (seed 0)"
    )
    L = float(np.sum(np.hypot(*np.diff(P_[:, :2], axis=0).T)))
    return {
        SID: dict(
            route=None,
            hazard=tuple(TASK["throat_xy"]),
            desc="gym course v2: ramp, crouch door leaf, humps, valleys, funnel, stairs",
            start=(float(TASK["start"][0]), LANE_Y),
            goal=(float(GOAL[0]), LANE_Y),
            floor=0.009,
            path=PATH,
            goal_z=GOAL_Z,
            t_max=6.0 * L + 60.0,
            fall_tilt=1.0,
            build="gen3_course_v2:build",
        )
    }


def build(stage, sid, sc):
    import gym_course as GC
    import gym_course_v2 as GV2

    off = GC.prepare_gym(stage)
    spec, Cls = GV2.course_spec(2, 0)
    info = Cls(stage).build(spec)
    print(
        f"[gymcourse2] built v2 ({info.get('n_parts')} parts), deactivated {len(off)}; robot {json.dumps(info.get('robot', {}))[:300]}",
        flush=True,
    )
    try:
        import rec_cam

        rec_cam.CAMS["gymcourse2"] = CourseCam2
    except Exception as e:
        print("[gymcourse2] rec cam not registered:", e, flush=True)


class CourseCam2:
    LAT, UP, BACK, AHEAD = -1.1, 2.4, 2.4, 1.0

    def __init__(self):
        self.ss = None
        self.focal = 14.0

    def _raw(self, s):
        se = max(s - self.BACK, -1.20)
        eye = np.array([X_FOOT + se, LANE_Y + self.LAT, surface_z(X_FOOT + s) + self.UP])
        tgt = np.array([X_FOOT + s + self.AHEAD, LANE_Y, surface_z(X_FOOT + s + self.AHEAD) + 0.5])
        return eye, tgt

    def pose(self, bs):
        s = float(bs["pos"][0]) - X_FOOT
        self.ss = s if self.ss is None else self.ss + 0.08 * (s - self.ss)
        return self._raw(self.ss)

    def lane_origin(self, eye):
        return np.array([eye[0], LANE_Y, eye[2]])

    def samples(self):
        return [(round(float(s), 2),) + self._raw(s) for s in np.arange(-1.2, 13.0, 0.5)]
