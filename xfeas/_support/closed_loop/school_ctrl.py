"""
School course v2 x controller (additive 2026-10-20; analysis/SCHOOL_COURSE_V2_URMA.md). Same pattern as
xfeas/_support/closed_loop/course_ctrl.py (CC1 / CC3), which is reused unchanged: its make_ct contact-logging robot factory is used as
is; only its module-level part table / grouping are filled for this course at build time (no file is modified).

  SC2  school course v2 (school_course_v2.SchoolCourseV2(make_spec_v2()), built IN MEMORY in assets/school.usd incl. its
       in-memory prop deactivation (prepare); the wall distances are the ones measured for the published build
       (experiments/3_school_course/v2/school_course_v2_spec.json "walls"), so the build is the verified one -- the built
       part AABBs are compared against the spec at build time). Start (-27.2, -5.8) facing +x, goal (-9.32, -5.8), strict
       success = base within 0.5 m of the goal (no goal_z), no map, relative goal only. Path = the spec task path, its
       points ordered along the lane (the spec lists the oil-exit / tunnel-entry points in reverse order because the two
       0.3 m margins overlap; all points are on y = -5.8, so ordering them is the same straight line).
  contact groups: bench, low_narrow (jambs + lintel), oil (decal), tunnel (rows + roof), sidle (rows), closures (every
       closure line + the gym-doorway block).
  REC_CAM=scchase   rear three-quarter chase camera, higher than ccchase (no beam to clip): eye BACK m behind, LAT m right
       of the lane, UP m above the floor (above every closure / wardrobe top), inside the hall; recording only.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, re, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT

# SC_COURSE (additive 2026-10-20): SC2 = v2 (default, unchanged) | SC2A = v2a (school_course_v2a.py, AMO-feasible variant)
COURSES = dict(
    SC2=dict(
        spec=xfeas.paths.COURSES + "/3_school_course/v2/school_course_v2_spec.json",
        mod="school_course_v2",
        cls="SchoolCourseV2",
        fn="make_spec_v2",
        desc="school course v2: bench 0.255 m, low+narrow 0.456 x 0.985 m, oil -> tunnel 1.0 m, sidle 0.39 m",
    ),
    SC2A=dict(
        spec=xfeas.paths.COURSES + "/3_school_course/v2a/school_course_v2a_spec.json",
        mod="school_course_v2a",
        cls="SchoolCourseV2a",
        fn="make_spec_v2a",
        desc="school course v2a: no bench, low+narrow 0.55 x 1.10 m, oil -> tunnel 1.10 m, sidle 0.39 m",
    ),
)
CUR = os.environ.get("SC_COURSE", "SC2")
SPEC_JSON = COURSES[CUR]["spec"]
SPEC = json.load(open(SPEC_JSON))
TASK = SPEC["build"]["task"]
FLOOR = float(TASK["start"]["z"])
LANE_Y = float(TASK["start"]["xy"][1])
START = tuple(map(float, TASK["start"]["xy"]))
GOAL = tuple(map(float, TASK["goal"]["xy"]))
X0 = START[0]
_P = sorted([tuple(map(float, p)) for p in TASK["path"]], key=lambda p: p[0])
PATH = [p for i, p in enumerate(_P) if i == 0 or p[0] - _P[i - 1][0] > 1e-6]
SEC = {s["name"]: (X0 + s["s"][0], X0 + s["s"][1]) for s in SPEC["build"]["sections"]}
NICE_SEC = dict(control_step_over="bench", low_narrow="low+narrow", oil="oil", tunnel="tunnel", sidle_strip="sidle")
# progress labels along x: obstacle spans and the approach gaps in front of them
ZONES = [("start straight", -99.0)]
for k in [k for k in ("control_step_over", "low_narrow", "oil", "tunnel", "sidle_strip") if k in SEC]:
    a, b = SEC[k]
    ZONES.append((f"approach {NICE_SEC[k]}", max(a - 0.6, ZONES[-1][1] + 1e-3)))
    ZONES.append((NICE_SEC[k], a))
    ZONES.append((f"past {NICE_SEC[k]}", b))
ZONES.append(("goal zone", GOAL[0] - 0.5))
ORDER = {n: i for i, (n, _) in enumerate(ZONES)}


def zone_of(x):
    k = 0
    for i, (_, x0) in enumerate(ZONES):
        if x >= x0:
            k = i
    return ZONES[k][0], k


def runner_scenes():
    import stairs_scenes as SS

    sid = CUR
    desc = COURSES[CUR]["desc"]
    SS.STAIR_SCENES[sid] = dict(path=PATH, floor=FLOOR, goal_z=None, desc=desc)
    L = float(np.sum(np.hypot(*np.diff(np.asarray(PATH)[:, :2], axis=0).T)))
    tmax = float(os.environ.get("SC2_TMAX", 6.0 * L + 60.0))
    hz = (0.5 * sum(SEC["low_narrow"]), LANE_Y)
    out = {
        sid: dict(
            route=None,
            hazard=hz,
            desc=desc,
            start=START,
            goal=GOAL,
            floor=FLOOR,
            path=PATH,
            goal_z=None,
            t_max=tmax,
            fall_tilt=1.0,
            build="school_ctrl:build_sc2",
        )
    }
    return out


def group_of(role):
    r = role.lower()
    if "closure" in r or r.startswith("gymdoor"):
        return "closures"
    if r.startswith("hurdle"):
        return "bench"
    if r.startswith("ln_"):
        return "low_narrow"
    if r.startswith("oil"):
        return "oil"
    if r.startswith("tunnel"):
        return "tunnel"
    if r.startswith("sidle"):
        return "sidle"
    return re.sub(r"[_\d].*$", "", r) or r


GROUPS = ("bench", "low_narrow", "oil", "tunnel", "sidle", "closures")


def build_sc2(stage, sid, sc):
    import importlib
    import course_ctrl as CC

    C = COURSES[CUR]
    M_ = importlib.import_module(C["mod"])
    spec = getattr(M_, C["fn"])()
    B = getattr(M_, C["cls"])(stage, spec)
    prep = B.prepare()
    print(f"[sc2] prepare: deactivated {len(prep['deactivated'])} loose props in memory", flush=True)
    WALLS = {}
    for k, v in SPEC["walls"].items():
        s_, side = k.split(":")
        WALLS[(float(s_), int(side))] = v

    def walls_at(s_, side):  # same rule as render_school_course_v2.walls_at, on the published measurements
        v = [w for (k, sd), w in WALLS.items() if sd == side and abs(k - s_) <= 0.45 and w is not None]
        return min(v) if v else 3.5

    info = B.build(walls_at=walls_at)
    pj = {p["role"]: p for p in B.parts_json()}
    ref = {p["role"]: p for p in SPEC["parts"]}
    dev = 0.0
    miss = sorted(set(ref) ^ set(pj))
    for r, p in ref.items():
        if r in pj:
            dev = max(
                dev,
                float(
                    np.max(np.abs(np.r_[pj[r]["aabb_min"], pj[r]["aabb_max"]] - np.r_[p["aabb_min"], p["aabb_max"]]))
                ),
            )
    print(
        f"[sc2] built {info['n_parts']} parts; vs published spec: max AABB deviation {dev * 1000:.1f} mm, role mismatch {miss}",
        flush=True,
    )
    if os.environ.get("CC_INFO"):
        json.dump(
            dict(
                n_parts=info["n_parts"],
                sections=info["sections"],
                task=info["task"],
                aabb_dev_m=dev,
                role_mismatch=miss,
                path_used=PATH,
                deactivated=[d["path"] for d in prep["deactivated"]],
            ),
            open(os.environ["CC_INFO"], "w"),
            indent=1,
            default=lambda o: np.asarray(o).tolist(),
        )
    # contact logging (course_ctrl.make_ct, unchanged): fill its part table + grouping for this course
    CC.PARTS.clear()
    for p in B.parts:
        CC.PARTS[str(p["path"])] = str(p["role"])
    CC.group_of = group_of
    CC.HURDLES = GROUPS
    print(f"[sc2] {len(CC.PARTS)} course parts registered for contact logging; groups {GROUPS}", flush=True)
    try:
        import rec_cam

        rec_cam.CAMS["scchase"] = SCChase
    except Exception as e:
        print("[sc2] rec cam not registered:", e, flush=True)


class SCChase:
    """rear three-quarter chase, higher than ccchase: eye BACK m behind the (smoothed) robot x, LAT m right of the lane
    (towards the hall's far wall, 3.4 m away), UP m above the floor (closures <= 1.0 m, wardrobes <= 1.7 m above the
    floor); looks AHEAD m in front of the robot at floor + ZT."""

    LAT, UP, BACK, AHEAD, ZT = -1.0, 2.0, 2.9, 0.9, 0.55

    def __init__(self):
        self.ss = None
        self.focal = 11.0
        self.xmin = -27.85  # hall west wall at x ~ -28.14

    def _raw(self, x):
        xe = max(x - self.BACK, self.xmin)
        eye = np.array([xe, LANE_Y + self.LAT, FLOOR + self.UP])
        tgt = np.array([x + self.AHEAD, LANE_Y, FLOOR + self.ZT])
        return eye, tgt

    def pose(self, bs):
        x = float(bs["pos"][0])
        self.ss = x if self.ss is None else self.ss + 0.10 * (x - self.ss)
        return self._raw(self.ss)

    def lane_origin(self, eye):
        return np.array([eye[0], LANE_Y, eye[2]])

    def samples(self):
        return [(round(float(x), 2),) + self._raw(x) for x in np.arange(START[0], GOAL[0] + 0.01, 1.0)]


# ------------------------------------------------------------------ fresh-process trial k (harness only)
class _TrialAsFactory:
    """--robot-factory school_ctrl:make_ct_trial  = course_ctrl.make_ct, plus: with env SC_TRIAL_AS=k (k > 0) every teleport
    of this process (the runner's trial 0 and its spawn retries) is moved to the runner's OWN trial-k start (same rng seed
    1000*k + digits(scene id) and draw order: lateral +-10 cm, then yaw +-5 deg). Used because after a fall SONIC could not be
    re-spawned in the same process (invalid_start); each remaining trial then runs in a fresh process. Method untouched."""

    def __call__(self, kind, prim_path, position, yaw=0.0):
        import math as _m
        import course_ctrl as CC

        rob = CC.make_ct(kind, prim_path, position, yaw)
        k = int(os.environ.get("SC_TRIAL_AS", "0") or 0)
        if k > 0:
            d = int("".join(ch for ch in CUR[1:] if ch.isdigit()) or 0)
            rng = np.random.default_rng(1000 * k + d)
            lat = float(rng.uniform(-0.10, 0.10))
            dyaw = float(rng.uniform(-_m.radians(5), _m.radians(5)))
            tel0 = rob.teleport

            def teleport(x, y, yw, *aa, **kk):
                x2, y2 = x - lat * _m.sin(yw), y + lat * _m.cos(yw)
                print(
                    f"[sc2] SC_TRIAL_AS={k}: start ({x:.3f},{y:.3f},{_m.degrees(yw):.2f} deg) -> ({x2:.3f},{y2:.3f},{_m.degrees(yw + dyaw):.2f} deg)",
                    flush=True,
                )
                return tel0(x2, y2, yw + dyaw, *aa, **kk)

            rob.teleport = teleport
        return rob

    @property
    def SPEC(self):
        import course_ctrl as CC

        return CC.make_ct.SPEC


make_ct_trial = _TrialAsFactory()
