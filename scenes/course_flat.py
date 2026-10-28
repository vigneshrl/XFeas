"""
Flat gym course v3flat x AMO (additive 2026-10-19; latest_results/amo_flat_course). episode_runner scenes that run from
the start ON the deck to the FOOT OF THE STAIRS, and a recording-only SIDE camera. Nothing here touches school.usd, the
v1/v2/v3 builders or any method; the course is gym_course_v3flat (in-memory build).

  CF110  v3flat, beam underside 1.10 m (variant A)     CF124  v3flat, beam underside 1.24 m (variant B, = v3)
         start (-24.126, 2.6) on the deck (z 0.179), goal = foot of the stairs (task goal_foot), success = base within
         0.5 m of it (episode_runner rule); path = the flat lane-centre polyline (mode-aware fall rule surface).
  REC_CAM=cfside  side camera on the -y side. Up to the beam it sits INSIDE the fence line (0.58 m off the lane centre,
         0.95 m above the deck = below both beam heights, wide lens) because from outside the fence the rails (top rail
         0.95 m above the deck) hide G1 under the beam; it then rises to a HIGH oblique view (3.3 m above the deck, 0.9 m
         off the lane centre) to look over the 1.38 m funnel wings, and comes down before the stairs. Follows the robot x
         (eye 0.6 m behind). Recording only: never seen by any method, not a physical object.
Contact logging: course_ctrl.make_ct (CC_BASE = sidestep_g1:make_g1c) with the v3flat parts registered here.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (THIS,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKET = xfeas_paths.COURSES + "/gym_obstacle_course/v3flat"
VAR = dict(CF110=("u110", 1.10), CF124=("u124", 1.24))
TASK = {sid: json.load(open(os.path.join(BUCKET, vt, "task.json"))) for sid, (vt, _) in VAR.items()
        if os.path.exists(os.path.join(BUCKET, vt, "task.json"))}
# v4 (2026-10-19, analysis/REBUILD_V4_PHASE01.md section 4): v3flat-b = v3flat u110 with the funnel sized by the oracle sweep
# (design gap 0.40 -> measured throat 0.414 m: AMO forward 0/3, AMO oracle sidestep 3/3; SONIC forward 1/3, sidestep 3/3)
BUCKET_B = xfeas_paths.COURSES + "/gym_obstacle_course/v3flat_b"
VAR_B = dict(CFB110=("u110", 1.10, 0.40))
GAP = dict(CF110=0.36, CF124=0.36, CFB110=0.40)
for _sid, (_vt, _u, _g) in VAR_B.items():
    if os.path.exists(os.path.join(BUCKET_B, _vt, "task.json")):
        TASK[_sid] = json.load(open(os.path.join(BUCKET_B, _vt, "task.json")))
LANE_Y = 2.6


def course(sid):
    T = TASK[sid]
    xs = float(T["goal_foot"][0]); dz = float(T["floor_z"])
    P = [p for p in T["path"] if p[0] < xs - 0.05] + [[xs, LANE_Y, dz]]
    P = np.asarray(P, float)
    for i in range(1, len(P)):
        if np.hypot(*(P[i, :2] - P[i - 1, :2])) < 1e-6:
            P[i, 0] += 0.01
    secs = [(s["name"], float(s["x"][0])) for s in T["sections"]]
    return dict(path=P, sections=secs, start=(float(T["start"][0]), LANE_Y), goal=(xs, LANE_Y), floor=dz, x_stairs=xs,
                beam_x=T["beam_x"], funnel_x=T["funnel_x"], x_end=xs + 0.3)


def section_of(sid, x):
    S = course(sid)["sections"]
    k = 0
    for i, (_, x0) in enumerate(S):
        if x >= x0:
            k = i
    return S[k][0], k


def runner_scenes():
    import stairs_scenes as SS
    out = {}
    for sid, (vt, u) in list(VAR.items()) + [(k, (v[0], v[1])) for k, v in VAR_B.items()]:
        if sid not in TASK:
            continue
        C = course(sid)
        path = [tuple(map(float, p)) for p in C["path"]]
        desc = f"gym course v3flat{'-b' if sid in VAR_B else ''} ({vt}): flat deck, beam {u:.2f} m, funnel {GAP[sid]:.2f} m (design) -> foot of the stairs"
        SS.STAIR_SCENES[sid] = dict(path=path, floor=C["floor"], goal_z=None, desc=desc)
        L = float(np.sum(np.hypot(*np.diff(C["path"][:, :2], axis=0).T)))
        out[sid] = dict(route=None, hazard=(0.5 * sum(C["beam_x"]), LANE_Y), desc=desc, start=C["start"], goal=C["goal"], floor=C["floor"],
                        path=path, t_max=6.0 * L + 60.0, fall_tilt=1.0, build=f"course_flat:build_{'b_' if sid in VAR_B else ''}{vt}")
    return out


def _build(stage, sid, u, gap=0.36):
    import gym_course as GC
    import gym_course_v3flat as GF
    import course_ctrl as CC
    off = GC.prepare_gym(stage)
    spec = GF.make_spec_v3flat(0, u=u, gap=gap)
    B = GF.GymCourseV3Flat(stage)
    info = B.build(spec)
    print(f"[cf] built v3flat u={u} ({info.get('n_parts')} parts, {info['n_removed']} ramp/hump/valley parts not placed), "
          f"beam s {json.dumps(info.get('crouch', {}).get('s'))}, rear deck {info['deck_rear']}", flush=True)
    if os.environ.get("CC_INFO"):
        json.dump(dict(crouch=info.get("crouch"), funnel=info.get("funnel"), robot=info.get("robot"), deck_rear=info.get("deck_rear")),
                  open(os.environ["CC_INFO"], "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    CC._register_parts(B.parts)
    try:
        import rec_cam
        rec_cam.CAMS["cfside"] = CFSide
    except Exception as e:
        print("[cf] rec cam not registered:", e, flush=True)


def build_b_u110(stage, sid, sc):
    _build(stage, sid, 1.10, gap=VAR_B["CFB110"][2])


def build_u110(stage, sid, sc):
    _build(stage, sid, 1.10)


def build_u124(stage, sid, sc):
    _build(stage, sid, 1.24)


class CFSide:
    """side camera on the -y side of the lane; (lat, eye height, target height) piecewise-linear in the robot x"""
    focal = 7.0

    def __init__(self):
        sid = os.environ.get("CF_SCENE", "CF110")
        C = course(sid)
        self.dz = C["floor"]
        b0, b1 = C["beam_x"]; f0, f1 = C["funnel_x"]
        # (x, lateral offset of the eye, eye height above the deck, target height above the deck, eye-behind, focal)
        LOW = (-0.58, 0.95, 0.62, 0.6, 7.0); HIGH = (-0.9, 3.30, 0.40, 0.6, 9.0); END = (-0.58, 1.05, 0.55, 0.9, 7.0)
        self.K = [(C["start"][0] - 1.0,) + LOW, (b1 + 0.5,) + LOW, (f0 - 1.6,) + HIGH, (f1 + 0.4,) + HIGH, (C["x_stairs"] - 0.5,) + END,
                  (C["x_stairs"] + 2.0,) + END]
        self.C = C; self.ss = None
        self.xmin = -24.30

    def _par(self, x):
        X = [k[0] for k in self.K]
        return [float(np.interp(x, X, [k[i] for k in self.K])) for i in (1, 2, 3, 4, 5)]

    def _raw(self, x):
        lat, ze, zt, back, foc = self._par(x)
        self.focal = foc
        xe = max(x - back, self.xmin)
        eye = np.array([xe, LANE_Y + lat, self.dz + ze])
        tgt = np.array([x + 0.25, LANE_Y + 0.05, self.dz + zt])
        return eye, tgt

    def pose(self, bs):
        x = float(bs["pos"][0])
        self.ss = x if self.ss is None else self.ss + 0.12 * (x - self.ss)
        return self._raw(self.ss)

    def lane_origin(self, eye):
        return np.array([eye[0], LANE_Y, eye[2]])

    def samples(self):
        return [(round(float(x), 2),) + self._raw(x) for x in np.arange(self.C["start"][0], self.C["x_stairs"] + 0.01, 0.5)]
