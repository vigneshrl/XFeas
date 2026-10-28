"""
Course x controller study (additive 2026-10-19; latest_results/course_controllers, analysis/COURSE_CONTROLLERS.md).
Registers two episode_runner scenes that run to the PLATFORM (full course), a generic hurdle-contact logging robot
factory wrapper, and a recording-only chase camera. Nothing here touches school.usd, the course builders or any
method; the scenes reuse the course builders unchanged (in-memory build, as GC0 / GC3F do).

  CC1  gym course v1 (gym_course.make_spec(0), GymCourse) == GC0: start (-23.926, 2.6), goal (-10.44, 2.6) on the
       platform, path = GC0's walking-surface path, success: base within 0.5 m of the goal and above goal_z + 0.15.
  CC3  gym course v3 (gym_course_v3.course_spec(3, 0): beam underside 1.24 m on pillars + funnel gap 0.36 m), start as
       GC3F (-24.126, 2.6), goal = task_v3 goal (-9.59, 2.6) on the platform, path = task_v3 path, same success rule.

  --robot-factory course_ctrl:make_ct    base factory from env CC_BASE (e.g. sidestep_g1:make_g1c,
       urma_robot:make_urma_g1, g1_sonic:make_g1s); adds PhysX contact reports on every robot rigid body and logs
       robot <-> course-part contacts per part group (beam, funnel, fence, rail, hump, valley, ramp, stairs, deck ...),
       split into foot links and body (non-foot) links; one json line per trial -> $CC_CONTACT_OUT. Measurement only.
  REC_CAM=ccchase   rear three-quarter chase camera that follows the walking surface (env CC_COURSE=v1|v3); eye low
       enough to stay under the v3 beam, inside the fence line. Never seen by any method.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math, os, re, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "scenes"), os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK3 = json.load(open(xfeas_paths.COURSES + "/gym_obstacle_course/v3/task_v3.json"))
LANE_Y = 2.6
PARTS = {}            # prim path -> role (filled at build)


def _split(P):
    P = np.asarray(P, float).copy()
    for i in range(1, len(P)):
        if np.hypot(*(P[i, :2] - P[i - 1, :2])) < 1e-6:
            P[i, 0] += 0.01
    return P


P1 = _split([(-23.926, 2.6, 0.009), (-22.726, 2.6, 0.009), (-22.026, 2.6, 0.179), (-12.375, 2.6, 0.179),
             (-12.365, 2.6, 0.348), (-12.065, 2.6, 0.518), (-11.765, 2.6, 0.688), (-11.465, 2.6, 0.858), (-10.44, 2.6, 0.858)])
P3 = _split(TASK3["path"])

# sections along x (v1 from gym_course_scene.SECTIONS + RISERS; v3 from task_v3)
X1F = -22.726
SEC1 = [("lane", -99.0), ("ramp", X1F), ("deck", X1F + 0.7009), ("hump 1", X1F + 2.7009), ("valley 1", X1F + 4.2564),
        ("hump 2", X1F + 5.6064), ("valley 2", X1F + 7.1618), ("deck", X1F + 8.3618), ("stairs", -12.375), ("platform", -11.465)]
SEC3 = [(s["name"], float(s["x"][0])) for s in TASK3["sections"]]
COURSE = dict(
    v1=dict(path=P1, sections=SEC1, start=(-23.926, LANE_Y), goal=(-10.44, LANE_Y), goal_z=0.846, x_stairs=-12.375, x_end=-10.44 + 1.0),
    v3=dict(path=P3, sections=SEC3, start=(float(TASK3["start"][0]), LANE_Y), goal=(float(TASK3["goal"][0]), LANE_Y),
            goal_z=float(TASK3["goal"][2]) - 0.012, x_stairs=float(TASK3["goal_foot"][0]), x_end=float(TASK3["sections"][-1]["x"][1])))


def surface_z(course, x):
    P = COURSE[course]["path"]
    return float(np.interp(x, P[:, 0], P[:, 2]))


def section_of(course, x):
    S = COURSE[course]["sections"]
    k = 0
    for i, (_, x0) in enumerate(S):
        if x >= x0:
            k = i
    return S[k][0], k


def runner_scenes():
    import stairs_scenes as SS
    out = {}
    for sid, c, hz, desc in (("CC1", "v1", (-17.0, LANE_Y), "gym course v1: ramp, deck, humps, valleys, 4 stairs, platform"),
                             ("CC3", "v3", (0.5 * sum(TASK3["crouch_x"]), LANE_Y),
                              "gym course v3: ramp, beam 1.24 m, humps, valleys, funnel 0.36 m, 4 stairs, platform")):
        C = COURSE[c]
        path = [tuple(map(float, p)) for p in C["path"]]
        SS.STAIR_SCENES[sid] = dict(path=path, floor=0.009, goal_z=C["goal_z"], desc=desc)
        L = float(np.sum(np.hypot(*np.diff(C["path"][:, :2], axis=0).T)))
        out[sid] = dict(route=None, hazard=hz, desc=desc, start=C["start"], goal=C["goal"], floor=0.009, path=path, goal_z=C["goal_z"],
                        t_max=6.0 * L + 60.0, fall_tilt=1.0, build=f"course_ctrl:build_{c}")
    return out


def _register_parts(parts):
    PARTS.clear()
    for p in parts:
        PARTS[str(p["path"])] = str(p["role"])
    print(f"[cc] {len(PARTS)} course parts registered for contact logging", flush=True)
    try:
        import rec_cam
        rec_cam.CAMS["ccchase"] = CCChase
    except Exception as e:
        print("[cc] rec cam not registered:", e, flush=True)


def build_v1(stage, sid, sc):
    import gym_course as GC
    off = GC.prepare_gym(stage)
    B = GC.GymCourse(stage)
    info = B.build(GC.make_spec(0))
    print(f"[cc] built v1 ({info['n_parts']} parts), deactivated {len(off)} in memory; robot {json.dumps(info['robot'])[:300]}", flush=True)
    _register_parts(B.parts)


def build_v3(stage, sid, sc):
    import gym_course as GC
    import gym_course_v3 as GV3
    off = GC.prepare_gym(stage)
    spec, Cls = GV3.course_spec(3, 0)
    B = Cls(stage)
    info = B.build(spec)
    print(f"[cc] built v3 ({info.get('n_parts')} parts), gap {spec['funnel']['gap']}, crouch {json.dumps(info.get('crouch', {}).get('s'))}", flush=True)
    if os.environ.get("CC_INFO"):
        json.dump(dict(crouch=info.get("crouch"), funnel=info.get("funnel"), robot=info.get("robot")), open(os.environ["CC_INFO"], "w"),
                  indent=1, default=lambda o: np.asarray(o).tolist())
    _register_parts(B.parts)


def group_of(role):
    r = role.lower()
    for g in ("beam", "funnel", "fence", "hump", "valley", "ramp"):
        if r.startswith(g) or f"_{g}" in r:
            return g
    if r.startswith("hand") or (r.startswith("plat") and "rail" in r) or "rail" in r:
        return "rail"
    if r.startswith("plat"):
        return "platform"
    if "stair" in r or "step" in r or "riser" in r or "tread" in r:
        return "stairs"
    return re.sub(r"[_\d].*$", "", r) or r


HURDLES = ("beam", "funnel", "fence", "rail")


def _role(path):
    p = path
    while p and p != "/":
        if p in PARTS:
            return PARTS[p]
        p = p.rsplit("/", 1)[0]
    return None


# ------------------------------------------------------------------ contact-logging robot factory
def make_ct(kind, prim_path, position, yaw=0.0):
    import atexit, importlib
    from pxr import UsdPhysics, PhysxSchema, Usd, PhysicsSchemaTools
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    mod, fn = os.environ["CC_BASE"].split(":")
    rob = getattr(importlib.import_module(mod), fn)(kind, prim_path, position, yaw)
    stage = omni.usd.get_context().get_stage()
    nb = 0
    for q in Usd.PrimRange(stage.GetPrimAtPath(prim_path), Usd.TraverseInstanceProxies()):
        if q.HasAPI(UsdPhysics.RigidBodyAPI) and not q.IsInstanceProxy():
            api = PhysxSchema.PhysxContactReportAPI.Apply(q)
            api.CreateThresholdAttr().Set(0.0)
            nb += 1
    HZ = float(getattr(make_ct.SPEC, "phys_hz", 200) or 200)
    CT = dict(n=0, trial=-1, out=os.environ.get("CC_CONTACT_OUT", ""), G={}, cache={})

    def _is_foot(link):
        l = link.lower()
        return any(w in l for w in ("ankle", "foot", "toe", "sole"))

    def _rec():
        G = {}
        for (g, part), v in CT["G"].items():
            e = G.setdefault(g, dict(contact_s=0.0, impulse_Ns=0.0, body_contact_s=0.0, body_impulse_Ns=0.0, links={}))
            e["steps"] = e.get("steps", set()) | v["steps"]
            e["bsteps"] = e.get("bsteps", set()) | v["bsteps"]
            e["impulse_Ns"] += v["imp"]; e["body_impulse_Ns"] += v["bimp"]
            for k, x in v["links"].items():
                e["links"][k] = e["links"].get(k, 0.0) + x
        for g, e in G.items():
            e["contact_s"] = round(len(e.pop("steps")) / HZ, 3); e["body_contact_s"] = round(len(e.pop("bsteps")) / HZ, 3)
            e["impulse_Ns"] = round(e["impulse_Ns"], 2); e["body_impulse_Ns"] = round(e["body_impulse_Ns"], 2)
            e["links"] = {k: round(x, 2) for k, x in sorted(e["links"].items(), key=lambda kv: -kv[1])[:5]}
        hz = {g: G[g] for g in HURDLES if g in G}
        return dict(trial=CT["trial"], groups=G, hurdle_contact=bool(hz), hurdles=hz, phys_hz=HZ, base=os.environ["CC_BASE"])

    def _flush(final=False):
        if CT["trial"] >= 0 and CT["out"]:
            r = _rec()
            open(CT["out"], "a").write(json.dumps(r) + "\n")
            print(f"[cccontact] trial {CT['trial']}: hurdles {json.dumps(r['hurdles'])}", flush=True)

    def _on_contact(headers, data):
        for h in headers:
            ra = str(PhysicsSchemaTools.intToSdfPath(h.actor0)); rb = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            a = str(PhysicsSchemaTools.intToSdfPath(h.collider0)); b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            if ra.startswith(prim_path) and not rb.startswith(prim_path):
                link, other = ra.split("/")[-1], b
            elif rb.startswith(prim_path) and not ra.startswith(prim_path):
                link, other = rb.split("/")[-1], a
            else:
                continue
            if other not in CT["cache"]:
                CT["cache"][other] = _role(other)
            role = CT["cache"][other]
            if role is None:
                continue
            imp = 0.0
            for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                im = data[k].impulse; imp += float(math.sqrt(im[0] ** 2 + im[1] ** 2 + im[2] ** 2))
            v = CT["G"].setdefault((group_of(role), role), dict(steps=set(), bsteps=set(), imp=0.0, bimp=0.0, links={}))
            v["steps"].add(CT["n"]); v["imp"] += imp
            if not _is_foot(link):
                v["bsteps"].add(CT["n"]); v["bimp"] += imp
            v["links"][link] = v["links"].get(link, 0.0) + imp
            if TLOG:                             # CC_CONTACT_TLOG (additive 2026-10-19): time-resolved contacts, 0.1 s bins
                kb = (CT["trial"], int((CT["n"] - CT.get("n_tel", 0)) / (0.1 * HZ)), group_of(role), role, link)
                CT["tb"][kb] = CT["tb"].get(kb, 0.0) + imp
    TLOG = os.environ.get("CC_CONTACT_TLOG", "")
    CT["tb"] = {}

    def _tflush():
        if TLOG and CT["tb"]:
            with open(TLOG, "a") as fo:
                for (tr, b, g, role, link), imp in sorted(CT["tb"].items()):
                    fo.write(json.dumps(dict(trial=tr, t_since_teleport=round(b * 0.1, 1), group=g, role=role, link=link, foot=_is_foot(link), impulse_Ns=round(imp, 4))) + "\n")
            CT["tb"] = {}
    CT["sub"] = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
    _step0, _tel0 = rob.step, rob.teleport

    def step(*aa, **kk):
        CT["n"] += 1
        if CT["out"] and CT["trial"] >= 0 and CT["n"] % int(HZ) == 0:      # live snapshot (survives a hard exit)
            json.dump(dict(_rec(), snapshot=True), open(CT["out"].replace(".jsonl", f"_t{CT['trial']}.json"), "w"))
        if TLOG and CT["n"] % int(HZ) == 0:
            _tflush()
        return _step0(*aa, **kk)

    def teleport(*aa, **kk):
        _flush()
        _tflush()
        CT.update(G={}, trial=CT["trial"] + 1, n_tel=CT["n"])
        return _tel0(*aa, **kk)
    rob.step, rob.teleport = step, teleport
    atexit.register(_flush)
    print(f"[cccontact] base {os.environ['CC_BASE']}; contact reports on {nb} robot bodies; {len(PARTS)} course parts -> {CT['out']}", flush=True)
    return rob


class _LazyFactory:
    """make_ct with a lazy .SPEC (the base module is imported only after episode_runner has started the SimulationApp)"""

    def __call__(self, *aa, **kk):
        return _make_ct(*aa, **kk)

    @property
    def SPEC(self):
        import importlib
        mod, fn = os.environ["CC_BASE"].split(":")
        return getattr(importlib.import_module(mod), fn).SPEC


_make_ct = make_ct
make_ct = _LazyFactory()


# ------------------------------------------------------------------ recording-only chase camera
class CCChase:
    """rear three-quarter chase: eye BACK m behind the (smoothed) robot x, LAT m beside the lane centre (inside the
    fence line), UP m above the walking surface under the eye; looks AHEAD m in front of the robot at surface + ZT.
    UP is low enough that the sight line passes under the v3 beam (underside 1.24 m above the deck)."""
    LAT, UP, BACK, AHEAD, ZT = -0.45, 1.45, 2.3, 0.6, 0.55

    def __init__(self):
        self.c = os.environ.get("CC_COURSE", "v1")
        self.ss = None; self.focal = 11.0
        self.xmin = -24.30                              # gym west wall at x -24.445

    def _raw(self, x):
        xe = max(x - self.BACK, self.xmin)
        eye = np.array([xe, LANE_Y + self.LAT, surface_z(self.c, xe) + self.UP])
        xt = x + self.AHEAD
        tgt = np.array([xt, LANE_Y, surface_z(self.c, x) + self.ZT])
        return eye, tgt

    def pose(self, bs):
        x = float(bs["pos"][0])
        self.ss = x if self.ss is None else self.ss + 0.10 * (x - self.ss)
        return self._raw(self.ss)

    def lane_origin(self, eye):
        return np.array([eye[0], LANE_Y, eye[2]])

    def samples(self):
        C = COURSE[self.c]
        return [(round(float(x), 2),) + self._raw(x) for x in np.arange(C["start"][0], C["goal"][0] + 0.01, 0.5)]
