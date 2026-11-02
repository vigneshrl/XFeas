"""
Climb TEST scenes for episode_runner (SCENES[sid]['build'] = 'climb_scenes:build'), built IN MEMORY from the frozen
geometry in data/climb/test_scenes.json (written once by make_test_scenes.py, which stacks the mats with raycasts and
verifies heights). assets/school.usd is never modified. Explicit straight lanes in the school GYM (floor z ~0.01).

  G1  stacked school gym mats (copies of the school's own mats, exact triangle-mesh colliders), goal ON TOP
  G2  stairs, goal on the landing. REAL since 2026-10-10: a terrace of real wooden planks (SM_literature_props_1, laid
      across the lane, scale 1.00) -- 2 planks (0.080 m) per riser, the same front edges (s 3.0/3.3/3.6/3.9) and riser
      heights (0.08 m) as the frozen box stairs; the landing reaches s_goal + 0.4 (the box landing ran to 6.4 m; the
      lowest layers would otherwise hit the gym radiator at y 7.68); 3.0 m wide instead of 1.5 m. Primitive boxes only
      with XFEAS_LEGACY_PRIMITIVES=1.

Also registers the recording side camera 'climbside' in rec_cam.CAMS (env REC_CAM=climbside): a camera beside the lane,
perpendicular to it, dollying with the robot, framing the obstacle edge so the legs and the step are visible.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os
import numpy as np

ROOT = xfeas.paths.ROOT
GEOM = os.path.join(ROOT, "data", "climb", "test_scenes.json")


# climb PROBE lanes (climb_probe.py: training lane oh0003, obstacle at s = 2.0) registered as scenes so they can be
# rendered / run like the others; built with climb_lib.family -> ClimbBuilder (REAL plank stacks / terraces by default)
PROBE = {"PL20": ("oh0003", "ledge20", 2.0), "PL12": ("oh0003", "ledge12", 2.0), "PS08": ("oh0003", "stairs08", 2.0)}


def probe_scenes():
    import climb_lib as CL

    L = json.load(open(os.path.join(ROOT, "data", "overhang", "lanes.json")))["lanes"]
    out = {}
    for sid, (lid, fam, haz) in PROBE.items():
        ln = next(l for l in L if l["id"] == lid)
        P0 = np.asarray(ln["start"], float)
        yaw = float(ln["yaw"])
        d = np.array([math.cos(yaw), math.sin(yaw)])
        _, s_goal, top = CL.family(fam, haz)
        out[sid] = dict(
            route=None,
            hazard=tuple(P0 + d * haz),
            desc=f"climb probe lane {lid}: {fam} (real planks)",
            start=tuple(P0),
            goal=tuple(P0 + d * s_goal),
            floor=float(ln["floor"]),
            build="climb_scenes:build_probe",
            probe=(lid, fam, haz),
        )
    return out


def build_probe(stage, sid, sc):
    import climb_lib as CL
    import obstacle_lib as OB

    lid, fam, haz = sc["probe"]
    parts, s_goal, top = CL.family(fam, haz)
    yaw = math.atan2(*(np.asarray(sc["goal"]) - np.asarray(sc["start"]))[::-1])
    pool = OB.make_pool(stage, n=8, root=f"/World/ClimbScene_{sid}/Boxes")
    pool.place([p for p in parts if p["kind"] == "box"], np.asarray(sc["start"], float), yaw, sc["floor"])
    print(f"[climb_scenes] {sid}: {fam} on {lid}: {getattr(pool, 'log', None)}", flush=True)


def runner_scenes():
    """SCENES entries for episode_runner (start/goal/floor/hazard from the frozen geometry)."""
    try:
        extra = probe_scenes()
    except Exception as e:  # probe lanes are optional
        print("[climb_scenes] probe scenes not registered:", e, flush=True)
        extra = {}
    if not os.path.exists(GEOM):
        return extra
    G = json.load(open(GEOM))
    out = {}
    for sid, g in G.items():
        desc = g["desc"]
        if g["boxes"]:
            import real_assets as RA

            if not RA.LEGACY:
                desc = desc.replace("box stairs", "plank stairs (real school planks)")
        out[sid] = dict(
            route=None,
            hazard=tuple(g["hazard"]),
            desc=desc,
            start=tuple(g["start"]),
            goal=tuple(g["goal"]),
            floor=g["floor"],
            build="climb_scenes:build",
        )
    out.update(extra)
    return out


def build(stage, sid, sc):
    import hazard_copy as HC
    import obstacle_lib as OB

    G = json.load(open(GEOM))[sid]
    srcs = sorted({m["src"] for m in G["mats"]})
    HC.SOURCES = dict(HC.SOURCES, mat=list(dict.fromkeys(list(HC.SOURCES["mat"]) + srcs)))  # in memory only
    lib = HC.CopyLib(stage)
    for i, m in enumerate(G["mats"]):
        lib.make(f"/World/ClimbScene_{sid}/mat{i}", m["src"], m["x"], m["y"], m["yaw"], m["z_bottom_world"])
    import real_assets as RA

    if G["boxes"] and not RA.LEGACY:
        rl = RA.RealLib(stage, root=f"/World/ClimbScene_{sid}/Real")
        bs = sorted(G["boxes"], key=lambda b: b["s0"])
        geo = rl.plank_terrace(
            np.asarray(G["box_origin"], float),
            G["box_yaw"],
            G["floor"],
            [(b["s0"], b["z1"]) for b in bs],
            bs[0]["l0"],
            bs[0]["l1"],
            float(G["s_goal"]) + 0.4,
            scale=1.0,
        )
        print(f"[climb_scenes] {sid}: REAL plank terrace {geo}", flush=True)
    elif G["boxes"]:
        P0 = np.asarray(G["start"], float)
        yaw = G["yaw"]
        pool = OB.BoxPool(stage, n=max(8, len(G["boxes"])), root=f"/World/ClimbScene_{sid}/Boxes")
        pool.place(
            G["boxes"],
            np.asarray(G["box_origin"], float),
            G["box_yaw"],
            G["floor"],
            color=tuple(G.get("box_color", (0.45, 0.36, 0.28))),
            mat=G.get("box_mat"),
        )
    print(f"[climb_scenes] {sid}: {G['desc']} ({len(G['mats'])} mats, {len(G['boxes'])} boxes)", flush=True)
    try:
        import rec_cam

        rec_cam.CAMS["climbside"] = lambda: ClimbSide(G)
    except Exception as e:
        print("[climb_scenes] rec cam not registered:", e, flush=True)


def register_cam(sid):
    """recording side camera for a (baked) climb scene"""
    G = json.load(open(GEOM))[sid]
    import rec_cam

    rec_cam.CAMS["climbside"] = lambda: ClimbSide(G)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class ClimbSide:
    """side camera for the climb scenes: on the side G['cam_side'] of the lane, CAM_L m from the lane centre, eye
    0.45-0.85 m above the floor, looking perpendicular at the robot; it follows the robot along the lane (smoothed) and
    stops at the far end of the obstacle."""

    X0 = 0.0

    def __init__(self, G):
        self.G = G
        self.P0 = np.asarray(G["start"], float)
        self.yaw = G["yaw"]
        self.fl = G["floor"]
        self.d = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.n = np.array([-self.d[1], self.d[0]])
        self.side = float(G.get("cam_side", 1.0))
        self.L = float(G.get("cam_dist", 2.2))
        self.ss = None
        self.focal = 10.0
        self.s_edge = float(G["s_edge"])
        self.s_goal = float(G["s_goal"])

    def pose(self, bs):
        s = float((np.asarray(bs["pos"][:2]) - self.P0) @ self.d)
        s = min(max(s, 0.6), self.s_goal)
        self.ss = s if self.ss is None else self.ss + 0.12 * (s - self.ss)
        u = _smooth((self.ss - (self.s_edge - 1.6)) / 1.2)  # 0 approach .. 1 at the edge
        top = float(self.G.get("top", 0.2))
        ez = self.fl + 0.95 - 0.25 * u + 0.6 * top * u
        eye_xy = self.P0 + self.d * (self.ss + 0.15) + self.n * self.side * (self.L - 0.4 * u)
        tgt_xy = self.P0 + self.d * (self.ss + 0.25)
        self.focal = 10.0 - 1.5 * u
        return np.r_[eye_xy, ez], np.r_[tgt_xy, self.fl + 0.30 + 0.5 * top * u]

    def samples(self):
        return []
