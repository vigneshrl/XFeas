"""
Procedural NARROW-PASSAGE module (in memory only; assets/school.usd is never modified).

    start ----- approach (d_app m) ----->| cross wall, thickness L, ONE gap of width w |-----> goal
    two side walls (corridor half width cw) run from behind the start to past the goal, so the gap is the only way.

REAL ASSETS (default since 2026-10-10, real_assets.py): every wall is a row of duplicated school furniture with its
own meshes/materials and exact triangle-mesh colliders --
  cross wall (the one with the gap): wardrobes (BP_school_wardrobe_2/_1, the T1 object family: TEST scenes only),
      fronts facing the robot, the inner side faces EXACTLY at the gap edges; the outer piece may be turned sideways
      and may sit <= 6 cm inside the adjacent (side / hallway) wall so the closure is complete;
  side walls (N1/N2 only): rows of blue school lockers (BP_school_closet_1), fronts facing the corridor, front plane at
      the corridor half width cw.
The old primitive walls (UsdGeom.Cube + flat colour) are kept behind XFEAS_LEGACY_PRIMITIVES=1 (build_primitive) only
to reproduce old results. Coordinates: lane frame (s along the heading from the start, e to the left), converted to
world with (origin, heading).

    import sidestep_scene as NS
    info = NS.build(stage, NS.N_SCENES["N1"])       # -> dict with the boxes' world footprints (for the raycast audit)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np

# Test scenes (never used for training; keep every training lane >= 3 m away from these footprints).
# N1: G1 narrow passage in the courtyard (ground plane z=0, open sky, no school lights reach it -> real night).
N_SCENES = {
    "N1": dict(
        origin=(-15.5, 11.0),
        heading=math.pi / 2,
        floor=0.0,
        d_app=3.5,
        L=0.6,
        w=0.39,
        gap_e=0.0,
        cw=1.1,
        wall_h=2.0,
        back=1.0,
        past=3.0,
        goal_after=2.0,
        color=(0.55, 0.52, 0.48),
    ),
    # N2: Spot check (optional). Spot's collision length (1.59 m incl. the arm) exceeds its width (0.55 m), so no gap
    # favours a sidestep; this gap (0.75 m) is passable FORWARD and a sidestep cannot fit. Wider corridor so the long
    # body can turn in it.
    "N2": dict(
        origin=(-10.5, 11.0),
        heading=math.pi / 2,
        floor=0.0,
        d_app=4.0,
        L=0.6,
        w=0.75,
        gap_e=0.0,
        cw=1.6,
        wall_h=2.0,
        back=1.0,
        past=3.0,
        goal_after=2.0,
        color=(0.55, 0.52, 0.48),
    ),
}

# HW1: BLOCKED HALLWAY (G1 sidestep, scripts/stairs/hallway_check.py). A REAL school hallway: west wing, ground floor
# (floor z 0.681), the corridor section y 8.25-13.1 between the stair hall and the washroom door; walls x -27.96 /
# -24.69 measured by raycasts = 3.27 m wide. A block wall 0.6 m thick, 1.8 m high closes the ENTIRE width (each end
# embedded 0.10 m into the hallway wall) except one gap 0.39 m wide at the centre. No side walls are added.
# Same-floor distance >= 3.0 m from every test scene and overhang/climb lane (latest_results/hallway_sidestep/README.md).
N_SCENES["HW1"] = dict(
    origin=(-26.325, 8.9),
    heading=math.pi / 2,
    floor=0.681,
    d_app=2.2,
    L=0.6,
    w=0.39,
    gap_e=0.0,
    cw=1.735,
    wall_h=1.8,
    back=0.0,
    past=0.0,
    goal_after=1.25,
    color=(0.40, 0.38, 0.35),
    real_walls=True,
    cam_h=3.85,
    rec_cam="hall_cam:HallTop",
    rec_cam_name="halltop",
    hall_half=(1.635, 1.636),  # (left = -x side, right = +x side) for heading +y, raycast 2026-10-10
    cross_kinds=("wardrobe",),
)  # real wall: per side one wardrobe at the gap + one turned sideways


def lane_to_world(sc, s, e):
    ox, oy = sc["origin"]
    h = sc["heading"]
    return ox + s * math.cos(h) - e * math.sin(h), oy + s * math.sin(h) + e * math.cos(h)


def geometry(sc):
    """start / goal / boxes in the lane frame: list of (name, s0, s1, e0, e1)."""
    d, L, w, g, cw = sc["d_app"], sc["L"], sc["w"], sc["gap_e"], sc["cw"]
    s_end = d + L + sc["past"]
    t = 0.1
    boxes = [
        ("side_L", -sc["back"], s_end, cw, cw + t),
        ("side_R", -sc["back"], s_end, -cw - t, -cw),
        ("cross_L", d, d + L, g + w / 2, cw),
        ("cross_R", d, d + L, -cw, g - w / 2),
    ]
    if sc.get("real_walls"):  # HW*: the school's own hallway walls are the side walls
        boxes = boxes[2:]
    return dict(
        start=lane_to_world(sc, 0.0, 0.0),
        goal=lane_to_world(sc, d + L + sc["goal_after"], g),
        boxes=boxes,
        s_wall=(d, d + L),
    )


def build(stage, sc, root="/World/NarrowPassage"):
    import real_assets as RA

    if RA.LEGACY:
        return build_primitive(stage, sc, root)
    return build_real(stage, sc, root)


def build_real(stage, sc, root="/World/NarrowPassage"):
    """real-asset walls (see module doc) -> dict(G, pieces=[...], plan=...)"""
    import real_assets as RA
    from pxr import UsdGeom

    G = geometry(sc)
    UsdGeom.Xform.Define(stage, root)
    lib = RA.RealLib(stage, root=root + "/Real")
    h = sc["heading"]
    fz = sc["floor"]
    d = np.array([math.cos(h), math.sin(h)])
    n = np.array([-d[1], d[0]])
    d0, L, w, g, cw = sc["d_app"], sc["L"], sc["w"], sc["gap_e"], sc["cw"]
    hw = sc.get("hall_half", (cw, cw))  # (left, right) wall distance from the lane centre
    out = dict(cross=[], side=[])
    for sgn, edge, wall in ((+1, g + w / 2, hw[0]), (-1, g - w / 2, hw[1])):
        start = np.array(lane_to_world(sc, d0, edge))
        r = lib.furniture_line(
            start,
            n * sgn,
            -d,
            wall - abs(edge),
            fz,
            sc.get("cross_kinds", RA.WALL_KINDS_TEST),
            max_embed=0.06,
            srange=(0.95, 1.05),
            depth_target=L,
        )
        r.pop("recs")
        out["cross"].append(r)
    if not sc.get("real_walls"):  # courtyard: side walls of lockers, fronts facing the corridor
        s_a, s_b = -sc["back"], d0 + L + sc["past"]
        for sgn in (+1, -1):
            start = np.array(lane_to_world(sc, s_a, sgn * cw))
            r = lib.furniture_line(
                start, d, -n * sgn, s_b - s_a, fz, ("locker",), max_embed=0.3, srange=(0.95, 1.05), rotated_ok=False
            )
            r.pop("recs")
            out["side"].append(r)
    print(
        f"[narrow] real walls: cross {[c['pieces'] for c in out['cross']]} side {[len(c['pieces']) for c in out['side']]}",
        flush=True,
    )
    _register_cam(sc)
    return dict(G, real=out, boxes=[])


def _register_cam(sc):
    if sc.get("rec_cam"):  # recording-only camera of this scene (module:Class(sc)), REC_CAM=<name>
        try:
            import importlib, rec_cam

            _m, _c = sc["rec_cam"].split(":")
            rec_cam.CAMS[sc.get("rec_cam_name", "scene")] = lambda: getattr(importlib.import_module(_m), _c)(sc)
        except Exception as e:
            print("[narrow] rec cam not registered:", e, flush=True)


def build_primitive(stage, sc, root="/World/NarrowPassage"):
    """LEGACY primitive walls (UsdGeom.Cube, flat colour) -- only for reproducing results from before 2026-10-10"""
    from pxr import UsdGeom, UsdPhysics, Gf, UsdShade, Sdf

    G = geometry(sc)
    UsdGeom.Xform.Define(stage, root)
    mat = UsdShade.Material.Define(stage, root + "/Mat")
    sh = UsdShade.Shader.Define(stage, root + "/Mat/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*sc["color"]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    h = sc["heading"]
    fz = sc["floor"]
    H = sc["wall_h"]
    out = []
    for name, s0, s1, e0, e1 in G["boxes"]:
        if s1 - s0 < 1e-3 or e1 - e0 < 1e-3:
            continue
        cx, cy = lane_to_world(sc, 0.5 * (s0 + s1), 0.5 * (e0 + e1))
        cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, fz + H / 2))
        xf.AddRotateZOp().Set(math.degrees(h))
        xf.AddScaleOp().Set(Gf.Vec3d(s1 - s0, e1 - e0, H))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        out.append(dict(name=name, s=(s0, s1), e=(e0, e1)))
    if sc.get("rec_cam"):  # recording-only camera of this scene (module:Class(sc)), REC_CAM=<name>
        try:
            import importlib, rec_cam

            _m, _c = sc["rec_cam"].split(":")
            rec_cam.CAMS[sc.get("rec_cam_name", "scene")] = lambda: getattr(importlib.import_module(_m), _c)(sc)
        except Exception as e:
            print("[narrow] rec cam not registered:", e, flush=True)
    return dict(G, boxes=out)


def raycast_gap(pxq, carb, sc, heights=(0.1, 0.5, 0.9, 1.3, 1.6), root="/World/NarrowPassage"):
    """Audit: at several heights and along the passage length, cast rays ACROSS the lane (from the gap centre to each
    side) and report the free width actually seen by PhysX; and one ray down the lane centre (gap must be open)."""
    d, L, g = sc["d_app"], sc["L"], sc["gap_e"]
    h = sc["heading"]
    nx, ny = -math.sin(h), math.cos(h)
    res = []
    for z in heights:
        for s in np.linspace(d + 0.05, d + L - 0.05, 4):
            ox, oy = lane_to_world(sc, s, g)
            hl = pxq.raycast_closest(carb.Float3(ox, oy, sc["floor"] + z), carb.Float3(nx, ny, 0.0), 3.0)
            hr = pxq.raycast_closest(carb.Float3(ox, oy, sc["floor"] + z), carb.Float3(-nx, -ny, 0.0), 3.0)
            ok = bool(hl and hl.get("hit") and hr and hr.get("hit"))
            res.append(
                dict(
                    z=z,
                    s=round(float(s), 2),
                    width=round(float(hl["distance"] + hr["distance"]), 3) if ok else None,
                    left=str(hl.get("collision", "")) if hl else "",
                    right=str(hr.get("collision", "")) if hr else "",
                )
            )
    sx, sy = lane_to_world(sc, d - 1.0, g)  # starts 1 m before the wall (robot stands at s=0)
    fwd = pxq.raycast_closest(
        carb.Float3(sx, sy, sc["floor"] + 0.9), carb.Float3(math.cos(h), math.sin(h), 0.0), 1.0 + L + sc["past"] - 0.2
    )
    ws = [r["width"] for r in res if r["width"] is not None]
    return dict(
        rows=res,
        min_width=min(ws) if ws else None,
        max_width=max(ws) if ws else None,
        all_walls_ours=all(r["left"].startswith(root) and r["right"].startswith(root) for r in res),
        centre_ray_hit=str(fwd.get("collision")) if fwd and fwd.get("hit") else None,
    )


def runner_scenes():
    """episode_runner SCENES entries (explicit straight lanes) for the N* scenes."""
    out = {}
    for k, sc in N_SCENES.items():
        G = geometry(sc)
        hz = lane_to_world(sc, 0.5 * sum(G["s_wall"]), sc["gap_e"])
        out[k] = dict(
            route=None,
            hazard=hz,
            start=G["start"],
            goal=G["goal"],
            floor=sc["floor"],
            desc=f"narrow passage: gap {sc['w']:.2f} m wide x {sc['L']:.2f} m long",
        )
    return out
