"""
Overhang evaluation scenes, built IN MEMORY by episode_runner (SCENES[sid]['build'] = 'overhang_scenes:build').
assets/school.usd is never modified. Each scene = one object laid across an explicit straight lane.

  V1  VALIDATION scene (used to decide selector changes, never D1): a seen object type (door leaf type 6) at a
      validation-zone lane (data/overhang/lanes.json 'val'), underside 0.93 m (Spot crouch band 0.88-1.01 m).
  T1  HELD-OUT TEST scene, chosen before any data existed (lanes.json reserved 'Ta', upper-floor east corridor; no
      training lane within 3 m): an object type that is in NO data set -- a wardrobe (BP_school_wardrobe_1) lying
      flat, raised so its underside is 0.93 m.
  V3  (REAL since 2026-10-10) a real wooden plank (SM_literature_props_1) laid flat across the lane, uniform scale 1.15
      -> 0.575 m deep x 3.45 m wide x 0.046 m, underside 0.91 m (was a primitive slab 0.6 x 2.4 x 0.12 m).
  H1/H2/W2 (REAL since 2026-10-10, build_cost): H2/W2 = real plank 0.425 m deep (scale 0.85) at underside 0.92 m, then a
      stack of 3 real planks (top 0.11 m, 0.47 m deep, front edge at s = 4.0) instead of the 11 cm primitive curb;
      H1 = a real plank lintel at 1.30 m (the primitive posts are not rebuilt: standing passes either way).
  NO FLOATING OBJECTS (2026-10-10): every raised plank rests on two real support columns beside the lane (inner faces
  0.65 m from the lane centre, so the passage stays open): low school cabinets / bookcases / desks, if needed standing
  on a pile of real planks, tops exactly at the underside height (real_assets.RealLib.support_column).
  The primitive versions remain behind XFEAS_LEGACY_PRIMITIVES=1.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np

OBJ = dict(
    V1=("door", "BP_school_door_1/SM_school_door_6_2/InstancedSM_school_door_6_2"),
    V2=("door", "BP_school_door_1/SM_school_door_6_2/InstancedSM_school_door_6_2"),
    V3=("prim", None),
    T1=("wardrobe", "BP_school_wardrobe_1"),
)
S_FRONT = 4.0


D1_OLD_DOOR = "/World/School/Geom/BP_school_door_5/SM_school_door_3_3/InstancedSM_school_door_3_04"


def _lowest_over(A, P0, d, n, floor, s_lo, s_hi, l_lo, l_hi):
    """lowest object vertex (height above floor) over the lane-frame window, or None"""
    s_ = (A[:, :2] - P0) @ d
    l_ = (A[:, :2] - P0) @ n
    m = (s_ >= s_lo) & (s_ <= s_hi) & (l_ >= l_lo) & (l_ <= l_hi)
    return float(A[m, 2].min() - floor) if m.any() else None


def build_d1(stage, sid, sc):
    """D1 SUPPORTED (2026-10-10; replaces the thick door BP_school_door_5 that was raised into mid-air and passed
    through both corridor walls): a real panelled school door leaf (InstancedSM_school_door_4_2, 2.58 x 0.73 m) laid
    flat across the corridor at D1's place, underside sc['u'] over the lane, resting on
      * the open side: a low school cabinet (BP_school_closet_3), inner face 0.65 m from the lane centre;
      * the wall side (wall 0.72 m from the lane centre): a narrow school shelf unit (SM_school_closet_4, 0.23 m)
        lying on its side against the wall, inner face 0.48 m from the lane centre;
    both column tops = the leaf's lowest vertex over them. Lane moved 0.15 m away from the wall (x -4.15) so Spot
    keeps >= 0.2 m to the narrow support. Old raised door: D1_LEGACY_DOOR=1 (episode_runner)."""
    import real_assets as RA

    old = stage.GetPrimAtPath(D1_OLD_DOOR)  # the floating raised thick door must not coexist with the new D1
    if old.IsValid() and old.IsActive():
        old.SetActive(False)
    lib = RA.RealLib(stage, root=f"/World/OverhangScene_{sid}")
    P0 = np.asarray(sc["start"], float)
    g = np.asarray(sc["goal"], float)
    yaw = math.atan2(*(g - P0)[::-1])
    d = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-d[1], d[0]])
    Lw, Dw, Tw = lib.flat_dims("door_leaf")
    s0 = sc["s_front"]
    sl = sc.get("leaf_scale", 1.0)
    xy = P0 + d * (s0 + 0.5 * Dw * sl) + n * sc.get("leaf_lat", 0.0)
    rec = lib.place(
        "door_leaf", xy[0], xy[1], sc["floor"] + sc["u"], "flat", yaw + math.pi / 2 + lib.flat_yaw("door_leaf"), sl
    )
    A = RA.world_points(rec["copy"]["prim"])
    s_c = s0 + 0.5 * Dw * sl
    sup = []
    # open side (+n): low cabinet (top = lowest vertex over its footprint, iterate once for the footprint size)
    T = _lowest_over(A, P0, d, n, sc["floor"], s_c - 0.5, s_c + 0.5, 0.65, 1.03)
    c = lib.support_column(
        P0,
        yaw,
        sc["floor"],
        s_c,
        +1,
        T,
        l_in=0.65,
        kinds=("closet3", "bookcase3", "desk", "table"),
        max_lat=sc.get("room_pos", 0.45),
    )
    sup.append(c)
    # wall side (-n): shelf unit on its side
    T2 = _lowest_over(A, P0, d, n, sc["floor"], s_c - 0.95, s_c + 0.95, -0.71, -0.48)
    sup.append(lib.place_beside("shelf", P0, yaw, sc["floor"], s_c, -1, 0.48, top=T2, orient="on_side"))
    geo = dict(kind="door_leaf", u=sc["u"], scale=sl, depth=round(Dw * sl, 3), width=round(Lw * sl, 3), supports=sup)
    print(f"[overhang_scenes] {sid}: SUPPORTED door leaf {geo}", flush=True)
    return geo


def supported_planks(lib, P0, yaw, floor, s0, s1, u):
    """real plank(s) across the lane with underside u, resting on two real support columns (tops = u) outside the lane"""
    geo = lib.raised_planks(P0, yaw, floor, s0, s1, -1.2, 1.2, u)
    half = 0.5 * (geo["l_range"][1] - geo["l_range"][0])
    s_c = s0 + 0.5 * geo["depth"]
    sup = [lib.support_column(P0, yaw, floor, s_c, side, u, max_lat=half - 0.65 - 0.10) for side in (+1, -1)]
    if any(x is None for x in sup):
        raise RuntimeError(f"no real support for a plank at u={u} (half width {half:.2f})")
    geo["supports"] = sup
    return geo


def build(stage, sid, sc):
    import overhang_lib as OL
    import real_assets as RA

    cls, src = OBJ[sid]
    if cls == "prim" and not RA.LEGACY:  # V3: real plank instead of the primitive slab
        lib = RA.RealLib(stage, root=f"/World/OverhangScene_{sid}")
        P0 = np.asarray(sc["start"], float)
        g = np.asarray(sc["goal"], float)
        yaw = math.atan2(*(g - P0)[::-1])
        s0 = sc.get("s_front", S_FRONT)
        geo = supported_planks(lib, P0, yaw, sc["floor"], s0, s0 + 0.6, sc["u"])
        print(f"[overhang_scenes] {sid}: REAL plank underside {sc['u']:.2f} m, {geo}", flush=True)
        return geo
    lib = OL.OverhangLib(
        stage, np.random.default_rng(0), sources={cls: [src]} if src else {}, root=f"/World/OverhangScene_{sid}"
    )
    P0 = np.asarray(sc["start"], float)
    g = np.asarray(sc["goal"], float)
    yaw = math.atan2(*(g - P0)[::-1])
    if cls == "prim":  # V3: plain slab (a seen object family), exact underside
        src = "prim_box"
        spec = dict(
            cls="prim",
            depth=0.6,
            width=2.4,
            thick=0.12,
            color=(0.55, 0.45, 0.35),
            mat=None,
            yaw_rel=0.0,
            slant=0.0,
            roll=0.0,
            u=sc["u"],
        )
    else:
        c = lib.copies[cls][0]
        spec = dict(
            cls=cls,
            copy=c,
            name=src,
            across=True,
            yaw_rel=sc.get("obj_yaw", 0.0),
            slant=sc.get("obj_slant", 0.0),
            roll=0.0,
            scale=1.0,
            u=sc["u"],
        )
    geo = lib.place(spec, P0, yaw, sc["floor"], sc.get("s_front", S_FRONT), 0.0)
    if not RA.LEGACY:  # NO FLOATING OBJECTS (V1/V2/T1): the raised object rests on two real
        rl = RA.RealLib(stage, root=f"/World/OverhangScene_{sid}/Supports")  # support columns beside the lane
        geo["supports"] = RA.supports_under(
            rl, c["prim"], P0, yaw, sc["floor"], room=sc.get("support_room", (0.6, 0.6))
        )  # tops = the door's
        if any(x is None for x in geo["supports"]):  # lowest vertex there
            raise RuntimeError(f"{sid}: the raised object does not reach both support sites")
    print(f"[overhang_scenes] {sid}: {src} underside {sc['u']:.2f} m, {geo}", flush=True)
    return geo


# ---- crouch-COST scenes (overhang-2 study), defined AFTER the overhang-2 data was collected; never in any data set.
# Parts in the lane frame (obstacle_lib boxes). Lane for H1/H2: upper floor, inside the D1 exclusion rectangle (no
# training lane anywhere near), (-4.43, 6.5) -> (-4.56, 0.3). W2 = validation copy of H2 on the V2/V3 validation lane.
COST_PARTS = dict(
    H1=lambda OB: [
        dict(s0=3.0, s1=3.25, l0=0.75, l1=0.9, z0=-0.02, z1=1.5),
        dict(s0=3.0, s1=3.25, l0=-0.9, l1=-0.75, z0=-0.02, z1=1.5),
        dict(s0=3.0, s1=3.25, l0=-0.9, l1=0.9, z0=1.30, z1=1.5),
    ],
    H2=lambda OB: OB.overhang(2.0, 0.92, depth=0.4, thick=0.12) + OB.curb(4.0, 0.11, depth=0.25),
    W2=lambda OB: OB.overhang(2.0, 0.92, depth=0.4, thick=0.12) + OB.curb(4.0, 0.11, depth=0.25),
)


def build_cost(stage, sid, sc):
    import obstacle_lib as OB
    import real_assets as RA

    P0 = np.asarray(sc["start"], float)
    g = np.asarray(sc["goal"], float)
    yaw = math.atan2(*(g - P0)[::-1])
    if not RA.LEGACY:
        lib = RA.RealLib(stage, root=f"/World/CostScene_{sid}")
        if sid == "H1":
            geo = [supported_planks(lib, P0, yaw, sc["floor"], 3.0, 3.25, 1.30)]
        else:  # H2 / W2: plank overhang, then a 3-plank curb
            geo = [
                supported_planks(lib, P0, yaw, sc["floor"], 2.0, 2.4, 0.92),
                lib.plank_block(P0, yaw, sc["floor"], 4.0, 4.25, -1.1, 1.1, 0.11, orient="across"),
            ]
        print(f"[overhang_scenes] {sid}: REAL {geo}", flush=True)
        return geo
    pool = OB.BoxPool(stage, n=8, root=f"/World/CostScene_{sid}")
    boxes = COST_PARTS[sid](OB)
    pool.place(boxes, P0, yaw, sc["floor"], color=(0.45, 0.33, 0.22))
    print(f"[overhang_scenes] {sid}: {len(boxes)} boxes {boxes}", flush=True)
