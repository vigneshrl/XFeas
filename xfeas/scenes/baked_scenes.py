"""
Test scenes BAKED into assets/school.usd (scripts/xembody/bake_test_scenes.py, 2026-10-10).

The bake writes every scene's real-asset objects into the school's root layer under their usual roots
(/World/OverhangScene_D1, /World/CostScene_H2, /World/NarrowPassage/..., /World/ClimbScene_G2, ...) and a marker prim
    /World/BakedTestScenes   customData {"<sid>": ["<root>", ...], ...}
Scene builders must not build a baked scene again (episode_runner.build_scene does the check), and a scene that shares
its lane with a baked one (V2/W2 on V3's lane, T0 on T1's, H1 on H2's) gets the other scene's objects DEACTIVATED in
memory while it runs (any baked object whose bounding box touches the running scene's lane corridor).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math
import numpy as np

MARKER = "/World/BakedTestScenes"


def baked(stage):
    """-> {sid: [roots]} from the marker (empty if the file has no baked scenes)"""
    p = stage.GetPrimAtPath(MARKER)
    if not p.IsValid():
        return {}
    cd = p.GetCustomDataByKey("baked_scenes")
    try:
        return json.loads(cd) if isinstance(cd, str) else {}
    except Exception:
        return {}


def write_marker(stage, mapping, info=None):
    from pxr import UsdGeom

    p = UsdGeom.Xform.Define(stage, MARKER).GetPrim()
    p.SetCustomDataByKey("baked_scenes", json.dumps(mapping))
    if info:
        p.SetCustomDataByKey("bake_info", json.dumps(info))


def aabb(stage, roots):
    from pxr import Usd, UsdGeom

    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    for r in roots:
        p = stage.GetPrimAtPath(r)
        if not p.IsValid():
            continue
        rg = bbc.ComputeWorldBound(p).ComputeAlignedRange()
        if rg.IsEmpty():
            continue
        mn = np.minimum(mn, np.array(rg.GetMin()))
        mx = np.maximum(mx, np.array(rg.GetMax()))
    return (mn, mx) if mn[0] < 1e8 else None


def lane_hits_box(start, yaw, length, floor, box, half_w=0.55, z_h=2.0, s_margin=0.3):
    """does a straight lane corridor (half width half_w, height z_h above its floor) touch the axis-aligned box?
    (separating-axis test of the lane rectangle against the box footprint, plus z overlap)"""
    mn, mx = box
    if mx[2] < floor + 0.02 or mn[2] > floor + z_h:
        return False
    d = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-d[1], d[0]])
    P0 = np.asarray(start, float)[:2] - d * s_margin
    L = length + 2 * s_margin
    rect = np.array([P0 + n * half_w, P0 - n * half_w, P0 + d * L + n * half_w, P0 + d * L - n * half_w])
    bx = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), d, n):
        a, b = rect @ ax, bx @ ax
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def prepare(stage, sid, sc_geom):
    """called before building scene `sid`: -> (is_baked, deactivated_roots). sc_geom = dict(start, yaw, len, floor)."""
    B = baked(stage)
    off = []
    for other, roots in B.items():
        if other == sid:
            continue
        box = aabb(stage, roots)
        if box is not None and lane_hits_box(sc_geom["start"], sc_geom["yaw"], sc_geom["len"], sc_geom["floor"], box):
            for r in roots:
                p = stage.GetPrimAtPath(r)
                if p.IsValid() and p.IsActive():
                    p.SetActive(False)
                    off.append(r)
    return sid in B, off
