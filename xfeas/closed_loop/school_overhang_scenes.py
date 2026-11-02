"""The school-course test scenes: the course cut 1.5 m after its first doorway, with the lintel at 1.10 m (SC2A_O1),
lowered to 0.85 m (SC2A_O1L), or removed (SC2A_O1N, nothing to crouch under).

Details:
School course v2a truncated after its FIRST overhang. Additive: school_ctrl /
course4_ep are imported, never modified.

Course SC2A (school_ctrl COURSES['SC2A'], experiments/3_school_course/v2a/school_course_v2a_spec.json): v2's bench is
removed, so the first obstacle on the lane is section 'low_narrow' = a doorway of two real cloakroom units (jambs, 0.55 m
gap) under a lintel (SM_literature_props_1). The FIRST OVERHANG is that lintel (spec part 'ln_lintel'; its x span and its
underside are read from the spec below, nothing is hard-coded). The geometry is built by school_ctrl.build_sc2 exactly as
for SC2A (whole course, unchanged); only the episode ends earlier:
  goal  = (lintel x_max + GOAL_PAST, lane y), GOAL_PAST = 1.5 m; strict 0.5 m success (episode_runner), relative goal, no map
  path  = the SC2A task path cut at the goal (straight lane y = -5.8, so the stock path follower drives the same line)
  t_max = school_ctrl rule 6 x L + 60 s (env O1_TMAX overrides: smoke only); hazard (logging only) = lintel centre.
Scene id 'SC2A_O1'. Needs SC_COURSE=SC2A at import (school_ctrl reads it).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import copy, os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import numpy as np

SID = "SC2A_O1"
SID_LOW, SID_NONE = "SC2A_O1L", "SC2A_O1N"
LOW_TOP = float(os.environ.get("O1_LOW_TOP", 0.85))
GOAL_PAST = 1.5


def overhang():
    import school_ctrl as SCC

    assert SCC.CUR == "SC2A", f"SC_COURSE must be SC2A (is {SCC.CUR})"
    p = next(q for q in SCC.SPEC["parts"] if q["role"] == "ln_lintel")
    lo, hi = np.asarray(p["aabb_min"], float), np.asarray(p["aabb_max"], float)
    return dict(
        x0=float(lo[0]),
        x1=float(hi[0]),
        z_under=float(lo[2]),
        under_h=float(lo[2] - SCC.FLOOR),
        y=SCC.LANE_Y,
        floor=SCC.FLOOR,
        role="ln_lintel",
        section=[v for k, v in SCC.SEC.items() if k == "low_narrow"][0],
    )


def register(ER):
    import school_ctrl as SCC
    import stairs_scenes as SS

    ER.SCENES.update(SCC.runner_scenes())  # SC2A itself (as course4_ep does)
    oh = overhang()
    gx = oh["x1"] + GOAL_PAST
    path = [p for p in SCC.PATH if p[0] < gx - 0.05] + [(gx, SCC.LANE_Y, SCC.FLOOR)]
    L = float(np.sum(np.hypot(*np.diff(np.asarray(path)[:, :2], axis=0).T)))
    sc = copy.deepcopy(ER.SCENES["SC2A"])
    sc.update(
        goal=(gx, SCC.LANE_Y),
        path=path,
        t_max=float(os.environ.get("O1_TMAX") or 6.0 * L + 60.0),
        hazard=(0.5 * (oh["x0"] + oh["x1"]), SCC.LANE_Y),
        desc=f"school course v2a to {GOAL_PAST} m past the first overhang (lintel {oh['under_h']:.2f} m, 0.55 m doorway)",
        overhang=oh,
    )
    ER.SCENES[SID] = sc
    SS.STAIR_SCENES[SID] = dict(path=path, floor=SCC.FLOOR, goal_z=None, desc=sc["desc"])
    # variants (10-27 evening, gen5b closed-loop test): same course build, same start / goal / path / t_max
    #   SC2A_O1L  the same doorway with LOW_TOP m jambs (the same cloakroom partitions, uniformly scaled as v2a scales them)
    #             and the same plank lintel seated on them -> a lintel Spot must crouch under (stock Spot top 0.98 m)
    #   SC2A_O1N  the same doorway with the lintel DEACTIVATED (jambs only): no overhang, the over-intervention control
    for sid, fn, d in (
        (SID_LOW, "build_low", f"lintel lowered to {LOW_TOP:.2f} m (jambs {LOW_TOP:.2f} m)"),
        (SID_NONE, "build_none", "lintel removed (0.55 m doorway, jambs only)"),
    ):
        v = copy.deepcopy(sc)
        v.update(
            build=f"school_overhang_scenes:{fn}",
            desc=f"school course v2a first doorway, {d}; goal {GOAL_PAST} m past it",
        )
        if sid == SID_NONE:
            v["overhang"] = dict(v["overhang"], under_h=None, removed=True)
        ER.SCENES[sid] = v
        SS.STAIR_SCENES[sid] = dict(path=path, floor=SCC.FLOOR, goal_z=None, desc=v["desc"])
    print(
        f"[school_overhang_scenes] {SID}: start {sc['start']} goal {sc['goal']} L {L:.2f} m t_max {sc['t_max']:.0f} s; first overhang "
        f"{oh['role']} x {oh['x0']:.3f}..{oh['x1']:.3f}, underside {oh['under_h']:.3f} m above the floor",
        flush=True,
    )
    return sc


def _lintel_paths():
    import course_ctrl as CC

    return [p for p, r in CC.PARTS.items() if r == "ln_lintel"]


def _bbox(stage, path):
    from pxr import Usd, UsdGeom

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = bc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin(), float), np.array(r.GetMax(), float)


def build_low(stage, sid, sc):
    """school_ctrl.build_sc2 with v2a's low_narrow jamb_top = LOW_TOP (the plank is seated on the jambs by v2a's own seat())"""
    import school_ctrl as SCC
    import school_course_v2a as V

    f0 = V.make_spec_v2a

    def spec_low(**over):
        sp = f0(**over)
        sp["low_narrow"]["jamb_top"] = LOW_TOP
        # the uniformly scaled partitions are shorter across the hall, so the closure gap to the walls changes: close it with
        # the school's own cabinets / lockers / wardrobes (real_assets scale limit 0.85..1.15 kept)
        sp["low_narrow"]["closure"] = ["closet3", "bookcase3", "locker", "wardrobe1", "wardrobe"]
        sp["low_narrow"]["closure_srange"] = [0.85, 1.15]
        return sp

    V.make_spec_v2a = spec_low
    try:
        SCC.build_sc2(stage, sid, sc)
    finally:
        V.make_spec_v2a = f0
    lp = _lintel_paths()
    assert len(lp) == 1, lp
    lo, hi = _bbox(stage, lp[0])
    sc["overhang"].update(x0=float(lo[0]), x1=float(hi[0]), z_under=float(lo[2]), under_h=float(lo[2] - SCC.FLOOR))
    print(
        f"[school_overhang_scenes] {sid}: built lintel {lp[0]} x {lo[0]:.3f}..{hi[0]:.3f}, underside {lo[2] - SCC.FLOOR:.3f} m above the floor "
        f"(jamb top {LOW_TOP:.2f})",
        flush=True,
    )


def build_none(stage, sid, sc):
    """school_ctrl.build_sc2 unchanged, then the lintel prim is deactivated (and dropped from the contact table)"""
    import school_ctrl as SCC
    import course_ctrl as CC

    SCC.build_sc2(stage, sid, sc)
    lp = _lintel_paths()
    assert len(lp) == 1, lp
    stage.GetPrimAtPath(lp[0]).SetActive(False)
    CC.PARTS.pop(lp[0], None)
    print(f"[school_overhang_scenes] {sid}: lintel {lp[0]} deactivated (jambs only)", flush=True)
