#!/usr/bin/env python
"""episode_runner entry for the gen4 Phase 4 course evaluation (additive 2026-10-21): everything gen3_episode registers +
the full-course scenes CC1 (= gym levels L1, v1 course to the platform) / CC3 (= L2, v3 course: beam 1.24 m + funnel
0.36 m, to the platform) of course_ctrl, the v3flat-b scene CFB110 (beam 1.10 m, funnel 0.40 m, goal = foot of the
stairs) of course_flat, and the gen4 arms (gen4_methods). episode_runner itself is unchanged (EP_SKIP_TRIALS resume only)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "isaac_sim", "controllers", "sonic"), os.path.join(ROOT, "scenes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gen3_episode  # noqa: F401
import episode_runner as ER
import course_ctrl as CC
import course_flat as CF
ER.SCENES.update(CC.runner_scenes())
ER.SCENES.update(CF.runner_scenes())
# CC1F (2026-10-21): the plain v1 course (ramp, deck, humps, valleys -- no door / beam / funnel) ending at the FOOT of the
# stairs (SONIC's planner modes do not climb stairs): same build as CC1, path cut before riser 1 (x -12.375), goal 0.35 m
# before it on the deck (z 0.179), no goal_z.
import copy as _copy, stairs_scenes as _SS
_XF = -12.375 - 0.35
_c = _copy.deepcopy(ER.SCENES["CC1"])
_c["path"] = [p for p in _c["path"] if p[0] < _XF - 0.05] + [(_XF, CC.LANE_Y, 0.179)]
_c.update(goal=(_XF, CC.LANE_Y), goal_z=None, desc="gym course v1 plain (ramp, deck, humps, valleys) -> foot of the stairs")
_c["t_max"] = 6.0 * float(sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(_c["path"][:-1], _c["path"][1:]))) + 60.0
ER.SCENES["CC1F"] = _c
_SS.STAIR_SCENES["CC1F"] = dict(path=_c["path"], floor=0.009, goal_z=None, desc=_c["desc"])
# school course v2 / v2a (2026-10-21): registered only when SC_COURSE is set (school_ctrl reads it at import)
if os.environ.get("SC_COURSE"):
    sys.path.insert(0, os.path.join(ROOT, "scenes"))
    import school_ctrl as SCC
    ER.SCENES.update(SCC.runner_scenes())
import jog_scene as _JS                       # GJ1F: Spot gym L1 with the 45 deg jog, G1 goal = foot of the stairs (2026-10-22)
ER.SCENES.update(_JS.runner_scenes())
ER.BUILTIN["gen4_stairs"] = "gen4_methods:StairsXFeasGen4"
ER.BUILTIN["gen4_straight"] = "gen4_methods:StraightXFeasGen4"
if __name__ == "__main__":
    ER.main()
