#!/usr/bin/env python
"""Episode entry point for the gym obstacle course (used by run_gym_cell.sh).

Details:
episode_runner entry for gen5 on the gym obstacle course (additive 2026-10-28): everything xfeas/_support/closed_loop/course4_ep.py
registers (CC1F = v1 fenced lane to the foot of the stairs, GC2 = v2 crouch door + funnel) + GC2F = GC2 cut at the foot
of the stairs (as CC1F is cut from CC1; G1 controllers here do not climb stairs): path ends 0.35 m before riser 1
(x -12.065), goal on the deck there, no goal_z.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys, copy

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import course4_ep  # noqa: F401
import episode_runner as ER
import stairs_scenes as SS
import gen3_course_v2 as GV2

_XF = -12.065 - 0.35
_c = copy.deepcopy(ER.SCENES["GC2"])
_c["path"] = [p for p in _c["path"] if p[0] < _XF - 0.05] + [(_XF, GV2.LANE_Y, GV2.surface_z(_XF))]
_c.update(
    goal=(_XF, GV2.LANE_Y),
    goal_z=None,
    desc="gym course v2 (ramp, crouch door leaf, humps, valleys, 0.39 m funnel) -> foot of the stairs",
)
_c["t_max"] = (
    6.0 * float(sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(_c["path"][:-1], _c["path"][1:])))
    + 60.0
)
ER.SCENES["GC2F"] = _c
SS.STAIR_SCENES["GC2F"] = dict(path=_c["path"], floor=0.009, goal_z=None, desc=_c["desc"])
ER.BUILTIN["gen5_stairs"] = "xfeas_method:StairsXFeasGen5"
if __name__ == "__main__":
    ER.main()
