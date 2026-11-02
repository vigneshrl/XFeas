#!/usr/bin/env python
"""Episode entry point for the school-course scenes (used by run_school_cell.sh).

Details:
episode_runner entry for the gen5 closed-loop test (additive 2026-10-27): everything xfeas/_support/closed_loop/course4_ep.py registers
(imported, unchanged) + the scene SC2A_O1 (school_overhang_scenes.py: school course v2a to 1.5 m past its first overhang) + the gen5
arms (xfeas_method). episode_runner itself is unchanged.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
assert os.environ.get("SC_COURSE") == "SC2A", "SC_COURSE=SC2A required"
import course4_ep  # noqa: F401  (registers the gen4 / course / school scenes; its main() is not run)
import episode_runner as ER
import school_overhang_scenes

school_overhang_scenes.register(ER)
ER.BUILTIN["gen5_stairs"] = "xfeas_method:StairsXFeasGen5"
if __name__ == "__main__":
    ER.main()
