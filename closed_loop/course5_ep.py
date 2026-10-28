#!/usr/bin/env python
"""episode_runner entry for the gen5 closed-loop test (additive 2026-10-27): everything closed_loop/course4_ep.py registers
(imported, unchanged) + the scene SC2A_O1 (school_o1.py: school course v2a to 1.5 m past its first overhang) + the gen5
arms (gen5_methods). episode_runner itself is unchanged."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
assert os.environ.get("SC_COURSE") == "SC2A", "SC_COURSE=SC2A required"
import course4_ep  # noqa: F401  (registers the gen4 / course / school scenes; its main() is not run)
import episode_runner as ER
import school_o1
school_o1.register(ER)
ER.BUILTIN["gen5_stairs"] = "gen5_methods:StairsXFeasGen5"
if __name__ == "__main__":
    ER.main()
