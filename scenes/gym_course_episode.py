#!/usr/bin/env python
"""episode_runner.py with the gym obstacle course scene GC0 registered (gym_course_scene.py). Additive 2026-10-16:
episode_runner itself is unchanged; this only adds the scene + the stairs path-follower arms, then calls its main().
  methods: stairs_stock (stock setpoints, path follower), stairs_gen2:StairsGen2 (gen2 layer on the same follower)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "scenes"), os.path.join(ROOT, "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import episode_runner as ER
import gym_course_scene as GS
ER.SCENES.update(GS.runner_scenes())
ER.BUILTIN.setdefault("stairs_stock", "stairs_methods:StairsStock")
ER.BUILTIN["course_gen2"] = "stairs_gen2:StairsGen2"
if __name__ == "__main__":
    ER.main()
