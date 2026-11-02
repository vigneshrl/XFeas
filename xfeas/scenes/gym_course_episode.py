#!/usr/bin/env python
"""episode_runner.py with the gym obstacle course scene GC0 registered (gym_course_scene.py). Additive 2026-10-16:
episode_runner itself is unchanged; this only adds the scene + the stairs path-follower arms, then calls its main().
  methods: stairs_stock (stock setpoints, path follower), stairs_gen2:StairsGen2 (gen2 layer on the same follower)"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import episode_runner as ER
import gym_course_scene as GS

ER.SCENES.update(GS.runner_scenes())
ER.BUILTIN.setdefault("stairs_stock", "stairs_methods:StairsStock")
ER.BUILTIN["course_gen2"] = "stairs_gen2:StairsGen2"
if __name__ == "__main__":
    ER.main()
