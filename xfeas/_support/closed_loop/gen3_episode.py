"""episode_runner entry for the gen3 SONIC zero-shot test (gen3 model agent, 2026-10-18; additive): everything zs_episode
registers (zsside camera, GC0 + stairs arms, posture oracle) + the v2 course GC2 (gen3_course_v2) + the gen3 arms
(gen3_methods). episode_runner itself is unchanged."""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import zs_episode  # noqa: F401  (registers zsside cam, GC0, stairs arms, posture oracle into episode_runner)
import episode_runner as ER
import gen3_course_v2 as GC2

ER.SCENES.update(GC2.runner_scenes())
ER.BUILTIN["gen3_straight"] = "gen3_methods:StraightXFeasGen3"
ER.BUILTIN["gen3_stairs"] = "gen3_methods:StairsXFeasGen3"
if __name__ == "__main__":
    ER.main()
