"""episode_runner entry for the gen3 SONIC zero-shot test (gen3 model agent, 2026-10-18; additive): everything zs_episode
registers (zsside camera, GC0 + stairs arms, posture oracle) + the v2 course GC2 (gen3_course_v2) + the gen3 arms
(gen3_methods). episode_runner itself is unchanged."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "isaac_sim", "controllers", "sonic")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import zs_episode  # noqa: F401  (registers zsside cam, GC0, stairs arms, posture oracle into episode_runner)
import episode_runner as ER
import gen3_course_v2 as GC2
ER.SCENES.update(GC2.runner_scenes())
ER.BUILTIN["gen3_straight"] = "gen3_methods:StraightXFeasGen3"
ER.BUILTIN["gen3_stairs"] = "gen3_methods:StairsXFeasGen3"
if __name__ == "__main__":
    ER.main()
