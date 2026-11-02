"""
Baseline wrappers for the comparison videos (latest_results/comparison_videos, 2026-10-15). NEW file; shared code is
not modified. Each wrapper is the released baseline adapter with the smallest change needed to run in a harness:

* VAMOSCmp = vamos_gpu0.VAMOSGPU0 (released VAMOS adapter, sidecar on GPU 0). One change: the released adapter seeds
  its sidecar with int(scene[1:]), which crashes for scene names that are not "<letter><int>" (D1g, terrain_<type>).
  Here the seed uses the digits of the scene name (D1g -> 1, as episode_runner does for its own start jitter) or,
  if there are none (terrain tiles), crc32(name) % 1000. Planner, weights, sensors, controller: unchanged.
* NavLocoCmp = navloco_spot.NavLocoSpot (released navigation-locomotion planner -> the robot's stock controller), with a
  robot-neutral label (it is also run on G1/AMO). Code path unchanged; head-camera mount = spec.cam of the robot.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import zlib
import vamos_gpu0
import navloco_spot


class VAMOSCmp(vamos_gpu0.VAMOSGPU0):
    label = "VAMOS (released planner) + stock controller"

    def act(self, obs):
        if not self.started:
            sc = str(self.ep["scene"])
            dg = "".join(ch for ch in sc[1:] if ch.isdigit())
            seed_name = "X" + (dg if dg and not sc.startswith("terrain_") else str(zlib.crc32(sc.encode()) % 1000))
            ep0 = self.ep
            self.ep = dict(ep0, scene=seed_name)
            try:
                return super().act(obs)
            finally:
                self.ep = ep0
        return super().act(obs)


class NavLocoCmp(navloco_spot.NavLocoSpot):
    name = "navloco_cmp"
    label = "navigation-locomotion (released planner) + stock controller"
