"""
VAMOS for baselines round 1 = cmp_methods.VAMOSCmp (released vamos_plugin.VAMOS, sidecar on GPU 0, scene-name seed
fix) with LOGGING ONLY added; planning, weights, sensors, controller and config are unchanged:
  * the sidecar's own debug dump (released adapter: <dbg>/predNNN_maps.npz = the 33x33 heightmap, the VALUE MAP the
    re-ranking used, the per-candidate costs and the candidate paths in the base frame; predNNN_value.png = the
    release's traversability_estimator visualisation) is pointed at our data dir instead of the adapter's evidence/;
  * for every planning step (a VLM prediction that produced a path) we keep the robot pose the plan was made at and
    the world-frame path that was sent to the carrot follower, so the value map can be drawn in the world frame.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, os
import numpy as np
import cmp_methods


class VAMOSR1(cmp_methods.VAMOSCmp):
    label = "VAMOS (released full stack) + Spot stock controller"

    def reset(self, ep):
        k = os.environ.get("R1_TRIAL")  # fresh-process trial k under episode_runner (--trials 1):
        if k not in (None, ""):  # the sidecar seed / evidence name use the runner's trial k
            ep = dict(ep, trial=int(k))
        super().reset(ep)
        root = os.environ.get("VAMOS_R1_MAPDIR")
        if root:
            self.dbg = root
        self.plans, self._nsel = [], 0

    def act(self, obs):
        n0 = len(self.log)
        r = super().act(obs)
        for e in self.log[n0:]:
            if "sel" in e:  # a planning step that published a path
                bs = obs["bs"]
                self.plans.append(
                    dict(
                        k=self._nsel,
                        t=e["t"],
                        x=float(bs["pos"][0]),
                        y=float(bs["pos"][1]),
                        z=float(bs["pos"][2]),
                        yaw=float(bs["yaw"]),
                        path_w=[[round(float(v), 3) for v in q[:2]] for q in self.wps],
                        v_sel=e.get("v_sel"),
                        v_min=e.get("v_min"),
                        v_max=e.get("v_max"),
                        sel=e.get("sel"),
                        n_proj=e.get("n_proj"),
                    )
                )
                self._nsel += 1
                try:
                    json.dump(self.plans, open(os.path.join(self.dbg, "plans.json"), "w"))
                except Exception as ex:
                    print("[vamos_r1] plans.json not written:", ex, flush=True)
        return r
