"""XFeas on the terrain tiles (ice, oil, sand, rubble, leaves, ...): drives straight to the goal while XFeas picks
the commands.

Details:
gen5 / gen5b learned layer on the terrain tiles: the straight-drive harness of ft_gen4_terrain.Gen4Terrain
(Spot: terrain_harness_spot.py; G1: terrain_harness_g1.py --mode feel) with xfeas_method._Gen5Layer instead of the gen4 layer.

Same decision code as the closed-loop course arm (StairsXFeasGen5): every 0.25 s the model's pick is sent directly; no
driver, trigger, stand-up rule or commit hold (the _Gen5Layer guards raise if any is reached). Only the harness glue
differs: straight drive to the goal (door_methods.straight_act) instead of the scene polyline, and no stairs-scene path
(progress s is logged as 0). Speed fraction: FT_GEN5_SPEED if set (the Spot EVORA runner passes 0.0), else the runner's.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import numpy as np  # noqa: E402
import local_nav as LN  # noqa: E402
import xfeas_method as G5  # noqa: E402
from door_methods import straight_act  # noqa: E402

SPEED = os.environ.get("FT_GEN5_SPEED", "")


class _NoPath:
    def update(self, x, y):
        return 0.0, None


class Gen5Terrain(G5._Gen5Layer, LN.LocalNav):
    name = "terrain_method"
    label = "+ XFeas gen5 (terrain, straight drive)"

    def setup(self, ctx):
        ds = LN.DepthSensor
        LN.DepthSensor = lambda *a, **k: None
        try:
            LN.LocalNav.setup(self, ctx)
            self.kind_cam = str(ctx["kind"])
            self._g5_setup(ctx)
        finally:
            LN.DepthSensor = ds
        self.label = os.environ.get("XFEAS_LABEL", self.label)

    def reset(self, ep):
        ep = dict(ep, speed_frac=float(SPEED)) if SPEED else ep
        LN.LocalNav.reset(self, ep)
        # == _Gen5Layer._g5_reset minus the stairs-scene path
        if getattr(self, "_n_exec", 0):
            self._verify_line("end of previous episode")
        self._g4_reset(ep)
        self.v_cap, self.vy_cmd = self.v_nom, 0.0
        self.cmd_now, self.ch_now = None, None
        self._t_sw, self._n_switch = None, 0
        self._n_exec, self._t_dec0 = 0, None
        self._top_pass = []
        self._pp5 = _NoPath()
        if self.diag:
            p = os.path.join(self.diag, f"decisions5_t{self.ep_trial}.jsonl")
            if os.path.exists(p):
                os.remove(p)

    def act(self, obs):
        self._g5_tick(obs)
        v0 = self.v
        self.v = self.v_cap
        try:
            base = straight_act(self, obs)
        finally:
            self.v = v0
        tw = np.asarray(base["twist"], float).copy()
        tw[1] = self.vy_cmd
        self.vrob.command(tw)
        return dict(twist=tw, info=(self.last or base["info"]), decisions=self.log)

    def flush(self):
        try:
            self._verify_line("end of run")
            self._diag_flush()
        except Exception:
            pass
