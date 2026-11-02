"""
StairsGen2 (additive 2026-10-15): the XFeas-Gen2 layer (gen_methods2._Gen2Layer: 4 Hz decisions, learned selection,
macros) on top of the SAME scripted stairwell path follower as StairsStock / StairsOracle (stairs_methods.py). The
layer sets speed cap + body height / pitch / roll (and runs its go2 / detour / sidestep macros when it picks them);
steering is the path follower's except while a detour / sidestep macro drives. Env as gen_methods2 (XFEAS_GEN2_CKPT,
XFEAS_GEN2_SEL, XFEAS_LABEL).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import numpy as np
import gen_methods2 as GM2
from stairs_methods import StairsStock


class StairsGen2(GM2._Gen2Layer, StairsStock):
    name = "stairs_gen2"
    label = "+ XFeas-Gen2 layer (learned selection) on the stairwell path follower"

    def setup(self, ctx):
        StairsStock.setup(self, ctx)
        self._gen_setup(ctx)
        self.label = (
            os.environ.get("XFEAS_LABEL")
            or f"+ XFeas-Gen2 ({os.path.basename(os.path.dirname(self.gdec.ckpt))}) learned selection"
        )

    def reset(self, ep):
        StairsStock.reset(self, ep)
        self._gen_reset()

    def setpoints(self, obs, s, k):  # posture is commanded by the layer; speed = the layer's cap
        v = self.cap if (self.cap is not None and self.cap > 0) else self.v
        return None, v, 0.0

    def act(self, obs):
        self._gen_tick(obs)
        tw = self._macro_twist(obs)
        if tw is not None:
            return dict(twist=np.asarray(tw, float), info=self.last, decisions=self.log)
        r = StairsStock.act(self, obs)
        return dict(twist=r["twist"], info=self.last or r["info"], decisions=self.log)
