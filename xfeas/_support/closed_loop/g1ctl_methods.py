"""
Hand-set ORACLE arms for the G1 controller-bridge characterisation (analysis/G1_CONTROLLER_BRIDGES.md). Additive
2026-10-17. Not methods: privileged, fixed commands chosen from FLAT-GROUND characterisation only (never tuned on a
test scene).
  posture_oracle : straight drive to the goal at ORACLE_V m/s with a fixed posture setpoint
                   (ORACLE_H body height, ORACLE_PITCH torso pitch, ORACLE_MODE for mode-switch controllers)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os
import numpy as np
import local_nav as LN
from door_methods import straight_act


class PostureOracle(LN.LocalNav):
    name = "posture_oracle"
    label = "ORACLE: fixed crouch / lean setpoint (straight drive)"

    def reset(self, ep):
        super().reset(ep)
        self.sp_set = False
        rob = self.ctx["rob"]
        if hasattr(rob, "reset_setpoints"):
            rob.reset_setpoints()

    def act(self, obs):
        if not self.sp_set:
            sp = {}
            if os.environ.get("ORACLE_H"):
                sp["body_height"] = float(os.environ["ORACLE_H"])
            if os.environ.get("ORACLE_PITCH"):
                sp["pitch"] = float(os.environ["ORACLE_PITCH"])
            if os.environ.get("ORACLE_MODE"):
                sp["mode"] = os.environ["ORACLE_MODE"]
            obs["rob"].command(sp)
            self.sp_set = True
            self.info = "oracle: " + ", ".join(f"{k} {v}" for k, v in sp.items())
        if os.environ.get("ORACLE_V"):
            self.v = float(os.environ["ORACLE_V"])
        r = straight_act(self, obs)
        r["info"] = self.info
        return r
