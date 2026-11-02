"""
Arms for the climb test scenes (straight drive to the goal, no planner, like door_methods / oh_methods):

  StraightStock (door_methods)      stock controller, default setpoints, nominal speed
  ClimbOracle                       ORACLE (not a method): one FIXED setpoint for the whole run, chosen from the headroom
                                    probe on TRAINING lanes (env CLIMB_ORACLE="h,pitch,v", default the probe's best cell)
  StraightXFeasHoldV                the unchanged XFeas hold selector (oh_methods.StraightXFeasHold) whose speed box may
                                    go up to CLIMB_VMAX (the controller's own speed range) while its STOCK reference stays
                                    the nominal speed: stock candidate = (nominal speed, default posture)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np
import local_nav as LN
from door_methods import straight_act

ORACLE = os.environ.get("CLIMB_ORACLE", "0.40,-0.35,0.9")


class ClimbOracle(LN.LocalNav):
    name = "climb_oracle"
    label = "ORACLE: best fixed setpoint (from the training-lane probe)"

    def reset(self, ep):
        super().reset(ep)
        self.sp_set = False
        self.h, self.p, self.vv = map(float, ORACLE.split(","))

    def act(self, obs):
        if not self.sp_set:
            obs["rob"].command(dict(body_height=self.h, pitch=self.p, roll=0.0))
            self.sp_set = True
        self.v = self.vv
        r = straight_act(self, obs)
        r["info"] = f"oracle: h {self.h:.2f} m  pitch {self.p:+.2f}  v {self.vv:.1f} m/s"
        return r


try:
    from oh_methods import StraightXFeasHold

    class StraightXFeasHoldV(StraightXFeasHold):
        name = "xfeashold_v"
        label = "+ XFeas layer (ours, hold): speed + posture setpoints (straight drive)"
        VMAX = float(os.environ.get("CLIMB_VMAX", "0.9"))

        def reset(self, ep):
            super().reset(ep)
            self.v_nom = self.v

        def act(self, obs):
            # speed box upper end = the controller's range (CLIMB_VMAX); until the first decision (t < 2 s) the robot
            # drives at the nominal speed (cap = v_nom), afterwards at the chosen speed.
            self.v = self.VMAX
            if self.cap is None:
                self.cap = self.v_nom
            return StraightXFeasHold.act(self, obs)
except ImportError:
    pass
