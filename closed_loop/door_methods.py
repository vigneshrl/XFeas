"""Crouch test (scene D1): drive STRAIGHT to the goal with no planner (the camera planner's obstacle band would refuse
to go under the door), so the only difference between the two arms is the posture/speed setpoint.

  StraightStock  -- stock controller, default posture, nominal speed.
  StraightXFeas  -- the unchanged XFeasPosture layer (v7 model, same selector, MARGIN, low-pass, 4 Hz) choosing
                    speed + body height / pitch / roll; only its path source is the straight heading controller.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
import numpy as np
import local_nav as LN


AMO_TURN = os.environ.get("XFEAS_STRAIGHT_AMO_TURN", "0") not in ("", "0")


def straight_act(self, obs):
    bs = obs["bs"]
    x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
    gx, gy = obs["goal_xy"]
    err = (math.atan2(gy - y, gx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
    wz = float(np.clip(2.0 * err, -self.wz_cap, self.wz_cap))
    v = self.v if abs(err) < 0.5 else 0.0
    if v == 0.0 and AMO_TURN and str(getattr(self, "kind", "")).startswith("g1"):
        # (additive 2026-10-16, GEN2_G1_DEBUG; XFEAS_STRAIGHT_AMO_TURN=1, default off = as before) AMO's released
        # deployment is in 'stand' mode for |vx| < 0.1 (g1_amo.py: dyaw forced 0, gait clock frozen), so vx = 0 with a
        # turn command never turns: a heading error > 0.5 rad would freeze the robot. Turn at the spec's v_min instead.
        v = float(self.spec.v_min)
    return dict(twist=np.array([v, 0.0, 0.0, wz]), info="straight to goal (no planner)")


class StraightStock(LN.LocalNav):
    name = "stock_straight"
    label = "STOCK controller, default posture (straight drive)"
    act = straight_act


class StraightXFeas(LN.XFeasPosture):
    name = "xfeaspos_straight"
    label = "+ XFeas layer (ours): speed + posture setpoints (straight drive)"

    def act(self, obs):
        orig = LN.LocalNav.act                     # XFeasPosture.act calls LocalNav.act for the path: swap it for
        LN.LocalNav.act = straight_act             # the straight controller during this call only
        try:
            return LN.XFeasPosture.act(self, obs)
        finally:
            LN.LocalNav.act = orig


class StraightCrouch(LN.LocalNav):
    """ORACLE arm (not a method): stock controller held at its LOWEST commanded body height for the whole run --
    tells whether a crouch can pass the door at all with this controller."""
    name = "crouch_straight"
    label = "ORACLE: fixed lowest body height (straight drive)"

    def reset(self, ep):
        super().reset(ep)
        self.sp_set = False

    def act(self, obs):
        if not self.sp_set:
            lo = self.spec.extra["setpoint_ranges"]["body_height"][0]
            obs["rob"].command(dict(body_height=lo, pitch=0.0, roll=0.0)); self.sp_set = True
            self.info = f"oracle: body_height {lo:.2f} m (lowest)"
        r = straight_act(self, obs); r["info"] = self.info
        return r
