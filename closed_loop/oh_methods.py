"""
Selector variant for the overhang study (straight drive, like door_methods). Used ONLY if the validation scene V1 /
validation data show the unchanged layer (door_methods:StraightXFeas = local_nav.XFeasPosture) needs it.

StraightXFeasHold = XFeasPosture + COMMIT/HYSTERESIS (LAYER_DIAGNOSIS fix #4):
  * a non-stock posture that the model prefers by >= MARGIN is adopted at once (no low-pass toward a LOWER body:
    the 0.5 low-pass needs ~1 s to reach a crouch, i.e. 0.6 m at 0.6 m/s) and HELD for at least HOLD_S seconds;
  * while held, a new non-stock choice replaces it (and restarts the hold); a stock choice is ignored until the hold
    expires, then the posture returns to stock through the usual low-pass.
Reason: the overhang leaves the head camera's view at < ~1.5 m (camera pitched 26 deg down), so the model's evidence
for crouching exists only on the approach; the decision must persist while the robot passes under.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np
import local_nav as LN
from door_methods import straight_act


class StraightXFeasHold(LN.XFeasPosture):
    name = "xfeashold_straight"
    label = "+ XFeas layer (ours, commit/hold): speed + posture setpoints (straight drive)"
    HOLD_S = 2.0

    def reset(self, ep):
        super().reset(ep)
        self.hold_until = -1.0

    def act(self, obs):
        t, bs, rob, sp, dec = obs["t"], obs["bs"], obs["rob"], self.spec, self.dec
        dec.buf.tick(rob, bs)
        if dec.desc is None:
            dec.desc = rob.descriptor(self.ctx["stage"])
        if t >= self.t_next and dec.buf.full():
            self.t_next = t + self.PERIOD
            rgb, dep = dec._snapshot(bs)
            pb, pj = dec.buf.arrays()
            v_hi = max(min(self.v, sp.max_cmd[0]), sp.v_min + 0.01)
            box = dict(self.box, v=[max(sp.v_min, 0.1), v_hi], v_ref=max(sp.max_cmd[0], 0.3))
            ch = dec._ask(rgb, dep, pb, pj, dict(setpoint=box, seed=int(t * 10)))
            gain = float(ch["p_success"]) - float(ch["p_stock"])
            if gain >= self.MARGIN and (abs(ch["dh"]) > 1e-3 or abs(ch["pitch"]) > 1e-3 or abs(ch["roll"]) > 1e-3):
                self.hold_until = t + self.HOLD_S
                for k in ("dh", "pitch", "roll"):
                    tgt = float(ch[k])
                    if k == "dh" and tgt < self.sp_now[k]:
                        self.sp_now[k] = tgt                      # lowering: at once
                    else:
                        self.sp_now[k] += 0.5 * (tgt - self.sp_now[k])
                self.cap = float(ch["v"]); held = "new"
            elif t < self.hold_until:
                self.cap = min(float(ch["v_stock"]), self.cap or 9.0); held = "hold"
                ch = dict(ch, p_success=ch["p_stock"])
            else:
                ch = dict(ch, dh=0.0, pitch=0.0, roll=0.0, v=ch["v_stock"], p_success=ch["p_stock"])
                self.cap = float(ch["v"]); held = "stock"
                for k in ("dh", "pitch", "roll"):
                    self.sp_now[k] += 0.5 * (0.0 - self.sp_now[k])
            if self.posture:
                rob.command(dict(body_height=self.nom + self.sp_now["dh"], pitch=self.sp_now["pitch"], roll=self.sp_now["roll"]))
            self.log.append(dict(t=round(t, 2), x=round(float(bs["pos"][0]), 2), y=round(float(bs["pos"][1]), 2),
                                 v=self.cap, dh=round(self.sp_now["dh"], 3), pitch=round(self.sp_now["pitch"], 3),
                                 roll=round(self.sp_now["roll"], 3), p=ch.get("p_success"), p_stock=ch.get("p_stock"),
                                 gain=round(gain, 3), state=held))
            self.last = (f"XFeas[{held}]: v {self.cap:.2f}  dh {self.sp_now['dh']:+.2f}  pitch {self.sp_now['pitch']:+.2f}  "
                         f"P={ch.get('p_success'):.2f} (stock {ch.get('p_stock'):.2f})")
        base = straight_act(self, obs)
        tw = np.asarray(base["twist"], float).copy()
        if self.cap is not None and tw[0] > self.cap > 0:
            r = self.cap / tw[0]; tw[0] = self.cap; tw[3] *= r
        return dict(twist=tw, info=self.last or base["info"], decisions=self.log)


class StraightCrouchThenStand(LN.LocalNav):
    """PRIVILEGED ORACLE arm (not a method) for overhang-then-obstacle scenes: lowest body height until the robot is
    closer than SCENES[scene]['stand_goal_dist'] to the goal (i.e. just past the overhang), then the stock posture."""
    name = "crouchstand_straight"
    label = "ORACLE: lowest height, stand up after the overhang (privileged)"

    def reset(self, ep):
        super().reset(ep)
        import sys
        sc = sys.modules["__main__"].SCENES[ep["scene"]]
        self.stand_d = float(sc.get("stand_goal_dist", -1.0)); self.state = None

    def act(self, obs):
        want = "low" if obs["goal_dist"] > self.stand_d else "stock"
        if want != self.state:
            R = self.spec.extra["setpoint_ranges"]["body_height"]
            import sys
            nom = getattr(sys.modules.get(type(obs["rob"]).__module__), "NOMINAL_H", 0.52)
            obs["rob"].command(dict(body_height=R[0] if want == "low" else nom, pitch=0.0, roll=0.0)); self.state = want
        r = straight_act(self, obs); r["info"] = f"oracle: {self.state} posture"
        return r
