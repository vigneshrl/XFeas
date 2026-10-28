"""
Ours on the FULL map-free navigation stack (episode_runner method 'xfeas_fullstack'):
    camera planner (local_nav.LocalNav: head RGB-D -> obstacle points -> local A* + DWA + recovery)
  + XFeas layer (v8 checkpoint via XFEAS_CKPT; commit/hold selector of scripts/overhang/oh_methods.StraightXFeasHold)
    sending speed + body height / pitch / roll setpoints to the UNCHANGED ReLIC controller.

The ONLY planner change (posture-aware obstacle band, a general rule from the robot's own geometry):
  LocalNav's DepthSensor keeps depth points whose height above the floor lies in a fixed band
  [lowest LASER_H, highest LASER_H + 0.15] = [0.15, 0.75] m for Spot. That band does not describe the robot's body:
  standing Spot is 1.01 m tall (so stock drives into things between 0.75 and 1.01 m, e.g. the D1 door at 0.92 m), and
  a crouched Spot is lower. Here the band's UPPER limit follows the robot's CURRENT COMMANDED TOP HEIGHT:
        upper = top(body_height_cmd) + 0.02 m
        top(h) = linear through Spot's measured tops: 1.01 m at the nominal body height 0.52, 0.86 m at the lowest 0.25
  (extrapolated to 1.08 m at 0.65). Points above the robot's commanded top cannot be hit and are not obstacles;
  everything between the floor band and the top is. Points stay in the planner's short-term memory with their
  height, and the band is applied when the memory is read, so a point seen while standing stops being an obstacle
  once the body is commanded low enough (and vice versa). Nothing about D1 is used: no thresholds were fit to it.
  Pitch / roll are not included in top() (documented limitation).
  The STOCK arm ('stock' = local_nav.LocalNav) keeps its default band.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np
import local_nav as LN
import oh_methods as OH

TOP_NOM, TOP_LOW, H_NOM, H_LOW = 1.01, 0.86, 0.52, 0.25
TOP_MARGIN = 0.02


def top_of(body_height):
    return TOP_LOW + (TOP_NOM - TOP_LOW) * (float(body_height) - H_LOW) / (H_NOM - H_LOW)


class XFeasFullStack(OH.StraightXFeasHold):
    name = "xfeas_fullstack"
    label = "+ XFeas layer (ours): posture setpoints + posture-aware camera planner (full stack)"

    def setup(self, ctx):
        super().setup(ctx)
        self.band_stock = tuple(self.laser.band)
        hi = self.ctx["spec"].extra.get("setpoint_ranges", {}).get("body_height", [H_LOW, H_NOM])[1]
        self.laser.band = (self.band_stock[0], max(self.band_stock[1], top_of(hi) + TOP_MARGIN))   # scan the full range
        print(f"[fullstack] stock band {self.band_stock}; scan band {self.laser.band}; upper limit follows top(body_height_cmd)+{TOP_MARGIN}", flush=True)

    def reset(self, ep):
        super().reset(ep)
        self.upper = top_of(self.nom if self.posture else H_NOM) + TOP_MARGIN
        self.band_log = []

    def _points(self, t, x, y):
        P = LN.LocalNav._points(self, t, x, y)
        if len(P):
            keep = (P[:, 2] <= self.upper) | (P[:, 2] == 0.0)          # h == 0: points marked infeasible (avoid_mem)
            P = P[keep]
        return P

    def act(self, obs):
        t, bs, rob, sp, dec = obs["t"], obs["bs"], obs["rob"], self.spec, self.dec
        dec.buf.tick(rob, bs)
        if dec.desc is None:
            dec.desc = rob.descriptor(self.ctx["stage"])
        if t >= self.t_next and dec.buf.full():                     # --- XFeas, identical to StraightXFeasHold.act
            self.t_next = t + self.PERIOD
            rgb, dep = dec._snapshot(bs)
            pb, pj = dec.buf.arrays()
            v_hi = max(min(self.v, sp.max_cmd[0]), sp.v_min + 0.01)
            box = dict(self.box, v=[max(sp.v_min, 0.1), v_hi], v_ref=max(sp.max_cmd[0], 0.3))
            ch = dec._ask(rgb, dep, pb, pj, dict(setpoint=box, seed=int(t * 10)))
            gain = float(ch["p_success"]) - float(ch["p_stock"])
            p_best, p_stock = float(ch["p_success"]), float(ch["p_stock"])
            if gain >= self.MARGIN and (abs(ch["dh"]) > 1e-3 or abs(ch["pitch"]) > 1e-3 or abs(ch["roll"]) > 1e-3):
                self.hold_until = t + self.HOLD_S
                for k in ("dh", "pitch", "roll"):
                    tgt = float(ch[k])
                    if k == "dh" and tgt < self.sp_now[k]:
                        self.sp_now[k] = tgt
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
            bh = (self.nom + self.sp_now["dh"]) if self.posture else H_NOM
            if self.posture:
                rob.command(dict(body_height=bh, pitch=self.sp_now["pitch"], roll=self.sp_now["roll"]))
            self.upper = top_of(bh) + TOP_MARGIN                      # --- the planner change: band follows the posture
            self.log.append(dict(t=round(t, 2), x=round(float(bs["pos"][0]), 2), y=round(float(bs["pos"][1]), 2),
                                 v=self.cap, dh=round(self.sp_now["dh"], 3), pitch=round(self.sp_now["pitch"], 3),
                                 roll=round(self.sp_now["roll"], 3), p=ch.get("p_success"), p_stock=ch.get("p_stock"),
                                 p_best=round(p_best, 3), gain=round(gain, 3), state=held, body_height=round(bh, 3),
                                 band_upper=round(self.upper, 3), mode=self.mode))
            self.last = (f"XFeas[{held}]: v {self.cap:.2f}  body h {bh:.2f}  pitch {self.sp_now['pitch']:+.2f}  "
                         f"P={p_best:.2f} (stock {p_stock:.2f})  band<= {self.upper:.2f} m")
        base = LN.LocalNav.act(self, obs)                             # --- the camera planner (unchanged code)
        if base.get("giveup"):
            return dict(base, decisions=self.log)
        tw = np.asarray(base["twist"], float).copy()
        if self.cap is not None and tw[0] > self.cap > 0:
            r = self.cap / tw[0]; tw[0] = self.cap; tw[3] *= r
        info = self.last or base["info"]
        if self.mode != "drive":
            info = f"{info}  [planner: {self.mode}]"
        return dict(twist=tw, info=info, decisions=self.log)


class XFeasFullStackFixedBand(XFeasFullStack):
    """ABLATION (diagnostics only): identical to XFeasFullStack but the planner keeps the STOCK obstacle band
    [0.15, 0.75] m, i.e. no posture-aware band. Separates the band rule from the posture choice."""
    name = "xfeas_fullstack_fixedband"
    label = "ablation: XFeas layer + camera planner with the stock (fixed) obstacle band"

    def setup(self, ctx):
        super().setup(ctx)
        self.laser.band = tuple(self.band_stock)
        print(f"[fullstack-ablation] fixed stock band {self.laser.band}", flush=True)

    def _points(self, t, x, y):
        return LN.LocalNav._points(self, t, x, y)
