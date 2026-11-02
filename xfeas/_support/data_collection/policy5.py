"""
gen5 BEHAVIOUR policy of the switching collector (pure python). analysis/GEN5_DATA_SPEC.md 5.3.

Every decision tick (0.25 s):
  p = 0.5  KEEP (the command in force)
  p = 0.2  NOMINAL (the stock command)
  p = 0.3  a random candidate (KEEP / NOMINAL excluded) with weights:
             1 for every candidate;
             x3 for posture / lateral candidates (crouch c_h < 0, pitch c_p != 0, lateral c_vy != 0, or a mode whose measured
                top is >= 5 cm below the standing top) while the robot is within 2.5 m of a hazard (path arclength) or on it;
             x3 for upright forward-moving candidates (measured e_vx >= 0.6 x the nominal measured speed, no crouch / pitch /
                lateral) while the robot is FAR from every hazard (> 2.5 m before it): without this the smoke showed that
                uniform draws over a mostly-crouch grid stall slow controllers (AMO) before the hazard (1.4 m in 18 s);
             x4 (on top) for the AGGRESSIVE set when the situation is an aggressive family (climb / step-over / stairs /
                crawl / slot ...) and the robot is near the hazard: the 3 highest measured-speed candidates, the 3 lowest
                measured-top candidates, the 2 highest measured-|vy| candidates, every high-gear mode (e_vx > 1.2 x the
                nominal measured speed).
Episode STYLE (drawn once per episode, logged): 'spec' (exactly the above) or 'persist' (after a NOMINAL / random switch the
new command is HELD for a dwell of U{2..12} ticks = 0.5-3 s: the dwell ticks are KEEP with forced = 1, propensity 1). The
persist style exists because 0.5 / 0.2 / 0.3 holds a switched command for 0.5 s on average (a crouch held through a 1 m
overhang needs ~3 s): without it, the data would almost never contain a successful sustained crouch / sidestep / climb.
Logged per tick: chosen index, mode (keep / nominal / random / forced), propensity of the chosen index under the policy.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np
import descriptors as DS

KI = DS.KI


class BehaviourPolicy5:
    def __init__(
        self,
        cands,
        X0,
        rng,
        style="spec",
        aggr=False,
        p_keep=0.5,
        p_nom=0.2,
        commit=0.0,
        front_reach=0.25,
        rear_reach=0.35,
    ):
        self.C, self.rng, self.style, self.aggr = cands, rng, style, bool(aggr)
        self.commit = float(commit or 0.0)
        self.rear_reach = float(rear_reach)
        self.commit_log = []
        self.p_keep, self.p_nom = p_keep, p_nom
        self.dwell = 0
        N = len(cands)
        self.rand_idx = np.array([i for i, c in enumerate(cands) if c["kind"] not in ("KEEP", "NOMINAL")], int)
        X = X0
        nom = [i for i, c in enumerate(cands) if c["kind"] == "NOMINAL"][0]
        stand_top = float(X[nom, KI["e_top"]]) if X[nom, KI["e_top"]] > 0 else 0.0
        v_nom = max(float(X[nom, KI["e_vx"]]), 0.05)
        self.posture = np.zeros(N, bool)
        self.aggr_set = np.zeros(N, bool)
        self.forward = np.zeros(N, bool)
        for i in self.rand_idx:
            x = X[i]
            self.posture[i] = (
                (x[KI["c_h"]] < -1e-3)
                or abs(x[KI["c_p"]]) > 1e-3
                or abs(x[KI["c_vy"]]) > 1e-3
                or (x[KI["has_mode"]] > 0 and stand_top > 0 and 0 < x[KI["e_top"]] < stand_top - 0.05)
            )
        for i in self.rand_idx:
            x = X[i]
            self.forward[i] = (not self.posture[i]) and x[KI["e_vx"]] >= 0.6 * v_nom and x[KI["c_mode_static"]] < 0.5
        R = self.rand_idx
        if len(R):
            ev, et, ey = (
                X[R, KI["e_vx"]],
                np.where(X[R, KI["e_top"]] > 0, X[R, KI["e_top"]], 9.0),
                np.abs(X[R, KI["e_vy"]]),
            )
            for arr, k, sgn in ((ev, 3, -1), (et, 3, 1), (ey, 2, -1)):
                o = np.argsort(sgn * arr, kind="stable")[:k]
                self.aggr_set[R[o]] = True
            self.aggr_set[R[(X[R, KI["has_mode"]] > 0) & (ev > 1.2 * v_nom)]] = True
        self.nom = nom
        self.keep = [i for i, c in enumerate(cands) if c["kind"] == "KEEP"][0]
        self.lowmove = np.zeros(N, bool)
        self.e_vx = np.asarray(X[:, KI["e_vx"]], float)
        self.commit_d = None
        if self.commit > 0 and len(R):
            self.lat = np.abs(np.asarray(X[:, KI["c_vy"]], float)) > 1e-3
            mv = (X[R, KI["e_vx"]] >= 0.03) & (X[R, KI["e_fell"]] < 0.5) & (X[R, KI["e_top"]] > 0) & ~self.lat[R]
            if mv.any():
                t_lo = float(X[R[mv], KI["e_top"]].min())
                self.lowmove[R[mv & (X[R, KI["e_top"]] <= t_lo + 0.06)]] = True
            self.commit_d = float(front_reach) + float(self.rng.uniform(-0.2, 1.4))

    def weights(self, near):
        w = np.ones(len(self.rand_idx))
        if not near:
            w = w * np.where(self.forward[self.rand_idx], 3.0, 1.0)
        if near:
            w = w * np.where(self.posture[self.rand_idx], 3.0, 1.0)
            if self.aggr:
                w = w * np.where(self.aggr_set[self.rand_idx], 4.0, 1.0)
            if self.commit > 0:
                w = w * np.where(self.lowmove[self.rand_idx], 8.0, 1.0)
        return w / w.sum()

    def choose(self, near, d_front=None, d_end=None, t=None):
        if self.dwell > 0:
            self.dwell -= 1
            return self.keep, dict(mode="forced", prop=1.0)
        u = self.rng.random()
        if u < self.p_keep:
            return self.keep, dict(mode="keep", prop=self.p_keep)
        if u < self.p_keep + self.p_nom or len(self.rand_idx) == 0:
            i, info = self.nom, dict(mode="nominal", prop=self.p_nom)
        else:
            wn = near
            if self.commit > 0 and d_front is not None and d_front > self.commit_d + 0.5:
                wn = False  # commit mode: approach weighting until 0.5 m before commit_d
            w = self.weights(wn)
            j = int(self.rng.choice(len(self.rand_idx), p=w))
            i, info = int(self.rand_idx[j]), dict(mode="random", prop=(1 - self.p_keep - self.p_nom) * float(w[j]))
        if self.style == "persist":
            self.dwell = int(self.rng.integers(2, 13))
        if (
            self.commit > 0
            and near
            and info["mode"] == "random"
            and d_front is not None
            and d_end is not None
            and d_front <= self.commit_d
            and d_end > -self.rear_reach
            and self.posture[i]
            and not self.lat[i]
            and self.e_vx[i] >= 0.02
        ):
            if self.rng.random() < self.commit:
                f = float(self.rng.uniform(0.6, 1.6))
                need = (float(d_end) + self.rear_reach) / max(float(self.e_vx[i]), 0.03)
                self.dwell = int(min(math.ceil(f * need / 0.25), 180))
                self.commit_log.append(
                    dict(
                        t=None if t is None else round(float(t), 2),
                        i=int(i),
                        d_front=round(float(d_front), 3),
                        d_end=round(float(d_end), 3),
                        f=round(f, 3),
                        dwell=self.dwell,
                        low=bool(self.lowmove[i]),
                    )
                )
        return i, info
