#!/usr/bin/env python
"""Decides which commands are tried from which moments: switching at several distances from the obstacle, holding
vs standing up once crouched, and scenes with nothing to avoid. Deterministic, no random switching.

Details:
gen5b BRANCH SCHEDULE (pure python, deterministic)..

No random switching: every run starts from a saved moment (or the spawn) with ONE command that is then held to the end of
the run (goal / fall / stall / t_max). Steering always stays with the stock path follower. What is tried is chosen from the
controller's OWN candidate menu (candidates) by the MEASURED effect (descriptors e_top / e_vx / e_vy / e_fell), never by name.

command classes (pick_classes):
  stock   the NOMINAL candidate
  slow    upright (no crouch / pitch / lateral, measured top within 6 cm of stock), e_vx closest to 0.5 x stock e_vx
  fast    upright, the highest e_vx >= 1.15 x stock e_vx
  low1    the lowest measured top among MOVING (e_vx >= 0.03, not fallen in the probe, no lateral) candidates with
          top >= 5 cm below stock; ties (1 cm) -> e_vx closest to stock e_vx
  low2    the next lowest moving top in ANOTHER transition class, >= 3 cm above low1
  latL / latR  the largest measured +/- e_vy (upright height), preferring e_vx >= 0.3 x stock
  stop    only when the menu has no slow / fast / posture (PHP-like menus)
  slowP   (per posture P) same class as P, e_vx closest to 0.5 x e_vx(P), moving
Schedules (plan, with the 10-27 time cut: A distances {2.0, 1.5, 1.0, 0.5}, B points {-0.5, mid, end, +0.5, +1.5}):
  A  stock prefix; moments at d before the hazard start; options = every class (stock = the prefix run itself)
  B  (low-obstacle families) prefix = stock, then P in {low1, low2} from d = 1.5; moments relative to the hazard;
     options keep P (= the B prefix run) / stock (stand up) / the other posture / slowP
  C  (null) 2 moments; options stock (prefix run) / slow / low1 / fast
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import numpy as np
import descriptors as DS

KI = DS.KI
A_D = (2.0, 1.5, 1.0, 0.5)
B_PTS = (
    ("pre0.5", "front", -0.5),
    ("mid", "mid", 0.0),
    ("end", "end", 0.0),
    ("post0.5", "end", 0.5),
    ("post1.5", "end", 1.5),
)
B_FROM_D = 1.5
LOW_FAMS = ("crawl", "overhang", "env_overhang")
SIDE_FAMS = ("slot", "env_lowclutter", "lowclutter", "clutter", "env_gap")


def pick_classes(C, X, lateral=True):
    """-> {class: candidate index}"""
    N = len(C)
    xn = X[1]
    top_s = float(xn[KI["e_top"]]) if xn[KI["e_top"]] > 0 else 9.0
    v_n = max(float(xn[KI["e_vx"]]), 0.05)
    ok = np.array(
        [
            C[i]["kind"] not in ("KEEP", "NOMINAL") and X[i, KI["e_fell"]] < 0.5 and X[i, KI["e_missing"]] < 0.5
            for i in range(N)
        ]
    )
    top = np.where(X[:, KI["e_top"]] > 0, X[:, KI["e_top"]], 9.0)
    ev = X[:, KI["e_vx"]]
    ey = X[:, KI["e_vy"]]
    lat = np.abs(X[:, KI["c_vy"]]) > 1e-3
    static = X[:, KI["c_mode_static"]] > 0.5
    upright = (X[:, KI["c_h"]] > -1e-3) & (np.abs(X[:, KI["c_p"]]) < 1e-3) & ~lat & (top >= top_s - 0.06) & ~static
    out = {"stock": 1}
    m = ok & upright & (ev >= 0.03) & (ev <= 0.8 * v_n)
    if m.any():
        i = np.where(m)[0]
        out["slow"] = int(i[np.argmin(np.abs(ev[i] - 0.5 * v_n))])
    m = ok & upright & (ev >= 1.15 * v_n)
    if m.any():
        i = np.where(m)[0]
        out["fast"] = int(i[np.argmax(ev[i])])
    mv = ok & (ev >= 0.03) & ~lat & ~static & (top <= top_s - 0.05)
    if mv.any():
        i = np.where(mv)[0]
        o = sorted(i, key=lambda k: (round(float(top[k]), 2), abs(float(ev[k]) - v_n)))
        out["low1"] = int(o[0])
        c1, t1 = C[o[0]]["cls"], float(top[o[0]])
        o2 = [k for k in o if C[k]["cls"] != c1 and float(top[k]) >= t1 + 0.03]
        if o2:
            out["low2"] = int(o2[0])
    if lateral:
        for nm, sg in (("latL", 1.0), ("latR", -1.0)):
            m = ok & lat & (sg * ey >= 0.05) & (X[:, KI["c_h"]] > -1e-3)
            if m.any():
                i = np.where(m)[0]
                sc = sg * ey[i] - np.where(ev[i] >= 0.3 * v_n, 0.0, 10.0)
                out[nm] = int(i[np.argmax(sc)])
    if not any(k in out for k in ("slow", "fast", "low1")):
        m = np.array([C[i]["kind"] not in ("KEEP", "NOMINAL") for i in range(N)]) & (np.abs(ev) < 0.03)
        if m.any():
            out["stop"] = int(np.where(m)[0][0])
    return out


def slow_in(C, X, p):
    ev = X[:, KI["e_vx"]]
    m = (
        np.array([C[i]["cls"] == C[p]["cls"] and i != p and X[i, KI["e_fell"]] < 0.5 for i in range(len(C))])
        & (ev >= 0.02)
        & (ev <= 0.8 * ev[p])
    )
    if not m.any():
        return None
    i = np.where(m)[0]
    return int(i[np.argmin(np.abs(ev[i] - 0.5 * ev[p]))])


def plan(family5, C, X, robot, hazards, arc_goal, front_reach=0.0):
    """-> dict(classes, A=[(label, arc)], B={P: [(label, arc)]}, opts); arcs are BASE path arclengths. d (A) and pre0.5 (B) are
    measured from the body FRONT (base + front_reach: G1 0.25 m, Spot 0.90 m incl. the stowed arm, gensim5.COMMIT_REACH);
    mid / end / post points are base positions relative to the hazard (the body is under it / its rear clears it)."""
    lateral = family5 in SIDE_FAMS
    cl = pick_classes(C, X, lateral=lateral)
    out = dict(classes=cl, A=[], B={}, opts_A=[], opts_B={}, null=family5 == "null")
    if family5 == "null" or not hazards:
        g = float(arc_goal)
        out["A"] = [("null_a", g - 3.0), ("null_b", g - 1.5)]
        out["opts_A"] = [k for k in ("slow", "low1", "fast") if k in cl]
        return out
    h = min(hazards, key=lambda z: z["arc0"])
    a0, a1 = float(h["arc0"]), float(h["arc1"])
    out["A"] = [(f"d{d:.1f}", a0 - front_reach - d) for d in A_D]
    out["opts_A"] = [k for k in cl if k != "stock"]
    if family5 in LOW_FAMS and robot in ("g1", "spot"):
        for P in ("low1", "low2"):
            if P not in cl:
                continue
            other = "low2" if P == "low1" else "low1"
            pts = []
            for lab, ref, off in B_PTS:
                r = a0 - front_reach if ref == "front" else a1 if ref == "end" else 0.5 * (a0 + a1)
                pts.append((lab, r + off))
            out["B"][P] = [
                p for p in pts if p[1] > a0 - front_reach - B_FROM_D + 0.05 and p[1] < float(arc_goal) - 0.05
            ]
            o = {"stock": cl["stock"]}
            if other in cl:
                o["other"] = cl[other]
            sp = slow_in(C, X, cl[P])
            if sp is not None:
                o["slowP"] = sp
            out["opts_B"][P] = o
    return out
