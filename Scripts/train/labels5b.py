#!/usr/bin/env python
"""
gen5b BRANCH LABELS (pure numpy). analysis/GEN5B_DATA_PLAN.md 2.

branch_labels(run, t_b, ...) -> dict, from the branch moment t_b to the end of the run that holds the branch command:
  outcome over the rest of the run:
    success (path goal reached with |e| < 0.5 m, the strict rule of the collector; crossed_low is NOT a success),
    fell (mode-aware fall rule of the run), fell_tilt (G1 only: tilt > TILT_FALL at any tick after t_b, the 10-26 relabel
    field; success_tilt = success and not fell_tilt), stall, timeout, contact (any non-foot hazard-body bin >= 1e-3 N s after
    t_b; env_curb: scene-body bins with the base within 1 m of the curb count, labels5 rule), contact_Ns, t_goal (s, success
    only), progress (m of path arclength from the branch to the end), max_tilt, min_top (lowest body top above the floor)
  short window H = 1.5 s (labels5): w_fall, w_stall (NaN if a fall ends it, 0 if the goal is reached inside), w_contact,
    w_progress
run arrays: ticks (n, 12) [t, arc, e, x, y, z, yaw, tilt, top, vx, vy, d_haz], bins hb / hf / sb with origin run t0,
end_reason, t_end.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import numpy as np

H = 1.5
CONTACT_MIN = 1e-3
CURB_MARGIN = 1.0
TILT_FALL = 1.2
BIN_S = 0.02
OUTCOMES = dict(success=0, fell=1, stall=2, timeout=3, crossed_low=4)
LAB_KEYS = ("success", "fell", "fell_tilt", "success_tilt", "stall", "timeout", "contact", "contact_Ns", "t_goal", "progress", "max_tilt",
            "min_top", "w_fall", "w_stall", "w_contact", "w_progress", "outcome", "t_rest")


def branch_labels(ticks, hb, hf, sb, t0, end_reason, t_end, t_b, arc_b, robot, hazards=(), bin_s=BIN_S):
    ticks = np.asarray(ticks, np.float64).reshape(-1, 12)
    tt, arc, tilt, top = ticks[:, 0], ticks[:, 1], ticks[:, 7], ticks[:, 8]
    m = tt >= t_b - 1e-6
    ts = np.r_[t_b, tt[m]]; ar = np.r_[arc_b, arc[m]]
    arc_at = lambda x: float(np.interp(x, ts, ar))
    hb = np.asarray(hb, np.float64).copy()
    curbs = [h for h in hazards if h.get("kind") == "env_curb"]
    if curbs and len(hb) and len(sb):
        tb = t0 + (np.arange(len(hb)) + 0.5) * bin_s
        ab = np.interp(tb, ts, ar)
        cm = np.zeros(len(hb), bool)
        for h in curbs:
            cm |= (ab >= h["arc0"] - CURB_MARGIN) & (ab <= h["arc1"] + CURB_MARGIN)
        hb = hb + np.where(cm, np.asarray(sb, np.float64)[:len(hb)], 0.0)
    b0 = max(0, int((t_b - t0) / bin_s))
    b_end = int((t_end - t0) / bin_s) + 1
    bw = int((min(t_b + H, t_end) - t0) / bin_s) + 1
    after = hb[b0:b_end]
    fell = end_reason == "fell"
    succ = end_reason == "success"
    ft = bool(robot == "g1" and np.any(tilt[m] > TILT_FALL))
    ends_in = t_end <= t_b + H + 1e-9
    te = min(t_b + H, t_end)
    wp = arc_at(te) - arc_b
    L = dict(success=float(succ), fell=float(fell), fell_tilt=float(fell or ft), success_tilt=float(succ and not ft),
             stall=float(end_reason == "stall"), timeout=float(end_reason == "timeout"),
             contact=float(np.any(after >= CONTACT_MIN)), contact_Ns=float(after.sum()),
             t_goal=float(t_end - t_b) if succ else float("nan"), progress=float(arc_at(t_end) - arc_b),
             max_tilt=float(np.max(tilt[m])) if m.any() else float("nan"), min_top=float(np.nanmin(top[m])) if m.any() and np.isfinite(top[m]).any() else float("nan"),
             w_fall=float(fell and ends_in), w_contact=float(np.any(hb[b0:bw] >= CONTACT_MIN)), w_progress=float(wp),
             outcome=float(OUTCOMES.get(end_reason, 3)), t_rest=float(t_end - t_b))
    L["w_stall"] = float("nan") if (fell and ends_in) else (0.0 if (succ and ends_in) else float(wp < 0.1))
    return L
