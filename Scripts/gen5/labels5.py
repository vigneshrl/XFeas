#!/usr/bin/env python
"""
gen5 LABELS (pure python / numpy), computed offline from the logged future of every episode. analysis/GEN5_DATA_SPEC.md 6.

labels(z) -> dict of arrays over the episode's DECISION ticks (z = an episode npz as a dict); per tick k at time t_k
(the candidate actually chosen at k is chosen[k]; horizon H = 1.5 s):
  y_fall       1 if the episode ended by a (mode-aware) fall at t_end in [t_k, t_k + H], else 0
  y_stall      1 if the path progress over [t_k, min(t_k + H, t_end)] is < 0.1 m; 0 if the goal was reached inside the
               window; NaN if a fall ends the episode inside the window (the fall label covers it)
  y_contact    1 if any non-foot robot link has hazard contact (impulse >= 1e-3 N s per pair and step, 0.02 s bins) in
               [t_k, t_k + H]  (+ y_contact_Ns: the impulse sum, y_contact_foot: the same for feet)
  y_progress   path arclength progress over [t_k, min(t_k + H, t_end)] (m); after a success the remaining window adds 0
  outcome      episode outcome code (OUTCOMES: success 0, fall 1, stall 2, timeout 3, crossed_low 4) for every tick
  y_success    1 if the episode ended in success
  next_dec     index (into the episode's tick arrays) of the decision tick at t_k + H (the first tick with
               t >= t_k + H - 1e-6), -1 if the episode ends before (terminal inside the horizon)
  term_in_h    1 if the episode ends inside (t_k, t_k + H]; then the bootstrapped value target is the outcome
  t_to_end     t_end - t_k
Options (2026-10-26):
  curb contact   env_curb hazards are the scene's own curb, so the simulator logs that contact as scene_body, not haz_body;
                 scene_body bins while the base is within CURB_MARGIN m of an env_curb hazard count as hazard contact
  tilt_fall      if set (rad): a G1 episode whose tilt exceeds it is relabelled 'fell' at the first such tick (the live
                 fall rule exempts tilt during low G1 postures, so a G1 that tips over can end as 'success'); off by default
Progress uses the 10 Hz history records (base x, y projected on the path polyline), so it is finer than the tick grid.
    python3 scripts/gen5/labels5.py <episode.npz> ...        # prints a label summary per episode
"""
import json, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import steer5 as ST

H = 1.5
OUTCOMES = dict(success=0, fell=1, stall=2, timeout=3, crossed_low=4)
CONTACT_MIN = 1e-3
CURB_MARGIN = 1.0


def arc_series(z):
    """(times relative to the episode clock, path arclength) from the 10 Hz records + the ticks"""
    pf = ST.PathFollower(np.asarray(z["path_world"]))
    t0 = float(z["t_clock0"])
    rt = np.asarray(z["rec_t"], np.float64) - t0
    ach = np.asarray(z["rec_ach"], np.float64)
    m = rt >= -1e-6
    ts = list(rt[m]); arcs = [pf.project(a[9:11])[0] for a in ach[m]]
    ts += list(np.asarray(z["t"], np.float64)); arcs += list(np.asarray(z["arc"], np.float64))
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(arcs)[o]


def curb_bins(z, t, bs, n):
    """bool (n,) over contact bins: base within CURB_MARGIN of an env_curb hazard (all False if there is none)"""
    m = np.zeros(n, bool)
    hz = [h for h in json.loads(str(z["hazards_json"])) if h["kind"] == "env_curb"] if "hazards_json" in z else []
    if not hz or "bins_scene_body" not in z:
        return m
    ab = np.interp((np.arange(n) + 0.5) * bs, t, np.asarray(z["arc"], np.float64))
    for h in hz:
        m |= (ab >= h["arc0"] - CURB_MARGIN) & (ab <= h["arc1"] + CURB_MARGIN)
    return m


def tilt_relabel(z, reason, t_end, tilt_fall):
    """-> (reason, t_end) after the optional G1 tilt rule"""
    if tilt_fall is None or reason == "fell" or str(z.get("robot", "")) != "g1" or "tilt" not in z:
        return reason, t_end
    t = np.asarray(z["t"], np.float64); tl = np.asarray(z["tilt"], np.float64)
    i = np.where((tl > tilt_fall) & (t <= t_end + 1e-9))[0]
    return ("fell", float(t[i[0]])) if len(i) else (reason, t_end)


def labels(z, tilt_fall=None):
    t = np.asarray(z["t"], np.float64)
    dec = np.asarray(z["dec"], bool)
    reason, t_end = tilt_relabel(z, str(z["end_reason"]), float(z["t_end"]), tilt_fall)
    K = np.where(dec & (t < t_end + 1e-9))[0] if t_end < float(z["t_end"]) else np.where(dec)[0]
    ts, arcs = arc_series(z)
    arc_at = lambda x: float(np.interp(x, ts, arcs))
    amax_to = lambda x: float(np.max(arcs[ts <= x + 1e-9])) if np.any(ts <= x + 1e-9) else float(arcs[0])
    bs = float(z["bin_s"]); hb = np.asarray(z["bins_haz_body"], np.float64); hf = np.asarray(z["bins_haz_foot"], np.float64)
    cb = curb_bins(z, t, bs, len(hb))
    if cb.any():
        hb = hb + np.where(cb, np.asarray(z["bins_scene_body"], np.float64)[:len(hb)], 0.0)
    out = {k: np.zeros(len(K), np.float32) for k in ("y_fall", "y_stall", "y_contact", "y_contact_Ns", "y_contact_foot", "y_progress",
                                                     "term_in_h", "t_to_end")}
    out["next_dec"] = np.full(len(K), -1, np.int32)
    for j, k in enumerate(K):
        tk = t[k]; t1 = tk + H
        ends_in = t_end <= t1 + 1e-9
        out["term_in_h"][j] = float(ends_in)
        out["t_to_end"][j] = t_end - tk
        out["y_fall"][j] = float(reason == "fell" and ends_in)
        te = min(t1, t_end)
        prog = arc_at(te) - arc_at(tk)
        out["y_progress"][j] = prog
        if reason == "fell" and ends_in:
            out["y_stall"][j] = np.nan
        elif reason == "success" and ends_in:
            out["y_stall"][j] = 0.0
        else:
            out["y_stall"][j] = float(prog < 0.1)
        b0, b1 = int(tk / bs), int(min(t1, t_end) / bs) + 1
        out["y_contact_Ns"][j] = float(hb[b0:b1].sum()); out["y_contact"][j] = float(np.any(hb[b0:b1] >= CONTACT_MIN))
        out["y_contact_foot"][j] = float(np.any(hf[b0:b1] >= CONTACT_MIN))
        nx = np.where((t >= t1 - 1e-6) & dec)[0]
        if not ends_in and len(nx):
            out["next_dec"][j] = int(nx[0])
    out["tick_index"] = K.astype(np.int32)
    out["outcome"] = np.full(len(K), OUTCOMES.get(reason, 3), np.int8)
    out["end_reason"] = reason; out["t_end"] = t_end
    out["y_success"] = np.full(len(K), float(reason == "success"), np.float32)
    out["chosen"] = np.asarray(z["chosen"])[K].astype(np.int32)
    return out


def summary(z, L):
    n = len(L["tick_index"])
    f = lambda k: float(np.nanmean(L[k])) if n else float("nan")
    return dict(sid=str(z["sid"]), end=L["end_reason"], n_dec=n, fall=f("y_fall"), stall=f("y_stall"), contact=f("y_contact"),
                progress=f("y_progress"), term=f("term_in_h"), next_ok=float(np.mean(L["next_dec"] >= 0)) if n else float("nan"))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        z = np.load(p, allow_pickle=False)
        z = {k: z[k] for k in z.files}
        L = labels(z, tilt_fall=float(os.environ["TILT_FALL"]) if os.environ.get("TILT_FALL") else None)
        print(json.dumps(summary(z, L)))
