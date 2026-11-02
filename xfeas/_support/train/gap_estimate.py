#!/usr/bin/env python
"""
Perceived gap geometry for the closed-loop SIDESTEP macro of gen2 (no privileged scene info). gen2 agent, 2026-10-14.
From ONE decision depth frame (gravity-aligned head camera, cmap.points): obstacle points 0.08-1.0 m above the floor;
wall front x0 = 5th percentile of their forward distance (within +-1.2 m, < 3.5 m); in the 0.5 m slab behind the
front, a 2 cm lateral occupancy histogram; the free interval (>= 0.12 m) closest to the robot's centre line is the gap.
Returns dict(x0, x1, e, w, ok) in the ROOT frame at the decision (x forward, y left), or ok=False.
Validated on gen_v1 env_gap decision frames against the generator's own (w, e): see validate().
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import cmap as CM


def estimate(depth, kind, h_lo=0.08, h_hi=1.0, bin_=0.02, min_w=0.12):
    P = CM.points(depth, kind, stride=1)
    zf, _ = CM.floor_level(P, kind)
    h = P[:, 2] - zf
    O = P[(h > h_lo) & (h < h_hi) & (np.abs(P[:, 1]) < 1.2) & (P[:, 0] < 3.5) & (P[:, 0] > 0.3)]
    if len(O) < 30:
        return dict(ok=False, why="no obstacle points")
    x0 = float(np.percentile(O[:, 0], 5))
    S = O[(O[:, 0] >= x0 - 0.05) & (O[:, 0] <= x0 + 0.5)]
    edges = np.arange(-1.2, 1.2 + 1e-9, bin_)
    occ = np.histogram(S[:, 1], edges)[0] > 0
    free, runs, start = ~occ, [], None
    for i, f in enumerate(np.append(free, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    runs = [(edges[a], edges[b]) for a, b in runs if (edges[b] - edges[a]) >= min_w and a > 0 and b < len(occ)]
    if not runs:
        return dict(ok=False, why="no interior free interval", x0=x0)
    lo, hi = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1])))
    side_pts = O[((np.abs(O[:, 1] - lo) < 0.15) | (np.abs(O[:, 1] - hi) < 0.15))]
    x1 = float(np.clip(np.percentile(side_pts[:, 0], 95) if len(side_pts) else x0 + 0.4, x0 + 0.2, x0 + 1.0))
    return dict(
        ok=True, x0=round(x0, 3), x1=round(x1, 3), e=round(float(0.5 * (lo + hi)), 3), w=round(float(hi - lo), 3)
    )


FLANK = 0.30  # lateral band outside each gap edge searched for the flanking wall


def gap_features(depth, kind, **kw):
    """estimate() + the flank evidence used by robust_gap (additive 2026-10-16, GEN2_G1_DEBUG): h_left / h_right =
    90th percentile height above the floor of ALL points (up to 2 m) within FLANK m outside the gap's left / right
    edge in the slab x0-0.05 .. x1 (0 if none); n_left / n_right = their counts"""
    g = estimate(depth, kind, **kw)
    f = dict(
        ok=bool(g.get("ok")),
        w=float(g.get("w", 0.0)),
        e=float(g.get("e", 0.0)),
        x0=float(g.get("x0", -1.0)),
        x1=float(g.get("x1", -1.0)),
        h_left=0.0,
        h_right=0.0,
        n_left=0,
        n_right=0,
    )
    if not g.get("ok"):
        return f
    P = CM.points(depth, kind, stride=1)
    zf, _ = CM.floor_level(P, kind)
    h = P[:, 2] - zf
    lo, hi = g["e"] - 0.5 * g["w"], g["e"] + 0.5 * g["w"]
    sl = (P[:, 0] >= g["x0"] - 0.05) & (P[:, 0] <= g["x1"]) & (h > 0.02) & (h < 2.0)
    L = sl & (P[:, 1] >= hi) & (P[:, 1] <= hi + FLANK)
    R = sl & (P[:, 1] <= lo) & (P[:, 1] >= lo - FLANK)
    for nm, m in (("left", L), ("right", R)):
        f[f"n_{nm}"] = int(m.sum())
        f[f"h_{nm}"] = round(float(np.percentile(h[m], 90)), 3) if m.sum() >= 10 else 0.0
    return f


def robust_gap(depth, kind, h_wall, w_max, **kw):
    """the sidestep gate of the fixed gen2 layer: a perceived gap counts only if BOTH flanks are walls >= h_wall above
    the floor and its width <= w_max (the robot's sidestep band); thresholds from gap_gate_fit.py (training split)"""
    f = gap_features(depth, kind, **kw)
    if not f["ok"]:
        return dict(ok=False, why="no gap", feat=f)
    if min(f["h_left"], f["h_right"]) < h_wall:
        return dict(
            ok=False, why=f"flank below wall height ({min(f['h_left'], f['h_right']):.2f} < {h_wall:.2f} m)", feat=f
        )
    if f["w"] > w_max:
        return dict(ok=False, why=f"wider than the sidestep band ({f['w']:.2f} > {w_max:.2f} m)", feat=f)
    return dict(ok=True, x0=f["x0"], x1=f["x1"], e=f["e"], w=f["w"], feat=f)


def validate(n=80):
    import glob

    root = os.path.join(THIS, "..", "..", "data", "gen_v1")
    res = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "sits_*.jsonl"))):
        D = os.path.dirname(f)
        rob = D.split(os.sep)[-1]
        for l in open(f):
            if not l.strip():
                continue
            d = json.loads(l)
            if d["subfamily"] != "env_gap":
                continue
            sp = d["spec"] if isinstance(d["spec"], dict) else eval(d["spec"])  # noqa: S307
            ro = d["rollouts"] if isinstance(d["rollouts"], list) else eval(d["rollouts"])  # noqa: S307
            z = np.load(os.path.join(D, "ep", ro[0]["ep"]), allow_pickle=False)
            est = estimate(z["depth"], rob)
            res.append((rob, sp["params"]["w"], sp["params"]["e"] - float(sp.get("lat0", 0.0)), d["d_dec"], est))
            if len(res) >= n:
                break
        if len(res) >= n:
            break
    ok = [r for r in res if r[4]["ok"]]
    ew = np.array([[r[1], r[4]["w"], r[2], r[4]["e"], r[3], r[4]["x0"]] for r in ok])
    out = dict(n=len(res), found=len(ok))
    if len(ok):
        out.update(
            w_med_abs_err=round(float(np.median(np.abs(ew[:, 0] - ew[:, 1]))), 3),
            e_med_abs_err=round(float(np.median(np.abs(ew[:, 2] - ew[:, 3]))), 3),
            e_med_abs_err_flipped=round(float(np.median(np.abs(ew[:, 2] + ew[:, 3]))), 3),
            x0_minus_ddec_med=round(float(np.median(ew[:, 5] - ew[:, 4])), 3),
        )
    return out


def hazard_extent(depth, kind, half_w=0.45, h_lo=0.05, h_hi=2.0):
    """lane hazard geometry for go2 / detour candidates: x0 = first point 5-200 cm above the floor within +-half_w of
    the centre line (5th percentile), x1 = its far end (95th percentile within x0..x0+1.5 m), lateral free room left /
    right at the hazard (distance from the centre line to the nearest such point beyond half_w, capped at 1.2 m)"""
    P = CM.points(depth, kind, stride=2)
    zf, _ = CM.floor_level(P, kind)
    h = P[:, 2] - zf
    O = P[(h > h_lo) & (h < h_hi) & (P[:, 0] > 0.3) & (P[:, 0] < 5.0)]
    lane = O[np.abs(O[:, 1]) < half_w]
    if len(lane) < 20:
        return dict(ok=False)
    x0 = float(np.percentile(lane[:, 0], 5))
    sl = lane[lane[:, 0] < x0 + 1.5]
    x1 = float(np.percentile(sl[:, 0], 95))
    band = O[(O[:, 0] > x0 - 0.3) & (O[:, 0] < x1 + 0.3)]
    left = band[band[:, 1] > half_w][:, 1]
    right = band[band[:, 1] < -half_w][:, 1]
    return dict(
        ok=True,
        x0=round(x0, 3),
        x1=round(max(x1, x0 + 0.1), 3),
        room_left=round(float(min(left.min() if len(left) else 1.2, 1.2)), 3),
        room_right=round(float(min(-right.max() if len(right) else 1.2, 1.2)), 3),
    )


def hazard_features(depth, kind, half_w=0.45, h_lo=0.05, h_hi=2.0):
    """hazard_extent() + the evidence for robust_hazard (additive 2026-10-16, GEN2_G1_DEBUG): h90 = 90th percentile
    height above the floor of the lane hazard points (x0 .. x0 + 1.5 m, |y| < half_w)"""
    hz = hazard_extent(depth, kind, half_w, h_lo, h_hi)
    f = dict(hz, h90=0.0)
    if not hz.get("ok"):
        return f
    P = CM.points(depth, kind, stride=2)
    zf, _ = CM.floor_level(P, kind)
    h = P[:, 2] - zf
    m = (h > h_lo) & (h < h_hi) & (np.abs(P[:, 1]) < half_w) & (P[:, 0] >= hz["x0"]) & (P[:, 0] <= hz["x0"] + 1.5)
    f["h90"] = round(float(np.percentile(h[m], 90)), 3) if m.sum() >= 10 else 0.0
    return f


def robust_hazard(depth, kind, h_min):
    """the go2 / detour gate of the fixed gen2 layer: the lane hazard must be a discrete obstacle reaching >= h_min
    (per robot, from hazard_gate_fit in gap_gate_fit.py, training split) -- not ground relief"""
    f = hazard_features(depth, kind)
    if not f.get("ok"):
        return dict(f, ok=False, why="no hazard")
    if f["h90"] < h_min:
        return dict(f, ok=False, why=f"hazard below obstacle height ({f['h90']:.2f} < {h_min:.2f} m)")
    return f


def hazard_onset(depth, kind, cam_h, half_w=0.45, h_a=0.02, h_top=0.10, bin_w=0.05):
    """ADDITIVE 2026-10-17 (analysis/GYM_COURSE_6DEG.md, go2 commit reference). Where the lane surface starts to rise
    above the robot's support level, from the decision depth frame. hazard_extent()'s x0 is the first point > 5 cm
    above a floor estimated from the points 0.2-1.2 m ahead; on a ramp those points are already on the slope, so the
    floor estimate rises with them and x0 lands far up the slope (1.14 m ahead of the root with the toe under the robot
    on the 6 deg course entrance, reproduced on a synthetic depth frame), past where the robot stalls.
    Here the support level is the camera-height prior plus the commanded body-height offset (cam_h, passed in), the lane
    profile is the median height per 5 cm bin (|y| < half_w), and the onset is where a line fitted to the first rising
    bins (from the first bin above h_a up to h_top, at most 0.6 m) meets the support level. If the nearest visible bin
    is already above h_a (the rise started under / behind the camera's near limit) the same fit extrapolates backwards.
    A step face (fit slope > 60 deg) -> onset = the first bin above h_a. -> dict(ok, onset, x_first, slope_deg)"""
    P = CM.points(depth, kind, stride=2)
    L = P[(np.abs(P[:, 1]) < half_w) & (P[:, 0] > 0.15) & (P[:, 0] < 5.0)]
    if len(L) < 50:
        return dict(ok=False)
    hl = L[:, 2] + float(cam_h)
    edges = np.arange(0.15, 5.0 + 1e-9, bin_w)
    idx = np.digitize(L[:, 0], edges) - 1
    xs, hs = [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() >= 5:
            xs.append(0.5 * (edges[k] + edges[k + 1]))
            hs.append(float(np.median(hl[m])))
    xs, hs = np.array(xs), np.array(hs)
    up = np.where((hs > h_a) & (hs < 2.0))[0]
    if len(xs) < 3 or not len(up):
        return dict(ok=False)
    i0 = int(up[0])
    sel = [i for i in range(i0, len(xs)) if xs[i] <= xs[i0] + 0.6 and hs[i] <= h_top]
    if i0 > 0:
        sel = [i0 - 1] + sel
    if len(sel) < 2:
        return dict(
            ok=True, onset=round(float(xs[i0] - 0.5 * bin_w), 3), x_first=round(float(xs[i0]), 3), slope_deg=90.0
        )
    a, b = np.polyfit(xs[sel], hs[sel], 1)
    sl = math.degrees(math.atan(a)) if a > 0 else 0.0
    if a <= 1e-3 or sl > 60.0:
        onset = float(xs[i0] - 0.5 * bin_w)
    else:
        onset = float(-b / a)
    return dict(ok=True, onset=round(max(onset, -0.5), 3), x_first=round(float(xs[i0]), 3), slope_deg=round(sl, 1))


if __name__ == "__main__":
    print(json.dumps(validate()))
