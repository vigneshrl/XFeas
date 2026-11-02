"""
Non-learned geometric feasibility check (analysis/GEN3_SPEC.md section 8): clearance map (gen_v2 clearmap, (4,60,32):
ground, overhead, free, n_pts; x -0.5..5.5 m rows of 0.1, y -1.6..1.6 m cols of 0.1, decision base frame) + probe limits.
Pure numpy.

    f = feas_vector(clear_map, fp, caps)        -> (16,) f32  + dict
    fa = feas_act(f_dict, prog, driver)         -> (4,)  f32
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os
import numpy as np

FEAS_KEYS = [
    "h_free_min",
    "x_at_h_free",
    "gap_w_min",
    "x_at_gap",
    "h_ground_max",
    "stand_minus_hfree",
    "fits_under_stock",
    "fits_under_best",
    "fits_fwd_stock",
    "fits_fwd_best",
    "fits_side",
    "margin_top_best",
    "margin_w_fwd",
    "margin_w_side",
    "lateral_capable",
    "any_fit",
]
X0, DX, Y0, DY = -0.5, 0.1, -1.6, 0.1
BLOCK_H = 0.15  # a cell whose ground cluster top is above this is an obstacle (wall / furniture / clutter)
MARGIN = 0.03


def _xs(nx):
    return X0 + DX * (np.arange(nx) + 0.5)


def _ys(ny):
    return Y0 + DY * (np.arange(ny) + 0.5)


def obs_points(depth_hist, cam_hist, base_pose_dec, floor_z):
    """decision-frame points (x fwd, y left, h above the lane floor) from the obs memory frames (gen_v2 obs keys);
    robot self-pixels removed as in clearmap"""
    import os, sys
    import clearmap as CM

    D = np.array(depth_hist, np.float32)
    W = []
    for d, M in zip(D, cam_hist):
        d = d.copy()
        d[CM.self_mask(d)] = 0.0
        W.append(CM.backproject(d, M))
    T = (float(base_pose_dec[0]), float(base_pose_dec[1]), float(base_pose_dec[3]))
    P = CM.to_base(np.concatenate(W, 0), T)
    P[:, 2] -= float(floor_z)
    return P


def gap_from_points(P, x_rng=(0.2, 5.0), h_low=(0.15, 0.5), h_wall=0.8, y_wall=1.2, y_ctr=0.5, slab=0.1, min_pts=3):
    """narrowest lateral free interval around the lane centre BETWEEN TWO WALLS, over x-slabs of 0.1 m -> (width, x) or (9.9, nan).
    2 cm lateral bins; a bin BLOCKS when it has >= 2 points in the lower-body band (walls touch the floor, overhead slabs do
    not); a blocking run counts as a WALL only if some bin of it also has points above h_wall (steps, curbs and low clutter
    are not walls, so a step face in front never reads as a gap)."""
    if P is None or len(P) == 0:
        return 9.9, float("nan")
    m = (
        (P[:, 0] >= x_rng[0])
        & (P[:, 0] <= x_rng[1])
        & (np.abs(P[:, 1]) < y_wall)
        & (P[:, 2] >= h_low[0])
        & (P[:, 2] <= 2.0)
    )
    Q = P[m]
    best, xb = 9.9, float("nan")
    if len(Q) == 0:
        return best, xb
    edges = np.arange(-y_wall, y_wall + 1e-9, 0.02)
    nb = len(edges) - 1
    for x0 in np.arange(x_rng[0], x_rng[1], slab):
        S = Q[(Q[:, 0] >= x0) & (Q[:, 0] < x0 + slab)]
        if len(S) < min_pts:
            continue
        low = S[S[:, 2] <= h_low[1]]
        blk = np.histogram(low[:, 1], edges)[0] >= 2
        if blk.sum() < 2:
            continue
        hi_ = S[S[:, 2] >= h_wall]
        tall = np.histogram(hi_[:, 1], edges)[0] >= 1
        # contiguous blocking runs and whether each is a wall
        runs = []
        i = 0
        while i < nb:
            if blk[i]:
                j = i
                while j + 1 < nb and blk[j + 1]:
                    j += 1
                runs.append((i, j, bool(tall[max(0, i - 1) : min(nb, j + 2)].any())))
                i = j + 1
            else:
                i += 1
        w_best = None
        for (a0, a1, wa), (b0, b1, wb) in zip(runs[:-1], runs[1:]):
            lo, hi = edges[a1 + 1], edges[b0]
            if not (wa and wb) or hi - lo < 0.04:
                continue
            if hi > -y_ctr and lo < y_ctr:
                w_best = max(w_best or 0.0, float(hi - lo))
        if w_best is not None and w_best < best:
            best, xb = w_best, float(x0 + slab / 2)
    return best, xb


def wall_cells(P, shape, h_low=(0.15, 0.5)):
    """(NX, NY) bool: clearance-map cells with lower-body-band points (walls / furniture / clutter, not overhead slabs)"""
    out = np.zeros(shape, bool)
    if P is None or len(P) == 0:
        return out
    m = (P[:, 2] >= h_low[0]) & (P[:, 2] <= h_low[1])
    ix = np.floor((P[m, 0] - X0) / DX).astype(int)
    iy = np.floor((P[m, 1] - Y0) / DY).astype(int)
    ok = (ix >= 0) & (ix < shape[0]) & (iy >= 0) & (iy < shape[1])
    cnt = np.zeros(shape, int)
    np.add.at(cnt, (ix[ok], iy[ok]), 1)
    w = cnt >= 3
    d = w.copy()  # dilate 1 cell (wall faces are seen from one side only)
    d[1:] |= w[:-1]
    d[:-1] |= w[1:]
    d[:, 1:] |= w[:, :-1]
    d[:, :-1] |= w[:, 1:]
    return d


X_OVER_V03, X_OVER_V04 = 0.6, 0.0  # v0.4: self-body points removed per frame (feas_mem), memory covers x < 0.6
H_OVER_MIN_V04 = 0.5  # v0.4: top of the lower-body band; a lower "overhead" is a step / terrain fold


def floor_connected(ground, xs, x_from=-0.5, return_ref=False):
    """v0.4: (NX, NY) bool, cell ground is WALKABLE floor reachable from the robot: marching away from the base along x in
    each column, a cell's ground counts as floor when it is within BLOCK_H above (or 0.3 m below) the last floor cell's
    ground (start: the floor under the robot, 0). Ramps / decks / gentle terrain stay floor; a slab top seen without
    the floor below it (camera above the slab) or a box top does not. Replaces v0.3's absolute ground < BLOCK_H test,
    which made a deck 0.17 m above the robot's floor (GC2 ramp head) read as an obstacle top."""
    out = np.zeros(ground.shape, bool)
    refs = np.zeros(ground.shape, np.float32)
    for j in range(ground.shape[1]):
        ref = 0.0
        for i in np.where(xs >= x_from)[0]:
            g = ground[i, j]
            if np.isfinite(g) and ref - 0.3 <= g <= ref + BLOCK_H:
                out[i, j] = True
                ref = float(g)
            refs[i, j] = ref
    return (out, refs) if return_ref else out


def wall_cells_local(P, ref, h_low=(0.15, 0.5)):
    """v0.4 wall cells: as wall_cells, but the lower-body band is measured above the LOCAL floor reference of the cell
    (floor_connected ref), not above the floor under the robot -- a deck / ramp head 0.17 m up is floor, not a wall"""
    shape = ref.shape
    out = np.zeros(shape, bool)
    if P is None or len(P) == 0:
        return out
    ix = np.floor((P[:, 0] - X0) / DX).astype(int)
    iy = np.floor((P[:, 1] - Y0) / DY).astype(int)
    ok = (ix >= 0) & (ix < shape[0]) & (iy >= 0) & (iy < shape[1])
    ix, iy, h = ix[ok], iy[ok], P[ok, 2] - ref[ix[ok], iy[ok]]
    m = (h >= h_low[0]) & (h <= h_low[1])
    cnt = np.zeros(shape, int)
    np.add.at(cnt, (ix[m], iy[m]), 1)
    w = cnt >= 3
    d = w.copy()
    d[1:] |= w[:-1]
    d[:-1] |= w[1:]
    d[:, 1:] |= w[:, :-1]
    d[:, :-1] |= w[:, 1:]
    return d


def geometry(cm, half_w=0.30, x_rng=(0.2, 5.0), y_wall=1.2, P=None, rules="v03"):
    cm = np.asarray(cm, np.float32)
    ground, over, free = cm[0], cm[1], cm[2]
    xs, ys = _xs(cm.shape[1]), _ys(cm.shape[2])
    rows = np.where((xs >= x_rng[0]) & (xs <= x_rng[1]))[0]
    cols = np.where(np.abs(ys) <= half_w)[0]
    if P is not None:
        gap_w, x_gap = gap_from_points(P, x_rng=x_rng, y_wall=y_wall)
    else:
        gap_w, x_gap = 9.9, np.nan
    # overhead: lowest free height over observed corridor cells whose ground is FLOOR (an obstacle top is not a passage),
    # with nothing standing on the floor below (wall_cells), not within 0.6 m of the base (the robot's own arms reach
    # ~0.45 m forward) and not beyond a wall gap (the space behind a wall is seen through the gap only)
    h_free, x_hf = 9.9, np.nan
    # (only for a NARROW wall gap: overhang support columns 1+ m apart also read as two walls)
    xo_max = x_rng[1] if not (np.isfinite(x_gap) and gap_w < 0.9) else min(x_rng[1], x_gap + 0.15)
    v04 = rules == "v04"
    x_over = X_OVER_V04 if v04 else max(x_rng[0], X_OVER_V03)
    orows = np.where((xs >= x_over) & (xs <= xo_max))[0]
    sub = free[np.ix_(orows, cols)]
    gsub_o = ground[np.ix_(orows, cols)]
    ok = np.isfinite(sub) & (sub < 9.0) & np.isfinite(gsub_o) & (gsub_o < BLOCK_H)
    if v04:
        fl, ref = floor_connected(ground, xs, return_ref=True)
        ok = np.isfinite(sub) & (sub < 9.0) & fl[np.ix_(orows, cols)] & (sub >= H_OVER_MIN_V04)
        if P is not None:
            ok &= ~wall_cells_local(P, ref)[np.ix_(orows, cols)]
    elif P is not None:
        ok &= ~wall_cells(P, ground.shape)[np.ix_(orows, cols)]
    if ok.any():
        i = np.argmin(np.where(ok, sub, 99.0))
        r, c = np.unravel_index(i, sub.shape)
        h_free, x_hf = float(sub[r, c]), float(xs[orows[r]])
    gsub = ground[np.ix_(rows, cols)]
    h_g = float(np.nanmax(gsub)) if np.isfinite(gsub).any() else 0.0
    if P is None:  # coarse fallback gap from the 0.1 m cells
        for r in rows:
            g = ground[r]
            blk = np.isfinite(g) & (g > BLOCK_H)
            free_c = np.where(~blk & (np.abs(ys) <= 0.5))[0]
            if len(free_c) == 0:
                continue
            c0 = free_c[np.argmin(np.abs(ys[free_c]))]
            lft = c0
            while lft - 1 >= 0 and not blk[lft - 1]:
                lft -= 1
            rgt = c0
            while rgt + 1 < len(g) and not blk[rgt + 1]:
                rgt += 1
            if (ys[lft] - DY / 2 > -y_wall) and (ys[rgt] + DY / 2 < y_wall):
                w = (rgt - lft + 1) * DY
                if w < gap_w:
                    gap_w, x_gap = float(w), float(xs[r])
    return dict(h_free_min=h_free, x_at_h_free=x_hf, gap_w_min=gap_w, x_at_gap=x_gap, h_ground_max=h_g)


def feas_vector(cm, fp, caps, P=None, rules="v03"):
    g = geometry(cm, P=P, rules=rules)
    st, mtw = float(fp["stand_top"]), float(fp["min_top_walk"])
    sw = float(fp["stand_width"])
    mww = float(np.nan_to_num(fp["min_width_walk"], nan=sw))
    mdw = float(np.nan_to_num(fp["min_depth_walk"], nan=9.9))
    lat = bool(caps.get("lateral_ok", False))
    if not lat and os.environ.get("XFEAS_GEN3_LAT_PER_SIDE", "1") == "1":  # 2026-10-21: either side measured to work
        lat = any(abs(float(caps.get(k, 0.0) or 0.0)) > 0.06 for k in ("vy_max_L", "vy_max_R"))
    hf, gw = g["h_free_min"], g["gap_w_min"]
    d = dict(g)
    d.update(
        stand_minus_hfree=st - hf,
        fits_under_stock=float(st + MARGIN <= hf),
        fits_under_best=float(mtw + MARGIN <= hf),
        fits_fwd_stock=float(sw <= gw),
        fits_fwd_best=float(mww <= gw),
        fits_side=float(lat and float(fp["stand_depth"]) <= gw),
        margin_top_best=hf - mtw,
        margin_w_fwd=gw - mww,
        margin_w_side=(gw - float(fp["stand_depth"])) if lat else -9.9,
        lateral_capable=float(lat),
    )
    d["any_fit"] = float(d["fits_under_best"] and (d["fits_fwd_best"] or d["fits_side"]))
    v = np.array([np.nan_to_num(float(d[k]), nan=-1.0) for k in FEAS_KEYS], np.float32)
    return v, d


def feas_act(fd, prog, driver):
    """margin_top, margin_wid, fits, driver_feasible for the op of the phase that crosses the hazard (S for turn_side_turn,
    else B)"""
    k = "S" if "S" in prog["phases"] else "B"
    p = prog["phases"][k].get("pred")
    top = driver.stand_top if p is None else float(p["top"])
    wid = (
        (driver.stand_d if p is None else float(p["depth"]))
        if k == "S"
        else (driver.stand_w if p is None else float(p["width"]))
    )
    mt = float(fd["h_free_min"]) - top
    mw = float(fd["gap_w_min"]) - wid
    fits = float(mt >= MARGIN and mw >= 0.0)
    return np.array([mt, mw, fits, float(prog.get("driver_feasible", True))], np.float32)
