"""
XFEAS_GEN3_TRIGGER=dist (additive 2026-10-19; analysis/GEN3_HEADCAM.md): DISTANCE-triggered execution of gen3 programs.
Pure numpy. Used by gen3_methods._Gen3Layer; default (unset) = the time-based behaviour (program starts at the decision,
two_phase commit point from the one-time decision-frame hazard x0).

Which programs are triggered (everything else starts at the decision as before):
  posture   constant / two_phase whose action-phase (B) posture differs from the nominal posture (crouch / lean / narrow)
            hazard kind 'over' if the action's top_b < stand_top - 0.03 (a crouch), else 'lane'
  sidestep  turn_side_turn, hazard kind 'gap'
Hazard edges are measured on the fused ego-motion-compensated MEMORY points (feas_mem.mem_points: decision base frame, x fwd,
y left, h above the floor) at every memory update (each decision while waiting, every PERIOD during programs), stored in
WORLD coordinates, and the live distance is (edge - base) . lane heading at every control tick (odometry between updates):
  'over'  points with h in [H_OVER_MIN 0.5 (feas v0.4 lower-body band top), stand_top + top_clear] within |y| <= 0.30
          (feas corridor half width) -- anything that would hit the standing robot but leaves the lower body free
  'lane'  points with h in [0.05, 2.0] within |y| <= 0.45 (gap_estimate.hazard_extent band / half width)
          run = first contiguous run of 0.1 m x-bins with >= 3 points (one empty bin bridged), x >= -0.5;
          front x0 = 2nd percentile, end x1 = 98th percentile of the run's points
  'gap'   gap_entrance(): the first 0.1 m bin where the free lateral interval around the centre line (0.08-1.0 m points)
          is narrower than the probe's standing width + 2 x 0.05 m; end = last bin of that narrow run (v2; v1 used
          gap_estimate.estimate's 5th-percentile wall front, which the lane's fence rails pulled to 0.35 m)
Trigger / release rules (every constant from the probe or from existing geometry, none tuned):
  posture start   d_live(front) <= d_start = t_reach(op) * v_meas + x_front(op) + POS_MARGIN
                  t_reach(op): the probe's own measured settle time of that posture (trace: time after the command change
                  until the top stays within 2 cm of its settled value; ops without a trace segment: the slowest measured);
                  v_meas: measured forward speed (body frame) at that tick; x_front(op) = max(op depth - stand_depth / 2,
                  stand_depth / 2): the frontmost body point ahead of the pelvis in that posture (probe body depth);
                  POS_MARGIN = 0.10 m (one memory cell).
  posture release (stand up / program end) only when the hazard end is behind the head:
                  x1_live <= min(x_head_live, x_head_stand) - (stand_depth / 2 + POS_MARGIN)
                  (x_head = the head-camera eye relative to the pelvis along the heading, live from torso_link; the
                  standing value is taken on flat ground before the first program; the end must clear the post-stand-up
                  head too, and half the standing body depth behind it). x1 is the max over updates (a run only grows
                  as more of it is seen). This replaces the two_phase d_level stand-up and the s_done end for these programs.
  association     an update whose run starts > ASSOC (0.5 m) beyond the tracked end (gap: whose front moved > 0.5 m) is
                  another hazard and is ignored.
  sidestep start  d_live(gap front) <= d_commit + 0.05 (TurnSideCtl3's own turn point), then the executor runs as before
                  with s_end updated live.
While waiting the robot walks the nominal posture at the nominal speed and the layer keeps re-deciding every PERIOD:
a stock decision cancels the waiting program; another non-stock program replaces it (same macro + same posture: kept).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
import numpy as np

H_OVER_MIN = 0.5
HALF_W_OVER = 0.30
HALF_W_LANE = 0.45
POS_MARGIN = 0.10
BIN = 0.1
ASSOC = 0.5           # a new run starting > 0.5 m beyond the tracked end (gap: front moved > 0.5 m) is another hazard
# v4 execution fix (a) (2026-10-19, analysis/REBUILD_V4_PHASE01.md; XFEAS_GEN3_END_ROBUST, default 1 = on; 0 = the old
# max-over-updates end): the tracked END is the per-coordinate MEDIAN of the last END_K updates' world ends instead of the
# furthest end ever seen. In run B' (gen3_headcam) ONE update (t 10.6 s) read the run 1.6 m too long (x1 3.72 vs ~2.1; every
# other update within -0.19..+0.02 m of the true end) and the max kept it -> stand-up 2.15 m late. A median over 5 updates
# ignores up to 2 outliers; a run that really grows (its end coming into view) is followed with a lag of <= 3 updates.
END_ROBUST = os.environ.get("XFEAS_GEN3_END_ROBUST", "1") == "1"
END_K = 5


def probe_settle_times(probe, tol=0.02):
    """{(body_height, pitch) rounded to 2 dp: seconds} from the probe trace (10 Hz), see module doc"""
    import virtual as VC
    T = probe.get("trace") or {}
    if not T or "cmd" not in T:
        return {}
    t = np.asarray(T["t"], float); c = np.asarray(T["cmd"], float); a = np.asarray(T["ach"], float)
    k = list(VC.ACH_KEYS).index("top")
    # posture command = body_height, pitch AND the mode column (index 6; modal controllers such as SONIC change posture by mode
    # with body_height / pitch unset = NaN). NaN-safe: NaN == NaN counts as unchanged. (Before 2026-10-21 only cols 3:5 with a
    # NaN-unsafe compare: SONIC's whole probe was one segment and every crouch read a 406 s settle time -> crouch at first sight.)
    pc = c[:, [3, 4, 6]] if c.shape[1] > 6 else c[:, 3:5]
    def _chg(u, v):
        both_nan = np.isnan(u) & np.isnan(v)
        return bool(np.any((np.abs(u - v) > 1e-3) & ~both_nan) or np.any(np.isnan(u) != np.isnan(v)))
    ch = [0] + [i for i in range(1, len(t)) if _chg(pc[i], pc[i - 1])] + [len(t)]
    out = {}
    for s, e in zip(ch[:-1], ch[1:]):
        if s == 0 or e - s < 5:
            continue
        top = a[s:e, k]; fin = float(np.median(top[-5:]))
        for j in range(len(top)):
            if np.all(np.abs(top[j:] - fin) <= tol):
                ts = float(t[s + j] - t[s])
                out[(round(float(c[s, 3]), 2), round(float(c[s, 4]), 2))] = ts
                out.setdefault("_by_top", []).append((fin, ts))          # settled top height -> settle time (any controller)
                break
    return out


def _run(xs, x_lo=-0.5, min_pts=3):
    if len(xs) == 0:
        return None
    xs = np.sort(xs[xs >= x_lo])
    if len(xs) == 0:
        return None
    b = np.floor((xs - x_lo) / BIN).astype(int)
    cnt = np.bincount(b)
    occ = cnt >= min_pts
    if not occ.any():
        return None
    i0 = int(np.argmax(occ)); i1 = i0
    j = i0 + 1
    while j < len(occ):
        if occ[j]:
            i1 = j; j += 1
        elif j + 1 < len(occ) and occ[j + 1]:
            i1 = j + 1; j += 2
        else:
            break
    sel = xs[(b >= i0) & (b <= i1)]
    return float(np.percentile(sel, 2)), float(np.percentile(sel, 98)), int(len(sel))


def hazard_edges(P, kind, stand_top=1.27, top_clear=0.03, stand_w=0.49):
    """P (n, 3) memory points [x fwd, y left, h above floor] -> dict(ok, x0, x1, n) in the base frame of P"""
    if P is None or len(P) == 0:
        return dict(ok=False)
    if kind == "gap":
        return gap_entrance(P, stand_w)
    if kind == "over":
        m = (P[:, 2] >= H_OVER_MIN) & (P[:, 2] <= stand_top + top_clear) & (np.abs(P[:, 1]) <= HALF_W_OVER)
    else:
        m = (P[:, 2] >= 0.05) & (P[:, 2] <= 2.0) & (np.abs(P[:, 1]) <= HALF_W_LANE)
    r = _run(P[m, 0])
    if r is None:
        return dict(ok=False)
    return dict(ok=True, x0=round(r[0], 3), x1=round(r[1], 3), n=r[2])


def gap_entrance(P, stand_w, h_lo=0.08, h_hi=1.0, y_max=1.2, x_lo=0.0, x_hi=5.0, margin=0.05, min_pts=2):
    """v2 (2026-10-19, after run A v1: the fence rails along the lane made estimate()'s 5th-percentile 'wall front' read
    0.35 m ahead, 4.4 m before the funnel). Gap ENTRANCE = the first 0.1 m x-bin (x >= x_lo) where the free lateral
    interval around the robot's centre line (nearest obstacle point, 0.08-1.0 m above the floor, on each side within
    y_max) is narrower than the robot's own standing width (probe fp stand_width) + 2 x margin, i.e. where walking on
    forward stops fitting; end = the last bin of that contiguous narrow run. A side with no points counts as open."""
    O = P[(P[:, 2] > h_lo) & (P[:, 2] < h_hi) & (np.abs(P[:, 1]) < y_max) & (P[:, 0] >= x_lo) & (P[:, 0] < x_hi)]
    if len(O) < 10:
        return dict(ok=False, why="no obstacle points")
    need = float(stand_w) + 2 * margin
    b = np.floor((O[:, 0] - x_lo) / BIN).astype(int)
    nb = int(b.max()) + 1
    narrow = np.zeros(nb, bool); wid = np.full(nb, 2 * y_max)
    for i in range(nb):
        Y = O[b == i, 1]
        L = Y[Y > 0]; R = Y[Y <= 0]
        if len(L) >= min_pts and len(R) >= min_pts:
            wid[i] = float(np.min(L) - np.max(R))
            narrow[i] = wid[i] < need
    if not narrow.any():
        return dict(ok=False, why="nothing narrower than the robot", need=round(need, 3))
    i0 = int(np.argmax(narrow)); i1 = i0
    while i1 + 1 < nb and (narrow[i1 + 1] or (i1 + 2 < nb and narrow[i1 + 2])):
        i1 += 1 + (not narrow[i1 + 1])
    return dict(ok=True, x0=round(x_lo + i0 * BIN, 3), x1=round(x_lo + (i1 + 1) * BIN, 3), w=round(float(wid[i0:i1 + 1].min()), 3), need=round(need, 3))


def gap_edges(P, h_lo=0.08, h_hi=1.0, bin_=0.02, min_w=0.12):
    """gap_estimate.estimate() after its points stage, on memory points (heights already above the floor)"""
    O = P[(P[:, 2] > h_lo) & (P[:, 2] < h_hi) & (np.abs(P[:, 1]) < 1.2) & (P[:, 0] < 3.5) & (P[:, 0] > 0.3)]
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
            runs.append((start, i)); start = None
    runs = [(edges[a], edges[b]) for a, b in runs if (edges[b] - edges[a]) >= min_w and a > 0 and b < len(occ)]
    if not runs:
        return dict(ok=False, why="no interior free interval", x0=x0)
    lo, hi = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1])))
    side_pts = O[((np.abs(O[:, 1] - lo) < 0.15) | (np.abs(O[:, 1] - hi) < 0.15))]
    x1 = float(np.clip(np.percentile(side_pts[:, 0], 95) if len(side_pts) else x0 + 0.4, x0 + 0.2, x0 + 1.0))
    return dict(ok=True, x0=round(x0, 3), x1=round(x1, 3), e=round(float(0.5 * (lo + hi)), 3), w=round(float(hi - lo), 3))


class Track:
    """world-anchored hazard edges of ONE waiting / running program"""

    def __init__(self, kind, stand_w=0.49):
        self.kind, self.stand_w = kind, float(stand_w)
        self.f_w = None; self.e_w = None; self.n_upd = 0; self.last = None
        self.e_hist = []

    def update(self, P, pose, stand_top, top_clear):
        """P in the base frame of pose (x, y, yaw); -> the edges dict"""
        h = hazard_edges(P, self.kind, stand_top, top_clear, self.stand_w)
        self.last = h
        if not h.get("ok"):
            return h
        x, y, yaw = pose
        d = np.array([math.cos(yaw), math.sin(yaw)])
        f_w = np.array([x, y]) + h["x0"] * d
        e_w = np.array([x, y]) + h["x1"] * d
        if self.f_w is not None:                 # association: the SAME hazard (a later one appearing first is ignored)
            if self.kind == "gap":
                if abs(float((f_w - self.f_w) @ d)) > ASSOC:
                    h = dict(h, rejected=True); self.last = h
                    return h
            elif float((f_w - self.e_w) @ d) > ASSOC:
                h = dict(h, rejected=True); self.last = h
                return h
        self.f_w = f_w                           # front: latest estimate (it is what the trigger approaches)
        if self.e_w is None or self.kind == "gap":
            self.e_w = e_w
            self.e_hist = [e_w]
        elif END_ROBUST:                         # v4 fix (a): median of the last END_K world ends (outlier-robust)
            self.e_hist = (self.e_hist + [e_w])[-END_K:]
            self.e_w = np.median(np.stack(self.e_hist), 0)
        else:                                    # end: furthest seen so far along the heading
            self.e_w = e_w if (e_w - self.e_w) @ d > 0 else self.e_w
        self.n_upd += 1
        return h

    def dist(self, pos, heading):
        """live (front, end) distances ahead of pos along heading (None if never seen)"""
        if self.f_w is None:
            return None, None
        d = np.array([math.cos(heading), math.sin(heading)]); p = np.asarray(pos[:2], float)
        return float((self.f_w - p) @ d), float((self.e_w - p) @ d)
