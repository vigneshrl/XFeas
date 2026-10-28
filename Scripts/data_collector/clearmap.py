"""
gen_v2 clearance map (pure numpy; analysis/GEN_V2_SPEC.md section 5). Recomputable from the obs npz:
    clear_map(depth_hist, cam_hist, T_dec, floor_z, K)  ->  (4, 60, 32) float16
Camera convention (gensim.GenSim.snapshot / visible): M = CAMERA->WORLD matrix (the camera prim's xform) in USD
ROW-vector form ([p_cam, 1] @ M gives world coordinates; eye = M[3, :3]; the camera looks along -z with +y up). Pixel (u, v) of an N x N image with focal f (px) and centre
c = N/2: ray_cam = ((u + .5 - c)/f, -(v + .5 - c)/f, -1); a depth value d (distance to the image plane) gives the camera
point d * ray_cam.
Grid: decision base frame (yaw only), x forward in [X0, X1), y left in [Y0, Y1), cell CELL. Heights relative to floor_z.
Per cell: heights sorted; ground cluster = lowest point + every next point within GAP of the previous one;
ground = top of that cluster; overhead = first point above the gap (NaN if none); free = overhead - ground (9.9 if no
overhead; NaN if unobserved); n = log1p(count).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json
import numpy as np

X0, X1, Y0, Y1, CELL = -0.5, 5.5, -1.6, 1.6, 0.1
NX, NY = int(round((X1 - X0) / CELL)), int(round((Y1 - Y0) / CELL))
GAP = 0.12
DMIN, DMAX = 0.15, 8.0
HMIN, HMAX = -0.6, 2.5
FX224 = 12.0 / 20.955 * 224.0


def intrinsics(N):
    f = 12.0 / 20.955 * N
    return np.array([[f, 0, N / 2.0], [0, f, N / 2.0], [0, 0, 1.0]], np.float32)


MINPOOL_SRC = 224          # source resolution of the min-pooled N x N depth (min_pool_depth)


def minpool_subpixel(d, S=MINPOOL_SRC, rel=0.002):
    """(additive 2026-10-19, env XFEAS_PERC_MINPOOL_FIX=1) sub-pixel ray position of each min-pooled depth pixel.
    min_pool_depth keeps the NEAREST source pixel of each block, but backproject() casts it along the block-CENTRE ray:
    on a surface seen obliquely the nearest source pixel lies at the block side towards decreasing depth, so the
    centre-ray point sits in front of the surface -- floor points come out too HIGH and overhang-underside points
    (seen from below) too LOW (gen_v3 training overhangs: ground +0.8 cm, underside -0.9 cm, h_free -1.8 cm median).
    Per axis, if the pooled depth decreases towards one neighbour (central difference > rel * d), the ray is moved to
    the centre of the extreme source pixel of the block on that side (exact for locally planar depth; at an occlusion
    edge it moves into the near surface, which is the one the min belongs to). -> (u_eff, v_eff) in N-pixel units."""
    N = d.shape[0]
    idx = (np.arange(N + 1) * S / N).astype(int)
    lo = (idx[:-1] + 0.5) * N / S; hi = (idx[1:] - 0.5) * N / S; ce = np.arange(N) + 0.5
    dd = np.where(np.isfinite(d) & (d > DMIN), d, np.nan).astype(np.float64)
    gu = np.full_like(dd, 0.0); gv = np.full_like(dd, 0.0)
    gu[:, 1:-1] = dd[:, 2:] - dd[:, :-2]; gv[1:-1, :] = dd[2:, :] - dd[:-2, :]
    gu = np.nan_to_num(gu); gv = np.nan_to_num(gv)
    thr = rel * np.nan_to_num(dd, nan=1.0)
    ue = np.where(gu > thr, lo[None, :], np.where(gu < -thr, hi[None, :], ce[None, :]))
    ve = np.where(gv > thr, lo[:, None], np.where(gv < -thr, hi[:, None], ce[:, None]))
    return ue, ve


def backproject(depth, M):
    """depth (N,N) distance to image plane, M cam->world (row-vector) -> world points (n,3)"""
    import os as _os
    N = depth.shape[0]
    f = 12.0 / 20.955 * N; c = N / 2.0
    v, u = np.mgrid[0:N, 0:N]
    d = depth.astype(np.float64)
    ok = (d > DMIN) & (d < DMAX) & np.isfinite(d)
    if _os.environ.get("XFEAS_PERC_MINPOOL_FIX", "0") == "1" and N < MINPOOL_SRC:
        ue, ve = minpool_subpixel(d)
        xc = (ue[ok] - c) / f; yc = -(ve[ok] - c) / f
    else:
        xc = (u[ok] + 0.5 - c) / f; yc = -(v[ok] + 0.5 - c) / f
    P = np.stack([xc * d[ok], yc * d[ok], -d[ok], np.ones(ok.sum())], 1)
    W = P @ M
    return W[:, :3]


def to_base(W, T_dec):
    """world points -> decision base frame (yaw only). T_dec = (x, y, yaw) of the base at the decision"""
    x, y, yaw = T_dec
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = W[:, 0] - x, W[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy, W[:, 2]], 1)


def grid(Pb, floor_z):
    h = Pb[:, 2] - floor_z
    ix = np.floor((Pb[:, 0] - X0) / CELL).astype(int); iy = np.floor((Pb[:, 1] - Y0) / CELL).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (h > HMIN) & (h < HMAX)
    ix, iy, h = ix[ok], iy[ok], h[ok]
    out = np.full((4, NX, NY), np.nan, np.float32)
    out[3] = 0.0
    if len(h) == 0:
        return out
    key = ix * NY + iy
    o = np.lexsort((h, key)); key, h = key[o], h[o]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]; ends = np.r_[starts[1:], len(key)]
    for a, b in zip(starts, ends):
        hh = h[a:b]
        k = key[a]; i, j = divmod(int(k), NY)
        gaps = np.flatnonzero(np.diff(hh) > GAP)
        if len(gaps):
            g = hh[gaps[0]]; ov = hh[gaps[0] + 1]
            out[0, i, j] = g; out[1, i, j] = ov; out[2, i, j] = ov - g
        else:
            out[0, i, j] = hh[-1]; out[2, i, j] = 9.9
        out[3, i, j] = np.log1p(b - a)
    return out


SELF_D, SELF_ROWS = 0.6, 0.39


EXACT_SELF_MASK = False     # v4 (isaac_sim/camera_mount.py): robot pixels were removed exactly at render -> no heuristic
# offline tools on v4 data: env XFEAS_SELF_MASK=exact


def self_mask(depth):
    """pixels of ONE frame that see the ROBOT ITSELF (Spot's arm / gripper swaying in the top of the head image):
    top 39 % of rows (elevation above ~-20 deg, where the floor is > 1.4 m away) and closer than 0.6 m, dilated by 2 px.
    v4: all-False when EXACT_SELF_MASK (set by camera_mount) or env XFEAS_SELF_MASK=exact -- the heuristic would delete
    real structure within 0.6 m (e.g. a beam a head camera sees close)."""
    import os as _os
    d = np.asarray(depth, np.float32); N = d.shape[-1]
    if EXACT_SELF_MASK or _os.environ.get("XFEAS_SELF_MASK", "") == "exact":
        return np.zeros((N, N), bool)
    m = np.zeros((N, N), bool); r = int(SELF_ROWS * N)
    m[:r] = d[:r] < SELF_D
    k = max(1, N // 64)
    out = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    out[r + k:] = False
    return out


def clear_map(depth_hist, cam_hist, T_dec, floor_z, stride=1):
    """depth_hist (K,N,N), cam_hist (K,4,4) cam->world, T_dec (x, y, yaw), floor_z (world z of the floor) -> (4,NX,NY) f16.
    Robot self-pixels (self_mask) are removed first."""
    D = np.array(depth_hist, np.float32)
    for k in range(len(D)):
        D[k][self_mask(D[k])] = 0.0                  # depth 0 -> dropped by DMIN
    W = [backproject(d if stride == 1 else _pool(d, stride), M) for d, M in zip(D, cam_hist)]
    Pb = to_base(np.concatenate(W, 0), T_dec)
    return grid(Pb, floor_z).astype(np.float16)


def _pool(d, k):
    N = d.shape[0] // k
    return np.asarray(d)[:N * k, :N * k].reshape(N, k, N, k).min((1, 3))


def meta():
    return json.dumps(dict(frame="decision base, yaw only; x forward, y left", x=[X0, X1], y=[Y0, Y1], cell=CELL, shape=[4, NX, NY],
                           channels=["ground", "overhead", "free", "log1p_n"], gap=GAP, depth_range=[DMIN, DMAX],
                           h_range=[HMIN, HMAX], floor="lane floor under the robot at the decision", no_overhead=9.9))


def min_pool_depth(dep224, N=128):
    """224 depth -> N x N by min over the source pixels each target pixel covers (thin obstacles survive)"""
    d = np.asarray(dep224, np.float32)
    S = d.shape[0]
    idx = (np.arange(N + 1) * S / N).astype(int)
    out = np.empty((N, N), np.float32)
    rowmin = np.minimum.reduceat(d, idx[:-1], axis=0)
    out[:] = np.minimum.reduceat(rowmin, idx[:-1], axis=1)
    return out


def _selftest():
    # self-test: a synthetic camera 1 m above a floor at z=0, looking along +x pitched down 26 deg, with a slab at
    # x 2.0-2.5, z 0.9-0.95 over y +-1 and a 0.15 m box at x 1.5-1.7
    import math
    N = 128; f = 12.0 / 20.955 * N; c = N / 2
    eye = np.array([0.0, 0.0, 1.0]); pitch = -0.45
    fwd = np.array([math.cos(pitch), 0, math.sin(pitch)]); right = np.array([0, -1.0, 0]); up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], 0)                         # rows: camera x, y, z axes in world
    M = np.eye(4); M[:3, :3] = R.T; M[3, :3] = -eye @ R.T      # row-vector world->cam
    dep = np.full((N, N), 60.0, np.float32)
    for v in range(N):
        for u in range(N):
            rc = np.array([(u + .5 - c) / f, -(v + .5 - c) / f, -1.0])
            rw = rc @ R                                       # direction in world (row-vector inverse rotation)
            best = None
            for (lo, hi) in (((2.0, -1.0, 0.9), (2.5, 1.0, 0.95)), ((1.5, -0.3, 0.0), (1.7, 0.3, 0.15)), ((-5, -5, -1), (20, 5, 0.0))):
                lo, hi = np.array(lo), np.array(hi)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1 = (lo - eye) / rw; t2 = (hi - eye) / rw
                tn = np.nanmax(np.minimum(t1, t2)); tf = np.nanmin(np.maximum(t1, t2))
                if tn <= tf and tf > 0:
                    t = tn if tn > 0 else tf
                    best = t if best is None else min(best, t)
            if best is not None:
                dep[v, u] = best * 1.0                        # |rc_z| = 1 -> distance to image plane = t
    m = clear_map(dep[None], np.linalg.inv(M)[None], (0.0, 0.0, 0.0), 0.0).astype(np.float32)
    ix = lambda x: int((x - X0) / CELL); iy = lambda y: int((y - Y0) / CELL)
    print("floor cell x=1.0:", m[:, ix(1.0), iy(0.0)])
    print("box cell x=1.6:", m[:, ix(1.6), iy(0.0)])
    print("slab cell x=2.2:", m[:, ix(2.2), iy(0.5)])
    return m


if __name__ == "__main__":
    m = _selftest()
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for x in np.arange(0.5, 3.0, 0.1):
        i = int((x - X0) / CELL)
        print(round(x, 2), m[:, i, int((0.0 - Y0) / CELL)], m[:, i, int((0.5 - Y0) / CELL)])
