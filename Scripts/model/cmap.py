#!/usr/bin/env python
"""
Depth CLEARANCE MAP for XFeas-Gen2 (gen2 agent, 2026-10-14). Pure numpy, no Isaac.

Why: gen1 had to INFER an overhang from an 8x8 pooled DINO grid (and partly from the room it was in). The head camera is
GRAVITY-ALIGNED in pitch and roll: every collector (gensim.GenSim._cam_pose / snapshot, xfeas_decider._snapshot,
collect_*.py) sets the camera with SetLookAt(eye, eye + (cos yaw cos p, sin yaw cos p, sin p), world-up), where
(fwd, up, p) = spec.cam of the robot and only the base YAW is used. So a single depth frame + the robot kind gives metric
3-D points in a yaw-aligned, gravity-aligned robot frame without any stored pose -- old data contributes.

Camera (identical in every collector): focal 12 mm, aperture 20.955 mm, 224 x 224 -> fx = fy = 128.27 px, principal
point at the image centre; annotator distance_to_image_plane (= z along the optical axis). Pixel ray of (u, v):
((u + .5 - 112) / fx, -(v + .5 - 112) / fx, -1) in the camera frame (gensim.visible, same convention).

Self-occlusion: points < 0.5 m from the camera are dropped (Spot's stowed arm/gripper, Jackal's own front).
Floor: the base height is not stored, so the floor level under the robot is ESTIMATED from the depth itself: 15th
percentile height of the points 0.2 - 1.2 m ahead of the root within +-0.35 m laterally (the ground right in front of the feet),
clamped to +-0.25 m of the per-kind prior (camera height above floor for that robot at its nominal posture, measured on
null-lane frames by calib(); written into PRIOR_CAM_H). If fewer than 30 such points exist (wall right in front), the
prior is used. The flag channel records which one was used.

Map: robot-centric, x -0.5..5.5 m ahead of the root (48 cells of 0.125 m), y -1.25..+1.25 m (20 cells), channels (C_MAP = 14):
  0..9   fraction of the cell's points in height-above-floor bins
         (<-0.05 drop | -0.05..0.05 floor | .05-.15 | .15-.30 | .30-.50 | .50-.70 | .70-.90 | .90-1.20 | 1.20-1.60 | >1.60)
  10     observed (any point)                      11  log1p(n points) / 5
  12     lowest point above 0.10 m (m / 2; 1.0 if none) -> the overhead clearance / obstacle bottom
  13     highest point below 0.60 m (m / 0.6; 0 if none) -> step / curb / clutter top
History (gen_v2): points of older frames can be added with their pose relative to the current base frame
(build_map(..., extra=[(depth, kind, (dx, dy, dyaw, dz))])); the gravity alignment makes this a planar rigid transform.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np

IMG = 224
FX = IMG * 12.0 / 20.955
# spec.cam (fwd, up, pitch) per robot kind -- copied from robots.SPECS / controllers/*.py (selftest re-reads the sources)
CAM = {"spotp": (0.52, -0.02, -0.45), "g1a": (0.1, 0.55, -0.5), "g1c": (0.1, 0.55, -0.5), "g1": (0.1, 0.55, -0.5),
       "jackal": (0.2, 0.25, -0.3), "iris": (0.15, -0.05, -0.35), "spot": (0.45, 0.15, -0.35), "h1": (0.15, 0.65, -0.5),
       "carter": (0.3, 0.55, -0.35), "anymal": (0.5, 0.2, -0.35), "go1": (0.28, 0.08, -0.3), "jetbot": (0.08, 0.1, -0.3)}
DEFAULT_CAM = (0.3, 0.3, -0.4)
# camera height above the floor at nominal posture (m), median of the near-floor estimate over null/flat frames
# (calib() on gen_v1 + old sets, 2026-10-14; see GEN2_MODEL.md). Unknown kinds: estimate only, clamp to [0.15, 2.0].
PRIOR_CAM_H = {"spotp": 0.50, "g1c": 1.295, "g1a": 1.295, "g1": 1.295, "jackal": 0.31, "iris": 0.82, "carter": 0.79,
               "h1": 1.545, "spot": 0.67}
# calib(): gen_v1 null frames (spotp 0.501 n 86, g1c 1.295 n 26, jackal 0.313 n 5, iris 0.823 n 31) and old sets
# (carter 0.790, h1 1.545, spot 0.669, g1a 1.297, n 80 each). Validation on gen_v1 env_overhang decision frames:
# measured underside vs the spec's u, when the slab is in view: median |err| 0.000-0.001 m, p90 <= 0.018 m (g1c 59,
# spotp 33, iris 8 frames). Only 33/74 Spot slabs with u < 1 m are in view at the decision frame (camera FOV).
X0, X1, NX = -0.5, 5.5, 48          # gen_v2 decisions are 1-3 m before the hazard (GEN_V2_SPEC section 5 uses -0.5..5.5)
Y0, Y1, NY = -1.25, 1.25, 20
HB = np.array([-0.05, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.20, 1.60], np.float32)   # 10 bins
C_MAP = 14
SELF_R = 0.5          # Spot's arm and Jackal's own front sit < 0.5 m from the head camera (checked on gen_v1 frames)
_UV = None


def _rays():
    global _UV
    if _UV is None:
        u = (np.arange(IMG, dtype=np.float32) + 0.5 - IMG / 2) / FX
        _UV = (u[None, :].repeat(IMG, 0), -u[:, None].repeat(IMG, 1))        # x_right/z, y_up/z
    return _UV


def cam_of(kind):
    k = str(kind)
    if k in CAM:
        return CAM[k]
    for p in ("spotp", "g1", "spot", "h1", "jackal", "iris", "carter", "anymal", "go1"):
        if k.startswith(p):
            return CAM[p]
    return DEFAULT_CAM


# ---- XFEAS_CAM_MOUNT=head (additive 2026-10-19, analysis/GEN3_HEADCAM.md) ----------------------------------------------
# A camera that is NOT the gravity-aligned spec.cam camera (the G1 head D435 mount: full 6-DoF pose from torso_link) has
# no fixed pitch, so points() cannot use cam_of(kind). While a mount is set (set_mount(M, base) by the caller, for the
# frame being processed), points() back-projects with that frame's EXACT camera->world matrix M (USD row vectors, same
# intrinsics) and returns the same convention [x fwd of the ROOT (base yaw frame), y left, z relative to the CAMERA
# height M[3, 2]]; world_to_decision() uses the given base instead of eye - fwd * look. Unset (default): unchanged.
_MOUNT = None


_CAM_H = None         # v4 (camera_mount): this frame's eye height above the robot's lowest body point (floor prior)


def set_mount(M=None, base=None, cam_h=None):
    """M (4, 4) camera->world of the CURRENT decision frame, base (x, y, yaw) of the robot root at that frame; None = off.
    cam_h (v4, isaac_sim/camera_mount.py): the frame's camera height above the robot's lowest body point, used as the
    floor prior of floor_level() instead of the fixed per-kind PRIOR_CAM_H (a head camera moves with the posture)."""
    global _MOUNT, _CAM_H
    _MOUNT = None if M is None else (np.asarray(M, np.float64).reshape(4, 4), tuple(float(v) for v in np.asarray(base).ravel()[:3]))
    _CAM_H = None if (M is None or cam_h is None or not np.isfinite(cam_h)) else float(cam_h)


def _points_mount(depth, stride, max_d):
    M, (bx, by, byaw) = _MOUNT
    N = np.asarray(depth).shape[0]
    W = points_world(depth, M, N * 12.0 / 20.955, N / 2.0, stride=stride, max_d=max_d).astype(np.float64)
    c, s = math.cos(byaw), math.sin(byaw)
    dx, dy = W[:, 0] - bx, W[:, 1] - by
    return np.stack([c * dx + s * dy, -s * dx + c * dy, W[:, 2] - M[3, 2]], 1).astype(np.float32)


def points(depth, kind, stride=2, max_d=8.0):
    """depth (224,224) -> (N,3) points [x fwd of ROOT, y left, z up relative to CAMERA height]"""
    if _MOUNT is not None:
        return _points_mount(depth, stride, max_d)
    fwd, up, p = cam_of(kind)
    xr, yu = _rays()
    d = np.asarray(depth, np.float32)[::stride, ::stride]
    xr, yu = xr[::stride, ::stride], yu[::stride, ::stride]
    rng = d * np.sqrt(1.0 + xr * xr + yu * yu)                              # euclidean range from the camera
    m = np.isfinite(d) & (rng > SELF_R) & (d < max_d)                        # < SELF_R: the robot's own arm / front
    d, xr, yu = d[m], xr[m] * d[m], yu[m] * d[m]
    c, s = math.cos(p), math.sin(p)
    X = d * c - yu * s + fwd
    Y = -xr
    Z = d * s + yu * c
    return np.stack([X, Y, Z], 1)


def points_world(depth, M, f, c, stride=2, max_d=8.0):
    """gen_v2 frame with its camera->world matrix M (USD row-vector convention, GEN_V2_SPEC section 5) -> (N,3) world"""
    d = np.asarray(depth, np.float32)[::stride, ::stride]
    H, W = np.asarray(depth).shape
    u = (np.arange(W, dtype=np.float32) + 0.5 - c) / f
    v = -(np.arange(H, dtype=np.float32) + 0.5 - c) / f
    xr, yu = u[None, :].repeat(H, 0)[::stride, ::stride], v[:, None].repeat(W, 1)[::stride, ::stride]
    rng = d * np.sqrt(1.0 + xr * xr + yu * yu)
    m = np.isfinite(d) & (rng > SELF_R) & (d > 0.05) & (d < max_d)
    pc = np.stack([xr[m] * d[m], yu[m] * d[m], -d[m], np.ones(int(m.sum()), np.float32)], 1)
    return (pc @ np.asarray(M, np.float64))[:, :3].astype(np.float32)


def world_to_decision(Pw, M_dec, kind):
    """world points -> the DECISION root frame convention of points(): x fwd of the root (yaw-only), y left,
    z relative to the decision CAMERA height"""
    fwd, _, _ = cam_of(kind)
    eye = np.asarray(M_dec, np.float64)[3, :3]
    look = -np.asarray(M_dec, np.float64)[2, :3]                 # camera looks along -z
    yaw = math.atan2(look[1], look[0])
    c, s = math.cos(yaw), math.sin(yaw)
    root = eye[:2] - fwd * np.array([c, s])
    if _MOUNT is not None:                                        # XFEAS_CAM_MOUNT=head: the given base, not eye - fwd * look
        yaw = _MOUNT[1][2]; c, s = math.cos(yaw), math.sin(yaw); root = np.array(_MOUNT[1][:2])
    d = Pw[:, :2] - root
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, Pw[:, 2] - eye[2]], 1).astype(np.float32)


def floor_level(P, kind):
    """-> (z_floor relative to camera, used_prior flag)"""
    near = P[(P[:, 0] > 0.2) & (P[:, 0] < 1.2) & (np.abs(P[:, 1]) < 0.35)]
    prior = PRIOR_CAM_H.get(str(kind)) if _CAM_H is None else _CAM_H
    if prior is None:
        for k, v in PRIOR_CAM_H.items():
            if str(kind).startswith(k):
                prior = v
                break
    if len(near) >= 30:
        z = float(np.percentile(near[:, 2], 15))     # lowest dominant level (props in front pull a median up)
        if prior is not None and abs(-z - prior) > 0.25:
            return -float(np.clip(-z, prior - 0.25, prior + 0.25)), 1.0
        return z, 0.0
    if prior is not None:
        return -prior, 1.0
    return -0.5, 1.0


def rasterise(P, zf):
    """points (rel. camera height) + floor -> (C_MAP, NX, NY) float32"""
    M = np.zeros((C_MAP, NX, NY), np.float32)
    M[12] = 1.0
    if not len(P):
        return M
    h = P[:, 2] - zf
    ix = np.floor((P[:, 0] - X0) / (X1 - X0) * NX).astype(np.int64)
    iy = np.floor((P[:, 1] - Y0) / (Y1 - Y0) * NY).astype(np.int64)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (h < 2.2)
    ix, iy, h = ix[ok], iy[ok], h[ok]
    if not len(h):
        return M
    cell = ix * NY + iy
    n = np.bincount(cell, minlength=NX * NY).astype(np.float32)
    hb = np.searchsorted(HB, h, side="right")                               # 0..9
    cnt = np.bincount(cell * 10 + hb, minlength=NX * NY * 10).reshape(NX * NY, 10).astype(np.float32)
    frac = cnt / np.maximum(n[:, None], 1.0)
    M[:10] = frac.T.reshape(10, NX, NY)
    M[10] = (n > 0).reshape(NX, NY)
    M[11] = (np.log1p(n) / 5.0).reshape(NX, NY)
    lo = np.full(NX * NY, 2.0, np.float32)
    sel = h > 0.10
    np.minimum.at(lo, cell[sel], h[sel])
    M[12] = (lo / 2.0).reshape(NX, NY)
    hi = np.zeros(NX * NY, np.float32)
    sel = (h < 0.60) & (h > -0.5)
    np.maximum.at(hi, cell[sel], np.maximum(h[sel], 0.0))
    M[13] = (hi / 0.6).reshape(NX, NY)
    return M


def transform(P, pose):
    """points of an OLDER frame (its root frame) -> current root frame; pose = (dx, dy, dyaw, dz) of the old root
    expressed in the current root frame (gravity-aligned, so a planar rigid transform + height shift)"""
    dx, dy, dyaw, dz = pose
    c, s = math.cos(dyaw), math.sin(dyaw)
    Q = P.copy()
    Q[:, 0] = c * P[:, 0] - s * P[:, 1] + dx
    Q[:, 1] = s * P[:, 0] + c * P[:, 1] + dy
    Q[:, 2] = P[:, 2] + dz
    return Q


def build_map(depth, kind, extra=(), stride=2, world_frames=None):
    """current depth (+ optional older frames with poses, or gen_v2 frames with exact camera matrices) -> (map, info)
    world_frames = (M_dec, [(depth_k, M_k, f_k, c_k), ...]) : gen_v2 obs (cam_hist), fused exactly"""
    P = points(depth, kind, stride)
    zf, used_prior = floor_level(P, kind)
    allP = [P]
    for dep_k, kind_k, pose in extra:
        allP.append(transform(points(dep_k, kind_k, stride), pose))
    if world_frames is not None:
        M_dec, frs = world_frames
        for dep_k, M_k, f_k, c_k in frs:
            allP.append(world_to_decision(points_world(dep_k, M_k, f_k, c_k, stride=1 if np.asarray(dep_k).shape[0] <= 128 else 2), M_dec, kind))
    P = np.concatenate(allP, 0) if len(allP) > 1 else P
    return rasterise(P, zf), dict(cam_h=-zf, used_prior=used_prior, n_frames=len(allP))


def calib(paths_by_kind, n=200, seed=0):
    """median near-floor camera height per kind on null / flat frames (no prior) -> dict"""
    rng = np.random.default_rng(seed)
    out = {}
    for kind, paths in paths_by_kind.items():
        paths = [paths[i] for i in rng.choice(len(paths), min(n, len(paths)), replace=False)]
        hs = []
        for p in paths:
            z = np.load(p, allow_pickle=False)
            P = points(z["depth"], kind)
            near = P[(P[:, 0] > 0.2) & (P[:, 0] < 1.2) & (np.abs(P[:, 1]) < 0.35)]
            if len(near) >= 30:
                hs.append(-float(np.median(near[:, 2])))
        out[kind] = dict(median=float(np.median(hs)) if hs else None, iqr=(float(np.percentile(hs, 25)), float(np.percentile(hs, 75))) if hs else None, n=len(hs))
    return out


def selftest():
    import os, re
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "..", "xembody", "controllers", "spot_posture.py")).read()
    m = re.search(r"cam=\(([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)\)", src)
    assert tuple(map(float, m.groups())) == CAM["spotp"], m.groups()
    src = open(os.path.join(here, "..", "xembody", "controllers", "g1_amo.py")).read()
    m = re.search(r"cam=\(([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)\)", src)
    assert tuple(map(float, m.groups())) == CAM["g1a"], m.groups()
    # synthetic: flat floor 0.5 m below a spotp camera + a slab whose underside is 0.45 m above the floor at x 2..2.5 m
    fwd, up, p = CAM["spotp"]
    xr, yu = _rays()
    c, s = math.cos(p), math.sin(p)
    dirz = s + yu * c                                     # z component per unit image-plane depth
    dirx = c - yu * s
    d_floor = np.where(dirz < -1e-3, -0.5 / np.minimum(dirz, -1e-3), 60.0)
    d_slab = np.where(dirz > 1e-3, 0.15 / np.where(dirz > 1e-3, dirz, 1.0), 60.0)   # underside 0.65 m above floor
    xs = d_slab * dirx + fwd
    dep = np.where((xs > 2.0) & (xs < 2.5), d_slab, d_floor).astype(np.float32)
    M, info = build_map(dep, "spotp")
    assert abs(info["cam_h"] - 0.5) < 0.02, info
    ix = int((2.2 - X0) / (X1 - X0) * NX)
    i0, i1 = int((2.0 - X0) / (X1 - X0) * NX), int((2.5 - X0) / (X1 - X0) * NX)
    under = float(M[12, i0:i1, NY // 2].min() * 2.0)                                # cells x 2.0..2.5 m
    assert under == pytest_approx(0.65, 0.03), under
    i1m = int((1.0 - X0) / (X1 - X0) * NX)
    assert M[1, i1m, NY // 2] > 0.9, M[:, i1m, NY // 2]                              # floor cell at x 1.0 m
    # exact-camera path (gen_v2 cam_hist) == gravity-aligned path for the decision frame (lookAt matrix as gensim)
    yaw, base = 0.7, np.array([3.0, -2.0, 0.5])
    fwd_, up_, p_ = CAM["spotp"]
    eye = base + np.array([fwd_ * math.cos(yaw), fwd_ * math.sin(yaw), up_])
    f = np.array([math.cos(yaw) * math.cos(p_), math.sin(yaw) * math.cos(p_), math.sin(p_)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u2 = np.cross(r, f)
    Mc = np.eye(4); Mc[0, :3], Mc[1, :3], Mc[2, :3], Mc[3, :3] = r, u2, -f, eye
    P1 = points(dep, "spotp", stride=2)
    P2 = world_to_decision(points_world(dep, Mc, FX, IMG / 2, stride=2), Mc, "spotp")
    assert P1.shape == P2.shape and np.abs(P1 - P2).max() < 1e-3, np.abs(P1 - P2).max()
    print(f"[cmap] selftest OK: cam_h {info['cam_h']:.3f}, slab underside {under:.3f} m at x 2.0-2.5 (true 0.65)")


def pytest_approx(v, tol):
    class _A:
        def __eq__(self, o):
            return abs(o - v) <= tol
    return _A()


if __name__ == "__main__":
    selftest()
