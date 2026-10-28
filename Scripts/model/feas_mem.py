"""
gen3 perception fix (2026-10-18; additive, flag-gated -- feas.py v0.3 behaviour is unchanged by default).
Pure numpy.

  * single_frame_feas(dep224, fp, caps)       the closed-loop arm's CURRENT path (gen3_methods._feas, v0.3 rules,
                                              decision frame only), reproduced for diagnostics
  * FrameMemory                               ego-motion-compensated keyframe memory of head-camera depth frames with
                                              their camera->world matrices (cam_hist convention, USD row vectors)
  * mem_points(frames, T_dec, floor_z)        fuse frames into decision-frame points (x fwd, y left, h above floor);
                                              per frame the robot's OWN body region is dropped in THAT frame's base
                                              frame (non-floor points < SELF_X ahead of the capturing base: the arms
                                              reach ~0.45 m forward -- the reason for v0.3's x >= 0.6 window)
  * feas_from_points(P, fp, caps, rules)      clearance grid + feas vector (rules "v04")

Rules v0.4 (feas.geometry(..., rules="v04")), every constant from geometry already in v0.3 / the robot body:
  R1  overhead window x >= X_OVER_V04 (0.0 m, the base) instead of 0.6 m: the self-body points are removed per frame,
      so an overhang seen earlier is still represented when it is above / just ahead of the robot (memory).
  R2  an "overhead" whose free height is below the top of the lower-body band (0.5 m, v0.3 h_low[1]) is NOT an
      overhang: it is a step / terrain fold / rail (the ground-cluster split of a sloped or sparsely sampled surface).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import os
import sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (THIS, os.path.join(THIS, "..", "data_collector"), os.path.join(THIS, "..", "..", "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import feas as FE
import clearmap as CLM

SELF_X = 0.6          # v0.3's self-body window (arms ~0.45 m ahead), now applied per frame in that frame's base frame
SELF_H = 0.15         # floor / low points (below the wall band) near the feet are kept
CAM_FWD = 0.1         # G1 head-camera mount ahead of the base (spec.cam fwd of g1s / g1c / g1l / g1u)


def single_frame_feas(dep, fp, caps, rules="v03"):
    """gen3_methods._feas exactly (decision depth only, gravity-aligned g1 camera, floor estimated from the depth)"""
    import cmap as CMAP
    P = CMAP.points(dep, "g1", stride=1)
    zf, _ = CMAP.floor_level(P, "g1")
    Pb = np.stack([P[:, 0], P[:, 1], P[:, 2] - zf], 1)
    cm = CLM.grid(Pb, 0.0)
    fv, fd = FE.feas_vector(cm, fp, caps, P=Pb, rules=rules)
    return fv, fd, Pb, cm, -zf


def base_of_cam(M, fwd=CAM_FWD):
    M = np.asarray(M, np.float64)
    eye = M[3, :3]; look = -M[2, :3]
    yaw = math.atan2(look[1], look[0])
    return np.array([eye[0] - fwd * math.cos(yaw), eye[1] - fwd * math.sin(yaw)]), yaw, eye


def frame_world_points(depth, M, floor_z, self_x=SELF_X, self_h=SELF_H, fwd=CAM_FWD, base=None):
    """one frame -> world points, robot self pixels (clearmap.self_mask) and its own body region removed.
    base (x, y, yaw) of the capturing robot root (additive 2026-10-19, XFEAS_CAM_MOUNT=head: a head-mounted camera is not
    at root + fwd along the look direction); None = derived from M as before"""
    d = np.array(depth, np.float32)
    d[CLM.self_mask(d)] = 0.0
    W = CLM.backproject(d, np.asarray(M, np.float64))
    if self_x > 0 and len(W):
        if base is not None:
            b, yaw = np.asarray(base[:2], np.float64), float(base[2])
        else:
            b, yaw, _ = base_of_cam(M, fwd)
        Pk = CLM.to_base(W, (b[0], b[1], yaw))
        keep = ~((Pk[:, 0] < self_x) & (Pk[:, 0] > -0.5) & (np.abs(Pk[:, 1]) < 0.5) & (W[:, 2] - floor_z > self_h))
        W = W[keep]
    return W


def mem_points(frames, T_dec, floor_z, self_x=SELF_X):
    """frames: [(depth (N,N), M cam->world)], T_dec (x, y, yaw), floor_z world -> decision-frame (x, y, h)"""
    W = [frame_world_points(f[0], f[1], floor_z, self_x, base=(f[2] if len(f) > 2 else None)) for f in frames]
    W = np.concatenate(W, 0) if W else np.zeros((0, 3))
    P = CLM.to_base(W, T_dec)
    P[:, 2] -= float(floor_z)
    return P


def feas_from_points(P, fp, caps, rules="v04"):
    cm = CLM.grid(P, 0.0)
    fv, fd = FE.feas_vector(cm, fp, caps, P=P, rules=rules)
    return fv, fd, cm


class FrameMemory:
    """keyframes: the newest frame always + one keyframe per >= KEY_MOVE m of base travel or >= KEY_TURN rad of yaw;
    at most N_KEY keyframes (~N_KEY * KEY_MOVE m of approach) and MAX_AGE s. Poses = camera->world matrices (in the
    simulator from the robot's odometry, the same source as cam_hist in collection)."""
    KEY_MOVE, KEY_TURN, N_KEY, MAX_AGE = 0.25, 0.2, 16, 60.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.keys, self.newest = [], None

    def add(self, t, depth, M, base=None):
        """base (x, y, yaw) of the robot root at the frame (additive, XFEAS_CAM_MOUNT=head); None = from M (default)"""
        if base is not None:
            b, yaw = np.asarray(base[:2], np.float64), float(base[2])
        else:
            b, yaw, _ = base_of_cam(M)
        fr = (float(t), np.asarray(depth, np.float32), np.asarray(M, np.float64), b, yaw, base is not None)
        if not self.keys or np.hypot(*(b - self.keys[-1][3])) >= self.KEY_MOVE or abs(math.remainder(yaw - self.keys[-1][4], 2 * math.pi)) >= self.KEY_TURN:
            self.keys.append(fr)
            self.keys = self.keys[-self.N_KEY:]
            self.newest = None
        else:
            self.newest = fr
        self.keys = [k for k in self.keys if t - k[0] <= self.MAX_AGE]

    def frames(self):
        f = lambda k: (k[1], k[2]) if not k[5] else (k[1], k[2], (float(k[3][0]), float(k[3][1]), float(k[4])))
        out = [f(k) for k in self.keys]
        if self.newest is not None:
            out.append(f(self.newest))
        return out

    def n(self):
        return len(self.keys) + (self.newest is not None)
