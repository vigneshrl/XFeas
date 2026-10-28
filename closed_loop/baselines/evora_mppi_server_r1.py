#!/usr/bin/env python
"""
[baselines_r1 copy of baselines/travblox_eval/arl4robots/v2/evora_mppi_server.py, 2026-10-21]
Planning is byte-for-byte the v2 sidecar. The ONLY addition is read-only logging for the video's right panel
(enabled when the reset message carries `logdir`): every planning step (dt 0.1 s) it stores the exact PMF grid,
obstacle map and unknown/visited masks that are handed to TDM_Numba.set_TDM_from_PMF_grid, MPPI's returned control
sequence, and MPPI_Numba.get_state_rollout() -- the released library's own visualisation call (workflow step 5 in
mppi.py, called after solve() and before shift_and_update(), exactly as the released notebooks do; it only writes the
library's state_rollout_batch_d buffer and uses no RNG, so the plan and the next plan are unchanged).
Bridge fix (documented in evora_plugin_r1.py): optional `selfmask` (robot's own pixels) is excluded from the map
back-projection; TractionNet input is unchanged.

BAND v2 (2026-10-21, user decision; EVORA_BAND=local, the default from now on): the v2 bridge measured the 0.20 / 1.80 m
ground / obstacle band from the START floor height, so terrain that rises > 0.2 m (pyramids, dunes, tall grass on a wave
relief) became walls. Now the band is measured from the LOCAL ground:
  * per cell zmin = lowest back-projected point ever observed in that cell (over all frames);
  * local ground G(cell) = min of zmin over the cell and its 8 neighbours (0.6 m x 0.6 m; covers the base of a wall face
    that falls into the next cell, but spans at most ~0.15 m of a 0.242 slope);
  * every observed point is kept in a per-cell height histogram (5 cm bins, floor - 1 m .. floor + 3 m), so obstacle
    hits are RE-EVALUATED against the current G at every planning step (no stale hits from an early, too-high ground
    estimate): obstacle cell = >= 2 points with G + 0.20 < z < G + 1.80 (same thresholds and hit count as v2);
  * ground points for the traction PMF: z <= G + 0.20 (v2: -0.30 < z - floor <= 0.20).
Everything else (inflation, unknown handling, PMF aggregation, planner) is unchanged. EVORA_BAND=start restores v2.

EVORA sidecar for the map-free episode runner (env: $EVORA_PY = gsam2 + numba-cuda).

AS RELEASED: the planner is mit-acl/mppi_numba (baselines/mppi_numba, unmodified) -- MPPI_Numba +
TDM_Numba.set_TDM_from_PMF_grid, with the configuration of the released example
planner_example_vis_gif.ipynb (use_tdm=True, i.e. the ICRA23/EVORA planner that samples full traction
maps from the PMF grid; T=10 s, dt=0.1, 1024 control rollouts x 1024 grid samples, cvar_alpha=0.8,
alpha_dyn=1.0, u_std=[2,2], lambda=1, goal_tolerance=1.0, v_post_rollout=0.01, vrange=[0,3.0],
wrange=[-pi,pi], max_speed_padding=5.0, seed=1). Obstacle / unknown penalties are the library defaults
(1e5 / 1e2). Same values for every robot.

PREVIOUSLY TRAINED CHECKPOINT: traversability = EVORA's TractionNet (12-bin linear-traction PMF) from
baselines/travblox_eval/out/evora_ckpt.pt (the released tdm_models carry no class names).

GLUE (written for this harness, documented in the README):
  * online BEV PMF map: per-pixel PMF from the head camera, back-projected with depth; points within
    0.20 m of the start floor are averaged per 0.2 m cell (EVORA's BEV cell aggregation of a PMF);
  * obstacle map: depth points 0.20-1.80 m above the floor (>= 2 hits per cell), inflated by the robot's
    half-width + 0.1 m (MPPI_Numba rolls out a point robot); no unknown map (as in the released
    examples: unobserved cells get nominal traction and no penalty);
    cells the robot has stood on are known free with nominal traction;
  * the same PMF is used for the angular TDM (TractionNet predicts only linear traction);
  * map window 30 x 30 m centred between start and goal, fixed per episode;
  * MPPI's (v, w) is returned as the body twist [v, 0, 0, w]; the robot adapter clips it.
  * numpy >= 1.24 removed np.float, which mppi_numba uses at import: aliased np.float = float.

Protocol (little-endian): uint32 n + npz{cmd: 'reset'|'step', ...} -> uint32 n + npz{...}
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import io, json, math, os, struct, sys, time
import numpy as np

np.float = float                                   # compatibility alias only (mppi.py line 32/33)
MPPI_ROOT = xfeas_paths.BASELINES + "/mppi_numba"
TBX = xfeas_paths.BASELINES + "/travblox_eval"
sys.path.insert(0, MPPI_ROOT); sys.path.insert(0, TBX)
from mppi_numba.mppi import MPPI_Numba             # noqa: E402
from mppi_numba.terrain import TDM_Numba           # noqa: E402
from mppi_numba.config import Config               # noqa: E402

_av = sys.argv; sys.argv = [sys.argv[0]]
import evora_baseline as EV                        # noqa: E402  (TractionNet, BIN_VALUES)
sys.argv = _av
import cv2, torch                                  # noqa: E402

LOG = sys.stderr
CKPT = os.environ.get("EVORA_CKPT", os.path.join(TBX, "out", "evora_ckpt.pt"))
RES, SIZE = 0.2, 30.0                              # map cell (m), window (m)
N = int(SIZE / RES)
GROUND_H, OBS_LO, OBS_HI, MAXR = 0.20, 0.20, 1.80, 6.0
BAND = os.environ.get("EVORA_BAND", "local")       # "local" (band v2, default) | "start" (v2 bridge: start-floor band)
DZ, NZ = 0.05, 80                                  # band v2 histogram: 5 cm bins, floor - 1 m .. floor + 3 m

# ---- released example configuration (planner_example_vis_gif.ipynb) ----
CFG = dict(T=10.0, dt=0.1, num_grid_samples=1024, num_control_rollouts=1024, max_speed_padding=5.0,
           num_vis_state_rollouts=100, seed=1, use_tdm=True, use_det_dynamics=False,
           use_nom_dynamics_with_speed_map=False, use_costmap=False)        # max_map_dim: Config default (250,250)
MAX_SPEED = 3.0
PARAMS = dict(goal_tolerance=1.0, v_post_rollout=0.01, cvar_alpha=0.8, alpha_dyn=1.0, lambda_weight=1.0,
              num_opt=1, u_std=np.array([2.0, 2.0]), vrange=np.array([0, MAX_SPEED]),
              wrange=np.array([-np.pi, np.pi]))


class Evora:
    def __init__(self):
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        self.net = EV.TractionNet(); self.net.load_state_dict(ck["state_dict"]); self.net = self.net.to(self.dev).eval()
        self.res_in, self.maxd = int(ck.get("res", EV.RES)), float(ck.get("maxd", EV.MAXD))
        self.cfg = Config(**CFG)
        self.mppi = MPPI_Numba(self.cfg)
        self.lin, self.ang = TDM_Numba(self.cfg), TDM_Numba(self.cfg)
        LOG.write(f"[evora-mppi] ready: {os.path.basename(CKPT)} ({ck.get('n_frames')} frames), "
                  f"mppi_numba use_tdm, {CFG['num_control_rollouts']}x{CFG['num_grid_samples']}\n"); LOG.flush()

    # ------------------------------------------------------------------ episode
    def reset(self, z):
        st, gl = z["start"].astype(float), z["goal"].astype(float)
        c = 0.5 * (st + gl)
        self.x0, self.y0 = c[0] - SIZE / 2, c[1] - SIZE / 2
        self.goal = gl; self.floor = float(z["floor"]); self.infl = float(z["inflate"])
        self.psum = np.zeros((EV.NBINS, N, N), np.float32); self.pcnt = np.zeros((N, N), np.float32)
        self.hits = np.zeros((N, N), np.float32); self.visited = np.zeros((N, N), bool)
        self.zmin = np.full((N, N), np.inf, np.float32)                       # band v2: lowest point seen per cell
        self.zh = np.zeros((NZ, N, N), np.int32); self.z0 = self.floor - 1.0  # band v2: per-cell height histogram
        from numba import cuda
        self.mppi.reset()                                # zero control sequence for the new episode
        self.mppi.u_cur_d = cuda.to_device(np.zeros((self.cfg.num_steps, 2), np.float32))
        self.useq = None
        self.logdir = str(z["logdir"]) if "logdir" in z.files else ""
        self.lbuf, self.lchunk, self.lstep = [], 0, 0
        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
            np.savez_compressed(os.path.join(self.logdir, "map_meta.npz"), origin=np.array([self.x0, self.y0]), res=RES, n=N,
                                goal=self.goal, start=st, bin_values=EV.BIN_VALUES, infl=self.infl, floor=self.floor)
        return dict(ok=1)

    # ------------------------------------------------------------------ logging (read-only, video panel)
    def _log(self, pos, yaw, pmf, obst, unk, useq, roll, v, w, solve_ms):
        etr = (pmf.astype(np.float32) / 100.0 * EV.BIN_VALUES[:, None, None]).sum(0)        # E[traction] per cell
        self.lbuf.append(dict(step=self.lstep, pos=np.asarray(pos, np.float32), yaw=np.float32(yaw), v=np.float32(v),
                              w=np.float32(w), etr=etr.astype(np.float16), obst=np.packbits(obst.astype(bool)),
                              unk=np.packbits(unk.astype(bool)), visited=np.packbits(self.visited),
                              useq=(useq if useq is not None else np.zeros((0, 2))).astype(np.float16),
                              roll=(roll[:, ::2, :2] if roll is not None else np.zeros((0, 0, 2))).astype(np.float32),
                              solve_ms=np.float32(solve_ms)))
        self.lstep += 1
        if len(self.lbuf) >= 10:                     # small chunks: a runner that never sends 'flush' loses < 1 s
            self._flush()

    def _flush(self):
        if not self.logdir or not self.lbuf:
            return
        out = {}
        for i, d in enumerate(self.lbuf):
            for k, v in d.items():
                out[f"{i}_{k}"] = v
        out["n"] = np.int32(len(self.lbuf))
        np.savez_compressed(os.path.join(self.logdir, f"chunk_{self.lchunk:04d}.npz"), **out)
        self.lchunk += 1; self.lbuf = []

    def rc(self, x, y):
        return ((np.asarray(y) - self.y0) / RES).astype(int), ((np.asarray(x) - self.x0) / RES).astype(int)

    # ------------------------------------------------------------------ perception -> map
    def integrate(self, rgb, dep, eye, f, selfmask=None):
        R = self.res_in
        rs = lambda a, i: cv2.resize(a, (R, R), interpolation=i)
        d = np.clip(rs(np.nan_to_num(dep, nan=0., posinf=0.), cv2.INTER_NEAREST), 0, self.maxd)
        x = np.concatenate([rs(rgb, cv2.INTER_AREA), (d / self.maxd * 255.0)[..., None]], -1).astype(np.uint8)
        with torch.no_grad():
            p = torch.softmax(self.net(EV.to_input(x[None]).to(self.dev)), 1)[0].permute(1, 2, 0).cpu().numpy()
        dd = rs(np.nan_to_num(dep, nan=0., posinf=60.), cv2.INTER_NEAREST)
        if selfmask is not None and selfmask.any():
            dd = dd.copy(); dd[rs(selfmask.astype(np.uint8), cv2.INTER_NEAREST) > 0] = 0.0     # robot body: not projected
        fpx = 12.0 / 20.955 * R
        r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
        uu, vv = np.meshgrid(np.arange(R) + 0.5, np.arange(R) + 0.5)
        a = (uu - R / 2) / fpx; b = (vv - R / 2) / fpx
        P = eye[None, None] + dd[..., None] * (f[None, None] + a[..., None] * r - b[..., None] * u)
        P = P.reshape(-1, 3); p = p.reshape(-1, EV.NBINS); dflat = dd.reshape(-1)
        h = P[:, 2] - self.floor
        rng = np.hypot(P[:, 0] - eye[0], P[:, 1] - eye[1])
        ok = (dflat > 0.05) & (dflat < 59.0) & (rng < MAXR)
        rr, cc = self.rc(P[:, 0], P[:, 1]); ok &= (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
        if BAND == "local":
            from scipy import ndimage
            z = P[:, 2]
            np.minimum.at(self.zmin, (rr[ok], cc[ok]), z[ok].astype(np.float32))
            G = ndimage.minimum_filter(self.zmin, size=3, mode="constant", cval=np.inf)
            hl = np.full(len(z), np.inf); hl[ok] = z[ok] - G[rr[ok], cc[ok]]
            b = np.floor((z - self.z0) / DZ).astype(int)
            okb = ok & (b >= 0) & (b < NZ)
            np.add.at(self.zh, (b[okb], rr[okb], cc[okb]), 1)
            g = ok & (hl <= GROUND_H)
            for k in range(EV.NBINS):
                np.add.at(self.psum[k], (rr[g], cc[g]), p[g, k])
            np.add.at(self.pcnt, (rr[g], cc[g]), 1.0)
            o = ok & (hl > OBS_LO) & (hl < OBS_HI)
            return int(g.sum()), int(o.sum())
        g = ok & (h > -0.30) & (h <= GROUND_H)
        for k in range(EV.NBINS):
            np.add.at(self.psum[k], (rr[g], cc[g]), p[g, k])
        np.add.at(self.pcnt, (rr[g], cc[g]), 1.0)
        o = ok & (h > OBS_LO) & (h < OBS_HI)
        np.add.at(self.hits, (rr[o], cc[o]), 1.0)
        return int(g.sum()), int(o.sum())

    def local_hits(self):
        """band v2: points per cell with G + OBS_LO < z < G + OBS_HI, from the height histogram and the CURRENT G"""
        from scipy import ndimage
        G = ndimage.minimum_filter(self.zmin, size=3, mode="constant", cval=np.inf)
        seen = np.isfinite(G)
        C = np.concatenate([np.zeros((1, N, N), np.int32), np.cumsum(self.zh, 0)], 0)      # C[i] = points in bins < i
        lo = np.clip(np.floor((np.where(seen, G, 0) + OBS_LO - self.z0) / DZ).astype(int) + 1, 0, NZ)
        hi = np.clip(np.floor((np.where(seen, G, 0) + OBS_HI - self.z0) / DZ).astype(int), 0, NZ)
        hits = np.take_along_axis(C, hi[None], 0)[0] - np.take_along_axis(C, lo[None], 0)[0]
        return np.where(seen & (hi > lo), hits, 0).astype(np.float32)

    def maps(self):
        from scipy import ndimage
        if BAND == "local":
            self.hits = self.local_hits()
        occ = self.hits >= 2
        occ &= ~self.visited
        k = int(math.ceil(self.infl / RES))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        obst = ndimage.binary_dilation(occ, structure=(np.hypot(xx, yy) * RES <= self.infl + 1e-6)) if k > 0 else occ
        known = (self.pcnt > 0) | self.visited | occ
        unknown = ~known
        pmf = np.zeros((EV.NBINS, N, N), np.float32); pmf[-1] = 1.0           # nominal traction where unobserved
        m = self.pcnt > 0
        pmf[:, m] = self.psum[:, m] / self.pcnt[m]
        pmf[:, self.visited & ~m] = 0.0; pmf[-1, self.visited & ~m] = 1.0
        q = np.floor(pmf * 100).astype(np.int32)                             # int8 PMF summing to exactly 100
        rem = 100 - q.sum(0)
        top = np.argmax(pmf - q / 100.0, axis=0)
        np.add.at(q, (top.ravel(), np.repeat(np.arange(N), N), np.tile(np.arange(N), N)), rem.ravel())
        return q.astype(np.int8), obst.astype(np.int8), unknown.astype(np.int8)

    # ------------------------------------------------------------------ plan
    def step(self, z):
        t0 = time.time()
        pos, yaw = z["pos"].astype(float), float(z["yaw"])
        if bool(z["has_img"]):
            ng, no = self.integrate(z["rgb"], z["depth"], z["eye"].astype(float), z["f"].astype(float),
                                    z["selfmask"] if "selfmask" in z.files else None)
        else:
            ng = no = 0
        r0, c0 = self.rc(pos[0], pos[1])
        if 0 <= r0 < N and 0 <= c0 < N:
            self.visited[max(0, r0 - 1):r0 + 2, max(0, c0 - 1):c0 + 2] = True
        else:
            return dict(v=0.0, w=0.0, info="outside map window", ok=0)
        pmf, obst, unk = self.maps()
        tdm = dict(res=RES, xlimits=np.array([self.x0, self.x0 + N * RES]), ylimits=np.array([self.y0, self.y0 + N * RES]),
                   bin_values=EV.BIN_VALUES, bin_values_bounds=np.array([0.0, 1.0]),
                   det_dynamics_cvar_alpha=PARAMS["alpha_dyn"])
        # unknown_map=None, as in the released examples (unobserved cells: nominal traction, no penalty)
        self.lin.set_TDM_from_PMF_grid(pmf, tdm, obstacle_map=obst, unknown_map=None)
        self.ang.set_TDM_from_PMF_grid(pmf, tdm, obstacle_map=obst, unknown_map=None)
        prm = dict(PARAMS, dt=self.cfg.dt, x0=np.array([pos[0], pos[1], yaw]), xgoal=self.goal.copy(),
                   timesteps=self.cfg.num_steps)
        self.mppi.setup(prm, self.lin, self.ang)
        useq = self.mppi.solve()
        if os.environ.get("EVORA_DUMP"):
            self.nstep = getattr(self, "nstep", 0) + 1
            if self.nstep % 10 == 1:
                os.makedirs(os.environ["EVORA_DUMP"], exist_ok=True)
                np.savez_compressed(os.path.join(os.environ["EVORA_DUMP"], f"m_{self.nstep:05d}.npz"), pmf=pmf, obst=obst,
                                    visited=self.visited, hits=self.hits, pos=pos, yaw=yaw, goal=self.goal,
                                    origin=np.array([self.x0, self.y0]), useq=useq if useq is not None else np.zeros(0))
        if useq is None:
            if getattr(self, "logdir", ""):
                self._log(pos, yaw, pmf, obst, unk, None, None, 0.0, 0.0, 1000 * (time.time() - t0))
            return dict(v=0.0, w=0.0, info="mppi solve condition not met", ok=0)
        v, w = float(useq[0, 0]), float(useq[0, 1])
        if getattr(self, "logdir", ""):
            roll = self.mppi.get_state_rollout()          # released vis call (100 rollouts of the chosen useq over sampled maps)
            self._log(pos, yaw, pmf, obst, unk, useq, roll, v, w, 1000 * (time.time() - t0))
        self.mppi.shift_and_update(prm["x0"], useq, num_shifts=1)            # warm start, as in the notebooks
        # mean traction / CVaR-style risk 1 m ahead (for logging only)
        ah = self.rc(pos[0] + math.cos(yaw), pos[1] + math.sin(yaw))
        mean_tr = float((pmf[:, ah[0], ah[1]].astype(float) / 100.0 * EV.BIN_VALUES).sum()) if (0 <= ah[0] < N and 0 <= ah[1] < N) else -1
        return dict(v=v, w=w, ok=1, n_ground=ng, n_obs=no, known=int((unk == 0).sum()), n_obst=int(obst.sum()),
                    trac_ahead=round(mean_tr, 3), solve_ms=round(1000 * (time.time() - t0), 1))


def rd(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b


def main():
    E = Evora()
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        h = rd(fin, 4)
        if h is None:
            try:
                E._flush()                               # stdin closed (plugin __del__ / runner exit): write the tail
            except Exception:
                pass
            break
        z = np.load(io.BytesIO(rd(fin, struct.unpack("<I", h)[0])), allow_pickle=False)
        cmd = str(z["cmd"])
        try:
            if cmd == "flush":
                E._flush(); out = dict(ok=1)
            else:
                out = E.reset(z) if cmd == "reset" else E.step(z)
        except Exception as e:
            import traceback; traceback.print_exc(file=LOG); LOG.flush()
            out = dict(v=0.0, w=0.0, ok=0, info=f"error: {e}")
        b = io.BytesIO(); np.savez(b, out=json.dumps(out)); b = b.getvalue()
        fout.write(struct.pack("<I", len(b))); fout.write(b); fout.flush()


if __name__ == "__main__":
    main()
