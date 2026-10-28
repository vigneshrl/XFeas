#!/usr/bin/env python
"""
Dataset for XFeas-Gen2 (gen2 agent, 2026-10-14). Same manifest rows as gen1 (build_manifest.py) and everything
dataset_gen.GenDataset returns, plus

  rgb    (K, 3, 224, 224) uint8   frame 0 = decision frame; 1.. = older frames (gen_v2 history), zero where absent
  depth  (K, 1, 224, 224)         metres
  fmask  (K,) bool, fage (K,) s   frame present / age before the decision (frame 0: True, 0)
  cmap   (C_MAP, 32, 16)          clearance map (cmap.py) of the decision depth (+ older depths when their poses exist)
  meta   (N_META,)                no_hist, n_frames / K, map_used_prior, cam_h / 1.5, map_present, hist_span / 2 s
  ap     (32,)                    act_v2 (GEN_V2_SPEC 3.2) normalised per robot (x mask) + its mask; see act_v2_of
  y_slip, y_slippery              per-rollout slip (gen_v2) / low-friction surface labels (see slip_labels) -- NaN = masked
Single-frame rows (gen_v1, Phase-1, old sets) -> K_eff = 1 (frames 1.. absent, no_hist = 1). History rows can be
dropped to single-frame at train time (--hist-drop p) so the model never depends on history being present.

gen_v2 (analysis/GEN_V2_SPEC.md v0.2): history from obs/<sid>.npz (frames_of), act_v2 / act_v2_mask, slipped, mu_patch;
absent keys -> single-frame / gen_v1 behaviour.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "..", "closed_loop"))
import sp_tokens as SP
import dataset_xe as DX
import dataset_gen as DG
import cmap as CM
import model_gen2 as MG2

T_HIST = DX.T_HIST


def _resize(x, n, mode):
    import cv2
    return cv2.resize(np.asarray(x), (n, n), interpolation=cv2.INTER_AREA if mode == "area" else cv2.INTER_NEAREST)
SLIPPERY_TILES = ("ice", "oil_spill", "wet_tile")


MU_SLIPPERY = 0.35


def slip_labels(z, f, source):
    """-> (y_slip, y_slippery).
    y_slip      per ROLLOUT: gen_v2 `slipped` (GEN_V2_SPEC 4.2: stance-foot slip_mean > 0.12 or p90 > 0.35 m/s); NaN when
                not measured (gen_v1, old, wheeled / aerial).
    y_slippery  the SURFACE is low-friction: gen_v2 mu_patch < 0.35 when known; gen terrain rows: the Isaac Lab tile's own
                'slippery' flag (ice / oil_spill / wet_tile); other generator rows run on the environments' own floors with
                default friction -> 0; old sets -> NaN (some school hazards had oil patches)."""
    y_slip = float("nan")
    if "slipped" in f and "slip_mean" in f and np.isfinite(float(z["slip_mean"])):
        y_slip = float(bool(z["slipped"]))
    if not source.startswith("gen"):
        return y_slip, float("nan")
    if "mu_patch" in f and np.isfinite(float(z["mu_patch"])):
        return y_slip, float(float(z["mu_patch"]) < MU_SLIPPERY)
    try:
        lane = json.loads(str(z["lane"])) if "lane" in f else {}
    except Exception:
        lane = {}
    P = lane.get("params", {}) or {}
    if "slippery" in P:
        return y_slip, float(bool(P["slippery"]))
    if str(P.get("type", "")) in SLIPPERY_TILES:
        return y_slip, 1.0
    return y_slip, 0.0


_EGAP = {}


def _egap(path):
    """gen_v1 sidestep rollouts: the gap centre e_gap is in the sits jsonl (rollouts[].sp.e_gap), not in the npz"""
    D = os.path.dirname(os.path.dirname(path))
    if D not in _EGAP:
        m = {}
        for fn in sorted(os.listdir(D)):
            if fn.startswith("sits_") and fn.endswith(".jsonl"):
                for l in open(os.path.join(D, fn)):
                    if "sidestep" not in l:
                        continue
                    d = json.loads(l)
                    ro = d["rollouts"] if isinstance(d["rollouts"], list) else eval(d["rollouts"])   # noqa: S307
                    for r in ro:
                        if isinstance(r.get("sp"), dict) and "e_gap" in r["sp"]:
                            m[r["ep"]] = float(r["sp"]["e_gap"])
        _EGAP[D] = m
    return _EGAP[D].get(os.path.basename(path), 0.0)


def act_v2_of(z, f, source, b, kind, path=""):
    """-> (act_v2 (16,) physical, mask (16,)) in the GEN_V2_SPEC section 3.2 layout"""
    A = np.zeros(16, np.float32); M = np.zeros(16, np.float32)
    if "act_v2" in f and "act_v2_mask" in f:
        return np.asarray(z["act_v2"], np.float32), np.asarray(z["act_v2_mask"], np.float32)
    if not source.startswith("gen"):
        return A, M                                         # old sets: absent
    sk = int(z["skill"])
    if sk == 4:                                             # gen_v1 sidestep (gensim_v1.SidestepCtl)
        try:
            P = json.loads(str(z["action_params"]))
        except Exception:
            P = {}
        side, vy = float(P.get("side", 0.0)), float(P.get("vy", 0.0))
        A[0] = b["nominal"]["v"]; A[8] = 0.6 if kind.startswith("g1") else 1.1
        A[10] = _egap(path) if path else 0.0; A[11] = -side * math.pi / 2; A[12] = side * vy
        M[[0, 8, 10, 11, 12]] = 1.0
        return A, M
    ch = np.asarray(z["action"], np.float32)
    if ch.shape[-1] < 7:
        return A, M
    v, dh, p, r = float(ch[0, 0]), float(ch[0, 4]), float(ch[0, 5]), float(ch[0, 6])
    A[0:4] = [v, dh, p, r]; A[4:8] = [v, dh, p, r]
    M[0:4] = 1.0
    return A, M


def norm_act_v2(A, b):
    """physical act_v2 -> normalised (GEN_V2_SPEC 3.2 units: posture piecewise-linear on the robot's box, speeds /
    max_cmd, lat_off / 1 m, distances / 3 m; head_off / (pi/2))"""
    y = np.zeros(16, np.float32)
    for i in (0, 4):
        y[i] = A[i] / max(b["vx_fwd"], 1e-6)
    for i, k in ((1, "body_height"), (5, "body_height"), (2, "pitch"), (6, "pitch"), (3, "roll"), (7, "roll")):
        if b[k] is None:
            continue
        lo, hi = b[k]; nom = b["nominal"][k]
        x = A[i] + (nom if k == "body_height" else 0.0)
        y[i] = float(SP._pw(np.float32(x), lo, nom, hi))
    for i in (8, 9, 13, 14, 15):
        y[i] = A[i] / 3.0
    y[10] = A[10] / 1.0
    y[11] = A[11] / (math.pi / 2)
    y[12] = A[12] / max(b["vy"], 1e-6) if b["vy"] > 1e-6 else 0.0
    return np.clip(y, -3, 3).astype(np.float32)


def ap_vector(z, f, source, b, kind, path=""):
    A, M = act_v2_of(z, f, source, b, kind, path)
    return np.concatenate([norm_act_v2(A, b) * M, M]).astype(np.float32)


_OBS = {}


def frames_of(z, f, path=""):
    """-> (frames newest first [(rgb, depth, age_s)], world_frames or None). gen_v1 / old: the decision frame only.
    gen_v2: obs/<sid>.npz (GEN_V2_SPEC section 5): rgb_hist / depth_hist at t = -1.5, -1.0, -0.5, 0 s (128 px) with
    camera->world matrices cam_hist; the decision frame itself stays the 224 px rgb / depth of the rollout npz."""
    out = [(z["rgb"], z["depth"], 0.0)]
    if "obs_file" not in f:
        return out, None
    ap_ = os.path.abspath(path)                         # manifest paths may be relative to Scripts/ (gen2 fix 2026-10-15)
    key = os.sep + "data" + os.sep + "gen_v2" + os.sep
    if key not in ap_:
        raise FileNotFoundError(f"{path}: obs_file present but the path is not under data/gen_v2")
    op = os.path.join(ap_[: ap_.index(key) + len(key)], str(z["obs_file"]))
    if not os.path.exists(op):
        raise FileNotFoundError(f"{op}: obs file missing (would silently drop the history)")
    if op not in _OBS:
        if len(_OBS) > 256:
            _OBS.clear()
        o = np.load(op, allow_pickle=False)
        _OBS[op] = {k: o[k] for k in ("rgb_hist", "depth_hist", "t_hist", "cam_hist", "K128")}
    o = _OBS[op]
    t = np.asarray(o["t_hist"], np.float32)
    order = np.argsort(-t)                                  # newest first
    f128, c128 = float(o["K128"][0, 0]), float(o["K128"][0, 2])
    M_dec = o["cam_hist"][order[0]]
    wf = []
    for i in order[1:]:
        dp = np.nan_to_num(np.asarray(o["depth_hist"][i], np.float32), posinf=20.0)
        out.append((o["rgb_hist"][i], dp, float(-t[i])))
        wf.append((dp, o["cam_hist"][i], f128, c128))
    return out, (M_dec, wf)


try:
    import torch
    from torch.utils.data import Dataset

    class GenDataset2(Dataset):
        def __init__(self, rows, norm, prop_norm="global", augment=False, derive_old=True, k_hist=MG2.K_HIST,
                     hist_drop=0.0, seed=0, use_map=True):
            self.rows, self.norm, self.mode, self.augment, self.derive = rows, norm, prop_norm, augment, derive_old
            self.k_hist, self.hist_drop, self.use_map = k_hist, hist_drop, use_map
            self._cache = {}
            self._color = DX.EpisodeDataset([], augment=True)._color_aug if augment else None
            self._rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.rows)

        def _desc(self, desc_s, box_s):
            k = (desc_s, box_s)
            if k not in self._cache:
                d = json.loads(desc_s)
                js, body = DX.featurize_descriptor(d)
                b = SP.robot_box(d, box_s or None)
                self._cache[k] = (js, body, b, SP.sp_tokens(b), float(d.get("fall_tilt", 0.8)), str(d.get("kind", "?")))
            return self._cache[k]

        def _pn(self, robot):
            P = self.norm["prop"]
            return P.get(robot, P["global"]) if self.mode == "robot" else (P["global"] if self.mode == "global" else None)

        def __getitem__(self, i):
            r = self.rows[i]
            z = np.load(r["path"], allow_pickle=False)
            f = set(z.files)
            js, body, b, tok, fall_tilt, kind = self._desc(str(z["desc"]), str(z["box"]) if "box" in f else "")
            pb, pj = DG._pad_hist(z["prop_base"], z["prop_joints"], T_HIST)
            pb = pb.astype(np.float32); pj = pj.astype(np.float32)
            pn = self._pn(r["robot"])
            if pn is not None:
                pb = np.clip((pb - np.asarray(pn["base_mean"], np.float32)) / np.asarray(pn["base_std"], np.float32), -10, 10)
                pj = np.clip((pj - np.asarray(pn["joint_mean"], np.float32)) / np.asarray(pn["joint_std"], np.float32), -10, 10)
            a_phys = SP.chunk_to_k(z["action"])
            a_n = SP.normalise(a_phys, b)
            # ---- frames (newest first), history drop augmentation
            fr, wf = frames_of(z, f, r["path"])
            fr = fr[: self.k_hist]
            if len(fr) > 1 and self.augment and self._rng.random() < self.hist_drop:
                fr, wf = fr[:1], None
            if wf is not None:
                wf = (wf[0], wf[1][: len(fr) - 1])
            K = len(fr)
            rgb = np.zeros((K, 224, 224, 3), np.uint8); dep = np.zeros((K, 224, 224), np.float32)
            for k, (im, dp, _) in enumerate(fr):
                if im.shape[0] != 224:                     # gen_v2 history frames are 128 px: resize (the encoder
                    im = _resize(im, 224, "area"); dp = _resize(dp, 224, "nearest")   # interpolates to 224 anyway)
                rgb[k] = im[..., :3]; dep[k] = np.nan_to_num(dp, posinf=60.0)
            if self._color is not None:
                rgb = np.stack([self._color(x) for x in rgb]) if K == 1 else self._color_seq(rgb)
            # ---- clearance map (decision depth + gen_v2 history depths fused with their exact camera matrices)
            if self.use_map:
                M, info = CM.build_map(dep[0], kind, world_frames=wf)
            else:
                M, info = np.zeros((CM.C_MAP, CM.NX, CM.NY), np.float32), dict(cam_h=0.0, used_prior=1.0)
            span = max([a for (_, _, a) in fr] + [0.0])
            meta = np.array([float(K == 1), K / self.k_hist, info["used_prior"], info["cam_h"] / 1.5, float(self.use_map),
                             span / 2.0, 0.0, 0.0], np.float32)
            nan = float("nan")
            ysl = slip_labels(z, f, r["source"])
            # ---- cost labels (identical to dataset_gen)
            t_norm = nan
            if "t_goal" in f and np.isfinite(float(z["t_goal"])) and bool(z["success"]):
                t_ref = float(z["t_ref"]) if "t_ref" in f else SP.t_ref_from_lane(json.loads(str(z["lane"])) if "lane" in f else {}, b)
                if np.isfinite(t_ref) and t_ref > 0:
                    t_norm = float(np.clip(float(z["t_goal"]) / t_ref, 0.0, DG.T_NORM_MAX))
            energy = nan
            if "energy_per_m" in f and np.isfinite(float(z["energy_per_m"])):
                E = self.norm["energy"]; mu, sd = (E.get(r["robot"]) or E["global"])[:2]
                energy = float(np.clip((math.log1p(max(float(z["energy_per_m"]), 0.0)) - mu) / sd, -6, 6))
            if "posture_dev" in f and np.isfinite(float(z["posture_dev"])):
                pdev = float(z["posture_dev"])
            else:
                pdev = SP.posture_dev_from_action(a_phys, b) if self.derive else nan
            if "tilt_margin" in f and np.isfinite(float(z["tilt_margin"])):
                tmar = float(z["tilt_margin"])
            else:
                tmar = float(1.0 - float(z["max_tilt"]) / max(fall_tilt, 1e-3)) if self.derive else nan
            tmar = float(np.clip(tmar, -1.0, 1.0)) if np.isfinite(tmar) else nan
            return dict(
                rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))),
                depth=torch.from_numpy(dep[:, None]), fage=torch.tensor([a for (_, _, a) in fr], dtype=torch.float32),
                cmap=torch.from_numpy(M), meta=torch.from_numpy(meta), ap=torch.from_numpy(ap_vector(z, f, r["source"], b, kind, r["path"])),
                base=torch.from_numpy(pb), joints=torch.from_numpy(pj),
                jstatic=torch.from_numpy(js), body=torch.from_numpy(body), sp_tok=torch.from_numpy(tok),
                action=torch.from_numpy(a_n), action_phys=torch.from_numpy(a_phys),
                skill=int(z["skill"]),
                y_success=float(z["success"]), y_fall=float(z["fell"]), y_collision=float(z["collision"]),
                y_stall=float(z["stall"]), y_slip=ysl[0], y_slippery=ysl[1],
                y_progress=float(z["progress"]), y_max_tilt=float(z["max_tilt"]),
                y_t_norm=t_norm, y_energy=energy, y_posture_dev=pdev, y_tilt_margin=tmar,
                idx=i, region=r["region"], robot=r["robot"], hazard=r["hazard"], source=r["source"])

        def _color_seq(self, rgb):
            """same colour jitter for every frame of one sample (history must stay consistent)"""
            st = self._color.__self__._rng.bit_generator.state
            out = []
            for x in rgb:
                self._color.__self__._rng.bit_generator.state = st
                out.append(self._color(x))
            return np.stack(out)

    def collate2(batch):
        K = max(b["rgb"].shape[0] for b in batch)
        J = max(b["joints"].shape[1] for b in batch)
        B = len(batch)
        out = {}
        rgb = torch.zeros(B, K, 3, 224, 224, dtype=torch.uint8); dep = torch.zeros(B, K, 1, 224, 224)
        fmask = torch.zeros(B, K, dtype=torch.bool); fage = torch.zeros(B, K)
        for i, b in enumerate(batch):
            k = b["rgb"].shape[0]
            rgb[i, :k] = b["rgb"]; dep[i, :k] = b["depth"]; fmask[i, :k] = True; fage[i, :k] = b["fage"]
        out.update(rgb=rgb, depth=dep, fmask=fmask, fage=fage)
        for k in ("cmap", "meta", "ap", "base", "body", "sp_tok", "action", "action_phys"):
            out[k] = torch.stack([b[k] for b in batch])
        T = batch[0]["joints"].shape[0]
        joints = torch.zeros(B, T, J, batch[0]["joints"].shape[2])
        jstatic = torch.zeros(B, J, DX.N_JOINT_STATIC)
        jmask = torch.zeros(B, J, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["joints"].shape[1]
            joints[i, :, :n] = b["joints"]; jstatic[i, :n] = b["jstatic"]; jmask[i, :n] = True
        out.update(joints=joints, jstatic=jstatic, jmask=jmask, skill=torch.tensor([b["skill"] for b in batch]),
                   idx=torch.tensor([b["idx"] for b in batch]))
        for k in batch[0]:
            if k.startswith("y_"):
                out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
        for k in ("region", "robot", "hazard", "source"):
            out[k] = [b[k] for b in batch]
        return out
except ImportError:
    pass
