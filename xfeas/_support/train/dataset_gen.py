#!/usr/bin/env python
"""
Dataset for XFeas-Gen training: reads manifest rows (build_manifest.py) and returns everything dataset_xe returns plus

  sp_tok   (N_SP, F_SP)   descriptor setpoint-range tokens of the robot's box (sp_tokens.py)
  action   (K=10, 7)      chunk NORMALISED to [-1, 1] of the robot's own box; K is 10 for training AND the selector
                          (dataset_xe zero-padded to 16 while the selectors scored 10 -- not repeated here)
  y_t_norm       t_goal / t_ref (successes of generator rollouts only; NaN = masked)
  y_energy       per-robot z-score of log1p(energy_per_m) (generator rows only; NaN = masked)
  y_posture_dev  stored label (generator) or the same formula on the commanded chunk (old rows with a posture box; 0 for
                 bodies without posture dims, exactly what gensim.posture_dev returns)
  y_tilt_margin  stored label, or 1 - max_tilt / fall_tilt (the gensim definition) for old rows

Proprio is normalised with stats from the TRAINING split (compute_norm), per channel; mode 'global' (default: one set of
stats for every body, so an unseen robot is treated exactly like the training robots) or 'robot' (per-robot stats with a
global fall-back for unseen robots). The stats travel inside the checkpoint.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import sp_tokens as SP
import dataset_xe as DX

T_HIST = DX.T_HIST
T_NORM_MAX = 5.0


def read_manifest(path):
    return [json.loads(l) for l in open(path)]


def _pad_hist(pb, pj, T):
    pb, pj = pb[-T:], pj[-T:]
    if len(pb) < T:
        pad = T - len(pb)
        pb = np.concatenate([np.repeat(pb[:1], pad, 0), pb])
        pj = np.concatenate([np.repeat(pj[:1], pad, 0), pj])
    return pb, pj


def _stats(x):
    m = x.mean(0)
    s = x.std(0)
    small = s < 1e-3
    m = np.where(small, 0.0, m)
    s = np.where(small, 1.0, s)
    return m.astype(np.float32).tolist(), s.astype(np.float32).tolist()


def compute_norm(rows, n_prop=1500, n_energy=6000, seed=0, threads=8):
    """train rows -> {'prop': {'global': {...}, '<robot>': {...}}, 'energy': {'global': [mu, sd], '<robot>': [...]}}"""
    rng = np.random.default_rng(seed)
    tr = [r for r in rows if r["split"] == "train"]
    pick = [tr[i] for i in rng.choice(len(tr), min(n_prop, len(tr)), replace=False)]

    def load_prop(r):
        z = np.load(r["path"], allow_pickle=False)
        pb, pj = _pad_hist(z["prop_base"], z["prop_joints"], T_HIST)
        return r["robot"], pb[::4].astype(np.float32), pj[::4].reshape(-1, pj.shape[-1]).astype(np.float32)

    with ThreadPoolExecutor(threads) as ex:
        P = list(ex.map(load_prop, pick))
    out = dict(prop={}, energy={}, n_prop=len(P))
    for key in ["global"] + sorted({p[0] for p in P}):
        sel = [p for p in P if key == "global" or p[0] == key]
        bm, bs = _stats(np.concatenate([p[1] for p in sel]))
        jm, js = _stats(np.concatenate([p[2] for p in sel]))
        out["prop"][key] = dict(base_mean=bm, base_std=bs, joint_mean=jm, joint_std=js, n=len(sel))
    # never rescale the binary is_wheel channel (index 6) or the already range-normalised channels 3..5
    for v in out["prop"].values():
        for c in (3, 4, 5, 6):
            v["joint_mean"][c], v["joint_std"][c] = 0.0, 1.0
    cost = [r for r in tr if r.get("has_cost")]
    cost = [cost[i] for i in rng.choice(len(cost), min(n_energy, len(cost)), replace=False)] if cost else []

    def load_e(r):
        z = np.load(r["path"], allow_pickle=False)
        return r["robot"], float(z["energy_per_m"])

    with ThreadPoolExecutor(threads) as ex:
        E = [e for e in ex.map(load_e, cost) if np.isfinite(e[1])]
    if E:
        for key in ["global"] + sorted({e[0] for e in E}):
            v = np.log1p(np.maximum([e[1] for e in E if key == "global" or e[0] == key], 0.0))
            out["energy"][key] = [float(v.mean()), float(max(v.std(), 1e-3)), int(len(v))]
    else:
        out["energy"]["global"] = [0.0, 1.0, 0]
    return out


try:
    import torch
    from torch.utils.data import Dataset

    class GenDataset(Dataset):
        def __init__(self, rows, norm, prop_norm="global", augment=False, derive_old=True, hist=T_HIST):
            self.rows, self.norm, self.mode, self.augment, self.derive, self.hist = (
                rows,
                norm,
                prop_norm,
                augment,
                derive_old,
                hist,
            )
            self._cache = {}
            self._color = DX.EpisodeDataset([], augment=True)._color_aug if augment else None

        def __len__(self):
            return len(self.rows)

        def _desc(self, desc_s, box_s):
            k = (desc_s, box_s)
            if k not in self._cache:
                d = json.loads(desc_s)
                js, body = DX.featurize_descriptor(d)
                b = SP.robot_box(d, box_s or None)
                self._cache[k] = (js, body, b, SP.sp_tokens(b), float(d.get("fall_tilt", 0.8)))
            return self._cache[k]

        def _pn(self, robot):
            P = self.norm["prop"]
            return (
                P.get(robot, P["global"]) if self.mode == "robot" else (P["global"] if self.mode == "global" else None)
            )

        def __getitem__(self, i):
            r = self.rows[i]
            z = np.load(r["path"], allow_pickle=False)
            f = set(z.files)
            js, body, b, tok, fall_tilt = self._desc(str(z["desc"]), str(z["box"]) if "box" in f else "")
            pb, pj = _pad_hist(z["prop_base"], z["prop_joints"], self.hist)
            pb = pb.astype(np.float32)
            pj = pj.astype(np.float32)
            pn = self._pn(r["robot"])
            if pn is not None:
                pb = np.clip(
                    (pb - np.asarray(pn["base_mean"], np.float32)) / np.asarray(pn["base_std"], np.float32), -10, 10
                )
                pj = np.clip(
                    (pj - np.asarray(pn["joint_mean"], np.float32)) / np.asarray(pn["joint_std"], np.float32), -10, 10
                )
            a_phys = SP.chunk_to_k(z["action"])
            a_n = SP.normalise(a_phys, b)
            rgb = z["rgb"]
            if self._color is not None:
                rgb = self._color(rgb)
            nan = float("nan")
            # ---- cost labels
            t_norm = nan
            if "t_goal" in f and np.isfinite(float(z["t_goal"])) and bool(z["success"]):
                t_ref = (
                    float(z["t_ref"])
                    if "t_ref" in f
                    else SP.t_ref_from_lane(json.loads(str(z["lane"])) if "lane" in f else {}, b)
                )
                if np.isfinite(t_ref) and t_ref > 0:
                    t_norm = float(np.clip(float(z["t_goal"]) / t_ref, 0.0, T_NORM_MAX))
            energy = nan
            if "energy_per_m" in f and np.isfinite(float(z["energy_per_m"])):
                E = self.norm["energy"]
                mu, sd = (E.get(r["robot"]) or E["global"])[:2]
                energy = float(np.clip((math.log1p(max(float(z["energy_per_m"]), 0.0)) - mu) / sd, -6, 6))
            if "posture_dev" in f and np.isfinite(float(z["posture_dev"])):
                pdev = float(z["posture_dev"])
            else:
                pdev = SP.posture_dev_from_action(a_phys, b) if self.derive else nan
            if "tilt_margin" in f and np.isfinite(float(z["tilt_margin"])):
                tmar = float(z["tilt_margin"])
            else:
                tmar = float(1.0 - float(z["max_tilt"]) / max(fall_tilt, 1e-3)) if self.derive else nan
            tmar = float(np.clip(tmar, -1.0, 1.0)) if np.isfinite(tmar) else nan  # a fall can push it to -3
            return dict(
                rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                depth=torch.from_numpy(z["depth"][None].astype(np.float32)),
                base=torch.from_numpy(pb),
                joints=torch.from_numpy(pj),
                jstatic=torch.from_numpy(js),
                body=torch.from_numpy(body),
                sp_tok=torch.from_numpy(tok),
                action=torch.from_numpy(a_n),
                action_phys=torch.from_numpy(a_phys),
                skill=int(z["skill"]),
                y_success=float(z["success"]),
                y_fall=float(z["fell"]),
                y_collision=float(z["collision"]),
                y_stall=float(z["stall"]),
                y_progress=float(z["progress"]),
                y_max_tilt=float(z["max_tilt"]),
                y_t_norm=t_norm,
                y_energy=energy,
                y_posture_dev=pdev,
                y_tilt_margin=tmar,
                idx=i,
                region=r["region"],
                robot=r["robot"],
                hazard=r["hazard"],
                source=r["source"],
            )

    def collate(batch):
        J = max(b["joints"].shape[1] for b in batch)
        out = {}
        for k in ("rgb", "depth", "base", "body", "sp_tok", "action", "action_phys"):
            out[k] = torch.stack([b[k] for b in batch])
        B, T = len(batch), batch[0]["joints"].shape[0]
        joints = torch.zeros(B, T, J, batch[0]["joints"].shape[2])
        jstatic = torch.zeros(B, J, DX.N_JOINT_STATIC)
        jmask = torch.zeros(B, J, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["joints"].shape[1]
            joints[i, :, :n] = b["joints"]
            jstatic[i, :n] = b["jstatic"]
            jmask[i, :n] = True
        out.update(
            joints=joints,
            jstatic=jstatic,
            jmask=jmask,
            skill=torch.tensor([b["skill"] for b in batch]),
            idx=torch.tensor([b["idx"] for b in batch]),
        )
        for k in batch[0]:
            if k.startswith("y_"):
                out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
        for k in ("region", "robot", "hazard", "source"):
            out[k] = [b[k] for b in batch]
        return out
except ImportError:
    pass
