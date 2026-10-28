#!/usr/bin/env python
"""
gen5b SIDE-BY-SIDE loader (2026-10-27; style of scripts/gen5/dataset5g4.py). analysis/GEN5B_DATA_PLAN.md.

One ITEM = one branch MOMENT of one gen5b pair (situation x controller):
  context    dataset_gen4.build_context (UNCHANGED) on a gen4-format obs dict assembled from the moment snapshot (obs_at5b:
             current head frame 224, history frames -1.5 / -1 / -0.5 s, camera matrices, base poses, 2 s proprio, h(t) packed
             with virtual.pack_records, feas v0.4) -- the same recipe as dataset5.obs_at
  menu       X (N x 35) = describe5 descriptors of the whole menu given the command in force at the moment (logged)
  candidates the K branches tried from this moment, side by side: cand (K x 35) = X[b_cand], cand_idx (menu indices), real
             labels per branch (labels5b: success / fell / fell_tilt / stall / contact / t_goal / progress + 1.5 s window)
  state      decision_state (time since the last switch / 5 s, any switch yet), cur_cmd (5)
    PY=/p/cral/vignesh/envs/gsam2/bin/python
    $PY scripts/gen5b/dataset5b.py build --root data/gen_v5b --out data/gen_train/gen5b/manifest.jsonl [--smoke]
    $PY scripts/gen5b/dataset5b.py check --manifest ... --n 8      # items build; descriptors recomputed == logged
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
G5D = os.path.join(ROOT, "scripts", "gen5")
sys.path[:0] = [THIS, G5D]
import dataset5 as D5                       # sets the gen5 paths, canonical joints, robot_desc5
import ctrl5 as K5
import describe5 as DS
import candidates5 as CA
import labels5b as LB
import virtual as VC
import dataset_gen4 as DG

T_SWITCH_CAP = D5.T_SWITCH_CAP
Y_KEYS = LB.LAB_KEYS


def build(root, smoke=False):
    root = os.path.abspath(root)
    rows, drop = [], defaultdict(int)
    done = None
    if not smoke and os.path.exists(os.path.join(root, "queue", "state.json")):
        st = json.load(open(os.path.join(root, "queue", "state.json")))
        done = {k for k, v in st.items() if v.get("status") == "done"}
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "sits_*.jsonl"))):
        part = os.path.basename(f)[5:-6]
        if done is not None and part not in done:
            drop["part_not_done"] += 1; continue
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if K5.is_held_out(r["ctrl_id"]):
                raise SystemExit(f"held-out controller {r['ctrl_id']}")
            if any(h in r["env"].lower() for h in D5.HELD_ENV):
                raise SystemExit(f"held-out env {r['env']}")
            fn = os.path.join(root, r["br_file"])
            if not os.path.exists(fn):
                drop["missing"] += 1; continue
            z = np.load(fn, allow_pickle=False)
            bm = np.asarray(z["b_moment"])
            for i in range(len(z["m_label"])):
                ks = np.where(bm == i)[0]
                if len(ks) < 2:
                    drop["lt2_branches"] += 1; continue
                rows.append(dict(key=f"{r['sid']}#{i}", br=fn, moment=i, sid=r["sid"], ctrl_id=r["ctrl_id"], base=r["base"], robot=r["robot"],
                                 env=r["env"], family5=r["family5"], split=r.get("zone_split", "train"), light=r["light"]["mode"],
                                 group=str(z["m_group"][i]), label=str(z["m_label"][i]), spec=K5.spec_path(r["ctrl_id"]),
                                 probe=os.path.join(ROOT, r["probe_file"]), trans=os.path.join(ROOT, r["trans_file"]),
                                 branches=[int(k) for k in ks], cls=[str(z["b_cls"][k]) for k in ks],
                                 y_success=[float(z["y_success"][k]) for k in ks]))
    return rows, dict(drop)


def obs_at5b(z, i):
    """gen4-format obs dict of moment i (same keys / recipe as dataset5.obs_at)"""
    import cv2
    import clearmap as CMg
    o = dict(rgb=z["rgb"][i][-1], depth=np.asarray(z["depth"][i], np.float32), cam_dec=np.asarray(z["cam"][i][-1], np.float64))
    o["t_hist"] = np.asarray(z["t_hist"][i], np.float32)
    o["rgb_hist"] = np.stack([cv2.resize(np.asarray(x), (128, 128), interpolation=cv2.INTER_AREA) for x in z["rgb"][i]]).astype(np.uint8)
    o["depth_hist"] = np.asarray(z["depth128"][i]).astype(np.float16)
    o["cam_hist"] = np.asarray(z["cam"][i], np.float64)
    o["base_hist_world"] = np.asarray(z["base_world"][i], np.float64)
    o["cam_h_hist"] = np.asarray(z["cam_h"][i], np.float32)
    o["selfmask_hist"] = np.asarray(z["selfmask"][i], np.float32)
    o["K128"] = CMg.intrinsics(128); o["K224"] = CMg.intrinsics(224)
    pn = int(z["prop_n"][i]); nP = z["prop_base"].shape[1]
    o["prop_base"] = np.asarray(z["prop_base"][i][nP - pn:], np.float32)
    o["prop_joints"] = np.asarray(z["prop_joints"][i][nP - pn:], np.float32)
    rn = int(z["rec_n"][i])
    R = [(float(z["rec_t"][i][k]), z["rec_cmd"][i][k], z["rec_env"][i][k], z["rec_ach"][i][k], z["rec_q"][i][k], z["rec_dq"][i][k]) for k in range(rn)]
    geo = dict(P0=np.asarray(z["P0"]), d=np.asarray(z["d"]), n=np.asarray(z["n"]), yaw=float(z["lane_yaw"]))
    H = VC.pack_records(R, geo, float(z["floor_z"]), float(z["m_clock"][i]), VC.HIST_N)
    for kk, v in H.items():
        o[f"hist_{kk}"] = v
    o["joint_names"] = z["joint_names"]; o["ach_keys"] = z["ach_keys"]; o["hist_roles"] = z["hist_roles"]
    o["feas"] = np.asarray(z["feas"][i], np.float32)
    o["cam_source"] = z["cam_source"]; o["cam_mount"] = z["cam_mount"]
    return o


def decision_state(z, i):
    dt = float(z["m_t_since"][i])
    return np.array([min(max(dt, 0.0), T_SWITCH_CAP) / T_SWITCH_CAP, float(bool(z["m_any_switch"][i]))], np.float32)


def item_np(r, z, norm=None, cfg=None):
    """numpy item (no torch): context + menu + side-by-side candidates with labels"""
    i = r["moment"]; ks = r["branches"]
    o = obs_at5b(z, i)
    spec = json.load(open(r["spec"])); probe = json.load(open(r["probe"]))
    desc = D5.robot_desc5(r["robot"])
    if not norm:                              # neutral normalisation (check only; training passes the gen5 norm)
        import gen4_common as G4C
        norm = dict(base_mean=[0.0] * 13, base_std=[1.0] * 13, joint_mean=[0.0] * 7, joint_std=[1.0] * 7,
                    fp_mean=[0.0] * G4C.N_FP, fp_std=[1.0] * G4C.N_FP)
    ctx, feas_raw = DG.build_context(o, spec, probe, desc, norm, dict(cfg or {}), train=False, rng=np.random.default_rng(0),
                                     aug=DG.DEFAULT_AUG, ptrace_cache={})
    X = np.asarray(z["X"][i], np.float32).copy()
    X[:, DS.KI["is_keep"]] = 0.0; X[0, DS.KI["is_keep"]] = 1.0
    X[:, DS.KI["is_nominal"]] = 0.0; X[1, DS.KI["is_nominal"]] = 1.0
    ci = np.asarray(z["b_cand"])[ks].astype(np.int64)
    out = dict(ctx=ctx, menu=X, cand=X[ci], cand_idx=ci, cls=[str(z["b_cls"][k]) for k in ks], keep=np.asarray(z["b_keep"])[ks],
               state=decision_state(z, i), cur_cmd=np.nan_to_num(np.asarray(z["m_cmd_prev"][i], np.float32), nan=0.0), feas_raw=feas_raw)
    for k in Y_KEYS:
        out[f"y_{k}"] = np.asarray(z[f"y_{k}"])[ks].astype(np.float32)
    return out


try:
    import torch
    from torch.utils.data import Dataset

    class BranchDataset(Dataset):
        def __init__(self, rows, norm, cfg=None):
            self.rows, self.norm, self.cfg = rows, norm, dict(cfg or {})
            self.cache = D5.EpCache()

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, j):
            r = self.rows[j]
            z = self.cache.get(r["br"])
            it = item_np(r, z, self.norm, self.cfg)
            t = torch.from_numpy
            out = {k: t(np.asarray(v)) for k, v in it.pop("ctx").items()}
            out.update(menu=t(it["menu"]), cand=t(it["cand"]), cand_idx=t(it["cand_idx"]), state=t(it["state"]), cur_cmd=t(it["cur_cmd"]), idx=j)
            for k in Y_KEYS:
                out[f"y_{k}"] = t(it[f"y_{k}"])
            return out
except ImportError:
    pass


def check(manifest, n):
    rows = [json.loads(l) for l in open(manifest)]
    rng = np.random.default_rng(0)
    pick = [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]
    worst = 0.0
    for r in pick:
        z = dict(np.load(r["br"], allow_pickle=False))
        it = item_np(r, z)
        fin = all(np.isfinite(np.asarray(v, np.float64)).all() for k, v in it["ctx"].items() if np.asarray(v).dtype.kind == "f")
        spec = json.load(open(r["spec"])); P = json.load(open(r["probe"])); T = json.load(open(r["trans"]))
        C = CA.enumerate_candidates(spec)
        cmd = json.loads(str(z["m_cmd_prev_json"][r["moment"]]))
        X = DS.Describer(spec, P, T).describe(C, cmd, float(z["m_vx"][r["moment"]]))
        d = float(np.max(np.abs(X.astype(np.float16).astype(np.float32) - np.asarray(z["X"][r["moment"]], np.float32))))
        worst = max(worst, d)
        print(f"[gen5b check] {r['key']:48s} {r['group']:8s} {r['label']:8s} K={len(r['branches'])} cls={it['cls']} succ={it['y_success'].tolist()} "
              f"ctx finite {fin} desc max|diff| {d:.4f}")
    print(f"[gen5b check] {len(pick)} items OK; descriptor max |recomputed - logged| = {worst:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build"); b.add_argument("--root", default=os.path.join(ROOT, "data", "gen_v5b")); b.add_argument("--out", required=True)
    b.add_argument("--smoke", action="store_true")
    c = sub.add_parser("check"); c.add_argument("--manifest", required=True); c.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        rows, drop = build(a.root, a.smoke)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[gen5b] {len(rows)} moments, {sum(len(r['branches']) for r in rows)} branches; dropped {drop}")
    elif a.cmd == "check":
        check(a.manifest, a.n)


if __name__ == "__main__":
    main()
