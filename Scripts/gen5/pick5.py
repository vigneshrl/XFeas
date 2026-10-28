#!/usr/bin/env python
"""
Offline POLICY readout (2026-10-26, Stage B check): what would the selector pick? On held-out episodes, score the whole menu
at every decision tick and take argmax U = mean_k V - KAPPA * std_k V - LAM * P(fall)  (GEN5_DESIGN 4.2). Report, against
the distance to the next hazard, the picked command's measured body top (e_top, relative to the stock walk), speed (e_vx),
and how often it is KEEP / NOMINAL; split by low obstacle where the crouch fits / is too tall, and obstacle-free episodes
(null: picking a crouch there = over-intervention).

    python scripts/gen5/pick5.py --ckpt <run>/model.pt --manifest data/gen_train/gen5/manifest.jsonl --out <run>/pick
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--kappa", type=float, default=1.0)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--mode", default="pi", choices=["pi", "q"], help="pi = the IQL chooser (argmax of the policy logit); q = argmax U")
ap.add_argument("--max-eps", type=int, default=150)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import torch
import dataset5 as D5
import model5 as M5
import describe5 as DS

LOW = {"crawl", "overhang", "env_overhang"}
BINS = [(-1.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99.0)]   # no hazard -> dist 9


def main():
    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    m = M5.build(cfg, pretrained=True).to(args.device).eval()
    m.load_state_dict(d["model"], strict=False)
    rows = D5.fold_rows([json.loads(l) for l in open(args.manifest)], cfg["holdout_base"])
    ev = [r for r in rows if r["split"] == args.split and (r["family5"] in LOW or r["family5"] == "null")]
    rng = np.random.default_rng(0)
    groups = defaultdict(list)
    for r in ev:
        groups["low" if r["family5"] in LOW else "null"].append(r["sid"])
    keep = set()
    for g, s in groups.items():
        s = sorted(set(s)); rng.shuffle(s); keep |= set(s[: args.max_eps])
    ev = [r for r in ev if r["sid"] in keep]
    E = defaultdict(list)
    for i, r in enumerate(ev):
        E[r["sid"]].append(i)
    bs = [ix[s:s + 32] for ix in E.values() for s in range(0, len(ix), 32)]
    ds = D5.TickDataset(ev, norm, cfg, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=bs, num_workers=args.workers, collate_fn=D5.collate5)
    R = defaultdict(list)
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(cand=b["cand"], state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
            V = torch.sigmoid(o["value"].float()); Pf = torch.sigmoid(o["fall"].float()).mean(0)
            U = V.mean(0) - args.kappa * V.std(0) - args.lam * Pf
            if args.mode == "pi":
                U = o["pi"].float().mean(0)
            U = torch.where(b["cand_mask"], U, torch.full_like(U, -9.0))
            a = U.argmax(1)
            X = b["cand"]
            pick = X.gather(1, a[:, None, None].expand(-1, 1, X.shape[-1]))[:, 0]
            for j, i in enumerate(b["idx"].tolist()):
                r = ev[i]
                z = ds.cache.get(r["ep"]); k = r["tick"]
                hz = json.loads(str(z["hazards_json"]))
                arc = float(z["arc"][k])
                dist = min([h["arc0"] - arc for h in hz if h["arc1"] - arc > 0] or [9.0])
                hf = float(np.nan_to_num(b["feas_raw"][j, M5.FI["h_free_min"]].item(), nan=9.9))
                R["sid"].append(r["sid"]); R["fam"].append("low" if r["family5"] in LOW else "null"); R["dist"].append(dist)
                R["hfree"].append(hf)
                top_nom = float(X[j, 1, DS.KI["e_top"]])
                mv = (X[j, :, DS.KI["e_moving"]] > 0.5) & (X[j, :, DS.KI["c_vx"]] > 0.05) & b["cand_mask"][j]
                R["top_min"].append(float(X[j, mv, DS.KI["e_top"]].min()) if mv.any() else top_nom)
                R["top_nom"].append(top_nom); R["top_pick"].append(float(pick[j, DS.KI["e_top"]]))
                R["vx_pick"].append(float(pick[j, DS.KI["e_vx"]])); R["a"].append(int(a[j]))
                R["u_gap"].append(float(U[j, a[j]] - U[j, 1]))
    A = {k: np.asarray(v) for k, v in R.items()}
    hf_ep = {}
    for s, h in zip(A["sid"], A["hfree"]):
        if h < 9.0:
            hf_ep[s] = min(hf_ep.get(s, 9.9), h)
    hfe = np.array([hf_ep.get(s, 9.9) for s in A["sid"]])
    case = np.where(A["fam"] == "null", "no obstacle",
                    np.where((hfe < 9.0) & (A["top_min"] < hfe - 0.03) & (A["top_nom"] > hfe), "crouch fits", "crouch too tall / none"))
    out = {}
    for c in ("crouch fits", "crouch too tall / none", "no obstacle"):
        tb = []
        for lo, hi in BINS:
            mk = (case == c) & (A["dist"] >= lo) & (A["dist"] < hi)
            if mk.sum() < 5:
                continue
            low = A["top_pick"][mk] < A["top_nom"][mk] - 0.1
            tb.append(dict(dist=f"{lo}-{hi}", n=int(mk.sum()), crouch_share=round(float(low.mean()), 3),
                           keep_share=round(float((A["a"][mk] == 0).mean()), 3), nominal_share=round(float((A["a"][mk] == 1).mean()), 3),
                           top_below_stock=round(float((A["top_nom"][mk] - A["top_pick"][mk]).mean()), 3),
                           vx_pick=round(float(A["vx_pick"][mk].mean()), 3)))
        out[c] = tb
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "pick_raw.npz"), case=case, **A)
    json.dump(dict(mode=args.mode, kappa=args.kappa, lam=args.lam, holdout=cfg["holdout_base"], table=out), open(os.path.join(args.out, "pick.json"), "w"), indent=1)
    for c, tb in out.items():
        print(c)
        for t in tb:
            print("  ", t)


if __name__ == "__main__":
    main()
