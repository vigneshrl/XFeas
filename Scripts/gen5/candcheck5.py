#!/usr/bin/env python
"""
Does the model USE the candidate? (2026-10-26, Stage A check). Tick-level AUROC can be high from the scene alone (a beam
ahead -> contact likely, whatever the command). For selection the score must change with the command. On ticks where the
chosen command differs from KEEP (row 0) in measured effect, predict each label twice:
  chosen   with the descriptor of the command that was actually sent (the real input)
  swapped  with the KEEP descriptor instead (what the model would say for 'keep the current command')
If the model reads the candidate, AUROC(chosen) > AUROC(swapped) and the predictions differ (mean |diff|). If the two are
equal, the heads only describe the situation and cannot rank commands.

    python scripts/gen5/candcheck5.py --ckpt <run>/model.pt --manifest data/gen_train/gen5/manifest.jsonl --split test
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--max-rows", type=int, default=6000)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import torch
import dataset5 as D5
import model5 as M5
import describe5 as DS


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    m = M5.build(cfg, pretrained=True).to(args.device).eval()
    m.load_state_dict(d["model"], strict=False)
    rows = D5.fold_rows([json.loads(l) for l in open(args.manifest)], cfg["holdout_base"])
    ev = [r for r in rows if r["split"] == args.split and r["chosen"] != 0]
    rng = np.random.default_rng(0)
    if len(ev) > args.max_rows:
        sids = sorted({r["sid"] for r in ev}); rng.shuffle(sids)
        keep, n = set(), 0
        cnt = defaultdict(int)
        for r in ev:
            cnt[r["sid"]] += 1
        for s in sids:
            if n >= args.max_rows:
                break
            keep.add(s); n += cnt[s]
        ev = [r for r in ev if r["sid"] in keep]
    ds = D5.TickDataset(ev, norm, cfg, augment=False)
    E = defaultdict(list)
    for i, r in enumerate(ev):
        E[r["sid"]].append(i)
    bs = [ix[s:s + 48] for ix in E.values() for s in range(0, len(ix), 48)]
    dl = torch.utils.data.DataLoader(ds, batch_sampler=bs, num_workers=args.workers, collate_fn=D5.collate5)
    P = defaultdict(list)
    eff = [DS.KI[k] for k in ("e_top", "e_vx", "e_vy", "e_pitch")]
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in b.items()}
            C = b["cand"]; ch = b["chosen"]
            Xc = C.gather(1, ch[:, None, None].expand(-1, 1, C.shape[-1])); Xk = C[:, :1]
            diff = (Xc[:, 0, eff] - Xk[:, 0, eff]).abs().max(-1).values
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(cand=torch.cat([Xc, Xk], 1), state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
            for h in ("fall", "stall", "contact"):
                p = torch.sigmoid(o[h].float()).mean(0)
                P[f"{h}_c"].append(p[:, 0].cpu().numpy()); P[f"{h}_k"].append(p[:, 1].cpu().numpy())
                P[f"y_{h}"].append(b[f"y_{h}"].cpu().numpy())
            pr = o["progress"].float().mean(0)
            P["prog_c"].append(pr[:, 0].cpu().numpy()); P["prog_k"].append(pr[:, 1].cpu().numpy()); P["y_progress"].append(b["y_progress"].cpu().numpy())
            P["eff_diff"].append(diff.float().cpu().numpy())
    A = {k: np.concatenate(v) for k, v in P.items()}
    big = A["eff_diff"] > 0.1
    res = dict(holdout=cfg["holdout_base"], split=args.split, n=int(len(big)), n_effect_differs=int(big.sum()))
    for nm, mk in (("all_switch_ticks", np.ones_like(big)), ("effect_differs", big)):
        r = {}
        for h in ("fall", "stall", "contact"):
            y = A[f"y_{h}"]; ok = mk & np.isfinite(y)
            r[h] = dict(auroc_chosen=round(auroc(y[ok], A[f"{h}_c"][ok]), 3), auroc_keep_instead=round(auroc(y[ok], A[f"{h}_k"][ok]), 3),
                        mean_abs_diff=round(float(np.abs(A[f"{h}_c"][ok] - A[f"{h}_k"][ok]).mean()), 3), rate=round(float(y[ok].mean()), 3))
        y = A["y_progress"]; ok = mk & np.isfinite(y)
        r["progress"] = dict(mae_chosen=round(float(np.abs(A["prog_c"][ok] - y[ok]).mean()), 3),
                             mae_keep_instead=round(float(np.abs(A["prog_k"][ok] - y[ok]).mean()), 3),
                             mean_abs_diff=round(float(np.abs(A["prog_c"][ok] - A["prog_k"][ok]).mean()), 3))
        res[nm] = r
    od = os.path.dirname(os.path.abspath(args.ckpt))
    json.dump(res, open(os.path.join(od, f"candcheck_{args.split}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
