#!/usr/bin/env python
"""
Stage A timing check (2026-10-26): on held-out episodes with a LOW obstacle (crawl / overhang / env_overhang), score the
WHOLE menu at every decision tick and compare, against the distance to the obstacle,
  upright  = the NOMINAL command (menu row 1: the controller's stock walk)
  crouch   = the moving candidate with the LOWEST measured body top (e_top; e_moving = 1, c_vx > 0)
on P(contact), P(stall), P(fall) and progress over the next 1.5 s. If Stage A learned anything useful, upright's P(contact)
rises as the obstacle gets close (and crouch's stays lower), and crouch's progress is lower far away (slow), i.e. the two
curves cross. No rule picks a distance anywhere; this only reads the model.

    python scripts/gen5/timing5.py --ckpt <run>/model.pt --manifest data/gen_train/gen5/manifest.jsonl --out <run>/timing
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
ap.add_argument("--families", default="crawl,overhang,env_overhang")
ap.add_argument("--max-eps", type=int, default=120)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import torch
import dataset5 as D5
import model5 as M5
import describe5 as DS

BINS = np.arange(-1.0, 4.01, 0.25)


def main():
    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    m = M5.build(cfg, pretrained=True).to(args.device).eval()
    m.load_state_dict(d["model"], strict=False)
    rows = D5.fold_rows([json.loads(l) for l in open(args.manifest)], cfg["holdout_base"])
    fams = set(args.families.split(","))
    ev = [r for r in rows if r["split"] == args.split and r["family5"] in fams]
    sids = sorted({r["sid"] for r in ev})[: args.max_eps]
    ev = [r for r in ev if r["sid"] in set(sids)]
    ds = D5.TickDataset(ev, norm, cfg, augment=False)
    R = defaultdict(list)
    with torch.no_grad():
        for i in range(len(ev)):
            b = D5.collate5([ds[i]])
            b = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in b.items()}
            X = b["cand"][0]
            mv = (X[:, DS.KI["e_moving"]] > 0.5) & (X[:, DS.KI["c_vx"]] > 0.05)
            if mv.sum() < 2:
                continue
            top = torch.where(mv, X[:, DS.KI["e_top"]], torch.full_like(X[:, 0], 9.0))
            ic = int(torch.argmin(top)); iu = 1
            if float(X[ic, DS.KI["e_top"]]) > float(X[iu, DS.KI["e_top"]]) - 0.1:
                continue                                         # this controller has no real low posture
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(cand=b["cand"][:, [iu, ic]], state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
            z = ds.cache.get(ev[i]["ep"]); k = ev[i]["tick"]
            hz = json.loads(str(z["hazards_json"]))
            arc = float(z["arc"][k])
            dist = min(h["arc0"] - arc for h in hz) if hz else np.nan
            R["dist"].append(dist); R["sid"].append(ev[i]["sid"]); R["chosen_is"].append(int(b["chosen"][0]) in (iu, ic) and int(b["chosen"][0]))
            R["top_u"].append(float(X[iu, DS.KI["e_top"]])); R["top_c"].append(float(X[ic, DS.KI["e_top"]]))
            R["hfree"].append(float(np.nan_to_num(b["feas_raw"][0, M5.FI["h_free_min"]].item(), nan=9.9)))
            for h in ("fall", "stall", "contact", "value"):
                p = torch.sigmoid(o[h][..., 0, :].float()).mean(0)        # (2,) members averaged
                R[f"{h}_u"].append(float(p[0])); R[f"{h}_c"].append(float(p[1]))
            pr = o["progress"][..., 0, :].float().mean(0)
            R["prog_u"].append(float(pr[0])); R["prog_c"].append(float(pr[1]))
    A = {k: np.asarray(v) for k, v in R.items()}
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "timing_raw.npz"), **A)

    def table(sel):
        tab = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            mk = sel & (A["dist"] >= lo) & (A["dist"] < hi)
            if mk.sum() < 5:
                continue
            tab.append(dict(d=round((lo + hi) / 2, 3), n=int(mk.sum()), **{f"{h}_{s}": round(float(A[f"{h}_{s}"][mk].mean()), 3)
                                                                         for h in ("contact", "stall", "fall", "prog", "value") for s in "uc"}))
        return tab
    # 'fits': the obstacle's free height (perceived at this tick, or seen at any tick of the episode) is above the crouch
    # top and below the upright top -- the case where crouching is the right call
    hf_ep = {}
    for s_, h_ in zip(A["sid"], A["hfree"]):
        if h_ < 9.0:
            hf_ep[s_] = min(hf_ep.get(s_, 9.9), h_)
    hfe = np.array([hf_ep.get(s_, 9.9) for s_ in A["sid"]])
    fits = (hfe < 9.0) & (A["top_c"] < hfe - 0.03) & (A["top_u"] > hfe)
    nofit = (hfe < 9.0) & (A["top_c"] >= hfe - 0.03)
    tab = table(np.ones(len(A["dist"]), bool)); tab_f = table(fits); tab_n = table(nofit)
    json.dump(dict(n_eps=len(set(A["sid"].tolist())) if len(A.get("sid", [])) else 0, n_ticks=len(A.get("dist", [])),
                   n_fits=int(fits.sum()), n_nofit=int(nofit.sum()), table=tab, table_crouch_fits=tab_f, table_crouch_too_tall=tab_n),
              open(os.path.join(args.out, "timing.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_ = [(nm_, t_) for nm_, t_ in (("all", tab), ("crouch fits under", tab_f), ("crouch too tall", tab_n)) if t_]
    if rows_:
        fig, axs = plt.subplots(len(rows_), 5, figsize=(20, 3.4 * len(rows_)), squeeze=False)
        for ax, (rn, tb) in zip(axs, rows_):
            x = [t["d"] for t in tb]
            for a, (h, nm) in zip(ax, (("contact", "P(hazard contact, 1.5 s)"), ("stall", "P(stall, 1.5 s)"), ("fall", "P(fall, 1.5 s)"),
                                       ("prog", "progress over 1.5 s (m)"), ("value", "V = P(eventual success)"))):
                a.plot(x, [t[f"{h}_u"] for t in tb], "-o", ms=3, label="upright (NOMINAL)")
                a.plot(x, [t[f"{h}_c"] for t in tb], "-o", ms=3, label="crouch (lowest moving)")
                a.axvline(0, color="k", lw=0.6); a.set_xlabel("distance to obstacle start (m)"); a.set_title(f"{rn}: {nm}", fontsize=9); a.invert_xaxis()
            ax[0].legend(fontsize=8)
        fig.suptitle(f"Stage A, held-out {cfg['holdout_base']}: {len(set(A['sid'].tolist()))} low-obstacle episodes, {len(A['dist'])} ticks")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "timing.png"), dpi=110)
    print(json.dumps(dict(n_ticks=len(A.get("dist", [])), table=tab[:3]), indent=0)[:1500])


if __name__ == "__main__":
    main()
