#!/usr/bin/env python
"""
Stage B bootstrapped VALUE targets (2026-10-26; GEN5_DESIGN 4.2 fitted value iteration). With a frozen model (the previous
round), score the WHOLE menu at every decision tick of the train / val rows, then for each row i:
  episode ends inside the 1.5 s window (term_in_h) or no next decision tick  ->  v_i = y_success (the actual outcome)
  else, with j = next_row(i) (the tick 1.5 s later in the same episode):
        v_i = 0.5 * V(j, command actually chosen at j)                          (what the data collector then did)
            + 0.5 * max_a [ mean_k V_k(j, a) - KAPPA * std_k V_k(j, a) ]       (the best command at j, PESSIMISTIC)
V = sigmoid(value logit), k = ensemble members. The 50/50 blend and the std penalty keep the max from chasing commands the
data rarely tried (over-estimation); KAPPA is the same kind of constant as the selector's, fixed here, not tuned per scene.
Held-out (test) rows are never scored or targeted.
--iql (2026-10-26, Stage I): score ONLY the command the data sent (no max over the menu) and write per row
  q_data = mean_k sigmoid(Q_k(s, a_data)), v_data = mean_k sigmoid(V_k(s)) (state value), and
  q_tgt  = y_success if the episode ends in the window (or no next tick), else v_data(next_row)   (IQL's Q target).

    python scripts/gen5/target5.py --ckpt <round r-1>/model.pt --manifest data/gen_train/gen5/manifest.jsonl --out <round r>/vtarget.npz
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--kappa", type=float, default=1.0)
ap.add_argument("--blend", type=float, default=0.5, help="weight of the pessimistic max (rest = the chosen command at j)")
ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--iql", action="store_true")
ap.add_argument("--max-eps", type=int, default=0, help="(smoke) only the first N episodes")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import torch
import dataset5 as D5
import model5 as M5


def main():
    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    m = M5.build(cfg, pretrained=True).to(args.device).eval()
    miss = m.load_state_dict(d["model"], strict=False)
    assert all(k.startswith("ctx.vision.backbone") for k in miss.missing_keys)
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = D5.fold_rows(rows, cfg["holdout_base"])
    use = [i for i, r in enumerate(rows) if r["split"] in ("train", "val")]
    if args.max_eps:
        keep = set(list(dict.fromkeys(rows[i]["sid"] for i in use))[: args.max_eps])
        use = [i for i in use if rows[i]["sid"] in keep]
    sub = [rows[i] for i in use]
    E = defaultdict(list)
    for j, r in enumerate(sub):
        E[r["sid"]].append(j)
    bs = [ix[s:s + 32] for ix in E.values() for s in range(0, len(ix), 32)]
    ds = D5.TickDataset(sub, norm, cfg, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=bs, num_workers=args.workers, collate_fn=D5.collate5, pin_memory=True,
                                     prefetch_factor=4)
    v_chosen = np.full(len(rows), np.nan, np.float32); v_best = np.full(len(rows), np.nan, np.float32)
    t0, n = time.time(), 0
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(args.device, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}
            if args.iql:
                C = b["cand"].gather(1, b["chosen"][:, None, None].expand(-1, 1, b["cand"].shape[-1]))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(cand=C, state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
                gi = [use[j] for j in b["idx"].tolist()]
                v_chosen[gi] = torch.sigmoid(o["value"].float())[..., 0].mean(0).cpu().numpy()          # q_data
                v_best[gi] = torch.sigmoid(o["v_state"].float())[..., 0].mean(0).cpu().numpy()          # v_data
                n += len(gi)
                if n % 20000 < 32:
                    print(f"[target5] {n}/{len(use)} rows ({n / (time.time() - t0):.0f}/s)", flush=True)
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(cand=b["cand"], state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
            V = torch.sigmoid(o["value"].float())                          # (K, B, N)
            mu, sd = V.mean(0), V.std(0)
            pess = torch.where(b["cand_mask"], mu - args.kappa * sd, torch.full_like(mu, -9.0))
            best = pess.max(1).values.clamp(0, 1)
            ch = mu.gather(1, b["chosen"][:, None])[:, 0]
            gi = [use[j] for j in b["idx"].tolist()]
            v_best[gi] = best.cpu().numpy(); v_chosen[gi] = ch.cpu().numpy()
            n += len(gi)
            if n % 20000 < 32:
                print(f"[target5] {n}/{len(use)} rows ({n / (time.time() - t0):.0f}/s)", flush=True)
    if args.iql:
        q_tgt = np.full(len(rows), np.nan, np.float32)
        nb = nt = 0
        for i in use:
            r = rows[i]; j = r.get("next_row", -1); ys = r.get("y_success")
            if r.get("term_in_h") or j is None or j < 0 or not np.isfinite(v_best[j]):
                q_tgt[i] = np.nan if ys is None else float(ys); nt += 1
            else:
                q_tgt[i] = v_best[j]; nb += 1
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, q_tgt=q_tgt, q_data=v_chosen, v_data=v_best, ckpt=os.path.abspath(args.ckpt))
        tr = np.array([rows[i]["split"] == "train" for i in range(len(rows))])
        adv = v_chosen - v_best
        print(f"[target5] IQL: {nb} bootstrapped, {nt} outcome rows -> {args.out}; train mean q_tgt {np.nanmean(q_tgt[tr]):.3f}, "
              f"Q(s,a_data) {np.nanmean(v_chosen[tr]):.3f}, V(s) {np.nanmean(v_best[tr]):.3f}, advantage sd {np.nanstd(adv[tr]):.3f}, "
              f"share > 0 {np.nanmean(adv[tr] > 0):.2f} ({time.time() - t0:.0f}s)", flush=True)
        return
    v = np.full(len(rows), np.nan, np.float32)
    n_boot = n_term = 0
    for i in use:
        r = rows[i]; j = r.get("next_row", -1)
        ys = r.get("y_success")
        if r.get("term_in_h") or j is None or j < 0 or not np.isfinite(v_best[j]):
            v[i] = np.nan if ys is None else float(ys); n_term += 1
        else:
            v[i] = args.blend * v_best[j] + (1 - args.blend) * v_chosen[j]; n_boot += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, v=v, v_best=v_best, v_chosen=v_chosen, kappa=args.kappa, blend=args.blend, ckpt=os.path.abspath(args.ckpt))
    tr = np.array([rows[i]["split"] == "train" for i in range(len(rows))])
    ysa = np.array([np.nan if r.get("y_success") is None else r["y_success"] for r in rows], np.float32)
    print(f"[target5] {n_boot} bootstrapped, {n_term} terminal / outcome rows -> {args.out}; mean target {np.nanmean(v[tr]):.3f} "
          f"(Monte Carlo {np.nanmean(ysa[tr]):.3f}); mean best-at-next {np.nanmean(v_best[tr]):.3f}, chosen-at-next {np.nanmean(v_chosen[tr]):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
