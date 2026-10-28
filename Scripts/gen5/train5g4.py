#!/usr/bin/env python
"""
Train the gen5 model on gen4 SIDE-BY-SIDE data (2026-10-26; dataset5g4.py, analysis/GEN4_VS_GEN5_STATUS.md 6 option 1).
One item = one situation x controller: the gen4 snapshot context + K candidates (the controller commands gen4 actually held
from the snapshot, described by describe5) each with its REAL outcome. All K outcomes are supervised (the comparison the
gen5 per-tick data lacked), plus a within-situation ranking term on the success logit.

Heads (model5, reused): value <- success (goal reached holding this command), fall <- fell, contact <- hazard body contact,
stall <- stall, progress <- progress (m). The selection score is the same as gen4's: U = mean P(success) - KAPPA std - LAM P(fall).
Context init: the gen4 checkpoint of the SAME fold (--gen4-ckpt; e.g. gen4 full/f_g1c never saw AMO), not the gen4 release.

    PY=/p/cral/vignesh/envs/gsam2/bin/python
    $PY scripts/gen5/train5g4.py --out data/gen_train/gen5g4/runs/g1c --holdout-base g1c \
        --gen4-ckpt data/gen_train/gen4/full/f_g1c/ckpt/xfeas_gen4_ens.pt
    $PY scripts/gen5/train5g4.py --eval-only --ckpt <out>/model.pt --eval-split test
Eval (per situation): pick = argmax U over the tried commands; success rate of the pick vs the stock walk vs the best tried
command (oracle) vs a random tried command; capture (pick succeeds where stock fails), harm (pick fails where stock succeeds);
AUROC of P(success) over all candidates; within-situation pair accuracy. Same quantities as gen4's learned_selection.json.
"""
import argparse, json, os, sys, time, zlib
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=os.path.join(THIS, "..", "..", "data", "gen_train", "gen5g4", "manifest.jsonl"))
ap.add_argument("--out", default="")
ap.add_argument("--holdout-base", default="")
ap.add_argument("--gen4-ckpt", default="")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--ckpt", default="")
ap.add_argument("--eval-split", default="val")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr-ctx", type=float, default=1e-4)
ap.add_argument("--rank-w", type=float, default=0.5)
ap.add_argument("--kappa", type=float, default=0.5)
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--k-ens", type=int, default=5)
ap.add_argument("--boot", type=float, default=0.8)
ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--max-rows", type=int, default=0)
ap.add_argument("--log-every", type=int, default=25)
ap.add_argument("--no-resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.set_num_threads(4)
import canon5
import dataset5 as D5
import model5 as M5

HEAD_OF = dict(value=0, fall=1, contact=2, stall=3)          # model head <- label column (dataset5g4.LABELS)
LOSS_W = dict(value=1.0, fall=0.5, contact=0.3, stall=0.3, progress=0.2)
STATE0 = np.array([1.0, 0.0], np.float32)                    # gen4 snapshot: walking the stock command, no switch yet


class SitItems(Dataset):
    def __init__(self, rows, norm, cfg, augment=False, seed=0):
        self.rows, self.norm, self.cfg, self.augment = rows, norm, cfg, augment
        self.aug = dict(D5.DG.DEFAULT_AUG)
        self._rng = np.random.default_rng(seed)
        self._spec, self._probe, self._ptr = {}, {}, {}

    def __len__(self):
        return len(self.rows)

    def _json(self, d, p):
        if p not in d:
            d[p] = json.load(open(p))
        return d[p]

    def __getitem__(self, i):
        r = self.rows[i]
        on = np.load(r["obs"], allow_pickle=False)
        o = {k: on[k] for k in on.files}
        spec = self._json(self._spec, r["spec"]); probe = self._json(self._probe, r["probe"])
        desc = D5.robot_desc5(r["robot"])
        ctx, feas_raw = D5.DG.build_context(o, spec, probe, desc, self.norm, self.cfg, train=self.augment, rng=self._rng, aug=self.aug,
                                            ptrace_cache=self._ptr)
        t = torch.from_numpy
        out = {k: t(np.asarray(v)) for k, v in ctx.items()}
        fr = feas_raw.astype(np.float32) if feas_raw is not None else np.full(16, np.nan, np.float32)
        out.update(feas_raw=t(fr), state=t(STATE0.copy()), cand=t(np.asarray(r["X"], np.float32)), y=t(np.asarray(r["y"], np.float32)),
                   stock=torch.tensor(r["stock"], dtype=torch.bool), idx=i)
        return out


def collate(batch):
    B = len(batch); K = max(b["rgb"].shape[0] for b in batch); N = max(b["cand"].shape[0] for b in batch)
    out = {}
    rgb = torch.zeros(B, K, 3, 224, 224, dtype=torch.uint8); dep = torch.zeros(B, K, 1, 224, 224)
    fmask = torch.zeros(B, K, dtype=torch.bool); fage = torch.zeros(B, K)
    for i, b in enumerate(batch):
        k = b["rgb"].shape[0]; rgb[i, :k] = b["rgb"]; dep[i, :k] = b["depth"]; fmask[i, :k] = True; fage[i, :k] = b["fage"]
    out.update(rgb=rgb, depth=dep, fmask=fmask, fage=fage)
    for k in M5.CTX_KEYS:
        if k not in out:
            out[k] = torch.stack([b[k] for b in batch])
    out["feas_raw"] = torch.stack([b["feas_raw"] for b in batch]); out["state"] = torch.stack([b["state"] for b in batch])
    cand = torch.zeros(B, N, batch[0]["cand"].shape[1]); y = torch.full((B, N, 5), float("nan")); cm = torch.zeros(B, N, dtype=torch.bool)
    st = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["cand"].shape[0]; cand[i, :n] = b["cand"]; y[i, :n] = b["y"]; cm[i, :n] = True; st[i, :n] = b["stock"]
    out.update(cand=cand, y=y, cand_mask=cm, stock=st, idx=torch.tensor([b["idx"] for b in batch]))
    return out


def to(b, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}


def run(m, b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(cand=b["cand"], state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
    return {k: v.float() for k, v in o.items()}


def boot_mask(keys, K, p):
    return torch.tensor([[(zlib.crc32(f"{s}|{k}".encode()) % 10000) / 10000.0 < p for s in keys] for k in range(K)])


def loss_fn(o, b, bm):
    cm = b["cand_mask"]; y = b["y"]
    M = (bm[:, :, None] & cm[None]).float()                    # (K,B,N)
    L, parts = 0.0, {}
    for h, j in HEAD_OF.items():
        yy = y[..., j]; ok = M * torch.isfinite(yy)[None].float()
        l = F.binary_cross_entropy_with_logits(o[h], torch.nan_to_num(yy, nan=0.0)[None].expand_as(o[h]), reduction="none")
        l = (l * ok).sum() / ok.sum().clamp(min=1); L = L + LOSS_W[h] * l; parts[h] = float(l)
    yy = y[..., 4]; ok = M * torch.isfinite(yy)[None].float()
    l = F.smooth_l1_loss(o["progress"], torch.nan_to_num(yy, nan=0.0)[None].expand_as(o["progress"]), beta=0.1, reduction="none")
    l = (l * ok).sum() / ok.sum().clamp(min=1); L = L + LOSS_W["progress"] * l; parts["progress"] = float(l)
    if args.rank_w > 0:                                           # within-situation: successful commands above failed ones
        ys = torch.nan_to_num(y[..., 0], nan=-1.0)
        pos = (ys == 1) & cm; neg = (ys == 0) & cm
        pm = (pos[:, :, None] & neg[:, None, :])[None] & bm[:, :, None, None]
        if pm.any():
            d = o["value"][:, :, :, None] - o["value"][:, :, None, :]
            l = F.softplus(-d[pm]).mean(); L = L + args.rank_w * l; parts["rank"] = float(l)
    return L, parts


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(int(torch.initial_seed()) % (1 << 31))


def save_atomic(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def load_rows(holdout):
    rows = [json.loads(l) for l in open(args.manifest)]
    for r in rows:
        if holdout and r["base"] == holdout:
            r["split"] = "test"
    return rows


def gen4_norm(rows, holdout):
    """gen4 context normalisation on the training rows of this fold (dataset_gen4.compute_norm on gen4-format rows)"""
    g4 = [dict(obs=r["obs"], eps=[o["ep"] for o in r["opts"]], spec=r["spec"], probe=r["probe"], ctrl_id=r["ctrl_id"], base=r["base"],
               robot=r["robot"], split=r["split"]) for r in rows]
    return D5.DG.compute_norm(g4)


def train():
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = dict(backbone="dinov3_s", holdout_base=args.holdout_base, k_ens=args.k_ens, use_margin=True, use_fp=True, use_hist=True,
               use_map=True, use_frames_hist=True, k_hist=4, data="gen5g4", gen4_ckpt=os.path.abspath(args.gen4_ckpt) if args.gen4_ckpt else "",
               loss_w=LOSS_W, rank_w=args.rank_w, kappa=args.kappa, lam=args.lam, epochs=args.epochs, bs=args.bs)
    rows = load_rows(args.holdout_base)
    tr = [r for r in rows if r["split"] == "train"]
    if args.max_rows:
        tr = tr[: args.max_rows]
    os.makedirs(args.out, exist_ok=True)
    np_ = os.path.join(args.out, "norm.json")
    if os.path.exists(np_):
        norm = json.load(open(np_))
    else:
        norm = gen4_norm(rows, args.holdout_base); json.dump(norm, open(np_ + ".tmp", "w")); os.replace(np_ + ".tmp", np_)
    assert not (set(norm["fp_ctrls"]) & {r["ctrl_id"] for r in rows if r["split"] == "test"}), "held-out controllers in the norm"
    m = M5.build(cfg, pretrained=True).to(dev)
    if args.gen4_ckpt:
        lo, pa, sk, mi = M5.load_gen4(m, args.gen4_ckpt)
        print(f"[g4train] context from {args.gen4_ckpt}: {len(lo)} loaded, {len(pa)} widened", flush=True)
    ctx_p = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("ctx.")]
    mem_p = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("ctx.")]
    opt = torch.optim.AdamW([dict(params=ctx_p, lr=args.lr_ctx), dict(params=mem_p, lr=args.lr)], weight_decay=0.05)
    spe = len(tr) // args.bs
    total = args.max_steps or args.epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr_ctx, args.lr], total_steps=max(total, 20), pct_start=0.1)
    step, ep, hist, t_prev = 0, 0, [], 0.0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last) and not args.no_resume:
        d = torch.load(last, map_location="cpu", weights_only=False)
        assert d["total"] == total and d["cfg"]["holdout_base"] == args.holdout_base
        m.load_state_dict(d["model"], strict=False); opt.load_state_dict(d["opt"]); sched.load_state_dict(d["sched"])
        step, ep, hist, t_prev = d["step"], d["epoch"], d["history"], d["sec"]
        print(f"[g4train] RESUMED step {step}/{total} epoch {ep}", flush=True)
    ds = SitItems(tr, norm, cfg, augment=True, seed=args.seed + ep)
    print(f"[g4train] fold '{args.holdout_base}': {len(tr)} train situations, {sum(len(r['y']) for r in tr)} candidates, "
          f"bases {sorted({r['base'] for r in tr})}; {spe} steps/epoch, total {total}", flush=True)
    t0 = time.time()
    while step < total:
        g = torch.Generator().manual_seed(args.seed * 1000 + ep)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, generator=g, num_workers=args.workers, collate_fn=collate,
                        worker_init_fn=worker_init, drop_last=True, pin_memory=True, prefetch_factor=4)
        m.train(); acc, n, te = 0.0, 0, time.time()
        for b in dl:
            b = to(b, dev)
            bm = boot_mask([tr[i]["key"] for i in b["idx"].tolist()], args.k_ens, args.boot).to(dev)
            o = run(m, b)
            loss, parts = loss_fn(o, b, bm)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ctx_p + mem_p, 1.0)
            opt.step(); sched.step(); step += 1; n += 1; acc += float(loss)
            if step % args.log_every == 0 or step <= 2:
                rec = dict(step=step, ep=ep, loss=round(float(loss), 4), **{k: round(v, 4) for k, v in parts.items()}, sec=round(t_prev + time.time() - t0, 1))
                hist.append(rec); print("[g4train] " + json.dumps(rec), flush=True)
            if step >= total:
                break
        ep += 1
        print(f"[g4train] ep{ep - 1} mean loss {acc / max(n, 1):.4f} ({time.time() - te:.0f}s)", flush=True)
        save_atomic(dict(model={k: v for k, v in m.state_dict().items() if not k.startswith("ctx.vision.backbone")}, opt=opt.state_dict(),
                         sched=sched.state_dict(), step=step, epoch=ep, total=total, cfg=cfg, history=hist, sec=t_prev + time.time() - t0), last)
    path = os.path.join(args.out, "model.pt")
    save_atomic(dict(format=M5.FORMAT + "_g4", cfg=cfg, norm=norm, history=hist, saved=time.strftime("%Y-%m-%d %H:%M"),
                     model={k: v.detach().cpu() for k, v in m.state_dict().items() if not k.startswith("ctx.vision.backbone")}), path)
    print(f"[g4train] saved {path}", flush=True)


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(ckpt, split):
    dev = args.device
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    m = M5.build(cfg, pretrained=True).to(dev).eval(); m.load_state_dict(d["model"], strict=False)
    rows = [r for r in load_rows(cfg["holdout_base"]) if r["split"] == split]
    if args.max_rows:
        rows = rows[: args.max_rows]
    dl = DataLoader(SitItems(rows, norm, cfg), batch_size=16, shuffle=False, num_workers=args.workers, collate_fn=collate)
    S = []
    for b in dl:
        b = to(b, dev); o = run(m, b)
        ps = torch.sigmoid(o["value"]); pf = torch.sigmoid(o["fall"]).mean(0)
        U = ps.mean(0) - cfg.get("kappa", args.kappa) * ps.std(0) - cfg.get("lam", args.lam) * pf
        for j, i in enumerate(b["idx"].tolist()):
            n = len(rows[i]["y"])
            S.append(dict(i=i, U=U[j, :n].cpu().numpy(), p=ps.mean(0)[j, :n].cpu().numpy(), pstd=ps.std(0)[j, :n].cpu().numpy()))
    rng = np.random.default_rng(0)
    rec = defaultdict(list)
    allp, ally = [], []
    pair_ok = pair_n = 0
    for s in S:
        r = rows[s["i"]]; y = np.asarray(r["y"])[:, 0]; stock = np.asarray(r["stock"])
        a = int(np.argmax(s["U"])); st = int(np.argmax(stock)) if stock.any() else 0
        rec["pick"].append(y[a]); rec["stock"].append(y[st]); rec["oracle"].append(y.max()); rec["random"].append(y.mean())
        rec["pick_is_stock"].append(float(a == st)); rec["fam"].append(r["family"]); rec["base"].append(r["base"])
        if y[st] == 0:
            rec["capture"].append(y[a])
            if y.max() == 1:
                rec["capture_possible"].append(y[a])
        else:
            rec["harm"].append(1.0 - y[a])
        allp += list(s["p"]); ally += list(y)
        for u in range(len(y)):
            for v in range(len(y)):
                if y[u] == 1 and y[v] == 0:
                    pair_n += 1; pair_ok += float(s["p"][u] > s["p"][v])
    mean = lambda k: round(float(np.mean(rec[k])), 3) if rec[k] else None
    res = dict(split=split, holdout=cfg["holdout_base"], n_sits=len(S), pick_success=mean("pick"), stock_success=mean("stock"),
               oracle_success=mean("oracle"), random_success=mean("random"), capture=mean("capture"), n_capture=len(rec["capture"]),
               capture_when_possible=mean("capture_possible"), n_capture_possible=len(rec["capture_possible"]), harm=mean("harm"),
               n_harm=len(rec["harm"]), picks_stock=mean("pick_is_stock"), auroc_success=round(auroc(ally, allp), 3),
               pair_acc=round(pair_ok / max(pair_n, 1), 3), n_pairs=pair_n)
    fam = {}
    for f in sorted(set(rec["fam"])):
        mk = np.asarray(rec["fam"]) == f
        fam[f] = dict(n=int(mk.sum()), pick=round(float(np.asarray(rec["pick"])[mk].mean()), 3), stock=round(float(np.asarray(rec["stock"])[mk].mean()), 3),
                      oracle=round(float(np.asarray(rec["oracle"])[mk].mean()), 3))
    res["per_family"] = fam
    od = os.path.dirname(os.path.abspath(ckpt))
    json.dump(res, open(os.path.join(od, f"select_{split}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if args.eval_only:
        evaluate(args.ckpt, args.eval_split)
    else:
        train()
