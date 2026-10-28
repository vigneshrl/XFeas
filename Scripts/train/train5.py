#!/usr/bin/env python
"""
XFeas-Gen5 training, STAGE A (2026-10-26): the short-horizon heads (fall / stall / contact within 1.5 s, progress over 1.5 s)
for the candidate the data collector actually chose. GPU 0 only (CUDA_VISIBLE_DEVICES=0 from the launcher).

    PY=$XFEAS_PY
    $PY Scripts/train/train5.py --manifest data/gen_train/gen5/manifest.jsonl --out data/gen_train/gen5/runs/A_g1c --holdout-base g1c
    $PY Scripts/train/train5.py --eval-only --ckpt <out>/model.pt --manifest ... [--eval-split val|test] [--eval-every 4]

* Model: model5 (gen4 context encoder, weights from the gen4 release; K_ENS members = candidate encoder + decoder + heads).
* Only the CHOSEN candidate has labels, so training scores just that row of the menu (the decoder is per-candidate, so the
  score of a row does not depend on the rest of the menu; model5 self-test).
* Batches = whole groups of ticks from the same episodes (--per-ep ticks of --eps-per-batch episodes): an episode file is
  read once per group (0.17 s random vs 0.01 s within an episode), so the loader is not the bottleneck.
* Ensemble diversity: each member sees a fixed 80 % bootstrap of the EPISODES (hash of sid, member) + its own init.
* Losses: BCE fall (pos_weight 5, 2 % positives), stall (NaN = fall in the window -> masked), contact; smooth-L1 progress.
  Row weights from dataset5.set_weights (1 / sqrt(n per controller x family)).
* Folds: --holdout-base b moves b and all its variants to split 'test' (never trained on, not in the norm stats).
* STAGE B (--stage B): adds the VALUE head = logit of P(eventual success | state, chosen command), trained with BCE on a soft
  target per manifest row (--vtarget npz 'v', aligned with the manifest; NaN rows -> the episode outcome y_success = the
  Monte-Carlo value of the data collector). Round 0 = Monte Carlo; rounds 1.. = bootstrapped targets from target5.py.
  --init <model.pt> starts from a previous model (Stage A or the previous round); Stage A heads keep training.
* STAGE I (--stage I, IQL; Kostrikov et al. 2021 "Offline RL with Implicit Q-Learning"): the Stage B readout showed that
  argmax of Q over the whole menu picks commands the data never tried there (crouch-creep everywhere). IQL never asks Q
  about untried commands:  Q(s,a_data) <- BCE to q_tgt = y_success if the episode ends in the window, else V_frozen(s');
  V(s) <- expectile (TAU) of Q_frozen(s, a_data) over the data's commands (an in-data 'max');  chooser pi <- advantage-
  weighted imitation: -w log softmax_menu(pi)[a_data], w = min(exp(BETA (Q_frozen(s,a_data) - V_frozen(s))), WMAX), the
  softmax over the chosen row + N_NEG random other rows of the menu. Targets per manifest row come from target5.py --iql
  (the previous round, frozen). Without --iql: Q / V on the Monte-Carlo outcome, no chooser loss (round 0).
* RESUMABLE: <out>/last.pt every --ckpt-every steps and every epoch; a restart with the same args resumes.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, hashlib, json, os, sys, time, zlib
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="")
ap.add_argument("--out", default="")
ap.add_argument("--holdout-base", default="")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--ckpt", default="")
ap.add_argument("--eval-split", default="val", help="val | test | train")
ap.add_argument("--eval-every", type=int, default=4, help="evaluate every k-th decision tick")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--per-ep", type=int, default=12, help="ticks drawn per episode per epoch")
ap.add_argument("--eps-per-batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr-ctx", type=float, default=1e-4, help="lr of the (gen4-initialised) context encoder")
ap.add_argument("--k-ens", type=int, default=5)
ap.add_argument("--boot", type=float, default=0.8)
ap.add_argument("--no-gen4", action="store_true", help="context encoder from scratch (ablation)")
ap.add_argument("--stage", default="A", choices=["A", "B", "I"])
ap.add_argument("--iql", default="", help="(stage I) npz from target5.py --iql: q_tgt, q_data, v_data per manifest row")
ap.add_argument("--tau", type=float, default=0.7, help="IQL expectile")
ap.add_argument("--beta", type=float, default=10.0, help="IQL advantage temperature")
ap.add_argument("--wmax", type=float, default=20.0)
ap.add_argument("--n-neg", type=int, default=15, help="other menu rows in the chooser softmax")
ap.add_argument("--init", default="", help="start from this model.pt (Stage B rounds)")
ap.add_argument("--gen4-ckpt", default="", help="gen4 weights for the context init (default: the gen4 release, which saw every controller)")
ap.add_argument("--vtarget", default="", help="npz with 'v' (len = manifest rows) value targets; empty = Monte Carlo (y_success)")
ap.add_argument("--value-w", type=float, default=1.0)
ap.add_argument("--no-margin", action="store_true", help="no per-candidate fit margins (the first Stage A run, ablation)")
ap.add_argument("--max-eps", type=int, default=0, help="(smoke) use the first N train episodes")
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--ckpt-every", type=int, default=300)
ap.add_argument("--no-resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_num_threads(args.threads)
import dataset5 as D5
import model5 as M5

LOSS_W = dict(fall=1.0, stall=0.5, contact=0.5, progress=0.5)
POS_W = dict(fall=5.0, stall=1.0, contact=1.0)


def cfg_dict():
    return dict(backbone="dinov3_s", holdout_base=args.holdout_base, use_margin=not args.no_margin, k_ens=args.k_ens, boot=args.boot, gen4_init=not args.no_gen4,
                use_fp=True, use_hist=True, use_map=True, use_frames_hist=True, k_hist=4, loss_w=LOSS_W, pos_w=POS_W,
                epochs=args.epochs, per_ep=args.per_ep, eps_per_batch=args.eps_per_batch, lr=args.lr, lr_ctx=args.lr_ctx,
                aug=dict(D5.DG.DEFAULT_AUG), manifest=os.path.abspath(args.manifest) if args.manifest else "", stage=args.stage,
                init=os.path.abspath(args.init) if args.init else "", vtarget=os.path.abspath(args.vtarget) if args.vtarget else "",
                value_w=args.value_w, iql=os.path.abspath(args.iql) if args.iql else "", tau=args.tau, beta=args.beta,
                wmax=args.wmax, n_neg=args.n_neg)


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


def load_rows(holdout):
    rows = [json.loads(l) for l in open(args.manifest)]
    if any(D5.K5.is_held_out(r["ctrl_id"]) for r in rows):
        sys.exit("[train5] held-out controller rows in the manifest: refused")
    if holdout and holdout not in {r["base"] for r in rows}:
        sys.exit(f"[train5] hold-out base {holdout} not in the manifest")
    for i, r in enumerate(rows):
        r["mi"] = i                                          # manifest index (value targets are aligned with it)
    rows = D5.fold_rows(rows, holdout)
    D5.set_weights([r for r in rows if r["split"] == "train"])
    return rows


def by_episode(rows):
    E = defaultdict(list)
    for i, r in enumerate(rows):
        E[r["sid"]].append(i)
    return E


class EpisodeBatches(torch.utils.data.Sampler):
    """yields batches = per_ep random ticks of each of eps_per_batch episodes, sorted by episode (one file read per group)"""

    def __init__(self, ep_idx, per_ep, eps_per_batch, seed):
        self.E, self.k, self.g, self.seed = list(ep_idx.values()), per_ep, eps_per_batch, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(self.E))
        for s in range(0, len(order) - self.g + 1, self.g):
            b = []
            for e in order[s:s + self.g]:
                ix = self.E[e]
                b += list(rng.choice(ix, min(self.k, len(ix)), replace=False))
            yield b

    def __len__(self):
        return len(self.E) // self.g


def boot_mask(sids, K, p):
    """(K, B) bool: member k trains on episode sid iff hash(sid, k) < p (fixed for the whole run)"""
    return torch.tensor([[(zlib.crc32(f"{s}|{k}".encode()) % 10000) / 10000.0 < p for s in sids] for k in range(K)])


def to(b, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}


def chosen_rows(b):
    """(B,1,F) descriptor of the chosen candidate"""
    return b["cand"].gather(1, b["chosen"][:, None, None].expand(-1, 1, b["cand"].shape[-1]))


def menu_subset(b, n_neg):
    """(B,1+n_neg) candidate indices: the chosen row first, then random other valid rows (padded with -1)"""
    cm = b["cand_mask"].clone()
    cm.scatter_(1, b["chosen"][:, None], False)
    r = torch.rand(cm.shape, device=cm.device).masked_fill(~cm, -1.0)
    k = min(n_neg, cm.shape[1])
    v, ix = r.topk(k, 1)
    ix = torch.where(v >= 0, ix, torch.full_like(ix, -1))
    return torch.cat([b["chosen"][:, None], ix], 1)


def iql_loss(out, bm, w, vexp, adv):
    """value / v_state / pi outputs (K,B,1+n) with row 0 = the chosen command. vexp (B,) expectile targets for V(s),
    adv (B,) frozen advantages (NaN = no chooser loss); sub_ok (B,1+n) valid rows"""
    L, parts = 0.0, {}
    ok = torch.isfinite(vexp)[None] & bm
    if ok.any():
        u = torch.nan_to_num(vexp, nan=0.0)[None] - torch.sigmoid(out["v_state"][..., 0])
        l = (torch.abs(args.tau - (u < 0).float()) * u ** 2)
        l = (l * ok * w[None]).sum() / (ok * w[None]).sum()
        L = L + l; parts["v_state"] = float(l)
    ok = torch.isfinite(adv)[None] & bm
    if ok.any():
        aw = torch.exp(args.beta * torch.nan_to_num(adv, nan=0.0)).clamp(max=args.wmax)
        lg = out["pi"].masked_fill(~out["_sub_ok"][None], -1e4)
        nll = -torch.log_softmax(lg, -1)[..., 0]
        l = (nll * aw[None] * ok * w[None]).sum() / (ok * w[None]).sum()
        L = L + l; parts["pi"] = float(l); parts["adv_w"] = float(aw.mean())
    return L, parts


def loss_fn(out, b, bm, w, vt=None):
    """out: dict (K,B,1); bm (K,B) bootstrap mask; w (B,) row weights; vt (B,) value targets in [0, 1] (Stage B) or None"""
    L, parts = 0.0, {}
    if vt is not None:
        ok = torch.isfinite(vt)[None] & bm
        if ok.any():
            lg = out["value"][..., 0]
            l = F.binary_cross_entropy_with_logits(lg, torch.nan_to_num(vt, nan=0.0)[None].expand_as(lg), reduction="none")
            l = (l * ok * w[None]).sum() / (ok * w[None]).sum()
            L = L + args.value_w * l; parts["value"] = float(l)
    for k in ("fall", "stall", "contact"):
        y = b[f"y_{k}"]; ok = torch.isfinite(y)[None] & bm
        if ok.any():
            lg = out[k][..., 0]
            l = F.binary_cross_entropy_with_logits(lg, torch.nan_to_num(y, nan=0.0)[None].expand_as(lg), reduction="none",
                                                   pos_weight=torch.tensor(POS_W[k], device=lg.device))
            l = (l * ok * w[None]).sum() / (ok * w[None]).sum()
            L = L + LOSS_W[k] * l; parts[k] = float(l)
    y = b["y_progress"]; ok = torch.isfinite(y)[None] & bm
    if ok.any():
        pr = out["progress"][..., 0]
        l = F.smooth_l1_loss(pr, torch.nan_to_num(y, nan=0.0)[None].expand_as(pr), beta=0.1, reduction="none")
        l = (l * ok * w[None]).sum() / (ok * w[None]).sum()
        L = L + LOSS_W["progress"] * l; parts["progress"] = float(l)
    return L, parts


def rows_all_sorted(rows):
    """rows back in manifest order (rows carry 'mi')"""
    return sorted(rows, key=lambda r: r["mi"])


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(int(torch.initial_seed()) % (1 << 31))


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_loader(ds, sampler):
    return DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, collate_fn=D5.collate5, worker_init_fn=worker_init,
                      pin_memory=True, prefetch_factor=4 if args.workers > 0 else None, persistent_workers=False)


def train(rows, norm, cfg):
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    m = M5.build(cfg, pretrained=True).to(dev)
    VT = VEXP = ADV = None
    if args.stage == "I":
        ys = np.array([np.nan if r.get("y_success") is None else r["y_success"] for r in rows_all_sorted(rows)], np.float32)
        VT, VEXP = ys.copy(), ys.copy()
        if args.iql:
            Z = np.load(args.iql)
            assert len(Z["q_tgt"]) == len(ys)
            VT = np.where(np.isfinite(Z["q_tgt"]), Z["q_tgt"], ys).astype(np.float32)
            VEXP = np.where(np.isfinite(Z["q_data"]), Z["q_data"], ys).astype(np.float32)
            ADV = (Z["q_data"] - Z["v_data"]).astype(np.float32)
        print(f"[train5] stage I (IQL tau {args.tau} beta {args.beta}): Q targets {'Monte Carlo' if not args.iql else args.iql}; "
              f"mean Q target {np.nanmean(VT):.3f}" + (f"; advantage mean {np.nanmean(ADV):+.3f} sd {np.nanstd(ADV):.3f}" if ADV is not None else
                                                   "; no chooser loss (round 0)"), flush=True)
    if args.stage == "B":
        ys = np.array([np.nan if r.get("y_success") is None else r["y_success"] for r in rows_all_sorted(rows)], np.float32)
        VT = ys.copy()
        if args.vtarget:
            v = np.load(args.vtarget)["v"].astype(np.float32)
            assert len(v) == len(ys), f"vtarget {len(v)} rows != manifest {len(ys)}"
            VT = np.where(np.isfinite(v), v, ys)
        print(f"[train5] stage B value targets: {'Monte Carlo (y_success)' if not args.vtarget else args.vtarget}; "
              f"mean {np.nanmean(VT):.3f}", flush=True)
    if args.init:
        d0 = torch.load(args.init, map_location="cpu", weights_only=False)
        miss = m.load_state_dict(d0["model"], strict=False)
        new = [k for k in miss.missing_keys if not k.startswith("ctx.vision.backbone")]
        assert all(".pi." in k or ".vs." in k for k in new), new[:5]       # only the IQL heads may be new (init from Stage A / B)
        print(f"[train5] init from {args.init}" + (f" ({len(new)} new IQL head tensors)" if new else ""), flush=True)
    elif cfg["gen4_init"]:
        lo, pa, sk, mi = M5.load_gen4(m, args.gen4_ckpt) if args.gen4_ckpt else M5.load_gen4(m)
        if args.gen4_ckpt: print(f"[train5] gen4 weights {args.gen4_ckpt}", flush=True)
        print(f"[train5] gen4 init: {len(lo)} loaded, {len(pa)} widened, {len(sk)} gen4-only skipped, {len(mi)} new", flush=True)
    tr_rows = [r for r in rows if r["split"] == "train"]
    E = by_episode(tr_rows)
    if args.max_eps:
        E = dict(list(E.items())[: args.max_eps])
    ctx_p = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("ctx.")]
    mem_p = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("ctx.")]
    opt = torch.optim.AdamW([dict(params=ctx_p, lr=args.lr_ctx), dict(params=mem_p, lr=args.lr)], weight_decay=0.05)
    steps_per_ep = len(E) // args.eps_per_batch
    total = args.max_steps or args.epochs * steps_per_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr_ctx, args.lr], total_steps=max(total, 20), pct_start=0.1)
    step, ep0, hist, t_prev = 0, 0, [], 0.0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last) and not args.no_resume:
        d = torch.load(last, map_location="cpu", weights_only=False)
        assert d["cfg"]["holdout_base"] == cfg["holdout_base"] and d["total"] == total, "last.pt of another run"
        m.load_state_dict(d["model"], strict=False); opt.load_state_dict(d["opt"]); sched.load_state_dict(d["sched"])
        step, ep0, hist, t_prev = d["step"], d["epoch"], d["history"], d.get("sec", 0.0)
        print(f"[train5] RESUMED from {last}: step {step}/{total}, epoch {ep0}", flush=True)
    ds = D5.TickDataset(tr_rows, norm, cfg, augment=True, aug=cfg["aug"], seed=args.seed + 1000 * ep0)
    print(f"[train5] fold '{args.holdout_base or 'none'}': {len(E)} train episodes, {len(tr_rows)} train ticks, "
          f"bases {sorted({r['base'] for r in tr_rows})}; {steps_per_ep} steps/epoch x {args.epochs} = {total}; "
          f"batch {args.per_ep} ticks x {args.eps_per_batch} episodes; trainable ctx {sum(p.numel() for p in ctx_p) / 1e6:.2f}M "
          f"members {sum(p.numel() for p in mem_p) / 1e6:.2f}M", flush=True)

    def ckpt(ep_done):
        save_atomic(dict(model={k: v for k, v in m.state_dict().items() if not k.startswith("ctx.vision.backbone")}, opt=opt.state_dict(),
                         sched=sched.state_dict(), step=step, epoch=ep_done, total=total, cfg=cfg, history=hist,
                         sec=t_prev + time.time() - t0, saved=time.strftime("%Y-%m-%d %H:%M:%S")), last)

    t0 = time.time()
    ep = ep0
    while step < total:
        smp = EpisodeBatches(E, args.per_ep, args.eps_per_batch, seed=args.seed * 1000 + ep)
        m.train(); acc, n, t_ep = 0.0, 0, time.time()
        skip = step - ep * steps_per_ep                       # mid-epoch resume: skip the batches already done
        for bi, b in enumerate(make_loader(ds, smp)):
            if bi < skip:
                continue
            b = to(b, dev)
            sids = [tr_rows[i]["sid"] for i in b["idx"].tolist()]
            bm = boot_mask(sids, args.k_ens, args.boot).to(dev)
            w = torch.tensor([tr_rows[i]["weight"] for i in b["idx"].tolist()], device=dev, dtype=torch.float32)
            if args.stage == "I":
                sub = menu_subset(b, args.n_neg)
                sub_ok = sub >= 0
                C = b["cand"].gather(1, sub.clamp(min=0)[..., None].expand(-1, -1, b["cand"].shape[-1]))
            else:
                C = chosen_rows(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(cand=C, state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
            out = {k: v.float() for k, v in out.items()}
            mi = [tr_rows[i]["mi"] for i in b["idx"].tolist()]
            vt = None
            if VT is not None:
                vt = torch.tensor(VT[mi], device=dev)
            if args.stage == "I":
                out0 = {k: v[..., :1] for k, v in out.items()}
                loss, parts = loss_fn(out0, b, bm, w, vt)
                out["_sub_ok"] = sub_ok
                li, pi_parts = iql_loss(out, bm, w, torch.tensor(VEXP[mi], device=dev),
                                        torch.tensor(ADV[mi], device=dev) if ADV is not None else torch.full((len(mi),), float("nan"), device=dev))
                loss = loss + li; parts.update(pi_parts)
            else:
                loss, parts = loss_fn(out, b, bm, w, vt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(ctx_p + mem_p, 1.0)
            opt.step(); sched.step(); step += 1; n += 1; acc += float(loss)
            if step % args.log_every == 0 or step <= 2 or step == total:
                rec = dict(step=step, ep=ep, loss=round(float(loss), 4), **{k: round(v, 4) for k, v in parts.items()},
                           sec=round(t_prev + time.time() - t0, 1), it_s=round(n / (time.time() - t_ep), 2))
                hist.append(rec); print("[train5] " + json.dumps(rec), flush=True)
            if args.ckpt_every and step % args.ckpt_every == 0 and step < total:
                ckpt(ep)
            if step >= total:
                break
        print(f"[train5] ep{ep} mean loss {acc / max(n, 1):.4f} ({time.time() - t_ep:.0f}s)", flush=True)
        ep += 1
        ckpt(ep)
    return m, hist


# ------------------------------------------------------------------------------------------------ evaluation
def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(y, p, nb=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(nb) if (b == i).any()))


@torch.no_grad()
def predict(m, rows, norm, cfg, dev, every):
    sel = [i for i, r in enumerate(rows) if (r["tick"] // 1) % every == 0] if every > 1 else list(range(len(rows)))
    sub = [rows[i] for i in sel]
    E = by_episode(sub)
    ds = D5.TickDataset(sub, norm, cfg, augment=False)

    class Seq(torch.utils.data.Sampler):
        def __iter__(self):
            for ix in E.values():
                for s in range(0, len(ix), 48):
                    yield ix[s:s + 48]

        def __len__(self):
            return sum((len(ix) + 47) // 48 for ix in E.values())

    P = defaultdict(list)
    m.eval()
    for b in make_loader(ds, Seq()):
        b = to(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(cand=chosen_rows(b), state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in M5.CTX_KEYS})
        for k in ("fall", "stall", "contact"):
            pk = torch.sigmoid(out[k][..., 0].float())
            P[f"p_{k}"].append(pk.mean(0).cpu().numpy()); P[f"p_{k}_std"].append(pk.std(0).cpu().numpy())
        pv = torch.sigmoid(out["value"][..., 0].float())
        P["p_value"].append(pv.mean(0).cpu().numpy()); P["p_value_std"].append(pv.std(0).cpu().numpy())
        P["y_success"].append(b["y_success"].cpu().numpy())
        pr = out["progress"][..., 0].float()
        P["r_progress"].append(pr.mean(0).cpu().numpy()); P["r_progress_std"].append(pr.std(0).cpu().numpy())
        for k in ("y_fall", "y_stall", "y_contact", "y_progress"):
            P[k].append(b[k].cpu().numpy())
        P["row"].append(b["idx"].cpu().numpy())
    out = {k: np.concatenate(v) for k, v in P.items()}
    for k in ("sid", "base", "ctrl_id", "family5", "env", "end"):
        out[k] = np.array([sub[i][k] for i in out["row"]])
    out["tick"] = np.array([sub[i]["tick"] for i in out["row"]])
    return out


def metrics(P, mask=None):
    mk0 = np.ones(len(P["row"]), bool) if mask is None else mask
    res = dict(n=int(mk0.sum()))
    for k in ("fall", "stall", "contact"):
        y, p = P[f"y_{k}"], P[f"p_{k}"]; mk = mk0 & np.isfinite(y)
        res[f"auroc_{k}"] = auroc(y[mk], p[mk]); res[f"brier_{k}"] = float(np.mean((p[mk] - y[mk]) ** 2)) if mk.any() else float("nan")
        res[f"ece_{k}"] = ece(y[mk], p[mk]) if mk.any() else float("nan"); res[f"rate_{k}"] = float(np.mean(y[mk])) if mk.any() else float("nan")
    if "p_value" in P:
        y, p = P["y_success"], P["p_value"]; mk = mk0 & np.isfinite(y)
        res["auroc_value_vs_success"] = auroc(y[mk], p[mk]); res["brier_value"] = float(np.mean((p[mk] - y[mk]) ** 2)) if mk.any() else float("nan")
        res["rate_success"] = float(np.mean(y[mk])) if mk.any() else float("nan")
    y, p = P["y_progress"], P["r_progress"]; mk = mk0 & np.isfinite(y)
    res["mae_progress"] = float(np.abs(p[mk] - y[mk]).mean()) if mk.any() else float("nan")
    res["mae_progress_const"] = float(np.abs(np.median(y[mk]) - y[mk]).mean()) if mk.any() else float("nan")
    return res


def do_eval(ckpt, split):
    dev = args.device
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    m = M5.build(cfg, pretrained=True).to(dev)
    miss = m.load_state_dict(d["model"], strict=False)
    assert all(k.startswith("ctx.vision.backbone") for k in miss.missing_keys), miss.missing_keys[:5]
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = D5.fold_rows(rows, cfg["holdout_base"])
    ev = [r for r in rows if r["split"] == split]
    t0 = time.time()
    P = predict(m, ev, norm, cfg, dev, args.eval_every)
    res = dict(split=split, holdout_base=cfg["holdout_base"], ckpt=os.path.abspath(ckpt), seconds=round(time.time() - t0, 1), all=metrics(P))
    res["per_base"] = {b: metrics(P, P["base"] == b) for b in sorted(set(P["base"]))}
    od = os.path.dirname(os.path.abspath(ckpt))
    np.savez_compressed(os.path.join(od, f"{split}_preds.npz"), **P)
    json.dump(res, open(os.path.join(od, f"metrics_{split}.json"), "w"), indent=1)
    r = lambda x: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}
    print(f"[eval5] {split} all " + json.dumps(r(res["all"])), flush=True)
    for b, v in res["per_base"].items():
        print(f"[eval5] {split} {b:7s} " + json.dumps(r(v)), flush=True)
    return res


def main():
    if args.eval_only:
        do_eval(args.ckpt, args.eval_split)
        return
    cfg = cfg_dict()
    rows = load_rows(args.holdout_base)
    os.makedirs(args.out, exist_ok=True)
    norm_p = os.path.join(args.out, "norm.json")
    msha = sha(args.manifest)
    if os.path.exists(norm_p):
        norm = json.load(open(norm_p))
        if norm.get("manifest_sha") != msha or norm.get("holdout_base") != args.holdout_base:
            sys.exit(f"[train5] {norm_p} was computed for another manifest / fold")
    else:
        norm = D5.compute_norm(rows); norm.update(manifest_sha=msha, holdout_base=args.holdout_base)
        tmp = norm_p + ".tmp"; json.dump(norm, open(tmp, "w")); os.replace(tmp, norm_p)
    assert not (set(norm["fp_ctrls"]) & {r["ctrl_id"] for r in rows if r["split"] == "test"}), "held-out controllers in the norm stats"
    m, h = train(rows, norm, cfg)
    path = os.path.join(args.out, "model.pt")
    save_atomic(dict(format=M5.FORMAT, cfg=cfg, norm=norm, model={k: v.detach().cpu() for k, v in m.state_dict().items()
                                                                  if not k.startswith("ctx.vision.backbone")},
                     manifest_sha=msha, history=h, saved=time.strftime("%Y-%m-%d %H:%M")), path)
    print(f"[train5] saved {path}", flush=True)


if __name__ == "__main__":
    main()
