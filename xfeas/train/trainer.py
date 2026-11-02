#!/usr/bin/env python
"""Trains the XFeas model (`python train.py ...` calls this). Each training item is one moment in a scene where
several commands were actually tried from the same state, each with its real outcome; the model learns to
predict those outcomes and to rank the commands. Also evaluates a trained model offline (--eval-only).

Details:
Train the gen5 model on gen5b SIDE-BY-SIDE branches + gen4 side-by-side options.
Same recipe as xfeas/train/trainer_gen4data.py: every tried command of a moment is supervised with its REAL outcome, plus a within-
moment ranking term on the success logit. Two item sources in one shuffled set:
  gen5b  one branch MOMENT (branch_dataset): context from the moment snapshot, candidates = the K branch commands described given the
         command IN FORCE at the moment (so the current command enters through the descriptors: is_keep, transition cost),
         state = real decision_state (time since the last switch, any switch yet). Labels: success, fell, contact, stall,
         progress; G1 uses success_tilt / fell_tilt (tilt > 1.2 rad = fall, the 10-26 decision).
  gen4   one situation x controller (gen4_option_dataset), STATE0 (walking stock, no switch); every SCHOOL row is dropped (all 519 are
         within 3 m of the SC2A test course, the held-out rule).
Groups for eval: A = switch distance from a stock walk, B = held posture (keep / stand up / other / slow), C = no hazard
(label null_*), G4 = gen4 options.
--g5-manifest (joint, 10-27): adds gen5 RANDOM per-tick rows as SINGLE-candidate items (the command actually sent, real state),
  labelled Monte Carlo from the episode (success = episode reached the goal, fell = episode fell (tilt relabel already in the
  manifest), contact = any hazard contact in the episode, stall = stall / timeout end; progress unlabelled). School env and the
  held-out base are dropped; every --g5-every-th tick; train split only (never evaluated). Each batch is --g5-frac gen5 items.
--init <model.pt> (ft): start from a trained gen5 model (missing IQL heads allowed), --norm-from-init reuses its normalisation.

    PY=$XFEAS_PY
    $PY xfeas/train/trainer.py --out data/gen_train/gen5b/runs/g1c_ho --holdout-base g1c         --gen4-ckpt data/gen_train/gen4/full/f_g1c/ckpt/xfeas_gen4_ens.pt
    $PY xfeas/train/trainer.py --eval-only --ckpt <out>/model.pt --eval-split test
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, json, os, sys, time, zlib
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "gen_train", "gen5b", "manifest.jsonl"))
ap.add_argument("--g4-manifest", default=os.path.join(ROOT, "data", "gen_train", "gen5g4", "manifest.jsonl"))
ap.add_argument("--no-g4", action="store_true")
ap.add_argument("--drop-env", default="school")
ap.add_argument("--drop-sid", default="wa5_env_overhang_spotp_54170036")  # quarantined (audit FAIL: dark frames)
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
ap.add_argument("--g5-manifest", default="")
ap.add_argument("--g5-every", type=int, default=4)
ap.add_argument("--g5-frac", type=float, default=0.5)
ap.add_argument("--init", default="")
ap.add_argument("--norm-from-init", action="store_true")
args = ap.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(4)
import branch_dataset as DB  # gen5 paths, canonical joints, robot_desc5
import dataset5 as D5
import feasibility_model as M5
import gen4_common as G4C

HEAD_OF = dict(value=0, fall=1, contact=2, stall=3)  # model head <- y column; y[..., 4] = progress
LOSS_W = dict(value=1.0, fall=0.5, contact=0.3, stall=0.3, progress=0.2)
STATE0 = np.array([1.0, 0.0], np.float32)
GROUPS = ("A", "B", "C", "G4")


def group_of(r):
    if r["src"] == "g5":
        return "G5"
    if r["src"] == "g4":
        return "G4"
    if r["label"].startswith("null"):
        return "C"
    return r["group"][:1]


def load_rows(holdout):
    rows = []
    for l in open(args.manifest):
        r = json.loads(l)
        if args.drop_sid and r["sid"].startswith(args.drop_sid):
            continue
        r["src"] = "g5b"
        rows.append(r)
    if not args.no_g4:
        for l in open(args.g4_manifest):
            r = json.loads(l)
            if args.drop_env and r["env"] == args.drop_env:
                continue
            r["src"] = "g4"
            rows.append(r)
    for r in rows:
        r["grp"] = group_of(r)
        if holdout and r["base"] == holdout:
            r["split"] = "test"
    if args.g5_manifest:
        rows += load_g5(holdout)
    return rows


def load_g5(holdout):
    """gen5 random per-tick rows (train split only), Monte Carlo episode labels"""
    raw = [json.loads(l) for l in open(args.g5_manifest)]
    contact = defaultdict(float)
    for r in raw:
        contact[r["sid"]] = max(contact[r["sid"]], float(r.get("y_contact") or 0.0))
    out = []
    for r in raw:
        if (
            r["split"] != "train"
            or r["base"] == holdout
            or (args.drop_env and r["env"] == args.drop_env)
            or r["tick"] % args.g5_every
        ):
            continue
        r["src"] = "g5"
        r["grp"] = "G5"
        r["y"] = [
            [
                float(r["end"] == "success"),
                float(r["end"] == "fell"),
                contact[r["sid"]],
                float(r["end"] in ("stall", "timeout")),
                float("nan"),
            ]
        ]
        out.append(r)
    return out


def make_norm(rows):
    """gen4 context normalisation on the gen4 training rows; controller-fingerprint stats over every training controller"""
    tr4 = [
        dict(
            obs=r["obs"],
            spec=r["spec"],
            probe=r["probe"],
            ctrl_id=r["ctrl_id"],
            base=r["base"],
            robot=r["robot"],
            split="train",
        )
        for r in rows
        if r["src"] == "g4" and r["split"] == "train"
    ]
    norm = D5.DG.compute_norm(tr4)
    fps = {}
    for r in rows:
        if r["split"] == "train" and r["probe"] and r["ctrl_id"] not in fps:
            fps[r["ctrl_id"]] = G4C.fp_vector(json.load(open(r["probe"])))
    Fv = np.stack([v[0] for v in fps.values()])
    Pm = np.stack([v[1] for v in fps.values()])
    mu = np.array([Fv[Pm[:, i], i].mean() if Pm[:, i].any() else 0.0 for i in range(G4C.N_FP)])
    sd = np.array([max(Fv[Pm[:, i], i].std(), 0.02) if Pm[:, i].sum() > 1 else 1.0 for i in range(G4C.N_FP)])
    norm.update(fp_mean=mu.tolist(), fp_std=sd.tolist(), n_fp_ctrls=len(fps), fp_ctrls=sorted(fps))
    return norm


class Items(Dataset):
    def __init__(self, rows, norm, cfg, augment=False, seed=0):
        self.rows, self.norm, self.cfg, self.augment = rows, norm, cfg, augment
        self.aug = dict(D5.DG.DEFAULT_AUG)
        self._rng = np.random.default_rng(seed)
        self._json_c, self._ptr = {}, {}
        self.cache = D5.EpCache()

    def __len__(self):
        return len(self.rows)

    def _json(self, p):
        if p not in self._json_c:
            self._json_c[p] = json.load(open(p))
        return self._json_c[p]

    def _ctx(self, o, r):
        return D5.DG.build_context(
            o,
            self._json(r["spec"]),
            self._json(r["probe"]),
            D5.robot_desc5(r["robot"]),
            self.norm,
            self.cfg,
            train=self.augment,
            rng=self._rng,
            aug=self.aug,
            ptrace_cache=self._ptr,
        )

    def __getitem__(self, j):
        r = self.rows[j]
        if r["src"] == "g4":
            on = np.load(r["obs"], allow_pickle=False)
            ctx, feas_raw = self._ctx({k: on[k] for k in on.files}, r)
            cand = np.asarray(r["X"], np.float32)
            y = np.asarray(r["y"], np.float32)
            stock = np.asarray(r["stock"], bool)
            state = STATE0.copy()
        elif r["src"] == "g5":
            z = self.cache.get(r["ep"])
            k = r["tick"]
            ctx, feas_raw = self._ctx(D5.obs_at(z, k), r)
            X = np.asarray(z["X"][k], np.float32).copy()
            X[:, DB.DS.KI["is_keep"]] = 0.0
            X[0, DB.DS.KI["is_keep"]] = 1.0
            X[:, DB.DS.KI["is_nominal"]] = 0.0
            X[1, DB.DS.KI["is_nominal"]] = 1.0
            cand = X[int(z["chosen"][k])][None]
            y = np.asarray(r["y"], np.float32)
            stock = np.zeros(1, bool)
            state = D5.decision_state(z, k)
        else:
            z = self.cache.get(r["br"])
            i = r["moment"]
            ks = r["branches"]
            ctx, feas_raw = self._ctx(DB.obs_at5b(z, i), r)
            X = np.asarray(z["X"][i], np.float32).copy()  # same convention as branch_dataset.item_np
            X[:, DB.DS.KI["is_keep"]] = 0.0
            X[0, DB.DS.KI["is_keep"]] = 1.0
            X[:, DB.DS.KI["is_nominal"]] = 0.0
            X[1, DB.DS.KI["is_nominal"]] = 1.0
            cand = X[np.asarray(z["b_cand"])[ks].astype(np.int64)]
            g1 = r["robot"] == "g1"
            col = lambda k: np.asarray(z[f"y_{k}"])[ks].astype(np.float32)
            y = np.stack(
                [
                    col("success_tilt" if g1 else "success"),
                    col("fell_tilt" if g1 else "fell"),
                    col("contact"),
                    col("stall"),
                    col("progress"),
                ],
                1,
            )
            stock = np.asarray([c == "stock" for c in r["cls"]], bool)
            state = DB.decision_state(z, i)
        t = torch.from_numpy
        out = {k: t(np.asarray(v)) for k, v in ctx.items()}
        fr = feas_raw.astype(np.float32) if feas_raw is not None else np.full(16, np.nan, np.float32)
        out.update(feas_raw=t(fr), state=t(np.asarray(state, np.float32)), cand=t(cand), y=t(y), stock=t(stock), idx=j)
        return out


def collate(batch):
    B = len(batch)
    K = max(b["rgb"].shape[0] for b in batch)
    N = max(b["cand"].shape[0] for b in batch)
    out = {}
    rgb = torch.zeros(B, K, 3, 224, 224, dtype=torch.uint8)
    dep = torch.zeros(B, K, 1, 224, 224)
    fmask = torch.zeros(B, K, dtype=torch.bool)
    fage = torch.zeros(B, K)
    for i, b in enumerate(batch):
        k = b["rgb"].shape[0]
        rgb[i, :k] = b["rgb"]
        dep[i, :k] = b["depth"]
        fmask[i, :k] = True
        fage[i, :k] = b["fage"]
    out.update(rgb=rgb, depth=dep, fmask=fmask, fage=fage)
    for k in M5.CTX_KEYS:
        if k not in out:
            out[k] = torch.stack([b[k] for b in batch])
    out["feas_raw"] = torch.stack([b["feas_raw"] for b in batch])
    out["state"] = torch.stack([b["state"] for b in batch])
    cand = torch.zeros(B, N, batch[0]["cand"].shape[1])
    y = torch.full((B, N, 5), float("nan"))
    cm = torch.zeros(B, N, dtype=torch.bool)
    st = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["cand"].shape[0]
        cand[i, :n] = b["cand"]
        y[i, :n] = b["y"]
        cm[i, :n] = True
        st[i, :n] = b["stock"]
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
    cm = b["cand_mask"]
    y = b["y"]
    M = (bm[:, :, None] & cm[None]).float()  # (K,B,N)
    L, parts = 0.0, {}
    for h, j in HEAD_OF.items():
        yy = y[..., j]
        ok = M * torch.isfinite(yy)[None].float()
        l = F.binary_cross_entropy_with_logits(
            o[h], torch.nan_to_num(yy, nan=0.0)[None].expand_as(o[h]), reduction="none"
        )
        l = (l * ok).sum() / ok.sum().clamp(min=1)
        L = L + LOSS_W[h] * l
        parts[h] = float(l)
    yy = y[..., 4]
    ok = M * torch.isfinite(yy)[None].float()
    l = F.smooth_l1_loss(
        o["progress"], torch.nan_to_num(yy, nan=0.0)[None].expand_as(o["progress"]), beta=0.1, reduction="none"
    )
    l = (l * ok).sum() / ok.sum().clamp(min=1)
    L = L + LOSS_W["progress"] * l
    parts["progress"] = float(l)
    if args.rank_w > 0:  # within-moment: successful commands above failed ones
        ys = torch.nan_to_num(y[..., 0], nan=-1.0)
        pos = (ys == 1) & cm
        neg = (ys == 0) & cm
        pm = (pos[:, :, None] & neg[:, None, :])[None] & bm[:, :, None, None]
        if pm.any():
            d = o["value"][:, :, :, None] - o["value"][:, :, None, :]
            l = F.softplus(-d[pm]).mean()
            L = L + args.rank_w * l
            parts["rank"] = float(l)
    return L, parts


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(int(torch.initial_seed()) % (1 << 31))


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train():
    dev = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = dict(
        backbone="dinov3_s",
        holdout_base=args.holdout_base,
        k_ens=args.k_ens,
        use_margin=True,
        use_fp=True,
        use_hist=True,
        use_map=True,
        use_frames_hist=True,
        k_hist=4,
        data="gen5b+gen5g4" if not args.no_g4 else "gen5b",
        drop_env=args.drop_env,
        drop_sid=args.drop_sid,
        gen4_ckpt=os.path.abspath(args.gen4_ckpt) if args.gen4_ckpt else "",
        loss_w=LOSS_W,
        rank_w=args.rank_w,
        kappa=args.kappa,
        lam=args.lam,
        epochs=args.epochs,
        bs=args.bs,
    )
    rows = load_rows(args.holdout_base)
    tr = [r for r in rows if r["split"] == "train"]
    if args.max_rows:
        tr = tr[: args.max_rows // 2] + [r for r in tr if r["src"] == "g5b"][: args.max_rows // 2]
    os.makedirs(args.out, exist_ok=True)
    np_ = os.path.join(args.out, "norm.json")
    if os.path.exists(np_):
        norm = json.load(open(np_))
    elif args.init and args.norm_from_init:
        norm = torch.load(args.init, map_location="cpu", weights_only=False)["norm"]
        json.dump(norm, open(np_, "w"))
    else:
        norm = make_norm(rows)
        json.dump(norm, open(np_ + ".tmp", "w"))
        os.replace(np_ + ".tmp", np_)
    assert not (set(norm["fp_ctrls"]) & {r["ctrl_id"] for r in rows if r["split"] == "test"}), (
        "held-out controllers in the norm"
    )
    assert not any(r.get("env") == "school" for r in tr) or not args.drop_env, "school rows in training"
    m = M5.build(cfg, pretrained=True).to(dev)
    if args.gen4_ckpt:
        lo, pa, sk, mi = M5.load_gen4(m, args.gen4_ckpt)
        print(f"[trainer] context from {args.gen4_ckpt}: {len(lo)} loaded, {len(pa)} widened", flush=True)
    if args.init:
        mi = m.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"], strict=False)
        bad = [
            k for k in mi.missing_keys if not k.startswith("ctx.vision.backbone") and not (".pi." in k or ".vs." in k)
        ]
        assert not bad and not mi.unexpected_keys, (bad[:5], mi.unexpected_keys[:5])
        print(f"[trainer] init from {args.init} ({len(mi.missing_keys)} missing: backbone / IQL heads)", flush=True)
    cfg.update(
        init=os.path.abspath(args.init) if args.init else "",
        g5_manifest=args.g5_manifest,
        g5_every=args.g5_every,
        g5_frac=args.g5_frac,
    )
    ctx_p = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("ctx.")]
    mem_p = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("ctx.")]
    opt = torch.optim.AdamW([dict(params=ctx_p, lr=args.lr_ctx), dict(params=mem_p, lr=args.lr)], weight_decay=0.05)
    sbs = [i for i, r in enumerate(tr) if r["src"] != "g5"]
    g5i = [i for i, r in enumerate(tr) if r["src"] == "g5"]
    n5 = int(round(args.bs * args.g5_frac)) if g5i else 0
    spe = len(sbs) // (args.bs - n5)
    total = args.max_steps or args.epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr_ctx, args.lr], total_steps=max(total, 20), pct_start=0.1)
    step, ep, hist, t_prev = 0, 0, [], 0.0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last) and not args.no_resume:
        d = torch.load(last, map_location="cpu", weights_only=False)
        assert d["total"] == total and d["cfg"]["holdout_base"] == args.holdout_base
        m.load_state_dict(d["model"], strict=False)
        opt.load_state_dict(d["opt"])
        sched.load_state_dict(d["sched"])
        step, ep, hist, t_prev = d["step"], d["epoch"], d["history"], d["sec"]
        print(f"[trainer] RESUMED step {step}/{total} epoch {ep}", flush=True)
    ds = Items(tr, norm, cfg, augment=True, seed=args.seed + ep)
    cnt = defaultdict(int)
    for r in tr:
        cnt[f"{r['grp']}:{r['base']}" if r["src"] != "g5" else "G5"] += 1
    print(
        f"[trainer] fold '{args.holdout_base}': {len(tr)} train items ({dict(sorted(cnt.items()))}), "
        f"{sum(len(r['y'] if r['src'] in ('g4', 'g5') else r['branches']) for r in tr)} candidates; g5 {n5}/{args.bs} per batch; {spe} steps/epoch, total {total}",
        flush=True,
    )
    t0 = time.time()
    while step < total:
        g = torch.Generator().manual_seed(args.seed * 1000 + ep)
        rs = np.random.default_rng(args.seed * 1000 + ep)
        ps = rs.permutation(sbs)
        nb = args.bs - n5
        bat = [
            list(ps[j * nb : (j + 1) * nb]) + (list(rs.choice(g5i, n5, replace=False)) if n5 else [])
            for j in range(spe)
        ]
        dl = DataLoader(
            ds,
            batch_sampler=[[int(x) for x in b_] for b_ in bat],
            num_workers=args.workers,
            collate_fn=collate,
            worker_init_fn=worker_init,
            pin_memory=True,
            prefetch_factor=4,
        )
        m.train()
        acc, n, te = 0.0, 0, time.time()
        for b in dl:
            b = to(b, dev)
            bm = boot_mask([tr[i]["key"] for i in b["idx"].tolist()], args.k_ens, args.boot).to(dev)
            o = run(m, b)
            loss, parts = loss_fn(o, b, bm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctx_p + mem_p, 1.0)
            opt.step()
            sched.step()
            step += 1
            n += 1
            acc += float(loss)
            if step % args.log_every == 0 or step <= 2:
                rec = dict(
                    step=step,
                    ep=ep,
                    loss=round(float(loss), 4),
                    **{k: round(v, 4) for k, v in parts.items()},
                    sec=round(t_prev + time.time() - t0, 1),
                )
                hist.append(rec)
                print("[trainer] " + json.dumps(rec), flush=True)
            if step >= total:
                break
        ep += 1
        print(f"[trainer] ep{ep - 1} mean loss {acc / max(n, 1):.4f} ({time.time() - te:.0f}s)", flush=True)
        save_atomic(
            dict(
                model={k: v for k, v in m.state_dict().items() if not k.startswith("ctx.vision.backbone")},
                opt=opt.state_dict(),
                sched=sched.state_dict(),
                step=step,
                epoch=ep,
                total=total,
                cfg=cfg,
                history=hist,
                sec=t_prev + time.time() - t0,
            ),
            last,
        )
    path = os.path.join(args.out, "model.pt")
    save_atomic(
        dict(
            format=M5.FORMAT + "_g5b",
            cfg=cfg,
            norm=norm,
            history=hist,
            saved=time.strftime("%Y-%m-%d %H:%M"),
            model={k: v.detach().cpu() for k, v in m.state_dict().items() if not k.startswith("ctx.vision.backbone")},
        ),
        path,
    )
    print(f"[trainer] saved {path}", flush=True)


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarise(R):
    """R: per-moment dicts with y (success per candidate), p, U, stock idx, keep idx, cls"""
    rec = defaultdict(list)
    allp, ally = [], []
    pair_ok = pair_n = 0
    picks = defaultdict(int)
    for s in R:
        y, a, st = s["y"], int(np.argmax(s["U"])), s["st"]
        rec["pick"].append(y[a])
        rec["stock"].append(y[st])
        rec["oracle"].append(y.max())
        rec["random"].append(y.mean())
        rec["pick_is_stock"].append(float(a == st))
        picks[s["cls"][a]] += 1
        if s["keep"] is not None:
            rec["keep"].append(y[s["keep"]])
        if y[st] == 0:
            rec["capture"].append(y[a])
            if y.max() == 1:
                rec["capture_possible"].append(y[a])
        else:
            rec["harm"].append(1.0 - y[a])
        allp += list(s["p"])
        ally += list(y)
        for u in range(len(y)):
            for v in range(len(y)):
                if y[u] == 1 and y[v] == 0:
                    pair_n += 1
                    pair_ok += float(s["p"][u] > s["p"][v])
    mean = lambda k: round(float(np.mean(rec[k])), 3) if rec[k] else None
    return dict(
        n=len(R),
        pick=mean("pick"),
        stock=mean("stock"),
        keep=mean("keep"),
        oracle=mean("oracle"),
        random=mean("random"),
        capture=mean("capture"),
        n_capture=len(rec["capture"]),
        capture_when_possible=mean("capture_possible"),
        n_capture_possible=len(rec["capture_possible"]),
        harm=mean("harm"),
        n_harm=len(rec["harm"]),
        picks_stock=mean("pick_is_stock"),
        pick_cls=dict(sorted(picks.items())),
        auroc=round(auroc(ally, allp), 3),
        pair_acc=round(pair_ok / max(pair_n, 1), 3),
        n_pairs=pair_n,
    )


@torch.no_grad()
def evaluate(ckpt, split):
    dev = args.device
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, norm = d["cfg"], d["norm"]
    args.drop_env, args.drop_sid = cfg.get("drop_env", args.drop_env), cfg.get("drop_sid", args.drop_sid)
    m = M5.build(cfg, pretrained=True).to(dev).eval()
    m.load_state_dict(d["model"], strict=False)
    args.g5_manifest = ""
    rows = [r for r in load_rows(cfg["holdout_base"]) if r["split"] == split]
    if args.max_rows:
        rows = rows[: args.max_rows]
    dl = DataLoader(Items(rows, norm, cfg), batch_size=16, shuffle=False, num_workers=args.workers, collate_fn=collate)
    R = []
    for b in dl:
        b = to(b, dev)
        o = run(m, b)
        ps = torch.sigmoid(o["value"])
        pf = torch.sigmoid(o["fall"]).mean(0)
        U = ps.mean(0) - cfg.get("kappa", args.kappa) * ps.std(0) - cfg.get("lam", args.lam) * pf
        for j, i in enumerate(b["idx"].tolist()):
            r = rows[i]
            n = int(b["cand_mask"][j].sum())
            y = b["y"][j, :n, 0].cpu().numpy()
            stock = b["stock"][j, :n].cpu().numpy()
            cls = r["cls"] if r["src"] == "g5b" else [o_["kind"] for o_ in r["opts"]]
            R.append(
                dict(
                    r=r,
                    y=y,
                    U=U[j, :n].cpu().numpy(),
                    p=ps.mean(0)[j, :n].cpu().numpy(),
                    st=int(np.argmax(stock)) if stock.any() else 0,
                    keep=cls.index("keep") if "keep" in cls else None,
                    cls=cls,
                )
            )
    res = dict(split=split, holdout=cfg["holdout_base"], ckpt=os.path.abspath(ckpt), all=summarise(R))
    for g in GROUPS:
        sub = [s for s in R if s["r"]["grp"] == g]
        if sub:
            res[g] = summarise(sub)
    for key, name in (
        (lambda s: s["r"]["label"], "A_by_dist"),
        (lambda s: s["r"]["label"], "B_by_point"),
        (lambda s: s["r"]["base"], "by_base"),
    ):
        grp = name[0] if name[0] in "AB" else None
        sub = [s for s in R if grp is None or s["r"]["grp"] == grp]
        res[name] = {k: summarise([s for s in sub if key(s) == k]) for k in sorted({key(s) for s in sub})}
    od = os.path.dirname(os.path.abspath(ckpt))
    json.dump(res, open(os.path.join(od, f"select_{split}.json"), "w"), indent=1)
    short = {
        k: {
            kk: v[kk]
            for kk in (
                "n",
                "pick",
                "stock",
                "keep",
                "oracle",
                "random",
                "capture_when_possible",
                "harm",
                "pair_acc",
                "pick_cls",
            )
        }
        for k, v in res.items()
        if isinstance(v, dict) and "pick" in v
    }
    print(json.dumps(short, indent=1))
    return res


if __name__ == "__main__":
    if args.eval_only:
        evaluate(args.ckpt, args.eval_split)
    else:
        train()
