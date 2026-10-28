#!/usr/bin/env python
"""
XFeas-Gen5 model (2026-10-26; design analysis/GEN5_DESIGN.md 4). Scores the controller's OWN commands every 0.25 s.

Context (encoded ONCE per tick; reused from gen4 = model_gen4.XFeasGen3, weights loaded from the gen4 release):
  vision (frozen DINOv3-S, 4 frames) + clearance map + 2 s proprio + embodiment + feas + CapabilityEncoder (spec, self-test
  fingerprint / ops / trace, h(t)) -> memory tokens.
Candidates (new): describe5 descriptor (35: measured effect of the command, normalised raw command, flags incl. is_current,
  measured transition current -> candidate) + decision state (time since the last switch) -> query. The raw current command
  is NOT an input (mode indices / heights in metres are not comparable across controllers); the menu row flagged is_current
  carries its measured effect.
Members: K_ENS independent candidate encoders + decoders + heads over the SHARED context (cheap ensemble: the context is
  the expensive part; disagreement between members = uncertainty about a command, the pessimism term of the selector).
Fit margins (cfg use_margin; margin_feats): candidate body top / width vs the free height / gap ahead + time to reach it.
IQL (2026-10-26, after the Stage B readout showed argmax-over-the-menu exploits value errors on untried commands):
  'value' = Q(s, a) (logit of P(eventual success)), 'v_state' = V(s) (expectile of Q over the commands the data tried),
  'pi' = the CHOOSER (per-candidate logit, softmax over the menu), trained by advantage-weighted imitation of the data
  collector's commands; on the robot the pick is argmax of pi, never argmax of Q. Train: train5.py --stage I.
Heads per candidate (Stage A, supervised on the chosen candidate): fall, stall, contact (logits), progress (m over 1.5 s).
  Stage B adds 'value' (logit of P(eventual success)); the head exists from the start and is untrained in Stage A.

    python scripts/gen5/model5.py            # self-test (random inputs, cnn backbone)
"""
import os, sys
import torch
import torch.nn as nn

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import canon5                                                # BEFORE the gen4 model: canonical joint sizes
for _p in (os.path.join(THIS, "..", "gen4"), os.path.join(THIS, "..", "gen_train"), os.path.join(THIS, "..", "xembody")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import model_gen4 as M4
import model_xe as MX
import describe5 as DS

FORMAT = "xfeas_gen5_v1"
CLASS_HEADS = ("fall", "stall", "contact", "value")
REG_HEADS = ("progress",)
STAGE_A = ("fall", "stall", "contact", "progress")
K_ENS = 5
FEAS_KEYS = ("h_free_min", "x_at_h_free", "gap_w_min", "x_at_gap", "h_ground_max", "stand_minus_hfree", "fits_under_stock",
             "fits_under_best", "fits_fwd_stock", "fits_fwd_best", "fits_side", "margin_top_best", "margin_w_fwd", "margin_w_side",
             "lateral_capable", "any_fit")                    # feas v0.4 order (episode npz feas_keys)
FI = {k: i for i, k in enumerate(FEAS_KEYS)}
N_MARGIN = 8
N_STATE = 2                                                  # time since the last switch (/ T_SWITCH_CAP, capped), any switch yet
T_SWITCH_CAP = 5.0
CTX_KEYS = M4.CTX_KEYS
GEN4_CKPT = os.path.join(THIS, "..", "..", "data", "xe_train", "gen4", "xfeas_gen4_ens.pt")


def margin_feats(cand, feas_raw):
    """per-candidate FIT MARGINS (2026-10-26, after the Stage A check): the candidate's measured body size (self-test e_top /
    e_width) against the free space perceived ahead (feas: h_free_min / gap_w_min, 9.9 = none seen), and the time to reach
    that constraint at the candidate's measured speed. Plain differences of two inputs the model already has; no threshold.
    cand (B,N,F_DESC), feas_raw (B,16) -> (B,N,N_MARGIN)"""
    B, N, _ = cand.shape
    f = torch.nan_to_num(feas_raw.float(), nan=9.9)[:, None].expand(B, N, -1)
    top, wid = cand[..., DS.KI["e_top"]], cand[..., DS.KI["e_width"]]
    v = cand[..., DS.KI["e_vx"]].clamp(min=0.05)
    out = []
    for h, x, size in (("h_free_min", "x_at_h_free", top), ("gap_w_min", "x_at_gap", wid)):
        has = (f[..., FI[h]] < 9.0) & (f[..., FI[x]] >= 0)
        m = torch.where(has, (f[..., FI[h]] - size).clamp(-1, 1), torch.ones_like(size))
        d = torch.where(has, f[..., FI[x]].clamp(0, 5), torch.full_like(size, 5.0))
        out += [m, has.float(), d / 5.0, (d / v).clamp(max=10.0) / 10.0]
    return torch.stack(out, -1)


class Member(nn.Module):
    """one ensemble member: candidate encoder + decoder over the shared context + heads"""

    def __init__(self, d=192, heads=4, dec_layers=2, use_margin=True):
        super().__init__()
        self.use_margin = use_margin
        self.cand = MX.mlp(DS.F_DESC + N_STATE + (N_MARGIN if use_margin else 0), d, d)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, dec_layers)
        self.cls = nn.Linear(d, len(CLASS_HEADS))
        self.reg = nn.Linear(d, len(REG_HEADS))
        self.pi = nn.Linear(d, 1)                            # IQL chooser: per-candidate policy logit (softmax over the menu)
        self.vs = nn.Sequential(nn.Linear(d + N_STATE, d), nn.GELU(), nn.Linear(d, 1))   # IQL state value V(s) (no candidate)

    def forward(self, mem, pad, cand, state, feas_raw=None):
        B, N, _ = cand.shape
        parts = [cand, state[:, None].expand(B, N, -1)]
        if self.use_margin:
            parts.append(margin_feats(cand, feas_raw))
        q = self.cand(torch.cat(parts, -1)).reshape(B * N, 1, -1)
        M = mem.shape[1]
        memx = mem[:, None].expand(B, N, M, mem.shape[-1]).reshape(B * N, M, -1)
        padx = pad[:, None].expand(B, N, M).reshape(B * N, M)
        h = self.dec(q, memx, memory_key_padding_mask=padx)[:, 0]
        c, r = self.cls(h).reshape(B, N, -1), self.reg(h).reshape(B, N, -1)
        out = {k: c[..., i] for i, k in enumerate(CLASS_HEADS)}
        out.update({k: r[..., i] for i, k in enumerate(REG_HEADS)})
        out["pi"] = self.pi(h).reshape(B, N)
        keep = (~pad).float()[..., None]
        pooled = (mem * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        out["v_state"] = self.vs(torch.cat([pooled, state], -1)).expand(B, N)    # same for every row (state only)
        return out


class XFeasGen5(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.ctx = M4.build(cfg, pretrained=pretrained)     # gen4 context encoder; its candidate / decoder / heads are unused
        for n in ("cand", "ach", "ach_in", "macro", "dec", "cls", "reg"):
            if hasattr(self.ctx, n):
                delattr(self.ctx, n)
        self.members = nn.ModuleList([Member(use_margin=cfg.get("use_margin", False)) for _ in range(cfg.get("k_ens", K_ENS))])

    def encode(self, **ctx_in):
        mem, pad, _ = self.ctx.encode_context(**ctx_in)
        return mem, pad

    def score(self, enc, cand, state, feas_raw=None):
        """enc = encode(...); cand (B,N,F_DESC); state (B,N_STATE); feas_raw (B,16) raw feas (fit margins) -> dict of (K,B,N)"""
        mem, pad = enc
        outs = [m(mem, pad, cand, state, feas_raw) for m in self.members]
        return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}

    def forward(self, cand, state, feas_raw=None, **ctx_in):
        return self.score(self.encode(**ctx_in), cand, state, feas_raw)


def build(cfg, pretrained=True):
    return XFeasGen5(cfg, pretrained=pretrained)


def load_gen4(model, path=GEN4_CKPT, member=0):
    """gen4 release weights -> the context encoder. The h(t) / probe-trace patch MLPs (cap.hist.0 / cap.ptr.0) take frames
    that grew by the 8 new joints' q / dq: the old columns are copied to their new positions, the new ones start at 0.
    -> (loaded, partially_copied, skipped) key lists"""
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd4 = d["members"][member]
    own = model.ctx.state_dict()
    fmap = canon5.frame_index_map()
    Fo, Fn = canon5.F_FRAME_GEN4, canon5.F_FRAME
    loaded, partial, skipped = [], [], []
    new = {}
    for k, v in sd4.items():
        if k not in own:
            skipped.append(k); continue
        if own[k].shape == v.shape:
            new[k] = v; loaded.append(k); continue
        if k in ("cap.hist.0.weight", "cap.ptr.0.weight"):
            P = M4.P_HIST if k.startswith("cap.hist") else M4.P_PROBE
            assert v.shape[1] == P * Fo + P and own[k].shape[1] == P * Fn + P, (k, v.shape, own[k].shape)
            w = torch.zeros_like(own[k])
            for p in range(P):
                for fo, fn in enumerate(fmap):
                    w[:, p * Fn + fn] = v[:, p * Fo + fo]
            w[:, P * Fn:] = v[:, P * Fo:]                       # per-frame valid flags
            new[k] = w; partial.append(k); continue
        skipped.append(k)
    missing = model.ctx.load_state_dict(new, strict=False)
    assert all(k.startswith("vision.backbone") for k in missing.unexpected_keys), missing.unexpected_keys
    return loaded, partial, skipped, [k for k in missing.missing_keys if not k.startswith("vision.backbone")]


def _selftest():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = dict(backbone="cnn", use_margin=True)
    m = build(cfg, pretrained=False).to(dev).eval()
    lo, pa, sk, mi = load_gen4(m)
    print(f"[gen5] gen4 weights: {len(lo)} loaded, {len(pa)} widened {pa}, {len(sk)} skipped (gen4 heads / backbone), "
          f"not from gen4: {[k for k in mi][:6]}{'...' if len(mi) > 6 else ''} ({len(mi)})")
    B, N = 3, 7
    x = M4.dummy_ctx(B, dev, J=canon5.NJ)
    cand = torch.randn(B, N, DS.F_DESC, device=dev); st = torch.randn(B, N_STATE, device=dev)
    fr = torch.full((B, len(FEAS_KEYS)), 9.9, device=dev); fr[0, FI["h_free_min"]] = 0.8; fr[0, FI["x_at_h_free"]] = 1.0
    with torch.no_grad():
        o = m(cand=cand, state=st, feas_raw=fr, **x)
        o2 = m(cand=cand[:, :1], state=st, feas_raw=fr, **x)    # scoring one candidate == its row in the full menu
    mf = margin_feats(cand, fr)
    assert mf.shape == (B, N, N_MARGIN) and bool((mf[1:, :, 1] == 0).all()) and bool((mf[0, :, 1] == 1).all())
    assert all(v.shape == (K_ENS, B, N) for v in o.values()), {k: v.shape for k, v in o.items()}
    d1 = max((o[k][:, :, :1] - o2[k]).abs().max().item() for k in o)
    assert d1 < 1e-4, d1
    sp = (o["fall"].std(0)).mean().item()
    assert sp > 1e-4, "members identical"
    n_ctx = sum(p.numel() for p in m.ctx.parameters()) / 1e6; n_mem = sum(p.numel() for p in m.members.parameters()) / 1e6
    print(f"[gen5] outputs {sorted(o)} shape (K, B, N) = {tuple(o['fall'].shape)}; menu-independence diff {d1:.1e}; member spread {sp:.3f}")
    print(f"[gen5] params: context {n_ctx:.2f}M (cnn backbone here), {K_ENS} members {n_mem:.2f}M")
    print("[gen5] selftest OK")


if __name__ == "__main__":
    _selftest()
