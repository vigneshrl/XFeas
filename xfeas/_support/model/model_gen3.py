#!/usr/bin/env python
"""
XFeasGen3 (gen3 model agent, 2026-10-17): a decision layer that adapts ZERO-SHOT to a new controller of the same robot
(G1) from its command spec + a <= 30 s self-test probe + the in-context history h(t). Contract: analysis/GEN3_SPEC.md;
design: analysis/GEN3_MODEL.md.

Context (encoded ONCE per situation x controller; all of its candidates are scored against it):
  * perception as gen2: frame memory (decision frame 64 tokens + older frames 16 tokens each, absent frames masked),
    clearance map (cmap.py, 60 tokens), 2 s proprio (StateEncoder on the CANONICAL 29 G1 joints);
  * embodiment: body + canonical joint statics (max_cmd zeroed: it is a controller property);
  * feas (16, spec 8): one token;
  * CAPABILITY ENCODER (the in-context part): a transformer over
        spec tokens (roles / ranges / modes; never names)            group 0
        fingerprint (32 + present mask)                              group 1   } "probe" inputs
        probe operating-point table (<= 24 ops)                       group 2   }
        probe trace patches (32 s at 10 Hz, 8-frame patches)          group 3   }
        history h(t) patches (6 s at 10 Hz, 5-frame patches)          group 4
    -> N_CAP capability tokens (readouts). Absent groups are masked (key padding), never zero-filled into attention.
Candidates (env_act (16,) + mask + macro + requested-vs-feas margins + driver outputs, gen3_common.cand_feats):
  * ACHIEVABLE-ENVELOPE head: MLP(capability summary, candidate) -> what THIS controller will achieve for that request
    (v, top, pitch in phase B; |vy| in the side phase). Supervised from the rollouts' ach_trace; it is the head that forces
    the capability tokens to carry controller response.
  * risk decoder: query = candidate embedding + embedding of the (detached) predicted achievable envelope; memory =
    perception + embodiment + feas + capability tokens -> P(success), fall, stall, collision, slip; progress, t_norm,
    env_dev, tilt_margin, cost.
Ablation switches (cfg): use_fp (fingerprint + ops + probe trace), use_hist (h(t)), ranges_only (spec without modes),
use_feas_probe (feas idx 5..15), use_drv (driver outputs per candidate). desc_only = gen2b-style: all of them off.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
import model_xe as MX
import model_gen2 as MG2
import cmap as CM
import gen3_common as G

FORMAT = "xfeas_gen3_v1"
CLASS_HEADS = ("success", "fall", "stall", "collision", "slip")
REG_HEADS = ("progress", "t_norm", "env_dev", "tilt_margin", "cost")
K_HIST = 4
N_CAP = 4
P_HIST, P_PROBE = 5, 8
META_FIELDS = (
    "no_hist_frames",
    "n_frames",
    "map_used_prior",
    "cam_h",
    "map_present",
    "hist_span",
    "has_hist",
    "has_probe",
    "has_feas",
)
N_META = len(META_FIELDS)
ABLATIONS = {
    "full": {},
    "no_fp": dict(use_fp=False),
    "no_hist": dict(use_hist=False),
    "neither": dict(use_fp=False, use_hist=False),
    "desc_only": dict(use_fp=False, use_hist=False, ranges_only=True, use_feas_probe=False, use_drv=False),
}


class CapabilityEncoder(nn.Module):
    def __init__(self, d=192, heads=4, layers=3, use_fp=True, use_hist=True):
        super().__init__()
        self.spec = MX.mlp(G.F_SPEC, d, d)
        self.fp = MX.mlp(2 * G.N_FP, d, d)
        self.ops = MX.mlp(G.F_OPS, d, d)
        self.ptr = MX.mlp(P_PROBE * G.F_FRAME + P_PROBE, d, d)
        self.hist = MX.mlp(P_HIST * G.F_FRAME + P_HIST, d, d)
        self.grp = nn.Parameter(torch.randn(1, 5, d) * 0.02)
        self.ppos = nn.Parameter(torch.zeros(1, G.T_PROBE // P_PROBE, d))
        self.hpos = nn.Parameter(torch.zeros(1, G.T_HIST // P_HIST, d))
        self.readout = nn.Parameter(torch.randn(1, N_CAP, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.use_fp, self.use_hist = use_fp, use_hist

    @staticmethod
    def _patch(x, v, p):
        B, T, Fd = x.shape
        n = T // p
        xp = torch.cat([x[:, : n * p].reshape(B, n, p * Fd), v[:, : n * p].float().reshape(B, n, p)], -1)
        return xp, v[:, : n * p].reshape(B, n, p).any(-1)

    def forward(self, spec_tok, spec_mask, fp, fp_mask, ops, ops_mask, ptrace, ptrace_valid, hist, hist_valid):
        B = spec_tok.shape[0]
        toks, keep = [self.readout.expand(B, -1, -1)], [torch.ones(B, N_CAP, dtype=torch.bool, device=spec_tok.device)]
        toks.append(self.spec(spec_tok) + self.grp[:, 0])
        keep.append(spec_mask)
        if self.use_fp:
            toks.append((self.fp(torch.cat([fp, fp_mask.float()], -1)) + self.grp[:, 1])[:, None])
            keep.append(fp_mask.any(-1, keepdim=True))
            toks.append(self.ops(ops) + self.grp[:, 2])
            keep.append(ops_mask)
            xp, vp = self._patch(ptrace, ptrace_valid, P_PROBE)
            toks.append(self.ptr(xp) + self.ppos + self.grp[:, 3])
            keep.append(vp)
        if self.use_hist:
            xh, vh = self._patch(hist, hist_valid, P_HIST)
            toks.append(self.hist(xh) + self.hpos + self.grp[:, 4])
            keep.append(vh)
        x = torch.cat(toks, 1)
        k = torch.cat(keep, 1)
        return self.tf(x, src_key_padding_mask=~k)[:, :N_CAP]


class XFeasGen3(nn.Module):
    def __init__(
        self,
        d=192,
        backbone="dinov3_s",
        pretrained=True,
        freeze=True,
        unfreeze_last=0,
        dec_layers=3,
        heads=4,
        use_map=True,
        use_frames_hist=True,
        k_hist=K_HIST,
        use_fp=True,
        use_hist=True,
        use_vision=True,
    ):
        super().__init__()
        self.use_vision = use_vision
        if use_vision:
            self.vision = MX.VisionEncoder(d, backbone, pretrained, freeze, unfreeze_last=unfreeze_last)
            self.fidx = nn.Embedding(k_hist, d)
            self.fage = MX.mlp(1, d, d)
        self.state = MX.StateEncoder(d)
        self.robot = MX.RobotEncoder(d, heads)
        self.map = MG2.MapEncoder(d)
        self.meta = MX.mlp(N_META, d, d)
        self.feas = MX.mlp(G.F_FEAS, d, d)
        self.cap = CapabilityEncoder(d, heads, use_fp=use_fp, use_hist=use_hist)
        self.cand = MX.mlp(G.F_CAND, d, d)
        self.ach = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, G.N_ACH_HEAD))
        self.ach_in = MX.mlp(G.N_ACH_HEAD, d, d)
        self.macro = nn.Embedding(len(G.MACROS), d)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, dec_layers)
        self.cls = nn.Linear(d, len(CLASS_HEADS))
        self.reg = nn.Linear(d, len(REG_HEADS))
        self.use_map, self.use_frames_hist, self.k_hist = use_map, use_frames_hist, k_hist

    def encode_frames(self, rgb, depth, fmask, fage):
        B, K = fmask.shape
        if not self.use_frames_hist:
            K = 1
        cur = self.vision(rgb[:, 0], depth[:, 0])
        d = cur.shape[-1]
        toks = [cur + self.fidx.weight[0]]
        pads = [torch.zeros(B, cur.shape[1], dtype=torch.bool, device=cur.device)]
        if K > 1:
            sel = fmask[:, 1:K].reshape(-1)
            H = torch.zeros(B * (K - 1), 16, d, device=cur.device, dtype=cur.dtype)
            if sel.any():
                r = rgb[:, 1:K].reshape(B * (K - 1), *rgb.shape[2:])[sel]
                dp = depth[:, 1:K].reshape(B * (K - 1), *depth.shape[2:])[sel]
                g = self.vision(r, dp).transpose(1, 2).reshape(-1, d, 8, 8)
                H[sel] = F.avg_pool2d(g, 2).flatten(2).transpose(1, 2)
            H = (
                H.reshape(B, K - 1, 16, d)
                + self.fidx.weight[1:K][None, :, None]
                + self.fage(fage[:, 1:K, None].float())[:, :, None]
            )
            toks.append(H.reshape(B, (K - 1) * 16, d))
            pads.append((~fmask[:, 1:K])[:, :, None].expand(B, K - 1, 16).reshape(B, -1))
        return torch.cat(toks, 1), torch.cat(pads, 1)

    def encode_context(
        self,
        rgb,
        depth,
        fmask,
        fage,
        cmap,
        meta,
        base,
        joints,
        jstatic,
        jmask,
        body,
        feas,
        spec_tok,
        spec_mask,
        fp,
        fp_mask,
        ops,
        ops_mask,
        ptrace,
        ptrace_valid,
        hist,
        hist_valid,
    ):
        parts, pads = [], []
        if self.use_vision:
            tv, pv = self.encode_frames(rgb, depth, fmask, fage)
            parts.append(tv)
            pads.append(pv)
        if self.use_map:
            tm = self.map(cmap)
            parts.append(tm)
            pads.append(torch.zeros(tm.shape[:2], dtype=torch.bool, device=tm.device))
        cap = self.cap(spec_tok, spec_mask, fp, fp_mask, ops, ops_mask, ptrace, ptrace_valid, hist, hist_valid)
        rest = torch.cat(
            [
                self.state(base, joints, jstatic, jmask),
                self.robot(body, jstatic, jmask),
                self.meta(meta)[:, None],
                self.feas(feas)[:, None],
                cap,
            ],
            1,
        )
        parts.append(rest)
        pads.append(torch.zeros(rest.shape[:2], dtype=torch.bool, device=rest.device))
        return torch.cat(parts, 1), torch.cat(pads, 1), cap.mean(1)

    def score(self, ctx, cand, macro_id):
        """ctx = encode_context(...); cand (B,N,F_CAND); macro_id (B,N) -> dict of (B,N) + 'ach' (B,N,4)"""
        mem, pad, capm = ctx
        B, N, _ = cand.shape
        ce = self.cand(cand) + self.macro(macro_id)
        ach = self.ach(torch.cat([capm[:, None].expand(B, N, -1), ce], -1))
        q = (ce + self.ach_in(ach.detach())).reshape(B * N, 1, -1)
        M = mem.shape[1]
        memx = mem[:, None].expand(B, N, M, mem.shape[-1]).reshape(B * N, M, -1)
        padx = pad[:, None].expand(B, N, M).reshape(B * N, M)
        h = self.dec(q, memx, memory_key_padding_mask=padx)[:, 0]
        c, r = self.cls(h).reshape(B, N, -1), self.reg(h).reshape(B, N, -1)
        out = {k: c[..., i] for i, k in enumerate(CLASS_HEADS)}
        out.update({k: r[..., i] for i, k in enumerate(REG_HEADS)})
        out["ach"] = ach
        return out

    def forward(self, cand, macro_id, **ctx_in):
        return self.score(self.encode_context(**ctx_in), cand, macro_id)


CTX_KEYS = (
    "rgb",
    "depth",
    "fmask",
    "fage",
    "cmap",
    "meta",
    "base",
    "joints",
    "jstatic",
    "jmask",
    "body",
    "feas",
    "spec_tok",
    "spec_mask",
    "fp",
    "fp_mask",
    "ops",
    "ops_mask",
    "ptrace",
    "ptrace_valid",
    "hist",
    "hist_valid",
)


def build(cfg, pretrained=True):
    return XFeasGen3(
        backbone=cfg.get("backbone", "dinov3_s"),
        pretrained=pretrained and not cfg.get("no_pretrained", False),
        freeze=not cfg.get("unfreeze", False),
        unfreeze_last=cfg.get("unfreeze_last", 0),
        use_map=cfg.get("use_map", True),
        use_frames_hist=cfg.get("use_frames_hist", True),
        k_hist=cfg.get("k_hist", K_HIST),
        use_fp=cfg.get("use_fp", True),
        use_hist=cfg.get("use_hist", True),
        use_vision=cfg.get("use_vision", True),
    )


def dummy_ctx(B, dev, K=2, J=G.NJ, T=100):
    x = dict(
        rgb=torch.randint(0, 255, (B, K, 3, 224, 224), dtype=torch.uint8),
        depth=torch.rand(B, K, 1, 224, 224) * 5,
        fmask=torch.ones(B, K, dtype=torch.bool),
        fage=torch.rand(B, K),
        cmap=torch.rand(B, CM.C_MAP, CM.NX, CM.NY),
        meta=torch.rand(B, N_META),
        base=torch.randn(B, T, MX.N_BASE),
        joints=torch.randn(B, T, J, MX.N_JOINT_DYN),
        jstatic=torch.randn(B, J, MX.N_JOINT_STATIC),
        jmask=torch.ones(B, J, dtype=torch.bool),
        body=torch.randn(B, MX.N_BODY),
        feas=torch.randn(B, G.F_FEAS),
        spec_tok=torch.randn(B, G.N_SPEC_TOK, G.F_SPEC),
        spec_mask=torch.ones(B, G.N_SPEC_TOK, dtype=torch.bool),
        fp=torch.randn(B, G.N_FP),
        fp_mask=torch.ones(B, G.N_FP, dtype=torch.bool),
        ops=torch.randn(B, G.N_OPS, G.F_OPS),
        ops_mask=torch.ones(B, G.N_OPS, dtype=torch.bool),
        ptrace=torch.randn(B, G.T_PROBE, G.F_FRAME),
        ptrace_valid=torch.ones(B, G.T_PROBE, dtype=torch.bool),
        hist=torch.randn(B, G.T_HIST, G.F_FRAME),
        hist_valid=torch.ones(B, G.T_HIST, dtype=torch.bool),
    )
    return {k: v.to(dev) for k, v in x.items()}


def _selftest():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = build(dict(backbone="cnn"), pretrained=False).to(dev).eval()
    B, N = 3, 5
    x = dummy_ctx(B, dev)
    x["hist_valid"][0] = False  # row 0: no history at all
    cand = torch.randn(B, N, G.F_CAND, device=dev)
    mac = torch.randint(0, 4, (B, N), device=dev)
    with torch.no_grad():
        o1 = m(cand=cand, macro_id=mac, **x)
        x2 = dict(x)
        h2 = x["hist"].clone()
        h2[0] = 99.0
        x2["hist"] = h2  # garbage in row 0's INVALID history
        o2 = m(cand=cand, macro_id=mac, **x2)
        x3 = dict(x)
        h3 = x["hist"].clone()
        h3[1] += 1.0
        x3["hist"] = h3  # change row 1's VALID history
        o3 = m(cand=cand, macro_id=mac, **x3)
        x4 = dict(x)
        f4 = x["fp"].clone()
        f4[2] += 1.0
        x4["fp"] = f4
        o4 = m(cand=cand, macro_id=mac, **x4)
    leak = max((o1[k][0] - o2[k][0]).abs().max().item() for k in CLASS_HEADS)
    used_h = max((o1[k][1] - o3[k][1]).abs().max().item() for k in CLASS_HEADS + ("ach",))
    used_f = max((o1[k][2] - o4[k][2]).abs().max().item() for k in CLASS_HEADS + ("ach",))
    assert leak < 1e-4 and used_h > 1e-6 and used_f > 1e-6, (leak, used_h, used_f)
    for name, ab in ABLATIONS.items():  # every ablation builds and ignores what it drops
        ma = build(dict(backbone="cnn", **ab), pretrained=False).to(dev).eval()
        with torch.no_grad():
            a1 = ma(cand=cand, macro_id=mac, **x)
            a4 = ma(cand=cand, macro_id=mac, **x4)
            a3 = ma(cand=cand, macro_id=mac, **x3)
        df = (a1["success"] - a4["success"]).abs().max().item()
        dh = (a1["success"] - a3["success"]).abs().max().item()
        assert (df < 1e-5) == (not ab.get("use_fp", True)) and (dh < 1e-5) == (not ab.get("use_hist", True)), (
            name,
            df,
            dh,
        )
    print(
        f"[gen3] invalid-history leak {leak:.1e}; history used {used_h:.1e}; fingerprint used {used_f:.1e}; ablations OK"
    )
    print(
        f"[gen3] params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M (cnn backbone), heads {CLASS_HEADS + REG_HEADS} + ach {G.ACH_HEAD}"
    )
    print("[gen3] selftest OK")


if __name__ == "__main__":
    _selftest()
