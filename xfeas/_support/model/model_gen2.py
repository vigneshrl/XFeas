#!/usr/bin/env python
"""
XFeasGen2 (gen2 agent, 2026-10-14): XFeas-Gen with
  * FRAME MEMORY: up to K_HIST RGB-D frames (index 0 = the decision frame, 1.. = older). The frozen DINOv3-S + depth
    stem of model_xe encodes each PRESENT frame once (absent frames are never run through the backbone); the decision
    frame keeps its 8x8 grid (64 tokens), older frames are pooled to 4x4 (16 tokens) and get a learned frame-index
    embedding + an age (seconds) embedding. Absent frames are MASKED out of the decoder memory (key padding mask), and a
    meta token carries the "no history" flag, so gen_v1 / old single-frame rows are exact single-frame inputs.
  * CLEARANCE MAP (cmap.py): a metric, gravity-aligned 2.5-D map (C_MAP x 32 x 16 cells, 0..4 m ahead, +-1 m) built
    from the depth frame(s) -> small CNN -> 8x4 = 32 tokens with a learned position embedding.
  * SETPOINT-RANGE TOKENS per robot (model_gen.SetpointRobotEncoder, unchanged from gen1) + actions normalised to the
    robot's own box (sp_tokens.normalise).
  * NEW ACTION TOKENS in the GEN_V2_SPEC encoding: the structured action act_v2 (16 fields, section 3.2: phase-B and
    phase-A setpoints, d_commit, d_level, lat_off, head_off, vy, L_ramp, s_back, d_yaw) NORMALISED per robot
    (dataset_gen2.norm_act_v2) + its 16-d mask, as one 32-d vector added to every chunk step, next to the skill id
    (go 0, go2 3, sidestep 4, detour 7) and the rendered (10,7) chunk. gen_v1 rows get act_v2 derived from their
    setpoint (go) or sidestep parameters; old sets get an all-zero mask (absent).
  * HEADS: success, fall, collision, stall, slip (per-rollout gen_v2 `slipped`), slippery (surface is low-friction:
    gen_v1 terrain tile flag / gen_v2 mu_patch) (BCE) + progress, max_tilt, t_norm, energy, posture_dev, tilt_margin.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
import model_xe as MX
import model_gen as MG
import sp_tokens as SP
import cmap as CM

FORMAT = "xfeas_gen2_v1"
CLASS_HEADS = ("success", "fall", "collision", "stall", "slip", "slippery")
REG_HEADS = MG.REG_HEADS  # progress, max_tilt, t_norm, energy, posture_dev, tilt_margin
K_HIST = 4
ACT_V2 = (
    "v_b",
    "dh_b",
    "pitch_b",
    "roll_b",
    "v_a",
    "dh_a",
    "pitch_a",
    "roll_a",
    "d_commit",
    "d_level",
    "lat_off",
    "head_off",
    "vy",
    "L_ramp",
    "s_back",
    "d_yaw",
)  # GEN_V2_SPEC section 3.2
AP_FIELDS = tuple(f"n_{k}" for k in ACT_V2) + tuple(f"m_{k}" for k in ACT_V2)
N_AP = len(AP_FIELDS)
META_FIELDS = ("no_hist", "n_frames", "map_used_prior", "cam_h", "map_present", "hist_span", "spare0", "spare1")
N_META = len(META_FIELDS)


class MapEncoder(nn.Module):
    def __init__(self, d=192, c_in=CM.C_MAP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, d, 3, 2, 1),
        )
        self.pos = nn.Parameter(torch.zeros(1, (CM.NX // 4) * (CM.NY // 4), d))

    def forward(self, m):  # (B, C, NX, NY) -> (B, NX/4 * NY/4, d)
        return self.net(m).flatten(2).transpose(1, 2) + self.pos


class XFeasGen2(nn.Module):
    def __init__(
        self,
        d=192,
        k_max=16,
        backbone="dinov3_s",
        pretrained=True,
        freeze=True,
        unfreeze_last=0,
        dec_layers=3,
        heads=4,
        use_map=True,
        use_hist=True,
        k_hist=K_HIST,
    ):
        super().__init__()
        self.vision = MX.VisionEncoder(d, backbone, pretrained, freeze, unfreeze_last=unfreeze_last)
        self.state = MX.StateEncoder(d)
        self.robot = MG.SetpointRobotEncoder(d, heads)
        self.map = MapEncoder(d)
        self.meta = MX.mlp(N_META, d, d)
        self.fidx = nn.Embedding(k_hist, d)
        self.fage = MX.mlp(1, d, d)
        self.act = MX.mlp(MX.A_DIM, d, d)
        self.ap = MX.mlp(N_AP, d, d)
        self.apos = nn.Parameter(torch.zeros(1, k_max, d))
        self.skill = nn.Embedding(MX.N_SKILLS, d)
        self.readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, dec_layers)
        self.cls = nn.Linear(d, len(CLASS_HEADS))
        self.reg = nn.Linear(d, len(REG_HEADS))
        self.use_map, self.use_hist, self.k_hist = use_map, use_hist, k_hist

    # ------------------------------------------------------------------ context
    def encode_frames(self, rgb, depth, fmask, fage):
        """rgb (B,K,3,H,W) uint8, depth (B,K,1,H,W), fmask (B,K) bool (frame 0 always True), fage (B,K) s
        -> tokens (B, 64 + 16 (K-1), d), pad mask (B, same) True = absent"""
        B, K = fmask.shape
        if not self.use_hist:
            K = 1
        cur = self.vision(rgb[:, 0], depth[:, 0])  # (B, 64, d)
        d = cur.shape[-1]
        toks = [cur + self.fidx.weight[0]]
        pads = [torch.zeros(B, cur.shape[1], dtype=torch.bool, device=cur.device)]
        if K > 1:
            sel = fmask[:, 1:K].reshape(-1)
            H = torch.zeros(B * (K - 1), 16, d, device=cur.device, dtype=cur.dtype)
            if sel.any():
                r = rgb[:, 1:K].reshape(B * (K - 1), *rgb.shape[2:])[sel]
                dp = depth[:, 1:K].reshape(B * (K - 1), *depth.shape[2:])[sel]
                g = self.vision(r, dp)  # (n, 64, d)
                g = g.transpose(1, 2).reshape(-1, d, 8, 8)
                H[sel] = F.avg_pool2d(g, 2).flatten(2).transpose(1, 2)
            H = H.reshape(B, K - 1, 16, d)
            H = H + self.fidx.weight[1:K][None, :, None] + self.fage(fage[:, 1:K, None].float())[:, :, None]
            toks.append(H.reshape(B, (K - 1) * 16, d))
            pads.append((~fmask[:, 1:K])[:, :, None].expand(B, K - 1, 16).reshape(B, -1))
        return torch.cat(toks, 1), torch.cat(pads, 1)

    def encode_context(self, rgb, depth, fmask, fage, cmap, meta, base, joints, jstatic, jmask, body, sp_tok):
        tv, pv = self.encode_frames(rgb, depth, fmask, fage)
        parts, pads = [tv], [pv]
        if self.use_map:
            tm = self.map(cmap)
            parts.append(tm)
            pads.append(torch.zeros(tm.shape[:2], dtype=torch.bool, device=tm.device))
        rest = torch.cat(
            [
                self.state(base, joints, jstatic, jmask),
                self.robot(body, jstatic, jmask, sp_tok),
                self.meta(meta)[:, None],
            ],
            1,
        )
        parts.append(rest)
        pads.append(torch.zeros(rest.shape[:2], dtype=torch.bool, device=rest.device))
        return torch.cat(parts, 1), torch.cat(pads, 1)

    # ------------------------------------------------------------------ candidates
    def score(self, ctx, actions, skill, ap=None):
        """ctx = (mem (B,M,d), pad (B,M)); actions (B,N,K,7) NORMALISED; skill (B,N); ap (B,N,N_AP) -> dict of (B,N)"""
        mem, pad = ctx
        actions = MX.pad_actions(actions)
        B, N, K, _ = actions.shape
        if ap is None:
            ap = actions.new_zeros(B, N, N_AP)
        a = self.act(actions) + self.apos[:, :K][:, None] + self.skill(skill)[:, :, None] + self.ap(ap)[:, :, None]
        q = torch.cat([self.readout.expand(B * N, -1, -1), a.reshape(B * N, K, -1)], 1)
        M = mem.shape[1]
        memx = mem[:, None].expand(B, N, M, mem.shape[-1]).reshape(B * N, M, -1)
        padx = pad[:, None].expand(B, N, M).reshape(B * N, M)
        h = self.dec(q, memx, memory_key_padding_mask=padx)[:, 0]
        c, r = self.cls(h).reshape(B, N, -1), self.reg(h).reshape(B, N, -1)
        out = {k: c[..., i] for i, k in enumerate(CLASS_HEADS)}
        out.update({k: r[..., i] for i, k in enumerate(REG_HEADS)})
        return out

    def forward(self, actions, skill, ap=None, **ctx_in):
        return self.score(self.encode_context(**ctx_in), actions, skill, ap)


def build(cfg, pretrained=True):
    return XFeasGen2(
        backbone=cfg.get("backbone", "dinov3_s"),
        pretrained=pretrained and not cfg.get("no_pretrained", False),
        freeze=not cfg.get("unfreeze", False),
        unfreeze_last=cfg.get("unfreeze_last", 0),
        use_map=cfg.get("use_map", True),
        use_hist=cfg.get("use_hist", True),
        k_hist=cfg.get("k_hist", K_HIST),
    )


def _selftest():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = build(dict(), pretrained=False).to(dev).eval()
    B, K, T, J, N = 3, 4, 100, 19, 5
    jmask = torch.arange(J)[None] < torch.tensor([19, 12, 4])[:, None]
    fmask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    x = dict(
        rgb=torch.randint(0, 255, (B, K, 3, 224, 224), dtype=torch.uint8),
        depth=torch.rand(B, K, 1, 224, 224) * 5,
        fmask=fmask,
        fage=torch.tensor([[0, 0, 0, 0], [0, 0.5, 1.0, 0], [0, 0.5, 1.0, 1.5]]),
        cmap=torch.rand(B, CM.C_MAP, CM.NX, CM.NY),
        meta=torch.rand(B, N_META),
        base=torch.randn(B, T, MX.N_BASE),
        joints=torch.randn(B, T, J, MX.N_JOINT_DYN),
        jstatic=torch.randn(B, J, MX.N_JOINT_STATIC),
        jmask=jmask,
        body=torch.randn(B, MX.N_BODY),
        sp_tok=torch.randn(B, SP.N_SP, SP.F_SP),
    )
    x = {k: v.to(dev) for k, v in x.items()}
    acts = torch.randn(B, N, 10, 7, device=dev)
    sk = torch.zeros(B, N, dtype=torch.long, device=dev)
    with torch.no_grad():
        o1 = m(actions=acts, skill=sk, **x)
        x2 = dict(x)
        r2 = x["rgb"].clone()
        r2[0, 1:] = 255 - r2[0, 1:]
        x2["rgb"] = r2  # garbage in ABSENT frames of row 0
        o2 = m(actions=acts, skill=sk, **x2)
        x3 = dict(x)
        r3 = x["rgb"].clone()
        r3[1, 1] = 255 - r3[1, 1]
        x3["rgb"] = r3  # a PRESENT history frame
        o3 = m(actions=acts, skill=sk, **x3)
    used = max((o1[k][1] - o3[k][1]).abs().max().item() for k in o1)
    assert used > 1e-6, used
    leak = max((o1[k][0] - o2[k][0]).abs().max().item() for k in o1)
    moved = max((o1[k][2] - o2[k][2]).abs().max().item() for k in o1)
    print(
        f"[gen2] absent-frame leak {leak:.2e} (must be ~0); untouched row {moved:.2e}; present history frame changes output by {used:.2e}"
    )
    assert leak < 1e-4 and moved < 1e-4
    print(f"[gen2] params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M, heads {CLASS_HEADS + REG_HEADS}")
    print("[gen2] selftest OK")


if __name__ == "__main__":
    _selftest()
