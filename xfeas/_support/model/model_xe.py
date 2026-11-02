#!/usr/bin/env python
"""
XFeas -- robot-agnostic, action-conditioned feasibility model.

    F(RGB-D, proprio history, robot descriptor, candidate action) -> outcome distribution

Four token streams, one decoder:

  vision   DINOv3 ViT-S/16 on RGB (frozen by default) + a small conv stem on depth, fused per
           patch, pooled to an 8x8 grid of tokens. Encoded ONCE per frame.
  state    proprio history (T x J joints). Each joint token = its dynamic signals (incl. the
           limit-normalised "actuator utilisation" channels from robots.py) + its STATIC limits
           from the descriptor, so the same weights read a 12-joint Spot, 19-joint H1 or a
           2-wheel Carter. Joints are attention-pooled per timestep (masked), then a temporal
           transformer reads the history -> in-context system identification.
  robot    body descriptor (mass, bbox, max command, family) + pooled static joint limits.
  action   a K-step chunk of body-frame twists [vx, vy, vz, wz] in PHYSICAL units + a skill id.

Candidates are scored in a batch against one context (encode_context once, then
score(ctx, actions) for N candidates) -- that is what makes it usable inside a real-time
selector.

Heads: logits for P(success), P(fall), P(collision), P(stall); regressions for progress and
max tilt. Uncertainty comes from an ensemble at train time (train_xe.py), not from here.

    $XFEAS_PY xfeas/_support/model/model_xe.py --selftest
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BASE = 13  # must match robots.N_BASE
N_JOINT_DYN = 7  # must match robots.N_JOINT
N_JOINT_STATIC = 5  # lower, upper, log vmax, log tau_max, is_wheel
N_BODY = 12  # mass(log), bbox xyz, max_cmd(5), family one-hot(3)
N_SKILLS = 8  # go, stop, back_off, climb_step, pass_gap, crouch_under, fly_over, spare
CLASS_HEADS = ("success", "fall", "collision", "stall")
REG_HEADS = ("progress", "max_tilt")


def mlp(i, h, o, p=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class VisionEncoder(nn.Module):
    """RGB backbone + depth stem -> (B, G*G, d) tokens."""

    def __init__(self, d=192, backbone="dinov3_s", pretrained=True, freeze=True, grid=8, img=224, unfreeze_last=0):
        super().__init__()
        self.img, self.grid, self.backbone_name = img, grid, backbone
        if backbone == "cnn":  # scratch baseline, no pretraining
            self.rgb = nn.Sequential(
                nn.Conv2d(3, 32, 5, 2, 2),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.GELU(),
                nn.Conv2d(128, 192, 3, 2, 1),
                nn.GELU(),
            )
            cdim = 192
        else:
            import timm

            name = {
                "dinov3_s": "vit_small_patch16_dinov3",
                "dinov3_splus": "vit_small_plus_patch16_dinov3",
                "dinov3_b": "vit_base_patch16_dinov3",
            }.get(backbone, backbone)
            self.rgb = timm.create_model(
                name, pretrained=pretrained, num_classes=0, img_size=img, dynamic_img_size=True
            )
            cdim = self.rgb.num_features
            if freeze:
                for p in self.rgb.parameters():
                    p.requires_grad_(False)
        if freeze and unfreeze_last > 0 and backbone != "cnn":  # adapt only the top ViT blocks
            for blk in list(self.rgb.blocks)[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.rgb.norm.parameters():
                p.requires_grad_(True)
        self.frozen = freeze and backbone != "cnn" and unfreeze_last == 0
        self.depth = nn.Sequential(  # 16x16 patchify like the ViT
            nn.Conv2d(1, 32, 4, 4), nn.GELU(), nn.Conv2d(32, 64, 2, 2), nn.GELU(), nn.Conv2d(64, d, 2, 2)
        )
        self.proj = nn.Linear(cdim, d)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, d))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _rgb_grid(self, x):
        if self.backbone_name == "cnn":
            return self.rgb(x)
        tok = self.rgb.forward_features(x)  # (B, prefix + h*w, C)
        n_pref = getattr(self.rgb, "num_prefix_tokens", 1)
        tok = tok[:, n_pref:]
        h = w = int(math.isqrt(tok.shape[1]))
        return tok.transpose(1, 2).reshape(x.shape[0], -1, h, w)

    def forward(self, rgb, depth):
        """rgb (B,3,H,W) uint8 or float in [0,1]; depth (B,1,H,W) metres."""
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        rgb = F.interpolate(rgb, size=(self.img, self.img), mode="bilinear", align_corners=False)
        rgb = (rgb - self.mean) / self.std
        dep = F.interpolate(depth, size=(self.img, self.img), mode="nearest")
        dep = torch.log1p(dep.clamp(0, 20))  # metric, compressed
        with torch.set_grad_enabled(not self.frozen and self.training):
            g = self._rgb_grid(rgb)
        g = F.adaptive_avg_pool2d(g, self.grid).flatten(2).transpose(1, 2)  # (B, G*G, C)
        dg = F.adaptive_avg_pool2d(self.depth(dep), self.grid).flatten(2).transpose(1, 2)
        return self.proj(g) + dg + self.pos


class JointPool(nn.Module):
    """Masked attention pooling over a variable number of joint tokens (URMA-style)."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, tok, mask):
        """tok (N, J, d), mask (N, J) True = real joint. Robots with no joints (aerial) get a
        single all-masked row; the zero output is then carried by the base token alone."""
        empty = ~mask.any(1)
        m = mask.clone()
        m[empty, 0] = True  # keep softmax finite
        out, _ = self.att(self.q.expand(tok.shape[0], -1, -1), tok, tok, key_padding_mask=~m)
        out = out.squeeze(1)
        return out.masked_fill(empty[:, None], 0.0)


class StateEncoder(nn.Module):
    """Proprio history -> (B, S, d) state tokens (in-context system identification)."""

    def __init__(self, d=192, t_stride=4, layers=2, heads=4, max_t=128, n_out=4):
        super().__init__()
        self.t_stride, self.n_out = t_stride, n_out
        self.joint = mlp(N_JOINT_DYN + N_JOINT_STATIC, d, d)
        self.base = mlp(N_BASE, d, d)
        self.pool = JointPool(d, heads)
        self.tpos = nn.Parameter(torch.zeros(1, max_t, d))
        self.readout = nn.Parameter(torch.randn(1, n_out, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)

    def forward(self, base, joints, jstatic, jmask):
        """base (B,T,13) joints (B,T,J,7) jstatic (B,J,5) jmask (B,J)."""
        base, joints = base[:, :: self.t_stride], joints[:, :: self.t_stride]
        B, T, J, _ = joints.shape
        js = jstatic[:, None].expand(B, T, J, N_JOINT_STATIC)
        jt = self.joint(torch.cat([joints, js], -1)).reshape(B * T, J, -1)
        jm = jmask[:, None].expand(B, T, J).reshape(B * T, J)
        per_t = self.pool(jt, jm).reshape(B, T, -1) + self.base(base)
        x = torch.cat([self.readout.expand(B, -1, -1), per_t + self.tpos[:, :T]], 1)
        return self.tf(x)[:, : self.n_out]


class RobotEncoder(nn.Module):
    def __init__(self, d=192, heads=4):
        super().__init__()
        self.body = mlp(N_BODY, d, d)
        self.joint = mlp(N_JOINT_STATIC, d, d)
        self.pool = JointPool(d, heads)

    def forward(self, body, jstatic, jmask):
        return torch.stack([self.body(body), self.pool(self.joint(jstatic), jmask)], 1)  # (B,2,d)


A_DIM = 7  # action step: [vx, vy, vz, wz, d_body_height, pitch, roll]; old 4-d chunks are zero-padded
# (= default posture). Robots without posture setpoints always see zeros there.


def pad_actions(a):
    """(..., k<=A_DIM) -> (..., A_DIM) with zeros (default posture)."""
    if a.shape[-1] == A_DIM:
        return a
    z = a.new_zeros(*a.shape[:-1], A_DIM - a.shape[-1])
    return torch.cat([a, z], -1)


def load_compat(m, sd):
    """load a state dict saved with a narrower action input (e.g. v5's 4-d): the new input columns of the
    first action layer start at 0, so the loaded model scores old-style actions exactly as before."""
    sd = dict(sd)
    for k, v in m.state_dict().items():
        if k in sd and sd[k].shape != v.shape and k.startswith("act.") and sd[k].dim() == 2:
            w = v.clone().zero_()
            w[:, : sd[k].shape[1]] = sd[k]
            sd[k] = w
    m.load_state_dict(sd)
    return m


class XFeas(nn.Module):
    def __init__(
        self, d=192, k_max=16, backbone="dinov3_s", pretrained=True, freeze=True, unfreeze_last=0, dec_layers=3, heads=4
    ):
        super().__init__()
        self.vision = VisionEncoder(d, backbone, pretrained, freeze, unfreeze_last=unfreeze_last)
        self.state = StateEncoder(d)
        self.robot = RobotEncoder(d, heads)
        self.act = mlp(A_DIM, d, d)
        self.apos = nn.Parameter(torch.zeros(1, k_max, d))
        self.skill = nn.Embedding(N_SKILLS, d)
        self.readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, dec_layers)
        self.cls = nn.Linear(d, len(CLASS_HEADS))
        self.reg = nn.Linear(d, len(REG_HEADS))
        # ablation switches (train_xe --no-proprio / --no-robot / --no-vision)
        self.use_state = self.use_robot = self.use_vision = True

    def encode_context(self, rgb, depth, base, joints, jstatic, jmask, body):
        """Everything that does NOT depend on the candidate action. (B, M, d)."""
        parts = []
        if self.use_vision:
            parts.append(self.vision(rgb, depth))
        if self.use_state:
            parts.append(self.state(base, joints, jstatic, jmask))
        if self.use_robot:
            parts.append(self.robot(body, jstatic, jmask))
        return torch.cat(parts, 1)

    def score(self, ctx, actions, skill):
        """ctx (B,M,d); actions (B,N,K,4..A_DIM); skill (B,N) long -> dict of (B,N) tensors."""
        actions = pad_actions(actions)
        B, N, K, _ = actions.shape
        a = self.act(actions) + self.apos[:, :K][:, None] + self.skill(skill)[:, :, None]
        q = torch.cat([self.readout.expand(B * N, -1, -1), a.reshape(B * N, K, -1)], 1)
        mem = ctx[:, None].expand(B, N, *ctx.shape[1:]).reshape(B * N, *ctx.shape[1:])
        h = self.dec(q, mem)[:, 0]
        c, r = self.cls(h).reshape(B, N, -1), self.reg(h).reshape(B, N, -1)
        out = {k: c[..., i] for i, k in enumerate(CLASS_HEADS)}
        out.update({k: r[..., i] for i, k in enumerate(REG_HEADS)})
        return out

    def forward(self, rgb, depth, base, joints, jstatic, jmask, body, actions, skill):
        return self.score(self.encode_context(rgb, depth, base, joints, jstatic, jmask, body), actions, skill)


def _selftest():
    import argparse, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--backbone", default="dinov3_s")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = XFeas(backbone=a.backbone, pretrained=a.pretrained).to(dev).eval()
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    ntot = sum(p.numel() for p in m.parameters())
    print(f"[xfeas] params total={ntot / 1e6:.2f}M trainable={ntr / 1e6:.2f}M on {dev}")
    # a mixed batch: Spot (12 joints), H1 (19), Carter (2) padded to J=19
    B, T, K, J = 3, 100, 10, 19
    nj = torch.tensor([12, 19, 2])
    jmask = torch.arange(J)[None] < nj[:, None]
    x = dict(
        rgb=torch.randint(0, 255, (B, 3, 384, 384), dtype=torch.uint8),
        depth=torch.rand(B, 1, 384, 384) * 6,
        base=torch.randn(B, T, N_BASE),
        joints=torch.randn(B, T, J, N_JOINT_DYN) * jmask[:, None, :, None],
        jstatic=torch.randn(B, J, N_JOINT_STATIC),
        jmask=jmask,
        body=torch.randn(B, N_BODY),
    )
    x = {k: v.to(dev) for k, v in x.items()}
    acts = torch.randn(B, a.n, K, 4, device=dev) * 0.5
    skill = torch.randint(0, N_SKILLS, (B, a.n), device=dev)
    with torch.no_grad():
        out = m(**x, actions=acts, skill=skill)
    assert all(v.shape == (B, a.n) for v in out.values()), {k: v.shape for k, v in out.items()}
    assert all(torch.isfinite(v).all() for v in out.values())
    # padding invariance: garbage in padded joint slots must not change the output
    x2 = dict(x)
    x2["joints"] = x["joints"] + (~jmask.to(dev))[:, None, :, None] * 100.0
    with torch.no_grad():
        out2 = m(**x2, actions=acts, skill=skill)
    dmax = max((out[k] - out2[k]).abs().max().item() for k in out)
    print(f"[xfeas] padded-joint leak: {dmax:.2e} (must be ~0)")
    assert dmax < 1e-4
    # latency: one robot, one frame, N candidates (the selector's real call pattern)
    one = {k: v[:1] for k, v in x.items()}
    with torch.no_grad():
        for _ in range(5):
            m.score(m.encode_context(**one), acts[:1], skill[:1])
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            m.score(m.encode_context(**one), acts[:1], skill[:1])
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
    print(f"[xfeas] latency: 1 frame + {a.n} candidates = {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms")
    print("[xfeas] selftest OK")


if __name__ == "__main__":
    _selftest()
