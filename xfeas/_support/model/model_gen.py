#!/usr/bin/env python
"""
XFeasGen = XFeas (xfeas/_support/model/model_xe.py, unchanged) with
  * a robot stream that carries the SETPOINT BOX: one token per controllable dim (sp_tokens.SP_DIMS) built from
    [lo, hi, nominal, present, nominal_frac] + a learned per-dim embedding, next to the body and joint-limit tokens
  * actions given NORMALISED to the robot's own box (sp_tokens.normalise; de-normalise with sp_tokens.denormalise)
  * regression heads progress, max_tilt + cost heads t_norm, energy, posture_dev, tilt_margin
The vision / state encoders and the cross-attention decoder are model_xe's.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys
import torch
import torch.nn as nn

THIS = os.path.dirname(os.path.abspath(__file__))
import model_xe as MX
import sp_tokens as SP

CLASS_HEADS = MX.CLASS_HEADS
REG_HEADS = ("progress", "max_tilt", "t_norm", "energy", "posture_dev", "tilt_margin")
FORMAT = "xfeas_gen_v1"


class SetpointRobotEncoder(nn.Module):
    def __init__(self, d=192, heads=4):
        super().__init__()
        self.body = MX.mlp(MX.N_BODY, d, d)
        self.joint = MX.mlp(MX.N_JOINT_STATIC, d, d)
        self.pool = MX.JointPool(d, heads)
        self.sp = MX.mlp(SP.F_SP, d, d)
        self.sp_dim = nn.Parameter(torch.randn(1, SP.N_SP, d) * 0.02)

    def forward(self, body, jstatic, jmask, sp_tok):
        base = torch.stack([self.body(body), self.pool(self.joint(jstatic), jmask)], 1)
        return torch.cat([base, self.sp(sp_tok) + self.sp_dim], 1)  # (B, 2 + N_SP, d)


class XFeasGen(MX.XFeas):
    def __init__(
        self, d=192, k_max=16, backbone="dinov3_s", pretrained=True, freeze=True, unfreeze_last=0, dec_layers=3, heads=4
    ):
        super().__init__(d, k_max, backbone, pretrained, freeze, unfreeze_last, dec_layers, heads)
        self.robot = SetpointRobotEncoder(d, heads)
        self.reg = nn.Linear(d, len(REG_HEADS))

    def encode_context(self, rgb, depth, base, joints, jstatic, jmask, body, sp_tok):
        parts = []
        if self.use_vision:
            parts.append(self.vision(rgb, depth))
        if self.use_state:
            parts.append(self.state(base, joints, jstatic, jmask))
        if self.use_robot:
            parts.append(self.robot(body, jstatic, jmask, sp_tok))
        return torch.cat(parts, 1)

    def score(self, ctx, actions, skill):
        """ctx (B,M,d); actions (B,N,K,7) NORMALISED; skill (B,N) -> dict of (B,N)"""
        actions = MX.pad_actions(actions)
        B, N, K, _ = actions.shape
        a = self.act(actions) + self.apos[:, :K][:, None] + self.skill(skill)[:, :, None]
        q = torch.cat([self.readout.expand(B * N, -1, -1), a.reshape(B * N, K, -1)], 1)
        mem = ctx[:, None].expand(B, N, *ctx.shape[1:]).reshape(B * N, *ctx.shape[1:])
        h = self.dec(q, mem)[:, 0]
        c, r = self.cls(h).reshape(B, N, -1), self.reg(h).reshape(B, N, -1)
        out = {k: c[..., i] for i, k in enumerate(CLASS_HEADS)}
        out.update({k: r[..., i] for i, k in enumerate(REG_HEADS)})
        return out

    def forward(self, rgb, depth, base, joints, jstatic, jmask, body, sp_tok, actions, skill):
        return self.score(self.encode_context(rgb, depth, base, joints, jstatic, jmask, body, sp_tok), actions, skill)


def build(cfg, pretrained=True):
    return XFeasGen(
        backbone=cfg.get("backbone", "dinov3_s"),
        pretrained=pretrained and not cfg.get("no_pretrained", False),
        freeze=not cfg.get("unfreeze", False),
        unfreeze_last=cfg.get("unfreeze_last", 0),
    )
