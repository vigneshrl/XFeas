#!/usr/bin/env python
"""
Per-robot SETPOINT BOX, descriptor setpoint tokens, per-robot action normalisation and derived labels
(ROBUSTNESS_PLAN.md R3/R4). Pure numpy; shared by the manifest builder, the dataset, the trainer and the selector.

Action chunk columns (dataset_xe / run_gen write_ep, verified on data):
    0 vx (m/s)  1 vy  2 vz  3 wz (rad/s)  4 dh = body_height - NOMINAL_H (m)  5 pitch (rad)  6 roll (rad)
Old 4-d chunks are zero-padded (= stock posture).

Box of a robot (same rule as gen/gensim.py full_box, so Phase-1 / gen_v1 / old data agree):
    v            [max(v_min, 0.1), max_cmd.vx_fwd], nominal v_nom = max(0.6 * vx_fwd, v_min)
    body_height  desc.setpoint_ranges.body_height (absolute m), nominal NOMINAL_H of the controller
    pitch, roll  desc.setpoint_ranges, nominal 0
    vy, wz       [-max_cmd, +max_cmd], nominal 0
    gait_freq, footswing   generic dims, absent (masked) for every current controller
A gen_v1 episode's `box` json ({box: {v, body_height, pitch, roll}, nominal: {...}}) overrides v / posture / nominal.

Normalisation to [-1, 1] of the robot's own box:
    velocity cols (vx, vy, vz, wz): sign-preserving a / max_cmd (vx<0 by vx_rev). 0 stays 0 (stop / padded steps), the box
        top maps to +1; the v token carries where v_lo and v_nom sit in these units.
    posture cols (body height, pitch, roll): piecewise linear with the controller's NOMINAL -> 0, box lo -> -1, box hi -> +1
        ("lowest crouch" = -1 on every body; stock posture = 0 on every body, so zero-padded old chunks stay correct).
    absent dims -> 0. denormalise() is the exact inverse (used by the selector).

Setpoint token per dim (N_SP x F_SP): [lo, hi, nominal (physical units), present flag, (nominal - lo) / (hi - lo)].
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math
import numpy as np

SP_DIMS = ("v", "vy", "wz", "body_height", "pitch", "roll", "gait_freq", "footswing")
N_SP = len(SP_DIMS)
F_SP = 5
A_DIM = 7
K_CHUNK = 10
POSTURE = ("body_height", "pitch", "roll")
POSTURE_COL = {"body_height": 4, "pitch": 5, "roll": 6}
# controller nominal standing heights (controllers/spot_posture.py NOMINAL_H = 0.52, controllers/g1_amo.py NOMINAL_H = 0.75);
# checked against the source files by selftest()
NOMINAL_H = {"spotp": 0.52, "g1": 0.75}


def nominal_height(kind):
    if kind == "spotp" or kind.startswith("spotp"):
        return NOMINAL_H["spotp"]
    if kind.startswith("g1"):
        return NOMINAL_H["g1"]
    return None


def robot_box(desc, box_json=None):
    """desc dict (+ optional gen_v1 box json str/dict) -> box dict with physical ranges and nominal values."""
    mc = desc["max_cmd"]
    vmin = float(desc.get("v_min", 0.1) or 0.1)
    vxf = float(mc["vx_fwd"])
    vxr = float(mc.get("vx_rev", vxf) or vxf)
    b = dict(
        kind=desc.get("kind", "?"),
        vx_fwd=vxf,
        vx_rev=vxr,
        vy=float(mc.get("vy", 0.0)),
        vz=float(mc.get("vz", 0.0)),
        wz=float(mc.get("wz", 0.0)),
    )
    b["v"] = (max(vmin, 0.1), vxf)
    nom = dict(v=max(0.6 * vxf, vmin), body_height=0.0, pitch=0.0, roll=0.0)
    R = desc.get("setpoint_ranges") or {}
    nh = nominal_height(str(desc.get("kind", "")))
    for k in POSTURE:
        b[k] = tuple(map(float, R[k])) if k in R else None
    if nh is None or "body_height" not in R:
        b["body_height"] = None
    else:
        nom["body_height"] = nh
    if box_json:
        bj = json.loads(box_json) if isinstance(box_json, str) else box_json
        bx = bj.get("box", bj)
        if "v" in bx:
            b["v"] = tuple(map(float, bx["v"]))
        for k in POSTURE:
            if k in bx:
                lo, hi = map(float, bx[k])
                b[k] = (lo, hi) if hi - lo > 1e-6 else None
        n2 = bj.get("nominal")
        if isinstance(n2, dict):
            for k in ("v", "body_height", "pitch", "roll"):
                if k in n2 and n2[k] is not None:
                    nom[k] = float(n2[k])
    b["nominal"] = nom
    return b


def box_key(b):
    """hashable summary (manifest / stats)"""
    return json.dumps({k: b[k] for k in ("kind", "v", "body_height", "pitch", "roll", "nominal")}, sort_keys=True)


def sp_tokens(b):
    T = np.zeros((N_SP, F_SP), np.float32)

    def put(i, lo, hi, nom):
        T[i] = [lo, hi, nom, 1.0, (nom - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5]

    put(0, b["v"][0], b["v"][1], b["nominal"]["v"])
    if b["vy"] > 0:
        put(1, -b["vy"], b["vy"], 0.0)
    if b["wz"] > 0:
        put(2, -b["wz"], b["wz"], 0.0)
    for i, k in ((3, "body_height"), (4, "pitch"), (5, "roll")):
        if b[k] is not None:
            put(i, b[k][0], b[k][1], b["nominal"][k])
    return T  # gait_freq / footswing rows stay 0 (present = 0)


def _pw(x, lo, nom, hi):
    """piecewise-linear: lo -> -1, nom -> 0, hi -> +1"""
    dn, up = max(nom - lo, 1e-6), max(hi - nom, 1e-6)
    return np.where(x >= nom, (x - nom) / up, (x - nom) / dn)


def _pw_inv(y, lo, nom, hi):
    dn, up = max(nom - lo, 1e-6), max(hi - nom, 1e-6)
    return np.where(y >= 0, nom + y * up, nom + y * dn)


def normalise(a, b, clip=2.0):
    """physical (..., 7) -> normalised (..., 7)"""
    a = np.asarray(a, np.float32)
    if a.shape[-1] < A_DIM:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (A_DIM - a.shape[-1],), np.float32)], -1)
    y = np.zeros_like(a)
    vx = a[..., 0]
    y[..., 0] = np.where(vx >= 0, vx / max(b["vx_fwd"], 1e-6), vx / max(b["vx_rev"], 1e-6))
    for c, k in ((1, "vy"), (2, "vz"), (3, "wz")):
        y[..., c] = a[..., c] / b[k] if b[k] > 1e-6 else 0.0
    for k, c in POSTURE_COL.items():
        if b[k] is None:
            continue
        lo, hi = b[k]
        nom = b["nominal"][k]
        x = a[..., c] + (nom if k == "body_height" else 0.0)  # dh -> absolute body height
        y[..., c] = _pw(x, lo, nom, hi)
    return np.clip(y, -clip, clip).astype(np.float32)


def denormalise(y, b):
    """normalised (..., 7) -> physical (..., 7) (dh relative to nominal, like the collectors)"""
    y = np.asarray(y, np.float32)
    a = np.zeros_like(y)
    a[..., 0] = np.where(y[..., 0] >= 0, y[..., 0] * b["vx_fwd"], y[..., 0] * b["vx_rev"])
    for c, k in ((1, "vy"), (2, "vz"), (3, "wz")):
        a[..., c] = y[..., c] * b[k]
    for k, c in POSTURE_COL.items():
        if b[k] is None:
            continue
        lo, hi = b[k]
        nom = b["nominal"][k]
        x = _pw_inv(y[..., c], lo, nom, hi)
        a[..., c] = x - nom if k == "body_height" else x
    return a


def posture_dev_from_action(a, b):
    """mean over chunk steps of mean_k |setpoint_k - nominal_k| / box width (gen/gensim.py posture_dev, same formula)"""
    a = np.asarray(a, np.float32)
    if a.shape[-1] < A_DIM:
        return 0.0
    out = []
    for k, c in POSTURE_COL.items():
        if b[k] is None:
            continue
        lo, hi = b[k]
        x = a[..., c] + (b["nominal"][k] if k == "body_height" else 0.0)
        out.append(np.abs(x - b["nominal"][k]) / (hi - lo))
    return float(np.mean(out)) if out else 0.0


def chunk_to_k(a, k=K_CHUNK):
    """(n, d) chunk -> (k, A_DIM): truncate, or pad by holding the LAST step (a held setpoint), zero posture for 4-d"""
    a = np.asarray(a, np.float32)
    if a.shape[-1] < A_DIM:
        a = np.concatenate([a, np.zeros((len(a), A_DIM - a.shape[-1]), np.float32)], -1)
    if len(a) >= k:
        return a[:k]
    return np.concatenate([a, np.repeat(a[-1:], k - len(a), 0)], 0)


def t_ref_from_lane(lane, b):
    """gen Phase-1 episodes have no t_ref key: (s_goal - s_dec) / v_nom, s_dec = s_front - d_dec (run_gen.py)"""
    try:
        s_dec = float(lane["s_front"]) - float(lane["d_dec"])
        return (float(lane["s_goal"]) - s_dec) / max(b["nominal"]["v"], 1e-3)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def selftest():
    import os, re

    here = os.path.dirname(os.path.abspath(__file__))
    for f, key in (("spot_posture.py", "spotp"), ("g1_amo.py", "g1")):
        src = open(os.path.join(here, "..", "xembody", "controllers", f)).read()
        v = float(re.search(r"^NOMINAL_H\s*=\s*([0-9.]+)", src, re.M).group(1))
        assert abs(v - NOMINAL_H[key]) < 1e-9, (f, v)
    spot = dict(
        kind="spotp",
        max_cmd=dict(vx_fwd=1.0, vx_rev=1.0, vy=1.0, vz=0.0, wz=1.0),
        v_min=0.1,
        setpoint_ranges=dict(body_height=[0.25, 0.65], pitch=[-0.35, 0.35], roll=[-0.35, 0.35]),
    )
    g1 = dict(
        kind="g1a",
        max_cmd=dict(vx_fwd=0.5, vx_rev=0.5, vy=0.4, vz=0.0, wz=1.0),
        v_min=0.1,
        setpoint_ranges=dict(body_height=[0.3, 0.8], pitch=[-0.52, 1.57], roll=[-0.7, 0.7]),
    )
    jack = dict(kind="jackal", max_cmd=dict(vx_fwd=2.0, vx_rev=1.0, vy=0.0, vz=0.0, wz=2.0), v_min=0.1)
    rng = np.random.default_rng(0)
    for d in (spot, g1, jack):
        b = robot_box(d)
        a = rng.uniform(-1, 1, (10, 7)).astype(np.float32)
        a[:, 4] = a[:, 4] * 0.2
        y = normalise(a, b)
        a2 = denormalise(y, b)
        cols = (
            [0]
            + [c for c, k in ((1, "vy"), (2, "vz"), (3, "wz")) if b[k] > 1e-6]
            + [c for k, c in POSTURE_COL.items() if b[k] is not None]
        )
        err = np.abs(a2[:, cols] - np.clip(a[:, cols], -9, 9))
        ok = np.abs(y[:, cols]) < 2.0 - 1e-6
        assert (err[ok] < 1e-5).all() and np.isfinite(y).all(), (d["kind"], err.max())
        z = normalise(np.zeros((10, 4)), b)
        assert np.abs(z).max() < 1e-7  # zero-padded old chunk = stop + stock posture
        T = sp_tokens(b)
        assert T.shape == (N_SP, F_SP)
        print(
            f"[sp] {d['kind']}: box v={b['v']} bh={b['body_height']} pitch={b['pitch']} nominal={b['nominal']} "
            f"present={T[:, 3].astype(int).tolist()}"
        )
    b = robot_box(g1)
    lo_crouch = normalise(np.array([[0.3, 0, 0, 0, 0.30 - 0.75, 1.57, -0.7]]), b)
    assert np.allclose(lo_crouch[0, 4:], [-1, 1, -1]), lo_crouch
    assert abs(posture_dev_from_action(np.array([[0.5, 0, 0, 0, 0.0, 0.0, 0.0]]), b)) < 1e-9
    print("[sp] selftest OK")


if __name__ == "__main__":
    selftest()
