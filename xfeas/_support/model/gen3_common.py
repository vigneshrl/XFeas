#!/usr/bin/env python
"""
Shared model-side definitions for XFeas-Gen3 (gen3 model agent, 2026-10-17), written against analysis/GEN3_SPEC.md v0.1.
Pure numpy. Everything that turns the spec's data contract into model inputs lives here, so the dataset, the
selector and the offline eval featurize IDENTICALLY.

  spec_tokens(spec)            controller spec JSON -> (N_SPEC_TOK, F_SPEC) tokens + mask. Only the spec's model inputs
                               (GEN3_SPEC 1): commands[*].{role, type, unit, range, nominal, values}, modes[*].{group, moving,
                               static, speed_range, uses_height, height_range}, rate_hz. Names / doc text are never read.
  fp_vector(probe)             32-d fingerprint (spec 4) in FP_KEYS order (NaN -> 0 + present mask)
  ops_tokens(probe)            the probe operating-point table (spec 4) -> (N_OPS, F_OPS) tokens + mask
  frame_feats(...)             one 10 Hz frame of h(t) / probe trace (spec 6) -> F_FRAME features
  history_feats(obs, spec)     obs/<sid>.npz history (60 frames) -> (60, F_FRAME) + valid
  probe_trace_feats(probe, spec) probe 'trace' -> (T_PROBE, F_FRAME) + valid
  cand_feats(env_act, mask, macro_id, feas, feas_act, drv) -> F_CAND candidate vector (env_act normalised, mask,
                               unconstrained flags, macro one-hot, requested margins vs the feas geometry, driver outputs)
  feas_feats(feas)             16 -> 32 (value with the 9.9 sentinels capped + sentinel flags)
  embodiment(desc)             robot descriptor -> canonical G1 joint statics (29, 5) + body (12) with the CONTROLLER's
                               max_cmd zeroed (it is a controller property; the spec carries it)
  canon_joints(names, x)       per-joint array -> canonical 29 G1 joints + mask (unknown names dropped)
  gen_v2 mapping               act_v2 (dh / pitch / roll) -> env_act (top / wid) for g1c rows via the AMO flat-probe table
                               (G1_OVERHANG.md section 1, training lanes of a TRAINING controller)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math
import numpy as np

# ------------------------------------------------------------------------------------------------ spec vocabulary
ROLES = (
    "lin_vel_x",
    "lin_vel_y",
    "yaw_rate",
    "pelvis_height",
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "speed",
    "mode",
)
HIST_CMD_ROLES = ("lin_vel_x", "lin_vel_y", "yaw_rate", "pelvis_height", "torso_pitch", "torso_roll", "mode", "speed")
CONT_CMD_COLS = (0, 1, 2, 3, 4, 5, 7)  # hist_cmd columns that are continuous
MODE_COL = 6
MACROS = ("constant", "two_phase", "turn_side_turn", "detour")
MACRO_ID = {m: i for i, m in enumerate(MACROS)}
ENV_ACT = (
    "v_b",
    "top_b",
    "wid_b",
    "pitch_b",
    "v_a",
    "top_a",
    "wid_a",
    "pitch_a",
    "d_commit",
    "d_level",
    "lat_off",
    "head_off",
    "vy",
    "L_ramp",
    "s_back",
    "d_yaw",
)
ENV_MASK = dict(
    constant=[0, 1, 2, 3],
    two_phase=list(range(10)),
    turn_side_turn=[0, 1, 2, 8, 10, 11, 12],
    detour=[0, 1, 2, 3, 10, 11, 13, 14],
)
UNC = 9.0  # >= 9 (9.9 in the spec) = unconstrained / nothing there
FP_KEYS = (
    "stand_top",
    "stand_width",
    "stand_depth",
    "vx_ach_lo",
    "vx_ach_mid",
    "vx_ach_hi",
    "vx_cmd_lo",
    "vx_cmd_mid",
    "vx_cmd_hi",
    "lag_tau",
    "vy_ach_L",
    "vy_ach_R",
    "lateral_needs_vx",
    "wz_ach",
    "turn_needs_vx",
    "min_top_walk",
    "v_at_min_top_walk",
    "width_at_min_top_walk",
    "depth_at_min_top_walk",
    "min_top_any",
    "min_width_walk",
    "min_depth_walk",
    "max_pitch_walk",
    "n_ops_walk",
    "n_top_levels",
    "posture_discrete",
    "has_height_cmd",
    "has_pitch_cmd",
    "has_lateral_cmd",
    "tilt_max_probe",
    "n_falls_probe",
    "top_span_walk",
)
N_FP = len(FP_KEYS)
FEAS_KEYS = (
    "h_free_min",
    "x_at_h_free",
    "gap_w_min",
    "x_at_gap",
    "h_ground_max",
    "stand_top_minus_h_free",
    "fits_under_stock",
    "fits_under_best",
    "fits_fwd_stock",
    "fits_fwd_best",
    "fits_side",
    "margin_top_best",
    "margin_w_fwd",
    "margin_w_side",
    "lateral_capable",
    "any_fit",
)
FEAS_GEOM = (0, 1, 2, 3, 4)  # probe-independent part (gen2b-style ablation keeps only this)
N_FEAS = len(FEAS_KEYS)
F_FEAS = 2 * N_FEAS
ACH = ("vx_b", "vy_b", "wz", "z_base", "top", "roll", "pitch", "yaw_rel", "tilt", "s", "e", "contacts")
ACH_HEAD = ("v_ach", "top_ach", "pitch_ach", "vy_ach")  # achievable-envelope head (phase B / side phase)
N_ACH_HEAD = len(ACH_HEAD)

# ------------------------------------------------------------------------------------------------ embodiment (same robot)
G1_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
NJ = len(G1_JOINTS)
_JIDX = {n: i for i, n in enumerate(G1_JOINTS)}


JOINT_ALIASES = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}  # Isaac Lab G1 (g1l) naming


def _jkey(n):
    n = str(n).lower()
    n = JOINT_ALIASES.get(n, n)
    for suf in ("_joint", ""):
        k = n if n.endswith("_joint") else n + "_joint"
        if k in _JIDX:
            return _JIDX[k]
    return None


def canon_index(names):
    """joint names -> (src indices, dst indices) into the canonical G1 29 (unknown names, e.g. hand joints, dropped)"""
    src, dst = [], []
    for i, n in enumerate(names):
        j = _jkey(n)
        if j is not None and j not in dst:
            src.append(i)
            dst.append(j)
    return np.array(src, int), np.array(dst, int)


def canon_joints(names, x, axis=-2):
    """x (..., J, C) per-joint -> (..., NJ, C) canonical + mask (NJ,) bool"""
    x = np.asarray(x)
    src, dst = canon_index(names)
    shp = list(x.shape)
    shp[axis] = NJ
    out = np.zeros(shp, np.float32)
    ix_o = [slice(None)] * len(shp)
    ix_i = [slice(None)] * len(shp)
    ix_o[axis] = dst
    ix_i[axis] = src
    out[tuple(ix_o)] = x[tuple(ix_i)]
    m = np.zeros(NJ, bool)
    m[dst] = True
    return out, m


def embodiment(desc):
    """descriptor dict -> jstatic (NJ, 5), body (12,), jmask (NJ,). Same features as dataset_xe.featurize_descriptor,
    but on the canonical G1 joint list, and with max_cmd (a CONTROLLER property) zeroed."""
    import dataset_xe as DX

    js, body = DX.featurize_descriptor(desc)
    jsc, jm = canon_joints(desc["joint_names"], js, axis=0)
    body = body.copy()
    body[4:9] = 0.0
    return jsc.astype(np.float32), body.astype(np.float32), jm


# ------------------------------------------------------------------------------------------------ spec tokens
N_CMD_MAX, N_MODE_MAX = 12, 32
N_SPEC_TOK = 1 + N_CMD_MAX + N_MODE_MAX
F_SPEC = 3 + len(ROLES) + 8  # token type (3) + role one-hot + 8 numeric


def _f(x, d=0.0):
    try:
        x = float(x)
        return x if math.isfinite(x) else d
    except (TypeError, ValueError):
        return d


def load_spec(s):
    return s if isinstance(s, dict) else json.loads(str(s))


def spec_role(spec, role):
    for c in spec.get("commands", []):
        if c.get("role") == role:
            return c
    return None


def spec_modes(spec):
    return list(spec.get("modes") or [])


def mode_feat(md):
    """mode dict -> (5,) moving, static, uses_height, speed mid, height mid (model inputs only)"""
    sr = md.get("speed_range") or None
    hr = md.get("height_range") or None
    return np.array(
        [
            float(bool(md.get("moving"))),
            float(bool(md.get("static"))),
            float(bool(md.get("uses_height"))),
            (_f(sr[0]) + _f(sr[1])) / 2 if sr else 0.0,
            (_f(hr[0]) + _f(hr[1])) / 2 if hr else 0.0,
        ],
        np.float32,
    )


def spec_tokens(spec, ranges_only=False):
    """-> tok (N_SPEC_TOK, F_SPEC), mask (N_SPEC_TOK,) bool. Token 0 = global, 1..N_CMD_MAX = commands, then modes.
    Commands are ordered by role (names are not inputs); modes keep spec order but carry no identity beyond their
    doc-declared properties and their group's relative index / size. ranges_only: drop the mode tokens and the per-mode
    doc facts (gen2b-style descriptor ablation: command ranges only)."""
    spec = load_spec(spec)
    T = np.zeros((N_SPEC_TOK, F_SPEC), np.float32)
    M = np.zeros(N_SPEC_TOK, bool)
    cmds = sorted(spec.get("commands", []), key=lambda c: ROLES.index(c["role"]) if c.get("role") in ROLES else 99)
    modes = [] if ranges_only else spec_modes(spec)
    nom = spec.get("nominal") or {}
    T[0, 0] = 1
    M[0] = True
    T[0, 3 + len(ROLES) :] = [
        _f(spec.get("rate_hz"), 50) / 100.0,
        len(cmds) / 8.0,
        len(modes) / 16.0,
        _f(nom.get("speed"), 0.3),
        float(any(c.get("role") == "mode" for c in cmds)),
        0,
        0,
        0,
    ]
    for k, c in enumerate(cmds[:N_CMD_MAX]):
        i = 1 + k
        T[i, 1] = 1
        M[i] = True
        if c.get("role") in ROLES:
            T[i, 3 + ROLES.index(c["role"])] = 1
        rg = c.get("range") or None
        cont = c.get("type", "continuous") == "continuous"
        T[i, 3 + len(ROLES) :] = [
            float(cont),
            _f(rg[0]) if rg else 0.0,
            _f(rg[1]) if rg else 0.0,
            _f(c.get("nominal")) if cont else 0.0,
            float(rg is not None),
            len(c.get("values") or []) / 32.0 if not cont else 0.0,
            0,
            0,
        ]
    groups = []
    for md in modes:
        g = md.get("group", "")
        if g not in groups:
            groups.append(g)
    for k, md in enumerate(modes[:N_MODE_MAX]):
        i = 1 + N_CMD_MAX + k
        T[i, 2] = 1
        M[i] = True
        sr = md.get("speed_range") or None
        hr = md.get("height_range") or None
        g = groups.index(md.get("group", ""))
        T[i, 3 : 3 + 8] = [
            float(bool(md.get("moving"))),
            float(bool(md.get("static"))),
            float(bool(md.get("uses_height"))),
            float(sr is not None),
            _f(sr[0]) if sr else 0.0,
            _f(sr[1]) if sr else 0.0,
            float(hr is not None),
            (_f(hr[0]) + _f(hr[1])) / 2 if hr else 0.0,
        ]
        gsz = sum(1 for m in modes if m.get("group", "") == md.get("group", "")) / max(len(modes), 1)
        T[i, 11:] = 0.0
        T[i, 11:13] = [g / max(len(groups), 1), gsz]
    return T, M


# ------------------------------------------------------------------------------------------------ probe
def fp_vector(probe):
    """-> fp (32,), present (32,) bool. Accepts probe['fingerprint'] (list, in fp_keys order) or a dict."""
    fp = np.zeros(N_FP, np.float32)
    pm = np.zeros(N_FP, bool)
    if probe is None:
        return fp, pm
    raw = probe.get("fingerprint") if isinstance(probe, dict) else probe
    keys = probe.get("fp_keys", FP_KEYS) if isinstance(probe, dict) else FP_KEYS
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = zip(keys, list(raw if raw is not None else []))
    for k, v in items:
        if k in FP_KEYS:
            x = _f(v, float("nan"))
            if math.isfinite(x):
                fp[FP_KEYS.index(k)] = x
                pm[FP_KEYS.index(k)] = True
    return fp, pm


N_OPS = 24
OP_KINDS = ("stand", "speed", "posture", "mode", "lateral", "turn")
F_OPS = len(OP_KINDS) + 12


def ops_tokens(probe):
    """probe ops table -> (N_OPS, F_OPS), mask. Spec-level command dicts are NOT inputs (names); the measured response is."""
    T = np.zeros((N_OPS, F_OPS), np.float32)
    M = np.zeros(N_OPS, bool)
    ops = (probe or {}).get("ops") or []
    for k, o in enumerate(ops[:N_OPS]):
        if str(o.get("kind")) in OP_KINDS:
            T[k, OP_KINDS.index(str(o["kind"]))] = 1
        T[k, len(OP_KINDS) :] = [
            float(bool(o.get("moving"))),
            _f(o.get("vx_cmd")),
            _f(o.get("vx_ach")),
            np.clip(_f(o.get("ratio")), -2, 3),
            _f(o.get("vy_ach")),
            _f(o.get("top_min")),
            _f(o.get("top_p50")),
            _f(o.get("width")),
            _f(o.get("depth")),
            _f(o.get("pitch_ach")),
            _f(o.get("tilt_max")),
            float(bool(o.get("fell"))),
        ]
        M[k] = True
    return T, M


# ------------------------------------------------------------------------------------------------ history frames
F_FRAME = 7 + 7 + 5 + 1 + 6 + 12 + 1 + 2 * NJ  # cmd cont + present, mode feat + present, env, ach, valid, q/dq
T_HIST = 60
T_PROBE = 320
SC_DQ = 0.1


ACH_SPEC_KEYS = (
    "vx_b",
    "vy_b",
    "wz",
    "z_base",
    "top",
    "roll",
    "pitch",
    "yaw_rel",
    "tilt",
    "s",
    "e",
    "foot_dz",
)  # spec v0.2


def ach_canon(ach, keys=None):
    """achieved channels -> the 12 model channels, location-free: vx_b, vy_b, wz, z_base, top, roll, pitch, yaw rate,
    tilt, forward rate, lateral rate, contacts / 2. The collector stores world x / y / yaw in channels 9 / 10 / 7
    (ach_keys of the obs: x, y, yaw, n_contact); spec v0.1 says s, e, yaw - lane yaw. Either way absolute positions /
    headings are turned into 10 Hz rates (m/s, rad/s) so no scene location leaks into the capability encoder."""
    a = np.nan_to_num(np.asarray(ach, np.float32), nan=0.0).copy()
    keys = list(keys) if keys else list(ACH_SPEC_KEYS)
    T = len(a)
    d = lambda x: (np.r_[x[1:2] - x[0:1] if T > 1 else [0.0], x[1:] - x[:-1]] * 10.0).astype(np.float32)
    out = a[:, :12].copy()
    if keys[9] in ("x", "s"):
        out[:, 9] = d(a[:, 9])
    if keys[10] in ("y",):
        out[:, 10] = d(a[:, 10])
    if keys[7] == "yaw":  # absolute heading (v0.1 smoke files): rate only
        y = np.unwrap(a[:, 7])
        out[:, 7] = d(y)
    # yaw_rel (spec v0.2: yaw - lane yaw) and e (lateral lane offset) are lane-relative and kept as they are
    k11 = keys[11] if len(keys) > 11 else ""
    out[:, 11] = a[:, 11] * 5.0 if k11 == "foot_dz" else (a[:, 11] / 2.0 if k11 in ("n_contact", "contacts") else 0.0)
    return np.clip(out, -5, 5)


def frames_feats(cmd, env, ach, valid, q, dq, jnames, spec, ach_keys=None):
    """arrays over T frames -> (T, F_FRAME) f32. cmd (T,8) by HIST_CMD_ROLES (NaN = role absent), env (T,4) or None,
    ach (T,12) or None, valid (T,), q / dq (T,J) or None with jnames."""
    spec = load_spec(spec)
    T = len(valid)
    X = np.zeros((T, F_FRAME), np.float32)
    o = 0
    if cmd is not None:
        cmd = np.asarray(cmd, np.float32)
        c = cmd[:, list(CONT_CMD_COLS)]
        pres = np.isfinite(c)
        X[:, 0:7] = np.where(pres, c, 0.0)
        X[:, 7:14] = pres
        modes = spec_modes(spec)
        mi = cmd[:, MODE_COL]
        for t in range(T):
            if np.isfinite(mi[t]) and 0 <= int(mi[t]) < len(modes):
                X[t, 14:19] = mode_feat(modes[int(mi[t])])
                X[t, 19] = 1
    o = 20
    if env is not None:
        e = np.nan_to_num(np.asarray(env, np.float32), nan=0.0)
        X[:, o + 0] = e[:, 0]
        X[:, o + 1] = np.minimum(e[:, 1], 1.6)
        X[:, o + 2] = np.minimum(e[:, 2], 1.2)
        X[:, o + 3] = e[:, 3]
        X[:, o + 4] = e[:, 1] >= UNC
        X[:, o + 5] = e[:, 2] >= UNC
    o += 6
    if ach is not None:
        vv = np.asarray(valid, bool)
        a = ach_canon(ach, ach_keys)
        if vv.any():  # rates across the spawn boundary are meaningless
            a[: int(np.argmax(vv)) + 1, [7, 9, 10]] = 0.0
        X[:, o : o + 12] = a
    o += 12
    X[:, o] = np.asarray(valid, np.float32)
    o += 1
    if q is not None and jnames is not None:
        qc, _ = canon_joints(jnames, np.asarray(q, np.float32)[..., None], axis=-2)
        dqc, _ = (
            canon_joints(jnames, np.asarray(dq, np.float32)[..., None], axis=-2)
            if dq is not None
            else (np.zeros_like(qc), None)
        )
        X[:, o : o + NJ] = np.clip(qc[..., 0], -5, 5)
        X[:, o + NJ : o + 2 * NJ] = np.clip(dqc[..., 0] * SC_DQ, -5, 5)
    X *= np.asarray(valid, np.float32)[:, None]
    return np.nan_to_num(X)


def history_feats(obs, spec):
    """obs npz dict (spec 6) -> (T_HIST, F_FRAME), valid (T_HIST,) bool; absent -> zeros / all invalid"""
    if obs is None or "hist_valid" not in obs:
        return np.zeros((T_HIST, F_FRAME), np.float32), np.zeros(T_HIST, bool)
    v = np.asarray(obs["hist_valid"]).astype(bool)[-T_HIST:]
    jn = json.loads(str(obs["joint_names"])) if "joint_names" in obs else None
    ak = json.loads(str(obs["ach_keys"])) if "ach_keys" in obs else None
    g = lambda k: np.asarray(obs[k])[-T_HIST:] if k in obs else None
    X = frames_feats(g("hist_cmd"), g("hist_env"), g("hist_ach"), v, g("hist_q"), g("hist_dq"), jn, spec, ach_keys=ak)
    if len(v) < T_HIST:
        pad = T_HIST - len(v)
        X = np.concatenate([np.zeros((pad, F_FRAME), np.float32), X])
        v = np.concatenate([np.zeros(pad, bool), v])
    return X, v


def probe_trace_feats(probe, spec, joint_names=None, ach_keys=None):
    """probe['trace'] (same channels as the history; spec 4) -> (T_PROBE, F_FRAME), valid. Accepted layouts: dict with
    cmd/hist_cmd (T,8), env/hist_env (T,4), ach/hist_ach (T,12), valid (T,), q/dq (T,J) + joint_names. (tbd in spec v0.1)"""
    tr = (probe or {}).get("trace") if isinstance(probe, dict) else None
    if not tr:
        return np.zeros((T_PROBE, F_FRAME), np.float32), np.zeros(T_PROBE, bool)
    g = lambda *ks: next((np.asarray(tr[k], np.float32) for k in ks if k in tr and tr[k] is not None), None)
    cmd, env, ach = (
        g("cmd", "hist_cmd", "cmd_trace3"),
        g("env", "hist_env", "env_trace"),
        g("ach", "hist_ach", "ach_trace"),
    )
    n = len(next(x for x in (cmd, ach, env) if x is not None))
    v = g("valid", "hist_valid")
    v = np.ones(n, bool) if v is None else v.astype(bool)
    q, dq = g("q", "hist_q"), g("dq", "hist_dq")
    jn = tr.get("joint_names") or (
        joint_names if (q is not None and joint_names is not None and q.shape[1] == len(joint_names)) else None
    )
    X = frames_feats(cmd, env, ach, v, q, dq, jn, spec, ach_keys=tr.get("ach_keys") or ach_keys)
    X, v = X[:T_PROBE], v[:T_PROBE]
    if len(v) < T_PROBE:
        pad = T_PROBE - len(v)
        X = np.concatenate([X, np.zeros((pad, F_FRAME), np.float32)])
        v = np.concatenate([v, np.zeros(pad, bool)])
    return X, v


# ------------------------------------------------------------------------------------------------ feas / candidates
def feas_feats(feas, geom_only=False):
    f = np.nan_to_num(np.asarray(feas if feas is not None else np.full(N_FEAS, np.nan), np.float32), nan=0.0)
    out = np.zeros(F_FEAS, np.float32)
    big = f >= UNC
    out[:N_FEAS] = np.clip(np.where(big, 3.0, f), -3, 3)
    out[N_FEAS:] = big
    if geom_only:
        keep = np.zeros(N_FEAS, bool)
        keep[list(FEAS_GEOM)] = True
        out[:N_FEAS] *= keep
        out[N_FEAS:] *= keep
    return out


F_CAND = 16 + 16 + 4 + 16 + len(MACROS) + 6 + 5 + 5  # = 72


def norm_env_act(A):
    A = np.asarray(A, np.float32)
    y = np.zeros(16, np.float32)
    for i in (0, 4, 12):
        y[i] = A[i]
    for i in (1, 5):
        y[i] = min(A[i], 1.6) / 1.3
    for i in (2, 6):
        y[i] = min(A[i], 1.2)
    for i in (3, 7, 10):
        y[i] = A[i]
    for i in (8, 9, 13, 14, 15):
        y[i] = A[i] / 3.0
    y[11] = A[11] / (math.pi / 2)
    return np.clip(np.nan_to_num(y), -3, 3)


def cand_feats(A, Amask, macro_id, feas=None, feas_act=None, drv=None, known=None, use_drv=True):
    """one candidate -> (F_CAND,). known (16,) = field values are known (gen_v2 mapping can't recover wid); default all.
    feas (16,) for requested-margin features; feas_act (4,) / drv (dict) are the driver's (probe-derived) outputs and are
    zeroed (present flag 0) when use_drv is False (gen2b-style ablation)."""
    A = np.asarray(A, np.float32)
    Am = np.asarray(Amask, np.float32)
    kn = np.ones(16, np.float32) if known is None else np.asarray(known, np.float32)
    x = np.zeros(F_CAND, np.float32)
    x[0:16] = norm_env_act(A) * Am * kn
    x[16:32] = Am
    x[32:36] = [float(A[1] >= UNC), float(A[5] >= UNC), float(A[2] >= UNC), float(A[6] >= UNC)]
    x[36:52] = kn
    o = 52
    x[o + int(macro_id)] = 1
    o += len(MACROS)
    if feas is not None:
        f = np.nan_to_num(np.asarray(feas, np.float32), nan=UNC + 1)
        hf, gw = f[0], f[2]
        for k, (lim, req) in enumerate(((hf, A[1]), (hf, A[5] if Am[5] else A[1]), (gw, A[2]))):
            ok = lim < UNC and req < UNC and kn[1 if k < 2 else 2] > 0
            x[o + 2 * k] = np.clip(lim - req, -1.5, 1.5) if ok else 0.0
            x[o + 2 * k + 1] = float(ok)
    o += 6
    if use_drv and feas_act is not None:
        fa = np.nan_to_num(np.asarray(feas_act, np.float32))
        x[o : o + 4] = np.clip(fa, -1.5, 1.5)
        x[o + 4] = 1
    o += 5
    d = drv_pred(drv) if (use_drv and drv) else {}
    if d:
        x[o : o + 4] = [_f(d.get("top")), min(_f(d.get("width")), 1.2), min(_f(d.get("depth")), 1.2), _f(d.get("vx"))]
        x[o + 4] = 1
    return x


def drv_pred(drv):
    """driver output -> the prediction of the op that crosses the hazard (phase S for turn_side_turn, else B); {} when the
    phase runs the nominal posture (stock: pred None). Accepts xfeas/_support/data_collection/driver.py programs / the rollout 'drv' json
    ({phases: {B: {pred: ...}}}) and flat {pred: ...} dicts."""
    if not isinstance(drv, dict):
        return {}
    ph = drv.get("phases")
    if isinstance(ph, dict) and ph:
        p = ph.get("S") or ph.get("B") or next(iter(ph.values()))
        return (p or {}).get("pred") or {}
    return drv.get("pred") or {}


def ach_targets(z, f):
    """rollout npz -> achieved envelope (4,) NaN where unknown: phase-B frames of env_trace (intent == the B request),
    skipping the first 1 s after the block starts and anything after a fall: median vx_b, median top, median pitch;
    turn_side_turn: median |vy_b| while a lateral command is sent."""
    y = np.full(N_ACH_HEAD, np.nan, np.float32)
    if "ach_trace" not in f or "env_trace" not in f:
        return y
    a = np.asarray(z["ach_trace"], np.float32)
    e = np.asarray(z["env_trace"], np.float32)
    A = np.asarray(z["env_act"], np.float32)
    n = min(len(a), len(e))
    if n < 8:
        return y
    a, e = a[:n], e[:n]
    tilt_bad = a[:, 8] > 0.7
    stop = int(np.argmax(tilt_bad)) if tilt_bad.any() else n
    sel = (np.abs(e[:, 0] - A[0]) < 0.02) & ((np.abs(e[:, 1] - A[1]) < 0.02) | ((e[:, 1] >= UNC) & (A[1] >= UNC)))
    sel[stop:] = False
    idx = np.where(sel)[0]
    if len(idx) >= 1:
        idx = idx[idx >= idx[0] + 10]
    if len(idx) >= 5:
        y[0] = np.median(a[idx, 0])
        y[1] = np.median(a[idx, 4])
        y[2] = np.median(a[idx, 6])
    if "cmd_trace3" in f and int(z["macro_id"]) == MACRO_ID["turn_side_turn"]:
        c = np.asarray(z["cmd_trace3"], np.float32)[:stop]
        lat = np.where(np.nan_to_num(np.abs(c[:, 1])) > 0.05)[0]
        if len(lat) >= 5:
            y[3] = np.median(np.abs(a[lat, 1]))
    return y


# ------------------------------------------------------------------------------------------------ gen_v2 (g1c) mapping
AMO_H = (0.30, 0.45, 0.60, 0.75)
AMO_TOP = (0.914, 1.023, 1.127, 1.266)  # G1_OVERHANG.md table 1, pitch 0 (AMO = g1c, a training controller)
AMO_PITCH = (-0.3, 0.0, 0.4, 0.6, 1.0, 1.2, 1.4)
AMO_PITCH_TOP_H030 = (1.049, 0.914, 0.837, 0.811, 0.734, 0.696, 0.662)
AMO_STAND_TOP = 1.266


def amo_top(h, pitch):
    """AMO commanded (body height, pitch) -> standing top (m), from the flat probe; the pitch effect measured at h 0.30
    is applied as an offset at other heights (approximation, gen_v2 auxiliary rows only)"""
    t0 = float(np.interp(h, AMO_H, AMO_TOP))
    dp = float(np.interp(pitch, AMO_PITCH, AMO_PITCH_TOP_H030)) - AMO_PITCH_TOP_H030[1]
    return t0 + dp


GEN2_MACRO = {"go": "constant", "go2": "two_phase", "sidestep": "turn_side_turn", "detour": "detour"}


def env_act_from_act_v2(A2, M2, skill_v2, nom_h=0.75):
    """gen_v2 act_v2 (dh, pitch, roll) -> (env_act, mask, known, macro). dh -> top via amo_top (unconstrained when the
    posture is nominal); roll -> wid is NOT recoverable (known = 0)."""
    A2 = np.asarray(A2, np.float32)
    M2 = np.asarray(M2, np.float32)
    macro = GEN2_MACRO.get(str(skill_v2), "constant")
    A = np.zeros(16, np.float32)
    kn = np.ones(16, np.float32)
    for o in (0, 4):
        A[o] = A2[o]
        dh, p = float(A2[o + 1]), float(A2[o + 2])
        A[o + 1] = UNC + 0.9 if (abs(dh) < 1e-3 and abs(p) < 1e-3) else amo_top(nom_h + dh, p)
        A[o + 2] = UNC + 0.9
        kn[o + 2] = 0.0
        A[o + 3] = p
    A[8:16] = A2[8:16]
    if macro == "turn_side_turn":
        A[1] = UNC + 0.9
    M = np.zeros(16, np.float32)
    M[ENV_MASK[macro]] = 1.0
    if macro == "two_phase" and abs(A[11]) > 1e-9:
        M[[11, 15]] = 1
    return A, M, kn, macro


def g1c_fallback_spec():
    """g1c (AMO) spec from the GEN3_SPEC section 1 table, used ONLY when scripts/gen3/specs/g1c.json is absent (gen_v2
    auxiliary rows). No measured numbers."""
    return {
        "schema": "xfeas.ctrl_spec/1",
        "ctrl_id": "g1c",
        "base": "g1c",
        "virtual": False,
        "robot": "unitree_g1",
        "rate_hz": 50,
        "nominal": {"speed": 0.3},
        "modes": [],
        "commands": [
            {"role": "lin_vel_x", "type": "continuous", "unit": "m/s", "range": [-0.5, 0.5], "nominal": 0.0},
            {"role": "lin_vel_y", "type": "continuous", "unit": "m/s", "range": [-0.4, 0.4], "nominal": 0.0},
            {"role": "yaw_rate", "type": "continuous", "unit": "rad/s", "range": [-1, 1], "nominal": 0.0},
            {"role": "pelvis_height", "type": "continuous", "unit": "m", "range": [0.30, 0.80], "nominal": 0.75},
            {"role": "torso_pitch", "type": "continuous", "unit": "rad", "range": [-0.52, 1.57], "nominal": 0.0},
            {"role": "torso_roll", "type": "continuous", "unit": "rad", "range": [-0.7, 0.7], "nominal": 0.0},
            {"role": "torso_yaw", "type": "continuous", "unit": "rad", "range": [-1.57, 1.57], "nominal": 0.0},
        ],
    }


def selftest():
    s = g1c_fallback_spec()
    T, M = spec_tokens(s)
    assert M.sum() == 1 + 7 and T.shape == (N_SPEC_TOK, F_SPEC)
    s2 = json.loads(json.dumps(s))
    s2["commands"] = s2["commands"][::-1]
    for c in s2["commands"]:
        c["name"] = "renamed_" + c["role"]
    assert np.allclose(spec_tokens(s2)[0], T), "spec tokens must not depend on names / order"
    assert abs(amo_top(0.75, 0.0) - 1.266) < 1e-6 and abs(amo_top(0.30, 1.4) - 0.662) < 1e-6
    A, Mk, kn, mac = env_act_from_act_v2(
        np.array([0.3, -0.15, 0, 0, 0.3, -0.15, 0, 0] + [0] * 8), np.r_[np.ones(4), np.zeros(12)], "go"
    )
    assert mac == "constant" and abs(A[1] - 1.127) < 1e-3 and kn[2] == 0
    x = cand_feats(A, Mk, 0, feas=np.r_[1.2, 2.0, UNC + 0.9, 0, 0.0, np.zeros(11)], known=kn)
    assert x.shape == (F_CAND,) and abs(x[56] - (1.2 - A[1])) < 1e-5 and x[57] == 1
    X, v = history_feats(None, s)
    assert X.shape == (T_HIST, F_FRAME) and not v.any()
    print(f"[gen3_common] selftest OK (F_SPEC {F_SPEC}, F_FRAME {F_FRAME}, F_CAND {F_CAND}, F_OPS {F_OPS})")


if __name__ == "__main__":
    import os, sys

    selftest()
