"""
v11 selector utility with a POSTURE-DEPARTURE COST (select_xe.choose_setpoint delegates here when the setpoint box
carries 'pcost' or 'gain_z'; without them select_xe is unchanged).

Why: for ReLIC Spot, crouching on open floor costs nothing physically (OVERHANG2_RESULTS: empty lane 0.92 vs 0.89,
crouched even slightly faster), so success labels cannot teach "do not crouch"; v8/v9 leave stock on empty lanes in
75-100 % of decisions. Departing from the controller's stock posture has costs the labels do not see (energy, joint
load, a lower camera/sensor view, reduced obstacle step-over height, predictability), so the layer should leave stock
only for a REAL predicted gain.

Utility (same LCB p as select_xe: p = mean - KAPPA * std over ensemble x colour views):
    U(a) = p (1 + BETA v / v_ref) - LAMBDA (1 - p)  -  pcost * dev(a)
    dev(a) = sum_k |x_k| / W_k,   k in (dh, pitch, roll),  W_k = width of THIS robot's setpoint box in that dimension
             (Spot: dh 0.40, pitch 0.70, roll 0.70; G1: dh 0.45, pitch 1.85, roll 0.70) -> dimensionless, per robot
Optional gate 'gain_z' (form B): a non-stock choice i is kept only if mean_i - mean_stock >= gain_z * sqrt(s_i^2 + s_stock^2)
(predicted gain larger than its uncertainty), else the best stock-posture candidate is returned.
The stock posture (dev 0) is always a candidate. Returned dict = select_xe's + u_stock, dev, gain_mean, gain_sd.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import numpy as np


def posture_dev(X, sp):
    W = np.array([max(sp[k][1] - sp[k][0], 1e-6) for k in ("dh", "pitch", "roll")], float)
    W = np.array(sp.get("pwidth", W), float)
    return np.sum(np.abs(X[:, 1:4]) / W[None, :], 1)


def choose_setpoint_pc(sel, obs, sp, n=64, iters=2, seed=0, tta=True):
    import course as CO
    import select_xe as SX

    rng = np.random.default_rng(seed)
    C = float(sp.get("pcost", 0.0))
    GZ = float(sp.get("gain_z", 0.0))
    ctxs = []
    for rgb in sel.tta_views(obs["rgb"]) if tta else [obs["rgb"]]:
        x = sel._inputs(dict(obs, rgb=np.ascontiguousarray(rgb)))
        with sel.torch.no_grad():
            ctxs += [(m, m.encode_context(**x)) for m in sel.models]
    keys = ["v", "dh", "pitch", "roll"]
    lo = np.array([sp[k][0] for k in keys], float)
    hi = np.array([sp[k][1] for k in keys], float)
    util = lambda p, X: p * (1 + SX.BETA * X[:, 0] / sp["v_ref"]) - SX.LAMBDA * (1 - p) - C * posture_dev(X, sp)
    X = rng.uniform(lo, hi, (n, 4))
    X[0] = np.clip([hi[0], 0, 0, 0], lo, hi)
    X[1] = np.clip([lo[0], 0, 0, 0], lo, hi)  # stock posture
    allX, allP, allM, allS = [], [], [], []
    for it in range(iters + 1):
        ch = np.zeros((len(X), 10, 7), np.float32)
        ch[:, :, 0] = X[:, 0:1]
        ch[:, :, 4] = X[:, 1:2]
        ch[:, :, 5] = X[:, 2:3]
        ch[:, :, 6] = X[:, 3:4]
        pm, ps = sel.p_success_chunks(ctxs, list(ch), [CO.SKILL_ID["go"]] * len(X))
        p = np.clip(pm - SX.KAPPA * ps, 0, 1)
        allX.append(X.copy())
        allP.append(p)
        allM.append(pm)
        allS.append(ps)
        if it == iters:
            break
        top = X[np.argsort(-util(p, X))[: max(4, n // 4)]]
        X = np.clip(rng.normal(top.mean(0), np.maximum(top.std(0), 0.02 * (hi - lo + 1e-9)), (n, 4)), lo, hi)
        X[0] = np.clip([top[0, 0], 0, 0, 0], lo, hi)
    X, P, M, S = map(np.concatenate, (allX, allP, allM, allS))
    u = util(P, X)
    i = int(np.argmax(u))
    stock = np.all(np.abs(X[:, 1:]) < 1e-9, 1)
    j = int(np.argmax(np.where(stock, u, -1e9)))
    g_mean = float(M[i] - M[j])
    g_sd = float(np.sqrt(S[i] ** 2 + S[j] ** 2))
    gated = False
    if GZ > 0 and not stock[i] and g_mean < GZ * g_sd:
        i, gated = j, True
    dev = posture_dev(X[i : i + 1], sp)[0]
    return dict(
        skill="go",
        v=round(float(X[i, 0]), 3),
        dh=round(float(X[i, 1]), 3),
        pitch=round(float(X[i, 2]), 3),
        roll=round(float(X[i, 3]), 3),
        p_success=round(float(P[i]), 3),
        p_std=round(float(S[i]), 3),
        utility=round(float(u[i]), 3),
        p_stock=round(float(P[j]), 3),
        v_stock=round(float(X[j, 0]), 3),
        u_stock=round(float(u[j]), 3),
        dev=round(float(dev), 3),
        gain_mean=round(g_mean, 3),
        gain_sd=round(g_sd, 3),
        gated=gated,
        pcost=C,
        gain_z=GZ,
    )
