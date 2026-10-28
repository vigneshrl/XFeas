#!/usr/bin/env python
"""
Selector for an XFeas-Gen3 checkpoint (format xfeas_gen3_v1). gen3 model agent, 2026-10-17. Contract: GEN3_SPEC.md.

One decision = score every candidate envelope program for THIS controller (spec + probe + history), gate, pick, and
hand the chosen env_act to the non-learned driver (Scripts/data_collector/driver.py; a reference implementation of spec 5 is
used only when that file does not exist yet, e.g. for unit tests).

Candidates (candidates_from_probe): the oracle-search vocabulary of GEN3_SPEC 9, built from the probe's ops only --
stock; constant speed variants over the measured vx range; constant crouch at every distinct walking top (+ lean when the
probe found a pitch op); width-limited constant; two_phase crouch-then-stand / slow-approach-then-commit; detour
+-{0.5, 0.8} within lat_room; turn_side_turn both sides x vy {0.6, 1.0} vy_max when a two-wall gap is perceived.

Decision rule (decide(); IDENTICAL code in the offline replay, eval_offline_gen3.py):
  1. hard gates, applied before any learned score (never to stock):
     * driver: turn_side_turn on a controller without measured lateral motion -> not proposed (driver_infeasible);
     * geometry ("can't fit -> don't propose"), only where the geometry is unambiguous: an overhead seen in the corridor
       (h_free_min < 9 at 0.2..5 m) and the driver-chosen op's margin_top < -geom_tol, or a two-wall gap seen and
       margin_wid < -geom_tol;
     * perceptual support (gen2 F1 / F2): turn_side_turn needs a two-wall gap with gap_w <= gap_w_max (closed loop: the
       same gap in the newest history frame, gap_estimate.robust_gap); two_phase / detour need a discrete hazard
       (h_ground_max >= hazard_h_min, or an overhead below the standing top);
  2. OOD fallback (gen2 F3, extended to EVERY candidate class incl. pure speed changes): ensemble std of P(success) above
     the validation P99 of its class -> disallowed;
  3. margin (gen2 F3): LCB(i) - LCB(stock) > m;
  4. among the allowed (stock always allowed): argmax LCB - w . [t_norm, env_dev, 1 - tilt_margin, P_fall, P_slip, P_stall].
  (kappa, m, w) and the gate thresholds are fitted on validation situations of the TRAINING controllers only
  (eval_offline_gen3.py -> learned_selection.json, gate.json next to the checkpoint). Re-decide after a stall (gen2 F4)
  is the closed-loop wrapper's job (5 s, < 0.2 m progress).
Rule baseline (rule_pick): crouch if the overhead clearance < the probe's standing top (fits_under_stock = 0) -> the
least-deviating constant crouch the driver says fits; sidestep if the gap < the standing width (fits_fwd_stock = 0) and
lateral motion was measured -> turn_side_turn; otherwise stock. Same probe, same driver, no learning.

    sel = Selector3(ckpt); out = sel.choose(obs, spec, probe)          obs keys: see build_obs()
    $PY closed_loop/select_gen3.py --ckpt C --serve               (length-prefixed npz, like select_gen2 --serve)
    $PY closed_loop/select_gen3.py --selftest
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, io, json, math, os, struct, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
sys.path.insert(0, THIS)
sys.path.insert(0, THIS)
import gen3_common as G

UNC = G.UNC
CLASSES = ("speed", "posture", "two_phase", "turn_side_turn", "detour")
FEATS = ("t_norm", "env_dev", "1-tilt_margin", "p_fall", "p_slip", "p_stall")
DEFAULT_SEL = dict(kappa=0.5, m=0.15, w=[0.05, 0.3, 0.1, 0.0, 0.1, 0.0], note="default (no learned_selection.json)")
DEFAULT_GATE = dict(geom_tol=0.05, tau={c: 1.0 for c in CLASSES}, gap_w_max=1.2, hazard_h_min=0.15, overhead_slack=0.02,
                    note="defaults (no gate.json): OOD gate effectively off")


# ------------------------------------------------------------------------------------------------ helpers
def cls_of(macro_id, A, stand_top=None):
    m = G.MACROS[int(macro_id)]
    if m != "constant":
        return m
    st = stand_top if stand_top is not None else UNC
    unconstrained = (A[1] >= UNC or A[1] >= st - 0.01) and A[2] >= UNC and abs(A[3]) < 1e-3
    return "speed" if unconstrained else "posture"


def feas_act_of_cand(cand):
    """the driver's feas_act (margin_top, margin_wid, fits, driver_feasible) back out of a cand_feats vector"""
    o = 52 + len(G.MACROS) + 6
    if cand[o + 4] < 0.5:
        return np.full(4, np.nan, np.float32)
    return np.asarray(cand[o:o + 4], np.float32)


def overhead_seen(feas):
    """an UNAMBIGUOUS overhead: seen in the corridor (0.2..5 m), clearance >= 0.3 m (lower = a wall / obstacle, not
    something to pass under; the lowest crawl top of any G1 controller doc'd is 0.34 m) over floor-level ground
    (h_ground_max < 0.3 m). On the gen_v3 smoke gap situations feas reports gap walls as h_free 0.14-0.20 over 1.74 m
    'ground' -- this keeps the hard gate off there."""
    return (feas is not None and np.isfinite(feas[0]) and feas[0] < UNC and 0.2 <= feas[1] <= 5.0 and feas[0] >= 0.3
            and np.isfinite(feas[4]) and feas[4] < 0.3)


def gap_seen(feas):
    """an UNAMBIGUOUS two-wall opening: gap_w in [0.15, 9) m. Narrower readings are walls / step faces, not openings
    (gen_v3 smoke climb situations report gap_w 0.02 m at the step)."""
    return feas is not None and np.isfinite(feas[2]) and 0.15 <= feas[2] < UNC


def hard_gates(macro_id, A, fa, feas, gate, lateral, stand_top):
    """-> reason string or '' (allowed). Never called for stock."""
    m = G.MACROS[int(macro_id)]
    if m == "turn_side_turn" and not lateral:
        return "driver_infeasible:no_lateral"
    if feas is None or not np.isfinite(feas).any():
        return "" if m in ("constant",) else "no_feas_for_macro"
    tol_t, tol_w = gate.get("geom_tol_top", gate["geom_tol"]), gate.get("geom_tol_wid", gate["geom_tol"])
    if np.isfinite(fa).all():
        clr = os.environ.get("XFEAS_GEN3_TOP_CLEAR", "")
        if clr:                                   # additive 2026-10-19: op's measured top < underside - fixed clearance
            if overhead_seen(feas) and fa[0] < float(clr):
                return "cant_fit_top"
        elif overhead_seen(feas) and fa[0] < -tol_t:
            return "cant_fit_top"
        if gap_seen(feas) and m != "detour" and fa[1] < -tol_w:
            return "cant_fit_width"
    if m == "turn_side_turn" and not (gap_seen(feas) and feas[2] <= gate["gap_w_max"]):
        return "no_two_wall_gap"
    if m in ("two_phase", "detour") and gate.get("hazard_gate", True):
        st = stand_top if stand_top is not None else (feas[5] + feas[0] if overhead_seen(feas) else UNC)
        hazard = (np.isfinite(feas[4]) and feas[4] >= gate["hazard_h_min"]) or (overhead_seen(feas) and feas[0] < st + gate["overhead_slack"])
        if not hazard:
            return "no_discrete_hazard"
    return ""


V_WALK_MIN = 0.08          # probe-measured walking speed below which an op is not offered (coordinator 2026-10-18; spec 'moving' = 0.05)


def op_vx_of_program(prog, probe_ops):
    """min probe-measured vx_ach over the forward-travel phases' ops of a driver program (A / B; nominal = no limit)"""
    vs = []
    for ph, v in (prog.get("phases") or {}).items():
        if ph in ("A", "B") and v.get("op_id") is not None and v.get("op_id") in probe_ops:
            o = probe_ops[v["op_id"]]
            if o.get("kind") in ("posture", "mode", "speed"):
                vs.append(G._f(o.get("vx_ach")))
    return min(vs) if vs else np.inf


def decide(pm, ps, F, macro, A, fa, feas, stock, sel, gate, lateral=True, stand_top=None, hard=True, op_vx=None):
    """-> (pick, allowed (n,), reasons list, U (n,), P (n,)). Shared by the online selector and the offline replay."""
    n = len(pm)
    P = np.clip(pm - sel["kappa"] * ps, 0, 1)
    U = P - np.asarray(F, float) @ np.asarray(sel["w"], float)
    allowed = np.ones(n, bool); why = [""] * n
    for i in range(n):
        if i == stock:
            why[i] = "stock"; continue
        r = hard_gates(macro[i], A[i], fa[i], feas, gate, lateral, stand_top) if hard else ""
        if hard and not r and op_vx is not None and op_vx[i] < gate.get("v_walk_min", V_WALK_MIN):
            r = "op_not_walking"                                   # the probe measured this posture barely moves
        if not r:
            c = cls_of(macro[i], A[i], stand_top)
            if ps[i] > gate["tau"].get(c, 1.0):
                r = f"ood:{c}"
            elif not (P[i] - P[stock] > sel["m"]):
                r = "margin"
        if r:
            allowed[i] = False; why[i] = r
    Ua = np.where(allowed, U, -np.inf)
    return int(np.argmax(Ua)), allowed, why, U, P


def cost_feats(S, idx=None):
    """per-candidate cost features (FEATS order) from prediction arrays (dict: r_t_norm, r_env_dev, r_tilt_margin,
    p_fall, p_slip, p_stall)"""
    g = lambda k, d: (S[k] if idx is None else S[k][idx]) if k in S else d
    n = len(S["p_success"]) if idx is None else len(idx)
    z = np.zeros(n)
    return np.stack([np.clip(g("r_t_norm", z), 0, 3), np.clip(g("r_env_dev", z), 0, 2), 1 - np.clip(g("r_tilt_margin", z + 1), -1, 1),
                     g("p_fall", z), g("p_slip", z), g("p_stall", z)], 1)


def rule_pick(macro, A, fa, feas, stock, lateral=True, v_nom=0.3):
    """the simple rule baseline (same probe / driver): see module doc. -> candidate index"""
    if feas is None or not np.isfinite(feas).any():
        return stock
    n = len(macro)
    fits = lambda i: np.isfinite(fa[i]).all() and fa[i][0] >= 0 and fa[i][1] >= 0
    if overhead_seen(feas) and feas[6] < 0.5:                        # stock does not fit under: crouch
        c = [i for i in range(n) if G.MACROS[int(macro[i])] == "constant" and A[i][1] < UNC and fits(i)]
        if not c:
            c = [i for i in range(n) if G.MACROS[int(macro[i])] == "constant" and A[i][1] < UNC]
            return min(c, key=lambda i: A[i][1]) if c else stock
        return max(c, key=lambda i: (round(float(A[i][1]), 3), -abs(A[i][0] - v_nom)))
    if gap_seen(feas) and feas[8] < 0.5:                             # gap narrower than the standing width: sidestep
        if lateral:
            c = [i for i in range(n) if G.MACROS[int(macro[i])] == "turn_side_turn"]
            if c:
                return min(c, key=lambda i: abs(A[i][12]))
        c = [i for i in range(n) if G.MACROS[int(macro[i])] == "constant" and fits(i)]
        return max(c, key=lambda i: (round(float(A[i][1]), 3), -abs(A[i][0] - v_nom))) if c else stock
    return stock


# ------------------------------------------------------------------------------------------------ driver (reference)
def act_dict(macro, A, stock=False):
    """env_act (16,) -> the envelope action dict of Scripts/data_collector/driver.py (inverse of driver.act_vec)"""
    A = [float(x) for x in A]
    a = dict(macro=macro, v_b=A[0], top_b=A[1], wid_b=A[2], pitch_b=A[3])
    if stock:
        a["stock"] = True
    if macro == "two_phase":
        a.update(v_a=A[4], top_a=A[5], wid_a=A[6], pitch_a=A[7], d_commit=A[8], d_level=A[9])
        if abs(A[11]) > 1e-9:
            a.update(yaw_off=A[11], d_yaw=A[15])
    elif macro == "turn_side_turn":
        side = -math.copysign(1.0, A[11]) if abs(A[11]) > 1e-9 else math.copysign(1.0, A[12] or 1.0)
        a.update(d_commit=A[8], e_gap=A[10], side=side, vy=abs(A[12]))
    elif macro == "detour":
        a.update(lat_off=A[10], L_ramp=A[13], s_back=A[14])
    return a


class OfficialDriver:
    """adapter: scripts/gen3/driver.Driver(spec, probe).plan(act dict) + feas.feas_act -> the RefDriver interface"""

    def __init__(self, spec, probe):
        p = os.path.join(ROOT, "Scripts", "data_collector")
        if p not in sys.path:
            sys.path.insert(0, p)
        import driver as DRV
        import feas as FE
        self.D, self.FE = DRV.Driver(spec, probe), FE

    def plan(self, macro, A, stock=False):
        prog = self.D.plan(act_dict(macro, A, stock))
        return dict(prog, drv=dict(pred=G.drv_pred(prog), driver_feasible=bool(prog.get("driver_feasible", True))))

    def feas_act(self, prog, feas):
        fd = {k: float(feas[i]) for i, k in enumerate(G.FEAS_KEYS)}
        return np.asarray(self.FE.feas_act(fd, prog, self.D), np.float32)


def get_driver(spec, probe):
    """the data agent's driver (Scripts/data_collector/driver.py, GEN3_SPEC 5) when present, else RefDriver"""
    if os.path.exists(os.path.join(ROOT, "Scripts", "data_collector", "driver.py")) and os.environ.get("GEN3_REF_DRIVER", "0") != "1":
        return OfficialDriver(spec, probe), "Scripts/data_collector/driver.py"
    return RefDriver(spec, probe), "RefDriver (select_gen3; reference only, Scripts/data_collector/driver.py missing)"


class RefDriver:
    """GEN3_SPEC section 5, op matching only (no command synthesis): -> per phase the chosen op and its predicted
    (top, width, depth, vx), driver_feasible. Used ONLY until Scripts/data_collector/driver.py exists."""

    def __init__(self, spec, probe):
        self.spec, self.ops = G.load_spec(spec), list((probe or {}).get("ops") or [])
        lx = G.spec_role(self.spec, "lin_vel_x")
        self.vmax = float(lx["range"][1]) if lx and lx.get("range") else 1.0
        st = [o for o in self.ops if o.get("kind") == "stand"]
        self.stand = st[0] if st else (self.ops[0] if self.ops else dict(top_min=1.3, width=0.5, depth=0.3))

    def _phase(self, v, top, wid, pitch, side):
        walk = [o for o in self.ops if o.get("moving") and not o.get("fell") and o.get("kind") in ("speed", "posture", "mode", "stand")]
        if side:
            walk = [o for o in self.ops if o.get("kind") in ("lateral", "stand") and not o.get("fell")]
        st = self.stand
        if not side and top >= G._f(st.get("top_min"), 1.3) and wid >= 9:
            return dict(op=st.get("op_id", 0), top=G._f(st.get("top_min")), width=G._f(st.get("width")), depth=G._f(st.get("depth")), vx=v, ok=True)
        if not walk:
            return dict(op=-1, top=G._f(st.get("top_min"), 1.3), width=G._f(st.get("width"), 0.5), depth=G._f(st.get("depth"), 0.3), vx=v, ok=False)
        ext = "depth" if side else "width"
        best, bad = None, None
        for o in walk:
            tm, ex = G._f(o.get("top_min"), 9), G._f(o.get(ext), 9)
            vcap = max(G._f(o.get("ratio")), 0.0) * self.vmax
            viol = max(0.0, tm - top - 0.01) + max(0.0, ex - wid - 0.01)
            c = max(0.0, top - tm) + 0.5 * abs(pitch - G._f(o.get("pitch_ach"))) + 0.3 * max(0.0, v - vcap) / max(v, 1e-3) + 0.5 * G._f(o.get("tilt_max"))
            rec = dict(op=o.get("op_id", -1), top=tm, width=G._f(o.get("width")), depth=G._f(o.get("depth")), vx=min(v, vcap) if vcap > 0 else v)
            if viol <= 0 and (best is None or c < best[0]):
                best = (c, rec)
            if bad is None or viol < bad[0]:
                bad = (viol, rec)
        if best:
            return dict(best[1], ok=True)
        return dict(bad[1], ok=False)

    def plan(self, macro, A):
        A = np.asarray(A, float)
        side = macro == "turn_side_turn"
        ph = dict(B=self._phase(A[0], A[1], A[2], A[3], side))
        if macro == "two_phase":
            ph["A"] = self._phase(A[4], A[5], A[6], A[7], False)
        ok = all(p["ok"] for p in ph.values())
        if side and not any(o.get("kind") == "lateral" for o in self.ops):
            ok = False
        return dict(phases=ph, drv=dict(pred=ph["B"], driver_feasible=bool(ok)), driver="RefDriver")


def feas_act_from(drv_pred, feas, macro, ok):
    hf = feas[0] if (feas is not None and np.isfinite(feas[0]) and feas[0] < UNC) else None
    gw = feas[2] if (feas is not None and np.isfinite(feas[2]) and feas[2] < UNC) else None
    ext = drv_pred["depth"] if macro == "turn_side_turn" else drv_pred["width"]
    mt = (hf - drv_pred["top"]) if hf is not None else 3.0
    mw = (gw - ext) if gw is not None else 3.0
    return np.array([mt, mw, float(mt >= 0 and mw >= 0), float(ok)], np.float32)


# ------------------------------------------------------------------------------------------------ candidates
def candidates_from_probe(spec, probe, feas=None, lat_room=(1.0, 1.0), gap_e=0.0, v_nom=None):
    """-> list of (macro, env_act (16,), mask (16,), info)"""
    spec = G.load_spec(spec)
    ops = list((probe or {}).get("ops") or [])
    fp, fpm = G.fp_vector(probe)
    fpd = {k: (float(fp[i]) if fpm[i] else None) for i, k in enumerate(G.FP_KEYS)}
    v_nom = v_nom or float((spec.get("nominal") or {}).get("speed", 0.3))
    stand_top = fpd["stand_top"] or 1.3
    out = []

    def add(macro, A, info):
        A = np.asarray(A, np.float32)
        M = np.zeros(16, np.float32); M[G.ENV_MASK[macro]] = 1
        if macro == "two_phase" and abs(A[11]) > 1e-9:
            M[[11, 15]] = 1
        for c in out:
            if c[0] == macro and np.allclose(c[1], A, atol=1e-4):
                return
        out.append((macro, A, M, info))

    def const(v, top=UNC + 0.9, wid=UNC + 0.9, pitch=0.0):
        A = np.zeros(16, np.float32); A[0:4] = [v, top, wid, pitch]; A[4:8] = A[0:4]
        return A
    add("constant", const(v_nom), dict(name="stock"))
    lo = fpd["vx_ach_lo"] or 0.5 * v_nom; hi = fpd["vx_ach_hi"] or 2 * v_nom
    for v in sorted({round(x, 3) for x in (lo, 0.5 * (lo + hi), hi, 0.6 * v_nom, 1.5 * v_nom)}):
        if v > 0.05 and abs(v - v_nom) > 0.02:
            add("constant", const(v), dict(name=f"speed {v}"))
    walk = [o for o in ops if o.get("moving") and not o.get("fell") and o.get("kind") in ("posture", "mode", "speed")]
    tops = sorted({round(G._f(o.get("top_min"), 9), 3) for o in walk if G._f(o.get("top_min"), 9) < stand_top - 0.03})
    for t in tops:
        for v in (v_nom, max(lo, 0.6 * v_nom)):
            add("constant", const(v, t + 0.01), dict(name=f"crouch top {t}"))
        pa = [G._f(o.get("pitch_ach")) for o in walk if abs(G._f(o.get("top_min"), 9) - t) < 0.005]
        if pa and max(pa) > 0.1:
            add("constant", const(v_nom, t + 0.01, pitch=max(pa)), dict(name=f"lean top {t}"))
    wids = sorted({round(G._f(o.get("width"), 9), 3) for o in walk if G._f(o.get("width"), 9) < (fpd["stand_width"] or 9) - 0.03})
    for w in wids[:3]:
        add("constant", const(v_nom, wid=w + 0.01), dict(name=f"narrow {w}"))
    for t in tops[:3]:
        for dc in (0.3, 0.6):
            for dl in (0.0, 0.5):
                A = const(v_nom, t + 0.01); A[4:8] = [v_nom, UNC + 0.9, UNC + 0.9, 0.0]; A[8], A[9] = dc, dl
                add("two_phase", A, dict(name=f"crouch-then-stand {t} dc {dc} dl {dl}"))
    for dc in (0.3, 0.6):
        A = const(hi); A[4:8] = [max(lo, 0.5 * v_nom), UNC + 0.9, UNC + 0.9, 0.0]; A[8] = dc
        add("two_phase", A, dict(name=f"slow-then-commit dc {dc}"))
    for lat in (-0.8, -0.5, 0.5, 0.8):
        room = lat_room[0] if lat > 0 else lat_room[1]
        if abs(lat) <= room - 0.25:
            A = const(v_nom); A[10], A[13], A[14] = lat, 1.2, 0.5; A[11] = math.atan2(lat, 1.2)
            add("detour", A, dict(name=f"detour {lat}"))
    vyL, vyR = fpd["vy_ach_L"] or 0.0, fpd["vy_ach_R"] or 0.0
    if max(abs(vyL), abs(vyR)) > 0.05 and gap_seen(feas):
        dep = fpd["min_depth_walk"] or fpd["stand_depth"] or 0.3
        for side in (1.0, -1.0):
            vym = abs(vyL if side > 0 else vyR)
            for f in (0.6, 1.0):
                A = np.zeros(16, np.float32); A[0], A[1], A[2] = v_nom, UNC + 0.9, dep + 0.01
                A[8], A[10], A[11], A[12] = 0.6, gap_e, -side * math.pi / 2, side * f * vym
                add("turn_side_turn", A, dict(name=f"side {side:+.0f} vy {f * vym:.2f}"))
    return out


# ------------------------------------------------------------------------------------------------ selector
class Selector3:
    def __init__(self, ckpt, device="cuda:0", selection="", gate="", agree_ckpt=""):
        import torch
        import model_gen3 as M3
        import dataset_gen3 as D3
        self.torch, self.dev, self.M3, self.D3 = torch, device, M3, D3
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert d.get("format") == M3.FORMAT, f"{ckpt}: not an {M3.FORMAT} checkpoint"
        self.cfg, self.norm = d["cfg"], d["norm"]
        self.models = []
        for sd in d["members"]:
            m = M3.build(self.cfg, pretrained=False).to(device).eval(); m.load_state_dict(sd); self.models.append(m)
        cd = os.path.dirname(os.path.abspath(ckpt))
        sp = selection or os.path.join(cd, "learned_selection.json")
        S = json.load(open(sp)) if os.path.exists(sp) else {}
        self.sel = S.get("chosen") or DEFAULT_SEL
        gp = gate or os.path.join(cd, "gate.json")
        self.gate = dict(DEFAULT_GATE, **(json.load(open(gp)) if os.path.exists(gp) else {}))
        self.src = dict(selection=sp if S else "default", gate=gp if os.path.exists(gp) else "default")
        # hybrid-agree (eval_offline_gen3 'hybrid|agree'): a departure of this model is executed only if the conservative
        # model (desc_only, with ITS fitted selection / gates) also allows that candidate; otherwise stock
        self.agree = Selector3(agree_ckpt, device) if agree_ckpt else None
        if self.agree is not None:
            self.src["agree"] = os.path.abspath(agree_ckpt)

    def scores(self, obs, spec, probe, C, FA, DV, MID):
        """ensemble scores of the candidate set C for THIS model (its own context / candidate featurisation)"""
        torch = self.torch
        x, feas = self.context(obs, spec, probe)
        CF = [G.cand_feats(Ak, Mk, MID[k], feas=feas, feas_act=FA[k], drv=DV[k], use_drv=self.cfg.get("use_drv", True))
              for k, (macro, Ak, Mk, info) in enumerate(C)]
        cand = torch.from_numpy(np.stack(CF))[None].to(self.dev); mid = torch.from_numpy(np.array(MID))[None].to(self.dev)
        outs = []
        with torch.no_grad():
            for m in self.models:
                o = m(cand=cand, macro_id=mid, **x); outs.append({k: v[0].float().cpu().numpy() for k, v in o.items()})
        sg = lambda z: 1 / (1 + np.exp(-z))
        p = np.stack([sg(o["success"]) for o in outs])
        S = dict(p_success=p.mean(0), p_std=p.std(0))
        for k in ("fall", "slip", "stall", "collision"):
            S[f"p_{k}"] = np.mean([sg(o[k]) for o in outs], 0)
        for k in ("t_norm", "env_dev", "tilt_margin", "cost", "progress"):
            S[f"r_{k}"] = np.mean([o[k] for o in outs], 0)
        S["ach"] = np.mean([o["ach"] for o in outs], 0)
        return S

    def context(self, obs, spec, probe):
        ctx, feas_raw = self.D3.build_context(obs, spec, probe, self.D3.g1_desc(), obs, self.norm, self.cfg, train=False)
        ctx["fmask"] = np.ones(len(ctx["fage"]), bool)                 # collate3 adds it in training
        t = lambda x: self.torch.from_numpy(np.ascontiguousarray(np.asarray(x)))[None].to(self.dev)
        return {k: t(v) for k, v in ctx.items()}, feas_raw

    def choose(self, obs, spec, probe, lat_room=(1.0, 1.0), gap_e=0.0, rule=False):
        """obs: dict with the gen_v3 obs keys (rgb, depth, prop_base, prop_joints, desc, [rgb_hist, depth_hist, t_hist,
        cam_hist, K128], hist_* (spec 6), feas (16,)). -> dict(macro, env_act, program, table, ...)"""
        torch = self.torch
        spec = G.load_spec(spec)
        drv, drv_src = get_driver(spec, probe)
        x, feas = self.context(obs, spec, probe)
        fp, fpm = G.fp_vector(probe)
        C = candidates_from_probe(spec, probe, feas, lat_room, gap_e)
        stand_top = float(fp[0]) if fpm[0] else None
        lateral = bool(fpm[10] and max(abs(fp[10]), abs(fp[11])) > 0.05)
        progs, FA, CF, MID, A, OPV, DV = [], [], [], [], [], [], []
        pops = {o.get("op_id"): o for o in (probe or {}).get("ops") or []}
        for k, (macro, Ak, Mk, info) in enumerate(C):
            pr = drv.plan(macro, Ak, stock=(k == 0)) if isinstance(drv, OfficialDriver) else drv.plan(macro, Ak)
            dv = pr.get("drv", {}) if isinstance(pr, dict) else {}
            if isinstance(drv, OfficialDriver) and feas is not None:
                fa = drv.feas_act(pr, feas)
            else:
                pred = dv.get("pred") or dict(top=float(fp[0]) if fpm[0] else 1.3, width=float(fp[1]) if fpm[1] else 0.5,
                                              depth=float(fp[2]) if fpm[2] else 0.3, vx=float(Ak[0]))
                fa = feas_act_from(pred, feas, macro, dv.get("driver_feasible", True))
            progs.append(pr); FA.append(fa); MID.append(G.MACRO_ID[macro]); A.append(Ak); DV.append(dv)
            OPV.append(op_vx_of_program(pr, pops) if k else np.inf)
        S = self.scores(obs, spec, probe, C, FA, DV, MID)
        FAa, Aa = np.stack(FA), np.stack(A)
        if rule:
            i = rule_pick(MID, Aa, FAa, feas, 0, lateral); allowed, why, U, P = None, None, None, None
        else:
            i, allowed, why, U, P = decide(S["p_success"], S["p_std"], cost_feats(S), MID, Aa, FAa, feas, 0, self.sel, self.gate, lateral, stand_top,
                                           op_vx=np.array(OPV))
            if self.agree is not None and i != 0:
                Sd = self.agree.scores(obs, spec, probe, C, FA, DV, MID)
                al_d = decide(Sd["p_success"], Sd["p_std"], cost_feats(Sd), MID, Aa, FAa, feas, 0, self.agree.sel, self.agree.gate, lateral,
                              stand_top, op_vx=np.array(OPV))[1]
                if not al_d[i]:
                    why = list(why); why[i] = "vetoed_by_conservative_model"; i = 0
        tab = [dict(name=C[j][3].get("name"), macro=C[j][0], p=round(float(S["p_success"][j]), 3), std=round(float(S["p_std"][j]), 3),
                    lcb=None if P is None else round(float(P[j]), 3), U=None if U is None else round(float(U[j]), 3),
                    allowed=None if allowed is None else bool(allowed[j]), why=None if why is None else why[j],
                    ach=np.round(S["ach"][j], 3).tolist(), feas_act=np.round(FAa[j], 3).tolist()) for j in range(len(C))]
        return json.loads(json.dumps(dict(macro=C[i][0], env_act=np.round(C[i][1], 4).tolist(), name=C[i][3].get("name"), stock=bool(i == 0),
                                          p_success=float(S["p_success"][i]), p_stock=float(S["p_success"][0]), program=progs[i], driver=drv_src,
                                          rule=bool(rule), sel=self.sel, gate_src=self.src, table=tab, feas=np.round(np.nan_to_num(np.asarray(feas, float) if feas is not None else np.zeros(16), nan=-1), 3).tolist(),
                                          stand_top=stand_top, lateral=lateral), default=float))


def _read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def serve(sel):
    """request npz: the obs keys of Selector3.choose + spec_json, probe_json, lat_room (2,), gap_e, rule (0/1).
    reply npz: choice (json)"""
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    print("[select_gen3] ready", file=sys.stderr, flush=True)
    while True:
        try:
            n = struct.unpack("<I", _read(fin, 4))[0]
        except EOFError:
            return
        z = np.load(io.BytesIO(_read(fin, n)), allow_pickle=False)
        obs = {k: z[k] for k in z.files}
        spec = json.loads(str(obs.pop("spec_json"))); probe = json.loads(str(obs.pop("probe_json")))
        # XFEAS_CAM_MOUNT=head (additive 2026-10-19): the decision frame's exact camera->world matrix + root pose; the
        # model's clearance map (cmap.build_map) then back-projects with it (cmap.set_mount). Absent = unchanged.
        mM, mB = obs.pop("cam_mount_M", None), obs.pop("cam_mount_base", None)
        mH = obs.pop("cam_mount_camh", None)                 # v4 camera_mount: per-frame floor prior (absent = per-kind prior)
        import cmap as _CM
        _CM.set_mount(mM, mB, cam_h=None if mH is None else float(mH)) if mM is not None else _CM.set_mount(None)
        try:
            ch = sel.choose(obs, spec, probe, tuple(map(float, obs.pop("lat_room", (1.0, 1.0)))), float(obs.pop("gap_e", 0.0)),
                            rule=bool(int(obs.pop("rule", 0))))
        finally:
            _CM.set_mount(None)
        out = io.BytesIO(); np.savez(out, choice=json.dumps(ch)); b = out.getvalue()
        fout.write(struct.pack("<I", len(b))); fout.write(b); fout.flush()


def selftest():
    """gates / decide / rule / candidates on a hand-made case (no checkpoint needed)"""
    spec = G.g1c_fallback_spec()
    probe = dict(fingerprint={"stand_top": 1.27, "stand_width": 0.48, "stand_depth": 0.3, "vx_ach_lo": 0.1, "vx_ach_hi": 0.4,
                              "vy_ach_L": 0.3, "vy_ach_R": 0.3, "min_depth_walk": 0.3},
                 ops=[dict(op_id=0, kind="stand", moving=False, top_min=1.27, width=0.48, depth=0.3),
                      dict(op_id=1, kind="posture", moving=True, ratio=0.6, top_min=1.02, width=0.45, depth=0.3, pitch_ach=0.0, tilt_max=0.2),
                      dict(op_id=2, kind="posture", moving=True, ratio=0.3, top_min=0.80, width=0.42, depth=0.3, pitch_ach=0.6, tilt_max=0.4),
                      dict(op_id=3, kind="lateral", moving=False, vy_ach=0.3, top_min=1.27, width=0.48, depth=0.3)])
    feas = np.array([1.10, 2.0, UNC + 0.9, 0, 0.0, 0.17, 0, 1, 1, 1, 0, 0.3, 3, 3, 1, 1], np.float32)
    C = candidates_from_probe(spec, probe, feas)
    drv = RefDriver(spec, probe)
    FA = np.stack([feas_act_from(drv.plan(c[0], c[1])["drv"]["pred"], feas, c[0], True) for c in C])
    MID = np.array([G.MACRO_ID[c[0]] for c in C]); A = np.stack([c[1] for c in C])
    assert C[0][3]["name"] == "stock" and not any(c[0] == "turn_side_turn" for c in C)        # no gap -> no side macro
    r = rule_pick(MID, A, FA, feas, 0)
    assert abs(A[r][1] - 1.03) < 1e-3, (r, A[r])                                               # highest crouch that fits 1.10
    pm = np.full(len(C), 0.5); pm[0] = 0.1
    ps = np.full(len(C), 0.01)
    gate = dict(DEFAULT_GATE, tau={c: 0.05 for c in CLASSES})
    i, allowed, why, U, P = decide(pm, ps, np.zeros((len(C), 6)), MID, A, FA, feas, 0, dict(kappa=0, m=0.1, w=[0] * 6), gate, True, 1.27)
    for j in range(len(C)):
        if j and cls_of(MID[j], A[j], 1.27) == "speed":
            assert why[j] == "cant_fit_top", (j, why[j])                                          # speed-only can't fit under 1.10
    ps2 = ps.copy(); ps2[:] = 0.2
    i2, al2, why2, _, _ = decide(pm, ps2, np.zeros((len(C), 6)), MID, A, FA, feas, 0, dict(kappa=0, m=0.1, w=[0] * 6), gate, True, 1.27)
    assert i2 == 0 and all(w.startswith(("ood", "cant", "stock", "no_", "driver")) for w in why2), why2  # OOD -> stock fallback
    feas_gap = feas.copy(); feas_gap[0] = UNC + 0.9; feas_gap[2] = 0.40; feas_gap[8] = 0
    Cg = candidates_from_probe(spec, probe, feas_gap)
    assert any(c[0] == "turn_side_turn" for c in Cg)
    FAg = np.stack([feas_act_from(drv.plan(c[0], c[1])["drv"]["pred"], feas_gap, c[0], True) for c in Cg])
    rg = rule_pick(np.array([G.MACRO_ID[c[0]] for c in Cg]), np.stack([c[1] for c in Cg]), FAg, feas_gap, 0, True)
    assert Cg[rg][0] == "turn_side_turn"
    print(f"[select_gen3] selftest OK ({len(C)} candidates on the overhead case, {len(Cg)} on the gap case)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--selection", default="")
    ap.add_argument("--gate", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--agree-ckpt", default="", help="hybrid-agree: conservative (desc_only) checkpoint that must also allow a departure")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    sel = Selector3(a.ckpt, a.device, a.selection, a.gate, agree_ckpt=a.agree_ckpt)
    if a.serve:
        serve(sel)
