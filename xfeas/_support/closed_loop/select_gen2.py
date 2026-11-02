#!/usr/bin/env python
"""
Setpoint / macro selector for an XFeas-Gen2 checkpoint (format xfeas_gen2_v1). gen2 agent, 2026-10-14.

Candidates (all scored in ONE batch against one context; index 0 = STOCK, always a candidate):
  * 'go' setpoints: CEM over the robot's own box in normalised units (v, body height, pitch, roll), as select_gen;
  * SIDESTEP macros (skill 4) for side in {+1, -1} x the vy levels the oracle search used (G1 0.25/0.32/0.4, Spot
    0.3/0.5/0.8), gap centre from gap_estimate (perceived); act_v2 + chunk rendered by act_v2.py (GEN_V2_SPEC 3.3);
  * DETOUR (skill 7, lat_off +-0.4/0.7/1.0 bounded by the perceived lateral room) and GO2 programs (skill 3: slow
    approach, nose-up / crouch commit at d_commit, optional level-out) -- only when the checkpoint was trained on those
    skills (cfg 'trained_skills'; gen2a = gen_v1 data = go + sidestep only). Never extrapolated to untrained skills.
Decision rule = the LEARNED selection (learned_selection.json next to the checkpoint, fitted on validation situations
by eval_offline_gen2.py):
    gain_i = P_lcb(i) - P(stock);  a candidate may depart from stock only if gain_i > m;
    among the allowed: argmax  P_lcb(i) - w . [t_norm, sigmoid(energy), posture_dev (exact), 1 - tilt_margin, P_fall,
                                               P_slip * v / v_max]
Observation (dict): rgb (224,224,3), depth (224,224), prop_base (T,13), prop_joints (T,J,7), desc (json str), box (json
str, optional), robot (str), hist (optional list of (rgb, depth, age_s, (dx, dy, dyaw, dz) pose of the old root in the
current root frame)).
    sel = Selector2(ckpt); choice = sel.choose(obs)
    $PY xfeas/_support/closed_loop/select_gen2.py --ckpt C --serve     (framing of select_gen.py --serve)
    $PY xfeas/_support/closed_loop/select_gen2.py --ckpt C --test-npz some_episode.npz
"""

import os, sys  # noqa: E401  (standalone process, possibly in an isolated env: find the repo root itself)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import xfeas  # noqa: E402, F401  (puts every XFeas code folder on sys.path)
import argparse, io, json, math, os, struct, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import sp_tokens as SP
import cmap as CM
import act_v2 as AV
import gap_estimate as GE

SKILL_GO, SKILL_PASS_GAP = 0, 4
SIDESTEP_VY = {"g1": (0.25, 0.32, 0.4), "spot": (0.3, 0.5, 0.8)}
DEFAULT_SEL = dict(kappa=0.0, m=0.1, w=[0.0, 0.0, 0.2, 0.0, 0.2, 0.0], note="default (no learned_selection.json)")


def as_gen_v2_hist(rgb, depth):
    """closed-loop history frame (224) -> what training saw for gen_v2 history: 128 px (INTER_AREA rgb, min-pooled depth,
    GEN_V2_SPEC 5) resized back to 224 by dataset_gen2 (area for rgb, nearest for depth)"""
    import cv2

    rgb, depth = np.asarray(rgb), np.asarray(depth, np.float32)
    if rgb.shape[0] != 224:
        return rgb, depth
    r = cv2.resize(cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA), (224, 224), interpolation=cv2.INTER_AREA)
    d = cv2.erode(np.clip(np.nan_to_num(depth, posinf=20.0), 0, 20), np.ones((2, 2), np.uint8))
    d = cv2.resize(
        cv2.resize(d, (128, 128), interpolation=cv2.INTER_NEAREST), (224, 224), interpolation=cv2.INTER_NEAREST
    )
    return r, d


def sidestep_chunk(side, vy, g1):
    """physical (10, 7) chunk, identical layout to the gen_v1 sidestep rollouts (checked in selftest on a real npz)"""
    vxm = 0.1 if g1 else 0.0
    ch = np.zeros((SP.K_CHUNK, SP.A_DIM), np.float32)
    ch[0:3, 0] = vxm
    ch[0:3, 3] = -side * 1.0
    ch[3:9, 1] = side * vy
    ch[9, 0] = vxm
    ch[9, 3] = side * 1.0
    return ch


class Selector2:
    def __init__(self, ckpt, device="cuda:0", selection="", macros=True):
        import torch
        import model_gen2 as MG2
        import dataset_xe as DX
        import dataset_gen2 as DG2

        self.torch, self.dev, self.DX, self.MG2, self.DG2 = torch, device, DX, MG2, DG2
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert d.get("format") == MG2.FORMAT, f"{ckpt}: not an {MG2.FORMAT} checkpoint"
        self.cfg, self.norm = d["cfg"], d["norm"]
        self.models = []
        for sd in d["members"]:
            m = MG2.build(self.cfg, pretrained=False).to(device).eval()
            m.load_state_dict(sd)
            self.models.append(m)
        sp = selection or os.path.join(os.path.dirname(os.path.abspath(ckpt)), "learned_selection.json")
        S = json.load(open(sp)) if os.path.exists(sp) else {}
        pick = S.get("chosen") or (S.get(next(iter(S)), {}).get("insample") if S else None)
        self.sel = pick or DEFAULT_SEL
        self.sel_src = sp if pick else "default"
        self.macros = macros
        self.trained_heads = set(
            self.cfg.get("trained_heads", ["slippery"])
        )  # 'slip' only once gen_v2 slip labels trained
        self.trained_skills = set(
            self.cfg.get("trained_skills", ["go", "sidestep"])
        )  # gen_v1 training data: go + sidestep
        self.verbose = os.environ.get("XFEAS_GEN2_VERBOSE", "0") not in (
            "",
            "0",
        )  # additive: candidate table in the reply
        # FIX mode (GEN2_G1_DEBUG.md; XFEAS_GEN2_FIX=1, default 0 = gen2 as released): macro candidates only with
        # perceptual support (robust gap / discrete hazard); gates fitted on the TRAINING split, analysis/gen2_g1_fix/gate.json
        self.fix = os.environ.get("XFEAS_GEN2_FIX", "0") not in ("", "0")
        self.gate = (
            json.load(open(os.path.join(THIS, "..", "..", "analysis", "gen2_g1_fix", "gate.json")))
            if self.fix
            else None
        )

    def _std_max(self, skill, kind):
        """validation P99 of the ensemble std of this skill's rollouts for this robot (pooled over robots when < 30
        rows); analysis/gen2_g1_fix/gate.json 'std_max', fitted by gate_build.py on data/gen_train/gen2b val preds"""
        T = self.gate["std_max"]
        r = "g1c" if kind.startswith("g1") else ("spotp" if kind.startswith("spot") else kind)
        return float(T.get(f"{skill}|{r}", T.get(f"{skill}|all", 1.0)))

    def _fixed_gap(self, obs, dep0, kind):
        """FIX (GEN2_G1_DEBUG): robust gap = two flanking walls >= h_wall, width <= w_max (gap_gate_fit.py, training
        split) AND the same gap in the newest history frame (persistence: |de| < 0.15 m, |dx0| < 0.3 m after moving the
        old estimate into the current root frame); no history -> no sidestep yet"""
        G = self.gate
        g1 = kind.startswith("g1")
        wm = G["w_max"]["g1" if g1 else "spot"]
        cur = GE.robust_gap(dep0, kind, G["h_wall"], wm)
        log = dict(cur=dict(ok=bool(cur.get("ok")), why=cur.get("why"), feat=cur.get("feat")))
        if cur.get("ok"):
            hist = list(obs.get("hist") or [])
            if not hist:
                cur = dict(ok=False, why="no history frame for persistence")
            else:
                h = hist[0]
                old = GE.robust_gap(np.asarray(h[1], np.float32), kind, G["h_wall"], wm)
                if not old.get("ok"):
                    cur = dict(ok=False, why=f"not persistent (history: {old.get('why')})")
                else:
                    dx, dy, dyaw = float(h[3][0]), float(h[3][1]), float(h[3][2])
                    c, s_ = math.cos(dyaw), math.sin(dyaw)
                    x0o, eo = c * old["x0"] - s_ * old["e"] + dx, s_ * old["x0"] + c * old["e"] + dy
                    if abs(eo - cur["e"]) > 0.15 or abs(x0o - cur["x0"]) > 0.3:
                        cur = dict(
                            ok=False, why=f"not persistent (de {eo - cur['e']:+.2f}, dx0 {x0o - cur['x0']:+.2f})"
                        )
        log["final"] = dict(ok=bool(cur.get("ok")), why=cur.get("why"))
        self._gate_log = dict(self._gate_log or {}, gap=log)
        return cur

    def _inputs(self, obs, rgb):
        torch, DX, MG2 = self.torch, self.DX, self.MG2
        desc = json.loads(str(obs["desc"]))
        js, body = DX.featurize_descriptor(desc)
        b = SP.robot_box(desc, str(obs["box"]) if obs.get("box") is not None else None)
        kind = str(desc.get("kind", obs.get("robot", "?")))
        T = DX.T_HIST
        pb, pj = obs["prop_base"][-T:].astype(np.float32), obs["prop_joints"][-T:].astype(np.float32)
        if len(pb) < T:
            pad = T - len(pb)
            pb = np.concatenate([np.repeat(pb[:1], pad, 0), pb])
            pj = np.concatenate([np.repeat(pj[:1], pad, 0), pj])
        mode = self.cfg.get("prop_norm", "global")
        if mode != "none":
            Pn = self.norm["prop"]
            pn = Pn.get(str(obs.get("robot", "")), Pn["global"]) if mode == "robot" else Pn["global"]
            pb = np.clip((pb - np.float32(pn["base_mean"])) / np.float32(pn["base_std"]), -10, 10)
            pj = np.clip((pj - np.float32(pn["joint_mean"])) / np.float32(pn["joint_std"]), -10, 10)
        K = self.cfg.get("k_hist", MG2.K_HIST) if self.cfg.get("use_hist", True) else 1
        hist = list(obs.get("hist") or [])[: K - 1]
        frames = [(rgb, np.asarray(obs["depth"], np.float32), 0.0, None)] + [
            (*as_gen_v2_hist(h[0], h[1]), float(h[2]), h[3]) for h in hist
        ]
        Kf = len(frames)
        R = np.stack([f[0][..., :3] for f in frames]).astype(np.uint8).transpose(0, 3, 1, 2)
        D = np.stack([np.nan_to_num(f[1], posinf=60.0) for f in frames])[:, None]
        if self.cfg.get("use_map", True):
            M, info = CM.build_map(D[0, 0], kind, extra=[(f[1], kind, f[3]) for f in frames[1:] if f[3] is not None])
        else:
            M, info = np.zeros((CM.C_MAP, CM.NX, CM.NY), np.float32), dict(cam_h=0.0, used_prior=1.0)
        self._last_info = info
        span = max(f[2] for f in frames)
        meta = np.array(
            [
                float(Kf == 1),
                Kf / max(K, 1),
                info["used_prior"],
                info["cam_h"] / 1.5,
                float(self.cfg.get("use_map", True)),
                span / 2.0,
                0.0,
                0.0,
            ],
            np.float32,
        )
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x))[None].to(self.dev)
        x = dict(
            rgb=t(R),
            depth=t(D.astype(np.float32)),
            fmask=torch.ones(1, Kf, dtype=torch.bool, device=self.dev),
            fage=t(np.array([f[2] for f in frames], np.float32)),
            cmap=t(M),
            meta=t(meta),
            base=t(pb.astype(np.float32)),
            joints=t(pj.astype(np.float32)),
            jstatic=t(js),
            body=t(body),
            sp_tok=t(SP.sp_tokens(b)),
            jmask=torch.ones(1, js.shape[0], dtype=torch.bool, device=self.dev),
        )
        return x, b, kind, M

    @staticmethod
    def tta_views(rgb):
        g = np.repeat(rgb.mean(-1, keepdims=True), 3, -1).astype(np.uint8)
        return [rgb, rgb[..., [2, 0, 1]], rgb[..., [1, 2, 0]], g]

    def score(self, ctxs, chunks_n, skills, aps):
        torch = self.torch
        a = torch.from_numpy(chunks_n.astype(np.float32))[None].to(self.dev)
        s = torch.from_numpy(np.asarray(skills, np.int64))[None].to(self.dev)
        p_ = torch.from_numpy(aps.astype(np.float32))[None].to(self.dev)
        outs = []
        with torch.no_grad():
            for m, c in ctxs:
                o = m.score(c, a, s, p_)
                outs.append({k: v[0].float().cpu().numpy() for k, v in o.items()})
        sg = lambda x: 1 / (1 + np.exp(-x))
        p = np.stack([sg(o["success"]) for o in outs])
        r = {k: np.mean([o[k] for o in outs], 0) for k in ("t_norm", "energy", "posture_dev", "tilt_margin")}
        r.update(
            {f"p_{k}": np.mean([sg(o[k]) for o in outs], 0) for k in ("fall", "slip", "slippery", "collision", "stall")}
        )
        return dict(p_mean=p.mean(0), p_std=p.std(0), **r)

    def choose(self, obs, n=64, iters=2, seed=0, tta=True, sel=None):
        sel = sel or self.sel
        rng = np.random.default_rng(seed)
        ctxs, b, kind = [], None, None
        for rgb in self.tta_views(obs["rgb"]) if tta else [obs["rgb"]]:
            x, b, kind, M = self._inputs(obs, np.ascontiguousarray(rgb))
            _info = self._last_info
            with self.torch.no_grad():
                ctxs += [(m, m.encode_context(**x)) for m in self.models]
        v_lo, v_hi = b["v"][0] / b["vx_fwd"], b["v"][1] / b["vx_fwd"]
        lo = np.array([v_lo] + [-1.0 if b[k] is not None else 0.0 for k in SP.POSTURE])
        hi = np.array([v_hi] + [1.0 if b[k] is not None else 0.0 for k in SP.POSTURE])
        stock = np.array([b["nominal"]["v"] / b["vx_fwd"], 0.0, 0.0, 0.0])
        g1 = kind.startswith("g1")
        DG2 = self.DG2
        dep0 = np.asarray(obs["depth"], np.float32)
        macros = []  # (info dict, skill name, act_v2 A, mask M, physical chunk)
        self._gate_log = None
        if self.macros and "sidestep" in self.trained_skills and b["vy"] > 1e-3:
            gap = self._fixed_gap(obs, dep0, kind) if self.fix else GE.estimate(dep0, kind)
            if gap.get("ok"):
                for side in (1.0, -1.0):
                    for vy in SIDESTEP_VY["g1" if g1 else "spot"]:
                        if vy <= b["vy"] + 1e-6:
                            A, Mk = AV.make(
                                "sidestep",
                                v_nom=b["nominal"]["v"],
                                d_trig=0.6 if g1 else 1.1,
                                e_gap=gap["e"],
                                side=side,
                                vy=vy,
                            )
                            macros.append(
                                (
                                    dict(skill="pass_gap", side=side, vy=vy, gap=gap),
                                    "sidestep",
                                    A,
                                    Mk,
                                    AV.render("sidestep", A, g1=g1),
                                )
                            )
        hz = (
            GE.hazard_extent(dep0, kind)
            if self.macros and ({"go2", "detour"} & self.trained_skills)
            else dict(ok=False)
        )
        if self.fix and hz.get("ok"):  # FIX: go2 / detour need a discrete obstacle, not ground relief
            rh = GE.robust_hazard(dep0, kind, self.gate["hazard_h_min"].get("g1c" if g1 else "spotp", 0.0))
            self._gate_log = dict(
                self._gate_log or {}, hazard=dict(ok=bool(rh.get("ok")), why=rh.get("why"), h90=rh.get("h90"))
            )
            if not rh.get("ok"):
                hz = dict(ok=False, why=rh.get("why"))
        if hz.get("ok") and "detour" in self.trained_skills:
            for lat in (-1.0, -0.7, -0.4, 0.4, 0.7, 1.0):
                room = hz["room_left"] if lat > 0 else hz["room_right"]
                if abs(lat) <= room - 0.25:  # bounded by the perceived lateral room (spec: lat_room)
                    A, Mk = AV.make("detour", v=b["nominal"]["v"], lat_off=lat, L_ramp=1.2, s_back=0.5)
                    macros.append(
                        (
                            dict(skill="detour", lat_off=lat, hazard=hz),
                            "detour",
                            A,
                            Mk,
                            AV.render("detour", A, d_dec=hz["x0"], s_len=hz["x1"] - hz["x0"]),
                        )
                    )
        if hz.get("ok") and "go2" in self.trained_skills and b["pitch"] is not None:
            vA = max(b["v"][0], 0.3 * b["vx_fwd"])
            for dhB, pB in (
                (-0.12, b["pitch"][0]),
                (0.0, b["pitch"][0]),
                (-0.2, 0.0),
            ):  # nose-up commit / crouch-then-stand
                for d_commit in (0.3, 0.6):
                    for d_level in (0.0, 0.5):
                        dhB_ = (
                            max(dhB, b["body_height"][0] - b["nominal"]["body_height"])
                            if b["body_height"] is not None
                            else 0.0
                        )
                        A, Mk = AV.make(
                            "go2", v_a=vA, v_b=b["v"][1], dh_b=dhB_, pitch_b=pB, d_commit=d_commit, d_level=d_level
                        )
                        macros.append(
                            (
                                dict(skill="go2", A=np.round(A[:10], 3).tolist(), hazard=hz),
                                "go2",
                                A,
                                Mk,
                                AV.render("go2", A, d_dec=hz["x0"]),
                            )
                        )

        def go_chunks(X):
            ch = np.zeros((len(X), SP.K_CHUNK, SP.A_DIM), np.float32)
            ch[:, :, 0] = X[:, 0:1]
            ch[:, :, 4] = X[:, 1:2]
            ch[:, :, 5] = X[:, 2:3]
            ch[:, :, 6] = X[:, 3:4]
            return ch

        def go_ap(chn):
            out = []
            for c in chn:
                ph = SP.denormalise(c[0], b)
                A, Mk = AV.make("go", v=float(ph[0]), dh=float(ph[4]), pitch=float(ph[5]), roll=float(ph[6]))
                out.append(np.concatenate([DG2.norm_act_v2(A, b) * Mk, Mk]))
            return np.stack(out).astype(np.float32)

        def evaluate(X):
            chn = go_chunks(X)
            sk = [SKILL_GO] * len(X)
            aps = go_ap(chn)
            if macros:
                chn = np.concatenate([chn, np.stack([SP.normalise(mc[4], b) for mc in macros])])
                sk += [AV.SKILL[mc[1]] for mc in macros]
                aps = np.concatenate(
                    [
                        aps,
                        np.stack([np.concatenate([DG2.norm_act_v2(mc[2], b) * mc[3], mc[3]]) for mc in macros]).astype(
                            np.float32
                        ),
                    ]
                )
            S = self.score(ctxs, chn, sk, aps)
            phys = [SP.denormalise(c, b) for c in chn]
            pdev = np.array([SP.posture_dev_from_action(p, b) for p in phys])
            vabs = np.array([float(np.abs(p[:, 0]).max()) for p in phys])
            P = np.clip(S["p_mean"] - sel["kappa"] * S["p_std"], 0, 1)
            pslip = S["p_slip"] if "slip" in self.trained_heads else S["p_slippery"] * vabs / max(b["v"][1], 1e-3)
            F = np.stack(
                [
                    np.clip(S["t_norm"], 0, 3),
                    1 / (1 + np.exp(-S["energy"])),
                    pdev,
                    1 - np.clip(S["tilt_margin"], -1, 1),
                    S["p_fall"],
                    pslip,
                ],
                1,
            )
            U = P - F @ np.asarray(sel["w"], float)
            return S, P, U, F

        X = rng.uniform(lo, hi, (n, 4))
        X[0] = stock
        best = None
        for it in range(iters + 1):
            S, P, U, F = evaluate(X)
            if self.fix:
                # FIX (d): like with like -- the departure margin compares LCB with the stock LCB (not the stock mean),
                # and an EXTRAPOLATED macro (ensemble std above the validation P99 of its skill x robot) may not win
                allowed = (P - P[0]) > sel["m"]
                for k, mc in enumerate(macros):
                    j = len(X) + k
                    tau = self._std_max(mc[1], kind)
                    if S["p_std"][j] > tau:
                        allowed[j] = False
                        self._gate_log = dict(
                            self._gate_log or {},
                            n_extrapolated=(self._gate_log or {}).get("n_extrapolated", 0) + (it == 0),
                            std_max={mc[1]: tau},
                        )
            else:
                allowed = (P - S["p_mean"][0]) > sel["m"]
            allowed[0] = True
            Ua = np.where(allowed, U, -np.inf)
            i = int(np.argmax(Ua))
            if best is None or Ua[i] > best[0]:
                best = (Ua[i], i, X.copy(), S, P, U, F, allowed)
            if it == iters:
                break
            top = X[np.argsort(-np.where(allowed[:n], U[:n], U[:n] - 1.0))[: max(4, n // 4)]]
            X = np.clip(rng.normal(top.mean(0), np.maximum(top.std(0), 0.02 * (hi - lo + 1e-9)), (n, 4)), lo, hi)
            X[0] = stock
        _, i, Xb, S, P, U, F, allowed = best
        nom_h = b["nominal"]["body_height"]
        out = dict(
            p_success=round(float(P[i]), 3),
            p_std=round(float(S["p_std"][i]), 3),
            p_stock=round(float(S["p_mean"][0]), 3),
            gain=round(float(P[i] - S["p_mean"][0]), 3),
            utility=round(float(U[i]), 3),
            u_stock=round(float(U[0]), 3),
            p_fall=round(float(S["p_fall"][i]), 3),
            p_slip=round(float(S["p_slip"][i]), 3),
            p_slippery=round(float(S["p_slippery"][0]), 3),
            cost_terms=np.round(F[i], 3).tolist(),
            n_allowed=int(allowed.sum()),
            sel=sel,
            sel_src=self.sel_src,
            stock=bool(i == 0),
        )
        if self.verbose:  # (additive 2026-10-16, GEN2_G1_DEBUG) full candidate table
            ng = len(Xb)

            def row(j, name):
                return dict(
                    cand=name,
                    p_mean=round(float(S["p_mean"][j]), 3),
                    p_std=round(float(S["p_std"][j]), 3),
                    lcb=round(float(P[j]), 3),
                    gain=round(float(P[j] - S["p_mean"][0]), 3),
                    allowed=bool(allowed[j]),
                    U=round(float(U[j]), 3),
                    costs=np.round(F[j], 3).tolist(),
                )

            jg = int(np.argmax(U[:ng]))
            jga = int(np.argmax(np.where(allowed[:ng], U[:ng], -np.inf)))
            tab = [row(0, "stock")]
            for j, nm in ((jg, "go_bestU"), (jga, "go_best_allowed"), (int(np.argmax(P[:ng])), "go_bestP")):
                ph = SP.denormalise(go_chunks(Xb[j : j + 1])[0, 0], b)
                tab.append(
                    dict(
                        row(j, nm),
                        v=round(float(ph[0]), 3),
                        dh=round(float(ph[4]), 3),
                        pitch=round(float(ph[5]), 3),
                        roll=round(float(ph[6]), 3),
                    )
                )
            for k, mc in enumerate(macros):
                d_ = {kk: vv for kk, vv in mc[0].items() if kk not in ("gap", "hazard")}
                tab.append(dict(row(ng + k, mc[1]), **d_))
            out["verbose"] = dict(
                cands=tab,
                winner=int(i),
                winner_is_macro=bool(i >= ng),
                margin=sel["m"],
                kappa=sel["kappa"],
                gap=GE.estimate(dep0, kind) if (self.macros and "sidestep" in self.trained_skills) else None,
                hazard=hz,
                cmap=dict(
                    used_prior=round(float(_info["used_prior"]), 3),
                    cam_h=round(float(_info["cam_h"]), 3),
                    occ_frac=[round(float((M[c] > 0.5).mean()), 3) for c in range(min(M.shape[0], 6))],
                ),
                n_hist=len(list(obs.get("hist") or [])),
                gate=getattr(self, "_gate_log", None),
            )
        if i >= len(Xb):  # a macro
            mc = macros[i - len(Xb)]
            A = mc[2]
            out.update(
                mc[0],
                act_v2=np.round(A, 4).tolist(),
                v=round(float(A[0]), 3),
                dh=round(float(A[1]), 3),
                pitch=round(float(A[2]), 3),
                roll=round(float(A[3]), 3),
                body_height=round(float(nom_h + A[1]), 3) if b["body_height"] is not None else None,
            )
            return json.loads(json.dumps(out, default=float))
        phys = SP.denormalise(go_chunks(Xb[i : i + 1])[0, 0], b)
        out.update(
            skill="go",
            v=round(float(phys[0]), 3),
            dh=round(float(phys[4]), 3),
            body_height=round(float(nom_h + phys[4]), 3) if b["body_height"] is not None else None,
            pitch=round(float(phys[5]), 3),
            roll=round(float(phys[6]), 3),
            x_norm=np.round(Xb[i], 3).tolist(),
        )
        return out


def _read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


DUMP = os.environ.get("XFEAS_GEN2_DUMP", "")


def serve(sel):
    """request npz: rgb, depth, prop_base, prop_joints, desc, box, robot, task; optional hist_rgb (k,224,224,3),
    hist_depth (k,224,224), hist_age (k,), hist_pose (k,4). reply npz: choice (json)"""
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    serve.k = 0
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
    print("[select_gen2] ready", file=sys.stderr, flush=True)
    while True:
        try:
            n = struct.unpack("<I", _read(fin, 4))[0]
        except EOFError:
            return
        raw = _read(fin, n)
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        if DUMP:  # (additive, GEN2_G1_DEBUG) keep the request for offline replay
            open(os.path.join(DUMP, f"req_{os.getpid()}_{serve.k:05d}.npz"), "wb").write(raw)
            serve.k += 1
        obs = {k: z[k] for k in z.files}
        obs["desc"] = str(obs["desc"])
        task = json.loads(str(obs["task"])) if "task" in obs else {}
        if "hist_rgb" in obs and len(obs["hist_rgb"]):
            obs["hist"] = [
                (
                    obs["hist_rgb"][k],
                    obs["hist_depth"][k],
                    float(obs["hist_age"][k]),
                    tuple(map(float, obs["hist_pose"][k])),
                )
                for k in range(len(obs["hist_rgb"]))
            ]
        sp = task.get("setpoint", {}) if isinstance(task.get("setpoint", {}), dict) else {}
        choice = sel.choose(obs, n=sp.get("n", 64), iters=sp.get("iters", 2), seed=task.get("seed", 0))
        out = io.BytesIO()
        np.savez(out, choice=json.dumps(choice))
        b = out.getvalue()
        fout.write(struct.pack("<I", len(b)))
        fout.write(b)
        fout.flush()


def selftest():
    """sidestep chunk layout == a real gen_v1 sidestep rollout (both robots, both sides when present)"""
    import glob

    root = os.path.join(THIS, "..", "..", "data", "gen_v1")
    n = 0
    for rob in ("g1c", "spotp"):
        for f in sorted(glob.glob(os.path.join(root, "*", rob, "ep", "*_gap_*.npz")))[:400]:
            z = np.load(f, allow_pickle=False)
            if int(z["skill"]) != SKILL_PASS_GAP:
                continue
            P = json.loads(str(z["action_params"]))
            A, _ = AV.make("sidestep", v_nom=0.6, d_trig=1.1, side=float(P["side"]), vy=float(P["vy"]))
            ch = AV.render("sidestep", A, g1=rob.startswith("g1"))
            assert np.allclose(ch, z["action"], atol=1e-4), (f, ch, z["action"])
            n += 1
            if n >= 6:
                break
    print(f"[select_gen2] sidestep chunk layout matches {n} real rollouts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--selection", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--no-macros", action="store_true")
    ap.add_argument("--test-npz", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    sel = Selector2(a.ckpt, a.device, a.selection, macros=not a.no_macros)
    if a.serve:
        serve(sel)
    elif a.test_npz:
        z = np.load(a.test_npz, allow_pickle=False)
        obs = {k: z[k] for k in ("rgb", "depth", "prop_base", "prop_joints")}
        obs["desc"] = str(z["desc"])
        obs["robot"] = str(z["robot"])
        if "box" in z.files:
            obs["box"] = str(z["box"])
        print(json.dumps(sel.choose(obs, n=32, iters=1)))
