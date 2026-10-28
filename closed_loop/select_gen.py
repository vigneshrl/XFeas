#!/usr/bin/env python
"""
Setpoint selector for an XFeas-Gen checkpoint (train_gen.py). Searches the robot's OWN box in normalised units
(sp_tokens: v in [v_lo / vx_fwd, 1], each present posture dim in [-1, 1], nominal = 0) by CEM, scores candidates with
the cost-aware utility picked on validation (utility.json next to the checkpoint; success-only fall-back if absent):

    U = P * (1 - c * cost) - lambda * (1 - P),   P = clip(mean - kappa * std) over (ensemble x colour views)

and DE-NORMALISES the winner to physical setpoints {v, body_height, dh, pitch, roll}. The stock setpoint (v_nom, nominal
posture) is always a candidate (index 0); when no candidate has U > 0 the choice is the STOCK setpoint with refuse=True
(same rule as pick_utility.py; the caller may still decide to stop).

    sel = Selector("data/gen_train/NAME/ckpt/xfeas_gen_ens.pt"); choice = sel.choose_setpoint(obs)
    obs: rgb (H,W,3) uint8, depth (H,W), prop_base (T,13), prop_joints (T,J,7), desc (json str), optional box (json str),
         optional robot (name for per-robot proprio stats)
Server (same framing as select_xe.py --serve; task json {"setpoint": {...optional overrides}}):
    $XFEAS_PY closed_loop/select_gen.py --ckpt ... --serve
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, io, json, os, struct, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, THIS)
import sp_tokens as SP

SKILL_GO = 0


class Selector:
    def __init__(self, ckpt, device="cuda:0", utility=""):
        import torch
        import model_gen as MG
        import dataset_xe as DX
        self.torch, self.dev, self.DX = torch, device, DX
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert d.get("format") == MG.FORMAT, f"{ckpt}: not an {MG.FORMAT} checkpoint"
        self.cfg, self.norm = d["cfg"], d["norm"]
        self.models = []
        for sd in d["members"]:
            m = MG.build(self.cfg, pretrained=False).to(device).eval(); m.load_state_dict(sd); self.models.append(m)
        up = utility or os.path.join(os.path.dirname(os.path.abspath(ckpt)), "utility.json")
        U = json.load(open(up)) if os.path.exists(up) else {}
        self.util = U.get("chosen") or dict(c=0.0, lam=1.5, kappa=1.0, w=dict(t=0.25, e=0.25, p=0.25, m=0.25))
        self.util_status = U.get("status", "missing (success-only fall-back)")

    def _inputs(self, obs, rgb):
        torch, DX = self.torch, self.DX
        desc = json.loads(str(obs["desc"]))
        js, body = DX.featurize_descriptor(desc)
        b = SP.robot_box(desc, str(obs["box"]) if obs.get("box") is not None else None)
        T = DX.T_HIST
        pb, pj = obs["prop_base"][-T:].astype(np.float32), obs["prop_joints"][-T:].astype(np.float32)
        if len(pb) < T:
            pad = T - len(pb)
            pb = np.concatenate([np.repeat(pb[:1], pad, 0), pb]); pj = np.concatenate([np.repeat(pj[:1], pad, 0), pj])
        mode = self.cfg.get("prop_norm", "global")
        if mode != "none":
            Pn = self.norm["prop"]
            pn = Pn.get(str(obs.get("robot", "")), Pn["global"]) if mode == "robot" else Pn["global"]
            pb = np.clip((pb - np.float32(pn["base_mean"])) / np.float32(pn["base_std"]), -10, 10)
            pj = np.clip((pj - np.float32(pn["joint_mean"])) / np.float32(pn["joint_std"]), -10, 10)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x))[None].to(self.dev)
        x = dict(rgb=t(rgb.transpose(2, 0, 1)), depth=t(np.asarray(obs["depth"], np.float32)[None]), base=t(pb.astype(np.float32)),
                 joints=t(pj.astype(np.float32)), jstatic=t(js), body=t(body), sp_tok=t(SP.sp_tokens(b)),
                 jmask=torch.ones(1, js.shape[0], dtype=torch.bool, device=self.dev))
        return x, b

    @staticmethod
    def tta_views(rgb):
        g = np.repeat(rgb.mean(-1, keepdims=True), 3, -1).astype(np.uint8)
        return [rgb, rgb[..., [2, 0, 1]], rgb[..., [1, 2, 0]], g]

    def score(self, ctxs, chunks_n):
        """chunks_n (N, K, 7) normalised -> dict of (N,) arrays: P mean/std over members x views, cost-head means"""
        torch = self.torch
        a = torch.from_numpy(chunks_n.astype(np.float32))[None].to(self.dev)
        s = torch.full((1, len(chunks_n)), SKILL_GO, dtype=torch.long, device=self.dev)
        outs = []
        with torch.no_grad():
            for m, c in ctxs:
                o = m.score(c, a, s); outs.append({k: v[0].cpu().numpy() for k, v in o.items()})
        p = np.stack([1 / (1 + np.exp(-o["success"])) for o in outs])
        r = {k: np.mean([o[k] for o in outs], 0) for k in ("t_norm", "energy", "posture_dev", "tilt_margin")}
        return dict(p_mean=p.mean(0), p_std=p.std(0), **r)

    def utility(self, S, util):
        w = util["w"]; ws = np.array([w["t"], w["e"], w["p"], w["m"]], float)
        feat = np.stack([np.clip(S["t_norm"], 0, 3), 1 / (1 + np.exp(-S["energy"])), np.clip(S["posture_dev"], 0, 1),
                         1 - np.clip(S["tilt_margin"], -1, 1)], 1)
        cost = feat @ ws / max(ws.sum(), 1e-9)
        P = np.clip(S["p_mean"] - util["kappa"] * S["p_std"], 0, 1)
        return P * (1 - util["c"] * cost) - util["lam"] * (1 - P), P, cost

    def choose_setpoint(self, obs, n=64, iters=2, seed=0, tta=True, util=None):
        torch = self.torch
        util = util or self.util
        rng = np.random.default_rng(seed)
        ctxs, b = [], None
        for rgb in (self.tta_views(obs["rgb"]) if tta else [obs["rgb"]]):
            x, b = self._inputs(obs, np.ascontiguousarray(rgb))
            with torch.no_grad():
                ctxs += [(m, m.encode_context(**x)) for m in self.models]
        v_lo, v_hi = b["v"][0] / b["vx_fwd"], b["v"][1] / b["vx_fwd"]
        lo = np.array([v_lo] + [-1.0 if b[k] is not None else 0.0 for k in SP.POSTURE])
        hi = np.array([v_hi] + [1.0 if b[k] is not None else 0.0 for k in SP.POSTURE])
        stock = np.array([b["nominal"]["v"] / b["vx_fwd"], 0.0, 0.0, 0.0])

        def chunks(X):
            ch = np.zeros((len(X), SP.K_CHUNK, SP.A_DIM), np.float32)
            ch[:, :, 0] = X[:, 0:1]; ch[:, :, 4] = X[:, 1:2]; ch[:, :, 5] = X[:, 2:3]; ch[:, :, 6] = X[:, 3:4]
            return ch
        X = rng.uniform(lo, hi, (n, 4)); X[0] = stock
        allX, allU, allP, allC, allS = [], [], [], [], []
        for it in range(iters + 1):
            S = self.score(ctxs, chunks(X))
            if util.get("pdev", "pred") == "exact":      # gen1 fix: exact posture_dev of each candidate, not the head
                S["posture_dev"] = np.array([SP.posture_dev_from_action(SP.denormalise(ch, b), b) for ch in chunks(X)])
            U, P, C = self.utility(S, util)
            allX.append(X.copy()); allU.append(U); allP.append(P); allC.append(C); allS.append(S["p_std"])
            if it == iters:
                break
            top = X[np.argsort(-U)[: max(4, n // 4)]]
            X = np.clip(rng.normal(top.mean(0), np.maximum(top.std(0), 0.02 * (hi - lo + 1e-9)), (n, 4)), lo, hi)
            X[0] = stock
        X, U, P, C, Sd = map(np.concatenate, (allX, allU, allP, allC, allS))
        i = int(np.argmax(U))
        refuse = bool(U[i] <= 0)
        if refuse:                                   # nothing worth it: fall back to the stock setpoint (pick_utility.py rule)
            i = 0
        phys = SP.denormalise(chunks(X[i:i + 1])[0, 0], b)
        nom_h = b["nominal"]["body_height"]
        return dict(skill="go", v=round(float(phys[0]), 3), dh=round(float(phys[4]), 3),
                    body_height=round(float(nom_h + phys[4]), 3) if b["body_height"] is not None else None,
                    pitch=round(float(phys[5]), 3), roll=round(float(phys[6]), 3), x_norm=np.round(X[i], 3).tolist(),
                    p_success=round(float(P[i]), 3), p_std=round(float(Sd[i]), 3), cost=round(float(C[i]), 3),
                    utility=round(float(U[i]), 3), refuse=refuse, p_stock=round(float(P[0]), 3),
                    u_stock=round(float(U[0]), 3), util=util, util_status=self.util_status)


def _read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def serve(sel):
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    print("[select_gen] ready", file=sys.stderr, flush=True)
    while True:
        try:
            n = struct.unpack("<I", _read(fin, 4))[0]
        except EOFError:
            return
        z = np.load(io.BytesIO(_read(fin, n)), allow_pickle=False)
        obs = {k: z[k] for k in z.files}
        obs["desc"] = str(obs["desc"]); task = json.loads(str(obs["task"])) if "task" in obs else {}
        sp = task.get("setpoint", {}) if isinstance(task.get("setpoint", {}), dict) else {}
        choice = sel.choose_setpoint(obs, n=sp.get("n", 64), iters=sp.get("iters", 2), seed=task.get("seed", 0))
        out = io.BytesIO()
        np.savez(out, choice=json.dumps(choice), cand_v=np.zeros(1, np.float32), cand_p=np.zeros(1, np.float32))
        b = out.getvalue()
        fout.write(struct.pack("<I", len(b))); fout.write(b); fout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--test-npz", default="", help="score one episode npz as an observation and print the choice")
    a = ap.parse_args()
    sel = Selector(a.ckpt, a.device)
    if a.serve:
        serve(sel)
    elif a.test_npz:
        z = np.load(a.test_npz, allow_pickle=False)
        obs = {k: z[k] for k in ("rgb", "depth", "prop_base", "prop_joints")}
        obs["desc"] = str(z["desc"]); obs["robot"] = str(z["robot"])
        if "box" in z.files:
            obs["box"] = str(z["box"])
        print(json.dumps(sel.choose_setpoint(obs, n=32, iters=1)))
