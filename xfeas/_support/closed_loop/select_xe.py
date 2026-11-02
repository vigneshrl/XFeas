#!/usr/bin/env python
"""
XFeas ACTION SELECTOR (+ sidecar server).

Given one observation (RGB-D, proprio history, robot descriptor) and a task, choose a
CONTINUOUS action by scoring candidates with the model:

    candidates   v ~ U[v_min, vx_max] (the task fixes the skill, and for 'turn' the yaw rate)
    score        U(a) = P_succ(a) * (1 + BETA * v / vx_max) - LAMBDA * (1 - P_succ(a))
                 (succeed first, then prefer faster; a failure costs LAMBDA)
    refine       CEM: keep the top quarter, resample around them, twice
    refuse       U(refuse) = 0. If no candidate beats not attempting, the robot does not go --
                 that is the correct behaviour for a body that cannot make it (Carter at a
                 0.5 m gap, H1 at a curb), and it is the model's call, not a script's.

The context (vision, proprio, robot) is encoded ONCE; all candidates are scored in one batch.

Server mode (spawned by demo_xe.py from Isaac's python, which lacks timm/DINOv3):
    in  : uint32 n + .npz bytes {rgb, depth, prop_base, prop_joints, desc(json), task(json)}
    out : uint32 n + .npz bytes {choice(json), cand_v, cand_p}

    $XFEAS_PY xfeas/_support/closed_loop/select_xe.py --ckpt data/xe_train/xfeas_....pt --serve
"""

import os, sys  # noqa: E401  (standalone process, possibly in an isolated env: find the repo root itself)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import xfeas  # noqa: E402, F401  (puts every XFeas code folder on sys.path)
import argparse, io, json, os, struct, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))

BETA, LAMBDA = 0.5, 1.5
KAPPA = 1.0  # act on a LOWER CONFIDENCE BOUND: P_lcb = mean - KAPPA * std over (ensemble x
# colour views). An ensemble that disagrees is not a license to go; a 0.54
# mean from members that split is a 'don't know', not a coin flip to take.
TASK_SKILL = {"brake": ("stop_at_line", 1), "turn": ("go", 0), "gap": ("pass_gap", 4), "curb": ("climb_step", 3)}
K = 10


def chunk(task, v, wz):
    """Same encoding as skills.encode (kept dependency-free here)."""
    ch = np.zeros((K, 4), np.float32)
    if task == "turn":
        ch[:, 0] = v
        ch[4:, 3] = wz
    elif task == "brake":
        ch[:7, 0] = v
    else:
        ch[:, 0] = v
    return ch


class Selector:
    def __init__(self, ckpt, backbone="dinov3_s", device="cuda:0"):
        import torch
        import model_xe as MX

        self.torch, self.dev = torch, device
        sds = torch.load(ckpt, map_location="cpu")
        sds = sds if isinstance(sds, list) else [sds]
        self.models = []
        for sd in sds:
            m = MX.XFeas(backbone=backbone, pretrained=False).to(device).eval()
            MX.load_compat(m, sd)
            self.models.append(m)

    def _inputs(self, obs):
        import dataset_xe as DX

        torch = self.torch
        js, body = DX.featurize_descriptor(json.loads(obs["desc"]))
        T = DX.T_HIST
        pb, pj = obs["prop_base"][-T:], obs["prop_joints"][-T:]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None].to(self.dev)
        return dict(
            rgb=t(obs["rgb"].transpose(2, 0, 1)),
            depth=t(obs["depth"][None].astype(np.float32)),
            base=t(pb.astype(np.float32)),
            joints=t(pj.astype(np.float32)),
            jstatic=t(js),
            jmask=torch.ones(1, js.shape[0], dtype=torch.bool, device=self.dev),
            body=t(body),
        )

    def p_success(self, ctxs, task, vs, wz):
        torch = self.torch
        acts = np.stack([chunk(task, v, wz) for v in vs])
        a = torch.from_numpy(acts)[None].to(self.dev)
        s = torch.full((1, len(vs)), TASK_SKILL[task][1], dtype=torch.long, device=self.dev)
        with torch.no_grad():
            ps = [torch.sigmoid(m.score(c, a, s)["success"])[0].cpu().numpy() for m, c in ctxs]
        return np.mean(ps, 0), np.std(ps, 0)

    @staticmethod
    def tta_views(rgb):
        """Colour-permuted / greyscale copies: the prediction must not hinge on hue."""
        g = np.repeat(rgb.mean(-1, keepdims=True), 3, -1).astype(np.uint8)
        return [rgb, rgb[..., [2, 0, 1]], rgb[..., [1, 2, 0]], g]

    def choose(self, obs, task, v_min, v_max, wz=0.0, n=64, iters=2, seed=0, tta=True):
        torch = self.torch
        rng = np.random.default_rng(seed)
        ctxs = []  # one per (model, view); p_success averages all
        for rgb in self.tta_views(obs["rgb"]) if tta else [obs["rgb"]]:
            x = self._inputs(dict(obs, rgb=np.ascontiguousarray(rgb)))
            with torch.no_grad():
                ctxs += [(m, m.encode_context(**x)) for m in self.models]
        util = lambda p, v: p * (1 + BETA * v / v_max) - LAMBDA * (1 - p)
        vs = rng.uniform(v_min, v_max, n)
        all_v, all_p, all_s = [], [], []
        for it in range(iters + 1):
            pm, ps = self.p_success(ctxs, task, vs, wz)
            p = np.clip(pm - KAPPA * ps, 0.0, 1.0)
            all_v += vs.tolist()
            all_p += p.tolist()
            all_s += ps.tolist()
            if it == iters:
                break
            top = vs[np.argsort(-util(p, vs))[: max(4, n // 4)]]
            mu, sd = top.mean(), max(top.std(), 0.03)
            vs = np.clip(rng.normal(mu, sd, n), v_min, v_max)
        all_v, all_p, all_s = np.array(all_v), np.array(all_p), np.array(all_s)
        u = util(all_p, all_v)
        i = int(np.argmax(u))
        if u[i] <= 0.0:
            choice = dict(
                skill="refuse",
                v=0.0,
                wz=0.0,
                p_success=round(float(all_p.max()), 3),
                utility=0.0,
                best_v=round(float(all_v[i]), 3),
                p_std=round(float(all_s[i]), 3),
            )
        else:
            choice = dict(
                skill=TASK_SKILL[task][0],
                v=round(float(all_v[i]), 3),
                wz=float(wz),
                p_success=round(float(all_p[i]), 3),
                utility=round(float(u[i]), 3),
                p_std=round(float(all_s[i]), 3),
            )
        order = np.argsort(all_v)
        return choice, all_v[order], all_p[order]

    # ------------------------------------------------------------ exploration course
    def p_success_chunks(self, ctxs, chunks, sids):
        torch = self.torch
        a = torch.from_numpy(np.stack(chunks).astype(np.float32))[None].to(self.dev)
        s = torch.tensor(sids, dtype=torch.long, device=self.dev)[None]
        with torch.no_grad():
            ps = [torch.sigmoid(m.score(c, a, s)["success"])[0].cpu().numpy() for m, c in ctxs]
        return np.mean(ps, 0), np.std(ps, 0)

    def choose_course(self, obs, alts, v_ref, n=48, iters=2, seed=0, tta=True):
        """alts: [{skill, v:[lo,hi], wz:[lo,hi]|None, wz_sign, task: bool}]. Task alternatives
        (go / stop / turn / pass_gap / reverse_out) earn progress; a non-task back_off is the
        physical way to decline -- chosen only when nothing is worth the risk AND reversing is
        predicted safe, else the robot just stands (refuse)."""
        import course as CO

        rng = np.random.default_rng(seed)
        ctxs = []
        for rgb in self.tta_views(obs["rgb"]) if tta else [obs["rgb"]]:
            x = self._inputs(dict(obs, rgb=np.ascontiguousarray(rgb)))
            with self.torch.no_grad():
                ctxs += [(m, m.encode_context(**x)) for m in self.models]
        util = lambda p, v: p * (1 + BETA * v / v_ref) - LAMBDA * (1 - p)
        best, abort, curves = None, None, {}
        for alt in alts:
            lo, hi = alt["v"]
            wlo, whi = alt["wz"] if alt.get("wz") else (0.0, 0.0)
            sgn = alt.get("wz_sign", 1.0)
            vs, ws = rng.uniform(lo, hi, n), rng.uniform(wlo, whi, n)
            AV, AW, AP, AS = [], [], [], []
            for it in range(iters + 1):
                ch = [CO.encode(dict(skill=alt["skill"], v=v, wz=sgn * w))[0] for v, w in zip(vs, ws)]
                sid = CO.SKILL_ID[alt["skill"]]
                pm, ps = self.p_success_chunks(ctxs, ch, [sid] * len(vs))
                p = np.clip(pm - KAPPA * ps, 0, 1)
                AV += vs.tolist()
                AW += ws.tolist()
                AP += p.tolist()
                AS += ps.tolist()
                if it == iters:
                    break
                u = util(p, vs) if alt.get("task", True) else p
                top = np.argsort(-u)[: max(4, n // 4)]
                vs = np.clip(rng.normal(vs[top].mean(), max(vs[top].std(), 0.03), n), lo, hi)
                ws = np.clip(rng.normal(ws[top].mean(), max(ws[top].std(), 0.03), n), wlo, whi)
            AV, AW, AP, AS = map(np.array, (AV, AW, AP, AS))
            u = util(AP, AV) if alt.get("task", True) else AP - 1.0  # never beats a task
            i = int(np.argmax(u))
            cand = dict(
                skill=alt["skill"],
                v=round(float(AV[i]), 3),
                wz=round(float(sgn * AW[i]), 3),
                p_success=round(float(AP[i]), 3),
                p_std=round(float(AS[i]), 3),
                utility=round(float(u[i]), 3),
                task=bool(alt.get("task", True)),
            )
            order = np.argsort(AV)
            curves[alt["skill"] + ("" if alt.get("task", True) else "/abort")] = (AV[order], AP[order])
            if alt.get("task", True) and (best is None or cand["utility"] > best["utility"]):
                best = cand
            if not alt.get("task", True):
                abort = cand
        if best is not None and best["utility"] > 0:
            return best, curves
        if abort is not None and abort["p_success"] > 0.5:
            return dict(abort, skill="back_off", abort=True, reason="no safe forward option"), curves
        return dict(
            skill="refuse", v=0.0, wz=0.0, p_success=best["p_success"] if best else 0.0, reason="no safe option"
        ), curves


def choose_setpoint(sel, obs, sp, n=64, iters=2, seed=0, tta=True):
    """Feasibility-layer setpoint search. sp: {v: [lo, hi], dh: [lo, hi] (body height - nominal), pitch: [lo, hi],
    roll: [lo, hi], v_ref}. Dimensions with lo == hi are fixed (e.g. masked on a velocity-only robot). CEM over
    the box with the same LCB utility as speed selection; the default posture (dh = pitch = roll = 0) is always a
    candidate, so the layer only departs from the controller's stock behaviour when it predicts a gain."""
    if sp.get("pcost") or sp.get("gain_z"):  # v11 (additive): posture-departure cost / gain-vs-uncertainty gate
        import select_pcost as SPC  # xfeas/_support/closed_loop/select_pcost.py; absent keys -> unchanged below

        return SPC.choose_setpoint_pc(sel, obs, sp, n=n, iters=iters, seed=seed, tta=tta)
    import course as CO

    rng = np.random.default_rng(seed)
    ctxs = []
    for rgb in sel.tta_views(obs["rgb"]) if tta else [obs["rgb"]]:
        x = sel._inputs(dict(obs, rgb=np.ascontiguousarray(rgb)))
        with sel.torch.no_grad():
            ctxs += [(m, m.encode_context(**x)) for m in sel.models]
    keys = ["v", "dh", "pitch", "roll"]
    lo = np.array([sp[k][0] for k in keys], float)
    hi = np.array([sp[k][1] for k in keys], float)
    util = lambda p, v: p * (1 + BETA * v / sp["v_ref"]) - LAMBDA * (1 - p)
    X = rng.uniform(lo, hi, (n, 4))
    X[0] = np.clip([hi[0], 0, 0, 0], lo, hi)
    X[1] = np.clip([lo[0], 0, 0, 0], lo, hi)  # stock posture
    allX, allP, allS = [], [], []
    for it in range(iters + 1):
        ch = np.zeros((len(X), 10, 7), np.float32)
        ch[:, :, 0] = X[:, 0:1]
        ch[:, :, 4] = X[:, 1:2]
        ch[:, :, 5] = X[:, 2:3]
        ch[:, :, 6] = X[:, 3:4]
        pm, ps = sel.p_success_chunks(ctxs, list(ch), [CO.SKILL_ID["go"]] * len(X))
        p = np.clip(pm - KAPPA * ps, 0, 1)
        allX.append(X.copy())
        allP.append(p)
        allS.append(ps)
        if it == iters:
            break
        top = X[np.argsort(-util(p, X[:, 0]))[: max(4, n // 4)]]
        X = np.clip(rng.normal(top.mean(0), np.maximum(top.std(0), 0.02 * (hi - lo + 1e-9)), (n, 4)), lo, hi)
        X[0] = np.clip([top[0, 0], 0, 0, 0], lo, hi)
    X, P, S = np.concatenate(allX), np.concatenate(allP), np.concatenate(allS)
    u = util(P, X[:, 0])
    i = int(np.argmax(u))
    stock = np.all(np.abs(X[:, 1:]) < 1e-9, 1)
    j = int(np.argmax(np.where(stock, u, -1e9)))
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
    )


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
    print("[select] ready", file=sys.stderr, flush=True)
    while True:
        try:
            n = struct.unpack("<I", _read(fin, 4))[0]
        except EOFError:
            return
        z = np.load(io.BytesIO(_read(fin, n)), allow_pickle=False)
        obs = {k: z[k] for k in z.files}
        obs["desc"] = str(obs["desc"])
        task = json.loads(str(obs["task"]))
        out = io.BytesIO()
        if "setpoint" in task:  # feasibility layer: speed + posture setpoints
            choice = choose_setpoint(sel, obs, task["setpoint"], seed=task.get("seed", 0))
            np.savez(out, choice=json.dumps(choice), cand_v=np.zeros(1, np.float32), cand_p=np.zeros(1, np.float32))
        elif "alts" in task:  # exploration-course segment
            choice, curves = sel.choose_course(obs, task["alts"], task["v_ref"], seed=task.get("seed", 0))
            main = next((k for k in curves if not k.endswith("/abort")), None)
            cv, cp = curves[main] if main else (np.zeros(1), np.zeros(1))
            np.savez(out, choice=json.dumps(choice), cand_v=cv.astype(np.float32), cand_p=cp.astype(np.float32))
        else:
            choice, cv, cp = sel.choose(
                obs, task["task"], task["v_min"], task["v_max"], task.get("wz", 0.0), seed=task.get("seed", 0)
            )
            np.savez(out, choice=json.dumps(choice), cand_v=cv.astype(np.float32), cand_p=cp.astype(np.float32))
        b = out.getvalue()
        fout.write(struct.pack("<I", len(b)))
        fout.write(b)
        fout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="dinov3_s")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    sel = Selector(a.ckpt, a.backbone, a.device)
    if a.serve:
        serve(sel)
