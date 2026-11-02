#!/usr/bin/env python
"""The model server: a separate PyTorch process that loads a trained model, builds the candidate commands and their
descriptors for the current controller, scores them and returns the choice. `--selftest` checks a model offline.

Details:
Online selector for the gen5 model trained on gen4 side-by-side data (trainer_gen4data.py; 2026-10-27, closed-loop test).
Same process / IPC pattern as xfeas/_support/closed_loop/select_gen4.py (length-prefixed npz over stdin / stdout, gsam2 env, the job's
own GPU), but NOTHING of gen4's decision code is used: no select_gen3 candidates, no driver programs, no gates, no rule.

One request = the gen4-format obs dict the closed-loop layer builds every 0.25 s (gen4_methods._Gen4Layer perception:
decision frame, 4-frame memory picked by distance, 2 s proprio, h(t) above the support floor, feas v0.4, camera matrix of
the shared mount) + ctrl_id / spec_path / probe_path / robot. The model input is built EXACTLY as trainer_gen4data.SitItems
(augment off):
    ctx, feas_raw = dataset_gen4.build_context(o, spec, probe, dataset5.robot_desc5(robot), norm, cfg, train=False, ...)
    state = [1.0, 0.0]   (trainer_gen4data.STATE0)
    cand  = descriptors.Describer(spec, probe, None).describe(cands, cur_cmd=NOMINAL, cur_speed=nominal speed)
            (as gen4_option_dataset.build: the command in force is NOT fed; the model never saw it)
then the trainer_gen4data.collate layout (B = 1) and trainer_gen4data.run (bf16 autocast). Score (trainer_gen4data.evaluate):
    U = mean_k P(success) - kappa std_k P(success) - lam mean_k P(fall)   (kappa, lam from the checkpoint cfg; 0.5 / 0.5)
    pick = argmax U over the deployment candidates. No threshold, no margin over stock, no gate.

Deployment candidates per controller (deploy_cands): the DISTINCT commands gen4 actually held for that ctrl_id in
data/gen_train/gen5g4/manifest.jsonl (opts[*].cmd; base controller only, not v_ variants), clustered on a 0.05 grid per
role; the representative of a cluster is its most frequent exact command, except that the cluster holding the stock
command is represented by the stock command itself (kind NOMINAL, as in training: cmd incl. torso_roll 0, so is_current
is 0 exactly as in every training row). The nominal is always present (added from candidates.nominal_cmd if absent).
Each candidate's cls = candidates.transition_class(spec, cmd).

    $PY xfeas/closed_loop/model_server.py --serve                 # sidecar (started by xfeas_method.Gen5Client)
    $PY xfeas/closed_loop/model_server.py --selftest [--n 6]      # parity checks offline (see selftest)

gen5b MODE: the model input is built EXACTLY as
trainer.Items for a gen5b moment:
    probe / transitions = data/probes_v5/<ctrl>/{office_v4,transitions}.json (the gen5b collector's; the runner's probe_path
                          is not used), spec = controller_registry.spec_path(ctrl)
    menu  C = candidates.enumerate_candidates(spec); X = Describer(spec, P, T).describe(C, cur_cmd, vx_measured)
          (the command IN FORCE and the measured forward speed, as branch_sim at a moment), then is_keep := [1 at 0] and
          is_nominal := [1 at 1] (trainer convention)
    state = [min(t since the last switch, 5) / 5, any switch yet]  (branch_dataset.decision_state; sent by the layer)
    candidates = the command classes the collector tried (branch_schedule.pick_classes on the nominal description, lateral off;
          slowP = branch_schedule.slow_in), in the same state the data tried them: command in force upright (stock / slow / fast)
          -> group-A set {stock, slow, fast, low1, low2}; command in force a posture P (low1 / low2 or its slowP) -> group-B
          set {P (= keep), stock (= stand up), the other posture, slowP(P)}. Stock is always table row 0 (kind NOMINAL).
"""

import os, sys  # noqa: E401  (standalone process, possibly in an isolated env: find the repo root itself)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import xfeas  # noqa: E402, F401  (puts every XFeas code folder on sys.path)
import argparse, hashlib, io, json, os, struct, sys
from collections import Counter, defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import joint_layout  # gen5 canonical joints BEFORE any gen4 import
import dataset5 as D5
import descriptors as DS
import candidates as CA

MANIFEST = os.path.join(ROOT, "data", "gen_train", "gen5g4", "manifest.jsonl")
CKPT = os.path.join(ROOT, "data", "gen_train", "gen5g4", "runs", "g1c", "model.pt")
STATE0 = np.array([1.0, 0.0], np.float32)  # == trainer_gen4data.STATE0
GRID = 0.05


def _key(cmd):
    return tuple(
        sorted((k, (round(round(v / GRID) * GRID, 2) if isinstance(v, (int, float)) else v)) for k, v in cmd.items())
    )


def deploy_cands(ctrl_id, spec, manifest=MANIFEST):
    """-> list of candidate dicts {kind, cmd, cls, n_tried, n_cluster} (index 0 = NOMINAL)"""
    clus = defaultdict(Counter)
    stock = None
    for l in open(manifest):
        r = json.loads(l)
        if r["ctrl_id"] != ctrl_id:
            continue
        for o in r["opts"]:
            c = o["cmd"]
            k = _key(c)
            clus[k][json.dumps(c, sort_keys=True)] += 1
            if o["kind"] == "stock":
                stock = c if stock is None else stock
                assert CA.same_cmd(stock, c), (stock, c)
    nom = CA.nominal_cmd(spec)
    if stock is None:
        stock = nom
    out = []
    ks = _key(stock)
    if ks not in clus:
        clus[ks][json.dumps(stock, sort_keys=True)] += 0
    for k, cnt in sorted(clus.items(), key=lambda kv: (kv[0] != ks, -sum(kv[1].values()))):
        if k == ks:
            cmd, kind = dict(stock), "NOMINAL"
        else:
            cmd, kind = json.loads(cnt.most_common(1)[0][0]), "grid"
        out.append(
            dict(
                kind=kind,
                cmd=cmd,
                cls=CA.transition_class(spec, cmd),
                n_cluster=int(sum(cnt.values())),
                n_exact=int(cnt[json.dumps(cmd, sort_keys=True)]),
            )
        )
    assert out[0]["kind"] == "NOMINAL"
    return out


def collate1(b, M5):
    """trainer_gen4data.collate for B = 1 (same layout / dtypes)"""
    import torch

    out = {}
    K = b["rgb"].shape[0]
    N = b["cand"].shape[0]
    rgb = torch.zeros(1, K, 3, 224, 224, dtype=torch.uint8)
    dep = torch.zeros(1, K, 1, 224, 224)
    fmask = torch.zeros(1, K, dtype=torch.bool)
    fage = torch.zeros(1, K)
    rgb[0, :K] = b["rgb"]
    dep[0, :K] = b["depth"]
    fmask[0, :K] = True
    fage[0, :K] = b["fage"]
    out.update(rgb=rgb, depth=dep, fmask=fmask, fage=fage)
    for k in M5.CTX_KEYS:
        if k not in out:
            out[k] = b[k][None]
    out["feas_raw"] = b["feas_raw"][None]
    out["state"] = b["state"][None]
    out["cand"] = b["cand"][None].float()
    return out


class Selector5:
    def __init__(self, ckpt=CKPT, device="cuda:0", manifest=MANIFEST):
        import torch
        import feasibility_model as M5

        self.torch, self.M5, self.dev = torch, M5, device
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert str(d.get("format", "")).startswith(M5.FORMAT), d.get("format")
        self.g5b = str(d["format"]).endswith("_g5b")
        self.cfg, self.norm = d["cfg"], d["norm"]
        self.kappa, self.lam = float(self.cfg.get("kappa", 0.5)), float(self.cfg.get("lam", 0.5))
        m = M5.build(self.cfg, pretrained=True).to(device).eval()
        miss = m.load_state_dict(d["model"], strict=False)
        bad = [k for k in miss.missing_keys if not k.startswith("ctx.vision.backbone")]
        assert not bad and not miss.unexpected_keys, (bad[:5], miss.unexpected_keys[:5])
        self.m = m
        self.manifest = manifest
        self._spec, self._probe, self._ptr, self._cands, self._desc = {}, {}, {}, {}, {}
        self.src = dict(
            ckpt=os.path.abspath(ckpt),
            holdout=self.cfg.get("holdout_base"),
            kappa=self.kappa,
            lam=self.lam,
            k_ens=len(m.members),
            manifest=os.path.abspath(manifest),
            mode="g5b" if self.g5b else "g4",
        )
        self._g5b = {}

    # -------------------------------------------------------------- gen5b mode (trainer.Items convention)
    def g5b_ctrl(self, ctrl_id):
        if ctrl_id not in self._g5b:
            import controller_registry as K5
            import branch_schedule as SB

            spec = json.load(open(K5.spec_path(ctrl_id)))
            pd = os.path.join(K5.PROBES_V5, ctrl_id)
            P = json.load(open(os.path.join(pd, "office_v4.json")))
            T = json.load(open(os.path.join(pd, "transitions.json")))
            assert P["ctrl_id"] == ctrl_id and T["ctrl_id"] == ctrl_id
            C = CA.enumerate_candidates(spec)
            D = DS.Describer(spec, P, T)
            nom = CA.nominal_cmd(spec)
            X0 = D.describe(C, nom, float(nom.get("lin_vel_x", 0.3) or 0.3))  # == branch_collector.pair
            cl = SB.pick_classes(C, X0, lateral=False)
            for p in ("low1", "low2"):
                if p in cl:
                    sp = SB.slow_in(C, X0, cl[p])
                    if sp is not None:
                        cl[f"slowP_{p}"] = sp
            self._g5b[ctrl_id] = dict(
                spec=spec,
                P=P,
                T=T,
                C=C,
                D=D,
                cl=cl,
                spec_path=K5.spec_path(ctrl_id),
                probe_path=os.path.join(pd, "office_v4.json"),
            )
            print(
                f"[model_server] g5b {ctrl_id}: classes "
                + ", ".join(f"{k}={i} {json.dumps(C[i]['cmd'])}" for k, i in cl.items()),
                file=sys.stderr,
                flush=True,
            )
        return self._g5b[ctrl_id]

    def g5b_set(self, g, cur):
        """-> [(class name, menu index)] for the command in force (stock first)"""
        C, cl = g["C"], g["cl"]
        ci = next((i for i in range(1, len(C)) if CA.same_cmd(C[i]["cmd"], cur)), None)
        post = {
            cl.get("low1"): "low1",
            cl.get("slowP_low1"): "low1",
            cl.get("low2"): "low2",
            cl.get("slowP_low2"): "low2",
        }
        post.pop(None, None)
        if ci is not None and ci in post:  # group B: a posture P is in force
            P = post[ci]
            oth = "low2" if P == "low1" else "low1"
            names = [("stock", cl["stock"]), (P, cl[P])]
            if oth in cl:
                names.append((oth, cl[oth]))
            if f"slowP_{P}" in cl:
                names.append((f"slowP_{P}", cl[f"slowP_{P}"]))
            if ci not in [i for _, i in names]:
                names.append(("keep", ci))
        else:  # group A: upright command in force
            names = [(k, cl[k]) for k in ("stock", "slow", "fast", "low1", "low2") if k in cl]
        return names, ci

    def choose_g5b(self, obs, ctrl_id, robot, cur, t_since, any_sw, vx):
        import torch

        g = self.g5b_ctrl(ctrl_id)
        C = g["C"]
        if not cur:  # first decision: the controller runs its nominal command
            cur = CA.nominal_cmd(g["spec"])
        X = np.asarray(g["D"].describe(C, cur, float(vx)), np.float32).copy()
        X[:, DS.KI["is_keep"]] = 0.0
        X[0, DS.KI["is_keep"]] = 1.0
        X[:, DS.KI["is_nominal"]] = 0.0
        X[1, DS.KI["is_nominal"]] = 1.0
        names, ci = self.g5b_set(g, cur)
        idx = [i for _, i in names]
        desc = D5.robot_desc5(robot)
        ctx, feas_raw = D5.DG.build_context(
            obs,
            g["spec"],
            g["P"],
            desc,
            self.norm,
            self.cfg,
            train=False,
            rng=np.random.default_rng(0),
            aug=dict(D5.DG.DEFAULT_AUG),
            ptrace_cache=self._ptr,
        )
        state = np.array(
            [min(max(float(t_since), 0.0), D5.T_SWITCH_CAP) / D5.T_SWITCH_CAP, float(bool(any_sw))], np.float32
        )
        t = torch.from_numpy
        b = {k: t(np.asarray(v)) for k, v in ctx.items()}
        fr = feas_raw.astype(np.float32) if feas_raw is not None else np.full(16, np.nan, np.float32)
        b.update(feas_raw=t(fr), state=t(state), cand=t(X[idx]))
        b = {k: v.to(self.dev) for k, v in collate1(b, self.M5).items()}
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):  # == trainer.run
                o = self.m(
                    cand=b["cand"], state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in self.M5.CTX_KEYS}
                )
            o = {k: v.float() for k, v in o.items()}
        ps = torch.sigmoid(o["value"])[:, 0]
        pf = torch.sigmoid(o["fall"])[:, 0]
        pc = torch.sigmoid(o["contact"])[:, 0]
        pst = torch.sigmoid(o["stall"])[:, 0]
        U = (
            (ps.mean(0) - self.kappa * ps.std(0) - self.lam * pf.mean(0)).cpu().numpy().astype(np.float64)
        )  # == trainer.evaluate
        G = lambda z: z.mean(0).cpu().numpy().astype(np.float64)
        p, psd, pfa, pco, pstl = G(ps), ps.std(0).cpu().numpy().astype(np.float64), G(pf), G(pc), G(pst)
        prog = G(o["progress"][:, 0])
        j = int(np.argmax(U))
        tab = [
            dict(
                i=int(idx[k]),
                kind="NOMINAL" if nm == "stock" else "grid",
                cls=nm,
                cmd=C[idx[k]]["cmd"],
                is_current=bool(idx[k] == ci),
                U=round(float(U[k]), 4),
                p=round(float(p[k]), 4),
                p_std=round(float(psd[k]), 4),
                p_fall=round(float(pfa[k]), 4),
                p_contact=round(float(pco[k]), 4),
                p_stall=round(float(pstl[k]), 4),
                progress=round(float(prog[k]), 3),
            )
            for k, (nm, _) in enumerate(names)
        ]
        return dict(
            pick=j,
            cmd=C[idx[j]]["cmd"],
            kind=tab[j]["kind"],
            cls=names[j][0],
            stock=bool(j == 0),
            U=float(U[j]),
            p=float(p[j]),
            p_stock=float(p[0]),
            U_stock=float(U[0]),
            n_cands=len(idx),
            table=tab,
            state=state.tolist(),
            cur_cmd=cur,
            cur_idx=ci,
            feas_raw=np.round(np.nan_to_num(fr, nan=-1.0), 4).tolist(),
            model="gen5b",
            src=self.src,
        )

    def _json(self, cache, p):
        if p not in cache:
            cache[p] = json.load(open(p))
        return cache[p]

    def cands(self, ctrl_id, spec_path, probe_path):
        key = (ctrl_id, spec_path, probe_path)
        if key not in self._cands:
            spec = self._json(self._spec, spec_path)
            probe = self._json(self._probe, probe_path)
            C = deploy_cands(ctrl_id, spec, self.manifest)
            D = DS.Describer(spec, probe, None)
            nom = CA.nominal_cmd(spec)
            X = D.describe(C, nom, float(spec.get("nominal", {}).get("speed", 0.3)))  # == gen4_option_dataset.build
            self._cands[key] = (C, np.asarray(X, np.float32))
        return self._cands[key]

    def item(self, obs, spec, probe, robot, X):
        """== trainer_gen4data.SitItems.__getitem__ with augment False (rng unused when train=False)"""
        import torch

        desc = D5.robot_desc5(robot)
        ctx, feas_raw = D5.DG.build_context(
            obs,
            spec,
            probe,
            desc,
            self.norm,
            self.cfg,
            train=False,
            rng=np.random.default_rng(0),
            aug=dict(D5.DG.DEFAULT_AUG),
            ptrace_cache=self._ptr,
        )
        t = torch.from_numpy
        out = {k: t(np.asarray(v)) for k, v in ctx.items()}
        fr = feas_raw.astype(np.float32) if feas_raw is not None else np.full(16, np.nan, np.float32)
        out.update(feas_raw=t(fr), state=t(STATE0.copy()), cand=t(np.asarray(X, np.float32)))
        return out, fr

    def score(self, obs, spec, probe, robot, X):
        torch = self.torch
        b, fr = self.item(obs, spec, probe, robot, X)
        b = {k: v.to(self.dev) for k, v in collate1(b, self.M5).items()}
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):  # == trainer_gen4data.run
                o = self.m(
                    cand=b["cand"], state=b["state"], feas_raw=b["feas_raw"], **{k: b[k] for k in self.M5.CTX_KEYS}
                )
            o = {k: v.float() for k, v in o.items()}
        ps = torch.sigmoid(o["value"])[:, 0]
        pf = torch.sigmoid(o["fall"])[:, 0]  # (K, N)
        pc = torch.sigmoid(o["contact"])[:, 0]
        pst = torch.sigmoid(o["stall"])[:, 0]
        U = ps.mean(0) - self.kappa * ps.std(0) - self.lam * pf.mean(0)  # == trainer_gen4data.evaluate
        g = lambda z: z.cpu().numpy().astype(np.float64)
        return dict(
            U=g(U),
            p=g(ps.mean(0)),
            pstd=g(ps.std(0)),
            pfall=g(pf.mean(0)),
            pcontact=g(pc.mean(0)),
            pstall=g(pst.mean(0)),
            prog=g(o["progress"][:, 0].mean(0)),
        ), fr

    def choose(self, obs, ctrl_id, spec_path, probe_path, robot):
        spec = self._json(self._spec, spec_path)
        probe = self._json(self._probe, probe_path)
        C, X = self.cands(ctrl_id, spec_path, probe_path)
        S, fr = self.score(obs, spec, probe, robot, X)
        i = int(np.argmax(S["U"]))
        tab = [
            dict(
                i=j,
                kind=C[j]["kind"],
                cls=C[j]["cls"],
                cmd=C[j]["cmd"],
                U=round(float(S["U"][j]), 4),
                p=round(float(S["p"][j]), 4),
                p_std=round(float(S["pstd"][j]), 4),
                p_fall=round(float(S["pfall"][j]), 4),
                p_contact=round(float(S["pcontact"][j]), 4),
                p_stall=round(float(S["pstall"][j]), 4),
                progress=round(float(S["prog"][j]), 3),
            )
            for j in range(len(C))
        ]
        return dict(
            pick=i,
            cmd=C[i]["cmd"],
            kind=C[i]["kind"],
            cls=C[i]["cls"],
            stock=bool(i == 0),
            U=float(S["U"][i]),
            p=float(S["p"][i]),
            p_stock=float(S["p"][0]),
            U_stock=float(S["U"][0]),
            n_cands=len(C),
            table=tab,
            feas_raw=np.round(np.nan_to_num(fr, nan=-1.0), 4).tolist(),
            model="gen5g4",
            src=self.src,
        )


# ---------------------------------------------------------------------- IPC (same framing as select_gen4)
def _read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def encode_request(obs, ctrl_id, spec_path, probe_path, robot):
    b = io.BytesIO()
    np.savez(b, ctrl_id=ctrl_id, spec_path=spec_path, probe_path=probe_path, robot=robot, **obs)
    return b.getvalue()


def decode_request(raw):
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    obs = {k: z[k] for k in z.files}
    meta = {k: str(obs.pop(k)) for k in ("ctrl_id", "spec_path", "probe_path", "robot")}
    for k in [k for k in obs if k.startswith("g5_")]:  # gen5b decision state from the layer (not perception)
        meta[k] = obs.pop(k)
    return obs, meta


def serve(sel):
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    print(f"[model_server] ready {json.dumps(sel.src)}", file=sys.stderr, flush=True)
    while True:
        try:
            n = struct.unpack("<I", _read(fin, 4))[0]
        except EOFError:
            return
        obs, meta = decode_request(_read(fin, n))
        try:
            if sel.g5b:
                assert "g5_cur_cmd" in meta, "gen5b checkpoint needs the layer's decision state (g5_* keys)"
                ch = sel.choose_g5b(
                    obs,
                    meta["ctrl_id"],
                    meta["robot"],
                    (json.loads(str(meta["g5_cur_cmd"])) if str(meta["g5_cur_cmd"]) else None),
                    float(meta["g5_t_since"]),
                    bool(meta["g5_any_switch"]),
                    float(meta["g5_vx"]),
                )
            else:
                ch = sel.choose(obs, meta["ctrl_id"], meta["spec_path"], meta["probe_path"], meta["robot"])
        except Exception as e:  # reported to the layer, which raises (no silent fallback)
            import traceback

            traceback.print_exc(file=sys.stderr)
            ch = dict(error=repr(e))
        out = io.BytesIO()
        np.savez(out, choice=json.dumps(ch, default=float))
        b = out.getvalue()
        fout.write(struct.pack("<I", len(b)))
        fout.write(b)
        fout.flush()


class Gen5Client:
    PY = xfeas.paths.XFEAS_PY

    def __init__(self, ckpt=CKPT, log=None):
        import subprocess

        e = dict(os.environ)
        e.setdefault("CUDA_VISIBLE_DEVICES", "0")
        e["OMP_NUM_THREADS"] = e.get("OMP_NUM_THREADS", "2")
        self.p = subprocess.Popen(
            [self.PY, os.path.abspath(__file__), "--ckpt", ckpt, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(open(log, "a") if log else None),
            env=e,
        )
        self.pid = self.p.pid

    def ask(self, obs, ctrl_id, spec_path, probe_path, robot):
        raw = encode_request(obs, ctrl_id, spec_path, probe_path, robot)
        self.p.stdin.write(struct.pack("<I", len(raw)))
        self.p.stdin.write(raw)
        self.p.stdin.flush()
        hdr = self.p.stdout.read(4)
        if len(hdr) < 4:
            raise RuntimeError("[model_server] sidecar died")
        n = struct.unpack("<I", hdr)[0]
        z = np.load(io.BytesIO(_read(self.p.stdout, n)), allow_pickle=False)
        ch = json.loads(str(z["choice"]))
        if "error" in ch:
            raise RuntimeError(f"[model_server] sidecar error: {ch['error']}")
        return ch

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(10)
        except Exception:
            self.p.kill()


# ---------------------------------------------------------------------- offline parity
def selftest(sel, n=6, ctrls=("g1c", "spotp")):
    """(1) deployment descriptors == the training rows' X for the same command (every manifest row of the controller);
    (2) the ONLINE request path (select_gen4.request_from_obs: what the runner sends, cam_mount_* keys) gives the same scores as
    the stored-obs path trainer_gen4data uses, on n rows per controller; prints the pick."""
    import select_gen4 as S4

    rows = [json.loads(l) for l in open(sel.manifest)]
    for cid in ctrls:
        R = [r for r in rows if r["ctrl_id"] == cid]
        sp, pp = R[0]["spec"], R[0]["probe"]
        C, X = sel.cands(cid, sp, pp)
        print(f"[selftest] {cid}: {len(C)} deployment candidates (from {len(R)} situations); nominal {C[0]['cmd']}")
        for c, x in zip(C, X):
            print(f"    {c['kind']:8s} {c['cls']:18s} n {c['n_cluster']:4d} {json.dumps(c['cmd'])}")
        nchk = nbad = 0
        for r in R:
            for o, xr in zip(r["opts"], r["X"]):
                for c, x in zip(C, X):
                    if CA.same_cmd(c["cmd"], o["cmd"]) and (c["kind"] == "NOMINAL") == (o["kind"] == "stock"):
                        nchk += 1
                        nbad += int(np.max(np.abs(np.asarray(xr, np.float32) - x)) > 1e-4)
        print(f"[selftest] {cid}: descriptor parity vs training X: {nchk} exact-command matches, {nbad} mismatches")
        assert nchk > 0 and nbad == 0
        rng = np.random.default_rng(0)
        for r in [R[i] for i in rng.choice(len(R), size=min(n, len(R)), replace=False)]:
            on = np.load(r["obs"], allow_pickle=False)
            o = {k: on[k] for k in on.files}
            spec, probe = sel._json(sel._spec, sp), sel._json(sel._probe, pp)
            S1, _ = sel.score(o, spec, probe, r["robot"], X)
            req = S4.request_from_obs(o)
            ch = sel.choose(req, cid, sp, pp, r["robot"])
            dU = float(np.max(np.abs(S1["U"] - np.array([t["U"] for t in ch["table"]]))))
            top = sorted(ch["table"], key=lambda t: -t["U"])[:3]
            print(
                f"[selftest] {cid} {r['key'][:48]:48s} fam {r['family']:10s} online-vs-stored max|dU| {dU:.4f}; pick {ch['cls']} "
                f"{json.dumps(ch['cmd'])} U {ch['U']:.3f} (stock U {ch['U_stock']:.3f}); top3 "
                + "; ".join(f"{t['cls']} vx {t['cmd'].get('lin_vel_x')} U {t['U']:.3f}" for t in top)
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    sel = Selector5(a.ckpt, a.device)
    if a.serve:
        serve(sel)
    elif a.selftest:
        selftest(sel, a.n)
