#!/usr/bin/env python
"""
Online selector for the XFeas-Gen4 release (format xfeas_gen4_v1). Phase 4 of the v4 rebuild (2026-10-21).

Port of closed_loop/select_gen3.py to gen4 -- the decision code (candidates_from_probe, the gen3 driver programs,
hard_gates / decide / rule_pick, cost_feats) is IMPORTED from select_gen3 unchanged; only the featurisation and the model
are gen4:
  * context  = dataset_gen4.build_context(obs, spec, probe, robot_desc(robot), norm, cfg, train=False) -- the SAME function
    the training / offline-eval dataset calls: fixed body description per ROBOT (g1_desc.json / spot_desc.json), the
    canonical 29 + 19 joint list (gen4_common), the WHOLE probe trace (T_PROBE 3072), the clearance map back-projected with
    the stored camera matrix of the shared mount (obs keys cam_mount_M / cam_mount_base / cam_mount_camh -> mount_of);
  * candidates = gen4_common.cand_feats (byte-identical to gen3_common.cand_feats) with the driver's feas_act / drv;
  * model = model_gen4 ensemble (data/xe_train/gen4/xfeas_gen4_ens.pt), its norm (ckpt['norm'] == norm.json),
    learned_selection.json + gate.json next to the checkpoint;
  * `lateral` for decide / rule_pick = feas[14] (lateral_capable) > 0.5, as eval_offline_gen4.sit_arrays (the gates were
    fitted with that); stand_top = the probe's fp stand_top (eval_offline_gen4.stand_tops).
No gate is added or changed. XFEAS_GEN3_TOP_CLEAR must be UNSET (the fitted geometry gate is fa[0] < -geom_tol_top);
serve() refuses to start otherwise.

    sel = Selector4(ckpt); out = sel.choose(obs, spec, probe, robot="g1")
    $PY closed_loop/select_gen4.py --ckpt data/xe_train/gen4/xfeas_gen4_ens.pt --serve
    request npz = the obs keys + (spec_json | spec_path) + (probe_json | probe_path) + robot + lat_room + gap_e + rule (0/1)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, hashlib, io, json, math, os, struct, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "closed_loop"), THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gen4_common as G4
import select_gen3 as SG

RELEASE = os.path.join(ROOT, "data", "xe_train", "gen4", "xfeas_gen4_ens.pt")


def collection_mask(macro, A):
    """the env_act mask exactly as the gen_v4 collector stored it: scripts/gen3/driver.act_vec of the action dict
    (act_vec always sets m[0:4]; select_gen3.candidates_from_probe uses ENV_MASK, which omits pitch_b (index 3) for
    turn_side_turn -- found by the Phase 4 parity check; verified equal to the stored mask on 3550 rollouts)"""
    p = os.path.join(ROOT, "Scripts", "data_collector")
    if p not in sys.path:
        sys.path.insert(0, p)
    import driver as DR
    return DR.act_vec(SG.act_dict(macro, A))[1].astype(np.float32)


def candidates4(spec, probe, feas, lat_room=(1.0, 1.0), gap_e=0.0):
    """select_gen3.candidates_from_probe (same vocabulary) with the collection's env_act mask"""
    return [(m, A, collection_mask(m, A), info) for (m, A, _M, info) in SG.candidates_from_probe(spec, probe, feas, lat_room, gap_e)]


def lateral_of(feas):
    """eval_offline_gen4.sit_arrays: lateral = feas lateral_capable when the feas vector is finite there"""
    if feas is not None and np.isfinite(feas[14]):
        return bool(feas[14] > 0.5)
    return True


class Selector4:
    def __init__(self, ckpt=RELEASE, device="cuda:0", selection="", gate=""):
        import torch
        import model_gen4 as M4
        import dataset_gen4 as D4
        self.torch, self.dev, self.M4, self.D4 = torch, device, M4, D4
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert d.get("format") == M4.FORMAT, f"{ckpt}: not an {M4.FORMAT} checkpoint"
        self.cfg, self.norm = d["cfg"], d["norm"]
        cd = os.path.dirname(os.path.abspath(ckpt))
        nj = os.path.join(cd, "norm.json")
        if os.path.exists(nj):
            assert json.load(open(nj)) == self.norm, "norm.json != checkpoint norm"
        self.models = []
        for sd in d["members"]:
            m = M4.build(self.cfg, pretrained=False).to(device).eval(); m.load_state_dict(sd); self.models.append(m)
        sp = selection or os.path.join(cd, "learned_selection.json")
        S = json.load(open(sp)) if os.path.exists(sp) else {}
        self.sel = S.get("chosen") or SG.DEFAULT_SEL
        gp = gate or os.path.join(cd, "gate.json")
        self.gate = dict(SG.DEFAULT_GATE, **(json.load(open(gp)) if os.path.exists(gp) else {}))
        self.src = dict(ckpt=os.path.abspath(ckpt), selection=sp if S else "default", gate=gp if os.path.exists(gp) else "default",
                        n_members=len(self.models), fmt=d.get("format"))
        self._ptr = {}                                   # probe trace features per probe (hash)
        self._spec_cache, self._probe_cache = {}, {}

    # ------------------------------------------------------------------ featurisation (== dataset_gen4)
    def context(self, obs, spec, probe, robot="g1"):
        D4 = self.D4
        key = probe.get("_hash") if isinstance(probe, dict) else None
        cache = {}
        if key is not None and key in self._ptr:
            cache[id(probe)] = self._ptr[key]
        ctx, feas_raw = D4.build_context(obs, spec, probe, D4.robot_desc(robot), self.norm, self.cfg, train=False, ptrace_cache=cache)
        if key is not None and id(probe) in cache:
            self._ptr[key] = cache[id(probe)]
        ctx["fmask"] = np.ones(len(ctx["fage"]), bool)                 # collate4 adds it in training
        t = lambda x: self.torch.from_numpy(np.ascontiguousarray(np.asarray(x)))[None].to(self.dev)
        return {k: t(v) for k, v in ctx.items()}, feas_raw, ctx

    def cand_matrix(self, C, FA, DV, MID, feas):
        return np.stack([G4.cand_feats(Ak, Mk, MID[k], feas=feas, feas_act=FA[k], drv=DV[k], use_drv=self.cfg.get("use_drv", True))
                         for k, (macro, Ak, Mk, info) in enumerate(C)]).astype(np.float32)

    def scores_from(self, x, CF, MID):
        torch = self.torch
        cand = torch.from_numpy(CF)[None].to(self.dev); mid = torch.from_numpy(np.asarray(MID, np.int64))[None].to(self.dev)
        outs = []
        with torch.no_grad():
            for m in self.models:
                o = m(cand=cand, macro_id=mid, **x); outs.append({k: v[0].float().cpu().numpy() for k, v in o.items()})
        sg = lambda z: 1 / (1 + np.exp(-z))
        S = {}
        for k in self.M4.CLASS_HEADS:
            p = np.stack([sg(o[k]) for o in outs])
            S[f"p_{k}"] = p.mean(0); S[f"p_{k}_std"] = p.std(0)
        S["p_std"] = S["p_success_std"]
        for k in self.M4.REG_HEADS:
            S[f"r_{k}"] = np.mean([o[k] for o in outs], 0)
        S["ach"] = np.mean([o["ach"] for o in outs], 0)
        return S

    # ------------------------------------------------------------------ driver programs of the candidate set
    def programs(self, spec, probe, feas, C):
        drv, drv_src = SG.get_driver(spec, probe)
        fp, fpm = G4.fp_vector(probe)
        pops = {o.get("op_id"): o for o in (probe or {}).get("ops") or []}
        progs, FA, MID, A, OPV, DV = [], [], [], [], [], []
        for k, (macro, Ak, Mk, info) in enumerate(C):
            pr = drv.plan(macro, Ak, stock=(k == 0)) if isinstance(drv, SG.OfficialDriver) else drv.plan(macro, Ak)
            dv = pr.get("drv", {}) if isinstance(pr, dict) else {}
            if isinstance(drv, SG.OfficialDriver) and feas is not None:
                fa = drv.feas_act(pr, feas)
            else:
                pred = dv.get("pred") or dict(top=float(fp[0]) if fpm[0] else 1.3, width=float(fp[1]) if fpm[1] else 0.5,
                                              depth=float(fp[2]) if fpm[2] else 0.3, vx=float(Ak[0]))
                fa = SG.feas_act_from(pred, feas, macro, dv.get("driver_feasible", True))
            progs.append(pr); FA.append(fa); MID.append(G4.MACRO_ID[macro]); A.append(Ak); DV.append(dv)
            OPV.append(SG.op_vx_of_program(pr, pops) if k else np.inf)
        return progs, FA, MID, A, OPV, DV, drv_src

    def choose(self, obs, spec, probe, robot="g1", lat_room=(1.0, 1.0), gap_e=0.0, rule=False):
        spec = G4.load_spec(spec)
        x, feas, _ = self.context(obs, spec, probe, robot)
        fp, fpm = G4.fp_vector(probe)
        C = candidates4(spec, probe, feas, lat_room, gap_e)
        stand_top = float(fp[0]) if fpm[0] else None
        lateral = lateral_of(feas)
        progs, FA, MID, A, OPV, DV, drv_src = self.programs(spec, probe, feas, C)
        S = self.scores_from(x, self.cand_matrix(C, FA, DV, MID, feas), MID)
        FAa, Aa = np.stack(FA), np.stack(A)
        if not rule and os.environ.get("XFEAS_GEN4_PURE", "0") == "1":
            # 2026-10-21 "learned model only": the candidate with the highest predicted P(success); no hard gates, no OOD
            # gate, no margin over stock (the learned scores alone decide)
            i = int(np.argmax(np.asarray(S["p_success"], float))); allowed, why, U, P = None, None, None, None
        elif rule:
            i = SG.rule_pick(np.asarray(MID), Aa, FAa, feas, 0, lateral); allowed, why, U, P = None, None, None, None
        else:
            i, allowed, why, U, P = SG.decide(S["p_success"], S["p_std"], SG.cost_feats(S), np.asarray(MID), Aa, FAa, feas, 0, self.sel, self.gate,
                                              lateral, stand_top, op_vx=np.array(OPV))
        tab = [dict(name=C[j][3].get("name"), macro=C[j][0], p=round(float(S["p_success"][j]), 3), std=round(float(S["p_std"][j]), 3),
                    p_clean=round(float(S["p_clean"][j]), 3), p_fall=round(float(S["p_fall"][j]), 3), p_stall=round(float(S["p_stall"][j]), 3),
                    p_collision=round(float(S["p_collision"][j]), 3),
                    lcb=None if P is None else round(float(P[j]), 3), U=None if U is None else round(float(U[j]), 3),
                    allowed=None if allowed is None else bool(allowed[j]), why=None if why is None else why[j],
                    ach=np.round(S["ach"][j], 3).tolist(), feas_act=np.round(FAa[j], 3).tolist(),
                    env_act=np.round(C[j][1], 4).tolist()) for j in range(len(C))]
        return json.loads(json.dumps(dict(macro=C[i][0], env_act=np.round(C[i][1], 4).tolist(), name=C[i][3].get("name"), stock=bool(i == 0),
                                          p_success=float(S["p_success"][i]), p_stock=float(S["p_success"][0]), program=progs[i], driver=drv_src,
                                          rule=bool(rule), sel=self.sel, gate_src=self.src, table=tab, pick=int(i), robot=robot,
                                          feas=np.round(np.nan_to_num(np.asarray(feas, float) if feas is not None else np.zeros(16), nan=-1), 4).tolist(),
                                          stand_top=stand_top, lateral=lateral, model="gen4"), default=float))

    # ------------------------------------------------------------------ request helpers
    def load_spec_probe(self, obs):
        if "spec_path" in obs:
            p = str(obs.pop("spec_path"))
            if p not in self._spec_cache:
                self._spec_cache[p] = json.load(open(p))
            spec = self._spec_cache[p]
        else:
            spec = json.loads(str(obs.pop("spec_json")))
        if "probe_path" in obs:
            p = str(obs.pop("probe_path"))
            if p not in self._probe_cache:
                pr = json.load(open(p)); pr["_hash"] = hashlib.sha1(open(p, "rb").read()).hexdigest()
                self._probe_cache[p] = pr
            probe = self._probe_cache[p]
        else:
            s = str(obs.pop("probe_json"))
            probe = json.loads(s); probe["_hash"] = hashlib.sha1(s.encode()).hexdigest()
        return spec, probe


def request_from_obs(o):
    """a stored gen_v4 obs npz -> the request the ONLINE runner sends (gen4_methods._decide): decision frame + proprio + h(t)
    + the 4-frame memory + the decision camera matrix of the shared mount as cam_mount_* (NOT the obs-only keys cam_dec /
    base_hist_world / cam_h_hist). Used by the parity check."""
    t = np.asarray(o["t_hist"], np.float32)
    i_dec = int(np.argsort(-t)[0])
    req = dict(rgb=np.asarray(o["rgb"], np.uint8), depth=np.asarray(o["depth"], np.float32), prop_base=np.asarray(o["prop_base"], np.float32),
               prop_joints=np.asarray(o["prop_joints"], np.float32), joint_names=str(o["joint_names"]), fmt="gen_v4",
               ach_keys=str(o["ach_keys"]), feas=np.asarray(o["feas"], np.float32),
               rgb_hist=np.asarray(o["rgb_hist"], np.uint8), depth_hist=np.asarray(o["depth_hist"]), t_hist=t,
               cam_hist=np.asarray(o["cam_hist"], np.float64), K128=np.asarray(o["K128"]),
               cam_mount_M=np.asarray(o["cam_dec"], np.float64), cam_mount_base=np.asarray(o["base_hist_world"][i_dec], np.float64),
               cam_mount_camh=np.float64(o["cam_h_hist"][i_dec]))
    for k in ("hist_t", "hist_valid", "hist_cmd", "hist_env", "hist_ach", "hist_q", "hist_dq"):
        req[k] = np.asarray(o[k])
    return req


def _read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def decode_request(raw, sel):
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    obs = {k: z[k] for k in z.files}
    spec, probe = sel.load_spec_probe(obs)
    robot = str(obs.pop("robot", "g1"))
    lat_room = tuple(map(float, obs.pop("lat_room", (1.0, 1.0))))
    gap_e = float(obs.pop("gap_e", 0.0)); rule = bool(int(obs.pop("rule", 0)))
    return obs, spec, probe, robot, lat_room, gap_e, rule


def encode_request(obs, spec_path=None, probe_path=None, spec=None, probe=None, robot="g1", lat_room=(1.0, 1.0), gap_e=0.0, rule=False):
    b = io.BytesIO()
    extra = {}
    if spec_path:
        extra["spec_path"] = spec_path
    else:
        extra["spec_json"] = json.dumps(spec)
    if probe_path:
        extra["probe_path"] = probe_path
    else:
        extra["probe_json"] = json.dumps(probe)
    np.savez(b, robot=robot, lat_room=np.asarray(lat_room, np.float32), gap_e=np.float32(gap_e), rule=np.int32(rule), **extra, **obs)
    return b.getvalue()


def serve(sel):
    assert not os.environ.get("XFEAS_GEN3_TOP_CLEAR"), "XFEAS_GEN3_TOP_CLEAR is set: not the fitted gen4 geometry gate"
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    print(f"[select_gen4] ready {json.dumps(sel.src)} sel {sel.sel}", file=sys.stderr, flush=True)
    while True:
        try:
            n = struct.unpack("<I", _read(fin, 4))[0]
        except EOFError:
            return
        obs, spec, probe, robot, lat_room, gap_e, rule = decode_request(_read(fin, n), sel)
        ch = sel.choose(obs, spec, probe, robot, lat_room, gap_e, rule=rule)
        out = io.BytesIO(); np.savez(out, choice=json.dumps(ch)); b = out.getvalue()
        fout.write(struct.pack("<I", len(b))); fout.write(b); fout.flush()


class Gen4Client:
    """length-prefixed npz client of `select_gen4.py --serve` (gsam2 env sidecar on the job's own GPU)"""
    PY = xfeas_paths.XFEAS_PY

    def __init__(self, ckpt=RELEASE):
        import subprocess
        e = dict(os.environ)
        e.pop("XFEAS_GEN3_TOP_CLEAR", None)
        e.setdefault("CUDA_VISIBLE_DEVICES", "0")
        e["OMP_NUM_THREADS"] = e.get("OMP_NUM_THREADS", "2")
        self.p = subprocess.Popen([self.PY, os.path.abspath(__file__), "--ckpt", ckpt, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=e)

    def ask(self, obs, spec_path, probe_path, robot="g1", lat_room=(1.0, 1.0), gap_e=0.0, rule=False):
        raw = encode_request(obs, spec_path=spec_path, probe_path=probe_path, robot=robot, lat_room=lat_room, gap_e=gap_e, rule=rule)
        self.p.stdin.write(struct.pack("<I", len(raw))); self.p.stdin.write(raw); self.p.stdin.flush()
        hdr = self.p.stdout.read(4)
        if len(hdr) < 4:
            raise RuntimeError("[select_gen4] sidecar died")
        n = struct.unpack("<I", hdr)[0]
        z = np.load(io.BytesIO(_read(self.p.stdout, n)), allow_pickle=False)
        return json.loads(str(z["choice"]))

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(5)
        except Exception:
            self.p.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=RELEASE)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    sel = Selector4(a.ckpt, a.device)
    if a.serve:
        serve(sel)
