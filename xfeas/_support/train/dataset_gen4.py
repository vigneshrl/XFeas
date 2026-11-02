#!/usr/bin/env python
"""
Situation-level dataset for XFeas-Gen4 (Phase 3 of the v4 rebuild, 2026-10-20). Adapts xfeas/_support/train/dataset_gen3.py
to the gen_v4 record format (analysis/GEN4_SPEC.md). One ITEM = one situation x controller: the decision context (encoded
once) + all of its simulated, non-duplicate rollouts as candidates. Model inputs are the gen3 ones (GEN3_MODEL.md):
envelope action candidates, spec tokens, probe fingerprint + ops + trace, h(t), perception (frame memory + clearance map +
proprio), feas (16). Changes vs gen3 (all forced by the v4 data, none tuned):
  * records come from sits_<job>.jsonl (+ ep/ + obs/); only FINISHED queue jobs (state.json 'done', or 'audit_fail' with a
    later PASS re-audit audit/audit_job_<id>.json) are read -- running / pending jobs never enter;
  * probes only from data/probes_v4 (GEN4_SPEC 6); SONIC (g1s) anywhere -> build() refuses;
  * clearance map back-projects with the stored per-frame camera matrices of the shared mount (cam_dec /
    base_hist_world / cam_h_hist -> cmap.set_mount), exactly what the online sidecar does (select_gen3 cam_mount_*);
  * embodiment: a fixed body per ROBOT (G1: scripts/gen_train/g1_desc.json as gen3; Spot: scripts/gen4/spot_desc.json),
    canonical joints = G1 29 + Spot 19 (gen4_common); the controller is never identified by its joint list;
  * labels add 'clean' (pass_label == clean: success without body contact with the hazard, GEN4_SPEC 4);
  * feas = the obs 'feas' (v0.4 rules + memory + minpool, GEN4_SPEC 3) -- the v4 model input; feas_v03 is not used.
Splits: zone_split (train / val lanes) inside each base; leave-one-base-out folds over {g1c, g1g, g1l, g1u, spotp}, every
virtual variant v_<base>_kk goes with its base (fold_rows).

    PY=$XFEAS_PY
    $PY xfeas/_support/train/dataset_gen4.py build --root data/gen_v4 --out data/gen_train/gen4/manifest.jsonl [--exclude qa_excl.json]
    $PY xfeas/_support/train/dataset_gen4.py check --manifest data/gen_train/gen4/manifest.jsonl --n 8
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, glob, json, math, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gen4_common as G
import cmap as CM

ROOT = xfeas.paths.ROOT
DATA = os.path.join(ROOT, "data", "gen_v4")
PROBES = os.path.join(ROOT, "data", "probes_v4")
HELD_OUT_CTRL = ("g1s",)  # GEAR-SONIC: never in data / fitting (GEN4_SPEC 1)
BASES = ("g1c", "g1g", "g1l", "g1u", "spotp")
HELD_ENV = ("gq", "hospital", "interior")
N_MAX = 12
K_HIST = 4
T_PROP = 100
MAP_KIND = "g1"  # irrelevant with the mount set (points() uses the exact matrix, cam_h prior)
DESC_FILES = dict(g1=os.path.join(THIS, "g1_desc.json"), spot=os.path.join(THIS, "spot_desc.json"))
_DESC = {}


def robot_desc(robot):
    """fixed embodiment descriptor per robot (never per controller)"""
    robot = "spot" if str(robot).startswith("spot") else "g1"
    if robot not in _DESC:
        _DESC[robot] = json.load(open(DESC_FILES[robot]))
    return _DESC[robot]


def robot_of(ctrl_or_base):
    return "spot" if "spotp" in str(ctrl_or_base) else "g1"


def base_of(ctrl_id):
    c = str(ctrl_id)
    if c.startswith("v_"):
        return c[2:].rsplit("_", 1)[0]
    return c


def joint_names_of(o, desc, J):
    if "joint_names" in o:
        jn = json.loads(str(o["joint_names"]))
        if len(jn) == J:
            return jn
    if len(desc["joint_names"]) == J:
        return list(desc["joint_names"])
    raise ValueError(f"no joint names for {J} proprio joints")


# ------------------------------------------------------------------------------------------------ finished jobs
def finished_parts(root):
    """job ids whose data is complete: state 'done', or 'audit_fail' with a PASS re-audit json (the two terrain_g1c jobs
    that failed only the over-strict lane check, re-audited PASS after the fix)"""
    st = json.load(open(os.path.join(root, "queue", "state.json")))
    ok, why = [], {}
    for jid, v in st.items():
        s = v.get("status")
        if s == "done":
            ok.append(jid)
        elif s == "audit_fail":
            p = os.path.join(root, "audit", f"audit_job_{jid}.json")
            if os.path.exists(p) and json.load(open(p)).get("verdict") == "PASS":
                ok.append(jid)
                why[jid] = "audit_fail in state.json, re-audit PASS"
            else:
                why[jid] = "audit_fail (no PASS re-audit): excluded"
        else:
            why[jid] = f"{s}: excluded"
    return sorted(ok), why


def load_records(root, parts):
    R = []
    for f in glob.glob(os.path.join(root, "*", "*", "sits_*.jsonl")):
        part = os.path.basename(f)[5:-6]
        if part in parts:
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    r["_part"] = part
                    R.append(r)
    return R


def _exclusions(path):
    if not path:
        return set()
    J = json.load(open(path))
    if isinstance(J, list):
        return set(map(str, J))
    out = set()
    for k in ("exclude", "sids", "situations"):
        v = J.get(k)
        if isinstance(v, list):
            out |= {str(x if not isinstance(x, dict) else x.get("sid", "")) for x in v}
        elif isinstance(v, dict):
            out |= set(map(str, v))
    return out


def build_gen4(root, exclude=""):
    root = os.path.abspath(root)
    parts, part_why = finished_parts(root)
    R = load_records(root, set(parts))
    excl = _exclusions(exclude)
    out, drop = [], defaultdict(int)
    for r in sorted(R, key=lambda r: r["sid"]):
        sid = r["sid"]
        if r["ctrl_id"] in HELD_OUT_CTRL or r["base"] in HELD_OUT_CTRL or "g1s" in sid:
            raise SystemExit(f"[gen4 manifest] {sid}: SONIC data in {root} -- it must never be used for training")
        if any(h in r["env"].lower() for h in HELD_ENV):
            raise SystemExit(f"[gen4 manifest] {sid}: held-out environment {r['env']}")
        if r.get("run_id") != "gen_v4":
            drop["run_id"] += 1
            continue
        if sid in excl or r.get("geo_sit") in excl:
            drop["qa_excluded"] += 1
            continue
        pf = os.path.join(ROOT, r["probe_file"]) if not os.path.isabs(r["probe_file"]) else r["probe_file"]
        if (
            r.get("probe_src") != "probes_v4"
            or not os.path.abspath(pf).startswith(PROBES + os.sep)
            or not os.path.exists(pf)
        ):
            drop["probe_not_v4"] += 1
            continue
        bdir = os.path.join(root, r["env"], r["base"], "ep")
        keep = []
        # 'confirm' re-runs of a program are kept (gen3 recipe): 84 % of them have a different trace (PhysX is not
        # bit-deterministic here), i.e. they are real repeated trials, not copies (QA 2026-10-20)
        for x in sorted(r["rollouts"], key=lambda x: x["k"]):
            if not x.get("sim") or x.get("dup_of") not in (None, -1):
                continue
            p = os.path.join(bdir, x["ep"])
            if not os.path.exists(p):
                drop["ep_missing"] += 1
                continue
            keep.append((x, p))
        st = [kp for kp in keep if kp[0]["k"] == 0]
        if not st:
            drop["no_stock"] += 1
            continue
        keep = st[:1] + [kp for kp in keep if kp[0]["k"] != 0]
        if len(keep) < 2:
            drop["lt2_rollouts"] += 1
            continue
        obs = os.path.join(root, r["obs_file"])
        if not os.path.exists(obs):
            drop["no_obs"] += 1
            continue
        spec = os.path.join(root, "specs", r["ctrl_id"] + ".json")
        if not os.path.exists(spec):
            drop["no_spec"] += 1
            continue
        b = r["base"]
        assert b == base_of(r["ctrl_id"]), (sid, b, r["ctrl_id"])
        ys = [bool(x["success"]) for x, _ in keep]
        yc = [x.get("pass_label") == "clean" for x, _ in keep]
        out.append(
            dict(
                key=sid,
                fmt="gen_v4",
                obs=obs,
                eps=[p for _, p in keep],
                ctrl_id=r["ctrl_id"],
                base=b,
                robot=r["robot"],
                virtual=bool(r["virtual"]),
                env=r["env"],
                split=r.get("zone_split") or "train",
                bucket=r.get("bucket", ""),
                family=r["family"],
                family_oracle=r.get("family_oracle", r["family"]),
                subfamily=r.get("subfamily", ""),
                light=r["light"]["mode"],
                spec=spec,
                probe=os.path.abspath(pf),
                sit_id=r.get("geo_sit") or sid.split("__")[0],
                part=r["_part"],
                n_ok=int(sum(ys)),
                n_clean=int(sum(yc)),
                stock_ok=ys[0],
                stock_clean=yc[0],
                audited=True,
                n_infeasible=int(sum(not x.get("sim") for x in r["rollouts"])),
                n_dup=int(sum(x.get("dup_of") not in (None, -1) for x in r["rollouts"])),
            )
        )
    return out, dict(drop), dict(parts=len(parts), part_why=part_why)


def set_weights(rows):
    n = defaultdict(int)
    for r in rows:
        n[(r["ctrl_id"], r["env"])] += 1
    for r in rows:
        r["weight"] = 1.0 / math.sqrt(n[(r["ctrl_id"], r["env"])])
    return rows


def read_manifest(p):
    return [json.loads(l) for l in open(p)]


def fold_rows(rows, holdout_base=""):
    """leave-one-base-out: the held-out base and ALL of its virtual variants leave train AND val (split 'test')"""
    out = []
    for r in rows:
        r = dict(r)
        if holdout_base and r["base"] == holdout_base:
            r["split"] = "test"
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ norm
def compute_norm(rows, n=400, seed=0):
    rng = np.random.default_rng(seed)
    tr = [r for r in rows if r["split"] == "train"]
    pick = [tr[i] for i in rng.choice(len(tr), min(n, len(tr)), replace=False)] if tr else []
    B, Jc = [], []
    for r in pick:
        o = np.load(r["obs"], allow_pickle=False)
        desc = robot_desc(r.get("robot", robot_of(r["base"])))
        pb = np.asarray(o["prop_base"], np.float32)
        pj = np.asarray(o["prop_joints"], np.float32)
        jc, jm = G.canon_joints(joint_names_of(o, desc, pj.shape[1]), pj, axis=1)
        B.append(pb)
        Jc.append(jc[:, jm].reshape(-1, pj.shape[-1]))
    st = lambda x: (x.mean(0).tolist(), np.maximum(x.std(0), 1e-3).tolist())
    nb, nj = (
        (st(np.concatenate(B)), st(np.concatenate(Jc))) if B else (([0.0] * 13, [1.0] * 13), ([0.0] * 7, [1.0] * 7))
    )
    fps = {}
    for r in tr:
        if r["probe"] and r["ctrl_id"] not in fps:
            fps[r["ctrl_id"]] = G.fp_vector(json.load(open(r["probe"])))
    if fps:
        F = np.stack([v[0] for v in fps.values()])
        Pm = np.stack([v[1] for v in fps.values()])
        mu = np.array([F[Pm[:, i], i].mean() if Pm[:, i].any() else 0.0 for i in range(G.N_FP)])
        sd = np.array([max(F[Pm[:, i], i].std(), 0.02) if Pm[:, i].sum() > 1 else 1.0 for i in range(G.N_FP)])
    else:
        mu, sd = np.zeros(G.N_FP), np.ones(G.N_FP)
    return dict(
        base_mean=nb[0],
        base_std=nb[1],
        joint_mean=nj[0],
        joint_std=nj[1],
        fp_mean=mu.tolist(),
        fp_std=sd.tolist(),
        n_fp_ctrls=len(fps),
        fp_ctrls=sorted(fps),
    )


# ------------------------------------------------------------------------------------------------ featurisation
def color_aug(rng, rgb):
    x = rgb.astype(np.float32) / 255.0
    if rng.random() < 0.5:
        x = x[..., rng.permutation(3)]
    if rng.random() < 0.25:
        x = np.repeat(x.mean(-1, keepdims=True), 3, -1)
    x = x * rng.uniform(0.85, 1.15, 3)
    m = x.mean()
    x = (x - m) * rng.uniform(0.75, 1.25) + m * rng.uniform(0.7, 1.3)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _resize(x, n, nearest):
    import cv2

    return cv2.resize(np.asarray(x), (n, n), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


DEFAULT_AUG = dict(
    fp_noise=0.05, p_fp_drop=0.15, p_trace_drop=0.3, p_hist_drop=0.25, p_hist_trunc=0.5, p_frames_drop=0.3, color=True
)


def rollout_labels(z, f):
    nan = float("nan")
    g = lambda k: float(z[k]) if k in f and np.ndim(z[k]) == 0 else nan
    ok = bool(z["success"])
    y = dict(
        success=float(ok),
        fall=float(bool(z["fell"])),
        stall=float(bool(z["stall"])) if "stall" in f else nan,
        collision=float(bool(z["collision"])) if "collision" in f else nan,
        slip=nan,
        clean=float(str(z["pass_label"]) == "clean") if "pass_label" in f else nan,
        progress=g("progress"),
        tilt_margin=nan,
        t_norm=nan,
        env_dev=nan,
        cost=nan,
    )
    if "slipped" in f and "slip_mean" in f and np.isfinite(float(z["slip_mean"])):
        y["slip"] = float(bool(z["slipped"]))
    tm = g("tilt_margin")
    y["tilt_margin"] = float(np.clip(tm, -1, 1)) if np.isfinite(tm) else nan
    tg, tr = g("t_goal"), g("t_ref")
    if ok and np.isfinite(tg) and np.isfinite(tr) and tr > 0:
        y["t_norm"] = float(np.clip(tg / tr, 0, 5))
    ed = g("env_dev")
    c = g("cost")
    y["env_dev"] = ed if np.isfinite(ed) else nan
    y["cost"] = float(np.clip(c, 0, 5)) if (ok and np.isfinite(c)) else nan
    return y


def rollout_cand(z, f, feas, use_drv=True):
    A = np.asarray(z["env_act"], np.float32)
    M = np.asarray(z["env_act_mask"], np.float32)
    mid = int(z["macro_id"]) if "macro_id" in f else G.MACRO_ID.get(str(z["macro"]), 0)
    drv = None
    if "drv" in f:
        try:
            drv = json.loads(str(z["drv"]))
        except Exception:
            drv = None
    fa = np.asarray(z["feas_act"], np.float32) if "feas_act" in f else None
    return G.cand_feats(A, M, mid, feas=feas, feas_act=fa, drv=drv, use_drv=use_drv), mid, A, M


def frames_of(o, use_hist):
    """obs -> (frames newest first [(rgb, depth, age_s)], world_frames for cmap or None, decision-frame index)"""
    fr = [(np.asarray(o["rgb"])[..., :3], np.nan_to_num(np.asarray(o["depth"], np.float32), posinf=60.0), 0.0)]
    wf, i_dec = None, None
    if "t_hist" in o:
        t = np.asarray(o["t_hist"], np.float32)
        order = np.argsort(-t)
        i_dec = int(order[0])
        if use_hist and "rgb_hist" in o:
            f128, c128 = float(o["K128"][0, 0]), float(o["K128"][0, 2])
            M_dec = o["cam_hist"][order[0]]
            wfl = []
            for i in order[1:K_HIST]:
                dp = np.nan_to_num(np.asarray(o["depth_hist"][i], np.float32), posinf=20.0)
                fr.append((_resize(o["rgb_hist"][i][..., :3], 224, False), _resize(dp, 224, True), float(-t[i])))
                wfl.append((dp, o["cam_hist"][i], f128, c128))
            wf = (M_dec, wfl)
    return fr, wf, i_dec


def mount_of(o, i_dec):
    """(M, base, cam_h) of the decision frame for cmap.set_mount: obs keys of gen_v4 (cam_dec, base_hist_world,
    cam_h_hist) or the online sidecar keys (cam_mount_M, cam_mount_base, cam_mount_camh)"""
    if "cam_mount_M" in o:
        return o["cam_mount_M"], o["cam_mount_base"], (float(o["cam_mount_camh"]) if "cam_mount_camh" in o else None)
    if "cam_dec" in o and "base_hist_world" in o and i_dec is not None:
        ch = float(o["cam_h_hist"][i_dec]) if "cam_h_hist" in o else None
        return o["cam_dec"], o["base_hist_world"][i_dec], ch
    return None


def build_context(o, spec, probe, desc, norm, cfg, train=False, rng=None, aug=None, ptrace_cache=None):
    """decision context of one situation x controller -> (dict of numpy arrays = model_gen4.CTX_KEYS, feas_raw or None)"""
    aug = dict(DEFAULT_AUG, **(aug or {}))
    rng = rng if rng is not None else np.random.default_rng(0)
    if not cfg.get("use_fp", True):
        probe = None
    use_fh = cfg.get("use_frames_hist", True) and not (train and rng.random() < aug["p_frames_drop"])
    fr, wf, i_dec = frames_of(o, use_fh)
    K = len(fr)
    rgb = np.stack([f[0] for f in fr]).astype(np.uint8)
    dep = np.stack([f[1] for f in fr]).astype(np.float32)
    if rgb.shape[1] != 224:
        rgb = np.stack([_resize(x, 224, False) for x in rgb])
        dep = np.stack([_resize(x, 224, True) for x in dep])
    if train and aug["color"]:
        st = rng.bit_generator.state
        out = []
        for x in rgb:
            rng.bit_generator.state = st
            out.append(color_aug(rng, x))
        rgb = np.stack(out)
    if cfg.get("use_map", True):
        mt = mount_of(o, i_dec)
        try:
            if mt is not None:
                CM.set_mount(mt[0], mt[1], cam_h=mt[2])
            Mp, info = CM.build_map(dep[0], MAP_KIND, world_frames=wf if (wf and wf[1]) else None)
        finally:
            CM.set_mount(None)
    else:
        Mp, info = np.zeros((CM.C_MAP, CM.NX, CM.NY), np.float32), dict(cam_h=0.0, used_prior=1.0)
    N = norm
    pb = np.asarray(o["prop_base"], np.float32)[-T_PROP:]
    pj = np.asarray(o["prop_joints"], np.float32)[-T_PROP:]
    if len(pb) < T_PROP:
        pad = T_PROP - len(pb)
        pb = np.concatenate([np.repeat(pb[:1], pad, 0), pb])
        pj = np.concatenate([np.repeat(pj[:1], pad, 0), pj])
    pb = np.clip((pb - np.float32(N["base_mean"])) / np.float32(N["base_std"]), -10, 10)
    pjc, pjm = G.canon_joints(joint_names_of(o, desc, pj.shape[1]), pj, axis=1)
    pjc = np.clip((pjc - np.float32(N["joint_mean"])) / np.float32(N["joint_std"]), -10, 10)
    js, body, jm = G.embodiment(desc)
    jm = jm | pjm
    pjc *= jm[None, :, None]
    st_, sm = G.spec_tokens(spec, ranges_only=cfg.get("ranges_only", False))
    fp, fpm = G.fp_vector(probe)
    ops, opm = G.ops_tokens(probe)
    ak = json.loads(str(o["ach_keys"])) if "ach_keys" in o else None
    if probe is not None and ptrace_cache is not None and id(probe) in ptrace_cache:
        ptr, ptv = ptrace_cache[id(probe)]
    else:
        ptr, ptv = G.probe_trace_feats(probe, spec, joint_names=desc.get("joint_names"), ach_keys=ak)
        if probe is not None and ptrace_cache is not None:
            ptrace_cache[id(probe)] = (ptr, ptv)
    ptr, ptv = ptr.copy(), ptv.copy()
    if probe is not None and train:
        fp = fp * (1 + rng.normal(0, aug["fp_noise"], G.N_FP)).astype(np.float32)
        ops[:, len(G.OP_KINDS) + 1 :] *= (
            1 + rng.normal(0, aug["fp_noise"], ops[:, len(G.OP_KINDS) + 1 :].shape)
        ).astype(np.float32)
        if rng.random() < aug["p_fp_drop"]:
            fpm[:] = False
            opm[:] = False
            ptv[:] = False
        elif rng.random() < aug["p_trace_drop"]:
            ptv[:] = False
    fpn = np.where(fpm, (fp - np.float32(N["fp_mean"])) / np.float32(N["fp_std"]), 0.0).clip(-6, 6).astype(np.float32)
    ops *= opm[:, None]
    ptr *= ptv[:, None]
    if cfg.get("use_hist", True):
        hx, hv = G.history_feats(
            {k: o[k] for k in o if k.startswith("hist_") or k in ("joint_names", "ach_keys")}, spec
        )
        if train and hv.any():
            if rng.random() < aug["p_hist_drop"]:
                hv[:] = False
            elif rng.random() < aug["p_hist_trunc"]:
                L = int(rng.integers(5, G.T_HIST + 1))
                hv[: G.T_HIST - L] = False
            hx *= hv[:, None]
    else:
        hx, hv = np.zeros((G.T_HIST, G.F_FRAME), np.float32), np.zeros(G.T_HIST, bool)
    feas_raw = np.asarray(o["feas"], np.float32) if "feas" in o else None
    feas = G.feas_feats(feas_raw, geom_only=not cfg.get("use_feas_probe", True))
    span = max(f[2] for f in fr)
    meta = np.array(
        [
            float(K == 1),
            K / K_HIST,
            info["used_prior"],
            info["cam_h"] / 1.5,
            float(cfg.get("use_map", True)),
            span / 2.0,
            float(hv.any()),
            float(fpm.any()),
            float(feas_raw is not None),
        ],
        np.float32,
    )
    return dict(
        rgb=np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)),
        depth=dep[:, None],
        fage=np.array([f[2] for f in fr], np.float32),
        cmap=Mp.astype(np.float32),
        meta=meta,
        base=pb.astype(np.float32),
        joints=pjc.astype(np.float32),
        jstatic=js,
        jmask=jm,
        body=body,
        feas=feas,
        spec_tok=st_,
        spec_mask=sm,
        fp=fpn,
        fp_mask=fpm,
        ops=ops.astype(np.float32),
        ops_mask=opm,
        ptrace=ptr.astype(np.float32),
        ptrace_valid=ptv,
        hist=hx.astype(np.float32),
        hist_valid=hv,
    ), feas_raw


try:
    import torch
    from torch.utils.data import Dataset

    class SitDataset(Dataset):
        def __init__(self, rows, norm, cfg, augment=False, aug=None, seed=0, n_max=N_MAX):
            self.rows, self.norm, self.cfg, self.augment = rows, norm, cfg, augment
            self.aug = dict(DEFAULT_AUG, **(aug or {}))
            self._rng = np.random.default_rng(seed)
            self.n_max = n_max
            self._spec, self._probe, self._ptr = {}, {}, {}

        def __len__(self):
            return len(self.rows)

        def spec_of(self, r):
            p = r["spec"]
            if p not in self._spec:
                self._spec[p] = json.load(open(p))
            return self._spec[p]

        def probe_of(self, r):
            p = r.get("probe", "")
            if not p:
                return None
            if p not in self._probe:
                self._probe[p] = json.load(open(p))
            return self._probe[p]

        def __getitem__(self, i):
            r = self.rows[i]
            cfg, rng, aug = self.cfg, self._rng, self.aug
            train = self.augment
            on = np.load(r["obs"], allow_pickle=False)
            o = {k: on[k] for k in on.files}
            spec = self.spec_of(r)
            probe = self.probe_of(r) if cfg.get("use_fp", True) else None
            desc = robot_desc(r.get("robot", robot_of(r["base"])))
            ctx, feas_raw = build_context(
                o, spec, probe, desc, self.norm, cfg, train=train, rng=rng, aug=aug, ptrace_cache=self._ptr
            )
            eps = list(r["eps"])
            if train and len(eps) > self.n_max:
                rest = list(rng.choice(np.arange(1, len(eps)), self.n_max - 1, replace=False))
                eps = [eps[0]] + [eps[j] for j in sorted(rest)]
            C, MID, Y, ACH, AR, FAR = [], [], defaultdict(list), [], [], []
            for p in eps:
                z = np.load(p, allow_pickle=False)
                f = set(z.files)
                c, mid, A_, _ = rollout_cand(z, f, feas_raw, use_drv=cfg.get("use_drv", True))
                C.append(c)
                MID.append(mid)
                AR.append(A_)
                FAR.append(np.asarray(z["feas_act"], np.float32) if "feas_act" in f else np.full(4, np.nan, np.float32))
                for kk, v in rollout_labels(z, f).items():
                    Y[kk].append(v)
                ACH.append(G.ach_targets(z, f))
            t = torch.from_numpy
            out = {k: t(np.asarray(v)) for k, v in ctx.items()}
            out.update(
                cand=t(np.stack(C).astype(np.float32)),
                macro_id=torch.tensor(MID, dtype=torch.long),
                y_ach=t(np.stack(ACH).astype(np.float32)),
                env_act=t(np.stack(AR).astype(np.float32)),
                fa_raw=t(np.stack(FAR)),
                feas_raw=t(
                    feas_raw.astype(np.float32) if feas_raw is not None else np.full(G.N_FEAS, np.nan, np.float32)
                ),
                idx=i,
                ep_k=torch.tensor([int(os.path.basename(p)[:-4].rsplit("_", 1)[1]) for p in eps], dtype=torch.long),
            )
            for kk, v in Y.items():
                out[f"y_{kk}"] = torch.tensor(v, dtype=torch.float32)
            return out

    def collate4(batch):
        B = len(batch)
        K = max(b["rgb"].shape[0] for b in batch)
        N = max(b["cand"].shape[0] for b in batch)
        out = {}
        rgb = torch.zeros(B, K, 3, 224, 224, dtype=torch.uint8)
        dep = torch.zeros(B, K, 1, 224, 224)
        fmask = torch.zeros(B, K, dtype=torch.bool)
        fage = torch.zeros(B, K)
        for i, b in enumerate(batch):
            k = b["rgb"].shape[0]
            rgb[i, :k] = b["rgb"]
            dep[i, :k] = b["depth"]
            fmask[i, :k] = True
            fage[i, :k] = b["fage"]
        out.update(rgb=rgb, depth=dep, fmask=fmask, fage=fage)
        for k in (
            "cmap",
            "meta",
            "base",
            "joints",
            "jstatic",
            "jmask",
            "body",
            "feas",
            "spec_tok",
            "spec_mask",
            "fp",
            "fp_mask",
            "ops",
            "ops_mask",
            "ptrace",
            "ptrace_valid",
            "hist",
            "hist_valid",
            "feas_raw",
        ):
            out[k] = torch.stack([b[k] for b in batch])
        cand = torch.zeros(B, N, G.F_CAND)
        mid = torch.zeros(B, N, dtype=torch.long)
        rm = torch.zeros(B, N, dtype=torch.bool)
        ach = torch.full((B, N, G.N_ACH_HEAD), float("nan"))
        ea = torch.zeros(B, N, 16)
        far = torch.full((B, N, 4), float("nan"))
        ek = torch.full((B, N), -1, dtype=torch.long)
        ys = {k: torch.full((B, N), float("nan")) for k in batch[0] if k.startswith("y_") and k != "y_ach"}
        for i, b in enumerate(batch):
            n = b["cand"].shape[0]
            cand[i, :n] = b["cand"]
            mid[i, :n] = b["macro_id"]
            rm[i, :n] = True
            ach[i, :n] = b["y_ach"]
            ea[i, :n] = b["env_act"]
            far[i, :n] = b["fa_raw"]
            ek[i, :n] = b["ep_k"]
            for k in ys:
                ys[k][i, :n] = b[k]
        out.update(
            cand=cand,
            macro_id=mid,
            rmask=rm,
            y_ach=ach,
            env_act=ea,
            fa_raw=far,
            ep_k=ek,
            idx=torch.tensor([b["idx"] for b in batch]),
            **ys,
        )
        return out
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--root", default=DATA)
    b.add_argument("--exclude", default="", help="QA exclusion json (list of sids or {exclude: [...]})")
    b.add_argument("--out", required=True)
    s = sub.add_parser("check")
    s.add_argument("--manifest", required=True)
    s.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        rows, drop, info = build_gen4(a.root, a.exclude)
        set_weights(rows)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        json.dump(
            dict(drop=drop, **info, n_rows=len(rows), exclude=a.exclude), open(a.out + ".info.json", "w"), indent=1
        )
        cnt = defaultdict(lambda: defaultdict(int))
        for r in rows:
            cnt[r["base"]][r["split"]] += 1
        print(
            f"[gen4 manifest] {len(rows)} situations x controller from {info['parts']} finished jobs -> {a.out}; dropped {drop}"
        )
        for k, v in sorted(info["part_why"].items()):
            if "re-audit" in v:
                print(f"   {k}: {v}")
        for k, v in sorted(cnt.items()):
            print(f"   {k:6s} {dict(v)}  ctrls {len({r['ctrl_id'] for r in rows if r['base'] == k})}")
    elif a.cmd == "check":
        rows = read_manifest(a.manifest)
        norm = compute_norm(rows, n=50)
        rng = np.random.default_rng(0)
        pick = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), replace=False)]
        ds = SitDataset(pick, norm, {}, augment=True)
        bt = collate4([ds[i] for i in range(len(ds))])
        for k, v in bt.items():
            if hasattr(v, "shape"):
                print(
                    f"{k:14s} {tuple(v.shape)} {v.dtype} finite={bool(torch.isfinite(v.float()).all()) if v.dtype != torch.bool else '-'}"
                )


if __name__ == "__main__":
    main()
