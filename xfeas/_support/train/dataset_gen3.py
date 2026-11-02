#!/usr/bin/env python
"""
Situation-level dataset for XFeas-Gen3 (gen3 model agent, 2026-10-17). One ITEM = one situation x controller: the
decision context (encoded once) + all of its simulated, non-duplicate rollouts as candidates.

Manifest (jsonl, one line per situation x controller), built here:
    $PY xfeas/_support/train/dataset_gen3.py build --gen3-root data/gen_v3 [--gen3-audit data/gen_v3/audit/<excl>.json]
        [--gen2-manifest data/gen_train/gen2b/manifest.jsonl --gen2-weight 0.3] --out data/gen_train/gen3/manifest.jsonl
    keys: key, fmt ('gen_v3' | 'gen_v2map'), obs, eps [rollout paths, stock first], ctrl_id, base, virtual, env, split
          (= zone_split), bucket, subfamily, light, spec, probe, weight, sit_id, n_ok, stock_ok, audited
  * gen_v3: data/gen_v3/<env>/<base>/ep/<sid>_<kk>.npz grouped by sid (GEN3_SPEC 9). Rollouts with sim = False
    (driver_infeasible) or dup_of >= 0 are left out of the candidate list (the selector hard-gates driver-infeasible
    programs; a duplicate is the same program). SONIC ('g1s') can never enter: build() raises.
  * gen_v2map: gen_v2 G1 (AMO = g1c) rows of an audited gen2 manifest, mapped to env_act (gen3_common.env_act_from_act_v2:
    dh -> top through the AMO flat-probe table; wid unknown). They have NO probe, NO history, NO feas: those inputs are
    masked. base = 'g1c', so leave-one-base-out drops them with g1c.
Item tensors (see collate3): context = model_gen3.CTX_KEYS; candidates cand (N, F_CAND), macro_id (N,), rmask (N,);
labels y_<head> (N,) NaN = masked, y_ach (N, 4).
Zero-shot augmentation (train only): fingerprint noise (x (1 + N(0, fp_noise)) in physical units) and dropout of the
whole probe group (fp + ops + trace, p_fp_drop) or the trace alone (p_trace_drop); history dropout (p_hist_drop) and
random truncation to the last L in [5, 60] frames; frame-memory dropout (p_frames_drop); colour jitter (gen1/gen2).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, glob, json, math, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gen3_common as G
import cmap as CM

ROOT = xfeas.paths.ROOT
HELD_OUT_CTRL = ("g1s",)  # GEAR-SONIC: never in data / probes / fitting (GEN3_SPEC)
# REFERENCE upper bound only (gen3_sonic_eval.sh arm gen3_ftsonic): SONIC rows may enter a manifest only with this flag,
# only from a root that is not data/gen_v3, and train_gen3 then only writes to a *_ftsonic_REFERENCE output.
SONIC_REF_OK = os.environ.get("GEN3_ALLOW_SONIC_REFERENCE", "0") == "1"
N_MAX = 12
K_HIST = 4
T_PROP = 100
MAP_KIND = "g1"  # same G1 head camera for every controller (cmap prefix 'g1')


_G1_DESC = None


def g1_desc():
    """fixed G1 embodiment descriptor (g1_desc.json = the gen_v2 g1c descriptor: same body for every controller).
    gen_v3 rollouts carry no 'desc'; their joint ORDER comes from obs['joint_names'] (see joint_names_of)."""
    global _G1_DESC
    if _G1_DESC is None:
        _G1_DESC = json.load(open(os.path.join(THIS, "g1_desc.json")))
    return _G1_DESC


def desc_of(z0):
    """embodiment descriptor: the fixed G1 body for every gen_v3 row (spec v0.2 stores a stripped desc; the body is the
    same robot, and a per-controller joint list would identify the controller -- joint ORDER still comes from
    obs['joint_names']); gen_v2 rows keep their own (g1c) descriptor"""
    fmt = str(z0["fmt"]) if "fmt" in z0 else ""
    if fmt.startswith("gen_v3") or "desc" not in z0:
        return g1_desc()
    return json.loads(str(z0["desc"]))


def joint_names_of(o, desc, J):
    """names of the J proprio joint columns: obs joint_names (gen_v3, articulation dof order) when the count matches, else
    the descriptor's (gen_v2 / old)"""
    if "joint_names" in o:
        jn = json.loads(str(o["joint_names"]))
        if len(jn) == J:
            return jn
    if len(desc["joint_names"]) == J:
        return list(desc["joint_names"])
    raise ValueError(f"no joint names for {J} proprio joints")


def _s(z, k, d=""):
    try:
        return str(z[k]) if k in z.files else d
    except Exception:
        return d


def _light(s):
    try:
        return json.loads(s).get("mode", "") if s.startswith("{") else s
    except Exception:
        return s


def _read_ep_meta(p):
    z = np.load(p, allow_pickle=False)
    f = set(z.files)
    g = lambda k, d=None: (z[k].item() if z[k].ndim == 0 else z[k]) if k in f else d
    return dict(
        path=p,
        success=bool(g("success", False)),
        sim=bool(g("sim", True)),
        dup_of=int(g("dup_of", -1)),
        k=int(g("rollout_k", 0)),
        kind=_s(z, "kind"),
        ctrl_id=_s(z, "ctrl_id"),
        base=_s(z, "base"),
        virtual=bool(g("virtual", False)),
        zone=_s(z, "zone_split"),
        env=_s(z, "env"),
        bucket=_s(z, "bucket"),
        subfamily=_s(z, "subfamily"),
        light=_light(_s(z, "lighting")),
        obs=_s(z, "obs_file"),
        probe=_s(z, "probe_file"),
        sit_id=_s(z, "geo_sit") or _s(z, "sit_id"),
        fmt=_s(z, "fmt"),
        macro=_s(z, "macro"),
    )


def _exclusions(path):
    if not path:
        return None
    J = json.load(open(path))
    if isinstance(J, list):
        return set(map(str, J))
    out = set()
    for k in ("exclude", "situations", "sids", "excluded"):
        v = J.get(k)
        if isinstance(v, list):
            out |= {str(x if not isinstance(x, dict) else x.get("sid", x.get("sit", ""))) for x in v}
        elif isinstance(v, dict):
            out |= set(map(str, v))
    return out


def load_feas_override(root):
    """data/gen_v3/feas_v03.jsonl (scripts/gen3/recompute_feas.py): feas recomputed with the v0.3 check + per-rollout
    feas_act (keyed by rollout_k), one line per sid. Overrides the in-npz values of rows collected before the fix."""
    p = os.path.join(root, "feas_v03.jsonl")
    out = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                d = json.loads(l)
                out[d["sid"]] = (d["feas"], {int(k): v for k, v in (d.get("feas_act") or {}).items()})
    return out, p if out else ""


def ep_k(path):
    return int(os.path.basename(path)[:-4].rsplit("_", 1)[1])


def build_gen3(root, audit, threads=16):
    from concurrent.futures import ThreadPoolExecutor

    root = os.path.abspath(root)
    FO, fo_path = load_feas_override(root)
    if SONIC_REF_OK and os.path.basename(root.rstrip("/")) == "gen_v3":
        raise SystemExit("[gen3 manifest] GEN3_ALLOW_SONIC_REFERENCE is only for the separate SONIC reference root")
    eps = sorted(p for p in glob.glob(os.path.join(root, "*", "*", "ep", "*.npz")))
    with ThreadPoolExecutor(threads) as ex:
        metas = list(ex.map(_read_ep_meta, eps))
    excl = _exclusions(audit)
    S = defaultdict(list)
    for m in metas:
        sid = os.path.basename(m["path"])[:-4].rsplit("_", 1)[0]
        S[sid].append(m)
    out, drop = [], defaultdict(int)
    for sid, ms in sorted(S.items()):
        ms.sort(key=lambda m: m["k"])
        m0 = ms[0]
        if (
            m0["ctrl_id"] in HELD_OUT_CTRL or m0["base"] in HELD_OUT_CTRL or sid.endswith("__g1s")
        ) and not SONIC_REF_OK:
            raise SystemExit(f"[gen3 manifest] {sid}: SONIC data found in {root} -- it must never be used for training")
        if "smoke" in m0["probe"] or "smoke" in os.path.basename(
            m0["probe"]
        ):  # v0.1 smoke sessions (coordinator, 2026-10-17)
            drop["smoke"] += 1
            continue
        if excl is not None and (sid in excl or m0["sit_id"] in excl):
            drop["audit"] += 1
            continue
        keep = [m for m in ms if m["sim"] and m["dup_of"] < 0]
        st = [m for m in keep if m["k"] == 0 or m["kind"] == "stock"]
        if not st:
            drop["no_stock"] += 1
            continue
        keep = st[:1] + [m for m in keep if m is not st[0]]
        if len(keep) < 2:
            drop["lt2_rollouts"] += 1
            continue
        obs = (
            os.path.join(root, m0["obs"])
            if m0["obs"]
            else os.path.join(os.path.dirname(os.path.dirname(m0["path"])), "obs", sid + ".npz")
        )
        if not os.path.exists(obs):
            drop["no_obs"] += 1
            continue
        spec = os.path.join(root, "specs", m0["ctrl_id"] + ".json")
        probe = os.path.join(root, m0["probe"]) if m0["probe"] else ""
        if not os.path.exists(spec):
            drop["no_spec"] += 1
            continue
        ovr = FO.get(sid)
        out.append(
            dict(
                feas_ovr=ovr[0] if ovr else None,
                fa_ovr={str(k): v for k, v in ovr[1].items()} if ovr else None,
                feas_src=fo_path if ovr else "npz",
                key=sid,
                fmt="gen_v3",
                obs=obs,
                eps=[m["path"] for m in keep],
                ctrl_id=m0["ctrl_id"],
                base=m0["base"] or m0["ctrl_id"].split("_")[1 if m0["ctrl_id"].startswith("v_") else 0],
                virtual=m0["virtual"],
                env=m0["env"],
                split=m0["zone"] or "train",
                bucket=m0["bucket"],
                subfamily=m0["subfamily"],
                light=m0["light"],
                spec=spec,
                probe=probe if os.path.exists(probe) else "",
                sit_id=m0["sit_id"] or sid.split("__")[0],
                n_ok=int(sum(m["success"] for m in keep)),
                stock_ok=bool(keep[0]["success"]),
                audited=excl is not None,
                synthetic=m0["fmt"].endswith("synthetic"),
                n_infeasible=int(sum(not m["sim"] for m in ms)),
                n_dup=int(sum(m["dup_of"] >= 0 for m in ms)),
            )
        )
    return out, dict(drop)


def build_gen2map(manifest):
    rows = [json.loads(l) for l in open(manifest)]
    S = defaultdict(list)
    for r in rows:
        if str(r.get("source", "")).startswith("gen_v2") and r.get("robot") == "g1c":
            r["path"] = (
                r["path"] if os.path.isabs(r["path"]) else os.path.join(ROOT, r["path"])
            )  # gen2b stores Scripts-relative paths
            if not os.path.exists(r["path"]):
                raise FileNotFoundError(f"[gen3 manifest] gen_v2 row missing on disk: {r['path']}")
            S[r["sit"]].append(r)
    spec = os.path.join(ROOT, "xfeas", "_support", "model", "specs", "g1c.json")
    out = []
    for sid, rs in sorted(S.items()):
        rs.sort(key=lambda r: int(r.get("rollout_k", 0)))
        if not (rs[0].get("rollout_kind") == "stock" or int(rs[0].get("rollout_k", -1)) == 0) or len(rs) < 2:
            continue
        p0 = rs[0]["path"]
        obs = os.path.join(os.path.dirname(os.path.dirname(p0)), "obs", sid + ".npz")
        if not os.path.exists(obs):
            raise FileNotFoundError(
                f"[gen3 manifest] gen_v2 obs missing (would silently drop the history frames): {obs}"
            )
        out.append(
            dict(
                key=f"{sid}__g1c_v2",
                fmt="gen_v2map",
                obs=obs,
                eps=[r["path"] for r in rs],
                ctrl_id="g1c",
                base="g1c",
                virtual=False,
                env=rs[0]["env"],
                split=rs[0]["split"],
                bucket=rs[0]["bucket"],
                subfamily=rs[0]["subfamily"],
                light=rs[0].get("light", ""),
                spec=spec if os.path.exists(spec) else "g1c_fallback",
                probe="",
                sit_id=sid,
                n_ok=int(sum(bool(r["success"]) for r in rs)),
                stock_ok=bool(rs[0]["success"]),
                audited=bool(rs[0].get("audited", True)),
                synthetic=False,
            )
        )
    return out


def set_weights(rows, gen2_weight):
    n = defaultdict(int)
    for r in rows:
        n[(r["ctrl_id"], r["env"], r["fmt"])] += 1
    for r in rows:
        r["weight"] = (1.0 / math.sqrt(n[(r["ctrl_id"], r["env"], r["fmt"])])) * (
            gen2_weight if r["fmt"] == "gen_v2map" else 1.0
        )
    return rows


def read_manifest(p):
    return [json.loads(l) for l in open(p)]


def fold_rows(rows, holdout_base="", drop_gen2map=False):
    """leave-one-base-out: the held-out base and ALL of its virtual wrappers (and gen_v2 AMO rows if base g1c) leave
    train AND val; they form the 'test' set of the fold"""
    out = []
    for r in rows:
        r = dict(r)
        if drop_gen2map and r["fmt"] == "gen_v2map":
            continue
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
        z = np.load(r["eps"][0], allow_pickle=False)
        desc = desc_of({k: z[k] for k in ("fmt", "desc") if k in z.files})
        pb = np.asarray(o["prop_base"] if "prop_base" in o.files else z["prop_base"], np.float32)
        pj = np.asarray(o["prop_joints"] if "prop_joints" in o.files else z["prop_joints"], np.float32)
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
            fp, pm = G.fp_vector(json.load(open(r["probe"])))
            fps[r["ctrl_id"]] = (fp, pm)
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


# ------------------------------------------------------------------------------------------------ dataset
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


def rollout_labels(z, f, fmt):
    nan = float("nan")
    g = lambda k: float(z[k]) if k in f and np.ndim(z[k]) == 0 else nan
    ok = bool(z["success"])
    y = dict(
        success=float(ok),
        fall=float(bool(z["fell"])),
        stall=float(bool(z["stall"])) if "stall" in f else nan,
        collision=float(bool(z["collision"])) if "collision" in f else nan,
        progress=g("progress"),
        tilt_margin=nan,
        t_norm=nan,
        env_dev=nan,
        cost=nan,
        slip=nan,
    )
    if "slipped" in f and "slip_mean" in f and np.isfinite(float(z["slip_mean"])):
        y["slip"] = float(bool(z["slipped"]))
    tm = g("tilt_margin")
    y["tilt_margin"] = float(np.clip(tm, -1, 1)) if np.isfinite(tm) else nan
    tg, tr = g("t_goal"), g("t_ref")
    if ok and np.isfinite(tg) and np.isfinite(tr) and tr > 0:
        y["t_norm"] = float(np.clip(tg / tr, 0, 5))
    if fmt == "gen_v3":
        ed = g("env_dev")
        c = g("cost")
        y["env_dev"] = ed if np.isfinite(ed) else nan
        y["cost"] = float(np.clip(c, 0, 5)) if (ok and np.isfinite(c)) else nan
    return y


def rollout_cand(z, f, fmt, feas, use_drv=True, fa_ovr=None):
    if fmt == "gen_v2map":
        A, M, kn, mac = G.env_act_from_act_v2(z["act_v2"], z["act_v2_mask"], _s(z, "skill_v2", "go"))
        return G.cand_feats(A, M, G.MACRO_ID[mac], feas=None, known=kn, use_drv=False), G.MACRO_ID[mac], A, M
    A = np.asarray(z["env_act"], np.float32)
    M = np.asarray(z["env_act_mask"], np.float32)
    mid = int(z["macro_id"]) if "macro_id" in f else G.MACRO_ID.get(_s(z, "macro", "constant"), 0)
    drv = None
    if "drv" in f:
        try:
            drv = json.loads(str(z["drv"]))
        except Exception:
            drv = None
    fa = (
        np.asarray(fa_ovr, np.float32)
        if fa_ovr is not None
        else (np.asarray(z["feas_act"], np.float32) if "feas_act" in f else None)
    )
    return G.cand_feats(A, M, mid, feas=feas, feas_act=fa, drv=drv, use_drv=use_drv), mid, A, M


def frames_of(o, use_hist):
    """obs dict -> (frames newest first [(rgb, depth, age_s)], world_frames for cmap or None); gen_v2/gen_v3 obs layout"""
    fr = [(np.asarray(o["rgb"])[..., :3], np.nan_to_num(np.asarray(o["depth"], np.float32), posinf=60.0), 0.0)]
    wf = None
    if use_hist and "rgb_hist" in o and "t_hist" in o:
        t = np.asarray(o["t_hist"], np.float32)
        order = np.argsort(-t)
        f128, c128 = float(o["K128"][0, 0]), float(o["K128"][0, 2])
        M_dec = o["cam_hist"][order[0]]
        wfl = []
        for i in order[1:K_HIST]:
            dp = np.nan_to_num(np.asarray(o["depth_hist"][i], np.float32), posinf=20.0)
            fr.append((_resize(o["rgb_hist"][i][..., :3], 224, False), _resize(dp, 224, True), float(-t[i])))
            wfl.append((dp, o["cam_hist"][i], f128, c128))
        wf = (M_dec, wfl)
    return fr, wf


def build_context(o, spec, probe, desc, z0, norm, cfg, train=False, rng=None, aug=None):
    """decision context of one situation x controller -> (dict of numpy arrays = model_gen3.CTX_KEYS, feas_raw or None).
    o = obs npz as a dict; z0 = the stock rollout (dict, fallback for prop / feas); probe None = absent. Shared by the
    dataset and select_gen3, so training and deployment featurize identically."""
    aug = dict(DEFAULT_AUG, **(aug or {}))
    rng = rng if rng is not None else np.random.default_rng(0)
    if not cfg.get("use_fp", True):
        probe = None
    # ---- frames + map
    use_fh = cfg.get("use_frames_hist", True) and not (train and rng.random() < aug["p_frames_drop"])
    fr, wf = frames_of(o, use_fh)
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
        Mp, info = CM.build_map(dep[0], MAP_KIND, world_frames=wf if (wf and wf[1]) else None)
    else:
        Mp, info = np.zeros((CM.C_MAP, CM.NX, CM.NY), np.float32), dict(cam_h=0.0, used_prior=1.0)
    # ---- proprio (canonical joints), embodiment
    N = norm
    pb = np.asarray(o["prop_base"] if "prop_base" in o else z0["prop_base"], np.float32)[-T_PROP:]
    pj = np.asarray(o["prop_joints"] if "prop_joints" in o else z0["prop_joints"], np.float32)[-T_PROP:]
    if len(pb) < T_PROP:
        pad = T_PROP - len(pb)
        pb = np.concatenate([np.repeat(pb[:1], pad, 0), pb])
        pj = np.concatenate([np.repeat(pj[:1], pad, 0), pj])
    pb = np.clip((pb - np.float32(N["base_mean"])) / np.float32(N["base_std"]), -10, 10)
    pjc, pjm = G.canon_joints(joint_names_of(o, desc, pj.shape[1]), pj, axis=1)
    pjc = np.clip((pjc - np.float32(N["joint_mean"])) / np.float32(N["joint_std"]), -10, 10)
    js, body, jm = G.embodiment(desc)
    jm = jm | pjm  # joints the controller exposes (wrists: no limits in the G1 desc)
    pjc *= jm[None, :, None]
    # ---- spec / probe / history / feas
    st_, sm = G.spec_tokens(spec, ranges_only=cfg.get("ranges_only", False))
    fp, fpm = G.fp_vector(probe)
    ops, opm = G.ops_tokens(probe)
    ptr, ptv = G.probe_trace_feats(
        probe,
        spec,
        joint_names=desc.get("joint_names"),
        ach_keys=json.loads(str(o["ach_keys"])) if "ach_keys" in o else None,
    )
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
        hx, hv = G.history_feats({k: o[k] for k in o if k.startswith("hist_") or k == "joint_names"}, spec)
        if train and hv.any():
            if rng.random() < aug["p_hist_drop"]:
                hv[:] = False
            elif rng.random() < aug["p_hist_trunc"]:
                L = int(rng.integers(5, G.T_HIST + 1))
                hv[: G.T_HIST - L] = False
            hx *= hv[:, None]
    else:
        hx, hv = np.zeros((G.T_HIST, G.F_FRAME), np.float32), np.zeros(G.T_HIST, bool)
    feas_raw = (
        np.asarray(o["feas"], np.float32)
        if "feas" in o
        else (np.asarray(z0["feas"], np.float32) if "feas" in z0 else None)
    )
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
            self._spec, self._probe = {}, {}

        def __len__(self):
            return len(self.rows)

        def spec_of(self, r):
            p = r["spec"]
            if p not in self._spec:
                self._spec[p] = G.g1c_fallback_spec() if p == "g1c_fallback" else json.load(open(p))
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
            if r.get("feas_ovr") is not None:  # v0.3 feas recompute (feas_v03.jsonl)
                o["feas"] = np.asarray(r["feas_ovr"], np.float32)
            spec = self.spec_of(r)
            probe = self.probe_of(r) if cfg.get("use_fp", True) else None
            zn = np.load(r["eps"][0], allow_pickle=False)
            z0 = {k: zn[k] for k in zn.files}
            desc = desc_of(z0)
            ctx, feas_raw = build_context(o, spec, probe, desc, z0, self.norm, cfg, train=train, rng=rng, aug=aug)
            # ---- candidates
            eps = list(r["eps"])
            if train and len(eps) > self.n_max:
                rest = list(rng.choice(np.arange(1, len(eps)), self.n_max - 1, replace=False))
                eps = [eps[0]] + [eps[j] for j in sorted(rest)]
            else:
                eps = eps[: max(self.n_max, len(eps) if not train else self.n_max)]
            C, MID, Y, ACH, AR, FAR = [], [], defaultdict(list), [], [], []
            for k, p in enumerate(eps):
                z = zn if k == 0 else np.load(p, allow_pickle=False)
                f = set(z.files)
                fo = (r.get("fa_ovr") or {}).get(str(ep_k(p)))
                c, mid, A_, _ = rollout_cand(z, f, r["fmt"], feas_raw, use_drv=cfg.get("use_drv", True), fa_ovr=fo)
                C.append(c)
                MID.append(mid)
                AR.append(A_)
                fr_ = (
                    fo if fo is not None else (z["feas_act"] if "feas_act" in f else None)
                )  # ablation-independent (rule arm, gates)
                FAR.append(np.asarray(fr_, np.float32) if fr_ is not None else np.full(4, np.nan, np.float32))
                for kk, v in rollout_labels(z, f, r["fmt"]).items():
                    Y[kk].append(v)
                ACH.append(G.ach_targets(z, f) if r["fmt"] == "gen_v3" else np.full(G.N_ACH_HEAD, np.nan, np.float32))
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
            )
            for kk, v in Y.items():
                out[f"y_{kk}"] = torch.tensor(v, dtype=torch.float32)
            return out

    def collate3(batch):
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
        ys = {k: torch.full((B, N), float("nan")) for k in batch[0] if k.startswith("y_") and k != "y_ach"}
        for i, b in enumerate(batch):
            n = b["cand"].shape[0]
            cand[i, :n] = b["cand"]
            mid[i, :n] = b["macro_id"]
            rm[i, :n] = True
            ach[i, :n] = b["y_ach"]
            ea[i, :n] = b["env_act"]
            far[i, :n] = b["fa_raw"]
            for k in ys:
                ys[k][i, :n] = b[k]
        out.update(
            cand=cand,
            macro_id=mid,
            rmask=rm,
            y_ach=ach,
            env_act=ea,
            fa_raw=far,
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
    b.add_argument("--gen3-root", default=os.path.join(ROOT, "data", "gen_v3"))
    b.add_argument("--gen3-audit", default="")
    b.add_argument("--gen2-manifest", default="")
    b.add_argument("--gen2-weight", type=float, default=0.3)
    b.add_argument("--allow-unaudited", action="store_true")
    b.add_argument("--out", required=True)
    s = sub.add_parser("check")
    s.add_argument("--manifest", required=True)
    s.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        audit = a.gen3_audit
        if not audit:
            c = sorted(glob.glob(os.path.join(a.gen3_root, "audit", "*exclusion*.json"))) or sorted(
                glob.glob(os.path.join(a.gen3_root, "audit", "audit_*.json")), key=os.path.getmtime
            )
            audit = c[-1] if c else ""
        if audit:
            J = json.load(open(audit))
            if isinstance(J, dict) and J.get("bad"):
                print(
                    f"[gen3 manifest] audit {os.path.basename(audit)} flags: {J['bad']} (npz_missing {J.get('npz_missing')})"
                )
        if not audit and not a.allow_unaudited:
            sys.exit(
                f"[gen3 manifest] no audit exclusion json under {a.gen3_root}/audit (pass --gen3-audit or --allow-unaudited)"
            )
        rows, drop = build_gen3(a.gen3_root, audit)
        if a.gen2_manifest:
            rows += build_gen2map(a.gen2_manifest)
        set_weights(rows, a.gen2_weight)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        cnt = defaultdict(lambda: defaultdict(int))
        for r in rows:
            cnt[r["fmt"] + ":" + r["base"]][r["split"]] += 1
        print(f"[gen3 manifest] {len(rows)} situations -> {a.out}; audit {audit or 'NONE'}; dropped {drop}")
        for k, v in sorted(cnt.items()):
            print(f"   {k:18s} {dict(v)}  ctrls {len({r['ctrl_id'] for r in rows if r['fmt'] + ':' + r['base'] == k})}")
    elif a.cmd == "check":
        rows = read_manifest(a.manifest)
        norm = compute_norm(rows, n=50)
        ds = SitDataset(rows[: a.n], norm, {}, augment=True)
        bt = collate3([ds[i] for i in range(len(ds))])
        for k, v in bt.items():
            if hasattr(v, "shape"):
                print(
                    f"{k:14s} {tuple(v.shape)} {v.dtype} finite={bool(torch.isfinite(v.float()).all()) if v.dtype != torch.bool else '-'}"
                )


if __name__ == "__main__":
    main()
